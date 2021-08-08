import argparse
from threading import Condition, Lock

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from ros_x_habitat.msg import PointGoalWithGPSCompass, DepthImage
from ros_x_habitat.srv import EvalEpisode, ResetAgent, GetAgentTime
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int16
from src.constants.constants import (
    AgentResetCommands,
    NumericalMetrics,
    EvalEpisodeSpecialIDs,
    PACKAGE_NAME,
    ServiceNames,
)
from src.test.data.data import TestHabitatROSData
from src.utils import utils_logging


class MockHabitatEnvNode:
    def __init__(
        self,
        node_name: str,
        enable_physics_sim: bool = False,
        use_continuous_agent: bool = False,
        pub_rate: float = 5.0,
    ):
        r"""
        A simple node to mock the Habitat environment node. Publishes pre-recorded
        sensor values, and checks if the agent node produces correct actions.
        """
        if use_continuous_agent:
            assert enable_physics_sim

        self.node_name = node_name
        rospy.init_node(self.node_name)

        self.enable_physics_sim = enable_physics_sim
        self.use_continuous_agent = use_continuous_agent

        self.action_count = 0

        self.pub_queue_size = 10
        self.sub_queue_size = 10

        # set up logger
        self.logger = utils_logging.setup_logger(self.node_name)

        # shutdown is set to true by eval_episode() to indicate the
        # evaluator wants the node to shutdown
        self.shutdown_cv = Condition()
        with self.shutdown_cv:
            self.shutdown = False

        # agent action and variables to keep things synchronized
        # NOTE: self.action_cv's lock guards self.action, self.new_action_published,
        # self.action_count, and pseudo-guards the sensor reading arrays
        self.action = None
        self.observations = None
        self.new_action_published = False
        self.action_cv = Condition()

        # publish to sensor topics
        self.pub_rgb = rospy.Publisher("rgb", Image, queue_size=self.pub_queue_size)
        self.pub_depth = rospy.Publisher(
            "depth", numpy_msg(DepthImage), queue_size=self.pub_queue_size
        )
        self.pub_pointgoal_with_gps_compass = rospy.Publisher(
            "pointgoal_with_gps_compass",
            PointGoalWithGPSCompass,
            queue_size=self.pub_queue_size,
        )

        # subscribe from command topics
        if self.use_continuous_agent:
            self.sub = rospy.Subscriber(
                "cmd_vel", Twist, self.callback, queue_size=self.sub_queue_size
            )
        else:
            self.sub = rospy.Subscriber(
                "action", Int16, self.callback, queue_size=self.sub_queue_size
            )

        # wait until connections with the agent is established
        self.logger.info("env making sure agent is subscribed to sensor topics...")
        while (
            self.pub_rgb.get_num_connections() == 0
            or self.pub_depth.get_num_connections() == 0
            or self.pub_pointgoal_with_gps_compass.get_num_connections() == 0
        ):
            pass

        # mock eval_episode service server
        self.eval_service = rospy.Service(
            f"{PACKAGE_NAME}/{self.node_name}/{ServiceNames.EVAL_EPISODE}",
            EvalEpisode,
            self.eval_episode,
        )
        self.episode_counter_lock = Lock()
        with self.episode_counter_lock:
            self.episode_counter = 0

        self.logger.info("mock env initialized")

    def reset(self):
        r"""
        Resets the agent and the simulator. Requires being called only from
        the main thread.
        """
        return

    def eval_episode(self, request):
        r"""
        mocks the real eval_episode handler.
        :param request: evaluation parameters provided by evaluator, including
            last episode ID and last scene ID.
        :return: 1) episode ID and scene ID; 2) metrics including distance-to-
        goal, success and spl. All hard-coded.
        """
        resp = {
            "episode_id": "-1",
            "scene_id": "-1",
            NumericalMetrics.DISTANCE_TO_GOAL: 0.0,
            NumericalMetrics.SUCCESS: 0.0,
            NumericalMetrics.SPL: 0.0,
            NumericalMetrics.NUM_STEPS: 0,
            NumericalMetrics.SIM_TIME: 0.0,
            NumericalMetrics.RESET_TIME: 0.0,
        }
        if request.episode_id_last == EvalEpisodeSpecialIDs.REQUEST_SHUTDOWN:
            # if shutdown request, enable reset and return immediately
            with self.shutdown_cv:
                self.shutdown = True
                self.shutdown_cv.notify()
        else:
            with self.episode_counter_lock:
                # check if the evaluator requests correct episode and scene id
                if self.episode_counter == 0:
                    assert (
                        request.episode_id_last
                        == TestHabitatROSData.test_evaluator_episode_id_request
                    )
                    assert (
                        request.scene_id_last
                        == TestHabitatROSData.test_evaluator_scene_id
                    )

                if self.episode_counter == 0:
                    # respond with the test episode's metrics
                    resp = {
                        "episode_id": TestHabitatROSData.test_evaluator_episode_id_response,
                        "scene_id": TestHabitatROSData.test_evaluator_scene_id,
                        NumericalMetrics.DISTANCE_TO_GOAL: TestHabitatROSData.test_evaluator_distance_to_goal,
                        NumericalMetrics.SUCCESS: TestHabitatROSData.test_evaluator_success,
                        NumericalMetrics.SPL: TestHabitatROSData.test_evaluator_spl,
                    }
                    self.episode_counter += 1
                else:
                    # signal no more episodes
                    resp = {
                        "episode_id": EvalEpisodeSpecialIDs.RESPONSE_NO_MORE_EPISODES,
                        "scene_id": TestHabitatROSData.test_evaluator_scene_id,
                        NumericalMetrics.DISTANCE_TO_GOAL: 0.0,
                        NumericalMetrics.SUCCESS: 0.0,
                        NumericalMetrics.SPL: 0.0,
                    }

        return resp

    def cv2_to_depthmsg(self, depth_img: DepthImage):
        r"""
        Converts a Habitat depth image to a ROS DepthImage message.
        :param depth_img: depth image as a numpy array
        :returns: a ROS DepthImage message
        """
        depth_msg = DepthImage()
        depth_msg.height, depth_msg.width, _ = depth_img.shape
        depth_msg.step = depth_msg.width
        depth_msg.data = np.ravel(depth_img)
        return depth_msg

    def publish_sensor_observations(self, rgb, depth, ptgoal_with_gps_compass):
        r"""
        Publishes hard-coded sensor readings.
        :param rgb: RGB reading.
        :param depth: Depth reading.
        :param ptgoal_with_gps_compass: Pointgoal + GPS/Compass reading.
        """
        with self.action_cv:
            t_curr = rospy.Time.now()
            # rgb
            rgb_msg = CvBridge().cv2_to_imgmsg(
                rgb.astype(np.uint8), encoding="passthrough"
            )
            h = Header()
            h.stamp = t_curr
            rgb_msg.header = h
            # depth
            depth_msg = self.cv2_to_depthmsg(depth)
            h = Header()
            h.stamp = t_curr
            depth_msg.header = h
            # pointgoal + gps/compass
            ptgoal_with_gps_compass_msg = PointGoalWithGPSCompass()
            ptgoal_with_gps_compass_msg.distance_to_goal = ptgoal_with_gps_compass[0]
            ptgoal_with_gps_compass_msg.angle_to_goal = ptgoal_with_gps_compass[1]
            h = Header()
            h.stamp = t_curr
            ptgoal_with_gps_compass_msg.header = h
            # publish
            self.pub_rgb.publish(rgb_msg)
            self.pub_depth.publish(depth_msg)
            self.pub_pointgoal_with_gps_compass.publish(ptgoal_with_gps_compass_msg)

    def check_command(self, action_expected):
        r"""
        Checks if agent's command is correct. To the variables
        guarded by self.action_cv, this method works similarly
        to HabitatEnvNode.step().
        :param action_expected: ID of expected action
        """
        with self.action_cv:
            while self.new_action_published is False:
                self.action_cv.wait()
            self.new_action_published = False
            assert (
                self.action == action_expected
            ), f"wrong action: step={self.action_count}, expected action ID={action_expected}, received action ID={self.action}"
            self.action_count = self.action_count + 1

    def callback(self, cmd_msg):
        r"""
        Reads a command from agent and check if it is correct.
        """
        if self.use_continuous_agent:
            # TODO: do some checking
            pass
        else:
            with self.action_cv:
                self.action = cmd_msg.data
                self.new_action_published = True
                self.action_cv.notify()


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-physics-sim", default=False, action="store_true")
    parser.add_argument("--use-continuous-agent", default=False, action="store_true")
    parser.add_argument(
        "--sensor-pub-rate",
        type=float,
        default=5.0,
    )
    args = parser.parse_args()

    mock_env_node = MockHabitatEnvNode(
        node_name="mock_env_node",
        enable_physics_sim=args.enable_physics_sim,
        use_continuous_agent=args.use_continuous_agent,
        pub_rate=args.sensor_pub_rate,
    )

    with mock_env_node.shutdown_cv:
        while mock_env_node.shutdown is False:
            mock_env_node.shutdown_cv.wait()


if __name__ == "__main__":
    main()
