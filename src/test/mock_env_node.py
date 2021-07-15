from threading import Condition

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from ros_x_habitat.msg import PointGoalWithGPSCompass, DepthImage
from ros_x_habitat.srv import EvalEpisode, ResetAgent
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int16
from src.constants.constants import AgentResetCommands, NumericalMetrics


class MockHabitatEnvNode:
    def __init__(
        self,
        enable_physics: bool = True,
        episode_id: str = "-1",
        scene_id: str = "haha",
    ):
        r"""
        A simple node to mock the Habitat environment node. Publishes pre-recorded
        sensor values, and checks if the agent node produces correct actions.
        """
        self.enable_physics = enable_physics
        self.action_count = 0

        self.pub_queue_size = 10
        self.sub_queue_size = 10

        # agent action and variables to keep things synchronized
        # NOTE: self.action_cv's lock guards self.action, self.new_action_published,
        # self.action_count, and pseudo-guards the sensor reading arrays
        self.action = None
        self.observations = None
        self.new_action_published = False
        self.action_cv = Condition()

        # mock eval_episode service handler
        self.eval_service = rospy.Service(
            "eval_episode", EvalEpisode, self.eval_episode
        )
        self.episode_id = episode_id
        self.scene_id = scene_id

        # establish reset service with agent
        self.reset_agent = rospy.ServiceProxy("reset_agent", ResetAgent)

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
        if self.enable_physics:
            self.sub = rospy.Subscriber(
                "cmd_vel", Twist, self.callback, queue_size=self.sub_queue_size
            )
        else:
            self.sub = rospy.Subscriber(
                "action", Int16, self.callback, queue_size=self.sub_queue_size
            )

        # wait until connections with the agent is established
        print("env making sure agent is subscribed to sensor topics...")
        while (
            self.pub_rgb.get_num_connections() == 0
            or self.pub_depth.get_num_connections() == 0
            or self.pub_pointgoal_with_gps_compass.get_num_connections() == 0
        ):
            pass

        print("mock env initialized")

    def reset(self):
        r"""
        Resets the agent and the simulator. Requires being called only from
        the main thread.
        """
        # reset agent
        rospy.wait_for_service("reset_agent")
        try:
            resp = self.reset_agent(int(AgentResetCommands.RESET))
            assert resp.done
        except rospy.ServiceException:
            pass

    def eval_episode(self, request):
        r"""
        mocks the real eval_episode handler.
        :param request: evaluation parameters provided by evaluator, including
            last episode ID and last scene ID.
        :return: 1) episode ID and scene ID; 2) metrics including distance-to-
        goal, success and spl. All hard-coded.
        """
        assert request.episode_id_last == self.episode_id
        assert request.scene_id_last == self.scene_id

        resp = {
            "episode_id": "-1",
            "scene_id": "lol",
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
        if self.enable_physics is True:
            # TODO: do some checking
            pass
        else:
            with self.action_cv:
                self.action = cmd_msg.data
                self.new_action_published = True
                self.action_cv.notify()
