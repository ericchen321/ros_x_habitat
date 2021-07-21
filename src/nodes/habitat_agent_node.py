#!/usr/bin/env python
import argparse
import time
from math import radians
from threading import Condition
from threading import Lock
from typing import (
    Any,
    Dict,
)
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from habitat.config import Config
from habitat.sims.habitat_simulator.actions import _DefaultHabitatSimActions
from habitat_baselines.agents.ppo_agents import PPOAgent
from message_filters import TimeSynchronizer
from ros_x_habitat.msg import PointGoalWithGPSCompass, DepthImage
from ros_x_habitat.srv import ResetAgent, GetAgentTime
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from src.constants.constants import AgentResetCommands
import time
from src.utils import utils_logging


def get_default_config():
    c = Config()
    c.INPUT_TYPE = "blind"
    c.MODEL_PATH = "data/checkpoints/blind.pth"
    c.RESOLUTION = 256
    c.HIDDEN_SIZE = 512
    c.RANDOM_SEED = 7
    c.PTH_GPU_ID = 0
    c.GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
    return c


class HabitatAgentNode:
    r"""
    A class to represent a ROS node with a Habitat agent inside.
    The node subscribes to sensor topics, and publishes either
    discrete actions or continuous velocities to command topics.
    """

    def __init__(
        self,
        agent_config: Config,
        enable_physics: bool = False,
        control_period: float = 1.0,
        sensor_pub_rate: float = 5.0,
    ):
        self.agent_config = agent_config
        self.enable_physics = enable_physics
        self.sensor_pub_rate = float(sensor_pub_rate)

        # declare an agent instance
        self.agent = PPOAgent(agent_config)

        # set up logger
        self.logger = utils_logging.setup_logger("agent_node")

        # agent publish and subscribe queue size
        # TODO: make them configurable by constructor argument
        self.sub_queue_size = 10
        self.pub_queue_size = 10

        # establish reset protocol with env
        self.reset_service = rospy.Service("reset_agent", ResetAgent, self.reset_agent)

        # establish agent time protocol with env
        self.agent_time_service = rospy.Service("get_agent_time", GetAgentTime, self.get_agent_time)

        # control_period defined for continuous mode
        if self.enable_physics:
            self.control_period = control_period

        # the last action produced from the agent
        self.action = None

        # count the number of frames received from Habitat simulator;
        # reset to 0 every time an action is completed
        # not applicable in discrete mode
        if self.enable_physics:
            self.count_frames = 0

        # for timing
        self.agent_time = 0

        # lock guarding access to self.action, self.count_frames,
        # self.agent and self.agent_time
        self.lock = Lock()

        # shutdown triggers the node to be shutdown. Guarded by shutdown_cv
        self.shutdown_cv = Condition()
        with self.shutdown_cv:
            self.shutdown = False

        # publish to command topics
        if self.enable_physics:
            self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=self.pub_queue_size)
        else:
            self.pub = rospy.Publisher("action", Int16, queue_size=self.pub_queue_size)

        # subscribe to sensor topics
        if (
            self.agent_config.INPUT_TYPE == "rgb"
            or self.agent_config.INPUT_TYPE == "rgbd"
        ):
            self.sub_rgb = message_filters.Subscriber("rgb", Image)
        if (
            self.agent_config.INPUT_TYPE == "depth"
            or self.agent_config.INPUT_TYPE == "rgbd"
        ):
            self.sub_depth = message_filters.Subscriber("depth", numpy_msg(DepthImage))
        self.sub_pointgoal_with_gps_compass = message_filters.Subscriber(
            "pointgoal_with_gps_compass", PointGoalWithGPSCompass
        )

        # filter sensor topics with time synchronizer
        if self.agent_config.INPUT_TYPE == "rgb":
            self.ts = TimeSynchronizer(
                [self.sub_rgb, self.sub_pointgoal_with_gps_compass],
                queue_size=self.sub_queue_size,
            )
            self.ts.registerCallback(self.callback_rgb)
        elif self.agent_config.INPUT_TYPE == "rgbd":
            self.ts = TimeSynchronizer(
                [self.sub_rgb, self.sub_depth, self.sub_pointgoal_with_gps_compass],
                queue_size=self.sub_queue_size,
            )
            self.ts.registerCallback(self.callback_rgbd)
        else:
            self.ts = TimeSynchronizer(
                [self.sub_depth, self.sub_pointgoal_with_gps_compass],
                queue_size=self.sub_queue_size,
            )
            self.ts.registerCallback(self.callback_depth)

        self.logger.info("agent making sure env subscribed to command topic...")
        while self.pub.get_num_connections() == 0:
            pass

        self.logger.info("agent initialized")

    def reset_agent(self, request):
        r"""
        ROS service handler which resets the agent and related variables.
        :param request: command from the env node.
        :returns: True
        """
        if request.reset == AgentResetCommands.RESET:
            # reset the agent
            # NOTE: here we actually re-instantiate a new agent
            # before resetting it, because somehow PPOAgent.reset()
            # doesn't work and the agent retains memory from previous
            # episodes
            self.lock.acquire()
            self.count_frames = 0
            self.action = None
            self.agent_config.RANDOM_SEED = request.seed
            self.agent = PPOAgent(self.agent_config)
            self.agent.reset()
            self.lock.release()
            return True
        elif request.reset == AgentResetCommands.SHUTDOWN:
            # shut down the agent node
            with self.shutdown_cv:
                self.shutdown = True
                self.shutdown_cv.notify()
                return True
    
    def get_agent_time(self, request):
        r"""
        ROS service handler which returns the time that the agent takes
        to produce the last action.
        :param request: not used
        :returns: agent time
        """
        return self.agent_time

    def depthmsg_to_cv2(self, depth_msg):
        r"""
        Converts a ROS DepthImage message to a Habitat depth observation.
        :param depth_msg: ROS depth message
        :returns: depth observation as a numpy array
        """
        w = depth_msg.width
        h = depth_msg.height
        depth_img = np.reshape(depth_msg.data.astype(np.float32), (h, w))
        return depth_img

    def msgs_to_obs(
        self,
        rgb_msg: Image = None,
        depth_msg: DepthImage = None,
        pointgoal_with_gps_compass_msg: PointGoalWithGPSCompass = None,
    ) -> Dict[str, Any]:
        r"""
        Converts ROS messages into Habitat observations.
        :param rgb_msg: RGB sensor observations packed in a ROS message
        :param depth_msg: Depth sensor observations packed in a ROS message
        :param pointgoal_with_gps_compass_msg: Pointgoal + GPS/Compass sensor
            observations packed in a ROS message
        :return: Habitat observations
        """
        observations = {}

        # Convert RGB message
        if rgb_msg is not None:
            observations["rgb"] = (
                CvBridge().imgmsg_to_cv2(rgb_msg, "passthrough").astype(np.float32)
            )

        # Convert depth message
        if depth_msg is not None:
            observations["depth"] = self.depthmsg_to_cv2(depth_msg)
            # have to manually add channel info
            observations["depth"] = np.expand_dims(observations["depth"], 2).astype(
                np.float32
            )

        # Convert pointgoal + GPS/compass sensor message
        if pointgoal_with_gps_compass_msg is not None:
            observations["pointgoal_with_gps_compass"] = np.asarray(
                [
                    pointgoal_with_gps_compass_msg.distance_to_goal,
                    pointgoal_with_gps_compass_msg.angle_to_goal,
                ]
            ).astype(np.float32)

        return observations

    def action_to_msg(self, action: Dict[str, int]):
        r"""
        Converts action produced by Habitat agent to a ROS message.
        :param action: Action produced by Habitat agent.
        :returns: A ROS message of action or velocity command.
        """
        action_id = action["action"]
        msg = ...  # type: Union[Twist, Int16]

        if self.enable_physics:
            msg = Twist()
            msg.linear.x = 0
            msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.x = 0
            msg.angular.y = 0
            msg.angular.z = 0
            # Convert to Twist message in continuous mode
            if action_id == _DefaultHabitatSimActions.STOP:
                pass
            elif action_id == _DefaultHabitatSimActions.MOVE_FORWARD:
                msg.linear.z = 0.25 / self.control_period
            elif action_id == _DefaultHabitatSimActions.TURN_LEFT:
                msg.angular.y = radians(10.0) / self.control_period
            elif action_id == _DefaultHabitatSimActions.TURN_RIGHT:
                msg.angular.y = radians(-10.0) / self.control_period
        else:
            # Convert to Int16 message in discrete mode
            msg = Int16()
            msg.data = action_id

        return msg

    def callback_rgb(self, rgb_msg, pointgoal_with_gps_compass_msg):
        r"""
        Produces an action or velocity command periodically from RGB
        sensor observation, and publish to respective topics.
        :param rgb_msg: RGB sensor readings in ROS message format.
        :param pointgoal_with_gps_compass_msg: Pointgoal + GPS/Compass readings.
        """
        # convert current_observations from ROS to Habitat format
        observations = self.msgs_to_obs(
            rgb_msg=rgb_msg,
            pointgoal_with_gps_compass_msg=pointgoal_with_gps_compass_msg,
        )

        # produce an action/velocity once the last action has completed
        # and publish to relevant topics
        self.lock.acquire()
        if self.enable_physics:
            self.count_frames += 1
            if self.count_frames == (self.sensor_pub_rate * self.control_period) - 1:
                self.count_frames = 0
                self.action = self.agent.act(observations)
                vel_msg = self.action_to_msg(self.action)
                self.pub.publish(vel_msg)
        else:
            self.action = self.agent.act(observations)
            action_msg = self.action_to_msg(self.action)
            self.pub.publish(action_msg)
        self.lock.release()

    def callback_depth(self, depth_msg, pointgoal_with_gps_compass_msg):
        r"""
        Produce an action or velocity command periodically from depth
        sensor observation, and publish to respective topics.
        :param depth_msg: Depth sensor readings in ROS message format.
        :param pointgoal_with_gps_compass_msg: Pointgoal + GPS/Compass readings.
        """
        # convert current_observations from ROS to Habitat format
        observations = self.msgs_to_obs(
            depth_msg=depth_msg,
            pointgoal_with_gps_compass_msg=pointgoal_with_gps_compass_msg,
        )

        # produce an action/velocity once the last action has completed
        # and publish to relevant topics
        self.lock.acquire()
        if self.enable_physics:
            self.count_frames += 1
            if self.count_frames == (self.sensor_pub_rate * self.control_period) - 1:
                self.count_frames = 0
                self.action = self.agent.act(observations)
                vel_msg = self.action_to_msg(self.action)
                self.pub.publish(vel_msg)
        else:
            self.action = self.agent.act(observations)
            action_msg = self.action_to_msg(self.action)
            self.pub.publish(action_msg)
        self.lock.release()

    def callback_rgbd(self, rgb_msg, depth_msg, pointgoal_with_gps_compass_msg):
        r"""
        Produces an action or velocity command periodically from RGBD
        sensor observations, and publish to respective topics.
        :param rgb_msg: RGB sensor readings in ROS message format.
        :param depth_msg: Depth sensor readings in ROS message format.
        :param pointgoal_with_gps_compass_msg: Pointgoal + GPS/Compass readings.
        """
        # convert current_observations from ROS to Habitat format
        observations = self.msgs_to_obs(
            rgb_msg=rgb_msg,
            depth_msg=depth_msg,
            pointgoal_with_gps_compass_msg=pointgoal_with_gps_compass_msg,
        )

        # produce an action/velocity once the last action has completed
        # and publish to relevant topics
        self.lock.acquire()
        if self.enable_physics:
            self.count_frames += 1
            if self.count_frames == (self.sensor_pub_rate * self.control_period) - 1:
                self.count_frames = 0
                self.action = self.agent.act(observations)
                vel_msg = self.action_to_msg(self.action)
                self.pub.publish(vel_msg)
        else:
            # ------------ log agent time start ------------
            t_agent_start = time.clock()
            # ----------------------------------------------

            self.action = self.agent.act(observations)

            # ------------ log agent time end ------------
            t_agent_end = time.clock()
            self.agent_time = t_agent_end - t_agent_start

            action_msg = self.action_to_msg(self.action)
            self.pub.publish(action_msg)
        self.lock.release()


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="blind",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument("--enable-physics", default=False, action="store_true")
    parser.add_argument(
        "--control-period",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sensor-pub-rate",
        type=float,
        default=10,
    )
    args = parser.parse_args()

    # initialize the agent node
    rospy.init_node("agent_node")
    agent_config = get_default_config()
    agent_config.INPUT_TYPE = args.input_type
    agent_config.MODEL_PATH = args.model_path
    agent_node = HabitatAgentNode(
        agent_config=agent_config,
        enable_physics=args.enable_physics,
        control_period=args.control_period,
        sensor_pub_rate=args.sensor_pub_rate,
    )

    # shutdown the agent node after getting the signal
    with agent_node.shutdown_cv:
        while agent_node.shutdown is False:
            agent_node.shutdown_cv.wait()
        rospy.signal_shutdown("received request to shut down")
