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
from src.constants.constants import AgentResetCommands, PACKAGE_NAME, ServiceNames
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
        node_name: str,
        agent_config: Config,
        sensor_pub_rate: float = 5.0,
    ):
        r"""
        Instantiates a node incapsulating a Habitat agent.
        :param node_name: name of the node
        :param agent_config: agent configuration
        :sensor_pub_rate: the rate at which Gazebo (or some other ROS-based
            sim) publishes sensor observations
        """
        # initialize the node
        self.node_name = node_name
        rospy.init_node(self.node_name)

        self.agent_config = agent_config
        self.sensor_pub_rate = float(sensor_pub_rate)

        # agent publish and subscribe queue size
        # TODO: make them configurable by constructor argument
        self.sub_queue_size = 10
        self.pub_queue_size = 10

        # lock guarding access to self.action, self.count_steps,
        # self.agent and self.t_agent_elapsed
        self.lock = Lock()
        with self.lock:
            # the last action produced from the agent
            self.action = None

            # declare an agent instance
            self.agent = PPOAgent(agent_config)

            # for timing
            self.count_steps = None
            self.t_agent_elapsed = 0

        # shutdown triggers the node to be shutdown. Guarded by shutdown_cv
        self.shutdown_cv = Condition()
        with self.shutdown_cv:
            self.shutdown = False

        # set up logger
        self.logger = utils_logging.setup_logger(self.node_name)

        # establish agent reset service server
        self.reset_service = rospy.Service(
            f"{PACKAGE_NAME}/{self.node_name}/{ServiceNames.RESET_AGENT}",
            ResetAgent,
            self.reset_agent,
        )

        # establish agent time service server
        self.agent_time_service = rospy.Service(
            f"{PACKAGE_NAME}/{self.node_name}/{ServiceNames.GET_AGENT_TIME}",
            GetAgentTime,
            self.get_agent_time,
        )

        # publish to command topics
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
            with self.lock:
                self.count_steps = 0
                self.t_agent_elapsed = 0.0
                self.action = None
                self.agent_config.RANDOM_SEED = request.seed
                self.agent = PPOAgent(self.agent_config)
                self.agent.reset()
            return True
        elif request.reset == AgentResetCommands.SHUTDOWN:
            # shut down the agent node
            with self.shutdown_cv:
                self.shutdown = True
                self.shutdown_cv.notify()
                return True

    def get_agent_time(self, request):
        r"""
        ROS service handler which returns the average time for the agent
        to produce an action since the last reset.
        :param request: not used
        :returns: agent time
        """
        with self.lock:
            avg_agent_time = self.t_agent_elapsed / self.count_steps
        return avg_agent_time

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
        :param action: Discrete action produced by Habitat agent.
        :returns: A ROS message of action command.
        """
        action_id = action["action"]
        msg = ...  # type: Union[Twist, Int16]
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
        with self.lock:
            # TODO: implement time logging for rgb callback
            self.action = self.agent.act(observations)
            self.count_steps += 1
            action_msg = self.action_to_msg(self.action)
            self.pub.publish(action_msg)

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
        with self.lock:
            # TODO: implement time logging for depth callback
            self.action = self.agent.act(observations)
            self.count_steps += 1
            action_msg = self.action_to_msg(self.action)
            self.pub.publish(action_msg)

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
        with self.lock:
            # ------------ log agent time start ------------
            t_agent_start = time.clock()
            # ----------------------------------------------

            self.action = self.agent.act(observations)

            # ------------ log agent time end ------------
            t_agent_end = time.clock()
            self.t_agent_elapsed += t_agent_end - t_agent_start
            self.count_steps += 1

            action_msg = self.action_to_msg(self.action)
            self.pub.publish(action_msg)

    def spin_until_shutdown(self):
        r"""
        Put the current thread to sleep. Wake up and exit upon shutdown.
        """
        # shutdown the agent node after getting the signal
        with self.shutdown_cv:
            while self.shutdown is False:
                self.shutdown_cv.wait()
            rospy.signal_shutdown("received request to shut down")


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-name", default="agent_node", type=str)
    parser.add_argument(
        "--input-type",
        default="blind",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument(
        "--sensor-pub-rate",
        type=float,
        default=10,
    )
    args = parser.parse_args()
    agent_config = get_default_config()
    agent_config.INPUT_TYPE = args.input_type
    agent_config.MODEL_PATH = args.model_path

    # instantiate agent node
    agent_node = HabitatAgentNode(
        node_name=args.node_name,
        agent_config=agent_config,
        sensor_pub_rate=args.sensor_pub_rate,
    )

    # spins until receiving the shutdown signal
    agent_node.spin_until_shutdown()


if __name__ == "__main__":
    main()
