#!/usr/bin/env python
import argparse
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
)
import rospy
from geometry_msgs.msg import Twist
from ros_x_habitat.msg import PointGoalWithGPSCompass
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from message_filters import TimeSynchronizer
from cv_bridge import CvBridge, CvBridgeError
from habitat.config import Config
from habitat_baselines.agents.ppo_agents import PPOAgent
from habitat.sims.habitat_simulator.actions import _DefaultHabitatSimActions
from math import radians


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
    def __init__(self, agent_config: Config, enable_physics: bool = False, control_period: float = 1.0, sensor_pub_rate: int = 10):
        self.agent = PPOAgent(agent_config)
        self.input_type = agent_config.INPUT_TYPE
        self.enable_physics = enable_physics
        self.sensor_pub_rate = sensor_pub_rate

        # the last action produced from the agent
        self.action = None

        # control_period defined for continuous mode
        if self.enable_physics:
            self.control_period = control_period

        # count the number of frames received from Habitat simulator;
        # reset to 0 every time an action is completed
        # not applicable in discrete mode
        if self.enable_physics:
            self.count_frames = 0

        # publish to command topics
        if self.enable_physics:
            self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        else:
            self.pub = rospy.Publisher("action", Int16, queue_size=10)

        # subscribe to sensor topics
        if self.input_type == "rgb" or self.input_type == "rgbd":
            self.sub_rgb = rospy.Subscriber("rgb", Image)
        if self.input_type == "depth" or self.input_type == "rgbd":
            self.sub_depth = rospy.Subscriber("depth", Image)
        self.sub_pointgoal_with_gps_compass = rospy.Subscriber("pointgoal_with_gps_compass", PointGoalWithGPSCompass)
        
        # filter sensor topics with time synchronizer
        if self.input_type == "rgb":
            self.ts = TimeSynchronizer([self.sub_rgb, self.sub_pointgoal_with_gps_compass], 10)
            self.ts.registerCallback(self.callback_rgb)
        elif self.input_type == "rgbd":
            self.ts = TimeSynchronizer([self.sub_rgb, self.sub_depth, self.sub_pointgoal_with_gps_compass], 10)
            self.ts.registerCallback(self.callback_rgbd)
        else:
            self.ts = TimeSynchronizer([self.sub_depth, self.sub_pointgoal_with_gps_compass], 10)
            self.ts.registerCallback(self.callback_depth)

    def msgs2obs(self, msgs: List) -> Dict[str, Any]:
        r"""
        Decode ROS messages into Habitat observations.
        :param msg: Observations from simulator as a list of messages.
        :return: Habitat observations
        TODO: reduce hard-codeded sensor IDs
        """
        observations = {}
        
        # RGB agent
        if self.input_type == "rgb":
            observations["rgb"] = CvBridge().imgmsg_to_cv2(msgs[0], "rgb8")
            observations["pointgoal_with_gps_compass"] = [msgs[1].distance_to_goal, msgs[1].angle_to_goal]
        # RGBD agent
        elif self.input_type == "rgbd":
            observations["rgb"] = CvBridge().imgmsg_to_cv2(msgs[0], "rgb8")
            observations["depth"] = CvBridge().imgmsg_to_cv2(msgs[1], "rgb8")
            observations["pointgoal_with_gps_compass"] = [msgs[2].distance_to_goal, msgs[2].angle_to_goal]
        # Depth agent
        else:
            observations["depth"] = CvBridge().imgmsg_to_cv2(msgs[0], "rgb8")
            observations["pointgoal_with_gps_compass"] = [msgs[1].distance_to_goal, msgs[1].angle_to_goal]
        
        return observations
    
    def action2msg(self, action: Dict[str, int]):
        r"""
        Converts action produced by Habitat agent to a ROS message.
        :param action: Action produced by Habitat agent.
        :returns: A ROS message of action or velocity command.
        """
        action_id = action["action"]
        msg = ...  # type: Union[Twist, Int16]

        if self.enable_physics:
            msg = Twist()
            # Convert to Twist message in continuous mode
            if action_id == _DefaultHabitatSimActions.STOP:
                msg.linear.x = 0
                msg.linear.y = 0
                msg.linear.z = 0
                msg.angular.x = 0
                msg.angular.y = 0
                msg.angular.z = 0
            elif action_id == _DefaultHabitatSimActions.MOVE_FORWARD:
                msg.linear.x = 0
                msg.linear.y = 0
                msg.linear.z = 0.25 / self.control_period
                msg.angular.x = 0
                msg.angular.y = 0
                msg.angular.z = 0
            elif action_id == _DefaultHabitatSimActions.TURN_LEFT:
                msg.linear.x = 0
                msg.linear.y = 0
                msg.linear.z = 0
                msg.angular.x = 0
                msg.angular.y = radians(10.0) / self.control_period
                msg.angular.z = 0
            elif action_id == _DefaultHabitatSimActions.TURN_RIGHT:
                msg.linear.x = 0
                msg.linear.y = 0
                msg.linear.z = 0
                msg.angular.x = 0
                msg.angular.y = radians(-10.0) / self.control_period
                msg.angular.z = 0
        else:
            # Convert to Int16 message in discrete mode
            msg = Int16()
            msg.data = action_id
        
        return msg

    def callback_rgb(self, img_msg_rgb):
        r"""
        Produce an action or velocity command periodically from RGB
        sensor observation, and publish to respective topics.
        :param img_msg_rgb: RGB sensor readings in ROS message format.
        """
        # convert current_observations from ROS to Habitat format
        observations = self.msgs2obs([img_msg_rgb])

        # produce an action/velocity once the last action has completed
        # and publish to relevant topics
        if self.enable_physics:
            self.count_frames += 1
            if self.count_frames == (self.sensor_pub_rate * self.control_period) - 1:
                self.count_frames = 0
                self.action = self.agent.act(observations)
                vel_msg = self.action2msg(self.action)
                self.pub.publish(vel_msg)
        else:
            self.action = self.agent.act(observations)
            action_msg = self.action2msg(self.action)
            self.pub.publish(action_msg)
    
    def callback_depth(self, img_msg_depth):
        r"""
        Produce an action or velocity command periodically from depth
        sensor observation, and publish to respective topics.
        :param img_msg_depth: Depth sensor readings in ROS message format.
        """
        # convert current_observations from ROS to Habitat format
        observations = self.msgs2obs([img_msg_depth])

        # produce an action/velocity once the last action has completed
        # and publish to relevant topics
        if self.enable_physics:
            self.count_frames += 1
            if self.count_frames == (self.sensor_pub_rate * self.control_period) - 1:
                self.count_frames = 0
                self.action = self.agent.act(observations)
                vel_msg = self.action2msg(self.action)
                self.pub.publish(vel_msg)
        else:
            self.action = self.agent.act(observations)
            action_msg = self.action2msg(self.action)
            self.pub.publish(action_msg)
    
    def callback_rgbd(self, img_msg_rgb, img_msg_depth):
        r"""
        Produce an action or velocity command periodically from RGBD
        sensor observations, and publish to respective topics.
        :param img_msg_rgb: RGB sensor readings in ROS message format.
        :param img_msg_depth: Depth sensor readings in ROS message format.
        """
        # convert current_observations from ROS to Habitat format
        observations = self.msgs2obs([img_msg_rgb, img_msg_depth])

        # produce an action/velocity once the last action has completed
        # and publish to relevant topics
        if self.enable_physics:
            self.count_frames += 1
            if self.count_frames == (self.sensor_pub_rate * self.control_period) - 1:
                self.count_frames = 0
                self.action = self.agent.act(observations)
                vel_msg = self.action2msg(self.action)
                self.pub.publish(vel_msg)
        else:
            self.action = self.agent.act(observations)
            action_msg = self.action2msg(self.action)
            self.pub.publish(action_msg)


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="blind",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument('--enable-physics', default=False, action='store_true')
    parser.add_argument(
        "--control-period",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sensor-pub-rate",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    rospy.init_node("agent_node")
    agent_config = get_default_config()
    agent_config.INPUT_TYPE = args.input_type
    agent_config.MODEL_PATH = args.model_path
    HabitatAgentNode(agent_config=agent_config, enable_physics=args.enable_physics, control_period=args.control_period, sensor_pub_rate=args.sensor_pub_rate)
    rospy.spin()
