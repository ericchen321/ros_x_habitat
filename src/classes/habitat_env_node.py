#!/usr/bin/env python
from typing import Optional
import rospy
from rospy_message_converter import message_converter
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from classes.habitat_eval_rlenv import HabitatEvalRLEnv
from habitat.config.default import get_config


class HabitatEnvNode:
    def __init__(
        self, config_paths: Optional[str] = None, enable_physics: bool = False
    ):
        self.config = get_config(config_paths)
        self.enable_physics = enable_physics
        self.env = HabitatEvalRLEnv(
            config=self.config, enable_physics=self.enable_physics
        )
        self.observations = self.env.reset()
        self.pub = rospy.Publisher("observations", String, queue_size=10)
        self.sub = rospy.Subscriber("action", String, self.callback)
        if self.enable_physics:
            self.sub = rospy.Subscriber("cmd_vel", Twist, self.callback)
        # TODO: publish initial observations after __init__

    def callback(self, msg):
        agent_decision = msg.data
        # TODO: convert agent's decision from a String or a Twist to appropriate Action class that the Env accepts
        (self.observations, _, _, _) = self.env.step(agent_decision)
        # TODO: convert observations to Python dictionary then to a ROS String
        # e.g dictionary = { 'data': 'Howdy' }
        # message = message_converter.convert_dictionary_to_ros_message('std_msgs/String', dictionary)
        new_msg = String()
        new_msg.data = ""

        self.pub.publish(new_msg)


if __name__ == "__main__":
    rospy.init_node("env_node")
    # TODO: add CLI args parsing to support launching individually
    HabitatEnvNode(None)
    rospy.spin()
