#!/usr/bin/env python
import rospy
from rospy_message_converter import message_converter
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from habitat.config import Config
from habitat_baselines.agents.ppo_agents import PPOAgent


class HabitatAgentNode:
    def __init__(self, agent_config: Config, enable_physics: bool = False):
        self.agent = PPOAgent(agent_config)
        self.decision = None
        self.enable_physics = enable_physics
        self.pub = rospy.Publisher("action", String, queue_size=10)
        if self.enable_physics:
            self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.sub = rospy.Subscriber("observations", String, self.callback)

    def callback(self, msg):
        current_observations = message_converter.convert_ros_message_to_dictionary(msg)
        # TODO: convert current_observations from a Python dict to habitat.core.simulator.Observations class
        self.decision = self.agent.act(current_observations)
        # TODO: convert decision to either a String (discrete mode) or a Twist (physics mode)
        new_msg = String()
        if self.enable_physics:
            new_msg = Twist()
            new_msg.linear.x = 0
            new_msg.linear.y = 0
            new_msg.linear.z = 0
            new_msg.angular.x = 0
            new_msg.angular.y = 0
            new_msg.angular.z = 0
        else:
            new_msg.data = ""

        self.pub.publish(new_msg)


if __name__ == "__main__":
    rospy.init_node("agent_node")
    # TODO: add CLI args parsing to support launching individually
    HabitatAgentNode(None)
    rospy.spin()
