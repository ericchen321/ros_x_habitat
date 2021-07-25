PACKAGE_NAME = "ros_x_habitat"


import os
import shlex
import unittest
from subprocess import Popen

import numpy as np
import rospy
import rostest

from mock_env_node import MockHabitatEnvNode
from ros_x_habitat.srv import ResetAgent
from src.constants.constants import AgentResetCommands
from src.test.data.data import TestHabitatROSData


class HabitatROSAgentNodeDiscreteCase(unittest.TestCase):
    r"""
    Test cases for Habitat agent + Habitat sim through ROS.
    """

    def setUp(self):
        # define env node publish rate
        self.env_pub_rate = 5.0

        # set up agent reset service client
        self.reset_agent = rospy.ServiceProxy("reset_agent", ResetAgent)

    def tearDown(self):
        pass

    def test_agent_node_discrete(self):
        # start the agent node
        agent_node_args = shlex.split(
            f"python src/nodes/habitat_agent_node.py --node-name agent_node_under_test --input-type rgbd --model-path data/checkpoints/v2/gibson-rgbd-best.pth --sensor-pub-rate {self.env_pub_rate}"
        )
        Popen(agent_node_args)

        # init mock env node
        rospy.init_node("mock_env_node")
        mock_env_node = MockHabitatEnvNode(enable_physics_sim=False)

        # reset the agent
        rospy.wait_for_service("reset_agent")
        try:
            resp = self.reset_agent(int(AgentResetCommands.RESET), 7)
            assert resp.done
        except rospy.ServiceException:
            raise rospy.ServiceException

        # reset the mock env
        mock_env_node.reset()

        # publish pre-saved sensor observations
        r = rospy.Rate(self.env_pub_rate)
        for i in range(0, TestHabitatROSData.test_acts_and_obs_discrete_num_obs):
            mock_env_node.publish_sensor_observations(
                TestHabitatROSData.test_acts_and_obs_discrete_obs_rgb[i],
                TestHabitatROSData.test_acts_and_obs_discrete_obs_depth[i],
                TestHabitatROSData.test_acts_and_obs_discrete_obs_ptgoal_with_comp[i],
            )
            mock_env_node.check_command(TestHabitatROSData.test_acts_and_obs_discrete_acts[i])
            r.sleep()

        # shut down the agent
        rospy.wait_for_service("reset_agent")
        try:
            resp = self.reset_agent(int(AgentResetCommands.SHUTDOWN), 0)
            assert resp.done
        except rospy.ServiceException:
            raise rospy.ServiceException

        # shut down the mock env node
        rospy.signal_shutdown("test agent node in setting 2 done")


def main():
    rostest.rosrun(
        PACKAGE_NAME,
        "tests_habitat_ros_agent_node_discrete",
        HabitatROSAgentNodeDiscreteCase,
    )


if __name__ == "__main__":
    main()
