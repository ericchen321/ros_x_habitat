PACKAGE_NAME = "ros_x_habitat"


import unittest
import rospy
import rostest
import os
import numpy as np
from mock_env_node import MockHabitatEnvNode
from mock_agent_node import MockHabitatAgentNode
from mock_habitat_ros_evaluator import MockHabitatROSEvaluator
from src.classes.habitat_agent_node import HabitatAgentNode, get_default_config
from src.classes.constants import AgentResetCommands
from src.classes.habitat_env_node import HabitatEnvNode
from subprocess import Popen, call
import shlex


class HabitatROSAgentNodeDiscreteCase(unittest.TestCase):
    r"""
    Test cases for Habitat agent + Habitat sim through ROS.
    """
    def setUp(self):
        # load discrete test data
        self.episode_id = "49"
        self.scene_id = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
        self.num_readings = 27
        self.readings_rgb_discrete = []
        self.readings_depth_discrete = []
        self.readings_ptgoal_with_comp_discrete = []
        self.actions_discrete = []
        for i in range(0, self.num_readings):
            self.readings_rgb_discrete.append(
                np.load(
                    f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/test/obs/rgb-{self.episode_id}-{os.path.basename(self.scene_id)}-{i}.npy"
                )
            )
            self.readings_depth_discrete.append(
                np.load(
                    f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/test/obs/depth-{self.episode_id}-{os.path.basename(self.scene_id)}-{i}.npy"
                )
            )
            self.readings_ptgoal_with_comp_discrete.append(
                np.load(
                    f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/test/obs/pointgoal_with_gps_compass-{self.episode_id}-{os.path.basename(self.scene_id)}-{i}.npy"
                )
            )
            self.actions_discrete.append(
                np.load(
                    f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/test/acts/action-{self.episode_id}-{os.path.basename(self.scene_id)}-{i}.npy"
                )
            )
        
        # define env node publish rate
        self.env_pub_rate = 5.0
    
    def tearDown(self):
        pass

    def test_agent_node_discrete(self):
        # start the agent node
        agent_node_args = shlex.split(f"python classes/habitat_agent_node.py --input-type rgbd --model-path data/checkpoints/v2/gibson-rgbd-best.pth --sensor-pub-rate {self.env_pub_rate}")
        Popen(agent_node_args)

        # init mock env node
        rospy.init_node("mock_env_node")
        mock_env_node = MockHabitatEnvNode(enable_physics=False)

        # reset the mock env
        mock_env_node.reset()

        # publish pre-saved sensor observations
        r = rospy.Rate(self.env_pub_rate)
        for i in range(0, self.num_readings):
            mock_env_node.publish_sensor_observations(
                self.readings_rgb_discrete[i],
                self.readings_depth_discrete[i],
                self.readings_ptgoal_with_comp_discrete[i]
            )
            mock_env_node.check_command(self.actions_discrete[i])
            r.sleep()
        
        # shut down the agent node
        rospy.wait_for_service("reset_agent")
        resp = mock_env_node.reset_agent(int(AgentResetCommands.SHUTDOWN))
        assert resp.done
        
        # shut down the mock env node
        rospy.signal_shutdown("test agent node in setting 2 done")


def main():
    rostest.rosrun(PACKAGE_NAME, "tests_habitat_ros_agent_node_discrete", HabitatROSAgentNodeDiscreteCase)

if __name__ == "__main__":
    main()