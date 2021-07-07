PACKAGE_NAME = "ros_x_habitat"


import unittest
import rospy
import rostest
import os
import numpy as np
from mock_env_node import MockHabitatEnvNode
from mock_agent_node import MockHabitatAgentNode
from mock_habitat_ros_evaluator import MockHabitatROSEvaluator
from src.nodes.habitat_agent_node import HabitatAgentNode, get_default_config
from src.constants.constants import AgentResetCommands
from src.nodes.habitat_env_node import HabitatEnvNode
from src.evaluators.habitat_ros_evaluator import HabitatROSEvaluator
from subprocess import Popen, call
import shlex


class HabitatROSEvaluatorDiscreteCase(unittest.TestCase):
    r"""
    Test cases for Habitat agent + Habitat sim through ROS.
    """
    def setUp(self):
        # load discrete test data
        self.episode_id = "47"
        self.scene_id = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
        self.distance_to_goal = 0.1,
        self.success = 1.0,
        self.spl = 0.25
    
    def tearDown(self):
        pass

    def test_agent_node_discrete(self):
        # start the agent node
        agent_node_args = shlex.split(f"python src/nodes/habitat_agent_node.py --input-type rgbd --model-path data/checkpoints/v2/gibson-rgbd-best.pth --sensor-pub-rate 5.0")
        Popen(agent_node_args)

        # init the mock env node
        rospy.init_node("mock_env_node")
        MockHabitatEnvNode(
            enable_physics = False,
            episode_id = self.episode_id,
            scene_id = self.scene_id,
        )
        
        # start the evaluator
        evaluator_args = shlex.split(f"python eval_habitat_ros.py --input-type rgbd --model-path data/checkpoints/v2/gibson-rgbd-best.pth --task-config configs/pointnav_rgbd_val.yaml --episode-id {self.episode_id} --scene-id={self.scene_id} --sensor-pub-rate=5.0 --do-not-start-nodes-from-evaluator --log-dir=logs/test_habitat_ros_evaluator_discrete/ --tb-dir=tb/test_habitat_ros_evaluator_discrete/")
        call(evaluator_args)

        # shut down the agent node
        shutdown_agent_args = shlex.split(f"rosnode kill agent_node")
        call(shutdown_agent_args)
        


def main():
    rostest.rosrun(PACKAGE_NAME, "tests_habitat_ros_evaluator_discrete", HabitatROSEvaluatorDiscreteCase)

if __name__ == "__main__":
    main()