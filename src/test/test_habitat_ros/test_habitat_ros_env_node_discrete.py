PACKAGE_NAME = "ros_x_habitat"


import os
import shlex
import unittest
from subprocess import Popen

import numpy as np
import rostest

from mock_habitat_ros_evaluator import MockHabitatROSEvaluator
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
from src.constants.constants import NumericalMetrics


class HabitatROSEnvNodeDiscreteCase(unittest.TestCase):
    r"""
    Test cases for Habitat agent + Habitat sim through ROS.
    """

    def setUp(self):
        # load discrete test data
        self.episode_id = "49"
        self.scene_id = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
        self.num_readings = 47
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

    def test_env_node_discrete(self):
        # start the env node
        env_node_args = shlex.split(
            f"python src/nodes/habitat_env_node.py --node-name env_node_under_test --task-config configs/pointnav_rgbd_val.yaml --sensor-pub-rate {self.env_pub_rate}"
        )
        Popen(env_node_args)

        # start the mock agent node
        agent_node_args = shlex.split(
            f"python src/test/test_habitat_ros/mock_agent_node.py --sensor-pub-rate {self.env_pub_rate}"
        )
        Popen(agent_node_args)

        # init the mock evaluator node
        mock_evaluator = MockHabitatROSEvaluator()

        # mock-eval one episode
        dict_of_metrics = mock_evaluator.evaluate("48", self.scene_id)
        metrics = HabitatSimEvaluator.compute_avg_metrics(dict_of_metrics)
        assert (
            np.linalg.norm(metrics[NumericalMetrics.SUCCESS] - 1.0) < 1e-5
            and np.linalg.norm(metrics[NumericalMetrics.SPL] - 0.68244) < 1e-5
        )
        
        # shut down nodes
        mock_evaluator.shutdown_agent_node()
        mock_evaluator.shutdown_env_node()

def main():
    rostest.rosrun(
        PACKAGE_NAME,
        "tests_habitat_ros_env_node_discrete",
        HabitatROSEnvNodeDiscreteCase,
    )


if __name__ == "__main__":
    main()
