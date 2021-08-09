import os
import shlex
import unittest
from subprocess import Popen

import numpy as np
import rostest

from mock_habitat_ros_evaluator import MockHabitatROSEvaluator
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
from src.constants.constants import NumericalMetrics, PACKAGE_NAME
from src.test.data.data import TestHabitatROSData


class HabitatROSEnvNodeDiscreteCase(unittest.TestCase):
    r"""
    Test cases for Habitat agent + Habitat sim through ROS.
    """

    def setUp(self):
        # define env node pub rate
        self.env_pub_rate = 5.0

        # define the env node's name
        self.env_node_under_test_name = "env_node_under_test"

    def tearDown(self):
        pass

    def test_env_node_discrete(self):
        # start the env node
        env_node_args = shlex.split(
            f"python src/nodes/habitat_env_node.py --node-name {self.env_node_under_test_name} --task-config {TestHabitatROSData.test_acts_and_obs_discrete_task_config} --sensor-pub-rate {self.env_pub_rate}"
        )
        Popen(env_node_args)

        # start the mock agent node
        agent_node_args = shlex.split(
            f"python src/test/test_habitat_ros/mock_agent_node.py"
        )
        Popen(agent_node_args)

        # init the mock evaluator node
        mock_evaluator = MockHabitatROSEvaluator(
            node_name="mock_habitat_ros_evaluator_node",
            env_node_name=self.env_node_under_test_name,
            agent_node_name="mock_agent_node",
        )

        # mock-eval one episode
        dict_of_metrics = mock_evaluator.evaluate(
            str(int(TestHabitatROSData.test_acts_and_obs_discrete_episode_id) - 1),
            TestHabitatROSData.test_acts_and_obs_discrete_scene_id,
        )
        metrics = HabitatSimEvaluator.compute_avg_metrics(dict_of_metrics)
        print(f"success: {metrics[NumericalMetrics.SUCCESS]}")
        print(f"spl: {metrics[NumericalMetrics.SPL]}")
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
