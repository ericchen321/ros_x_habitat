import shlex
import unittest
from subprocess import Popen, call
import os
import rospy
import rostest
import numpy as np
from src.test.data.data import TestHabitatROSData
from src.constants.constants import NumericalMetrics, PACKAGE_NAME
from src.evaluators.habitat_ros_evaluator import HabitatROSEvaluator


class HabitatROSEvaluatorDiscreteCase(unittest.TestCase):
    r"""
    Test cases for Habitat agent + Habitat sim through ROS.
    """

    def setUp(self):
        # load discrete test data
        self.episode_id_request = TestHabitatROSData.test_evaluator_episode_id_request
        self.episode_id_response = TestHabitatROSData.test_evaluator_episode_id_response
        self.scene_id = TestHabitatROSData.test_evaluator_scene_id
        self.log_dir = TestHabitatROSData.test_evaluator_log_dir
        self.agent_seed = TestHabitatROSData.test_evaluator_agent_seed
        self.config_paths = TestHabitatROSData.test_evaluator_config_paths
        self.input_type = TestHabitatROSData.test_evaluator_input_type
        self.model_path = TestHabitatROSData.test_evaluator_model_path
        self.distance_to_goal = TestHabitatROSData.test_evaluator_distance_to_goal
        self.success = TestHabitatROSData.test_evaluator_success
        self.spl = TestHabitatROSData.test_evaluator_spl

        # create log dirs
        os.makedirs(name=self.log_dir, exist_ok=True)

        # clean up log files from previous runs
        try:
            os.remove(
                f"{self.log_dir}/episode={self.episode_id_response}-scene={self.scene_id}.log"
            )
        except FileNotFoundError:
            pass

    def tearDown(self):
        pass

    def test_evaluator_node_discrete(self):
        # start the mock agent node
        agent_node_args = shlex.split(
            f"python src/test/test_habitat_ros/mock_agent_node.py"
        )
        Popen(agent_node_args)

        # start the mock env node
        mock_env_node_args = shlex.split(
            f"python src/test/test_habitat_ros/mock_env_node.py"
        )
        Popen(mock_env_node_args)

        # start the evaluator
        evaluator = HabitatROSEvaluator(
            config_paths=self.config_paths,
            input_type=self.input_type,
            model_path=self.model_path,
            enable_physics=False,
            node_name="habitat_ros_evaluator_node_under_test",
            do_not_start_nodes=True,
        )

        # test HabitatROSEvaluator.evaluate()
        dict_of_metrics = evaluator.evaluate(
            episode_id_last=self.episode_id_request,
            scene_id_last=self.scene_id,
            log_dir=self.log_dir,
            agent_seed=self.agent_seed,
        )
        # check metrics from the mock env node
        assert (
            np.linalg.norm(
                dict_of_metrics[f"{self.episode_id_response},{self.scene_id}"][
                    NumericalMetrics.DISTANCE_TO_GOAL
                ]
                - self.distance_to_goal
            )
            < 1e-5
        )
        assert (
            np.linalg.norm(
                dict_of_metrics[f"{self.episode_id_response},{self.scene_id}"][
                    NumericalMetrics.SUCCESS
                ]
                - self.success
            )
            < 1e-5
        )
        assert (
            np.linalg.norm(
                dict_of_metrics[f"{self.episode_id_response},{self.scene_id}"][
                    NumericalMetrics.SPL
                ]
                - self.spl
            )
            < 1e-5
        )
        # check metrics from the agent node
        assert (
            np.linalg.norm(
                dict_of_metrics[f"{self.episode_id_response},{self.scene_id}"][
                    NumericalMetrics.AGENT_TIME
                ]
                - 0.0
            )
            < 1e-5
        )

        # check if the episodic log file is created
        assert os.path.isfile(
            f"{self.log_dir}/episode={self.episode_id_response}-scene={self.scene_id}.log"
        )

        # test HabitatROSEvaluator.shutdown_env_node()
        evaluator.shutdown_env_node()

        # test HabitatROSEvaluator.shutdown_agent_node()
        evaluator.shutdown_agent_node()


def main():
    rostest.rosrun(
        PACKAGE_NAME,
        "tests_habitat_ros_evaluator_discrete",
        HabitatROSEvaluatorDiscreteCase,
    )


if __name__ == "__main__":
    main()
