import os
import unittest

import numpy as np
from PIL import Image
from src.constants.constants import NumericalMetrics
from src.evaluators.habitat_evaluator import HabitatEvaluator


class TestHabitatEvaluatorContinuousCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator_continuous = HabitatEvaluator(
            config_paths="configs/pointnav_rgbd_with_physics.yaml",
            input_type="rgbd",
            model_path="data/checkpoints/v2/gibson-rgbd-best.pth",
            enable_physics=True,
        )

    def test_evaluate_one_episode_continuous(self):
        metrics_list = self.evaluator_continuous.evaluate(
            episode_id_last="48",
            scene_id_last="data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
            log_dir="logs",
            agent_seed=7,
        )
        avg_metrics = self.evaluator_continuous.compute_avg_metrics(metrics_list)
        assert (
            np.linalg.norm(avg_metrics[NumericalMetrics.DISTANCE_TO_GOAL] - 0.140662)
            < 1e-5
        )
        assert np.linalg.norm(avg_metrics[NumericalMetrics.SPL] - 0.793321) < 1e-5


if __name__ == "__main__":
    unittest.main()
