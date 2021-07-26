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
        assert np.linalg.norm(avg_metrics[NumericalMetrics.DISTANCE_TO_GOAL] - 0.140662) < 1e-5
        assert np.linalg.norm(avg_metrics[NumericalMetrics.SPL] - 0.793321) < 1e-5


    def test_generate_video_continuous(self):
        try:
            os.mkdir(
                 "videos/test_habitat_evaluator_continuous/"
            )
        except FileExistsError:
            pass
        
        try:
            os.remove(
                "videos/test_habitat_evaluator_continuous/episode=49-scene=van-gogh-room.glb-seed=7-ckpt=0-distance_to_goal=0.03-success=1.00-spl=0.68.mp4"
            )
        except FileNotFoundError:
            pass

        self.evaluator_continuous.generate_video(
            episode_id="49",
            scene_id="data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
            agent_seed=7,
        )
        assert os.path.isfile(
            "videos/test_habitat_evaluator_continuous/episode=49-scene=van-gogh-room.glb-seed=7-ckpt=0-distance_to_goal=0.14-success=1.00-spl=0.79.mp4"
        )

    def test_generate_map_continuous(self):
        try:
            os.mkdir(
                "habitat_maps/test_habitat_evaluator_continuous/"
            )
        except FileExistsError:
            pass

        # for now we only check if the code runs
        top_down_map = self.evaluator_continuous.generate_map(
            episode_id="49",
            scene_id="data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
            agent_seed=7,
            map_height=400,
        )
        map_img = Image.fromarray(top_down_map, "RGB")
        map_img.save("habitat_maps/test_habitat_evaluator_continuous/test_generate_map_continuous.png")


if __name__ == "__main__":
    unittest.main()
