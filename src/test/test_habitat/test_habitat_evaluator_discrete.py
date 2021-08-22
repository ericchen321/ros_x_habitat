import os
import unittest
import numpy as np
from PIL import Image
from src.constants.constants import NumericalMetrics
from src.evaluators.habitat_evaluator import HabitatEvaluator


class TestHabitatEvaluatorDiscreteCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator_discrete = HabitatEvaluator(
            config_paths="configs/pointnav_rgbd_val.yaml",
            input_type="rgbd",
            model_path="data/checkpoints/v2/gibson-rgbd-best.pth",
            enable_physics=False,
        )

    def test_compute_avg_metrics(self):
        spls = [0.2, 0.4, 0.0, 0.0]
        distance_to_goal = [0.01, 0.09, float("nan"), -1.0 * float("inf")]
        dict_of_metrics = {
            "episode-0": {
                NumericalMetrics.SPL: spls[0],
                NumericalMetrics.DISTANCE_TO_GOAL: distance_to_goal[0],
            },
            "episode-1": {
                NumericalMetrics.SPL: spls[1],
                NumericalMetrics.DISTANCE_TO_GOAL: distance_to_goal[1],
            },
            "episode-2": {
                NumericalMetrics.SPL: spls[2],
                NumericalMetrics.DISTANCE_TO_GOAL: distance_to_goal[2],
            },
            "episode-3": {
                NumericalMetrics.SPL: spls[3],
                NumericalMetrics.DISTANCE_TO_GOAL: distance_to_goal[3],
            },
        }
        avg_metrics = self.evaluator_discrete.compute_avg_metrics(dict_of_metrics)
        assert np.linalg.norm(avg_metrics[NumericalMetrics.SPL] - 0.3) < 1e-5
        assert (
            np.linalg.norm(avg_metrics[NumericalMetrics.DISTANCE_TO_GOAL] - 0.05) < 1e-5
        )

    def test_evaluate_one_episode_discrete(self):
        metrics_dict = self.evaluator_discrete.evaluate(
            episode_id_last="48",
            scene_id_last="data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
            log_dir="logs",
            agent_seed=7,
        )
        metrics_dict = self.evaluator_discrete.extract_metrics(
            metrics_dict, [NumericalMetrics.DISTANCE_TO_GOAL, NumericalMetrics.SPL]
        )
        avg_metrics = self.evaluator_discrete.compute_avg_metrics(metrics_dict)
        assert (
            np.linalg.norm(avg_metrics[NumericalMetrics.DISTANCE_TO_GOAL] - 0.026777)
            < 1e-5
        )
        assert np.linalg.norm(avg_metrics[NumericalMetrics.SPL] - 0.682441) < 1e-5

    def test_generate_video_one_episode_discrete(self):
        os.makedirs(
            name="videos/test_habitat_evaluator_discrete/one_episode/", exist_ok=True
        )
        self.evaluator_discrete.config.defrost()
        self.evaluator_discrete.config.VIDEO_DIR = (
            "videos/test_habitat_evaluator_discrete/one_episode/"
        )
        self.evaluator_discrete.config.freeze()

        episode_ids = ["3"]
        scene_ids = ["data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"]

        # eye-ball check produced videos
        self.evaluator_discrete.generate_videos(
            episode_ids=episode_ids, scene_ids=scene_ids, agent_seed=7
        )

    def test_generate_video_two_episodes_discrete(self):
        os.makedirs(
            name="videos/test_habitat_evaluator_discrete/two_episodes/", exist_ok=True
        )
        self.evaluator_discrete.config.defrost()
        self.evaluator_discrete.config.VIDEO_DIR = (
            "videos/test_habitat_evaluator_discrete/two_episodes/"
        )
        self.evaluator_discrete.config.freeze()

        episode_ids = ["0", "4"]
        scene_ids = [
            "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        ]

        # eye-ball check produced videos
        self.evaluator_discrete.generate_videos(
            episode_ids=episode_ids, scene_ids=scene_ids, agent_seed=7
        )

    def test_generate_maps_one_episode_discrete(self):
        os.makedirs(
            name="habitat_maps/test_habitat_evaluator_discrete/one_episode/",
            exist_ok=True,
        )

        episode_ids = ["3"]
        scene_ids = ["data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"]

        top_down_maps = self.evaluator_discrete.generate_maps(
            episode_ids=episode_ids,
            scene_ids=scene_ids,
            agent_seed=7,
            map_height=400,
        )

        # check iff only one map's generated
        assert len(top_down_maps) == 1

        # eye-ball check produced maps
        for episode_id, scene_id in zip(episode_ids, scene_ids):
            map_img = Image.fromarray(top_down_maps[f"{episode_id},{scene_id}"], "RGB")
            map_img.save(
                f"habitat_maps/test_habitat_evaluator_discrete/one_episode/episode={episode_id}-scene={os.path.basename(scene_id)}.png"
            )

    def test_generate_maps_two_episodes_discrete(self):
        os.makedirs(
            name="habitat_maps/test_habitat_evaluator_discrete/two_episodes/",
            exist_ok=True,
        )

        episode_ids = ["0", "4"]
        scene_ids = [
            "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        ]

        top_down_maps = self.evaluator_discrete.generate_maps(
            episode_ids=episode_ids,
            scene_ids=scene_ids,
            agent_seed=7,
            map_height=400,
        )

        # check iff two maps are generated
        assert len(top_down_maps) == 2

        # eye-ball check produced maps
        for episode_id, scene_id in zip(episode_ids, scene_ids):
            map_img = Image.fromarray(top_down_maps[f"{episode_id},{scene_id}"], "RGB")
            map_img.save(
                f"habitat_maps/test_habitat_evaluator_discrete/two_episodes/episode={episode_id}-scene={os.path.basename(scene_id)}.png"
            )

    def test_get_original_maps_two_episodes_discrete(self):
        os.makedirs(
            name="habitat_maps/test_get_original_maps/two_episodes/", exist_ok=True
        )

        episode_ids = ["0", "4"]
        scene_ids = [
            "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        ]

        top_down_maps = self.evaluator_discrete.get_blank_maps(
            episode_ids=episode_ids,
            scene_ids=scene_ids,
            map_height=400,
        )

        # check iff two maps are generated
        assert len(top_down_maps) == 2

        # eye-ball check produced maps
        for episode_id, scene_id in zip(episode_ids, scene_ids):
            map_img = Image.fromarray(top_down_maps[f"{episode_id},{scene_id}"], "RGB")
            map_img.save(
                f"habitat_maps/test_get_original_maps/two_episodes/episode={episode_id}-scene={os.path.basename(scene_id)}.pgm"
            )


if __name__ == "__main__":
    unittest.main()
