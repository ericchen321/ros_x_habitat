import numpy as np
from habitat.config import Config
from habitat.config.default import get_config
from typing import List, Tuple, Dict
from collections import defaultdict
from src.evaluators.evaluator import Evaluator


class HabitatSimEvaluator(Evaluator):
    r"""Abstract class for evaluating an agent in a Habitat simulation
    environment either with or without physics.

    Users should instantiate subclasses to 'HabitatSimEvaluator' for
    evaluation.
    """

    def __init__(
        self,
        config_paths: str,
        input_type: str,
        model_path: str,
        enable_physics: bool = False,
    ):
        # store experiment settings
        self.config = get_config(config_paths)
        self.input_type = input_type
        self.model_path = model_path
        self.enable_physics = enable_physics

    @classmethod
    def overwrite_simulator_config(cls, config):
        r"""
        Overwrite simulator and task configurations when physics is enabled.
        :param config: environment config to be overwritten.
        """
        for k in config.PHYSICS_SIMULATOR.keys():
            if isinstance(config.PHYSICS_SIMULATOR[k], Config):
                for inner_k in config.PHYSICS_SIMULATOR[k].keys():
                    config.SIMULATOR[k][inner_k] = config.PHYSICS_SIMULATOR[k][inner_k]
            else:
                config.SIMULATOR[k] = config.PHYSICS_SIMULATOR[k]
        try:
            from habitat.sims.habitat_simulator.habitat_simulator import (
                HabitatSim,
            )
            from src.sims.habitat_physics_simulator import HabitatPhysicsSim
            from src.tasks.habitat_physics_task import PhysicsNavigationTask
            from habitat.sims.habitat_simulator.actions import (
                HabitatSimV1ActionSpaceConfiguration,
            )
        except ImportError as e:
            print("Import HSIM failed")
            raise e

        return

    @classmethod
    def compute_avg_metrics(
        cls,
        dict_of_metrics: Dict[str, Dict[str, float]],
    ) -> Dict:
        r"""
        Compute average metrics from a list of metrics.
        :param dict_of_metrics: a collection of metrics for which a key
            identifies an episode, a value contains a dictionary of metrics
            from that episode. Each metrics dictionary should contain only
            numerically-valued metrics.
        :returns: average metrics as a dictionary.
        """
        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0
        for _, metrics in dict_of_metrics.items():
            # add metrics from one episode
            # first we drop any episode that contains nan, inf or -inf
            # values
            contain_invalid_values = False
            for _, metric_val in metrics.items():
                if np.isinf(metric_val) or np.isnan(metric_val):
                    contain_invalid_values = True
            # we only count episodes in which all metrics are valid
            if not contain_invalid_values:
                for metric_name, metric_val in metrics.items():
                    agg_metrics[metric_name] += metric_val
                count_episodes += 1
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        return avg_metrics

    @classmethod
    def extract_metrics(
        cls,
        dict_of_metrics: Dict[str, Dict[str, float]],
        metric_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        r"""
        Extract specified metrics from dict_of_metrics and return.
        :param dict_of_metrics: A dictionary of metrics from many episodes.
        :param metric_names: Names of metrics to extract. Each metric must
            have already been collected.
        :return: A dictionary of metrics from the same episodes, but contains
            only metrics with names specified in `metric_names`.
        """
        new_dict_of_metrics = {}
        for episode_identifier, episode_metrics in dict_of_metrics.items():
            new_dict_of_metrics[episode_identifier] = {
                metric_name: episode_metrics[metric_name]
                for metric_name in metric_names
            }
        return new_dict_of_metrics

    @classmethod
    def compute_pairwise_diff_of_metrics(
        cls,
        dict_of_metrics_baseline: Dict[str, Dict[str, float]],
        dict_of_metrics_compared: Dict[str, Dict[str, float]],
        metric_names: List[str],
        compute_percentage: bool,
    ) -> Dict[str, Dict[str, float]]:
        r"""
        Compute pairwise difference in metrics between `dict_of_metrics_baseline`
        and `dict_of_metrics_compared`. Require each metric being a scalar value.
        :param dict_of_metrics_baseline: metrics collected under the baseline setting
        :param dict_of_metrics_compared: metrics collected under the setting of our
            interest
        :param metric_names: names of the metrics to compute differences
        :param compute_percentage: if compute the difference in percentage or not
        :return a dictionary of per-episode metrics, but each metric's value is the
            (percentage) pair-wise difference
        """
        # precondition check
        assert len(dict_of_metrics_baseline) == len(dict_of_metrics_compared)

        pairwise_diff_dict_of_metrics = {}
        for (
            episode_identifier,
            episode_metrics_baseline,
        ) in dict_of_metrics_baseline.items():
            # iterate over episodes
            episode_metrics_compared = dict_of_metrics_compared[episode_identifier]
            pairwise_diff_dict_of_metrics[episode_identifier] = {}
            for metric_name in metric_names:
                # compute difference per metric
                if compute_percentage:
                    if (
                        np.linalg.norm(episode_metrics_baseline[metric_name] - 0.0)
                        < 1e-5
                    ):
                        # handle divide-by-zero - register invalid % change
                        metric_diff = float("nan")
                    else:
                        metric_diff = (
                            (
                                episode_metrics_compared[metric_name]
                                - episode_metrics_baseline[metric_name]
                            )
                            / episode_metrics_baseline[metric_name]
                        ) * 100.0
                else:
                    metric_diff = (
                        episode_metrics_compared[metric_name]
                        - episode_metrics_baseline[metric_name]
                    )
                pairwise_diff_dict_of_metrics[episode_identifier][
                    metric_name
                ] = metric_diff

        return pairwise_diff_dict_of_metrics

    def generate_videos(
        self,
        episode_ids: List[str],
        scene_ids: List[str],
        agent_seed: int = 7,
        *args,
        **kwargs,
    ) -> None:
        r"""
        Evaluate the episode of given episode ID's and scene ID's, and save their videos
        to <video_dir>/.

        :param episode_ids: List of episode ID's
        :param scene_id: List of scene ID's
        :param agent_seed: seed for initializing agent
        """
        raise NotImplementedError

    def generate_maps(
        self,
        episode_ids: List[str],
        scene_ids: List[str],
        agent_seed: int,
        map_height: int,
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        r"""
        Evaluate the episode of given episode ID's and scene ID's, with agent initialized
        by the given seed. Return the top-down maps from each episode.

        :param episode_ids: List of episode ID's
        :param scene_id: List of scene ID's
        :param agent_seed: seed for initializing agent
        :param map_height: desired height of the map

        :returns: Dictionary of Top-down maps with initial/goal position, shortest path and
            actual path.
        """
        raise NotImplementedError

    def get_blank_maps(
        self,
        episode_ids: List[str],
        scene_ids: List[str],
        map_height: int,
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        r"""
        Return the original (no agent marker, paths, etc.) top-down maps of the specified
        episodes.

        :param episode_ids: List of episode ID's
        :param scene_id: List of scene ID's
        :param map_height: desired height of the map

        :returns: Dictionary of blank Top-down maps with initial/goal position, shortest
            path and actual path.
        """
        raise NotImplementedError

    def evaluate_and_get_maps(
        self,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        map_height: int = 200,
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        r"""..
        Evaluate over episodes, starting from the last episode evaluated. Return evaluation
        metrics and top-down maps from the episodes.

        :param episode_id_last: ID of the last episode evaluated; -1 for evaluating
            from start
        :param scene_id_last: Scene ID of the last episode evaluated
        :param log_dir: logging directory
        :param map_height: height of top-down maps
        :return: a dictionary where each key is an episode's unique identifier as
            <episode-id>,<scene-id>; each value is the set of metrics (including top-down maps)
            from the episode.
        """
        raise NotImplementedError
