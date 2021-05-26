from classes.habitat_evaluator import HabitatEvaluator
from typing import Dict, Optional
from habitat.config.default import get_config
from classes.habitat_eval_rlenv import HabitatEvalRLEnv
from habitat.core.agent import Agent
from collections import defaultdict

# use TensorBoard to visualize
from classes.utils_tensorboard import TensorboardWriter, generate_video
from habitat.utils.visualizations.utils import observations_to_image
import numpy as np
from habitat.tasks.nav.nav import NavigationEpisode

# logging
import os
from classes import utils_logging


class HabitatDiscreteEvaluator(HabitatEvaluator):
    r"""Class to evaluate Habitat agents producing discrete actions in environments
    without dynamics.
    """

    def __init__(self, config_paths: Optional[str] = None) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        """
        config_env = get_config(config_paths)
        # embed top-down map and heading sensor in config
        config_env.defrost()
        config_env.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        # config_env.TASK.SENSORS.append("HEADING_SENSOR")
        config_env.freeze()

        self._env = HabitatEvalRLEnv(config=config_env, enable_physics=False)

    def evaluate(
        self,
        agent: Agent,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        video_dir: str = "videos/",
        tb_dir: str = "tb/",
    ) -> Dict[str, float]:
        r"""..

        Args:
            agent: agent to be evaluated in environment.
            num_episodes: count of number of episodes for which the
            evaluation should be run.

        Return:
            dict containing metrics tracked by environment.
        """
        num_episodes = len(self._env._env.episodes)
        assert num_episodes > 0, "environment should contain at least one episode"
        logger = utils_logging.setup_logger(__name__)
        logger.info(f"Total number of episodes in the environment: {num_episodes}")

        agg_metrics: Dict = defaultdict(float)

        writer = TensorboardWriter(
            tb_dir, flush_secs=30
        )  # flush_specs from base_trainer.py

        # locate the last episode evaluated
        if episode_id_last != "-1":
            # iterate to the last episode. If not found, the loop exits upon a
            # StopIteration exception
            last_ep_found = False
            while not last_ep_found:
                try:
                    self._env._env.reset()
                    e = self._env._env.current_episode
                    if (e.episode_id == episode_id_last) and (
                        e.scene_id == scene_id_last
                    ):
                        logger.info(
                            f"Last episode found: episode-id={episode_id_last}, scene-id={scene_id_last}"
                        )
                        last_ep_found = True
                except StopIteration:
                    logger.info("Last episode not found!")
                    raise StopIteration
            assert self._env._env.current_episode is not None
            assert (
                self._env._env.current_episode.episode_id == episode_id_last
                and self._env._env.current_episode.scene_id == scene_id_last
            )
        else:
            logger.info(
                f"No last episode specified. Proceed to evaluate from beginning"
            )

        # then evaluate the rest of the episodes from the environment
        count_episodes = 0
        while count_episodes < num_episodes:
            # initialize a new episode
            observations_per_episode = []
            agent.reset()
            observations_per_action = self._env._env.reset()
            current_episode = self._env._env.current_episode

            # get episode and scene id
            episode_id = int(current_episode.episode_id)
            scene_id = current_episode.scene_id
            logger = utils_logging.setup_logger(
                f"{__name__}-{episode_id}-{scene_id}",
                f"{log_dir}/{episode_id}-{os.path.basename(scene_id)}.log",
            )
            logger.info(f"episode id: {episode_id}")
            logger.info(f"scene id: {scene_id}")

            # act until one episode is over
            while not self._env._env.episode_over:
                action = agent.act(observations_per_action)
                observations_per_action = None
                info_per_action = None
                (observations_per_action, _, _, info_per_action) = self._env.step(
                    action
                )
                # generate an output image for the action. The image includes observations
                # and a top-down map showing the agent's state in the environment
                out_im_per_action = observations_to_image(
                    observations_per_action, info_per_action
                )
                observations_per_episode.append(out_im_per_action)

            # episode ended
            # get per-episode metrics. for now we only extract
            # distance-to-goal, success, spl
            metrics = self._env._env.get_metrics()
            per_ep_metrics = {
                k: metrics[k] for k in ["distance_to_goal", "success", "spl"]
            }
            # print distance_to_goal, success and spl
            for k, v in per_ep_metrics.items():
                logger.info(f"{k},{v}")
            # calculate aggregated distance_to_goal, success and spl
            for m, v in per_ep_metrics.items():
                agg_metrics[m] += v
            count_episodes += 1
            # generate video
            generate_video(
                video_option=["disk", "tensorboard"],
                video_dir=video_dir,
                images=observations_per_episode,
                episode_id=episode_id,
                scene_id=scene_id,
                checkpoint_idx=0,
                metrics=per_ep_metrics,
                tb_writer=writer,
            )

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics
