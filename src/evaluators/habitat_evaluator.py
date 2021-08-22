import os
import time
from traceback import print_exc
from typing import List, Tuple, Dict
import numpy as np
from habitat.config import Config
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.agents.ppo_agents import PPOAgent

from src.envs.habitat_eval_rlenv import HabitatEvalRLEnv
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
from src.constants.constants import NumericalMetrics
from src.utils import utils_logging
from src.utils.utils_visualization import (
    TensorboardWriter,
    generate_video,
    colorize_and_fit_to_height,
)


def get_default_config():
    c = Config()
    c.INPUT_TYPE = "blind"
    c.MODEL_PATH = "data/checkpoints/blind.pth"
    c.RESOLUTION = 256
    c.HIDDEN_SIZE = 512
    c.RANDOM_SEED = 7
    c.PTH_GPU_ID = 0
    c.GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
    return c


class HabitatEvaluator(HabitatSimEvaluator):
    r"""Class to evaluate a Habitat agent in a Habitat simulator instance
    without ROS as middleware.
    """

    def __init__(
        self,
        config_paths: str,
        input_type: str,
        model_path: str,
        enable_physics: bool = False,
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param agent: Habitat agent object
        :param enable_physics: use dynamic simulation or not

        """
        super().__init__(config_paths, input_type, model_path, enable_physics)

        # embed top-down map and heading sensor in config
        self.config.defrost()
        self.config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        self.config.freeze()

        # declare an agent instance
        self.agent = None

        # overwrite env config if physics enabled
        if self.enable_physics:
            self.overwrite_simulator_config(self.config)

        # define Habitat simulator instance
        self.env = HabitatEvalRLEnv(
            config=self.config, enable_physics=self.enable_physics
        )

    def evaluate_and_get_maps(
        self,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        agent_seed: int = 7,
        map_height: int = 200,
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        # make sure we have episodes to evaluate
        num_episodes = len(self.env._env.episodes)
        assert num_episodes > 0, "environment should contain at least one episode"

        # create a logger
        logger = utils_logging.setup_logger(__name__)
        logger.info(f"Total number of episodes in the environment: {num_episodes}")

        # reset episode iterator
        self.env.reset_episode_iterator()

        # locate the last episode evaluated
        if episode_id_last != "-1":
            self.env.iter_to_episode(episode_id_last, scene_id_last, logger)
        else:
            logger.info(
                f"No last episode specified. Proceed to evaluate from the next one"
            )

        # then evaluate the rest of the episodes from the environment
        count_episodes = 0
        episode_id = ""
        scene_id = ""
        dict_of_metrics = {}
        while count_episodes < num_episodes:
            try:
                count_steps = 0
                t_reset_elapsed = 0.0
                t_sim_elapsed = 0.0
                t_agent_elapsed = 0.0

                # instantiate an agent
                agent_config = get_default_config()
                agent_config.INPUT_TYPE = self.input_type
                agent_config.MODEL_PATH = self.model_path
                agent_config.RANDOM_SEED = agent_seed
                self.agent = PPOAgent(agent_config)
                self.agent.reset()

                # ------------ log reset time start ------------
                t_reset_start = time.clock()
                # --------------------------------------------

                observations_per_action = self.env.reset()

                # ------------  log reset time end  ------------
                t_reset_end = time.clock()
                t_reset_elapsed += t_reset_end - t_reset_start
                # --------------------------------------------

                current_episode = self.env._env.current_episode

                # get episode and scene id
                episode_id = str(current_episode.episode_id)
                scene_id = current_episode.scene_id
                logger_per_episode = utils_logging.setup_logger(
                    f"{__name__}-{episode_id}-{scene_id}",
                    f"{log_dir}/episode={episode_id}-scene={os.path.basename(scene_id)}.log",
                )
                logger_per_episode.info(f"episode id: {episode_id}")
                logger_per_episode.info(f"scene id: {scene_id}")

                # act until one episode is over
                info_per_action = None
                while not self.env._env.episode_over:

                    # ------------ log agent time start ------------
                    t_agent_start = time.clock()
                    # ----------------------------------------------

                    action = self.agent.act(observations_per_action)

                    # ------------ log agent time end ------------
                    t_agent_end = time.clock()
                    t_agent_elapsed += t_agent_end - t_agent_start
                    # --------------------------------------------

                    observations_per_action = None

                    # ------------ log sim time start ------------
                    t_sim_start = time.clock()
                    # --------------------------------------------

                    (observations_per_action, _, _, info_per_action) = self.env.step(
                        action
                    )

                    # ------------  log sim time end  ------------
                    t_sim_end = time.clock()
                    t_sim_elapsed += t_sim_end - t_sim_start
                    # --------------------------------------------

                    count_steps += 1

                # episode ended
                # collect metrics
                per_episode_metrics = self.env._env.get_metrics()
                per_episode_metrics[NumericalMetrics.NUM_STEPS] = count_steps
                per_episode_metrics[NumericalMetrics.SIM_TIME] = (
                    t_sim_elapsed / count_steps
                )
                per_episode_metrics[NumericalMetrics.RESET_TIME] = t_reset_elapsed
                per_episode_metrics[NumericalMetrics.AGENT_TIME] = (
                    t_agent_elapsed / count_steps
                )
                # colorize the map and replace "top_down_map" metric with it
                per_episode_metrics[
                    "top_down_map"
                ] = maps.colorize_draw_agent_and_fit_to_height(
                    info_per_action["top_down_map"],
                    map_height,
                )

                # print numerical metrics of this episode
                for metric_name in NumericalMetrics:
                    logger_per_episode.info(
                        f"{metric_name},{per_episode_metrics[metric_name]}"
                    )

                # add to the metrics list
                dict_of_metrics[f"{episode_id},{scene_id}"] = per_episode_metrics

                # increment episode counter
                count_episodes += 1

                # shut down the episode logger
                utils_logging.close_logger(logger_per_episode)

            except StopIteration:
                logger.info(f"Finished evaluation after: {count_episodes} episodes")
                logger.info(
                    f"Last episode evaluated: episode={episode_id}, scene={scene_id}"
                )
                break
            except OSError:
                logger.info(
                    f"Evaulation stopped after: {count_episodes} episodes due to OSError!"
                )
                logger.info(f"Current episode: episode={episode_id}, scene={scene_id}")
                print_exc()
                break

        # destroy the logger
        utils_logging.close_logger(logger)

        return dict_of_metrics

    def evaluate(
        self,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        agent_seed: int = 7,
        *args,
        **kwargs,
    ):
        dict_of_metrics = self.evaluate_and_get_maps(
            episode_id_last, scene_id_last, log_dir, agent_seed, 200
        )
        return dict_of_metrics

    def generate_videos(
        self,
        episode_ids: List[str],
        scene_ids: List[str],
        agent_seed: int = 7,
        *args,
        **kwargs,
    ) -> None:
        # create a logger
        logger = utils_logging.setup_logger(__name__)

        # precondition checks
        assert len(episode_ids) == len(scene_ids)

        num_episodes = len(episode_ids)
        count_episodes_visualized = 0

        # create agent config
        agent_config = get_default_config()
        agent_config.INPUT_TYPE = self.input_type
        agent_config.MODEL_PATH = self.model_path
        agent_config.RANDOM_SEED = agent_seed

        # reset episode iterator
        self.env.reset_episode_iterator()

        # set up Tensorboard writer
        writer = None
        if "tensorboard" in self.config.VIDEO_OPTION:
            writer = TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=30
            )  # flush_specs from base_trainer.py

        # visualize episodes in the given lists
        while count_episodes_visualized < num_episodes:
            try:
                observations_per_action = self.env.reset()
                info_per_action = None

                # get episode and scene id
                current_episode = self.env._env.current_episode
                episode_id = str(current_episode.episode_id)
                scene_id = current_episode.scene_id
                if (episode_id, scene_id) in zip(episode_ids, scene_ids):
                    # if the current episode is in the visualization list,
                    # we evaluate it
                    count_episodes_visualized += 1

                    # store observations over frames
                    # NOTE: we are not storing the initial observations returned
                    # from env.reset() because we cannot get the initial info for
                    # observations_to_image()
                    observations_per_episode = []

                    # instantiate an agent
                    self.agent = PPOAgent(agent_config)
                    self.agent.reset()

                    # act until the episode is over
                    while not self.env._env.episode_over:
                        action = self.agent.act(observations_per_action)
                        (
                            observations_per_action,
                            _,
                            _,
                            info_per_action,
                        ) = self.env.step(action)
                        out_im_per_action = observations_to_image(
                            observations_per_action, info_per_action
                        )
                        observations_per_episode.append(out_im_per_action)

                    # get metrics for video generation
                    metrics = self.env._env.get_metrics()
                    per_ep_metrics = {
                        k: metrics[k]
                        for k in [
                            NumericalMetrics.DISTANCE_TO_GOAL,
                            NumericalMetrics.SUCCESS,
                            NumericalMetrics.SPL,
                        ]
                    }

                    # generate video and tensorboard visualization
                    generate_video(
                        video_option=self.config.VIDEO_OPTION,
                        video_dir=self.config.VIDEO_DIR,
                        images=observations_per_episode,
                        episode_id=episode_id,
                        scene_id=scene_id,
                        agent_seed=agent_seed,
                        checkpoint_idx=0,
                        metrics=per_ep_metrics,
                        tb_writer=writer,
                    )
            except StopIteration:
                break

        logger.info(f"Given {num_episodes} episodes to generate videos")
        logger.info(f"Generated videos of {count_episodes_visualized} episodes")

        # destroy the logger
        utils_logging.close_logger(logger)

    def generate_maps(
        self,
        episode_ids: List[str],
        scene_ids: List[str],
        agent_seed: int,
        map_height: int,
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        # create a logger
        logger = utils_logging.setup_logger(__name__)

        # precondition checks
        assert len(episode_ids) == len(scene_ids)

        num_episodes = len(episode_ids)
        count_episodes_visualized = 0

        # create agent config
        agent_config = get_default_config()
        agent_config.INPUT_TYPE = self.input_type
        agent_config.MODEL_PATH = self.model_path
        agent_config.RANDOM_SEED = agent_seed

        # reset episode iterator
        self.env.reset_episode_iterator()

        # visualize episodes in the given lists
        dict_of_maps: Dict[str, np.ndarray] = {}
        while count_episodes_visualized < num_episodes:
            try:
                observations_per_action = self.env.reset()
                info_per_action = None

                # get episode and scene id
                current_episode = self.env._env.current_episode
                episode_id = str(current_episode.episode_id)
                scene_id = current_episode.scene_id
                if (episode_id, scene_id) in zip(episode_ids, scene_ids):
                    # if the current episode is in the visualization list,
                    # we evaluate it
                    count_episodes_visualized += 1

                    # instantiate an agent
                    self.agent = PPOAgent(agent_config)
                    self.agent.reset()

                    # act until the episode is over
                    while not self.env._env.episode_over:
                        action = self.agent.act(observations_per_action)
                        (
                            observations_per_action,
                            _,
                            _,
                            info_per_action,
                        ) = self.env.step(action)

                    # draw and append the map
                    top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                        info_per_action["top_down_map"],
                        map_height,
                    )
                    dict_of_maps[f"{episode_id},{scene_id}"] = top_down_map
            except StopIteration:
                break

        logger.info(f"Given {num_episodes} episodes to generate maps")
        logger.info(f"Generated maps of {count_episodes_visualized} episodes")

        # destroy the logger
        utils_logging.close_logger(logger)

        return dict_of_maps

    def get_blank_maps(
        self,
        episode_ids: List[str],
        scene_ids: List[str],
        map_height: int,
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        # create a logger
        logger = utils_logging.setup_logger(__name__)

        # precondition checks
        assert len(episode_ids) == len(scene_ids)

        num_episodes = len(episode_ids)
        count_episodes_visualized = 0

        # reset episode iterator
        self.env.reset_episode_iterator()

        # visualize episodes in the given lists
        dict_of_maps: Dict[str, np.ndarray] = {}
        while count_episodes_visualized < num_episodes:
            try:
                self.env.reset()

                # get episode and scene id
                current_episode = self.env._env.current_episode
                episode_id = str(current_episode.episode_id)
                scene_id = current_episode.scene_id
                if (episode_id, scene_id) in zip(episode_ids, scene_ids):
                    # if the current episode is in the visualization list,
                    # we get its top-down map
                    count_episodes_visualized += 1

                    # draw and append the map
                    top_down_map_raw = maps.get_topdown_map_from_sim(
                        sim=self.env._env._sim,
                        map_resolution=self.env._env._config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION,
                        draw_border=self.env._env._config.TASK.TOP_DOWN_MAP.DRAW_BORDER,
                    )
                    top_down_map = colorize_and_fit_to_height(
                        top_down_map_raw, map_height
                    )
                    dict_of_maps[f"{episode_id},{scene_id}"] = top_down_map
            except StopIteration:
                break

        logger.info(f"Given {num_episodes} episodes to generate blank top-down maps")
        logger.info(
            f"Generated blank top-down maps of {count_episodes_visualized} episodes"
        )

        # destroy the logger
        utils_logging.close_logger(logger)

        return dict_of_maps
