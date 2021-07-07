from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
from typing import Dict, List
from src.envs.habitat_eval_rlenv import HabitatEvalRLEnv
from habitat.core.agent import Agent
from collections import defaultdict

# use TensorBoard to visualize
from src.utils.utils_visualization import TensorboardWriter, generate_video
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import observations_to_image
import numpy as np
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.config import Config
from habitat_baselines.agents.ppo_agents import PPOAgent

# logging
import os
from src.utils import utils_logging
from traceback import print_exc

# sim timing
import time


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

    def evaluate(
        self,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        agent_seed: int = 7,
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        # make sure we have episodes to evaluate
        num_episodes = len(self.env._env.episodes)
        assert num_episodes > 0, "environment should contain at least one episode"
        self.logger.info(f"Total number of episodes in the environment: {num_episodes}")

        agg_metrics: Dict = defaultdict(float)

        # reset episode iterator
        self.env.reset_episode_iterator()

        # locate the last episode evaluated
        if episode_id_last != "-1":
            self.env.iter_to_episode(episode_id_last, scene_id_last, self.logger)
        else:
            self.logger.info(
                f"No last episode specified. Proceed to evaluate from the next one"
            )

        # then evaluate the rest of the episodes from the environment
        count_episodes = 0
        episode_id = ""
        scene_id = ""
        while count_episodes < num_episodes:
            try:
                count_steps = 0
                t_sim_elapsed = 0.0
                t_agent_elapsed = 0.0

                # initialize a new episode

                # ------------ log agent time start ------------
                t_agent_start = time.clock()
                # ----------------------------------------------

                # instantiate an agent
                agent_config = get_default_config()
                agent_config.INPUT_TYPE = self.input_type
                agent_config.MODEL_PATH = self.model_path
                agent_config.RANDOM_SEED = agent_seed
                self.agent = PPOAgent(agent_config)
                self.agent.reset()

                # ------------ log agent time end ------------
                t_agent_end = time.clock()
                t_agent_elapsed += t_agent_end - t_agent_start
                # --------------------------------------------

                # ------------ log sim time start ------------
                t_sim_start = time.clock()
                # --------------------------------------------

                observations_per_action = self.env.reset()

                # ------------  log sim time end  ------------
                t_sim_end = time.clock()
                t_sim_elapsed += t_sim_end - t_sim_start
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

                    (observations_per_action, _, _, _) = self.env.step(
                        action
                    )
                    count_steps += 1

                    # ------------  log sim time end  ------------
                    t_sim_end = time.clock()
                    t_sim_elapsed += t_sim_end - t_sim_start
                    # --------------------------------------------

                # episode ended
                # get per-episode metrics. for now we extract distance-to-goal, success, spl
                # from the environment, and we add sim_time and num_steps as two other metrics
                metrics = self.env._env.get_metrics()
                per_ep_metrics = {
                    k: metrics[k] for k in ["distance_to_goal", "success", "spl"]
                }
                per_ep_metrics["agent_time"] = t_agent_elapsed / count_steps
                per_ep_metrics["sim_time"] = t_sim_elapsed / count_steps
                per_ep_metrics["num_steps"] = count_steps
                # print metrics of this episode
                for k, v in per_ep_metrics.items():
                    logger_per_episode.info(f"{k},{v}")
                # calculate aggregated metrics over episodes eval'ed so far
                for m, v in per_ep_metrics.items():
                    agg_metrics[m] += v
                count_episodes += 1

                # shut down the episode logger
                utils_logging.close_logger(logger_per_episode)

            except StopIteration:
                self.logger.info(f"Finished evaluation after: {count_episodes} episodes")
                self.logger.info(
                    f"Last episode evaluated: episode={episode_id}, scene={scene_id}"
                )
                break
            except OSError:
                self.logger.info(
                    f"Evaulation stopped after: {count_episodes} episodes due to OSError!"
                )
                self.logger.info(f"Current episode: episode={episode_id}, scene={scene_id}")
                print_exc()
                break

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        utils_logging.close_logger(self.logger)

        return avg_metrics

    def generate_video(
        self,
        episode_id: str,
        scene_id: str,
        agent_seed: int = 7,
        *args,
        **kwargs
    ) -> None:
        # reset episode iterator
        self.env.reset_episode_iterator()

        # iterate to the given episode
        observations_per_action = None
        info_per_action = None
        observations_per_action = self.env.iter_to_episode(episode_id, scene_id, self.logger)

        # instantiate an agent
        agent_config = get_default_config()
        agent_config.INPUT_TYPE = self.input_type
        agent_config.MODEL_PATH = self.model_path
        agent_config.RANDOM_SEED = agent_seed
        self.agent = PPOAgent(agent_config)
        self.agent.reset()

        # store observations over frames
        observations_per_episode = []
        
        # act until the episode is over
        while not self.env._env.episode_over:
            action = self.agent.act(observations_per_action)
            (observations_per_action, _, _, info_per_action) = self.env.step(
                action
            )
            # generate an output image for the action. The image includes observations
            # and a top-down map showing the agent's state in the environment
            out_im_per_action = observations_to_image(
                observations_per_action, info_per_action
            )
            observations_per_episode.append(out_im_per_action)
        
        # get metrics for video generation
        metrics = self.env._env.get_metrics()
        per_ep_metrics = {
            k: metrics[k] for k in ["distance_to_goal", "success", "spl"]
        }

        # set up Tensorboard writer
        writer = None
        if "tensorboard" in self.config.VIDEO_OPTION:
            writer = TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=30
            )  # flush_specs from base_trainer.py

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

    def generate_map(
        self,
        episode_id: str,
        scene_id: str,
        agent_seed: int,
        map_height: int,
        *args,
        **kwargs
    ) -> np.ndarray:
        # reset episode iterator
        self.env.reset_episode_iterator()

        # iterate to the given episode
        observations_per_action = None
        info_per_action = None
        observations_per_action = self.env.iter_to_episode(episode_id, scene_id, self.logger)

        # instantiate an agent
        agent_config = get_default_config()
        agent_config.INPUT_TYPE = self.input_type
        agent_config.MODEL_PATH = self.model_path
        agent_config.RANDOM_SEED = agent_seed
        self.agent = PPOAgent(agent_config)
        self.agent.reset()

        # act until the episode is over
        while not self.env._env.episode_over:
            action = self.agent.act(observations_per_action)
            (observations_per_action, _, _, info_per_action) = self.env.step(
                action
            )

        # draw the map
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info_per_action["top_down_map"], map_height,
        )

        return top_down_map