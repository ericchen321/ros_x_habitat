from src.classes.evaluator import Evaluator
from typing import Dict
from habitat.config.default import get_config
from src.classes.habitat_eval_rlenv import HabitatEvalRLEnv
from habitat.core.agent import Agent
from collections import defaultdict

# use TensorBoard to visualize
from src.classes.utils_tensorboard import TensorboardWriter, generate_video
from habitat.utils.visualizations.utils import observations_to_image
import numpy as np
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.config import Config
from habitat_baselines.agents.ppo_agents import PPOAgent

# logging
import os
from classes import utils_logging
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

class HabitatEvaluator(Evaluator):
    r"""Class to evaluate a Habitat agent in a Habitat simulator instance
    without ROS as middleware.
    """

    def __init__(
        self,
        config_paths: str,
        agent_input_type: str,
        agent_model_path: str,
        enable_physics: bool = False,
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param agent: Habitat agent object
        :param enable_physics: use dynamic simulation or not

        """
        config_env = get_config(config_paths)
        # embed top-down map and heading sensor in config
        config_env.defrost()
        config_env.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        # config_env.TASK.SENSORS.append("HEADING_SENSOR")
        config_env.freeze()

        # store agent params and declare agent instance
        self.agent_input_type = agent_input_type
        self.agent_model_path = agent_model_path
        self.agent = None

        # define Habitat simulator instance
        self.enable_physics = enable_physics
        self.env = HabitatEvalRLEnv(
            config=config_env, enable_physics=self.enable_physics
        )

    def evaluate(
        self,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        make_videos: bool = False,
        video_dir: str = "videos/",
        tb_dir: str = "tb/",
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        r"""..
        Evaluate over episodes, starting from the last episode evaluated. Return evaluation
        metrics. ROS is not involved.

        :param episode_id_last: ID of the last episode evaluated; -1 for evaluating
            from start
        :param scene_id_last: Scene ID of the last episode evaluated
        :param log_dir: logging directory
        :param make_videos: toggle video production on/off
        :param video_dir: directory to store videos
        :param tb_dir: Tensorboard logging directory
        :return: dict containing metrics tracked by environment.
        """
        num_episodes = len(self.env._env.episodes)
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
                    self.env._env.reset()
                    e = self.env._env.current_episode
                    if (str(e.episode_id) == str(episode_id_last)) and (
                        e.scene_id == scene_id_last
                    ):
                        logger.info(
                            f"Last episode found: episode-id={episode_id_last}, scene-id={scene_id_last}"
                        )
                        last_ep_found = True
                except StopIteration:
                    logger.info("Last episode not found!")
                    raise StopIteration
        else:
            logger.info(
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
                observations_per_episode = []

                # ------------ log agent time start ------------
                t_agent_start = time.clock()
                # ----------------------------------------------

                # instantiate an agent
                agent_config = get_default_config()
                agent_config.INPUT_TYPE = self.agent_input_type
                agent_config.MODEL_PATH = self.agent_model_path
                self.agent = PPOAgent(agent_config)
                self.agent.reset()

                # ------------ log agent time end ------------
                t_agent_end = time.clock()
                t_agent_elapsed += t_agent_end - t_agent_start
                # --------------------------------------------

                # ------------ log sim time start ------------
                t_sim_start = time.clock()
                # --------------------------------------------

                observations_per_action = self.env._env.reset()

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
                    f"{log_dir}/{episode_id}-{os.path.basename(scene_id)}.log",
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
                    info_per_action = None

                    # ------------ log sim time start ------------
                    t_sim_start = time.clock()
                    # --------------------------------------------

                    (observations_per_action, _, _, info_per_action) = self.env.step(
                        action
                    )
                    count_steps += 1

                    # ------------  log sim time end  ------------
                    t_sim_end = time.clock()
                    t_sim_elapsed += t_sim_end - t_sim_start
                    # --------------------------------------------

                    # generate an output image for the action. The image includes observations
                    # and a top-down map showing the agent's state in the environment
                    out_im_per_action = observations_to_image(
                        observations_per_action, info_per_action
                    )
                    observations_per_episode.append(out_im_per_action)

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
                # generate video
                if make_videos:
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

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        utils_logging.close_logger(logger)

        return avg_metrics
