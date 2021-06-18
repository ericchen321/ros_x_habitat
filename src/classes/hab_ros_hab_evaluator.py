from src.classes.evaluator import Evaluator
from typing import Dict
from habitat.config.default import get_config
from habitat.core.agent import Agent
from collections import defaultdict
from ros_x_habitat.srv import *
from subprocess import Popen
import shlex
import rospy

# use TensorBoard to visualize
from src.classes.utils_tensorboard import TensorboardWriter, generate_video
from habitat.utils.visualizations.utils import observations_to_image
import numpy as np
from habitat.tasks.nav.nav import NavigationEpisode

# logging
import os
from classes import utils_logging
from traceback import print_exc

# sim timing
import time

class HabROSHabEvaluator(Evaluator):
    r"""Class to evaluate Habitat agents in Habitat environments with ROS
    as middleware.
    """

    def __init__(
        self,
        input_type: str,
        model_path: str,
        config_paths: str,
        sensor_pub_rate: float,
        enable_physics: bool = False,
    ) -> None:
        r"""..

        :param input_type: agent's input type, options: "rgb", "rgbd",
            "depth", "blind"
        :param model_path: path to agent's model
        :param config_paths: file to be used for creating the environment
        :param sensor_pub_rate: rate at which the env node publishes sensor
            readings
        :param enable_physics: use dynamic simulation or not
        """

        # check if agent input type is valid
        assert input_type in ["rgb", "rgbd", "depth", "blind"]

        if enable_physics:
            # TODO: pass extra arguments to define agent and sim with dynamics
            raise NotImplementedError
        else:
            # start the agent node
            agent_node_args = shlex.split(f"python classes/habitat_agent_node.py --input-type {input_type} --model-path {model_path} --sensor-pub-rate {sensor_pub_rate}")
            Popen(agent_node_args)

            # start the env node
            env_node_args = shlex.split(f"python classes/habitat_env_node.py --task-config configs/pointnav_rgbd_val.yaml --sensor-pub-rate {sensor_pub_rate}")
            Popen(env_node_args)

        # start the evaluator node
        rospy.init_node("evaluator_hab_ros_hab")
    
    def evaluate(
        self,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        make_videos: bool = False,
        video_dir: str = "videos/",
        tb_dir: str = "tb/",
        *args,
        **kwargs
    ) -> Dict[str, float]:
        r"""..
        Evaluate over episodes, starting from the last episode evaluated. Return evaluation
        metrics.

        :param episode_id_last: ID of the last episode evaluated; -1 for evaluating
            from start
        :param scene_id_last: Scene ID of the last episode evaluated
        :param log_dir: logging directory
        :param make_videos: toggle video production on/off
        :param video_dir: directory to store videos
        :param tb_dir: Tensorboard logging directory
        :return: dict containing metrics tracked by environment.
        """
        logger = utils_logging.setup_logger(__name__)

        count_episodes = 0
        agg_metrics: Dict = defaultdict(float)
        eval_episode = rospy.ServiceProxy('eval_episode', EvalEpisode)

        # evaluate episodes, starting from the one after the last episode
        # evaluated
        while not rospy.is_shutdown():
            rospy.wait_for_service("eval_episode")
            try:
                # request env node to evaluate an episode
                resp = None
                if count_episodes == 0:
                    # jump to the first episode we want to evaluate
                    resp = eval_episode(episode_id_last, scene_id_last)
                else:
                    # evaluate the next episode
                    resp = eval_episode("-1", "")
                
                if resp.episode_id == "-1":
                    # no more episodes
                    logger.info(f"Finished evaluation after: {count_episodes} episodes")
                    break
                else:
                    # get per-episode metrics
                    per_ep_metrics = {
                        "distance_to_goal": resp.distance_to_goal,
                        "success": resp.success,
                        "spl": resp.spl
                    }

                    # set up logger
                    episode_id = resp.episode_id
                    scene_id = resp.scene_id
                    logger_per_episode = utils_logging.setup_logger(
                        f"{__name__}-{episode_id}-{scene_id}",
                        f"{log_dir}/{episode_id}-{os.path.basename(scene_id)}.log",
                    )

                    # log episode ID and scene ID
                    logger_per_episode.info(f"episode id: {episode_id}")
                    logger_per_episode.info(f"scene id: {scene_id}")

                    # print metrics of this episode
                    for k, v in per_ep_metrics.items():
                        logger_per_episode.info(f"{k},{v}")
                    
                    # calculate aggregated metrics over episodes eval'ed so far
                    for m, v in per_ep_metrics.items():
                        agg_metrics[m] += v
                    
                    count_episodes += 1
            except rospy.ServiceException:
                logger.info(f"Evaluation call failed at {count_episodes}-th episode")
                break
        
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics