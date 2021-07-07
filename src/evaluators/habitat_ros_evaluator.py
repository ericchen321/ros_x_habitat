from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
from typing import Dict
from habitat.config.default import get_config
from habitat.core.agent import Agent
from collections import defaultdict
from ros_x_habitat.srv import *
from subprocess import Popen
import shlex
import rospy

# use TensorBoard to visualize
from src.utils.utils_visualization import TensorboardWriter, generate_video
from habitat.utils.visualizations.utils import observations_to_image
import numpy as np
from habitat.tasks.nav.nav import NavigationEpisode

# logging
import os
from src.utils import utils_logging
from traceback import print_exc

# sim timing
import time

class HabitatROSEvaluator(HabitatSimEvaluator):
    r"""Class to evaluate Habitat agents in Habitat environments with ROS
    as middleware.
    """

    def __init__(
        self,
        config_paths: str,
        input_type: str,
        model_path: str,
        sensor_pub_rate: float,
        do_not_start_nodes: bool = False,
        enable_physics: bool = False,
    ) -> None:
        r"""..

        :param input_type: agent's input type, options: "rgb", "rgbd",
            "depth", "blind"
        :param model_path: path to agent's model
        :param config_paths: file to be used for creating the environment
        :param sensor_pub_rate: rate at which the env node publishes sensor
            readings
        :param do_not_start_nodes: if True then the evaluator would not start
            the env node and the agent node.
        :param enable_physics: use dynamic simulation or not
        """

        # check if agent input type is valid
        assert input_type in ["rgb", "rgbd", "depth", "blind"]

        if enable_physics:
            # TODO: pass extra arguments to define agent and sim with dynamics
            raise NotImplementedError
        else:
            if do_not_start_nodes is False:
                # start the agent node
                agent_node_args = shlex.split(f"python src/nodes/habitat_agent_node.py --input-type {input_type} --model-path {model_path} --sensor-pub-rate {sensor_pub_rate}")
                Popen(agent_node_args)

                # start the env node
                env_node_args = shlex.split(f"python src/nodes/habitat_env_node.py --task-config {config_paths} --sensor-pub-rate {sensor_pub_rate}")
                Popen(env_node_args)

        # start the evaluator node
        rospy.init_node("evaluator_habitat_ros")
    
    def evaluate(
        self,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        agent_seed: int = 7,
        *args,
        **kwargs
    ) -> Dict[str, float]:
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
                        f"{log_dir}/episode={episode_id}-scene={os.path.basename(scene_id)}.log",
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
                    # shut down the episode logger
                    utils_logging.close_logger(logger_per_episode)
            except rospy.ServiceException:
                logger.info(f"Evaluation call failed at {count_episodes}-th episode")
                break
        
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        utils_logging.close_logger(logger)

        return avg_metrics

    def generate_video(
        self,
        episode_id: str,
        scene_id: str,
        agent_seed: int = 7,
        *args,
        **kwargs
    ) -> None:
        #TODO: we may need to implement it for Habitat agent + Gazebo or ROS planner + Habitat Sim
        raise NotImplementedError

    def generate_map(
        self,
        episode_id: str,
        scene_id: str,
        agent_seed: int,
        map_height: int,
        *args,
        **kwargs
    ) -> np.ndarray:
        #TODO: we may need to implement it for Habitat agent + Gazebo or ROS planner + Habitat Sim
        raise NotImplementedError