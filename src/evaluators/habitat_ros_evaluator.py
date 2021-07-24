# logging
import os
import shlex
from subprocess import Popen
from typing import List, Tuple, Dict
import numpy as np
import rospy
from ros_x_habitat.srv import EvalEpisode, ResetAgent
from src.constants.constants import NumericalMetrics
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
from src.constants.constants import AgentResetCommands
from src.utils import utils_logging


class HabitatROSEvaluator(HabitatSimEvaluator):
    r"""Class to evaluate Habitat agents in Habitat environments with ROS
    as middleware.
    """

    def __init__(
        self,
        env_node_name: str,
        agent_node_name: str,
        config_paths: str,
        input_type: str,
        model_path: str,
        sensor_pub_rate: float,
        do_not_start_nodes: bool = False,
        enable_physics: bool = False,
        use_continuous_agent: bool = False,
    ) -> None:
        r"""..

        :param env_node_name: name of the env node
        :param agent_node_name: name of the agent node
        :param config_paths: file to be used for creating the environment
        :param input_type: agent's input type, options: "rgb", "rgbd",
            "depth", "blind"
        :param model_path: path to agent's model
        :param sensor_pub_rate: rate at which the env node publishes sensor
            readings
        :param do_not_start_nodes: if True then the evaluator would not start
            the env node and the agent node.
        :param enable_physics: use dynamic simulation or not
        :param use_continuous_agent: if using a continuous agent (outputs velocity)
            or a discrete agent (outputs action). This must be false when physics
            has been disabled
        """

        # check if agent input type is valid
        assert input_type in ["rgb", "rgbd", "depth", "blind"]

        # parse args for agent node
        agent_node_args = shlex.split(
            f"python src/nodes/habitat_agent_node.py --node-name {agent_node_name} --input-type {input_type} --model-path {model_path} --sensor-pub-rate {sensor_pub_rate}"
        )

        # parse args for env node
        if enable_physics and use_continuous_agent:
            # physic sim + continuous agent
            env_node_args = shlex.split(
                f"python src/nodes/habitat_env_node.py --node-name {env_node_name} --task-config {config_paths} --enable-physics --use-continuous-agent --sensor-pub-rate {sensor_pub_rate}"
            )
        elif enable_physics and (not use_continuous_agent):
            # physics sim + discrete agent
            env_node_args = shlex.split(
                f"python src/nodes/habitat_env_node.py --node-name {env_node_name} --task-config {config_paths} --enable-physics --sensor-pub-rate {sensor_pub_rate}"
            )
        else:
            # discrete sim + discrete agent
            assert use_continuous_agent is False
            env_node_args = shlex.split(
                f"python src/nodes/habitat_env_node.py --node-name {env_node_name} --task-config {config_paths} --sensor-pub-rate {sensor_pub_rate}"
            )

        self.do_not_start_nodes = do_not_start_nodes
        if do_not_start_nodes is False:
            self.agent_process = Popen(agent_node_args)
            self.env_process = Popen(env_node_args)

        # start the evaluator node
        rospy.init_node("evaluator_habitat_ros")

        # set up agent reset service client
        self.reset_agent = rospy.ServiceProxy("reset_agent", ResetAgent)

    def evaluate(
        self,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        agent_seed: int = 7,
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        logger = utils_logging.setup_logger(__name__)

        count_episodes = 0
        dict_of_metrics = {}
        eval_episode = rospy.ServiceProxy("eval_episode", EvalEpisode)

        # evaluate episodes, starting from the one after the last episode
        # evaluated
        while not rospy.is_shutdown():
            # reset agent
            rospy.wait_for_service("reset_agent")
            try:
                resp = self.reset_agent(int(AgentResetCommands.RESET), agent_seed)
                assert resp.done
            except rospy.ServiceException:
                logger.info("Failed to reset agent!")
            
            # evaluate episode
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
                    per_episode_metrics = {
                        NumericalMetrics.DISTANCE_TO_GOAL: resp.distance_to_goal,
                        NumericalMetrics.SUCCESS: resp.success,
                        NumericalMetrics.SPL: resp.spl,
                        NumericalMetrics.NUM_STEPS: resp.num_steps,
                        NumericalMetrics.AGENT_TIME: resp.agent_time,
                        NumericalMetrics.SIM_TIME: resp.sim_time,
                        NumericalMetrics.RESET_TIME: resp.reset_time
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
                    for k, v in per_episode_metrics.items():
                        logger_per_episode.info(f"{k},{v}")

                    # add to the metrics list
                    dict_of_metrics[f"{episode_id},{scene_id}"] = per_episode_metrics

                    # increment episode counter
                    count_episodes += 1

                    # shut down the episode logger
                    utils_logging.close_logger(logger_per_episode)
                    
            except rospy.ServiceException:
                logger.info(f"Evaluation call failed at {count_episodes}-th episode")
                break

        utils_logging.close_logger(logger)

        return dict_of_metrics

    def shutdown_env_and_agent(
        self,
    ) -> None:
        r"""
        Shutdown env and agent node if the evaluator has instantiated
        them.
        """
        if self.do_not_start_nodes is False:
            self.env_process.kill()
            self.agent_process.kill()
    
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
        raise NotImplementedError

    def generate_video(
        self, episode_id: str, scene_id: str, agent_seed: int = 7, *args, **kwargs
    ) -> None:
        # TODO: we may need to implement it for Habitat agent + Gazebo or ROS planner + Habitat Sim
        raise NotImplementedError

    def generate_map(
        self,
        episode_id: str,
        scene_id: str,
        agent_seed: int,
        map_height: int,
        *args,
        **kwargs,
    ) -> np.ndarray:
        # TODO: we may need to implement it for Habitat agent + Gazebo or ROS planner + Habitat Sim
        raise NotImplementedError
