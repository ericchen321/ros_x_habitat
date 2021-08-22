import os
import shlex
from subprocess import Popen
from typing import List, Tuple, Dict
import numpy as np
import rospy
from ros_x_habitat.srv import EvalEpisode, ResetAgent, GetAgentTime
from src.constants.constants import NumericalMetrics
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
from src.constants.constants import (
    AgentResetCommands,
    EvalEpisodeSpecialIDs,
    PACKAGE_NAME,
    ServiceNames,
)
from src.utils import utils_logging


class HabitatROSEvaluator(HabitatSimEvaluator):
    r"""Class to evaluate Habitat agents in Habitat environments with ROS
    as middleware.
    """

    def __init__(
        self,
        config_paths: str,
        input_type: str,
        model_path: str,
        enable_physics: bool = False,
        node_name: str = "habitat_ros_evaluator_node",
        env_node_name: str = "env_node",
        agent_node_name: str = "agent_node",
        sensor_pub_rate: float = 5.0,
        do_not_start_nodes: bool = False,
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param input_type: agent's input type, options: "rgb", "rgbd",
            "depth", "blind"
        :param model_path: path to agent's model
        :param enable_physics: use dynamic simulation or not in the Habitat
            environment
        :param env_node_name: name of the env node
        :param agent_node_name: name of the agent node
        :param sensor_pub_rate: rate at which the env node publishes sensor
            readings
        :param do_not_start_nodes: if True then the evaluator would not start
            the env node and the agent node.
        """
        super().__init__(
            config_paths=config_paths,
            input_type=input_type,
            model_path=model_path,
            enable_physics=enable_physics,
        )

        # check if agent input type is valid
        assert input_type in ["rgb", "rgbd", "depth", "blind"]

        self.node_name = node_name
        self.env_node_name = env_node_name
        self.agent_node_name = agent_node_name

        # parse args for agent node
        agent_node_args = shlex.split(
            f"python src/nodes/habitat_agent_node.py --node-name {self.agent_node_name} --input-type {input_type} --model-path {model_path} --sensor-pub-rate {sensor_pub_rate}"
        )

        # parse args for env node
        if enable_physics:
            # physics sim + discrete agent
            env_node_args = shlex.split(
                f"python src/nodes/habitat_env_node.py --node-name {self.env_node_name} --task-config {config_paths} --enable-physics-sim --sensor-pub-rate {sensor_pub_rate}"
            )
        else:
            # discrete sim + discrete agent
            env_node_args = shlex.split(
                f"python src/nodes/habitat_env_node.py --node-name {self.env_node_name} --task-config {config_paths} --sensor-pub-rate {sensor_pub_rate}"
            )

        # start an agent node and an env node
        self.do_not_start_nodes = do_not_start_nodes
        if do_not_start_nodes is False:
            self.agent_process = Popen(agent_node_args)
            self.env_process = Popen(env_node_args)

        # start the evaluator node
        rospy.init_node(self.node_name)

        # resolve service names
        if self.do_not_start_nodes:
            self.eval_episode_service_name = (
                f"{PACKAGE_NAME}/mock_env_node/{ServiceNames.EVAL_EPISODE}"
            )
            self.reset_agent_service_name = (
                f"{PACKAGE_NAME}/mock_agent_node/{ServiceNames.RESET_AGENT}"
            )
            self.get_agent_time_service_name = (
                f"{PACKAGE_NAME}/mock_agent_node/{ServiceNames.GET_AGENT_TIME}"
            )
        else:
            self.eval_episode_service_name = (
                f"{PACKAGE_NAME}/{self.env_node_name}/{ServiceNames.EVAL_EPISODE}"
            )
            self.reset_agent_service_name = (
                f"{PACKAGE_NAME}/{self.agent_node_name}/{ServiceNames.RESET_AGENT}"
            )
            self.get_agent_time_service_name = (
                f"{PACKAGE_NAME}/{self.agent_node_name}/{ServiceNames.GET_AGENT_TIME}"
            )

        # set up eval episode service client
        self.eval_episode = rospy.ServiceProxy(
            self.eval_episode_service_name, EvalEpisode
        )
        # set up agent reset service client
        self.reset_agent = rospy.ServiceProxy(self.reset_agent_service_name, ResetAgent)
        # establish agent time service client
        self.get_agent_time = rospy.ServiceProxy(
            self.get_agent_time_service_name, GetAgentTime
        )

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

        # evaluate episodes, starting from the one after the last episode
        # evaluated
        while not rospy.is_shutdown():
            # reset agent
            rospy.wait_for_service(self.reset_agent_service_name)
            try:
                resp = self.reset_agent(int(AgentResetCommands.RESET), agent_seed)
                assert resp.done
            except rospy.ServiceException:
                logger.info("Failed to reset agent!")

            # evaluate one episode and get metrics from the env node
            rospy.wait_for_service(self.eval_episode_service_name)
            resp = None
            try:
                # request env node to evaluate an episode
                if count_episodes == 0:
                    # jump to the first episode we want to evaluate
                    resp = self.eval_episode(episode_id_last, scene_id_last)
                else:
                    # evaluate the next episode
                    resp = self.eval_episode(EvalEpisodeSpecialIDs.REQUEST_NEXT, "")
            except rospy.ServiceException:
                logger.info(f"Evaluation call failed at {count_episodes}-th episode")
                raise rospy.ServiceException

            if resp.episode_id == EvalEpisodeSpecialIDs.RESPONSE_NO_MORE_EPISODES:
                # no more episodes
                logger.info(f"Finished evaluation after: {count_episodes} episodes")
                break
            else:
                # extract per-episode metrics from the env
                per_episode_metrics = {
                    NumericalMetrics.DISTANCE_TO_GOAL: resp.distance_to_goal,
                    NumericalMetrics.SUCCESS: resp.success,
                    NumericalMetrics.SPL: resp.spl,
                    NumericalMetrics.NUM_STEPS: resp.num_steps,
                    NumericalMetrics.SIM_TIME: resp.sim_time,
                    NumericalMetrics.RESET_TIME: resp.reset_time,
                }

                # get the agent time of this episode
                rospy.wait_for_service(self.get_agent_time_service_name)
                try:
                    agent_time_resp = self.get_agent_time()
                    per_episode_metrics[
                        NumericalMetrics.AGENT_TIME
                    ] = agent_time_resp.agent_time
                except rospy.ServiceException:
                    logger.info(
                        f"Failed to get agent time at episode={resp.episode_id}, scene={resp.scene_id}"
                    )
                    raise rospy.ServiceException

                # set up per-episode logger
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

        utils_logging.close_logger(logger)

        return dict_of_metrics

    def shutdown_env_node(self):
        r"""
        Signal the env node to shutdown.
        """
        rospy.wait_for_service(self.eval_episode_service_name)
        try:
            resp = self.eval_episode(EvalEpisodeSpecialIDs.REQUEST_SHUTDOWN, "")
        except rospy.ServiceException:
            print("Shutting down env node failed")
            raise rospy.ServiceException

    def shutdown_agent_node(self):
        r"""
        Signal the agent node to shutdown.
        """
        rospy.wait_for_service(self.reset_agent_service_name)
        try:
            resp = self.reset_agent(int(AgentResetCommands.SHUTDOWN), 0)
            assert resp.done
        except rospy.ServiceException:
            print("Failed to shut down agent!")
            raise rospy.ServiceException

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

    def generate_videos(
        self,
        episode_ids: List[str],
        scene_ids: List[str],
        agent_seed: int = 7,
        *args,
        **kwargs,
    ) -> None:
        # TODO: we may need to implement it for Habitat agent + Gazebo or ROS planner + Habitat Sim
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
        # TODO: we may need to implement it for Habitat agent + Gazebo or ROS planner + Habitat Sim
        raise NotImplementedError

    def get_blank_maps(
        self,
        episode_ids: List[str],
        scene_ids: List[str],
        map_height: int,
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError
