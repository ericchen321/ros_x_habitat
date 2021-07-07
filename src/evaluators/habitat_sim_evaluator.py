from src.evaluators.evaluator import Evaluator
from habitat.config.default import get_config
from habitat.config import Config
from typing import List
from src.utils import utils_logging
import numpy as np


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
        enable_physics: bool = False
    ):
        # store experiment settings
        self.config = get_config(config_paths)
        self.input_type = input_type
        self.model_path = model_path
        self.enable_physics = enable_physics

        # create a logger
        self.logger = utils_logging.setup_logger(__name__)

    @classmethod
    def overwrite_simulator_config(cls, config):
        r"""
        Overwrite simulator and task configurations when physics is enabled.
        :param config: environment config to be overwritten.
        """
        for k in config.PHYSICS_SIMULATOR.keys():
            if isinstance(config.PHYSICS_SIMULATOR[k], Config):
                for inner_k in config.PHYSICS_SIMULATOR[k].keys():
                    config.SIMULATOR[k][inner_k] = config.PHYSICS_SIMULATOR[
                        k
                    ][inner_k]
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

    def generate_video(
        self,
        episode_id: str,
        scene_id: str,
        agent_seed: int = 7,
        *args,
        **kwargs
    ) -> None:
        r"""
        Evaluate the episode of given episode ID and scene ID, and save the video to <video_dir>/.

        :param episode_id: ID of the episode
        :param scene_id: ID of the scene
        :param agent_seed: seed for initializing agent
        """
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
        r"""
        Evaluate the episode of given episode ID and scene ID, with agent initialized by the 
        given seed. Return the top-down map.

        :param episode_id: ID of the episode
        :param scene_id: ID of the scene
        :param agent_seed: seed for initializing agent
        :param map_height: desired height of the map

        :returns: Top-down map with initial/goal position, shortest path and actual path.
        """
        raise NotImplementedError