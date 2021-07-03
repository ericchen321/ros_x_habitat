from src.classes.evaluator import Evaluator
from habitat.config.default import get_config
from habitat.config import Config


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
                    config.SIMULATOR[k][inner_k] = config.PHYSICS_SIMULATOR[
                        k
                    ][inner_k]
            else:
                config.SIMULATOR[k] = config.PHYSICS_SIMULATOR[k]
        try:
            from habitat.sims.habitat_simulator.habitat_simulator import (
                HabitatSim,
            )
            from src.classes.habitat_physics_simulator import HabitatPhysicsSim
            from src.classes.habitat_physics_task import PhysicsNavigationTask
            from habitat.sims.habitat_simulator.actions import (
                HabitatSimV1ActionSpaceConfiguration,
            )
        except ImportError as e:
            print("Import HSIM failed")
            raise e

        return
