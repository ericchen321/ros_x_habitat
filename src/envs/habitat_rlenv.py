from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.env import Env
from habitat.core.simulator import Observations
from habitat.utils import profiling_wrapper

from src.envs.physics_env import PhysicsEnv


class HabitatRLEnv(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over :ref:`Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: :ref:`get_reward_range()`, :ref:`get_reward()`,
    :ref:`get_done()`, :ref:`get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.

    This class is modifed upon RLEnv class from Habitat Lab so it can represent
    an environment either with or without Bullet physics enabled.
    """

    _env: Union[Env, PhysicsEnv]

    def __init__(
        self,
        config: Config,
        dataset: Optional[Dataset] = None,
        enable_physics: Optional[bool] = False,
    ) -> None:
        """Constructor

        :param config: config to construct :ref:`Env`
        :param dataset: dataset to construct :ref:`Env`.
        """

        self._env = None
        self.enable_physics = enable_physics
        if self.enable_physics:
            self._env = PhysicsEnv(config, dataset)
        else:
            self._env = Env(config, dataset)

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Episode]:
        return self._env.episodes

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        self._env.episodes = episodes

    @property
    def current_episode(self) -> Episode:
        return self._env.current_episode

    @profiling_wrapper.RangeContext("RLEnv.reset")
    def reset(self) -> Observations:
        return self._env.reset()

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the :ref:`step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    @profiling_wrapper.RangeContext("RLEnv.step")
    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.
        If physics is enabled, would call `PhysicsEnv.step_physics()`;
        otherwise would call `Env.step()`.

        :return: :py:`(observations, reward, done, info)`
        """
        if self.enable_physics:
            observations = self._env.step_physics(*args, **kwargs)
        else:
            observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def set_agent_velocities(
        self, linear_vel: np.ndarray, angular_vel: np.ndarray
    ) -> None:
        r"""
        Set linear and angular velocity for the agent in the environment.
        Can only be called when physics is turned on.
        :param linear_vel: linear velocity
        :param angular_vel: angular velocity
        """
        assert self.enable_physics
        self._env.set_agent_velocities(linear_vel, angular_vel)
