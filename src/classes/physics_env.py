#!/usr/bin/env python3

from typing import Any, Dict, Iterator, List, Optional, Type, Union
import numpy as np
from gym.spaces.dict_space import Dict as SpaceDict

from habitat.core.env import Env
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.simulator import Observations, Simulator
import habitat_sim as hsim


class PhysicsEnv(Env):
    r"""Fundamental environment class for :ref:`habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied tasks with simulator
    is abstracted inside :ref:`Env`. Acts as a base for other derived
    environment classes. :ref:`Env` consists of three major components:
    ``dataset`` (`episodes`), ``simulator`` (:ref:`sim`) and :ref:`task` and
    connects all the three components together.
    """

    observation_space: SpaceDict
    action_space: SpaceDict
    _config: Config
    _dataset: Optional[Dataset]
    number_of_episodes: Optional[int]
    _episodes: List[Type[Episode]]
    _current_episode_index: Optional[int]
    _current_episode: Optional[Type[Episode]]
    _episode_iterator: Optional[Iterator]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def step_physics(
        self, action: Union[int, str, Dict[str, Any]], time_step=1.0 / 60.0,
        control_period=1.0, **kwargs
    ) -> Observations:
        r"""Perform an action in the environment, with physics enabled, and
        return observations.

        :param action: action (belonging to :ref:`action_space`) to be
            performed inside the environment. Action is a name or index of
            allowed task's action and action arguments (belonging to action's
            :ref:`action_space`) to support parametrized and continuous
            actions.
        :param time_step: time step for physics simulation
        :param control_period:
        :return: observations after taking action in environment.
        """
        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"

        # Support simpler interface as well
        if isinstance(action, str) or isinstance(action, (int, np.integer)):
            action = {"action": action}

        # Step with physics
        observations = self.task.step_physics(
            action=action, episode=self.current_episode, time_step=time_step,
            control_period=control_period, id_agent_obj=self._id_agent_obj
        )

        self._task.measurements.update_measures(
            episode=self.current_episode, action=action, task=self.task
        )

        self._update_step_stats()

        return observations

    def enable_physics(self):
        r"""Enables physics in the environment and initialize the
        simulation environment.
        """
        # attach asset to agent
        locobot_template_id = \
        self._sim._sim.load_object_configs("data/objects/locobot_merged")[0]
        # print("locobot_template_id is " + str(locobot_template_id))
        self._id_agent_obj = self._sim._sim.add_object(locobot_template_id,
                                                       self._sim._sim.agents[
                                                           0].scene_node)
        # print("id of agent object is " + str(self._id_agent_obj))

        # set all objects in scene to be dynamic
        obj_ids = self._sim._sim.get_existing_object_ids()
        for obj_id in obj_ids:
            self._sim._sim.set_object_motion_type(
                hsim.physics.MotionType.DYNAMIC, obj_id)

    def disable_physics(self):
        r"""Disables physics in the environment and clear up the
        simulation environment.
        """
        # remove all objects in the scene, but keep their scene nodes
        obj_ids = self._sim._sim.get_existing_object_ids()
        for obj_id in obj_ids:
            self._sim._sim.remove_object(object_id=obj_id,
                                         delete_object_node=False,
                                         delete_visual_node=False)
