#!/usr/bin/env python3

from typing import Any, Dict, Iterator, List, Optional, Type, Union

import habitat_sim as hsim
import numpy as np
from gym.spaces import Dict as SpaceDict
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.env import Env
from habitat.core.simulator import Observations, Simulator


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

    def __init__(self, config: Config, dataset: Optional[Dataset] = None) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """
        super().__init__(config, dataset)

        # declare the physics object attributes manager;
        # this should be (re)-instantiated on every reset
        self.obj_templates_mgr = None

        # declare the rigid object manager;
        # this should be (re)-instantiated on every reset
        self.rigid_obj_mgr = None

        # declare the agent object
        self.agent_object = None

    def step_physics(
        self,
        action: Union[int, str, Dict[str, Any]] = None,
        time_step=1.0 / 60.0,
        control_period=1.0,
        **kwargs
    ) -> Observations:
        r"""Perform an action in the environment, with physics enabled, and
        return observations.

        :param action: action (belonging to :ref:`action_space`) to be
            performed inside the environment. Action is a name or index of
            allowed task's action and action arguments (belonging to action's
            :ref:`action_space`) to support parametrized and continuous
            actions. If Action is none, then iterate by one time step only.
        :param time_step: time step for physics simulation
        :param control_period:
        :return: observations after taking action in environment.
        """
        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        # NOTE: we relax the episode-is-not-over condition due
        # to the roam mode
        # assert (
        #    self._episode_over is False
        # ), "Episode over, call reset before calling step"

        # Support simpler interface as well
        if action is not None and (
            isinstance(action, str) or isinstance(action, (int, np.integer))
        ):
            action = {"action": action}

        # Step with physics
        # double-check to make sure the agent object is already define by this point
        assert self.agent_object is not None
        observations = self.task.step_physics(
            action=action,
            episode=self.current_episode,
            time_step=time_step,
            control_period=control_period,
            agent_object=self.agent_object,
        )

        self._task.measurements.update_measures(
            episode=self.current_episode, action=action, task=self.task
        )

        self._update_step_stats()

        return observations

    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.
        Also initialize physics stuff on the basis of the tutorial from
        https://aihabitat.org/docs/habitat-sim/managed-rigid-object-tutorial.html.

        :return: initial observations from the environment.
        """
        self._reset_stats()

        assert len(self.episodes) > 0, "Episodes list is empty"
        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth
        # it
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        self._current_episode = next(self._episode_iterator)

        # remove all objects in the scene, but keep their scene nodes
        obj_handles = self._sim.get_rigid_object_manager().get_object_handles()
        for obj_handle in obj_handles:
            self.rigid_obj_mgr.remove_object_by_handle(
                obj_handle, delete_object_node=False, delete_visual_node=False
            )

        # restart the simulator instance
        self.reconfigure(self._config)

        observations = self.task.reset(episode=self.current_episode)
        self._task.measurements.reset_measures(
            episode=self.current_episode, task=self.task
        )

        # re-instantiate the physics object attributes manager
        self.obj_templates_mgr = self._sim.get_object_template_manager()
        # re-instantiate the rigid object manager
        self.rigid_obj_mgr = self._sim.get_rigid_object_manager()

        # load locobot asset and attach it to the agent's scene node
        locobot_template_id = self.obj_templates_mgr.load_configs(
            "data/objects/locobot_merged"
        )[0]
        self.obj_templates_mgr.get_template_by_id(
            locobot_template_id
        ).angular_damping = 0.0
        self.agent_object = self.rigid_obj_mgr.add_object_by_template_id(
            locobot_template_id, self._sim.agents[0].scene_node
        )
        assert self.agent_object is not None

        # set all objects in the scene to be dynamic and collidable
        obj_handles = self.rigid_obj_mgr.get_object_handles()
        for obj_handle in obj_handles:
            obj = self.rigid_obj_mgr.get_object_by_handle(obj_handle)
            obj.motion_type = hsim.physics.MotionType.DYNAMIC
            obj.collidable = True

        return observations

    def set_agent_velocities(
        self, linear_vel: np.ndarray, angular_vel: np.ndarray
    ) -> None:
        r"""
        Set linear and angular velocity for the agent in the environment.
        Can only be called when physics is turned on.
        :param linear_vel: linear velocity
        :param angular_vel: angular velocity
        """
        vel_control = self.agent_object.velocity_control
        vel_control.controlling_lin_vel = True
        vel_control.controlling_ang_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.ang_vel_is_local = True
        vel_control.linear_velocity = linear_vel
        vel_control.angular_velocity = angular_vel
