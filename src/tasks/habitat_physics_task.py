from typing import Any, Optional, Union, Dict, Type

import habitat_sim as hsim
import numpy as np
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
)
from habitat.core.registry import registry
from habitat.core.simulator import (
    Simulator,
)
from habitat.tasks.nav.nav import (
    merge_sim_episode_config,
    SimulatorTaskAction,
    MoveForwardAction,
    TurnLeftAction,
    TurnRightAction,
    StopAction,
)
import pandas as pd
import math
import os
from habitat.utils.geometry_utils import angle_between_quaternions


@registry.register_task(name="Nav-Phys")
class PhysicsNavigationTask(EmbodiedTask):
    r"""
    Task to do point-goal navigation but with physics enabled. This class is the
    dual of habitat.nav.nav.PointGoalNavigationTask (which is the class for nav
    without physics).
    """

    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.df = pd.DataFrame(columns=["action", "desired_value", "actual_value"])
        self.df_name = None

    def reset(self, episode: Episode):
        observations = self._sim.reset()
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode, task=self
            )
        )

        for action_instance in self.actions.values():
            action_instance.reset(episode=episode, task=self)

        self.is_stop_called = False

        return observations

    def step_physics(
        self,
        action: Union[int, Dict[str, Any]],
        episode: Type[Episode],
        time_step: float,
        control_period: float,
        agent_object: hsim.physics.ManagedRigidObject,
    ):
        if action is None:
            # step by one frame
            observations = self._sim.step_physics(agent_object, time_step)
        else:
            # step multiple frames to complete an action
            if "action_args" not in action or action["action_args"] is None:
                action["action_args"] = {}
            action_name = action["action"]
            if isinstance(action_name, (int, np.integer)):
                action_name = self.get_action_name(action_name)
            assert (
                action_name in self.actions
            ), f"Can't find '{action_name}' action in {self.actions.keys()}."
            task_action = self.actions[action_name]

            # complete the action with physics and collect observations
            observations = None
            if isinstance(task_action, StopAction):
                # if given stop command, collect observations immediately
                observations = self._sim.get_observations_at()
                self.is_stop_called = True
            else:
                # Setting agent velocity controls based on each action type
                self._set_agent_velocities(
                    action=task_action,
                    agent_vel_control=agent_object.velocity_control,
                    control_period=control_period,
                )
                # step through all frames in the control period
                total_steps = round(control_period * 1.0 / time_step)

                # save previous position/rotation
                # comment out when not collecting actuation error data
                current_df_name = (
                    "actuation_data/"
                    + str(episode.scene_id).split("/")[-1]
                    + "_"
                    + str(episode.episode_id)
                    + "_actuation.csv"
                )
                if self.df_name != current_df_name:
                    self.df = self.df.iloc[0:0]
                    self.df_name = current_df_name

                current_position = self._sim.get_agent_state().position
                current_rotation = self._sim.get_agent_state().rotation
                for frame in range(0, total_steps):
                    observations = self._sim.step_physics(agent_object, time_step)
                    # if collision occurred, quit the loop immediately
                    # NOTE: this is not working yet
                    # if self._sim.previous_step_collided:
                    #    break
                # log position/rotation after stepping
                # NOTE: to get angle between quarternions, use angle_between_quaternions()
                # from geometry_utils in habitat
                # comment out when not collecting actuation error data
                new_position = self._sim.get_agent_state().position
                new_rotation = self._sim.get_agent_state().rotation
                displacement = np.linalg.norm(new_position - current_position)
                angle_diff = math.degrees(
                    angle_between_quaternions(new_rotation, current_rotation)
                )
                if isinstance(task_action, TurnLeftAction):
                    self.df = self.df.append(
                        {
                            "action": "TurnLeft",
                            "desired_value": 10.0,
                            "actual_value": angle_diff,
                        },
                        ignore_index=True,
                    )
                elif isinstance(task_action, TurnRightAction):
                    self.df = self.df.append(
                        {
                            "action": "TurnRight",
                            "desired_value": 10.0,
                            "actual_value": angle_diff,
                        },
                        ignore_index=True,
                    )
                elif isinstance(task_action, MoveForwardAction):
                    self.df = self.df.append(
                        {
                            "action": "MoveForward",
                            "desired_value": 0.25,
                            "actual_value": displacement,
                        },
                        ignore_index=True,
                    )
                else:
                    pass

                if not self.df.empty:
                    print("Updating csv: " + self.df_name)
                    self.df.to_csv(self.df_name, index=False)

        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )

        self._is_episode_active = self._check_episode_is_active(
            observations=observations, action=action, episode=episode
        )

        return observations

    def overwrite_sim_config(self, sim_config: Any, episode: Episode) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)

    def _set_agent_velocities(
        self,
        action: SimulatorTaskAction,
        agent_vel_control: hsim.physics.VelocityControl,
        control_period: float,
    ) -> None:
        r"""
        Given action, convert it to linear and angular velocities, and set
        the agent's control handle.
        :param action: class of given action.
        :param agent_vel_control: control handle of the agent, to be set.
        :param control_period: duration for which an action lasts.
        """
        # Make agent velocities controllable
        agent_vel_control.controlling_lin_vel = True
        agent_vel_control.controlling_ang_vel = True
        agent_vel_control.lin_vel_is_local = True
        agent_vel_control.ang_vel_is_local = True

        # Setting agent velocity controls based on each action type
        if isinstance(action, StopAction):
            agent_vel_control.linear_velocity = np.float32([0, 0, 0])
            agent_vel_control.angular_velocity = np.float32([0, 0, 0])
        elif isinstance(action, MoveForwardAction):
            # set linear velocity. 0.25m is defined as _C.SIMULATOR.FORWARD_STEP_SIZE
            # in habitat/config/default.py. By default, forward movement happens on the
            # local z axis.
            # TODO: programmatically load this value
            agent_vel_control.linear_velocity = np.float32(
                [0, 0, -0.25 / control_period]
            )
            agent_vel_control.angular_velocity = np.float32([0, 0, 0])
        elif isinstance(action, TurnLeftAction):
            # set angular velocity. 10 deg is defined as _C.SIMULATOR.TURN_ANGLE
            # in habitat/config/default.py. By default, turning left/right is about
            # the local y axis.
            # TODO: programmatically load this value
            agent_vel_control.linear_velocity = np.float32([0, 0, 0])
            agent_vel_control.angular_velocity = np.float32(
                [0, (np.deg2rad(10.0) / control_period), 0]
            )
        elif isinstance(action, TurnRightAction):
            agent_vel_control.linear_velocity = np.float32([0, 0, 0])
            agent_vel_control.angular_velocity = np.float32(
                [0, (np.deg2rad(-10.0) / control_period), 0]
            )
