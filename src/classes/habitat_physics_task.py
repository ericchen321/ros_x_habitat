from typing import Any, Optional, Union, Dict, Type
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
from habitat.sims.habitat_simulator.actions import _DefaultHabitatSimActions
from habitat.tasks.nav.nav import (
    SimulatorTaskAction,
    MoveForwardAction,
    TurnLeftAction,
    TurnRightAction,
    StopAction
)
import habitat_sim as hsim
from habitat.utils.geometry_utils import angle_between_quaternions
import magnum as mn


def merge_sim_episode_config(sim_config: Config, episode: Episode) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if episode.start_position is not None and episode.start_rotation is not None:
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@registry.register_task(name="Nav-v1")
class PhysicsNavigationTask(EmbodiedTask):
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
    
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
        agent_object_id: int
    ):
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
            total_steps = int(control_period * 1.0 / time_step)
            if isinstance(task_action, TurnLeftAction):
                agent_init_rotation = self._sim.agents[0].get_state().rotation
            for frame in range(0, total_steps):
                observations = self._sim.step_physics(agent_object, agent_object_id, time_step)
                # if collision occurred, quit the loop immediately
                # NOTE: this is not working yet
                #if self._sim.previous_step_collided:
                #    break
            if isinstance(task_action, TurnLeftAction):
                agent_final_rotation = self._sim.agents[0].get_state().rotation
                angle_between_rotations = np.rad2deg(angle_between_quaternions(agent_final_rotation, agent_init_rotation))
                print(f"rotated angle: {angle_between_rotations} degs")

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
            agent_vel_control.angular_velocity = np.float32([0, (np.deg2rad(10.0)/control_period), 0])
        elif isinstance(action, TurnRightAction):
            agent_vel_control.linear_velocity = np.float32([0, 0, 0])
            agent_vel_control.angular_velocity = np.float32([0, (np.deg2rad(-10.0)/control_period), 0])
