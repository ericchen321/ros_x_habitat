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

    def step_physics(
        self,
        action: Union[int, Dict[str, Any]],
        episode: Type[Episode],
        time_step,
        control_period,
        id_agent_obj,
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

        # step through all frames in the control period
        total_steps = int(control_period * 1.0 / time_step)
        observations = None
        for i in range(0, total_steps):
            observations_per_step = task_action.step_physics(
                **action["action_args"],
                task=self,
                time_step=time_step,
                control_period=control_period,
                id_agent_obj=id_agent_obj,
            )
            # we only collect observations from the last frame
            if i == total_steps - 1:
                observations = observations_per_step

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

    def is_episode_active(self):
        return super().is_episode_active
