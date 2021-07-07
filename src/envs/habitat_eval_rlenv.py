from logging import Logger

from habitat.core.simulator import Observations

from src.envs.habitat_rlenv import HabitatRLEnv


class HabitatEvalRLEnv(HabitatRLEnv):
    r"""Custom RL environment for Evaluator."""

    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def iter_to_episode(
        self, episode_id: str, scene_id: str, logger: Logger
    ) -> Observations:
        r"""
        Advance the environment's episode iterator to the given episode.
        :param episode_id: ID of the episode to iterate to
        :param scene_id: Scene ID of the episode to iterate to
        :returns: initial observations from the episode after reset.
        """
        # iterate to the last episode. If not found, the loop exits upon a
        # StopIteration exception
        last_ep_found = False
        while not last_ep_found:
            try:
                obs = self._env.reset()
                e = self._env.current_episode
                if (str(e.episode_id) == str(episode_id)) and (e.scene_id == scene_id):
                    logger.info(
                        f"Last episode found: episode-id={episode_id}, scene-id={scene_id}"
                    )
                    last_ep_found = True
                    return obs
            except StopIteration:
                logger.info("Last episode not found!")
                raise StopIteration

    def reset_episode_iterator(self) -> None:
        r"""
        Reset the environment's episode iterator.
        """
        # get a new episode iterator
        iter_option_dict = {
            k.lower(): v
            for k, v in self._env._config.ENVIRONMENT.ITERATOR_OPTIONS.items()
        }
        iter_option_dict["seed"] = self._env._config.SEED
        self._env._episode_iterator = self._env._dataset.get_episode_iterator(
            **iter_option_dict
        )
