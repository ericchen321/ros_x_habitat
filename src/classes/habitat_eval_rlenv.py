
from classes.habitat_rlenv import HabitatRLEnv

class HabitatEvalRLEnv(HabitatRLEnv):
    r""" Custom RL environment for HabitatEvaluator.
    """

    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()