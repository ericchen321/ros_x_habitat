from typing import Dict, Optional
from habitat.core.agent import Agent


class HabitatEvaluator:
    r"""Abstract class for evaluating Habitat agents in environments either with or
    without physics.

    Users should instantiate 'HabitatDiscreteEvaluator' or 'HabitatPhysicsEvaluator'
    as sublcasses of 'HabitatEvaluator'.
    """

    def evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None, control_period: Optional[float] = 1.0
    ) -> Dict[str, float]:
        r"""..

        Args:
            agent: agent to be evaluated in environment.
            num_episodes: number of episodes for which the evaluation should be run.
            control_period: number of seconds in which each action should complete. Not
                used by subclasses for which dynamics is not simulated.

        Return:
            dict containing metrics tracked by environment.
        """
        raise NotImplementedError