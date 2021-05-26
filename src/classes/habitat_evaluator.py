from typing import Dict, Optional
from habitat.core.agent import Agent


class HabitatEvaluator:
    r"""Abstract class for evaluating Habitat agents in environments either with or
    without physics.

    Users should instantiate 'HabitatDiscreteEvaluator' or 'HabitatPhysicsEvaluator'
    as sublcasses of 'HabitatEvaluator'.
    """

    def evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None, *args, **kwargs
    ) -> Dict[str, float]:
        r"""..

        Args:
            agent: agent to be evaluated in environment.
            num_episodes: number of episodes for which the evaluation should be run.

        Return:
            dict containing metrics tracked by environment.
        """
        raise NotImplementedError
