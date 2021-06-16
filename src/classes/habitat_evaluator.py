from typing import Dict, Optional
from habitat.core.agent import Agent


class HabitatEvaluator:
    r"""Abstract class for evaluating Habitat agents in environments either with or
    without physics.

    Users should instantiate 'HabitatDiscreteEvaluator' or 'HabitatPhysicsEvaluator'
    as sublcasses of 'HabitatEvaluator'.
    """

    def evaluate(
        self, agent: Agent, *args, **kwargs
    ) -> Dict[str, float]:
        r"""..

        Args:
            agent: agent to be evaluated in environment.

        Return:
            dict containing metrics tracked by environment.
        """
        raise NotImplementedError
