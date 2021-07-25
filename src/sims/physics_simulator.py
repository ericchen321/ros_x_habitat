import time

import attr
from habitat_sim.simulator import Simulator, Configuration


@attr.s(auto_attribs=True)
class PhysicsSimulator(Simulator):
    r"""Subclass of 'Simulator' class of habitat-sim. Provides a better
    implementation of step_physics().

    :property config: configuration for the simulator

    The simulator ties together the backend, the agent, controls functions,
    NavMesh collision checking/pathfinding, attribute template management,
    object manipulation, and physics simulation.
    """

    def step_physics(self, agent_object, dt):
        r"""
        Step for one frame with physics. Unlike Simulator.step(),
        this method 1) does not complete the given action in one frame,
        and 2) can simulate environments with the default agent only.

        :param agent_object: the object that the agent embodies in.
        :param dt: simulation time step.

        :returns: sensor observations from the default agent.
        """
        self._num_total_frames += 1
        agent = self.get_agent(self._default_agent_id)
        self._Simulator__last_state[self._default_agent_id] = agent.get_state()

        # step physics by dt
        step_start_Time = time.time()
        super().step_world(dt)
        self._previous_step_time = time.time() - step_start_Time

        # collision detection
        default_agent_observations = self.get_sensor_observations(
            agent_ids=[self._default_agent_id]
        )[self._default_agent_id]
        default_agent_observations["collided"] = agent_object.contact_test()

        return default_agent_observations

    def reconfigure(self, config: Configuration) -> None:
        self._sanitize_config(config)
        self._Simulator__set_from_config(config)
        self.config = config
