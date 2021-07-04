from habitat_sim.simulator import Simulator
import time
import attr


@attr.s(auto_attribs=True)
class PhysicsSimulator(Simulator):
    def step_physics(
        self,
        agent_object,
        agent_object_id,
        dt
    ):
        r"""
        Step for one frame with physics. Unlike Simulator.step(),
        this method 1) does not complete the given action in one frame,
        and 2) can simulate environments with the default agent only.
        :param agent_object: the object that the agent embodies in.
            NOTE: we pass this argument in despite not using it because
            we will use it in the future, once Facebook people have
            implemented ManagedRigidObject.contact_test().
        :param agent_object_handle: ID of the agent object.
            NOTE: this parameter would become obsolete once Facebook peopl
            have implemented ManagedRigidObject.contact_test().
        :param dt: simulation time step.
        """
        self._num_total_frames += 1
        agent = self.get_agent(self._default_agent_id)
        self._Simulator__last_state[self._default_agent_id] = agent.get_state()

        # step physics by dt
        step_start_Time = time.time()
        super().step_world(dt)
        self._previous_step_time = time.time() - step_start_Time

        # collision detection
        default_agent_observations = self.get_sensor_observations(agent_ids=[self._default_agent_id])[self._default_agent_id]
        # NOTE: use ManagedRigidObject.contact_test() once it's implemented
        default_agent_observations["collided"] = self.contact_test(agent_object_id)
        
        return default_agent_observations
