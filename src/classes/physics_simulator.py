from habitat_sim.simulator import Simulator
import time
import attr


@attr.s(auto_attribs=True)
class PhysicsSimulator(Simulator):

    def step_physics(self, dt):
        self._num_total_frames += 1
        collided = self.contact_test(object_id=0)

        # step physics by dt
        step_start_time = time.time()
        self.step_world(dt)
        _previous_step_time = time.time() - step_start_time

        observations = self.get_sensor_observations()
        # Whether or not the action taken resulted in a collision
        observations["collided"] = collided

        return observations
