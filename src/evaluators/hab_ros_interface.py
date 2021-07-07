#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import threading

import rospy
from geometry_msgs.msg import Twist
from habitat.config import Config
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

from src.envs.physics_env import PhysicsEnv

sys.path = [
    b for b in sys.path if "2.7" not in b
]  # remove path's related to ROS from environment or else certain packages like cv2 can't be imported

import habitat
import habitat_sim as hsim
import numpy as np

lock = threading.Lock()
rospy.init_node("habitat", anonymous=False)


class SimEnv(threading.Thread):

    _x_axis = 0
    _y_axis = 1
    _z_axis = 2
    _dt = 0.00478
    _sensor_rate = 50  # hz
    _r = rospy.Rate(_sensor_rate)

    def __init__(self, env_config_file):
        threading.Thread.__init__(self)
        self.config = habitat.get_config(env_config_file)
        self.overwrite_simulator_config()
        self.env = PhysicsEnv(config=self.config)
        # always assume height equals width
        self._sensor_resolution = {
            "RGB": self.env._sim.habitat_config["RGB_SENSOR"]["HEIGHT"],
            "DEPTH": self.env._sim.habitat_config["DEPTH_SENSOR"]["HEIGHT"],
        }
        self.obj_templates_mgr = self.env._sim._sim.get_object_template_manager()
        self.env._sim._sim.agents[0].move_filter_fn = self.env._sim._sim.step_filter
        self.observations = self.env.reset()

        # add a sphere
        # load some object templates from configuration files
        sphere_template_id = self.obj_templates_mgr.load_configs(
            str("data/test_assets/objects/sphere")
        )[0]
        id_sphere = self.env._sim._sim.add_object(sphere_template_id)

        self.env._sim._sim.set_translation(np.array([-2.63, 0.114367, 19.3]), id_sphere)

        # load and initialize the lobot_merged asset
        locobot_template_id = self.obj_templates_mgr.load_configs(
            str("data/objects/locobot_merged")
        )[0]
        # add robot object to the scene with the agent/camera SceneNode attached
        self.id_agent_obj = self.env._sim._sim.add_object(
            locobot_template_id, self.env._sim._sim.agents[0].scene_node
        )
        # set the agent's body to dynamic
        self.env._sim._sim.set_object_motion_type(
            hsim.physics.MotionType.DYNAMIC, self.id_agent_obj
        )

        self.env._sim._sim.agents[0].state.velocity = np.float32([0, 0, 0])
        self.env._sim._sim.agents[0].state.angular_velocity = np.float32([0, 0, 0])
        self.vel_control = self.env._sim._sim.get_object_velocity_control(
            self.id_agent_obj
        )
        self.env._sim._sim.set_translation(
            np.array([-1.7927, 0.114367, 19.1552]), self.id_agent_obj
        )

        self._pub_rgb = rospy.Publisher("~rgb", numpy_msg(Floats), queue_size=1)
        self._pub_depth = rospy.Publisher("~depth", numpy_msg(Floats), queue_size=1)

        self._pub_depth_and_pointgoal = rospy.Publisher(
            "depth_and_pointgoal", numpy_msg(Floats), queue_size=1
        )
        print("created habitat_plant succsefully")

    def run(self):
        """Publish sensor readings through ROS on a different thread.
        This method defines what the thread does when the start() method
        of the threading class is called
        """
        while not rospy.is_shutdown():
            lock.acquire()

            rgb_with_res = np.concatenate(
                (
                    np.float32(self.observations["rgb"].ravel()),
                    np.array(
                        [
                            self._sensor_resolution["RGB"],
                            self._sensor_resolution["RGB"],
                        ]
                    ),
                )
            )

            # multiply by 10 to get distance in meters
            depth_with_res = np.concatenate(
                (
                    np.float32(self.observations["depth"].ravel() * 10),
                    np.array(
                        [
                            self._sensor_resolution["DEPTH"],
                            self._sensor_resolution["DEPTH"],
                        ]
                    ),
                )
            )

            depth_np = np.float32(self.observations["depth"].ravel())
            pointgoal_np = np.float32(
                self.observations["pointgoal_with_gps_compass"].ravel()
            )
            lock.release()

            self._pub_rgb.publish(np.float32(rgb_with_res))
            self._pub_depth.publish(np.float32(depth_with_res))
            # self._pub_bc_sensor.publish(np.float32(bc_sensor_with_res))

            depth_pointgoal_np = np.concatenate((depth_np, pointgoal_np))
            self._pub_depth_and_pointgoal.publish(np.float32(depth_pointgoal_np))
            self._r.sleep()

    def update_observations(self):
        sim_obs = self.env._sim._sim.get_sensor_observations()
        self.observations = self.env._sim._sensor_suite.get_observations(sim_obs)
        self.observations.update(
            self.env._task.sensor_suite.get_observations(
                observations=self.observations,
                episode=self.env.current_episode,
            )
        )

    def overwrite_simulator_config(self):
        for k in self.config.PHYSICS_SIMULATOR.keys():
            if isinstance(self.config.PHYSICS_SIMULATOR[k], Config):
                for inner_k in self.config.PHYSICS_SIMULATOR[k].keys():
                    self.config.SIMULATOR[k][inner_k] = self.config.PHYSICS_SIMULATOR[
                        k
                    ][inner_k]
            else:
                self.config.SIMULATOR[k] = self.config.PHYSICS_SIMULATOR[k]
        try:
            from habitat.sims.habitat_simulator.habitat_simulator import (
                HabitatSim,
            )
            from src.sims.habitat_physics_simulator import HabitatPhysicsSim
            from src.tasks.habitat_physics_task import PhysicsNavigationTask
            from habitat.sims.habitat_simulator.actions import (
                HabitatSimV1ActionSpaceConfiguration,
            )
        except ImportError as e:
            print("Import HSIM failed")
            raise e

        return


def callback(vel, my_env):
    lock.acquire()
    pos = my_env.env.sim.get_agent_state(0).position
    # print(f"agent's position: {pos}.")
    my_env.vel_control.linear_velocity = np.array(
        [(1.0 * vel.linear.y), 0.0, (-1.0 * vel.linear.x)]
    )
    my_env.vel_control.angular_velocity = np.array([0.0, vel.angular.z, 0])
    my_env.vel_control.controlling_lin_vel = True
    my_env.vel_control.controlling_ang_vel = True
    # set velocity relative to the agent's own frame
    my_env.vel_control.lin_vel_is_local = True
    my_env.vel_control.ang_vel_is_local = True

    lock.release()


def main():

    # setup environment
    my_env = SimEnv(env_config_file="configs/pointnav_rgbd.yaml")
    # start the thread that publishes sensor readings
    my_env.start()

    # set to receive velocities from the controller
    rospy.Subscriber("/cmd_vel", Twist, callback, (my_env), queue_size=1)

    # run the simulation
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        lock.acquire()
        my_env.env._sim._sim.step_physics(1.0 / 60.0)
        my_env.env._update_step_stats()  # think this increments episode count
        my_env.update_observations()
        lock.release()
        rate.sleep()


if __name__ == "__main__":
    main()
