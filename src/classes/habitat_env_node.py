#!/usr/bin/env python
import argparse
from typing import (
    TYPE_CHECKING,
    Optional,
)
import numpy as np
import rospy
from rospy_message_converter import message_converter
from geometry_msgs.msg import Twist
from ros_x_habitat.msg import PointGoalWithGPSCompass
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int16
from habitat.core.simulator import Observations
from cv_bridge import CvBridge, CvBridgeError
from classes.habitat_eval_rlenv import HabitatEvalRLEnv
from habitat.config.default import get_config


class HabitatEnvNode:
    def __init__(
        self, config_paths: Optional[str] = None, enable_physics: bool = False
    ):
        r"""
        Instantiates a ROS node which encapsulates a Habitat simulator.
        The node subscribes to agent command topics, and publishes sensor
        readings to sensor topics.
        """
        # initialize Habitat environment
        self.config = get_config(config_paths)
        self.enable_physics = enable_physics
        self.env = HabitatEvalRLEnv(
            config=self.config, enable_physics=self.enable_physics
        )
        self.observations = self.env.reset()

        # publish to sensor topics
        for sensor_uuid, _ in self.observations.items():
            # we create one topic for each of RGB, Depth and GPS+Compass
            # sensor
            if sensor_uuid == "rgb":
                self.pub_rgb = rospy.Publisher(sensor_uuid, Image, queue_size=10)
            elif sensor_uuid == "depth":
                self.pub_depth = rospy.Publisher(sensor_uuid, Image, queue_size=10)
            elif sensor_uuid == "pointgoal_with_gps_compass":
                self.pub_pointgoal_with_gps_compass = rospy.Publisher(sensor_uuid, PointGoalWithGPSCompass, queue_size=10)

        # subscribe to command topics
        if self.enable_physics:
            self.sub = rospy.Subscriber("cmd_vel", Twist, self.callback)
        else:
            self.sub = rospy.Subscriber("action", Int16, self.callback)

        # publish initial observations after __init__
        # encode observations as ROS messages
        observations_ros = self.obs2msgs(self.observations)
        for sensor_uuid, _ in self.observations.items():
            # we publish to each of RGB, Depth and GPS+Compass sensor
            if sensor_uuid == "rgb":
                self.pub_rgb.publish(observations_ros["rgb"])
            elif sensor_uuid == "depth":
                self.pub_depth.publish(observations_ros["depth"])
            elif sensor_uuid == "pointgoal_with_gps_compass":
                self.pub_pointgoal_with_gps_compass.publish("pointgoal_with_gps_compass")
    
    def obs2msgs(self, observations_hab: Observations):
        r"""
        Converts Habitat observations to ROS observations.

        :param observations_hab: Habitat observations.
        :return: a dictionary containing RGB/depth/Pos+Orientation readings
        in ROS Image/Pose format.
        """
        observations_ros = {}

        # take the current sim time to later use as timestamp
        # for all simulator readings
        t_curr = rospy.Time.now()

        for sensor_uuid, _ in self.observations.items():
            sensor_data = self.observations[sensor_uuid]
            # we publish to each of RGB, Depth and GPS+Compass sensor
            if sensor_uuid == "rgb": 
                img_msg_rgb = CvBridge().cv2_to_imgmsg(sensor_data.astype(np.uint8), encoding="rgb8")
                img_msg_rgb.height, img_msg_rgb.width = sensor_data.shape
                h = Header()
                h.stamp = t_curr
                img_msg_rgb.header = h
                observations_ros[sensor_uuid] = img_msg_rgb
            elif sensor_uuid == "depth":
                img_msg_depth = CvBridge().cv2_to_imgmsg(sensor_data.astype(np.uint8), encoding="mono16")
                img_msg_depth.height, img_msg_depth.width = sensor_data.shape
                h = Header()
                h.stamp = t_curr
                img_msg_depth.header = h
                observations_ros[sensor_uuid] = img_msg_depth
            elif sensor_uuid == "pointgoal_with_gps_compass":
                ptgoal_with_gps_compass_msg = PointGoalWithGPSCompass()
                ptgoal_with_gps_compass_msg.distance_to_goal = sensor_data[0]
                ptgoal_with_gps_compass_msg.angle_to_goal = sensor_data[1]
                h = Header()
                h.stamp = t_curr
                ptgoal_with_gps_compass_msg.header = h
                observations_ros[sensor_uuid] = ptgoal_with_gps_compass_msg
        return observations_ros

    def publish_sensor_observations(self, rate: int = 10):
        r"""
        Publishes simulator sensor readings indefinitely
        at given rate.
        """
        r = rospy.Rate(rate)
        
        # publish observations at fixed rate
        while not rospy.is_shutdown():
            observations_ros = self.obs2msgs(self.observations)
            for sensor_uuid, _ in self.observations.items():
                # we publish to each of RGB, Depth and GPS+Compass sensor
                if sensor_uuid == "rgb":
                    self.pub_rgb.publish(observations_ros["rgb"])
                elif sensor_uuid == "depth":
                    self.pub_depth.publish(observations_ros["depth"])
                elif sensor_uuid == "pointgoal_with_gps_compass":
                    self.pub_pointgoal_with_gps_compass.publish("pointgoal_with_gps_compass")
            r.sleep()

    def callback(self, cmd_msg):
        # unpack agent action from ROS message, and send the action
        # to the simulator 
        if self.enable_physics is True:
            # TODO: invoke step_physics() or something to set velocity
            pass 
        else:
            agent_action = cmd_msg.data
            (self.observations, _, _, _) = self.env.step(agent_action)


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config", type=str, default="configs/pointnav_d_orignal.yaml"
    )
    parser.add_argument('--enable-physics', default=False, action='store_true')
    parser.add_argument(
        "--sensor-pub-rate",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    rospy.init_node("env_node")
    HabitatEnvNode(config_paths=args.task_config, enable_physics=args.enable_physics)
    HabitatEnvNode.publish_sensor_observations(args.sensor_pub_rate)
