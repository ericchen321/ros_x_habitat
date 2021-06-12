#!/usr/bin/env python
import argparse
from typing import (
    TYPE_CHECKING,
    Optional,
)
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from rospy.numpy_msg import numpy_msg
from ros_x_habitat.msg import PointGoalWithGPSCompass, DepthImage
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int16
from habitat.core.simulator import Observations
from cv_bridge import CvBridge, CvBridgeError
from src.classes.habitat_eval_rlenv import HabitatEvalRLEnv
from habitat.config.default import get_config
from threading import Condition


class HabitatEnvNode:
    r"""
    A class to represent a ROS node with a Habitat simulator inside.
    The node subscribes to agent command topics, and publishes sensor
    readings to sensor topics.
    """
    def __init__(
        self, config_paths: Optional[str] = None, enable_physics: bool = False, episode_id_last: str = "-1", scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    ):
        # initialize Habitat environment
        self.config = get_config(config_paths)
        # embed top-down map and heading sensor in config
        self.config.defrost()
        self.config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        # config_env.TASK.SENSORS.append("HEADING_SENSOR")
        self.config.freeze()
        self.enable_physics = enable_physics
        self.env = HabitatEvalRLEnv(
            config=self.config, enable_physics=self.enable_physics
        )

        # locate the last episode specified and iterate to the episode
        # after that one
        if episode_id_last != "-1":
            # iterate to the last episode. If not found, the loop exits upon a
            # StopIteration exception
            last_ep_found = False
            while not last_ep_found:
                try:
                    self.env.reset()
                    e = self.env._env.current_episode
                    if (e.episode_id == episode_id_last) and (
                        e.scene_id == scene_id_last
                    ):
                        print(f"Last episode found: episode-id={episode_id_last}, scene-id={scene_id_last}"
                        )
                        last_ep_found = True
                except StopIteration:
                    print("Last episode not found!")
                    raise StopIteration
        else:
            print(f"No last episode specified. Proceed to evaluate from beginning")
        self.observations = self.env.reset()

        # agent action and variables to keep things synchronized
        # NOTE: self.action_cv's lock guards self.action, self.new_action_published,
        # self.observations
        self.action = None
        self.new_action_published = False
        self.action_cv = Condition()

        # environment publish and subscribe queue size
        # TODO: make them configurable by constructor argument
        self.sub_queue_size = 10
        self.pub_queue_size = 10

        # publish to sensor topics
        for sensor_uuid, _ in self.observations.items():
            # we create one topic for each of RGB, Depth and GPS+Compass
            # sensor
            if sensor_uuid == "rgb":
                self.pub_rgb = rospy.Publisher(sensor_uuid, Image, queue_size=self.pub_queue_size)
            elif sensor_uuid == "depth":
                self.pub_depth = rospy.Publisher(sensor_uuid, numpy_msg(DepthImage), queue_size=self.pub_queue_size)
            elif sensor_uuid == "pointgoal_with_gps_compass":
                self.pub_pointgoal_with_gps_compass = rospy.Publisher(sensor_uuid, PointGoalWithGPSCompass, queue_size=10)

        # subscribe to command topics
        if self.enable_physics:
            self.sub = rospy.Subscriber("cmd_vel", Twist, self.callback, queue_size=self.sub_queue_size)
        else:
            self.sub = rospy.Subscriber("action", Int16, self.callback, queue_size=self.sub_queue_size)

        # wait until connections with the agent is established
        print("env making sure agent is subscribed to sensor topics...")
        while self.pub_rgb.get_num_connections() == 0 or self.pub_depth.get_num_connections() == 0 or self.pub_pointgoal_with_gps_compass.get_num_connections() == 0:
            pass

        print("env initialized")
    
    def cv2_to_depthmsg(self, depth_img: DepthImage):
        r"""
        Converts a Habitat depth image to a ROS DepthImage message.
        :param depth_img: depth image as a numpy array
        :returns: a ROS DepthImage message
        """
        depth_msg = DepthImage()
        depth_msg.height, depth_msg.width, _ = depth_img.shape
        depth_msg.step = depth_msg.width
        depth_msg.data = np.ravel(depth_img)
        return depth_msg
    
    def obs_to_msgs(self, observations_hab: Observations):
        r"""
        Converts Habitat observations to ROS messages.

        :param observations_hab: Habitat observations.
        :return: a dictionary containing RGB/depth/Pos+Orientation readings
        in ROS Image/Pose format.
        """
        observations_ros = {}

        # take the current sim time to later use as timestamp
        # for all simulator readings
        t_curr = rospy.Time.now()

        for sensor_uuid, _ in observations_hab.items():
            sensor_data = observations_hab[sensor_uuid]
            # we publish to each of RGB, Depth and GPS+Compass sensor
            if sensor_uuid == "rgb": 
                rgb_msg = CvBridge().cv2_to_imgmsg(sensor_data.astype(np.uint8), encoding="passthrough")
                h = Header()
                h.stamp = t_curr
                rgb_msg.header = h
                observations_ros[sensor_uuid] = rgb_msg
            elif sensor_uuid == "depth":
                depth_msg = self.cv2_to_depthmsg(sensor_data)
                h = Header()
                h.stamp = t_curr
                depth_msg.header = h
                observations_ros[sensor_uuid] = depth_msg
            elif sensor_uuid == "pointgoal_with_gps_compass":
                ptgoal_with_gps_compass_msg = PointGoalWithGPSCompass()
                ptgoal_with_gps_compass_msg.distance_to_goal = sensor_data[0]
                ptgoal_with_gps_compass_msg.angle_to_goal = sensor_data[1]
                h = Header()
                h.stamp = t_curr
                ptgoal_with_gps_compass_msg.header = h
                observations_ros[sensor_uuid] = ptgoal_with_gps_compass_msg
        return observations_ros

    def publish_sensor_observations(self):
        r"""
        Publishes current simulator sensor readings.
        """
        # pack observations in ROS message
        with self.action_cv:
            observations_ros = self.obs_to_msgs(self.observations)
            for sensor_uuid, _ in self.observations.items():
                # we publish to each of RGB, Depth and GPS+Compass sensor
                if sensor_uuid == "rgb":
                    self.pub_rgb.publish(observations_ros["rgb"])
                elif sensor_uuid == "depth":
                    self.pub_depth.publish(observations_ros["depth"])
                elif sensor_uuid == "pointgoal_with_gps_compass":
                    self.pub_pointgoal_with_gps_compass.publish(observations_ros["pointgoal_with_gps_compass"])
            #print("published observations")  

    def step(self):
        r"""
        Enact a new command and update sensor observations.
        """
        with self.action_cv:
            while self.new_action_published is False:
                self.action_cv.wait()
            self.new_action_published = False
            (self.observations, _, _, _) = self.env.step(self.action)

    def callback(self, cmd_msg):
        r"""
        Takes in a command from an agent and alert the simulator to enact
        it.
        :param cmd_msg: Either a velocity command or an action command.
        """
        # unpack agent action from ROS message, and send the action
        # to the simulator 
        if self.enable_physics is True:
            # TODO: invoke step_physics() or something to set velocity
            pass 
        else:
            with self.action_cv:
                self.action = cmd_msg.data
                self.new_action_published = True
                self.action_cv.notify()


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
    parser.add_argument("--episode-id", type=str, default="-1")
    parser.add_argument(
        "--scene-id",
        type=str,
        default="/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    )
    args = parser.parse_args()

    rospy.init_node("env_node")
    env_node = HabitatEnvNode(config_paths=args.task_config, enable_physics=args.enable_physics, episode_id_last=args.episode_id, scene_id_last=args.scene_id_last)
    
    # publish observations at fixed rate
    r = rospy.Rate(args.sensor_pub_rate)
    while not rospy.is_shutdown():
        env_node.publish_sensor_observations()
        env_node.step()
        r.sleep()
