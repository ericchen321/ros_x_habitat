#!/usr/bin/env python
import argparse
import numpy as np
import rospy
import message_filters
from message_filters import TimeSynchronizer
from ros_x_habitat.msg import PointGoalWithGPSCompass, DepthImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Int16
from geometry_msgs.msg import PoseStamped, Pose, Point
import quaternion
from habitat.tasks.utils import cartesian_to_polar
from tf.transformations import euler_from_quaternion, rotation_matrix
from visualization_msgs.msg import Marker, MarkerArray
from threading import Lock
from ros_x_habitat.srv import GetAgentPose
from src.constants.constants import PACKAGE_NAME, ServiceNames
from src.utils import utils_logging

class GazeboToHabitatAgent:
    r"""
    A class to represent a ROS node which subscribes from Gazebo sensor
    topics and publishes to Habitat agent's sensor topics.
    """

    def __init__(
        self,
        node_name: str,
        gazebo_rgb_topic_name: str,
        gazebo_depth_topic_name: str,
        gazebo_odom_topic_name: str,
        move_base_goal_topic_name: str,
        fetch_goal_from_move_base: bool=False,
        final_pointgoal_pos: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        r"""
        Instantiates the Gazebo->Habitat agent bridge.
        :param node_name: name of the bridge node
        :param gazebo_rgb_topic_name: name of the topic on which Gazebo
            publishes RGB observations
        :param gazebo_depth_topic_name: name of the topic on which Gazebo
            publishes depth observations
        :param gazebo_odom_topic_name: name of the topic on which Gazebo
            publishes odometry data
        :param move_base_goal_topic_name: name of the topic on which the
            move_base package publishes navigation goals
        :param fetch_goal_from_move_base: True if we fetch the goal position
            from topic <move_base_goal_topic_name>
        :param final_pointgoal_pos: goal location of navigation, measured
            in the world frame. If `fetch_goal_from_move_base` is True, this
            position is ignored
        """
        # initialize the node
        self.node_name = node_name
        rospy.init_node(self.node_name)

        # bridge's publish and subscribe queue size
        # TODO: make them configurable by constructor argument
        self.sub_queue_size = 1
        self.pub_queue_size = 1

        # set up logger
        self.logger = utils_logging.setup_logger(self.node_name)

        # set max number of steps for navigation
        # TODO: make them configurable by constructor argument
        self.max_steps = 500

        # last_action_done controls when to publish the next set
        # of observations
        self.last_action_lock = Lock()
        with self.last_action_lock:
            self.count_steps = 0
            self.last_action_done = True

        # current pose of the agent
        self.curr_pose_lock = Lock()
        with self.curr_pose_lock:
            self.prev_pos = None # NOTE: self.prev_pos is for visualization only
            self.curr_pos = None
            self.curr_rotation = None

        # initialize pointogal_reached flag
        self.pointgoal_reached_lock = Lock()
        with self.pointgoal_reached_lock:
            self.pointgoal_reached = False
        
        # initialize final pointgoal position
        self.final_pointgoal_lock = Lock()
        with self.final_pointgoal_lock:
            self.final_pointgoal_pos = None
            self.pointgoal_set = False
        if fetch_goal_from_move_base:
            # subscribe from goal position topic and get
            # the point-goal from there
            self.sub_goal = rospy.Subscriber(
                move_base_goal_topic_name,
                PoseStamped,
                self.callback_register_goal,
                queue_size=self.sub_queue_size
            )
        else:
            # get goal directly from constructor argument
            with self.final_pointgoal_lock:
                self.final_pointgoal_pos = final_pointgoal_pos
                self.logger.info(f"Final pointgoal position set to: {self.final_pointgoal_pos}")
                self.pointgoal_set = True

        # subscribe from Gazebo-facing sensor topics
        self.sub_rgb = message_filters.Subscriber(gazebo_rgb_topic_name, Image)
        self.sub_depth = message_filters.Subscriber(gazebo_depth_topic_name, Image)
        self.sub_odom = message_filters.Subscriber(gazebo_odom_topic_name, Odometry)
        self.ts = TimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_odom],
            queue_size=self.sub_queue_size,
        )
        self.ts.registerCallback(self.callback_obs_from_gazebo)

        # subscribe from `last_action_done/`
        self.sub_last_action_done = rospy.Subscriber(
            "last_action_done",
            Int16,
            self.callback_signal_last_action,
            queue_size=self.sub_queue_size
        )
        
        # publish to Habitat-agent-facing sensor topics
        self.pub_rgb = rospy.Publisher(
            "rgb",
            Image,
            queue_size=self.pub_queue_size
        )
        self.pub_depth = rospy.Publisher(
            "depth",
            DepthImage,
            queue_size=self.pub_queue_size
        )
        self.pub_pointgoal_with_gps_compass = rospy.Publisher(
            "pointgoal_with_gps_compass",
            PointGoalWithGPSCompass,
            queue_size=self.pub_queue_size
        )
        
        # establish get_agent_pose service server
        self.get_agent_pose_service = rospy.Service(
            f"{PACKAGE_NAME}/{node_name}/{ServiceNames.GET_AGENT_POSE}",
            GetAgentPose,
            self.get_agent_pose,
        )
        
        # publish initial/goal position for visualization
        self.marker_array_lock = Lock()
        with self.marker_array_lock:
            self.marker_array = MarkerArray()
        self.pub_init_and_goal_pos = rospy.Publisher(
            "visualization_marker_array",
            MarkerArray,
            queue_size=self.pub_queue_size
        )

        self.logger.info("gazebo -> habitat agent bridge initialized")
    
    def pos_to_point(self, pos):
        r"""
        Produce a ROS point from a position.
        :param pos: position as a (3, ) numpy array.
        :return a ROS point.
        """
        point = Point()
        point.x = pos[0]
        point.y = pos[1]
        point.z = pos[2]
        return point
    
    def add_pos_to_marker_array(self, pos_type, pos_0, pos_1=None, rot=None):
        r"""
        Add position(s) to the marker array for visualization.
        Require:
            1) self.marker_array_lock not being held by the calling
                thread
            2) the lock guarding `pos_0` and `pos_1` being held by the
                calling thread
            3) self.last_action_lock is being held by the calling thread
        :param pos_type: can only be one of the following:
            1) "curr" - path from previous to current position of the agent
            2) "init" - inital position of the agent
            3) "goal" - final goal position
        :param pos_0: position to add marker to
        :param pos_1: second position for drawing line segment; Only used if
            `pos_type` is "curr"
        :param rot: orientation of the marker
        """
        # precondition checks
        assert pos_type in ["curr", "init", "goal"]
        if pos_type == "curr":
            pos_1 is not None

        # code adapted from
        # https://answers.ros.org/question/11135/plotting-a-markerarray-of-spheres-with-rviz/
        pos_marker = Marker()
        pos_marker.id = self.count_steps
        pos_marker.header.frame_id = "odom"
        pos_marker.action = pos_marker.ADD
        
        if pos_type == "curr":
            pos_marker.type = pos_marker.LINE_STRIP
            point_0 = self.pos_to_point(pos_0)
            point_1 = self.pos_to_point(pos_1)
            pos_marker.points.append(point_0)
            pos_marker.points.append(point_1)
            pos_marker.scale.x = 0.05
            pos_marker.color.a = 1.0
            pos_marker.color.g = 1.0
        elif pos_type == "init":
            pos_marker.type = pos_marker.SPHERE
            pos_marker.pose.position.x = pos_0[0]
            pos_marker.pose.position.y = pos_0[1]
            pos_marker.pose.position.z = pos_0[2]
            pos_marker.scale.x = 0.4
            pos_marker.scale.y = 0.4
            pos_marker.scale.z = 0.4
            pos_marker.color.a = 1.0
            pos_marker.color.b = 1.0
        elif pos_type == "goal":
            pos_marker.type = pos_marker.SPHERE
            pos_marker.pose.position.x = pos_0[0]
            pos_marker.pose.position.y = pos_0[1]
            pos_marker.pose.position.z = pos_0[2]
            pos_marker.scale.x = 0.4
            pos_marker.scale.y = 0.4
            pos_marker.scale.z = 0.4
            pos_marker.color.a = 1.0
            pos_marker.color.r = 1.0
        
        with self.marker_array_lock:
            self.marker_array.markers.append(pos_marker)

    def publish_marker_array(self):
        r"""
        Publish the marker array.
        """
        with self.marker_array_lock:
            self.pub_init_and_goal_pos.publish(self.marker_array)
    
    def compute_pointgoal(self):
        r"""
        Compute distance-to-goal and angle-to-goal. Modified upon
        habitat.task.nav.nav.PointGoalSensor._compute_pointgoal().
        Require 1) `self.curr_pose_lock` and `self.final_pointgoal_lock`
        already acquired in the calling thread
        :return tuple 1) distance-to-goal, 2) angle-to-goal
        """
        # get the yaw angle
        _, _, yaw = euler_from_quaternion(self.curr_rotation)
        # build local->world matrix
        local_to_world = rotation_matrix(yaw, (0, 0, 1))
        # build world->local matrix
        world_to_local = np.linalg.inv(local_to_world)
        # compute direction vector in world frame
        direction_vector_world = np.zeros((4, ))
        direction_vector_world[0:3] = self.final_pointgoal_pos - self.curr_pos
        direction_vector_world[-1] = 0
        # compute direction vector in local frame
        direction_vector_local = np.matmul(world_to_local, direction_vector_world)[0:3]
        rho, phi = cartesian_to_polar(
            direction_vector_local[0], direction_vector_local[1]
        )
        return rho, phi
    
    def rgb_msg_to_img(self, rgb_msg, dim):
        r"""
        Extract RGB image from RGB message. Further compress the RGB
        image to size `dim` x `dim`.
        :param rgb_msg: RGB sensor reading from Gazebo
        :param dim: dimension of the RGB observation
        :return: RGB observation as a numpy array
        """
        rgb_img = CvBridge().imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        rgb_img = cv2.resize(rgb_img, (dim, dim), interpolation = cv2.INTER_AREA)
        return rgb_img

    def depth_msg_to_img(self, depth_msg, dim):
        r"""
        Extract depth image from depth message. Further compress the depth
        image to size `dim` x `dim`.
        :param depth_msg: Depth sensor reading from Gazebo
        :param dim: dimension of the depth observation
        :return: Depth observation as a numpy array
        """
        depth_img_original = np.copy(CvBridge().imgmsg_to_cv2(
            depth_msg,
            desired_encoding="passthrough"))
        # remove nan values by replacing with 0's
        # idea: https://github.com/stereolabs/zed-ros-wrapper/issues/67
        for row in range(depth_img_original.shape[0]):
            for col in range(depth_img_original.shape[1]):
                if np.isnan(depth_img_original[row][col]):
                    depth_img_original[row][col] = 0.0
        #depth_img_normalized = cv2.normalize(
        #    depth_img_original,
        #    None,
        #    alpha=0,
        #    beta=255.0,
        #    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        #)
        #depth_img_resized = cv2.resize(depth_img_normalized,
        #    (dim, dim),
        #    interpolation = cv2.INTER_AREA
        #)
        depth_img_resized = cv2.resize(depth_img_original,
            (dim, dim),
            interpolation = cv2.INTER_AREA
        )
        depth_img = np.array(depth_img_resized, dtype=np.float64)
        #depth_img_pil = PILImage.fromarray(depth_img)
        #depth_img_pil.save("gazebo_obs/depth.png", mode="L")
        return depth_img
    
    def update_pose(self, odom_msg):
        r"""
        Update current position, current rotation, previous position.
        Require `self.curr_pose_lock` already acquired by the calling
        thread
        :param odom_msg: Odometry data from Gazebo
        """
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(
            [odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z
            ]
        )
        self.curr_rotation = [
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
        ]
    
    def callback_obs_from_gazebo(self, rgb_msg, depth_msg, odom_msg):
        r"""
        Upon receiving a set of RGBD observations and odometry data from
        Gazebo, publishes
            1) the observations to `rgb/` and `depth/` topic for a Habitat
            agent to read;
            2) current GPS+Compass data.
        Only publishes iff all following conditions are satisfied:
            1) on reset or after the last action has been completed,
            2) pointogoal is set,
            3) pointgoal not reached yet.
        :param rgb_msg: RGB sensor reading from Gazebo
        :param depth_msg: Depth sensor reading from Gazebo
        :param odom_msg: Odometry data from Gazebo
        """
        with self.last_action_lock:
            with self.pointgoal_reached_lock:
                with self.final_pointgoal_lock:
                    if (
                        self.last_action_done
                        and self.pointgoal_set
                        and not self.pointgoal_reached
                        and not (self.count_steps >= self.max_steps)
                    ):
                        # publish a new set of observations if last action's done,
                        # pointgoal's set and navigation not completed yet
                        # get a header object and assign time
                        h = Header()
                        h.stamp = rospy.Time.now()
                        # create RGB message for Habitat
                        rgb_img = self.rgb_msg_to_img(rgb_msg, 720)
                        rgb_msg_for_hab = CvBridge().cv2_to_imgmsg(rgb_img, encoding="rgb8")
                        rgb_msg_for_hab.header = h
                        # create depth message for Habitat
                        depth_img = self.depth_msg_to_img(depth_msg, 720)
                        depth_msg_for_hab = DepthImage()
                        depth_msg_for_hab.height, depth_msg_for_hab.width = depth_img.shape
                        depth_msg_for_hab.step = depth_msg_for_hab.width
                        depth_msg_for_hab.data = np.ravel(depth_img)
                        depth_msg_for_hab.header = h
                        with self.curr_pose_lock:
                            # update pose and compute current GPS+Compass info
                            self.update_pose(odom_msg)
                            distance_to_goal, angle_to_goal = self.compute_pointgoal()
                        ptgoal_gps_msg = PointGoalWithGPSCompass()
                        ptgoal_gps_msg.distance_to_goal = distance_to_goal
                        ptgoal_gps_msg.angle_to_goal = angle_to_goal
                        ptgoal_gps_msg.header = h
                        # publish observations
                        self.pub_rgb.publish(rgb_msg_for_hab)
                        self.pub_depth.publish(depth_msg_for_hab)
                        self.pub_pointgoal_with_gps_compass.publish(ptgoal_gps_msg)
                        # block further sensor callbacks
                        self.last_action_done = False

                        # add the current position to the marker array for
                        # visualization
                        with self.curr_pose_lock:
                            if self.count_steps == 0:
                                self.logger.info(f"initial agent position: {self.curr_pos}")
                                self.add_pos_to_marker_array(
                                    "init",
                                    self.curr_pos
                                )
                            else:
                                self.add_pos_to_marker_array(
                                    "curr",
                                    self.prev_pos,
                                    self.curr_pos,
                                    self.curr_rotation
                                )
                    
                    elif(
                        self.last_action_done
                        and self.pointgoal_set
                        and (
                            self.pointgoal_reached
                            or self.count_steps >= self.max_steps
                        )
                    ):
                        # if the navigation is completed, publish data
                        # for visualization
                        self.logger.info(f"navigation completed after {self.count_steps} steps")
                        with self.curr_pose_lock:
                            self.logger.info(f"final agent position: {self.curr_pos}")
                            self.add_pos_to_marker_array(
                                "goal",
                                self.final_pointgoal_pos
                            )
                            self.publish_marker_array()
                        self.last_action_done = False
                    else:
                        # otherwise, drop the observations
                        return

    def callback_register_goal(self, goal_msg):
        r"""
        Register point-goal position w.r.t. the world frame.
        :param goal_msg: point-goal position from move_base
        """
        with self.final_pointgoal_lock:
            self.final_pointgoal_pos = np.array([
                goal_msg.pose.position.x,
                goal_msg.pose.position.y,
                goal_msg.pose.position.z
            ])
            self.pointgoal_set = True
            self.logger.info(f"Final pointgoal position set to: {self.final_pointgoal_pos}")

    def callback_signal_last_action(self, done_msg):
        r"""
        Signal the last action being done; toggle `pointgoal_reached`
        flag if the last action is `STOP`.
        """
        if done_msg.data == 1:
            # last action is `STOP`
            with self.pointgoal_reached_lock:
                self.pointgoal_reached = True
        
        with self.last_action_lock:
            # increment step counter and signal last action being done
            self.count_steps += 1
            self.last_action_done = True
    
    def get_agent_pose(self, request):
        r"""
        ROS service handler which returns the current pose of the agent.
        :param request: not used
        :returns: current pose of the agent
        """
        pose = Pose()
        with self.curr_pose_lock:
            pose.position.x = self.curr_pos[0]
            pose.position.y = self.curr_pos[1]
            pose.position.z = self.curr_pos[2]
            pose.orientation.x = self.curr_rotation[0]
            pose.orientation.y = self.curr_rotation[1]
            pose.orientation.z = self.curr_rotation[2]
            pose.orientation.w = self.curr_rotation[3]
        return pose
    
    def spin_until_shutdown(self):
        r"""
        Let the node spin until shutdown.
        """
        rospy.spin()


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node-name",
        default="gazebo_to_habitat_agent",
        type=str
    )
    parser.add_argument(
        "--gazebo-rgb-topic-name",
        default="camera/rgb/image_raw",
        type=str
    )
    parser.add_argument(
        "--gazebo-depth-topic-name",
        default="camera/depth/image_raw",
        type=str
    )
    parser.add_argument(
        "--gazebo-odom-topic-name",
        default="odom",
        type=str
    )
    parser.add_argument(
        "--move-base-goal-topic-name",
        default="/move_base_simple/goal",
        type=str
    )
    parser.add_argument(
        "--fetch-goal-from-move-base",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--pointgoal-location",
        nargs="+",
        type=float
    )
    args = parser.parse_args()

    # if the user is not providing pointgoal location, use the origin
    if args.pointgoal_location is None:
        pointgoal_list = [0.0, 0.0, 0.0]
    else:
        pointgoal_list = args.pointgoal_location
    
    # instantiate the bridge
    bridge = GazeboToHabitatAgent(
        node_name=args.node_name,
        gazebo_rgb_topic_name=args.gazebo_rgb_topic_name,
        gazebo_depth_topic_name=args.gazebo_depth_topic_name,
        gazebo_odom_topic_name=args.gazebo_odom_topic_name,
        move_base_goal_topic_name=args.move_base_goal_topic_name,
        fetch_goal_from_move_base=args.fetch_goal_from_move_base,
        final_pointgoal_pos=np.array(pointgoal_list),
    )

    # spins until receiving the shutdown signal
    bridge.spin_until_shutdown()


if __name__ == "__main__":
    main()