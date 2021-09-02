#!/usr/bin/env python
import argparse
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16
from geometry_msgs.msg import PoseStamped
import quaternion
from habitat.utils.geometry_utils import quaternion_rotate_vector
from threading import Lock
from habitat.sims.habitat_simulator.actions import _DefaultHabitatSimActions
from ros_x_habitat.srv import GetAgentPose
from src.constants.constants import PACKAGE_NAME, ServiceNames
from src.utils import utils_logging

class HabitatAgentToGazebo:
    r"""
    A class to represent a ROS node which subscribes from Habitat
    agent's action topic and converts the action to a sequence of
    velocities.
    """

    def __init__(
        self,
        node_name: str,
        control_period: float=1.0
    ):
        r"""
        Instantiates the Habitat agent->Gazebo bridge.
        :param node_name: name of the bridge node
        :param control_period: time it takes for a discrete action to complete,
            measured in seconds
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

        # register control period
        self.control_period = control_period

        # set up step counter
        self.step_lock = Lock()
        with self.step_lock:
            self.count_steps = 0

        # subscribe from Habitat-agent-facing action topic
        self.sub_action = rospy.Subscriber(
            "action", Int16, self.callback_action_from_agent, queue_size=self.sub_queue_size
        )

        # publish to Gazebo-facing velocity command topic
        self.pub_vel = rospy.Publisher("cmd_vel", Twist, queue_size=self.pub_queue_size)

        # publish to `last_action_done/`
        self.pub_last_action_done = rospy.Publisher(
            "last_action_done",
            Int16,
            queue_size=self.pub_queue_size
        )

        # establish get_agent_pose service client
        self.get_agent_pose_service_name = (
                f"{PACKAGE_NAME}/gazebo_to_habitat_agent/{ServiceNames.GET_AGENT_POSE}"
            )        
        self.get_agent_pose = rospy.ServiceProxy(
            self.get_agent_pose_service_name, GetAgentPose
        )

        self.logger.info("habitat agent -> gazebo bridge initialized")

    def create_vel_msg(
        self,
        linear_x,
        linear_y,
        linear_z,
        angular_x,
        angular_y,
        angular_z
    ):
        r"""
        Create a velocity message from the given linear and angular
        velocity components.
        :param linear_x: linear component x
        :param linear_y: linear component x
        :param linear_z: linear component z
        :param angular_x: angular component x
        :param angular_y: angular component y
        :param angular_z: angular component z
        :return: velocity message of type `Twist`
        """
        vel_msg = Twist()
        vel_msg.linear.x = linear_x
        vel_msg.linear.y = linear_y
        vel_msg.linear.z = linear_z
        vel_msg.angular.x = angular_x
        vel_msg.angular.y = angular_y
        vel_msg.angular.z = angular_z
        return vel_msg
    
    def callback_action_from_agent(self, action_msg):
        r"""
        Upon receiving a discrete action from a Habitat agent, converts
        the action to velocities and publishes to `cmd_vel/`. Signal the
        action being done on `last_action_done/` upon the action's
        completion.
        :param action_msg: A discrete action command
        """        
        # initialize done flag
        navigation_done = 0

        # convert action to Twist message and publish
        vel_msg = self.create_vel_msg(0, 0, 0, 0, 0, 0)
        action_id = action_msg.data
        if action_id == _DefaultHabitatSimActions.STOP.value:
            navigation_done = 1
        elif action_id == _DefaultHabitatSimActions.MOVE_FORWARD.value:
            linear_vel_local = np.array([-0.25 / self.control_period, 0, 0])
            # get the current pose of the agent
            rospy.wait_for_service(self.get_agent_pose_service_name)
            try:
                agent_pose = self.get_agent_pose()
                curr_rotation = [
                    agent_pose.pose.orientation.x,
                    agent_pose.pose.orientation.y,
                    agent_pose.pose.orientation.z,
                    agent_pose.pose.orientation.w
                ]
            except rospy.ServiceException:
                raise rospy.ServiceException
            # compute linear velocity in world frame
            linear_vel_world = quaternion_rotate_vector(
                np.quaternion(
                    curr_rotation[0],
                    curr_rotation[1],
                    curr_rotation[2],
                    curr_rotation[3]),
                linear_vel_local)
            vel_msg.linear.x = linear_vel_world[0]
            vel_msg.linear.y = linear_vel_world[1]
            vel_msg.linear.z = linear_vel_world[2]
        elif action_id == _DefaultHabitatSimActions.TURN_LEFT.value:
            vel_msg.angular.z = np.deg2rad(10.0)
        elif action_id == _DefaultHabitatSimActions.TURN_RIGHT.value:
            vel_msg.angular.z = np.deg2rad(-10.0)
        with self.step_lock:
            # increment by one step
            self.count_steps += 1
        self.pub_vel.publish(vel_msg)
        
        # actuate
        action_start_time = rospy.get_rostime().secs
        while rospy.get_rostime().secs < action_start_time + self.control_period:
            rospy.sleep(self.control_period/10.0)
        # issue a pseudo-stop to mitigate the delay from observation to actuation
        vel_msg = self.create_vel_msg(0, 0, 0, 0, 0, 0)
        self.pub_vel.publish(vel_msg)
        
        # signal the action being done
        navigation_done_msg = Int16()
        navigation_done_msg.data = navigation_done
        self.pub_last_action_done.publish(navigation_done_msg)

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
        default="habitat_agent_to_gazebo",
        type=str
    )
    args = parser.parse_args()

    # instantiate the bridge
    bridge = HabitatAgentToGazebo(
        node_name=args.node_name,
        control_period=2.0
    )

    # spins until receiving the shutdown signal
    bridge.spin_until_shutdown()


if __name__ == "__main__":
    main()