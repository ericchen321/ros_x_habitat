#!/usr/bin/env python
import argparse
import os
from math import radians
from threading import Condition
from threading import Lock
from typing import (
    Any,
    Dict,
)

import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from habitat.sims.habitat_simulator.actions import _DefaultHabitatSimActions
from message_filters import TimeSynchronizer
from ros_x_habitat.msg import PointGoalWithGPSCompass, DepthImage
from ros_x_habitat.srv import ResetAgent, GetAgentTime
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from src.constants.constants import AgentResetCommands, PACKAGE_NAME, ServiceNames
from src.test.data.data import TestHabitatROSData
from src.utils import utils_logging


class MockHabitatAgentNode:
    r"""
    A mock agent that checks if sensor readings from the environment
    node is correct, and publishes pre-recorded actions back to the
    environment.
    """

    def __init__(
        self,
        node_name: str,
        sensor_pub_rate: float = 5.0,
        use_habitat_physics_sim: bool = False,
    ):
        self.node_name = node_name
        rospy.init_node(self.node_name)

        self.sensor_pub_rate = float(sensor_pub_rate)

        self.sub_queue_size = 10
        self.pub_queue_size = 10

        self.use_habitat_physics_sim = use_habitat_physics_sim

        self.observations_count = 0

        # lock guarding access to self.count_frames and self.agent_time
        self.lock = Lock()
        with self.lock:
            self.agent_time = 0

        # shutdown triggers the node to be shutdown. Guarded by shutdown_cv
        self.shutdown_cv = Condition()
        with self.shutdown_cv:
            self.shutdown = False

        # set up logger
        self.logger = utils_logging.setup_logger(self.node_name)

        # establish agent reset service server
        self.reset_service = rospy.Service(
            f"{PACKAGE_NAME}/{self.node_name}/{ServiceNames.RESET_AGENT}",
            ResetAgent,
            self.reset_agent,
        )

        # establish agent time service server
        self.agent_time_service = rospy.Service(
            f"{PACKAGE_NAME}/{self.node_name}/{ServiceNames.GET_AGENT_TIME}",
            GetAgentTime,
            self.get_agent_time,
        )

        # publish to command topics
        self.pub = rospy.Publisher("action", Int16, queue_size=self.pub_queue_size)

        # subscribe to sensor topics
        self.sub_rgb = message_filters.Subscriber("rgb", Image)
        self.sub_depth = message_filters.Subscriber("depth", numpy_msg(DepthImage))
        self.sub_pointgoal_with_gps_compass = message_filters.Subscriber(
            "pointgoal_with_gps_compass", PointGoalWithGPSCompass
        )

        # filter sensor topics with time synchronizer
        self.ts = TimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_pointgoal_with_gps_compass],
            queue_size=self.sub_queue_size,
        )
        self.ts.registerCallback(self.callback_rgbd)

        self.logger.info("agent making sure env subscribed to command topic...")
        while self.pub.get_num_connections() == 0:
            pass

        self.logger.info("mock agent initialized")

    def reset_agent(self, request):
        r"""
        ROS service handler which resets the agent and related variables.
        :param request: command from the env node.
        :returns: True
        """
        if request.reset == AgentResetCommands.RESET:
            # fake-reset the agent
            with self.lock:
                self.agent_time = 0.0
            return True
        elif request.reset == AgentResetCommands.SHUTDOWN:
            # shut down the agent node
            with self.shutdown_cv:
                self.shutdown = True
                self.shutdown_cv.notify()
                return True

    def get_agent_time(self, request):
        r"""
        ROS service handler which returns the time that the agent takes
        to produce the last action.
        :param request: not used
        :returns: agent time
        """
        return self.agent_time

    def depthmsg_to_cv2(self, depth_msg):
        r"""
        Converts a ROS DepthImage message to a Habitat depth observation.
        :param depth_msg: ROS depth message
        :returns: depth observation as a numpy array
        """
        w = depth_msg.width
        h = depth_msg.height
        depth_img = np.reshape(depth_msg.data.astype(np.float32), (h, w))
        return depth_img

    def msgs_to_obs(
        self,
        rgb_msg: Image = None,
        depth_msg: DepthImage = None,
        pointgoal_with_gps_compass_msg: PointGoalWithGPSCompass = None,
    ) -> Dict[str, Any]:
        r"""
        Converts ROS messages into Habitat observations.
        :param rgb_msg: RGB sensor observations packed in a ROS message
        :param depth_msg: Depth sensor observations packed in a ROS message
        :param pointgoal_with_gps_compass_msg: Pointgoal + GPS/Compass sensor
            observations packed in a ROS message
        :return: Habitat observations
        """
        observations = {}

        # Convert RGB message
        if rgb_msg is not None:
            observations["rgb"] = (
                CvBridge().imgmsg_to_cv2(rgb_msg, "passthrough").astype(np.float32)
            )

        # Convert depth message
        if depth_msg is not None:
            observations["depth"] = self.depthmsg_to_cv2(depth_msg)
            # have to manually add channel info
            observations["depth"] = np.expand_dims(observations["depth"], 2).astype(
                np.float32
            )

        # Convert pointgoal + GPS/compass sensor message
        if pointgoal_with_gps_compass_msg is not None:
            observations["pointgoal_with_gps_compass"] = np.asarray(
                [
                    pointgoal_with_gps_compass_msg.distance_to_goal,
                    pointgoal_with_gps_compass_msg.angle_to_goal,
                ]
            ).astype(np.float32)

        return observations

    def action_to_msg(self, action: Dict[str, int]):
        r"""
        Converts action produced by Habitat agent to a ROS message.
        :param action: Discrete action produced by Habitat agent.
        :returns: A ROS message of action command.
        """
        action_id = action["action"]
        # produce action message
        msg = Int16()
        msg.data = action_id

        return msg

    def callback_rgbd(self, rgb_msg, depth_msg, pointgoal_with_gps_compass_msg):
        r"""
        Checks if simulator readings are correct;
        Produces an action or velocity command periodically from RGBD
        sensor observations, and publish to respective topics.
        :param rgb_msg: RGB sensor readings in ROS message format.
        :param depth_msg: Depth sensor readings in ROS message format.
        :param pointgoal_with_gps_compass_msg: Pointgoal + GPS/Compass readings.
        """
        # convert current_observations from ROS to Habitat format
        observations = self.msgs_to_obs(
            rgb_msg=rgb_msg,
            depth_msg=depth_msg,
            pointgoal_with_gps_compass_msg=pointgoal_with_gps_compass_msg,
        )

        with self.lock:
            # using Habitat sim
            if self.use_habitat_physics_sim:
                # check sensor observations' correctness - continuous case
                assert (
                    self.observations_count
                    < TestHabitatROSData.test_acts_and_obs_continuous_num_obs
                )
                assert (
                    np.linalg.norm(
                        observations["rgb"]
                        - TestHabitatROSData.test_acts_and_obs_continuous_obs_rgb[
                            self.observations_count
                        ]
                    )
                    < 1e-5
                ), f"RGB reading at step {self.observations_count} does not match"
                assert (
                    np.linalg.norm(
                        observations["depth"]
                        - TestHabitatROSData.test_acts_and_obs_continuous_obs_depth[
                            self.observations_count
                        ]
                    )
                    < 1e-5
                ), f"Depth reading at step {self.observations_count} does not match"
                assert (
                    np.linalg.norm(
                        observations["pointgoal_with_gps_compass"]
                        - TestHabitatROSData.test_acts_and_obs_continuous_obs_ptgoal_with_comp[
                            self.observations_count
                        ]
                    )
                    < 1e-5
                ), f"Pointgoal + GPS/Compass reading at step {self.observations_count} does not match"
                # produce the saved continuous action at this step
                action = {
                    "action": TestHabitatROSData.test_acts_and_obs_continuous_acts[
                        self.observations_count
                    ]
                }

            else:
                # check sensor observations' correctness - discrete case
                assert (
                    self.observations_count
                    < TestHabitatROSData.test_acts_and_obs_discrete_num_obs
                )
                assert (
                    np.linalg.norm(
                        observations["rgb"]
                        - TestHabitatROSData.test_acts_and_obs_discrete_obs_rgb[
                            self.observations_count
                        ]
                    )
                    < 1e-5
                ), f"RGB reading at step {self.observations_count} does not match"
                assert (
                    np.linalg.norm(
                        observations["depth"]
                        - TestHabitatROSData.test_acts_and_obs_discrete_obs_depth[
                            self.observations_count
                        ]
                    )
                    < 1e-5
                ), f"Depth reading at step {self.observations_count} does not match"
                assert (
                    np.linalg.norm(
                        observations["pointgoal_with_gps_compass"]
                        - TestHabitatROSData.test_acts_and_obs_discrete_obs_ptgoal_with_comp[
                            self.observations_count
                        ]
                    )
                    < 1e-5
                ), f"Pointgoal + GPS/Compass reading at step {self.observations_count} does not match"
                # produce the saved discrete action at this step
                action = {
                    "action": TestHabitatROSData.test_acts_and_obs_discrete_acts[
                        self.observations_count
                    ]
                }

                # publish a discrete action
                action_msg = self.action_to_msg(action)
                self.pub.publish(action_msg)
                self.observations_count += 1

    def spin_until_shutdown(self):
        # shutdown the mock agent node after getting the signal
        with self.shutdown_cv:
            while self.shutdown is False:
                self.shutdown_cv.wait()
            rospy.signal_shutdown("received request to shut down")


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sensor-pub-rate",
        type=float,
        default=10,
    )
    parser.add_argument("--use-habitat-physics-sim", default=False, action="store_true")
    args = parser.parse_args()

    # init mock agent node
    mock_agent_node = MockHabitatAgentNode(
        node_name="mock_agent_node",
        sensor_pub_rate=args.sensor_pub_rate,
        use_habitat_physics_sim=args.use_habitat_physics_sim,
    )

    mock_agent_node.spin_until_shutdown()


if __name__ == "__main__":
    main()
