#!/usr/bin/env python
import argparse
from threading import Condition

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from habitat.config.default import get_config
from habitat.core.simulator import Observations
from ros_x_habitat.msg import PointGoalWithGPSCompass, DepthImage
from ros_x_habitat.srv import EvalEpisode, ResetAgent, GetAgentTime
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int16
from src.constants.constants import AgentResetCommands, NumericalMetrics
from src.envs.habitat_eval_rlenv import HabitatEvalRLEnv
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
import time
from src.utils import utils_logging


class HabitatEnvNode:
    r"""
    A class to represent a ROS node with a Habitat simulator inside.
    The node subscribes to agent command topics, and publishes sensor
    readings to sensor topics.
    """

    def __init__(
        self,
        config_paths: str = None,
        enable_physics_sim: bool = False,
        use_continuous_agent: bool = False,
        pub_rate: float = 5.0,
    ):
        r"""
        Instantiates a node incapsulating a Habitat sim environment.
        :param config_paths: path to Habitat env config file
        :param enable_physics_sim: if true, turn on dynamic simulation
            with Bullet
        :param use_continuous_agent: if true, the agent would be one
            that produces continuous velocities. Must be false if using
            discrete simulator
        :pub_rate: the rate at which the node publishes sensor readings
        """
        # precondition check
        if use_continuous_agent:
            assert enable_physics_sim

        # set up logger
        self.logger = utils_logging.setup_logger("env_node")

        # declare timing variables
        self.t_reset_elapsed = None
        self.t_sim_elapsed = None
        self.t_agent_elapsed = None

        # initialize Habitat environment
        self.config = get_config(config_paths)
        # embed top-down map and heading sensor in config
        self.config.defrost()
        self.config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        self.config.freeze()
        self.enable_physics_sim = enable_physics_sim
        self.use_continuous_agent = use_continuous_agent
        # overwrite env config if physics enabled
        if self.enable_physics_sim:
            HabitatSimEvaluator.overwrite_simulator_config(self.config)
        # define environment
        self.env = HabitatEvalRLEnv(
            config=self.config, enable_physics=self.enable_physics_sim
        )

        # enable_eval is set to true by eval_episode() to allow
        # publish_sensor_observations() and step() to run
        # enable_eval is set to false in one of the two conditions:
        # 1) by publish_and_step() after an episode is done;
        # 2) by main() after all episodes have been evaluated.
        # all_episodes_evaluated is set to True by main() to indicate
        # no more episodes left to evaluate. eval_episodes() then signals
        # back to evaluator
        self.all_episodes_evaluated = False
        self.enable_eval = False
        self.enable_eval_cv = Condition()

        # enable_reset is set to true by eval_episode() to allow
        # reset() to run
        # enable_reset is set to false by reset() after simulator reset
        self.enable_reset = False
        self.episode_id_last = None
        self.scene_id_last = None
        self.enable_reset_cv = Condition()
        self.eval_service = rospy.Service(
            "eval_episode", EvalEpisode, self.eval_episode
        )

        # agent action and variables to keep things synchronized
        self.action = None
        self.observations = None
        self.step_counter = 0
        self.new_action_published = False
        self.action_cv = Condition()

        # establish reset service with agent
        self.reset_agent = rospy.ServiceProxy("reset_agent", ResetAgent)

        # establish agent time service with agent
        self.get_agent_time = rospy.ServiceProxy("get_agent_time", GetAgentTime)

        # define the max rate at which we publish sensor readings
        self.pub_rate = float(pub_rate)

        # environment publish and subscribe queue size
        # TODO: make them configurable by constructor argument
        self.sub_queue_size = 10
        self.pub_queue_size = 10

        # publish to sensor topics
        # we create one topic for each of RGB, Depth and GPS+Compass
        # sensor
        if "RGB_SENSOR" in self.config.SIMULATOR.AGENT_0.SENSORS:
            self.pub_rgb = rospy.Publisher("rgb", Image, queue_size=self.pub_queue_size)
        if "DEPTH_SENSOR" in self.config.SIMULATOR.AGENT_0.SENSORS:
            self.pub_depth = rospy.Publisher(
                "depth", numpy_msg(DepthImage), queue_size=self.pub_queue_size
            )
        if "POINTGOAL_WITH_GPS_COMPASS_SENSOR" in self.config.TASK.SENSORS:
            self.pub_pointgoal_with_gps_compass = rospy.Publisher(
                "pointgoal_with_gps_compass", PointGoalWithGPSCompass, queue_size=10
            )

        # subscribe from command topics
        if self.use_continuous_agent:
            self.sub = rospy.Subscriber(
                "cmd_vel", Twist, self.callback, queue_size=self.sub_queue_size
            )
        else:
            self.sub = rospy.Subscriber(
                "action", Int16, self.callback, queue_size=self.sub_queue_size
            )

        # wait until connections with the agent is established
        self.logger.info("env making sure agent is subscribed to sensor topics...")
        while (
            self.pub_rgb.get_num_connections() == 0
            or self.pub_depth.get_num_connections() == 0
            or self.pub_pointgoal_with_gps_compass.get_num_connections() == 0
        ):
            pass

        self.logger.info("env initialized")

    def reset(self):
        r"""
        Resets the agent and the simulator. Requires being called only from
        the main thread.
        """
        # reset the simulator
        with self.enable_reset_cv:
            while self.enable_reset is False:
                self.enable_reset_cv.wait()

            # locate the last episode specified
            if self.episode_id_last != "-1":
                # iterate to the last episode. If not found, the loop exits upon a
                # StopIteration exception
                last_ep_found = False
                while not last_ep_found:
                    try:
                        self.env.reset()
                        e = self.env._env.current_episode
                        if (str(e.episode_id) == str(self.episode_id_last)) and (
                            e.scene_id == self.scene_id_last
                        ):
                            self.logger.info(
                                f"Last episode found: episode-id={self.episode_id_last}, scene-id={self.scene_id_last}"
                            )
                            last_ep_found = True
                    except StopIteration:
                        self.logger.info("Last episode not found!")
                        raise StopIteration
            else:
                # self.logger.info(f"No last episode specified. Proceed to evaluate from the next one")
                pass

            # initialize observations and step counter
            with self.action_cv:
                # reset timing variables
                self.t_reset_elapsed = 0.0
                self.t_sim_elapsed = 0.0
                self.t_agent_elapsed = 0.0

                # ------------ log reset time start ------------
                t_reset_start = time.clock()
                # --------------------------------------------

                self.observations = self.env.reset()

                # ------------  log reset time end  ------------
                t_reset_end = time.clock()
                self.t_reset_elapsed += t_reset_end - t_reset_start
                # -------------------------------------------- 

                self.count_steps = 0

            # reset agent
            rospy.wait_for_service("reset_agent")
            try:
                resp = self.reset_agent(int(AgentResetCommands.RESET))
                assert resp.done
            except rospy.ServiceException:
                self.logger.info("Failed to reset agent!")

            self.enable_reset = False

    def eval_episode(self, request):
        r"""
        ROS service handler which evaluates one episode and returns evaluation
        metrics.
        :param request: evaluation parameters provided by evaluator, including
            last episode ID and last scene ID.
        :return: 1) episode ID and scene ID; 2) metrics including distance-to-
        goal, success and spl.
        """
        with self.enable_reset_cv:
            # unpack evaluator request
            self.episode_id_last = str(request.episode_id_last)
            self.scene_id_last = str(request.scene_id_last)

            # enable (env) reset
            assert self.enable_reset is False
            self.enable_reset = True
            self.enable_reset_cv.notify()

        # enable evaluation
        with self.enable_eval_cv:
            assert self.enable_eval is False
            self.enable_eval = True
            self.enable_eval_cv.notify()

        # wait for evaluation to be over
        with self.enable_eval_cv:
            while self.enable_eval is True:
                self.enable_eval_cv.wait()

            resp = None
            if self.all_episodes_evaluated is False:
                # collect episode info and metrics
                resp = {
                    "episode_id": str(self.env._env.current_episode.episode_id),
                    "scene_id": str(self.env._env.current_episode.scene_id),
                }
                metrics = self.env._env.get_metrics()
                metrics_dic = {
                    k: metrics[k] for k in [NumericalMetrics.DISTANCE_TO_GOAL, NumericalMetrics.SUCCESS, NumericalMetrics.SPL]
                }
                metrics_dic[NumericalMetrics.NUM_STEPS] = self.count_steps
                metrics_dic[NumericalMetrics.AGENT_TIME] = self.t_agent_elapsed / self.count_steps
                metrics_dic[NumericalMetrics.SIM_TIME] = self.t_sim_elapsed / self.count_steps
                metrics_dic[NumericalMetrics.RESET_TIME] = self.t_reset_elapsed
                resp.update(metrics_dic)
            else:
                # signal that no episode has been evaluated
                resp = {
                    "episode_id": "-1",
                    "scene_id": "-1",
                    NumericalMetrics.DISTANCE_TO_GOAL: 0.0,
                    NumericalMetrics.SUCCESS: 0.0,
                    NumericalMetrics.SPL: 0.0,
                    NumericalMetrics.NUM_STEPS: 0,
                    NumericalMetrics.AGENT_TIME: 0.0,
                    NumericalMetrics.SIM_TIME: 0.0,
                    NumericalMetrics.RESET_TIME: 0.0
                }
            return resp

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
                rgb_msg = CvBridge().cv2_to_imgmsg(
                    sensor_data.astype(np.uint8), encoding="passthrough"
                )
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
        Waits until evaluation is enabled, then publishes current simulator
        sensor readings. Requires to be called after simulator reset.
        """
        with self.enable_eval_cv:
            # wait for evaluation to be enabled
            while self.enable_eval is False:
                self.enable_eval_cv.wait()
            # publish sensor readings
            with self.action_cv:
                # pack observations in ROS message
                observations_ros = self.obs_to_msgs(self.observations)
                for sensor_uuid, _ in self.observations.items():
                    # we publish to each of RGB, Depth and Ptgoal/GPS+Compass sensor
                    if sensor_uuid == "rgb":
                        self.pub_rgb.publish(observations_ros["rgb"])
                    elif sensor_uuid == "depth":
                        self.pub_depth.publish(observations_ros["depth"])
                    elif sensor_uuid == "pointgoal_with_gps_compass":
                        self.pub_pointgoal_with_gps_compass.publish(
                            observations_ros["pointgoal_with_gps_compass"]
                        )

    def step(self):
        r"""
        Enact a new command and update sensor observations.
        Requires 1) being called only when evaluation has been enabled and
        2) being called only from the main thread.
        """
        with self.action_cv:
            # wait for new action before stepping
            while self.new_action_published is False:
                self.action_cv.wait()
            self.new_action_published = False
            # enact the action. 
            # NOTE: Here we call HabitatEvalRLEnv.step() which dispatches
            # to Env.step() or PhysicsEnv.step_physics() depending on
            # whether physics has been enabled

            # ------------ log sim time start ------------
            t_sim_start = time.clock()
            # --------------------------------------------

            (self.observations, _, _, _) = self.env.step(self.action)

            # ------------  log sim time end  ------------
            t_sim_end = time.clock()
            self.t_sim_elapsed += t_sim_end - t_sim_start
            # --------------------------------------------

            self.count_steps += 1

    def publish_and_step(self):
        r"""
        Complete an episode and alert eval_episode() upon completion. Requires
        to be called after simulator reset.
        """
        # publish observations at fixed rate
        r = rospy.Rate(self.pub_rate)
        while not self.env._env.episode_over:
            self.publish_sensor_observations()
            self.step()
            r.sleep()
        # now the episode is done, disable evaluation and alert eval_episode()
        with self.enable_eval_cv:
            assert self.enable_eval is True
            self.enable_eval = False
            self.enable_eval_cv.notify()

    def callback(self, cmd_msg):
        r"""
        Takes in a command from an agent and alert the simulator to enact
        it.
        :param cmd_msg: Either a velocity command or an action command.
        """
        # unpack agent action from ROS message, and send the action
        # to the simulator
        if self.use_continuous_agent:
            # TODO: possibly do nothing?
            pass
        else:
            with self.action_cv:
                # get the action
                self.action = cmd_msg.data
                # get the agent time of the current episode
                rospy.wait_for_service("get_agent_time")
                try:
                    agent_time_resp = self.get_agent_time()
                    self.t_agent_elapsed += agent_time_resp.agent_time
                except rospy.ServiceException:
                    self.logger.info("Failed to get agent time!")
                # set action publish flag and notify
                self.new_action_published = True
                self.action_cv.notify()


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config", type=str, default="configs/pointnav_d_orignal.yaml"
    )
    parser.add_argument("--enable-physics-sim", default=False, action="store_true")
    parser.add_argument("--use-continuous-agent", default=False, action="store_true")
    parser.add_argument(
        "--sensor-pub-rate",
        type=float,
        default=5.0,
    )
    parser.add_argument("--episode-id", type=str, default="-1")
    parser.add_argument(
        "--scene-id",
        type=str,
        default="/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    )
    args = parser.parse_args()

    # initialize the env node
    rospy.init_node("env_node")
    env_node = HabitatEnvNode(
        config_paths=args.task_config,
        enable_physics_sim=args.enable_physics_sim,
        use_continuous_agent=args.use_continuous_agent,
        pub_rate=args.sensor_pub_rate,
    )

    # iterate over episodes
    while not rospy.is_shutdown():
        try:
            env_node.reset()
            env_node.publish_and_step()
        except StopIteration:
            with env_node.enable_eval_cv:
                env_node.all_episodes_evaluated = True
                env_node.enable_eval = False
                env_node.enable_eval_cv.notify()
            # request agent to shut down
            rospy.wait_for_service("reset_agent")
            try:
                resp = env_node.reset_agent(int(AgentResetCommands.SHUTDOWN))
                assert resp.done
            except rospy.ServiceException:
                env_node.logger.info("Failed to shut down agent!")
            # shut down the env node
            rospy.signal_shutdown("no episodes to evaluate")
            break
