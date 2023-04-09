#!/usr/bin/env python
import argparse
from threading import Condition, Lock

import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from habitat.config.default import get_config
from habitat.core.simulator import Observations
from ros_x_habitat.msg import PointGoalWithGPSCompass, DepthImage
from ros_x_habitat.srv import EvalEpisode, ResetAgent, GetAgentTime, Roam
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header, Int16
from src.constants.constants import (
    EvalEpisodeSpecialIDs,
    NumericalMetrics,
    PACKAGE_NAME,
    ServiceNames,
)
from src.envs.habitat_eval_rlenv import HabitatEvalRLEnv
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
import time
from src.utils import utils_logging
from src.utils.utils_visualization import generate_video, observations_to_image_for_roam
from src.measures.top_down_map_for_roam import (
    TopDownMapForRoam,
    add_top_down_map_for_roam_to_config,
)


class HabitatEnvNode:
    r"""
    A class to represent a ROS node with a Habitat simulator inside.
    The node subscribes to agent command topics, and publishes sensor
    readings to sensor topics.
    """

    def __init__(
        self,
        node_name: str,
        config_paths: str = None,
        enable_physics_sim: bool = False,
        use_continuous_agent: bool = False,
        pub_rate: float = 5.0,
    ):
        r"""
        Instantiates a node incapsulating a Habitat sim environment.
        :param node_name: name of the node
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

        # initialize node
        self.node_name = node_name
        rospy.init_node(self.node_name)

        rospy.on_shutdown(self.on_exit_generate_video)

        # set up environment config
        self.config = get_config(config_paths)
        # embed top-down map in config
        self.config.defrost()
        self.config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        self.config.freeze()
        add_top_down_map_for_roam_to_config(self.config)

        # instantiate environment
        self.enable_physics_sim = enable_physics_sim
        self.use_continuous_agent = use_continuous_agent
        # overwrite env config if physics enabled
        if self.enable_physics_sim:
            HabitatSimEvaluator.overwrite_simulator_config(self.config)
        # define environment
        self.env = HabitatEvalRLEnv(
            config=self.config, enable_physics=self.enable_physics_sim
        )

        # shutdown is set to true by eval_episode() to indicate the
        # evaluator wants the node to shutdown
        self.shutdown_lock = Lock()
        with self.shutdown_lock:
            self.shutdown = False

        # enable_eval is set to true by eval_episode() to allow
        # publish_sensor_observations() and step() to run
        # enable_eval is set to false in one of the three conditions:
        # 1) by publish_and_step_for_eval() after an episode is done;
        # 2) by publish_and_step_for_roam() after a roaming session
        #    is done;
        # 3) by main() after all episodes have been evaluated.
        # all_episodes_evaluated is set to True by main() to indicate
        # no more episodes left to evaluate. eval_episodes() then signals
        # back to evaluator, and set it to False again for re-use
        self.all_episodes_evaluated = False
        self.enable_eval = False
        self.enable_eval_cv = Condition()

        # enable_reset is set to true by eval_episode() or roam() to allow
        # reset() to run
        # enable_reset is set to false by reset() after simulator reset
        self.enable_reset_cv = Condition()
        with self.enable_reset_cv:
            self.enable_reset = False
            self.enable_roam = False
            self.episode_id_last = None
            self.scene_id_last = None

        # agent velocities/action and variables to keep things synchronized
        self.command_cv = Condition()
        with self.command_cv:
            if self.use_continuous_agent:
                self.linear_vel = None
                self.angular_vel = None
            else:
                self.action = None
            self.count_steps = None
            self.new_command_published = False

        self.observations = None

        # timing variables and guarding lock
        self.timing_lock = Lock()
        with self.timing_lock:
            self.t_reset_elapsed = None
            self.t_sim_elapsed = None

        # video production variables
        self.make_video = False
        self.observations_per_episode = []
        self.video_frame_counter = 0
        self.video_frame_period = 1  # NOTE: frame rate defined as x steps/frame

        # set up logger
        self.logger = utils_logging.setup_logger(self.node_name)

        # establish evaluation service server
        self.eval_service = rospy.Service(
            f"{PACKAGE_NAME}/{node_name}/{ServiceNames.EVAL_EPISODE}",
            EvalEpisode,
            self.eval_episode,
        )

        # establish roam service server
        self.roam_service = rospy.Service(
            f"{PACKAGE_NAME}/{node_name}/{ServiceNames.ROAM}", Roam, self.roam
        )

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
            if self.use_continuous_agent:
                # if we are using a ROS-based agent, we publish depth images
                # in type Image
                self.pub_depth = rospy.Publisher(
                    "depth", Image, queue_size=self.pub_queue_size
                )
                # also publish depth camera info
                self.pub_camera_info = rospy.Publisher(
                    "camera_info", CameraInfo, queue_size=self.pub_queue_size
                )
            else:
                # otherwise, we publish in type DepthImage to preserve as much
                # accuracy as possible
                self.pub_depth = rospy.Publisher(
                    "depth", DepthImage, queue_size=self.pub_queue_size
                )
        if "POINTGOAL_WITH_GPS_COMPASS_SENSOR" in self.config.TASK.SENSORS:
            self.pub_pointgoal_with_gps_compass = rospy.Publisher(
                "pointgoal_with_gps_compass",
                PointGoalWithGPSCompass,
                queue_size=self.pub_queue_size
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

            # disable reset
            self.enable_reset = False

            # if shutdown is signalled, return immediately
            with self.shutdown_lock:
                if self.shutdown:
                    return

            # locate the last episode specified
            if self.episode_id_last != EvalEpisodeSpecialIDs.REQUEST_NEXT:
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
                # evaluate from the next episode
                pass

            # initialize timing variables
            with self.timing_lock:
                self.t_reset_elapsed = 0.0
                self.t_sim_elapsed = 0.0

            # ------------ log reset time start ------------
            t_reset_start = time.clock()
            # --------------------------------------------

            # initialize observations
            self.observations = self.env.reset()

            # ------------  log reset time end  ------------
            t_reset_end = time.clock()
            with self.timing_lock:
                self.t_reset_elapsed += t_reset_end - t_reset_start
            # --------------------------------------------

            # initialize step counter
            with self.command_cv:
                self.count_steps = 0

    def _enable_reset(self, request, enable_roam):
        r"""
        Helper method to set self.episode_id_last, self.scene_id_last,
        enable reset and alert threads waiting for reset to be enabled.
        :param request: request dictionary, should contain field
            `episode_id_last` and `scene_id_last`.
        :param enable_roam: if should enable free-roam mode or not.
        """
        with self.enable_reset_cv:
            # unpack evaluator request
            self.episode_id_last = str(request.episode_id_last)
            self.scene_id_last = str(request.scene_id_last)

            # enable (env) reset
            assert self.enable_reset is False
            self.enable_reset = True
            self.enable_roam = enable_roam
            self.enable_reset_cv.notify()

    def _enable_evaluation(self):
        r"""
        Helper method to enable evaluation and alert threads waiting for evalu-
        ation to be enabled.
        """
        with self.enable_eval_cv:
            assert self.enable_eval is False
            self.enable_eval = True
            self.enable_eval_cv.notify()

    def eval_episode(self, request):
        r"""
        ROS service handler which evaluates one episode and returns evaluation
        metrics.
        :param request: evaluation parameters provided by evaluator, including
            last episode ID and last scene ID.
        :return: 1) episode ID and scene ID; 2) metrics including distance-to-
        goal, success and spl.
        """
        # make a response dict
        resp = {
            "episode_id": EvalEpisodeSpecialIDs.RESPONSE_NO_MORE_EPISODES,
            "scene_id": "",
            NumericalMetrics.DISTANCE_TO_GOAL: 0.0,
            NumericalMetrics.SUCCESS: 0.0,
            NumericalMetrics.SPL: 0.0,
            NumericalMetrics.NUM_STEPS: 0,
            NumericalMetrics.SIM_TIME: 0.0,
            NumericalMetrics.RESET_TIME: 0.0,
        }

        if str(request.episode_id_last) == EvalEpisodeSpecialIDs.REQUEST_SHUTDOWN:
            # if shutdown request, enable reset and return immediately
            with self.shutdown_lock:
                self.shutdown = True
            with self.enable_reset_cv:
                self.enable_reset = True
                self.enable_reset_cv.notify()
            return resp
        else:
            # if not shutting down, enable reset and evaluation
            self._enable_reset(request=request, enable_roam=False)

            # enable evaluation
            self._enable_evaluation()

            # wait for evaluation to be over
            with self.enable_eval_cv:
                while self.enable_eval is True:
                    self.enable_eval_cv.wait()

                if self.all_episodes_evaluated is False:
                    # collect episode info and metrics
                    resp = {
                        "episode_id": str(self.env._env.current_episode.episode_id),
                        "scene_id": str(self.env._env.current_episode.scene_id),
                    }
                    metrics = self.env._env.get_metrics()
                    metrics_dic = {
                        k: metrics[k]
                        for k in [
                            NumericalMetrics.DISTANCE_TO_GOAL,
                            NumericalMetrics.SUCCESS,
                            NumericalMetrics.SPL,
                        ]
                    }
                    with self.timing_lock:
                        with self.command_cv:
                            metrics_dic[NumericalMetrics.NUM_STEPS] = self.count_steps
                            metrics_dic[NumericalMetrics.SIM_TIME] = (
                                self.t_sim_elapsed / self.count_steps
                            )
                            metrics_dic[
                                NumericalMetrics.RESET_TIME
                            ] = self.t_reset_elapsed
                    resp.update(metrics_dic)
                else:
                    # no episode is evaluated. Toggle the flag so the env node
                    # can be reused
                    self.all_episodes_evaluated = False
                return resp

    def roam(self, request):
        r"""
        ROS service handler which allows an agent to roam freely within a scene,
        starting from the initial position of the specified episode.
        :param request: episode ID and scene ID.
        :return: acknowledge signal.
        """
        # if not shutting down, enable reset and evaluation
        self._enable_reset(request=request, enable_roam=True)

        # set video production flag
        self.make_video = request.make_video
        self.video_frame_period = request.video_frame_period

        # enable evaluation
        self._enable_evaluation()

        return True

    def cv2_to_depthmsg(self, depth_img: np.ndarray):
        r"""
        Converts a Habitat depth image to a ROS DepthImage message.
        :param depth_img: depth image as a numpy array
        :returns: a ROS Image message if using continuous agent; or
            a ROS DepthImage message if using discrete agent
        """
        if self.use_continuous_agent:
            # depth reading should be denormalized, so we get
            # readings in meters
            assert self.config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH is False
            depth_img_in_m = np.squeeze(depth_img, axis=2)
            depth_msg = CvBridge().cv2_to_imgmsg(
                depth_img_in_m.astype(np.float32), encoding="passthrough"
            )
        else:
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
                sensor_msg = CvBridge().cv2_to_imgmsg(
                    sensor_data.astype(np.uint8), encoding="rgb8"
                )
            elif sensor_uuid == "depth":
                sensor_msg = self.cv2_to_depthmsg(sensor_data)
            elif sensor_uuid == "pointgoal_with_gps_compass":
                sensor_msg = PointGoalWithGPSCompass()
                sensor_msg.distance_to_goal = sensor_data[0]
                sensor_msg.angle_to_goal = sensor_data[1]
            # add header to message, and add the message to observations_ros
            if sensor_uuid in ["rgb", "depth", "pointgoal_with_gps_compass"]:
                h = Header()
                h.stamp = t_curr
                sensor_msg.header = h
                observations_ros[sensor_uuid] = sensor_msg

        return observations_ros

    def publish_sensor_observations(self):
        r"""
        Waits until evaluation is enabled, then publishes current simulator
        sensor readings. Requires to be called 1) after simulator reset and
        2) when evaluation has been enabled.
        """
        # pack observations in ROS message
        observations_ros = self.obs_to_msgs(self.observations)
        for sensor_uuid, _ in self.observations.items():
            # we publish to each of RGB, Depth and Ptgoal/GPS+Compass sensor
            if sensor_uuid == "rgb":
                self.pub_rgb.publish(observations_ros["rgb"])
            elif sensor_uuid == "depth":
                self.pub_depth.publish(observations_ros["depth"])
                if self.use_continuous_agent:
                    self.pub_camera_info.publish(
                        self.make_depth_camera_info_msg(
                            observations_ros["depth"].header,
                            observations_ros["depth"].height,
                            observations_ros["depth"].width,
                        )
                    )
            elif sensor_uuid == "pointgoal_with_gps_compass":
                self.pub_pointgoal_with_gps_compass.publish(
                    observations_ros["pointgoal_with_gps_compass"]
                )

    def make_depth_camera_info_msg(self, header, height, width):
        r"""
        Create camera info message for depth camera.
        :param header: header to create the message
        :param height: height of depth image
        :param width: width of depth image
        :returns: camera info message of type CameraInfo.
        """
        # code modifed upon work by Bruce Cui
        camera_info_msg = CameraInfo()
        camera_info_msg.header = header
        fx, fy = width / 2, height / 2
        cx, cy = width / 2, height / 2

        camera_info_msg.width = width
        camera_info_msg.height = height
        camera_info_msg.distortion_model = "plumb_bob"
        camera_info_msg.K = np.float32([fx, 0, cx, 0, fy, cy, 0, 0, 1])
        camera_info_msg.D = np.float32([0, 0, 0, 0, 0])
        camera_info_msg.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        return camera_info_msg

    def step(self):
        r"""
        Enact a new command and update sensor observations.
        Requires 1) being called only when evaluation has been enabled and
        2) being called only from the main thread.
        """
        
        with self.command_cv:
            # wait for new action before stepping
            while self.new_command_published is False:
                self.command_cv.wait()
            self.new_command_published = False

            # enact the action / velocities
            # ------------ log sim time start ------------
            t_sim_start = time.clock()
            # --------------------------------------------

            if self.use_continuous_agent:
                self.env.set_agent_velocities(self.linear_vel, self.angular_vel)
                print(self.linear_vel)
                (self.observations, _, _, info) = self.env.step()
            else:
                # NOTE: Here we call HabitatEvalRLEnv.step() which dispatches
                # to Env.step() or PhysicsEnv.step_physics() depending on
                # whether physics has been enabled
                (self.observations, _, _, info) = self.env.step(self.action)

            # ------------  log sim time end  ------------
            t_sim_end = time.clock()
            with self.timing_lock:
                self.t_sim_elapsed += t_sim_end - t_sim_start
            # --------------------------------------------

        # if making video, generate frames from actions
        if self.make_video:
            self.video_frame_counter += 1
            if self.video_frame_counter == self.video_frame_period - 1:
                # NOTE: for now we only consider the case where we make videos
                # in the roam mode, for a continuous agent
                out_im_per_action = observations_to_image_for_roam(
                    self.observations,
                    info,
                    self.config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH,
                )
                self.observations_per_episode.append(out_im_per_action)
                self.video_frame_counter = 0

        with self.command_cv:
            self.count_steps += 1

    def publish_and_step_for_eval(self):
        r"""
        Complete an episode and alert eval_episode() upon completion. Requires
        to be called after simulator reset.
        """
        # publish observations at fixed rate
        r = rospy.Rate(self.pub_rate)
        with self.enable_eval_cv:
            # wait for evaluation to be enabled
            while self.enable_eval is False:
                self.enable_eval_cv.wait()

            # publish observations and step until the episode ends
            while not self.env._env.episode_over:
                self.publish_sensor_observations()
                self.step()
                r.sleep()

            # now the episode is done, disable evaluation and alert eval_episode()
            self.enable_eval = False
            self.enable_eval_cv.notify()

    def publish_and_step_for_roam(self):
        r"""
        Let an agent roam within a scene until shutdown. Requires to be called
        1) after simulator reset, 2) shutdown_lock has not yet been acquired by
        the current thread.
        """
        # publish observations at fixed rate
        r = rospy.Rate(self.pub_rate)
        with self.enable_eval_cv:
            # wait for evaluation to be enabled
            while self.enable_eval is False:
                self.enable_eval_cv.wait()

            # publish observations and step until shutdown
            while True:
                with self.shutdown_lock:
                    if self.shutdown:
                        break
                self.publish_sensor_observations()
                self.step()
                r.sleep()

            # disable evaluation
            self.enable_eval = False

    def callback(self, cmd_msg):
        r"""
        Takes in a command from an agent and alert the simulator to enact
        it.
        :param cmd_msg: Either a velocity command or an action command.
        """
        # unpack agent action from ROS message, and send the action
        # to the simulator
        with self.command_cv:
            if self.use_continuous_agent:
                # set linear + angular velocity
                self.linear_vel = np.array(
                    [(1.0 * cmd_msg.linear.y), 0.0, (-1.0 * cmd_msg.linear.x)]
                )
                self.angular_vel = np.array([0.0, cmd_msg.angular.z, 0.0])
            else:
                # get the action
                self.action = cmd_msg.data

            # set action publish flag and notify
            self.new_command_published = True
            self.command_cv.notify()

    def simulate(self):
        r"""
        An infinite loop where the env node 1) keeps evaluating the next
        episode in its RL environment, if an EvalEpisode request is given;
        or 2) let the agent roam freely in one episode.
        Breaks upon receiving shutdown command.
        """
        # iterate over episodes
        while True:
            try:
                # reset the env
                self.reset()
                with self.shutdown_lock:
                    # if shutdown service called, exit
                    if self.shutdown:
                        rospy.signal_shutdown("received request to shut down")
                        break
                with self.enable_reset_cv:
                    if self.enable_roam:
                        self.publish_and_step_for_roam()
                    else:
                        # otherwise, evaluate the episode
                        self.publish_and_step_for_eval()
            except StopIteration:
                # set enable_reset and enable_eval to False, so the
                # env node can evaluate again in the future
                with self.enable_reset_cv:
                    self.enable_reset = False
                with self.enable_eval_cv:
                    self.all_episodes_evaluated = True
                    self.env.reset_episode_iterator()
                    self.enable_eval = False
                    self.enable_eval_cv.notify()

    def on_exit_generate_video(self):
        r"""
        Make video of the current episode, if video production is turned
        on.
        """
        if self.make_video:
            generate_video(
                video_option=self.config.VIDEO_OPTION,
                video_dir=self.config.VIDEO_DIR,
                images=self.observations_per_episode,
                episode_id="fake_episode_id",
                scene_id="fake_scene_id",
                agent_seed=0,
                checkpoint_idx=0,
                metrics={},
                tb_writer=None,
            )


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-name", type=str, default="env_node")
    parser.add_argument(
        "--task-config", type=str, default="configs/pointnav_d_orignal.yaml"
    )
    parser.add_argument("--enable-physics-sim", default=False, action="store_true")
    parser.add_argument("--use-continuous-agent", default=False, action="store_true")
    parser.add_argument(
        "--sensor-pub-rate",
        type=float,
        default=20.0,
    )
    args = parser.parse_args()

    # initialize the env node
    env_node = HabitatEnvNode(
        node_name=args.node_name,
        config_paths=args.task_config,
        enable_physics_sim=args.enable_physics_sim,
        use_continuous_agent=args.use_continuous_agent,
        pub_rate=args.sensor_pub_rate,
    )

    # run simulations
    env_node.simulate()


if __name__ == "__main__":
    main()
