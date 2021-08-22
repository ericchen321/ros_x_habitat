import roslaunch
import rospy
import shlex
from geometry_msgs.msg import Twist
from subprocess import Popen
from ros_x_habitat.srv import Roam
from src.constants.constants import PACKAGE_NAME, ServiceNames


class JoyHabitatRoamer:
    r"""
    Class to allow free-roaming in a Habitat environment through joystick
    control.
    """

    def __init__(
        self,
        launch_file_path: str,
        hab_env_node_path: str,
        hab_env_config_path: str,
        hab_env_node_name: str,
        video_frame_period: int,
    ):
        r"""
        :param launch_file_path: path to the launch file which launches the
            joy controller node and some other nodes for control and visualization
        :param hab_env_node_path: path to the Habitat env node file
        :param hab_env_config_path: path to the Habitat env config file
        :param hab_env_node_name: name given to the Habitat env node
        :param video_frame_period: period at which to record a frame; measured in
            steps per frame
        """
        # create a node
        rospy.init_node("joy_habitat_roamer", anonymous=True)

        # start the launch file
        # code adapted from http://wiki.ros.org/roslaunch/API%20Usage
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(self.uuid, [launch_file_path])
        self.launch.start()
        rospy.loginfo(f"launch file {launch_file_path} started")

        # start the env node
        self.hab_env_node_name = hab_env_node_name
        env_node_args = shlex.split(
            f"python {hab_env_node_path} --node-name {self.hab_env_node_name} --task-config {hab_env_config_path} --enable-physics-sim --use-continuous-agent"
        )
        self.env_process = Popen(env_node_args)

        # set up roam service client
        self.roam_service_name = (
            f"{PACKAGE_NAME}/{self.hab_env_node_name}/{ServiceNames.ROAM}"
        )
        self.roam = rospy.ServiceProxy(self.roam_service_name, Roam)

        # register frame period
        self.video_frame_period = video_frame_period

    def roam_until_shutdown(self, episode_id_last: str = "-1", scene_id_last: str = ""):
        r"""
        Roam in a specified scene until shutdown signalled.
        :param episode_id_last: last episode's ID before the one to roam in.
        :param scene_id_last: last episode's scene ID before the one to
            roam in.
        """
        # code adapted from http://wiki.ros.org/roslaunch/API%20Usage
        try:
            rospy.wait_for_service(self.roam_service_name)
            try:
                resp = self.roam(
                    episode_id_last, scene_id_last, True, self.video_frame_period
                )
                assert resp.ack
            except rospy.ServiceException:
                rospy.logerr("Failed to initiate the roam service!")
            self.launch.spin()
        finally:
            # After Ctrl+C, stop all nodes from running
            self.env_process.kill()
            self.launch.shutdown()
