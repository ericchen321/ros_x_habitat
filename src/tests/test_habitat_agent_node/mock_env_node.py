import numpy as np
import rospy
import argparse
from geometry_msgs.msg import Twist
from rospy.numpy_msg import numpy_msg
from ros_x_habitat.msg import PointGoalWithGPSCompass, DepthImage
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int16
from cv_bridge import CvBridge, CvBridgeError
from threading import Condition


# load sensor readings and actions from disk
num_readings = 27
readings_rgb_discrete = []
readings_depth_discrete = []
readings_ptgoal_with_comp_discrete = []
actions_discrete = []
for i in range(0, num_readings):
    readings_rgb_discrete.append(np.load(f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/tests/obs/rgb-{i}.npy"))
    readings_depth_discrete.append(np.load(f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/tests/obs/depth-{i}.npy"))
    readings_ptgoal_with_comp_discrete.append(np.load(f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/tests/obs/pointgoal_with_gps_compass-{i}.npy"))
    actions_discrete.append(np.load(f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/tests/acts/action-{i}.npy"))


class MockHabitatEnvNode:
    def __init__(self, enable_physics=True):
        r"""
        A simple node to mock the Habitat environment node. Publishes pre-recorded
        sensor values, and checks if the agent node produces correct actions.
        """
        self.enable_physics = enable_physics
        self.action_count = 0

        self.pub_queue_size = 10
        self.sub_queue_size = 10

        # agent action and variables to keep things synchronized
        # NOTE: self.action_cv's lock guards self.action, self.new_action_published,
        # self.action_count, and pseudo-guards the sensor reading arrays
        self.action = None
        self.new_action_published = False
        self.action_cv = Condition()

        # publish to sensor topics
        self.pub_rgb = rospy.Publisher("rgb", Image, queue_size=self.pub_queue_size)
        self.pub_depth = rospy.Publisher("depth", numpy_msg(DepthImage), queue_size=self.pub_queue_size)
        self.pub_pointgoal_with_gps_compass = rospy.Publisher("pointgoal_with_gps_compass", PointGoalWithGPSCompass, queue_size=self.pub_queue_size)

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

    def publish_sensor_observations(self, i):
        r"""
        Publishes hard-coded sensor readings.
        :param i: index of the reading.
        """
        with self.action_cv:
            t_curr = rospy.Time.now()
            # rgb
            rgb_msg = CvBridge().cv2_to_imgmsg(readings_rgb_discrete[i].astype(np.uint8), encoding="passthrough")
            h = Header()
            h.stamp = t_curr
            rgb_msg.header = h
            # depth
            depth_msg = self.cv2_to_depthmsg(readings_depth_discrete[i])
            h = Header()
            h.stamp = t_curr
            depth_msg.header = h
            # pointgoal + gps/compass
            ptgoal_with_gps_compass_msg = PointGoalWithGPSCompass()
            ptgoal_with_gps_compass_msg.distance_to_goal = readings_ptgoal_with_comp_discrete[i][0]
            ptgoal_with_gps_compass_msg.angle_to_goal = readings_ptgoal_with_comp_discrete[i][1]
            h = Header()
            h.stamp = t_curr
            ptgoal_with_gps_compass_msg.header = h
            # publish
            self.pub_rgb.publish(rgb_msg)
            self.pub_depth.publish(depth_msg)
            self.pub_pointgoal_with_gps_compass.publish(ptgoal_with_gps_compass_msg)
            print(f"published observation {i}")
        
    def check_command(self):
        r"""
        Checks if agent's command is correct. To the variables
        guarded by self.action_cv, this method works similarly
        to HabitatEnvNode.step().
        """
        with self.action_cv:
            while self.new_action_published is False:
                self.action_cv.wait()
            self.new_action_published = False
            assert self.action == actions_discrete[self.action_count], f"wrong action: step={self.action_count}, expected action ID={actions_discrete[self.action_count]}, received action ID={self.action}"
            self.action_count = self.action_count + 1

    def callback(self, cmd_msg):
        r"""
        Reads a command from agent and check if it is correct.
        """
        if self.enable_physics is True:
            # TODO: do some checking
            pass 
        else:
            with self.action_cv:
                self.action = cmd_msg.data
                self.new_action_published = True
                self.action_cv.notify()


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-physics', default=False, action='store_true')
    args = parser.parse_args()

    rospy.init_node("mock_env_node")
    mock_env_node = MockHabitatEnvNode(enable_physics=args.enable_physics)
    
    # publish pre-saved sensor observations
    r = rospy.Rate(10)
    for i in range(0, num_readings):
        mock_env_node.publish_sensor_observations(i)
        mock_env_node.check_command()
        r.sleep()
    rospy.spin()