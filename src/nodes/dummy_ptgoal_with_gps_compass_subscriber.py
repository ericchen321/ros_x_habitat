#!/usr/bin/env python
import rospy
from ros_x_habitat.msg import PointGoalWithGPSCompass


def callback(data):
    # rospy.loginfo(f"dist_to_goal: {data.distance_to_goal}, angle_to_goal: {data.angle_to_goal}")
    pass


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node("ptgoal_with_gps_compass_dummy_subscriber", anonymous=True)

    rospy.Subscriber("pointgoal_with_gps_compass", PointGoalWithGPSCompass, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == "__main__":
    listener()
