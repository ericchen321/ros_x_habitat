#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)


def callback(data):
    # to speed up
    vel_linear_scale_factor = 2.0
    vel_angular_scale_factor = 2.0

    # negative sign in vel_z because agent eyes look at negative z axis
    vel_max = 0.3  # m/s
    vel_z = 4 * data.axes[1] * vel_max

    # negative sign because pushing right produces negative number on joystick
    vel_x = -4 * data.axes[0] * vel_max
    yaw = data.axes[3] * 30 / 180 * 3.1415926
    pitch = data.axes[4] * 30

    # h = std_msgs.msg.Header()
    # h.stamp = rospy.Time.now()
    vel_msg = Twist()
    vel_msg.linear.x = vel_linear_scale_factor * vel_z
    vel_msg.linear.y = vel_linear_scale_factor * vel_x
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = vel_angular_scale_factor * pitch
    vel_msg.angular.z = vel_angular_scale_factor * yaw
    # vel_msg.header = h

    pub.publish(vel_msg)


def start():
    rospy.init_node("Joy2habitat")
    rospy.Subscriber("joy", Joy, callback)
    rospy.spin()


if __name__ == "__main__":
    start()
