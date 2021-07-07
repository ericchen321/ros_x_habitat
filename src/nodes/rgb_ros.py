#!/usr/bin/env python
# note need to run viewer with python2!!!

import numpy as np
import rospy
import std_msgs.msg
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import Image

rospy.init_node("nprgb2ros_rgb", anonymous=False)

pub = rospy.Publisher("ros_img_rgb", Image, queue_size=10)


def callback(data):
    img_raveled = data.data[0:-2]
    img_size = data.data[-2:].astype(int)
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    img = (np.reshape(img_raveled, (img_size[0], img_size[1], 3))).astype(np.uint8)
    image_message = CvBridge().cv2_to_imgmsg(img, encoding="rgb8")
    image_message.header = h
    pub.publish(image_message)


def listener():

    rospy.Subscriber("rgb", numpy_msg(Floats), callback)
    rospy.spin()


if __name__ == "__main__":
    listener()
