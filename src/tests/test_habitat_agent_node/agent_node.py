import rospy
import argparse
from src.classes.habitat_agent_node import get_default_config, HabitatAgentNode


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-physics', default=False, action='store_true')
    args = parser.parse_args()

    # set up test agent
    agent_config = get_default_config()
    agent_config.INPUT_TYPE = "rgbd"
    agent_config.MODEL_PATH = "/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/data/checkpoints/v2/gibson-rgbd-best.pth"

    # set up agent node under test
    rospy.init_node("agent_node")
    HabitatAgentNode(agent_config=agent_config, enable_physics=args.enable_physics, sensor_pub_rate=10)
    rospy.spin()
