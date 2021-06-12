import rospy
import argparse
from src.classes.habitat_env_node import HabitatEnvNode


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-physics", default=False, action="store_true")
    args = parser.parse_args()

    # set up env node under test
    rospy.init_node("env_node")
    env_node = HabitatEnvNode(
        config_paths="/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/configs/pointnav_rgbd_val.yaml",
        enable_physics=args.enable_physics,
        episode_id_last="48",
        scene_id_last="data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
    )

    # publish observations at fixed rate
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        env_node.publish_sensor_observations()
        env_node.step()
        r.sleep()
