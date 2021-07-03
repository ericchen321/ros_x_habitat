#!/usr/bin/env python3
# runs Setting 2 (Habitat agent + Habitat Sim, no ROS) experiments under 10 random
# seeds
import argparse
import subprocess


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="blind",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", default="", type=str)
    parser.add_argument(
        "--task-config", type=str, default="configs/pointnav_d_orignal.yaml"
    )
    parser.add_argument("--episode-id", type=str, default="-1")
    parser.add_argument(
        "--scene-id",
        type=str,
        default="data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    )
    parser.add_argument("--make-videos", default=False, action="store_true")
    args = parser.parse_args()

    subprocess.run("")


if __name__ == "__main__":
    main()