#!/usr/bin/env python3
# runs Setting 2 (Habitat agent + Habitat Sim, no ROS) experiments under 10 random
# seeds
import argparse
import subprocess
import csv
import shlex


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
    parser.add_argument(
        "--seed-file-path", type=str, default="random_seeds.csv"
    )
    parser.add_argument("--log-dir", type=str, default="logs/")
    parser.add_argument("--make-videos", default=False, action="store_true")
    parser.add_argument("--video-dir", type=str, default="videos/")
    parser.add_argument("--tb-dir", type=str, default="tb/")
    parser.add_argument("--make-maps", default=False, action="store_true")
    parser.add_argument("--map-dir", type=str, default="maps/")
    args = parser.parse_args()

    seeds = []
    with open(args.seed_file_path, newline='') as csv_file:
        csv_lines = csv.reader(csv_file)
        for line in csv_lines:
            seeds.append(int(line[0]))

    for seed in seeds:
        # create directory to store per-episode logs, tb files and videos
        create_log_dir_args = shlex.split(f"mkdir -p {args.log_dir}/seed={seed}/")
        subprocess.run(create_log_dir_args)
        create_tb_dir_args = shlex.split(f"mkdir -p {args.tb_dir}/seed={seed}/")
        subprocess.run(create_tb_dir_args)
        if args.make_videos:
            create_video_dir_args = shlex.split(f"mkdir -p {args.video_dir}/seed={seed}/")
            subprocess.run(create_video_dir_args)

        # create summary file and evaluate
        summary_file = open(f"{args.log_dir}/summary_seed={seed}.log", "w")
        eval_command = f"python eval_habitat.py --input-type {args.input_type} --model-path {args.model_path} --task-config {args.task_config} --episode-id {args.episode_id} --scene-id {args.scene_id} --agent-seed {seed} --log-dir {args.log_dir}/seed={seed}/ --tb-dir {args.tb_dir}/seed={seed} --video-dir {args.video_dir}/seed={seed} --map-dir={args.map_dir}/seed={seed}"
        if args.make_videos:
            eval_command += "--make-videos"
        if args.make_maps:
            eval_command += "--make-maps"
        eval_habitat_args = shlex.split(eval_command)
        subprocess.run(eval_habitat_args, stdout=summary_file)
        summary_file.close()


if __name__ == "__main__":
    main()