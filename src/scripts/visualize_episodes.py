import argparse
import os
from habitat.config.default import get_config
from src.evaluators.habitat_evaluator import HabitatEvaluator
from src.utils import utils_visualization, utils_files
from typing import Dict, List
import numpy as np


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
    parser.add_argument("--episodes-to-visualize-file-path", default="", type=str)
    parser.add_argument(
        "--episodes-to-visualize-file-has-header", default=False, action="store_true"
    )
    parser.add_argument("--seed-file-path", type=str, default="")
    parser.add_argument("--make-videos", default=False, action="store_true")
    parser.add_argument("--make-maps", default=False, action="store_true")
    parser.add_argument("--make-blank-maps", default=False, action="store_true")
    parser.add_argument("--map-height", type=int, default=200)
    parser.add_argument("--map-dir", type=str, default="habitat_maps/")

    args = parser.parse_args()

    # get exp config
    exp_config = get_config(args.task_config)

    # get seeds if provided; otherwise use default seed from Habitat
    seeds = []
    if args.seed_file_path != "":
        seeds = utils_files.load_seeds_from_file(args.seed_file_path)
    else:
        seeds = [exp_config.SEED]

    # get episode ID's and scene ID's of episodes to visualize
    episode_ids, scene_ids = utils_files.load_episode_identifiers(
        episodes_to_visualize_file_path=args.episodes_to_visualize_file_path,
        has_header=args.episodes_to_visualize_file_has_header,
    )

    # instantiate a discrete/continuous evaluator
    evaluator = None
    if "PHYSICS_SIMULATOR" in exp_config:
        evaluator = HabitatEvaluator(
            config_paths=args.task_config,
            input_type=args.input_type,
            model_path=args.model_path,
            enable_physics=True,
        )
    elif "SIMULATOR" in exp_config:
        evaluator = HabitatEvaluator(
            config_paths=args.task_config,
            input_type=args.input_type,
            model_path=args.model_path,
            enable_physics=False,
        )
    else:
        raise NotImplementedError

    # evaluate and generate videos
    if args.make_videos:
        # create video dir
        os.makedirs(name=f"{exp_config.VIDEO_DIR}", exist_ok=True)

        for seed in seeds:
            evaluator.generate_videos(episode_ids, scene_ids, seed)

    # evaluate and visualize top-down maps with agent position, shortest
    # and actual path
    if args.make_maps:
        # create map dir
        os.makedirs(name=f"{args.map_dir}", exist_ok=True)

        # create a list of per-seed maps for each episode
        maps: Dict[str, List[np.ndarray]] = {}
        for episode_id, scene_id in zip(episode_ids, scene_ids):
            maps[f"{episode_id},{scene_id}"] = []

        for seed in seeds:
            maps_one_seed = evaluator.generate_maps(
                episode_ids, scene_ids, seed, args.map_height
            )
            # add map from each episode to maps
            for episode_id, scene_id in zip(episode_ids, scene_ids):
                maps[f"{episode_id},{scene_id}"].append(
                    maps_one_seed[f"{episode_id},{scene_id}"]
                )

        # make grid of maps for each episode
        for episode_id, scene_id in zip(episode_ids, scene_ids):
            utils_visualization.generate_grid_of_maps(
                episode_id,
                scene_id,
                seeds,
                maps[f"{episode_id},{scene_id}"],
                args.map_dir,
            )

    # visualize blank top-down maps
    if args.make_blank_maps:
        # create map dir
        os.makedirs(name=f"{args.map_dir}", exist_ok=True)

        # create a blank map for each episode
        blank_maps = evaluator.get_blank_maps(episode_ids, scene_ids, args.map_height)

        for episode_id, scene_id in zip(episode_ids, scene_ids):
            utils_visualization.save_blank_map(
                episode_id,
                scene_id,
                blank_maps[f"{episode_id},{scene_id}"],
                args.map_dir,
            )


if __name__ == "__main__":
    main()
