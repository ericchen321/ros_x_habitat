import argparse
import os

from habitat.config.default import get_config

from src.evaluators.habitat_evaluator import HabitatEvaluator
from src.utils import utils_visualization, utils_files


# logging


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
    parser.add_argument("--seed-file-path", type=str, default="seed=7.csv")
    parser.add_argument("--make-videos", default=False, action="store_true")
    parser.add_argument("--make-maps", default=False, action="store_true")
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
        try:
            os.mkdirs(f"{exp_config.VIDEO_DIR}")
        except FileExistsError:
            pass

        for seed in seeds:
            evaluator.generate_video(args.episode_id, args.scene_id, seed)

    # evaluate and visualize top-down maps
    if args.make_maps:
        # create map dir
        try:
            os.mkdirs(f"{args.map_dir}")
        except FileExistsError:
            pass

        maps = []
        for seed in seeds:
            map_one_seed = evaluator.generate_map(
                args.episode_id, args.scene_id, seed, 200
            )
            maps.append(map_one_seed)
        utils_visualization.generate_grid_of_maps(args.episode_id, args.scene_id, seeds, maps, args.map_dir)


if __name__ == "__main__":
    main()
