import argparse
import os

from habitat.config.default import get_config
from src.evaluators.habitat_evaluator import HabitatEvaluator
from src.constants.constants import NumericalMetrics

from src.utils import utils_logging, utils_visualization, utils_files


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
    parser.add_argument("--seed-file-path", type=str, default="seeds/seed=7.csv")
    parser.add_argument("--log-dir", type=str, default="logs/")
    parser.add_argument("--make-maps", default=False, action="store_true")
    parser.add_argument("--map-dir", type=str, default="habitat_maps/")
    parser.add_argument("--make-plots", default=False, action="store_true")
    parser.add_argument("--plot-dir", type=str, default="plots/")
    args = parser.parse_args()

    # get exp config
    exp_config = get_config(args.task_config)

    # get seeds if provided; otherwise use default seed from Habitat
    seeds = []
    if args.seed_file_path != "":
        seeds = utils_files.load_seeds_from_file(args.seed_file_path)
    else:
        seeds = [exp_config.SEED]

    # create log dir
    os.makedirs(name=f"{args.log_dir}", exist_ok=True)

    # create logger and log experiment settings
    logger = utils_logging.setup_logger(
        __name__, f"{args.log_dir}/summary-all_seeds.log"
    )
    logger.info("Experiment configuration:")
    logger.info(exp_config)

    # instantiate a discrete/continuous evaluator
    evaluator = None
    if "PHYSICS_SIMULATOR" in exp_config:
        logger.info("Instantiating continuous simulator with dynamics")
        evaluator = HabitatEvaluator(
            config_paths=args.task_config,
            input_type=args.input_type,
            model_path=args.model_path,
            enable_physics=True,
        )
    elif "SIMULATOR" in exp_config:
        logger.info("Instantiating discrete simulator")
        evaluator = HabitatEvaluator(
            config_paths=args.task_config,
            input_type=args.input_type,
            model_path=args.model_path,
            enable_physics=False,
        )
    else:
        logger.info("Simulator not properly specified")
        raise NotImplementedError

    logger.info("Started evaluation")
    metrics_list = []
    avg_metrics_all_seeds = {}
    maps = []
    for seed in seeds:
        # create logger for each seed and log the seed
        logger_per_seed = utils_logging.setup_logger(
            f"{__name__}-seed={seed}", f"{args.log_dir}/summary-seed={seed}.log"
        )
        logger_per_seed.info(f"Seed = {seed}")

        # create (per-episode) log dir
        os.makedirs(name=f"{args.log_dir}/seed={seed}", exist_ok=True)

        # evaluate
        metrics_and_maps = evaluator.evaluate_and_get_maps(
            episode_id_last=args.episode_id,
            scene_id_last=args.scene_id,
            log_dir=f"{args.log_dir}/seed={seed}",
            agent_seed=seed,
            map_height=200,
        )

        # extract top-down-maps
        maps_per_seed = evaluator.extract_metrics(metrics_and_maps, ["top_down_map"])
        maps.append(maps_per_seed)

        # extract other metrics
        metrics_per_seed = evaluator.extract_metrics(
            metrics_and_maps, [metric_name for metric_name in NumericalMetrics]
        )
        metrics_list.append(metrics_per_seed)

        # compute average metrics of this seed
        avg_metrics_per_seed = evaluator.compute_avg_metrics(metrics_per_seed)

        # save to dict of avg metrics across all seeds
        avg_metrics_all_seeds[seed] = avg_metrics_per_seed

        # log average metrics of this seed
        logger_per_seed.info("Printing average metrics:")
        for k, v in avg_metrics_per_seed.items():
            logger_per_seed.info("{}: {:.3f}".format(k, v))
        utils_logging.close_logger(logger_per_seed)
    logger.info("Evaluation ended")

    # log average metrics across all seeds
    avg_metrics = evaluator.compute_avg_metrics(avg_metrics_all_seeds)
    logger.info("Printing average metrics:")
    for k, v in avg_metrics.items():
        logger.info("{}: {:.3f}".format(k, v))

    # make top-down maps for each episode
    if args.make_maps:
        # create map dir
        os.makedirs(name=f"{args.map_dir}", exist_ok=True)

        if len(maps) > 0:
            for episode_identifier, _ in maps[0].items():
                # plot maps for each episode. Here we assume the same
                # episode has been evaluated with all seeds
                maps_per_episode = []
                for seed_index in range(len(seeds)):
                    maps_per_episode.append(
                        maps[seed_index][episode_identifier]["top_down_map"]
                    )
                utils_visualization.generate_grid_of_maps(
                    episode_identifier.split(",")[0],
                    episode_identifier.split(",")[1],
                    seeds,
                    maps_per_episode,
                    args.map_dir,
                )
    logger.info("Generated top-down maps")

    # make box-and-whisker plots of metrics vs. seed
    if args.make_plots:
        # create plot dir
        os.makedirs(name=f"{args.plot_dir}", exist_ok=True)

        # create box plots of metrics vs seeds
        utils_visualization.visualize_variability_due_to_seed_with_box_plots(
            metrics_list, seeds, args.plot_dir
        )
    logger.info("Generated metric plots")

    utils_logging.close_logger(logger)


if __name__ == "__main__":
    main()
