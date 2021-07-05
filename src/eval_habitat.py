import argparse

import habitat
from habitat.config import Config
from habitat.config.default import get_config
from src.classes.habitat_evaluator import HabitatEvaluator
import random

# logging
from classes import utils_logging

logger = utils_logging.setup_logger(__name__)


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
        "--agent-seed", type=int, default=7,
    )
    parser.add_argument("--log-dir", type=str, default="logs/")
    parser.add_argument("--make-videos", default=False, action="store_true")
    parser.add_argument("--video-dir", type=str, default="videos/")
    parser.add_argument("--tb-dir", type=str, default="tb/")
    parser.add_argument("--make-maps", default=False, action="store_true")
    parser.add_argument("--map-dir", type=str, default="maps/")
    args = parser.parse_args()

    # instantiate a discrete/continuous evaluator
    exp_config = get_config(args.task_config)
    evaluator = None
    if "PHYSICS_SIMULATOR" in exp_config:
        logger.info("Instantiating continuous simulator with dynamics")
        evaluator = HabitatEvaluator(config_paths=args.task_config, input_type=args.input_type, model_path=args.model_path, agent_seed=args.agent_seed, enable_physics=True)
    elif "SIMULATOR" in exp_config:
        logger.info("Instantiating discrete simulator")
        evaluator = HabitatEvaluator(config_paths=args.task_config, input_type=args.input_type, model_path=args.model_path, agent_seed=args.agent_seed, enable_physics=False)
    else:
        logger.info("Simulator not properly specified")
        raise NotImplementedError

    logger.info("Started Evaluation")
    metrics = evaluator.evaluate(
        episode_id_last=args.episode_id,
        scene_id_last=args.scene_id,
        log_dir=args.log_dir,
        make_videos=args.make_videos,
        video_dir=args.video_dir,
        tb_dir=args.tb_dir,
        make_maps=args.make_maps,
        map_dir=args.map_dir
    )

    logger.info("Printing average metrics:")
    for k, v in metrics.items():
        logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
