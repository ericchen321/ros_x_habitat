import argparse

import habitat
from habitat.config import Config
from habitat.config.default import get_config
from src.classes.habitat_ros_evaluator import HabitatROSEvaluator
from habitat_baselines.agents.ppo_agents import PPOAgent
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
    parser.add_argument("--sensor-pub-rate", type=float, default=5.0)
    parser.add_argument("--do-not-start-nodes-from-evaluator", default=False, action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs/")
    parser.add_argument("--make-videos", default=False, action="store_true")
    parser.add_argument("--video-dir", type=str, default="videos/")
    parser.add_argument("--tb-dir", type=str, default="tb/")
    args = parser.parse_args()

    # instantiate a discrete/continuous evaluator
    exp_config = get_config(args.task_config)
    evaluator = None
    if "SIMULATOR" in exp_config:
        logger.info("Instantiating discrete simulator")
        evaluator = HabitatROSEvaluator(input_type=args.input_type, model_path=args.model_path, config_paths=args.task_config, sensor_pub_rate=args.sensor_pub_rate, do_not_start_nodes=args.do_not_start_nodes_from_evaluator, enable_physics=False)
    elif "PHYSICS_SIMULATOR" in exp_config:
        logger.info("Instantiating continuous simulator with dynamics")
        # TODO: pass in control period
        evaluator = HabitatROSEvaluator(input_type=args.input_type, model_path=args.model_path, config_paths=args.task_config, sensor_pub_rate=args.sensor_pub_rate, do_not_start_nodes=args.do_not_start_nodes_from_evaluator, enable_physics=True)
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
    )

    logger.info("Printing average metrics:")
    for k, v in metrics.items():
        logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
