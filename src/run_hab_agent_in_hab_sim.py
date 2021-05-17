import argparse

import habitat
from habitat.config import Config
from habitat.config.default import get_config
from classes.habitat_discrete_evaluator import HabitatDiscreteEvaluator
from habitat_baselines.agents.ppo_agents import PPOAgent


def get_default_config():
    c = Config()
    c.INPUT_TYPE = "blind"
    c.MODEL_PATH = "data/checkpoints/blind.pth"
    c.RESOLUTION = 256
    c.HIDDEN_SIZE = 512
    c.RANDOM_SEED = 7
    c.PTH_GPU_ID = 0
    c.GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
    return c


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
        "--task-config", type=str, default="$HOME/catkin_ws/src/ros-x-habitat/src/configs/pointnav_d_orignal.yaml"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50
    )
    args = parser.parse_args()

    # instantiate a discrete/continuous evaluator
    exp_config = get_config(args.task_config)
    evaluator = None
    if 'SIMULATOR' in exp_config:
        print('Instantiating discrete simulator')
        evaluator = HabitatDiscreteEvaluator(config_paths=args.task_config)
    elif 'PHYSICS_SIMULATOR' in exp_config:
        print('Instantiating continuous simulator with dynamics')
        raise NotImplementedError
    else:
        print('Simulator not properly specified')
        raise NotImplementedError

    agent_config = get_default_config()
    agent_config.INPUT_TYPE = args.input_type
    agent_config.MODEL_PATH = args.model_path
    num_episodes = args.num_episodes
    agent = PPOAgent(agent_config)

    print("Evaluating")
    metrics = evaluator.evaluate(agent, num_episodes=num_episodes)

    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
