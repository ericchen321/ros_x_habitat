from classes.habitat_evaluator import HabitatEvaluator
from typing import Dict, Optional
from habitat.config.default import get_config
from classes.habitat_eval_rlenv import HabitatEvalRLEnv
from habitat.core.agent import Agent
from collections import defaultdict

# use TensorBoard to visualize
from classes.utils_tensorboard import TensorboardWriter, generate_video
from habitat.utils.visualizations.utils import observations_to_image
import numpy as np
from habitat.tasks.nav.nav import NavigationEpisode

class HabitatDiscreteEvaluator(HabitatEvaluator):
    r"""Class to evaluate Habitat agents producing discrete actions in environments
    without dynamics.
    """

    def __init__(
        self, config_paths: Optional[str] = None
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        """
        config_env = get_config(config_paths)
        # embed top-down map and heading sensor in config
        config_env.defrost()
        config_env.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        #config_env.TASK.SENSORS.append("HEADING_SENSOR")
        config_env.freeze()

        self._env = HabitatEvalRLEnv(config=config_env, enable_physics=False)
    

    def evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None, control_period: Optional[float] = 1.0
    ) -> Dict[str, float]:
        r"""..

        Args:
            agent: agent to be evaluated in environment.
            num_episodes: count of number of episodes for which the
            evaluation should be run.
            control_period: number of seconds in which each action should complete. Not used
                by the discrete evaluator since every action is instantaneous

        Return:
            dict containing metrics tracked by environment.
        """
        if num_episodes is None:
            num_episodes = len(self._env._env.episodes)
        else:
            assert num_episodes <= len(self._env._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)

        writer = TensorboardWriter('tb_benchmark/', flush_secs=30) # flush_specs from base_trainer.py

        count_episodes = 0
        print("number of episodes: " + str(num_episodes))
        while count_episodes < num_episodes:
            print(f"Working on  {count_episodes+1}/{num_episodes}-th episode")
            observations_per_episode = []
            agent.reset()
            observations_per_action = self._env._env.reset()
            current_episode = self._env._env.current_episode
            print(f"episode id: {current_episode.episode_id}")
            print(f"episode scene id: {current_episode.scene_id}")
            
            frame_counter = 0
            # act until one episode is over
            while not self._env._env.episode_over:
                action = agent.act(observations_per_action)
                observations_per_action = reward_per_action = done_per_action = info_per_action = None
                (observations_per_action, 
                reward_per_action, 
                done_per_action, 
                info_per_action)  = self._env.step(action)
                # generate an output image for the action. The image includes observations
                # and a top-down map showing the agent's state in the environment
                out_im_per_action = observations_to_image(observations_per_action, info_per_action)
                observations_per_episode.append(out_im_per_action)
            
            # episode ended
            # get per-episode metrics. for now we only extract
            # distance-to-goal, success, spl
            metrics = self._env._env.get_metrics()
            per_ep_metrics = {k: metrics[k] for k in ['distance_to_goal', 'success', 'spl']}
            # print distance_to_goal, success and spl
            for k, v in per_ep_metrics.items():
                print(f'{k},{v}')
            # calculate aggregated distance_to_goal, success and spl
            for m, v in per_ep_metrics.items():
                agg_metrics[m] += v
            count_episodes += 1
            # generate video
            generate_video(
                video_option=["disk", "tensorboard"],
                video_dir='video_benchmark_dir',
                images=observations_per_episode,
                episode_id=current_episode.episode_id,
                scene_id=current_episode.scene_id,
                checkpoint_idx=0,
                metrics=per_ep_metrics,
                tb_writer=writer,
            )
            
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics
