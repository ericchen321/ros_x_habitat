from src.evaluators.evaluator import Evaluator
from typing import Dict
from habitat.config.default import get_config
from habitat.core.agent import Agent
from collections import defaultdict
from ros_x_habitat.srv import *
from subprocess import Popen
import shlex
import rospy
import numpy as np


class MockHabitatROSEvaluator(Evaluator):
    r"""Class to evaluate Habitat agents in Habitat environments with ROS
    as middleware.
    """

    def __init__(
        self,
    ) -> None:
        r"""..
        Constructor for the setting 3 mock evaluator.
        """

        # start the evaluator node
        rospy.init_node("mock_evaluator_habitat_ros")
    
    def evaluate(
        self,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        *args,
        **kwargs
    ) -> Dict[str, float]:
        r"""..
        Evaluate over episodes, starting from the last episode evaluated. Return evaluation
        metrics.

        :param episode_id_last: ID of the last episode evaluated; -1 for evaluating
            from start
        :param scene_id_last: Scene ID of the last episode evaluated
        --- The following parameters are unused:
        :param log_dir: logging directory
        :return: dict containing metrics tracked by environment.
        """

        count_episodes = 0
        agg_metrics: Dict = defaultdict(float)
        eval_episode = rospy.ServiceProxy('eval_episode', EvalEpisode)

        # evaluate episodes, starting from the one after the last episode
        # evaluated
        while not rospy.is_shutdown():
            rospy.wait_for_service("eval_episode")
            try:
                # request env node to evaluate an episode
                resp = None
                if count_episodes == 0:
                    # jump to the first episode we want to evaluate
                    resp = eval_episode(episode_id_last, scene_id_last)
                else:
                    # evaluate the next episode
                    resp = eval_episode("-1", "")
                
                if resp.episode_id == "-1":
                    # no more episodes
                    print(f"Finished evaluation after: {count_episodes} episodes")
                    break
                else:
                    # get per-episode metrics
                    per_ep_metrics = {
                        "distance_to_goal": resp.distance_to_goal,
                        "success": resp.success,
                        "spl": resp.spl
                    }
                    
                    # calculate aggregated metrics over episodes eval'ed so far
                    for m, v in per_ep_metrics.items():
                        agg_metrics[m] += v
                    
                    count_episodes += 1
            except rospy.ServiceException:
                print(f"Evaluation call failed at {count_episodes}-th episode")
                break
        
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics