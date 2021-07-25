from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np

import rospy
from ros_x_habitat.srv import *
from src.constants.constants import (
    AgentResetCommands,
    EvalEpisodeSpecialIDs,
    NumericalMetrics
)
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator
from src.constants.constants import EvalEpisodeSpecialIDs


class MockHabitatROSEvaluator(HabitatSimEvaluator):
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

        # register eval episode client
        self.eval_episode = rospy.ServiceProxy("eval_episode", EvalEpisode)

        # register reset_agent client
        self.reset_agent = rospy.ServiceProxy("reset_agent", ResetAgent)

    def evaluate(
        self,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
    
        dict_of_metrics = {}
        count_episodes = 0

        # evaluate episodes, starting from the one after the last episode
        # evaluated
        while not rospy.is_shutdown():
            rospy.wait_for_service("eval_episode")
            try:
                # request env node to evaluate an episode
                resp = None
                if count_episodes == 0:
                    print(f"requesting to evaluate from after {episode_id_last},{scene_id_last}")
                    # jump to the first episode we want to evaluate
                    resp = self.eval_episode(episode_id_last, scene_id_last)
                else:
                    # evaluate the next episode
                    print("requesting to evaluate the next episode!!!!!!!")
                    resp = self.eval_episode(EvalEpisodeSpecialIDs.REQUEST_NEXT, "")

                if resp.episode_id == EvalEpisodeSpecialIDs.RESPONSE_NO_MORE_EPISODES:
                    # no more episodes
                    print(f"Finished evaluation after: {count_episodes} episodes")
                    break
                else:
                    # get per-episode metrics
                    per_ep_metrics = {
                        NumericalMetrics.DISTANCE_TO_GOAL: resp.distance_to_goal,
                        NumericalMetrics.SUCCESS: resp.success,
                        NumericalMetrics.SPL: resp.spl,
                    }

                    dict_of_metrics[f"{resp.episode_id},{resp.scene_id}"] = per_ep_metrics
                    count_episodes += 1

            except rospy.ServiceException:
                print(f"Evaluation call failed at {count_episodes}-th episode")
                break

        return dict_of_metrics

    def shutdown_env_node(self):
        rospy.wait_for_service("eval_episode")
        try:
            resp = self.eval_episode(EvalEpisodeSpecialIDs.REQUEST_SHUTDOWN, "")
        except rospy.ServiceException:
            print("Shutting down env node failed")
            raise rospy.ServiceException

    def shutdown_agent_node(self):
        rospy.wait_for_service("reset_agent")
        try:
            resp = self.reset_agent(int(AgentResetCommands.SHUTDOWN), 0)
            assert resp.done
        except rospy.ServiceException:
            print("Failed to shut down agent!")
            raise rospy.ServiceException