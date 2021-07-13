from typing import Tuple, List, Dict


class Evaluator:
    r"""Abstract class for evaluating an agent in a simulation environment
    either with or without physics.

    Users should instantiate subclasses to 'Evaluator' for evaluation.
    """

    def evaluate(
        self,
        episode_id_last: str = "-1",
        scene_id_last: str = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        log_dir: str = "logs/",
        *args,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        r"""..
        Evaluate over episodes, starting from the last episode evaluated. Return evaluation
        metrics.

        :param episode_id_last: ID of the last episode evaluated; -1 for evaluating
            from start
        :param scene_id_last: Scene ID of the last episode evaluated
        :param log_dir: logging directory
        :return: a dictionary where each key is an episode's unique identifier as
            <episode-id>,<scene-id>; each value is the set of metrics from the episode.
        """
        raise NotImplementedError
