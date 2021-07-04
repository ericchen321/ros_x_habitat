from typing import Dict


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
        make_videos: bool = False,
        video_dir: str = "videos/",
        tb_dir: str = "tb/",
        make_maps: bool = False,
        map_dir: str = "maps/",
        *args,
        **kwargs
    ) -> Dict[str, float]:
        r"""..
        Evaluate over episodes, starting from the last episode evaluated. Return evaluation
        metrics.

        :param episode_id_last: ID of the last episode evaluated; -1 for evaluating
            from start
        :param scene_id_last: Scene ID of the last episode evaluated
        :param log_dir: logging directory
        :param make_videos: toggle video production on/off
        :param video_dir: directory to store videos
        :param tb_dir: Tensorboard logging directory
        :param map_maps: toggle overlayed map production on/off
        :param map_dir: directory to store maps
        :return: dict containing metrics tracked by environment.
        """
        raise NotImplementedError
