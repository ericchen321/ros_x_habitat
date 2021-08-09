import csv
from src.constants.constants import NumericalMetrics


def load_seeds_from_file(seed_file_path):
    r"""
    Load random seeds from file identified by seed_file_path.
    :param seed_file_path: path to a .csv file of seeds
    :returns: a list of the random seeds from the given file
    """
    seeds = []
    with open(seed_file_path, newline="") as csv_file:
        csv_lines = csv.reader(csv_file)
        for line in csv_lines:
            seeds.append(int(line[0]))
    return seeds


def load_episode_identifiers(episodes_to_visualize_file_path):
    r"""
    Load episode identifiers from the given file. Each episode must be specified
    by an episode ID and a scene ID.
    :param episodes_to_visualize_file_path: path to a .csv file of episode
        identifiers
    :returns: a list of episode ID's and a list of scene ID's. One-to-one
        correspondence
    """
    episode_ids = []
    scene_ids = []
    with open(episodes_to_visualize_file_path, newline="") as csv_file:
        csv_lines = csv.reader(csv_file)
        for line in csv_lines:
            episode_id = str(line[0])
            scene_id = str(line[1])
            assert episode_id is not None and episode_id != ""
            assert scene_id is not None and scene_id != ""
            episode_ids.append(episode_id)
            scene_ids.append(scene_id)

    return episode_ids, scene_ids


def extract_metrics_from_log_file(log_filepath, metric_names):
    r"""
    Create a dictionary of metrics for the episode logged in `log_filepath`.
    Require `metric_names` contain names only of numerical metrics.
    :param log_filepath: path to the log file of one episode
    :metric_names: metrics we want to extract
    :return: a tuple of three things:
        1) episode ID,
        2) scene ID,
        3) metrics dictionary
    """
    log_file = open(log_filepath, "r")
    log_file_lines = log_file.readlines()
    
    # get episode ID
    episode_id_line = log_file_lines[0]
    episode_id = str(episode_id_line.split(": ")[1]).rstrip("\n")

    # get scene ID
    scene_id_line = log_file_lines[1]
    scene_id = str(scene_id_line.split(": ")[1]).rstrip("\n")

    # for each metric, extract its value
    per_ep_metrics = {}
    metric_line_nums = {
        NumericalMetrics.DISTANCE_TO_GOAL: 2,
        NumericalMetrics.SUCCESS: 3,
        NumericalMetrics.SPL: 4,
        NumericalMetrics.NUM_STEPS: 5
    }
    for metric_name in metric_names:
        metric_line = log_file_lines[metric_line_nums[metric_name]]
        metric_val = float(metric_line.split(",")[2])
        per_ep_metrics[metric_name] = metric_val
    
    return (episode_id, scene_id, per_ep_metrics)