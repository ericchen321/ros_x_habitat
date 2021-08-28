import csv
from src.constants.constants import NumericalMetrics
from typing import List, Dict, Tuple
import glob
from datetime import datetime


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


def load_episode_identifiers(
    episodes_to_visualize_file_path: str, has_header: bool
) -> Tuple[List[str], List[str]]:
    r"""
    Load episode identifiers from the given file. Each episode must be specified
    by an episode ID and a scene ID.
    :param episodes_to_visualize_file_path: path to a .csv file of episode
        identifiers
    :param has_header: if the .csv file has header or not
    :returns: a list of episode ID's and a list of scene ID's. One-to-one
        correspondence
    """
    episode_ids = []
    scene_ids = []
    with open(episodes_to_visualize_file_path, newline="") as csv_file:
        csv_lines = csv.reader(csv_file)
        if has_header:
            next(csv_lines, None)
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
        NumericalMetrics.NUM_STEPS: 5,
        NumericalMetrics.SIM_TIME: 6,
        NumericalMetrics.RESET_TIME: 7,
        NumericalMetrics.AGENT_TIME: 8,
    }
    for metric_name in metric_names:
        metric_line = log_file_lines[metric_line_nums[metric_name]]
        metric_val = float(metric_line.split(",")[2])
        per_ep_metrics[metric_name] = metric_val

    return (episode_id, scene_id, per_ep_metrics)


def extract_seed_dir_paths(
    log_dir: str
) -> List[str]:
    r"""
    Extract seed directory paths.
    :param log_dir: path to directory containing all seeds
    """
    log_dirs_all_seeds = []
    for log_dir_per_seed in glob.glob(f"{log_dir}/*/"):
        log_dirs_all_seeds.append(log_dir_per_seed)
    return log_dirs_all_seeds


def extract_log_filepaths(
    list_of_log_dirs: List[str],
) -> List[List[str]]:
    r"""
    Return paths to per-episode log files in each of the given directories,
    respectively.
    :param list_of_log_dirs: list of directory paths
    :return: a list containing lists of paths to per-episode log files
        from each directory
    """
    list_of_log_filepaths = []
    for log_dir in list_of_log_dirs:
        log_filepaths = []
        for log_filepath in glob.glob(f"{log_dir}/*.log"):
            log_filepaths.append(log_filepath)
        list_of_log_filepaths.append(log_filepaths)
    return list_of_log_filepaths


def get_metric_name_appended_by_suffix(metric_name: str, suffix: str) -> str:
    r"""
    Return String <metric_name><suffix>.
    :param metric_name: name of a metric
    :param suffix: suffix to append
    :return: the metric name appended by the suffix
    """
    return f"{metric_name}{suffix}"


def get_metric_name_without_suffix(metric_name: str, suffix: str) -> str:
    r"""
    Return the metric name with the suffix removed.
    :param metric_name: name of a metric
    :param suffix: suffix to append
    :return: the metric name with the suffix removed
    """
    return metric_name.rstrip(suffix)


def get_metric_names_with_suffices(
    metric_names: List[str], suffices: List[str]
) -> List[List[str]]:
    r"""
    Return list of metric names followed by each suffix.
    :param metric_names: names of (original) metrics
    :param suffices: suffices to append to metric names
    :return: list of metric names lists; each list has metric names follow
        the format <metric_name><suffix>
    """
    list_of_metric_name_lists = []
    for suffix in suffices:
        metric_name_list = []
        for metric_name in metric_names:
            metric_name_list.append(
                get_metric_name_appended_by_suffix(metric_name, suffix)
            )
        list_of_metric_name_lists.append(metric_name_list)
    return list_of_metric_name_lists


def extract_metrics_from_each(
    metric_names: List[str],
    list_of_log_filepaths: List[List[str]],
) -> List[Dict[str, Dict]]:
    r"""
    Create a dictionary of metrics for each given list of paths to log files.
    :param metric_names: list of metric names to extract
    :param list_of_log_filepaths: list of log file path lists
    :return: a list of dictionaries of metrics; each list corresponds to a
        given log file path list
    """
    list_of_dict_of_metrics = []
    for log_filepaths in list_of_log_filepaths:
        dict_of_metrics = {}
        for log_filepath in log_filepaths:
            episode_id, scene_id, per_episode_metrics = extract_metrics_from_log_file(
                log_filepath=log_filepath, metric_names=metric_names
            )
            dict_of_metrics[f"{episode_id},{scene_id}"] = per_episode_metrics
        list_of_dict_of_metrics.append(dict_of_metrics)
    return list_of_dict_of_metrics


def extract_experiment_running_time_from_log_file(log_filepath):
    r"""
    Compute experiment running time from a summarative log file.
    :param log_filepath: path to the log file
    :return: experiment running time in hours
    """
    # read the log file
    with open(log_filepath, "r") as log_file:
        lines = log_file.readlines()
        start_time_line = lines[0]
        end_time_line = lines[1]

    # get start/end date
    start_time_string = start_time_line.split(",")[0]
    end_time_string = end_time_line.split(",")[0]

    # convert to datetime objects
    start_time = datetime.strptime(start_time_string, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S")

    elapsed_seconds = (end_time - start_time).total_seconds()
    return float(elapsed_seconds) / 3600.0
