import argparse
import os
from src.utils import utils_logging, utils_files
from src.constants.constants import NumericalMetrics
import csv
from typing import Tuple, List, Dict
import glob
import numpy as np


def get_episodes_success_in_1_fail_in_2(
    dict_of_metrics_1: Dict[str, Dict],
    dict_of_metrics_2: Dict[str, Dict],
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    r"""
    Extract episodes that are registered as a success in `dict_of_metrics_1` but
    as a fail case in `dict_of_metrics_2`.
    :param dict_of_metrics_1: dictionary of metrics from experiment 1
    :param dict_of_metrics_2: dictionary of metrics from experiment 2
    :return: a tuple of two dictionaries of metrics, each having only episodes that
        satisfy the criterion above
    """
    # precondition check
    assert len(dict_of_metrics_1) == len(dict_of_metrics_2)

    dict_of_metrics_1_subset = {}
    dict_of_metrics_2_subset = {}

    # find episodes that satisfy the criteria
    for episode_identifier, episode_metrics_1 in dict_of_metrics_1.items():
        episode_metrics_2 = dict_of_metrics_2[episode_identifier]
        if (
            np.linalg.norm(episode_metrics_1[NumericalMetrics.SUCCESS] - 1.0) < 1e-5
            and np.linalg.norm(episode_metrics_2[NumericalMetrics.SUCCESS] - 0.0) < 1e-5
        ):
            dict_of_metrics_1_subset[episode_identifier] = episode_metrics_1
            dict_of_metrics_2_subset[episode_identifier] = episode_metrics_2

    return dict_of_metrics_1_subset, dict_of_metrics_2_subset


def get_episodes_success_in_both_but_metrics_differ_by_a_lot(
    dict_of_metrics_1: Dict[str, Dict],
    dict_of_metrics_2: Dict[str, Dict],
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    r"""
    Extract episodes that are registered as a success in both `dict_of_metrics_1`
    and in `dict_of_metrics_2`, but spl differ by >= 50%.
    :param dict_of_metrics_1: dictionary of metrics from experiment 1
    :param dict_of_metrics_2: dictionary of metrics from experiment 2
    :return: a tuple of two dictionaries of metrics, each having only episodes that
        satisfy the criterion above
    """
    # precondition check
    assert len(dict_of_metrics_1) == len(dict_of_metrics_2)

    dict_of_metrics_1_subset = {}
    dict_of_metrics_2_subset = {}

    # find episodes that satisfy the criteria
    for episode_identifier, episode_metrics_1 in dict_of_metrics_1.items():
        episode_metrics_2 = dict_of_metrics_2[episode_identifier]
        # NOTE: criteria:
        # 1) Both are successes
        # 2.1) Exp 2 SPL < 50% Exp 1 SPL OR 2.2) Exp 2 SPL > 150% Exp 1 SPL
        if (
            np.linalg.norm(episode_metrics_1[NumericalMetrics.SUCCESS] - 1.0) < 1e-5
            and np.linalg.norm(episode_metrics_2[NumericalMetrics.SUCCESS] - 1.0) < 1e-5
        ):
            spl_ratio = (
                episode_metrics_2[NumericalMetrics.SPL]
                / episode_metrics_1[NumericalMetrics.SPL]
            )
            if spl_ratio < 0.5 or spl_ratio > 1.5:
                dict_of_metrics_1_subset[episode_identifier] = episode_metrics_1
                dict_of_metrics_2_subset[episode_identifier] = episode_metrics_2

    return dict_of_metrics_1_subset, dict_of_metrics_2_subset


def get_episodes_fail_in_1_success_in_2(
    dict_of_metrics_1: Dict[str, Dict],
    dict_of_metrics_2: Dict[str, Dict],
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    r"""
    Extract episodes that are registered as a fail case in `dict_of_metrics_1`,
    but as a success in `dict_of_metrics_2`.
    :param dict_of_metrics_1: dictionary of metrics from experiment 1
    :param dict_of_metrics_2: dictionary of metrics from experiment 2
    :return: a tuple of two dictionaries of metrics, each having only episodes that
        satisfy the criterion above
    """
    # precondition check
    assert len(dict_of_metrics_1) == len(dict_of_metrics_2)

    dict_of_metrics_1_subset = {}
    dict_of_metrics_2_subset = {}

    # find episodes that satisfy the criteria
    for episode_identifier, episode_metrics_1 in dict_of_metrics_1.items():
        episode_metrics_2 = dict_of_metrics_2[episode_identifier]
        if (
            np.linalg.norm(episode_metrics_2[NumericalMetrics.SUCCESS] - 1.0) < 1e-5
            and np.linalg.norm(episode_metrics_1[NumericalMetrics.SUCCESS] - 0.0) < 1e-5
        ):
            dict_of_metrics_1_subset[episode_identifier] = episode_metrics_1
            dict_of_metrics_2_subset[episode_identifier] = episode_metrics_2

    return dict_of_metrics_1_subset, dict_of_metrics_2_subset

def get_episodes_fail_in_both(
    dict_of_metrics_1: Dict[str, Dict],
    dict_of_metrics_2: Dict[str, Dict],
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    r"""
    Extract episodes that are registered as a fail case in both `dict_of_metrics_1`,
    and `dict_of_metrics_2`.
    :param dict_of_metrics_1: dictionary of metrics from experiment 1
    :param dict_of_metrics_2: dictionary of metrics from experiment 2
    :return: a tuple of two dictionaries of metrics, each having only episodes that
        satisfy the criterion above
    """
    # precondition check
    assert len(dict_of_metrics_1) == len(dict_of_metrics_2)

    dict_of_metrics_1_subset = {}
    dict_of_metrics_2_subset = {}

    # find episodes that satisfy the criteria
    for episode_identifier, episode_metrics_1 in dict_of_metrics_1.items():
        episode_metrics_2 = dict_of_metrics_2[episode_identifier]
        if (
            np.linalg.norm(episode_metrics_2[NumericalMetrics.SUCCESS] - 0.0) < 1e-5
            and np.linalg.norm(episode_metrics_1[NumericalMetrics.SUCCESS] - 0.0) < 1e-5
        ):
            dict_of_metrics_1_subset[episode_identifier] = episode_metrics_1
            dict_of_metrics_2_subset[episode_identifier] = episode_metrics_2

    return dict_of_metrics_1_subset, dict_of_metrics_2_subset


def zip_metrics_1_and_2(
    fieldnames: List[str],
    dict_of_metrics_1: Dict[str, Dict],
    dict_of_metrics_2: Dict[str, Dict],
) -> Dict[str, Dict]:
    r"""
    Merge metrics of the same episode from `dict_of_metrics_1` and `dict_of_metrics_2`.
    :param fieldnames: field names in the merged dictionary of metrics for each
        episode, eg. 'SPL_1', 'SUCCESS_2'
    :param dict_of_metrics_1: dictionary of metrics from experiment 1
    :param dict_of_metrics_2: dictionary of metrics from experiment 2
    :return: a dictionary of merged metrics from all episodes. Each per-episode
        dictionary also contains episode ID and scene ID, eg.
        {'0,van-gogh-room.glb': {'episode_id': 0, 'scene_id': 'van-gogh-room.glb',
        'distance_to_goal_1': 0.05, 'success_1': 1.0, 'spl_1': 0.98,
        'distance_to_goal_2': 0.10, 'success_2': 1.0, 'spl_2': 0.95}}
    """
    # precondition check
    assert len(dict_of_metrics_1) == len(dict_of_metrics_2)

    # merge metrics
    dict_of_metrics_merged = {}
    for episode_identifier, episode_metrics_1 in dict_of_metrics_1.items():
        dict_of_metrics_merged[episode_identifier] = {}
        for field_name in fieldnames:
            if field_name == "episode_id":
                dict_of_metrics_merged[episode_identifier][
                    field_name
                ] = episode_identifier.split(",")[0]
            elif field_name == "scene_id":
                dict_of_metrics_merged[episode_identifier][
                    field_name
                ] = episode_identifier.split(",")[1]
            else:
                if "_1" in field_name:
                    metric_name = utils_files.get_metric_name_without_suffix(
                        field_name, "_1"
                    )
                    dict_of_metrics_merged[episode_identifier][
                        field_name
                    ] = episode_metrics_1[metric_name]
                elif "_2" in field_name:
                    metric_name = utils_files.get_metric_name_without_suffix(
                        field_name, "_2"
                    )
                    episode_metrics_2 = dict_of_metrics_2[episode_identifier]
                    dict_of_metrics_merged[episode_identifier][
                        field_name
                    ] = episode_metrics_2[metric_name]
    return dict_of_metrics_merged


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir-1", type=str, default="")
    parser.add_argument("--log-dir-2", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="logs/compare_results/")
    parser.add_argument("--episode-dir", type=str, default="episodes/")
    parser.add_argument(
        "--mode",
        default="find_cases_success_in_1_fail_in_2",
        choices=[
            "find_cases_success_in_1_fail_in_2",
            "find_cases_success_in_both_but_metrics_differ_by_a_lot",
            "find_cases_fail_in_1_success_in_2",
            "find_cases_fail_in_both"],
    )
    args = parser.parse_args()

    # create log dir
    os.makedirs(name=f"{args.log_dir}", exist_ok=True)

    # create episode dir
    os.makedirs(name=f"{args.episode_dir}", exist_ok=True)

    # set log filename and episode csv ilename
    log_dir_1_no_trailing_slash = args.log_dir_1.rstrip("/")
    log_dir_2_no_trailing_slash = args.log_dir_2.rstrip("/")
    log_filename = f"compare={os.path.basename(log_dir_1_no_trailing_slash)}-vs-{os.path.basename(log_dir_2_no_trailing_slash)}-mode={args.mode}.log"
    episode_filename = f"compare={os.path.basename(log_dir_1_no_trailing_slash)}-vs-{os.path.basename(log_dir_2_no_trailing_slash)}=mode={args.mode}.csv"

    # create logger and log comparison settings
    logger = utils_logging.setup_logger(__name__, f"{args.log_dir}/{log_filename}")
    logger.info("Compared directories:")
    logger.info(args.log_dir_1)
    logger.info(args.log_dir_2)
    logger.info("Mode:")
    logger.info(args.mode)
    logger.info("Writing episode csv file to:")
    logger.info(f"{args.log_dir}/{episode_filename}")

    # get log file paths
    list_of_log_filepaths = utils_files.extract_log_filepaths(
        list_of_log_dirs=[args.log_dir_1, args.log_dir_2]
    )

    # set up metric names to extract
    metric_names = [
        NumericalMetrics.DISTANCE_TO_GOAL,
        NumericalMetrics.SUCCESS,
        NumericalMetrics.SPL,
        NumericalMetrics.NUM_STEPS,
    ]
    list_of_metric_names = utils_files.get_metric_names_with_suffices(
        metric_names, ["_1", "_2"]
    )
    metric_names_1 = list_of_metric_names[0]
    metric_names_2 = list_of_metric_names[1]

    # get metrics
    list_of_dict_of_metrics = utils_files.extract_metrics_from_each(
        metric_names=metric_names, list_of_log_filepaths=list_of_log_filepaths
    )

    # find episodes of interest
    dict_of_metrics_1 = list_of_dict_of_metrics[0]
    dict_of_metrics_2 = list_of_dict_of_metrics[1]
    if args.mode == "find_cases_success_in_1_fail_in_2":
        (
            dict_of_metrics_subset_1,
            dict_of_metrics_subset_2,
        ) = get_episodes_success_in_1_fail_in_2(
            dict_of_metrics_1=dict_of_metrics_1,
            dict_of_metrics_2=dict_of_metrics_2,
        )
    elif args.mode == "find_cases_success_in_both_but_metrics_differ_by_a_lot":
        (
            dict_of_metrics_subset_1,
            dict_of_metrics_subset_2,
        ) = get_episodes_success_in_both_but_metrics_differ_by_a_lot(
            dict_of_metrics_1=dict_of_metrics_1,
            dict_of_metrics_2=dict_of_metrics_2,
        )
    elif args.mode == "find_cases_fail_in_1_success_in_2":
        (
            dict_of_metrics_subset_1,
            dict_of_metrics_subset_2,
        ) = get_episodes_fail_in_1_success_in_2(
            dict_of_metrics_1=dict_of_metrics_1,
            dict_of_metrics_2=dict_of_metrics_2,
        )
    elif args.mode == "find_cases_fail_in_both":
        dict_of_metrics_subset_1, dict_of_metrics_subset_2 = get_episodes_fail_in_both(
            dict_of_metrics_1=dict_of_metrics_1,
            dict_of_metrics_2=dict_of_metrics_2,
        )
    
    # merge metrics from experiment 1 and 2
    dict_of_metrics_1_and_2 = zip_metrics_1_and_2(
        fieldnames=(["episode_id", "scene_id"] + metric_names_1 + metric_names_2),
        dict_of_metrics_1=dict_of_metrics_subset_1,
        dict_of_metrics_2=dict_of_metrics_subset_2,
    )

    # write the episodes of interest to the .csv file
    with open(f"{args.episode_dir}/{episode_filename}", "w", newline="") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=(["episode_id", "scene_id"] + metric_names_1 + metric_names_2),
        )
        csv_writer.writeheader()
        for _, id_and_metrics in dict_of_metrics_1_and_2.items():
            csv_writer.writerow(id_and_metrics)

    utils_logging.close_logger(logger)


if __name__ == "__main__":
    main()
