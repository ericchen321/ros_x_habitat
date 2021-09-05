import argparse
import os
from src.utils import utils_files, utils_visualization
from src.constants.constants import NumericalMetrics
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir-all-seeds", type=str, default="")
    parser.add_argument("--plot-dir", type=str, default="metric_plots/")
    args = parser.parse_args()

    # create plot dir
    os.makedirs(name=f"{args.plot_dir}", exist_ok=True)

    # get (per-seed) log dir paths
    list_of_log_dirs = utils_files.extract_seed_dir_paths(args.log_dir_all_seeds)

    # get seeds
    seeds = []
    for log_dir in list_of_log_dirs:
        seeds.append(log_dir.split("seed=")[1].rstrip("/"))

    # get log file paths
    list_of_log_filepaths = utils_files.extract_log_filepaths(
        list_of_log_dirs=list_of_log_dirs
    )

    # get metrics
    list_of_dict_of_metrics = utils_files.extract_metrics_from_each(
        metric_names=[
            NumericalMetrics.DISTANCE_TO_GOAL,
            NumericalMetrics.SUCCESS,
            NumericalMetrics.SPL,
            NumericalMetrics.NUM_STEPS,
            NumericalMetrics.SIM_TIME,
            NumericalMetrics.RESET_TIME,
            NumericalMetrics.AGENT_TIME,
        ],
        list_of_log_filepaths=list_of_log_filepaths,
    )

    # visualize (per-step)-agent-time, (per-step)-simulation-time,
    # (per-episode) reset time
    list_of_dict_of_metrics_for_box_plot = []
    for dict_of_metrics in list_of_dict_of_metrics:
        # remove other metrics
        dict_of_metrics_for_box_plot = HabitatSimEvaluator.extract_metrics(
            dict_of_metrics=dict_of_metrics,
            metric_names=[
                NumericalMetrics.SIM_TIME,
                NumericalMetrics.RESET_TIME,
                NumericalMetrics.AGENT_TIME,
            ],
        )
        list_of_dict_of_metrics_for_box_plot.append(dict_of_metrics_for_box_plot)
    utils_visualization.visualize_metrics_across_configs_with_box_plots(
        metrics_list=list_of_dict_of_metrics_for_box_plot,
        config_names=seeds,
        configs_or_seeds="seeds",
        plot_dir=args.plot_dir,
    )

    # visualize distance-to-goal, spl
    list_of_dict_of_metrics_for_histogram = []
    for dict_of_metrics in list_of_dict_of_metrics:
        # remove other metrics
        dict_of_metrics_for_histogram = HabitatSimEvaluator.extract_metrics(
            dict_of_metrics=dict_of_metrics,
            metric_names=[
                NumericalMetrics.DISTANCE_TO_GOAL,
                NumericalMetrics.SPL
            ],
        )
        list_of_dict_of_metrics_for_histogram.append(dict_of_metrics_for_histogram)
    utils_visualization.visualize_metrics_across_configs_with_histograms(
        metrics_list=list_of_dict_of_metrics_for_histogram,
        config_names=seeds,
        configs_or_seeds="seeds",
        plot_dir=args.plot_dir,
    )

    # visualize success
    utils_visualization.visualize_success_across_configs_with_pie_charts(
        metrics_list=list_of_dict_of_metrics,
        config_names=seeds,
        configs_or_seeds="seeds",
        plot_dir=args.plot_dir,
    )


if __name__ == "__main__":
    main()
