import argparse
import os
from src.utils import utils_files, utils_visualization
from src.constants.constants import NumericalMetrics
from src.evaluators.habitat_sim_evaluator import HabitatSimEvaluator


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir-discrete-no-ros", type=str, default="")
    parser.add_argument("--log-dir-discrete-ros", type=str, default="")
    parser.add_argument("--log-dir-continuous-no-ros", type=str, default="")
    parser.add_argument("--log-dir-continuous-ros", type=str, default="")
    parser.add_argument("--plot-dir", type=str, default="metric_plots/")
    parser.add_argument(
        "--plot-pairwise-diff-in-percentage", default=False, action="store_true"
    )
    args = parser.parse_args()

    # create plot dir
    os.makedirs(name=f"{args.plot_dir}", exist_ok=True)

    # get log file paths
    list_of_log_filepaths = utils_files.extract_log_filepaths(
        list_of_log_dirs=[
            args.log_dir_discrete_no_ros,
            args.log_dir_discrete_ros,
            args.log_dir_continuous_no_ros,
            args.log_dir_continuous_ros,
        ]
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

    # visualize number-of-steps, (per-step)-agent-time, (per-step)-simulation-time
    # with box plots
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
        config_names=[
            "-physics & -ROS",
            "-physics & +ROS",
            "+physics & -ROS",
            "+physics & +ROS",
        ],
        configs_or_seeds="configurations",
        plot_dir=args.plot_dir,
    )

    # visualize spl, distance_to_goal, num_steps with histograms
    list_of_dict_of_metrics_for_histogram = []
    for dict_of_metrics in list_of_dict_of_metrics:
        # remove other metrics
        dict_of_metrics_for_histogram = HabitatSimEvaluator.extract_metrics(
            dict_of_metrics=dict_of_metrics,
            metric_names=[
                NumericalMetrics.DISTANCE_TO_GOAL,
                NumericalMetrics.SPL,
                NumericalMetrics.NUM_STEPS,
            ],
        )
        list_of_dict_of_metrics_for_histogram.append(dict_of_metrics_for_histogram)
    utils_visualization.visualize_metrics_across_configs_with_histograms(
        metrics_list=list_of_dict_of_metrics_for_histogram,
        config_names=[
            "(a) -physics & -ROS",
            "(b) -physics & +ROS",
            "(c) +physics & -ROS",
            "(d) +physics & +ROS",
        ],
        configs_or_seeds="configurations",
        plot_dir=args.plot_dir,
    )


    # visualize success with pie charts
    utils_visualization.visualize_success_across_configs_with_pie_charts(
        metrics_list=list_of_dict_of_metrics,
        config_names=[
            "(a) -physics & -ROS",
            "(b) -physics & +ROS",
            "(c) +physics & -ROS",
            "(d) +physics & +ROS",
        ],
        configs_or_seeds="configurations",
        plot_dir=args.plot_dir,
    )

    # visualize total running time
    discrete_no_ros_time = utils_files.extract_experiment_running_time_from_log_file(
        f"{args.log_dir_discrete_no_ros}/../summary-seed=188076191.log"
    )
    discrete_ros_time = utils_files.extract_experiment_running_time_from_log_file(
        f"{args.log_dir_discrete_ros}/../summary-seed=188076191.log"
    )
    continuous_no_ros_time = utils_files.extract_experiment_running_time_from_log_file(
        f"{args.log_dir_continuous_no_ros}/../summary-seed=188076191.log"
    )
    continuous_ros_time = utils_files.extract_experiment_running_time_from_log_file(
        f"{args.log_dir_continuous_ros}/../summary-seed=188076191.log"
    )
    utils_visualization.visualize_running_times_with_bar_plots(
        running_times=[
            discrete_no_ros_time,
            discrete_ros_time,
            continuous_no_ros_time,
            continuous_ros_time,
        ],
        config_names=[
            "-physics & -ROS",
            "-physics & +ROS",
            "+physics & -ROS",
            "+physics & +ROS",
        ],
        plot_dir=args.plot_dir,
    )

    # visualize pair-wise differences of metrics
    # visualize effects of adding physics (Setting 2 vs 4)
    pairwise_diff_dict_of_metrics = (
        HabitatSimEvaluator.compute_pairwise_diff_of_metrics(
            dict_of_metrics_baseline=list_of_dict_of_metrics[0],
            dict_of_metrics_compared=list_of_dict_of_metrics[2],
            metric_names=[
                NumericalMetrics.DISTANCE_TO_GOAL,
                NumericalMetrics.SUCCESS,
                NumericalMetrics.SPL,
                NumericalMetrics.NUM_STEPS,
                NumericalMetrics.SIM_TIME,
                NumericalMetrics.RESET_TIME,
                NumericalMetrics.AGENT_TIME,
            ],
            compute_percentage=args.plot_pairwise_diff_in_percentage,
        )
    )
    utils_visualization.visualize_pairwise_percentage_diff_of_metrics(
        pairwise_diff_dict_of_metrics=pairwise_diff_dict_of_metrics,
        config_names=[
            "-physics & -ROS",
            "+physics & -ROS",
        ],
        diff_in_percentage=args.plot_pairwise_diff_in_percentage,
        plot_dir=args.plot_dir,
    )
    # visualize effects of adding ROS (Setting 2 vs 3)
    pairwise_diff_dict_of_metrics = (
        HabitatSimEvaluator.compute_pairwise_diff_of_metrics(
            dict_of_metrics_baseline=list_of_dict_of_metrics[0],
            dict_of_metrics_compared=list_of_dict_of_metrics[1],
            metric_names=[
                NumericalMetrics.DISTANCE_TO_GOAL,
                NumericalMetrics.SUCCESS,
                NumericalMetrics.SPL,
                NumericalMetrics.NUM_STEPS,
                NumericalMetrics.SIM_TIME,
                NumericalMetrics.RESET_TIME,
                NumericalMetrics.AGENT_TIME,
            ],
            compute_percentage=args.plot_pairwise_diff_in_percentage,
        )
    )
    utils_visualization.visualize_pairwise_percentage_diff_of_metrics(
        pairwise_diff_dict_of_metrics=pairwise_diff_dict_of_metrics,
        config_names=[
            "-physics & -ROS",
            "-physics & +ROS",
        ],
        diff_in_percentage=args.plot_pairwise_diff_in_percentage,
        plot_dir=args.plot_dir,
    )


if __name__ == "__main__":
    main()
