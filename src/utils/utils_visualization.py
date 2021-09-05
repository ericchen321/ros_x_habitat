#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# moved from habitat_baselines due to dependency issues

import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations.utils import images_to_video
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import seaborn as sns
from habitat.utils.visualizations import maps
from PIL import Image
from src.constants.constants import NumericalMetrics
from habitat.utils.visualizations.utils import draw_collision

cv2 = try_cv2_import()


class TensorboardWriter:
    def __init__(self, log_dir: str, *args: Any, **kwargs: Any):
        r"""A Wrapper for tensorboard SummaryWriter. It creates a dummy writer
        when log_dir is empty string or None. It also has functionality that
        generates tb video directly from numpy images.

        Args:
            log_dir: Save directory location. Will not write to disk if
            log_dir is an empty string.
            *args: Additional positional args for SummaryWriter
            **kwargs: Additional keyword args for SummaryWriter
        """
        self.writer = None
        if log_dir is not None and len(log_dir) > 0:
            self.writer = SummaryWriter(log_dir, *args, **kwargs)

    def __getattr__(self, item):
        if self.writer:
            return self.writer.__getattribute__(item)
        else:
            return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()

    def add_video_from_np_images(
        self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10
    ) -> None:
        r"""Write video into tensorboard from images frames.

        Args:
            video_name: name of video string.
            step_idx: int of checkpoint index to be displayed.
            images: list of n frames. Each frame is a np.ndarray of shape.
            fps: frame per second for output video.

        Returns:
            None.
        """
        if not self.writer:
            return
        # initial shape of np.ndarray list: N * (H, W, 3)
        frame_tensors = [torch.from_numpy(np_arr).unsqueeze(0) for np_arr in images]
        video_tensor = torch.cat(tuple(frame_tensors))
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        # final shape of video tensor: (1, n, 3, H, W)
        self.writer.add_video(video_name, video_tensor, fps=fps, global_step=step_idx)


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: int,
    scene_id: int,
    agent_seed: int,
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        scene_id: scene id for video naming.
        agent_seed: agent initialization seed for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    scene_id = os.path.basename(scene_id)
    video_name = (
        f"episode={episode_id}-scene={scene_id}-seed={agent_seed}-ckpt={checkpoint_idx}-"
        + "-".join(metric_strs)
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )


def generate_grid_of_maps(episode_id, scene_id, seeds, maps, map_dir):
    """
    Paste top-down-maps from agent initialized with the given seeds to a grid
    image. Save the grid image to <map_dir>/episode=<episode_id>-scene=<scene_id>.png.
    Code modified based on tutorial from
    https://matplotlib.org/stable/gallery/axes_grid1/simple_axesgrid.html

    Args:
        episode_id: episode's ID.
        scene_id: scene ID of the episode, starts with data/dataset/...
        seeds: seeds used to initialize the agents.
        maps: maps produced by the agents navigating in np.ndarray format. Should be in
            the same order as seeds.
        map_dir: directory to store the map
    """
    fig = plt.figure(figsize=(16.0, 4.0))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(2, int(np.ceil(len(seeds) / 2))),  # creates n/2 x 2 grid of axes
        axes_pad=0.4,  # pad between axes in inch.
    )

    for ax, im, seed in zip(grid, maps, seeds):
        # iterating over the grid to return the axes
        ax.set_title(f"Seed={seed}", fontdict=None, loc="center", color="k")
        ax.imshow(im)

    fig.savefig(
        f"{map_dir}/episode={episode_id}-scene={os.path.basename(scene_id)}.png"
    )
    plt.close(fig)


def colorize_and_fit_to_height(top_down_map_raw: np.ndarray, output_height: int):
    r"""Given the output of the TopDownMap measure, colorizes the map,
    and fits to a desired output height. Modified on the basis of
    maps.colorize_draw_agent_and_fit_to_height from habitat-lab

    :param top_down_map_raw: raw top-down map
    :param output_height: The desired output height
    """
    top_down_map = maps.colorize_topdown_map(top_down_map_raw, None)

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map to align with rgb view
    old_h, old_w, _ = top_down_map.shape
    top_down_height = output_height
    top_down_width = int(float(top_down_height) / old_h * old_w)
    # cv2 resize (dsize is width first)
    top_down_map = cv2.resize(
        top_down_map,
        (top_down_width, top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return top_down_map


def save_blank_map(episode_id: str, scene_id: str, blank_map: np.ndarray, map_dir: str):
    r"""
    Save the given blank map in .pgm format in <map_dir>/
    :param episode_id: episode ID
    :param scene_id: scene ID
    :param blank_map: blank top-down map of the specified episode
    :param map_dir: directory to save the map
    """
    map_img = Image.fromarray(blank_map, "RGB")
    map_img.save(
        f"{map_dir}/blank_map-episode={episode_id}-scene={os.path.basename(scene_id)}.pgm"
    )


def resolve_metric_unit(metric_name):
    r"""
    Return a string of the unit of the given metric.
    :param metric_name: name of the metric
    :return: a unit string
    """
    if metric_name == NumericalMetrics.DISTANCE_TO_GOAL:
        return "(meters)"
    elif (
        metric_name == NumericalMetrics.SUCCESS
        or metric_name == NumericalMetrics.SPL
        or metric_name == NumericalMetrics.NUM_STEPS):
        return ""
    elif (
        metric_name == NumericalMetrics.SIM_TIME
        or metric_name == NumericalMetrics.RESET_TIME
        or metric_name == NumericalMetrics.AGENT_TIME):
        return "(seconds)"


def visualize_variability_due_to_seed_with_box_plots(
    metrics_list: List[Dict[str, Dict[str, float]]],
    seeds: List[int],
    plot_dir: str,
):
    r"""
    Generate box plots from metrics and seeds. Requires same metrics collected
    from all seeds. Save the plots to <plot_dir>/<metric_name>-<n>_seeds.png,
    where <metric_name> is for eg. "spl", <n> is the number of seeds.
    Args:
        metrics_list: list of metrics collected from experiment run with the
            given seeds.
        seeds: seeds to initialize agents. Should be in the same order as
            metrics_list.
        plot_dir: directory to save the box plot.
    """
    # check if we have metrics from all seeds
    num_seeds = len(seeds)
    assert len(metrics_list) == num_seeds

    # return if no data
    if num_seeds == 0:
        return
    num_samples_per_seed = len(metrics_list[0])
    if num_samples_per_seed == 0:
        return

    # check if all seeds have the same number of data points
    # for i in range(num_seeds):
    #    assert  len(metrics_list[i]) == num_samples_per_seed

    # extract metric names
    metric_names = []
    for _, episode_metrics in metrics_list[0].items():
        for metric_name, _ in episode_metrics.items():
            metric_names.append(metric_name)
        break

    # build dataframe
    data = {}
    total_num_samples = num_samples_per_seed * num_seeds
    data["seed"] = np.ndarray((total_num_samples,))
    for metric_name in metric_names:
        data[metric_name] = np.ndarray((total_num_samples,))
    # populate each array
    total_sample_count = 0
    for seed_index in range(num_seeds):
        for _, episode_metrics in metrics_list[seed_index].items():
            # register a new sample
            data["seed"][total_sample_count] = seeds[seed_index]
            for metric_name in metric_names:
                data[metric_name][total_sample_count] = episode_metrics[metric_name]
            total_sample_count += 1
    df = pd.DataFrame(data)

    # drop invalid samples
    # code adapted from piRSquared's work on
    # https://stackoverflow.com/questions/45745085/python-pandas-how-to-remove-nan-and-inf-values
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    # create box-and-strip plot for each metric
    for metric_name in metric_names:
        fig = plt.figure(figsize=(12.8, 9.6))
        ax = fig.add_subplot(111)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        sns.boxplot(x="seed", y=metric_name, data=df, ax=ax)
        sns.stripplot(x="seed", y=metric_name, data=df, color=".25", size=2, ax=ax)
        fig.subplots_adjust(bottom=0.15)
        fig.savefig(f"{plot_dir}/{metric_name}-{num_seeds}_seeds.png")
        plt.close(fig)


def visualize_metrics_across_configs_with_box_plots(
    metrics_list: List[Dict[str, Dict[str, float]]],
    config_names: List[str],
    configs_or_seeds: str,
    plot_dir: str,
):
    r"""
    Generate box plots from metrics and experiment configurations. Requires same
    metrics collected across all configs. Save the plots to
    <plot_dir>/<metric_name>.png, where <metric_name> is for eg. "agent_time".
    Args:
        metrics_list: list of metrics collected from experiment run with the
            given seeds.
        config_names: names of experiment configurations. Should be in the same
            order as metrics_list.
        configs_or_seeds: if visualizing across configs or seeds. Can only be
            "configurations" or "seeds"
        plot_dir: directory to save the box plot.
    """
    # check configs_or_seeds
    assert configs_or_seeds in ["configurations", "seeds"]
    
    # check if we have metrics from all configs
    num_configs = len(config_names)
    assert len(metrics_list) == num_configs

    # return if no data
    if num_configs == 0:
        return
    num_samples_per_config = len(metrics_list[0])
    if num_samples_per_config == 0:
        return

    # check if all configs have the same number of data points
    # for i in range(num_configs):
    #    assert  len(metrics_list[i]) == num_samples_per_config

    # extract metric names
    metric_names = []
    for _, episode_metrics in metrics_list[0].items():
        for metric_name, _ in episode_metrics.items():
            metric_names.append(metric_name)
        break

    # build dataframe
    data = {}
    total_num_samples = num_samples_per_config * num_configs
    data[configs_or_seeds] = np.ndarray((total_num_samples,), dtype=object)
    for metric_name in metric_names:
        data[metric_name] = np.ndarray((total_num_samples,))
    # populate each array
    total_sample_count = 0
    for config_index in range(num_configs):
        for _, episode_metrics in metrics_list[config_index].items():
            # register a new sample
            data[configs_or_seeds][total_sample_count] = config_names[config_index]
            for metric_name in metric_names:
                data[metric_name][total_sample_count] = episode_metrics[metric_name]
            total_sample_count += 1
    df = pd.DataFrame(data)

    # drop invalid samples
    # code adapted from piRSquared's work on
    # https://stackoverflow.com/questions/45745085/python-pandas-how-to-remove-nan-and-inf-values
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    # create box-and-strip plot for each metric
    for metric_name in metric_names:
        fig = plt.figure(figsize=(12.8, 9.6))
        ax = fig.add_subplot(111)
        sns.set(font_scale=1.2, style="white")
        sns.boxplot(x=configs_or_seeds, y=metric_name, data=df, ax=ax)
        sns.stripplot(x=configs_or_seeds, y=metric_name, data=df, color=".25", size=2, ax=ax)
        ax.set_xlabel(f"{configs_or_seeds}", fontsize=22)
        if configs_or_seeds == "seeds":
            plt.xticks(rotation=90, ha="right")
            plt.subplots_adjust(bottom=0.2)
        else:
            plt.xticks(rotation=0)
            plt.subplots_adjust(bottom=0.1)
        ax.set_ylabel(f"{metric_name.value} {resolve_metric_unit(metric_name)}", fontsize=22)
        #fig.suptitle(f"{metric_name.value} across {num_configs} {configs_or_seeds}")
        fig.savefig(f"{plot_dir}/{metric_name}-{num_configs}_{configs_or_seeds}.png")
        plt.close(fig)


def visualize_success_across_configs_with_pie_charts(
    metrics_list: List[Dict[str, Dict[str, float]]],
    config_names: List[str],
    configs_or_seeds: str,
    plot_dir: str,
):
    r"""
    Generate pie charts to show success counts across experiment configurations. Require
    the success metric collected across all configs. Save the plot to <plot_dir>/success.png.
    Args:
        metrics_list: list of metrics collected from experiment run with the
            given seeds.
        config_names: names of experiment configurations. Should be in the same
            order as metrics_list.
        configs_or_seeds: if visualizing across configs or seeds. Can only be
            "configurations" or "seeds"
        plot_dir: directory to save the pie chart.
    """
    # check configs_or_seeds
    assert configs_or_seeds in ["configurations", "seeds"]

    # check if we have metrics from all configs
    num_configs = len(config_names)
    assert len(metrics_list) == num_configs

    # return if no data
    if num_configs == 0:
        return
    num_samples_per_config = len(metrics_list[0])
    if num_samples_per_config == 0:
        return

    # check if all configs have the same number of data points
    # for i in range(num_configs):
    #    assert  len(metrics_list[i]) == num_samples_per_config

    # build data for plotting
    success_counts = []
    for config_index in range(0, num_configs):
        # count success in this config
        dict_of_metrics = metrics_list[config_index]
        count = 0
        for _, per_episode_metrics in dict_of_metrics.items():
            count += per_episode_metrics[NumericalMetrics.SPL]
        success_counts.append(count)
    dict_of_success_counts = {}
    for config_index in range(0, num_configs):
        dict_of_success_counts[config_names[config_index]] = [
            success_counts[config_index],
            num_samples_per_config - success_counts[config_index],
        ]
    
    # create pie plots for all configs
    fig, axes = plt.subplots(
        int(np.ceil(num_configs/2)),
        2,
        sharey=True,
        figsize=(9.6, 12.8))
    axes_flattened = axes.ravel()
    for config_index in range(0, num_configs):
        config_name = config_names[config_index]
        data_per_config = dict_of_success_counts[config_name]
        axes_flattened[config_index].pie(data_per_config,
            labels=["success", "fail"],
            autopct='%1.1f%%',
            shadow=False,
            startangle=90)
        axes_flattened[config_index].set_title(config_name)
    #fig.suptitle(f"proportion of succeeded/failed episodes across {num_configs} {configs_or_seeds}")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/success-{num_configs}_{configs_or_seeds}.png")
    plt.close(fig)


def visualize_metrics_across_configs_with_histograms(
    metrics_list: List[Dict[str, Dict[str, float]]],
    config_names: List[str],
    configs_or_seeds: str,
    plot_dir: str,
):
    r"""
    Generate histograms from metrics and experiment configurations. Requires same
    metrics collected across all configs. Save the plots to
    <plot_dir>/<metric_name>.png, where <metric_name> is for eg. "spl".
    Args:
        metrics_list: list of metrics collected from experiment run with the
            given seeds.
        config_names: names of experiment configurations. Should be in the same
            order as metrics_list.
        configs_or_seeds: if visualizing across configs or seeds. Can only be
            "configurations" or "seeds"
        plot_dir: directory to save the histograms.
    """
    # check configs_or_seeds
    assert configs_or_seeds in ["configurations", "seeds"]

    # check if we have metrics from all configs
    num_configs = len(config_names)
    assert len(metrics_list) == num_configs

    # return if no data
    if num_configs == 0:
        return
    num_samples_per_config = len(metrics_list[0])
    if num_samples_per_config == 0:
        return

    # check if all configs have the same number of data points
    # for i in range(num_configs):
    #    assert  len(metrics_list[i]) == num_samples_per_config

    # extract metric names
    metric_names = []
    for _, episode_metrics in metrics_list[0].items():
        for metric_name, _ in episode_metrics.items():
            metric_names.append(metric_name)
        break

    # build data for plotting
    data_all_configs_all_metrics = {} # Dict[Dict[List[float]]]
    for config_index in range(0, num_configs):
        data_per_config_all_metrics = {} # Dict[List[float]]
        metrics_per_config = metrics_list[config_index]
        for metric_name in metric_names:
            data_per_config_per_metric = []
            for _, dict_of_metrics in metrics_per_config.items():
                if (
                    not np.isnan(dict_of_metrics[metric_name])
                    and not np.isinf(dict_of_metrics[metric_name])
                ):
                    data_per_config_per_metric.append(dict_of_metrics[metric_name])
            data_per_config_all_metrics[metric_name] = data_per_config_per_metric
        data_all_configs_all_metrics[config_names[config_index]] = data_per_config_all_metrics

    # plot histograms
    for metric_name in metric_names:
        fig, axes = plt.subplots(
            int(np.ceil(num_configs/2)),
            2,
            sharey=True,
            figsize=(12.8, 9.6))
        axes_flattened = axes.ravel()
        for config_index in range(0, num_configs):
            config_name = config_names[config_index]
            # create histogram per metric, per config
            data_to_plot = data_all_configs_all_metrics[config_name][metric_name]
            if (
                metric_name == NumericalMetrics.DISTANCE_TO_GOAL
                or metric_name == metric_name == NumericalMetrics.SPL
                or metric_name == NumericalMetrics.NUM_STEPS
            ):
                axes_flattened[config_index].hist(
                    data_to_plot,
                    bins=50
                )
            else:
                raise NotImplementedError
            axes_flattened[config_index].set_title(config_name, fontsize=18)
        # set common x and y label. Code adapted from
        # https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
        fig.text(0.5, 0.04, f"{metric_name.value} {resolve_metric_unit(metric_name)}", ha="center", fontsize=22)
        fig.text(0.04, 0.5, "number of episodes", va="center", rotation="vertical", fontsize=22)
        plt.xticks(rotation=0)
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.6)
        fig.savefig(f"{plot_dir}/{metric_name}-{num_configs}_{configs_or_seeds}.png")
        plt.close(fig)


def observations_to_image_for_roam(
    observation: Dict,
    info: Dict,
    max_depth: float,
) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step(). Modified upon
    habitat.utils.visualizations.observations_to_image().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().
        max_depth: max depth reading of the depth sensor.

    Returns:
        generated image of a single frame.
    """
    egocentric_view_l: List[np.ndarray] = []
    if "rgb" in observation:
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view_l.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        depth_map = observation["depth"].squeeze() * (255.0 / max_depth)
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view_l.append(depth_map)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation:
        rgb = observation["imagegoal"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view_l.append(rgb)

    assert len(egocentric_view_l) > 0, "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view_l, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if "top_down_map_for_roam" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map_for_roam"], egocentric_view.shape[0]
        )
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)
    return frame


def visualize_running_times_with_bar_plots(
    running_times: List[float], config_names: List[str], plot_dir: str
):
    r"""
    Visualize running times from multiple experiments as bar charts. Save the
    plot to <plot_dir>/running_time_across_configs.png.
    :param running_times: running times from a sequence of experiments.
    :param config_names: names of experiment configs, should be in same order
        as `running_times`
    :param plot_dir: directory to save the plot
    """
    # precondition check
    assert len(running_times) == len(config_names)

    fig = plt.figure(figsize=(12.8, 9.6))
    data = {}
    data["config"] = config_names
    data["running_time"] = running_times
    df = pd.DataFrame(data)
    sns.set(font_scale=1.2, style="white")
    ax = sns.barplot(x="config", y="running_time", data=df)
    # add bar labels;
    # solution from https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
    # TODO: upgrade matplotlib to 3.4.0
    # ax.bar_label(ax.containers[0])
    ax.set_xlabel("configurations", fontsize=22)
    plt.xticks(rotation=0)
    ax.set_ylabel("running_time (hours)", fontsize=22)
    fig.savefig(f"{plot_dir}/running_time-{len(config_names)}_configurations.png")
    plt.close(fig)


def visualize_pairwise_percentage_diff_of_metrics(
    pairwise_diff_dict_of_metrics: Dict[str, Dict[str, float]],
    config_names: List[str],
    diff_in_percentage: bool,
    plot_dir: str,
):
    r"""
    Visualize pair-wise difference in metrics across multiple configs. Save
    the plot to plot_dir/<metric_name>-pairwise_diff.png, where <metric_name>
    is for eg. "spl".
    :param pairwise_diff_dict_of_metrics: pair-wise difference in metrics between
        two experiment configs.
    :param config_names: list of names of two experiment configs
    :param diff_in_percentage: if the pair-wise difference is computed as percentage
        or not
    :param plot_dir: directory to save the plots
    """
    # precondition check
    num_configs = len(config_names)
    assert num_configs == 2

    # return if no data
    if num_configs == 0:
        return
    num_samples = len(pairwise_diff_dict_of_metrics)
    if num_samples == 0:
        return

    # extract metric names
    metric_names = []
    for _, episode_metrics in pairwise_diff_dict_of_metrics.items():
        for metric_name, _ in episode_metrics.items():
            metric_names.append(metric_name)
        break

    # build dataframe
    data = {}
    data["compared configs"] = np.ndarray((num_samples,), dtype=object)
    for metric_name in metric_names:
        if diff_in_percentage:
            data[f"{metric_name} difference (%)"] = np.ndarray((num_samples,))
        else:
            data[f"{metric_name} difference"] = np.ndarray((num_samples,))
    # populate each array
    total_sample_count = 0
    for _, episode_metrics in pairwise_diff_dict_of_metrics.items():
        # register a new sample
        data["compared configs"][
            total_sample_count
        ] = f"configs: {config_names[1]} vs {config_names[0]}"
        for metric_name in metric_names:
            if diff_in_percentage:
                data[f"{metric_name} difference (%)"][
                    total_sample_count
                ] = episode_metrics[metric_name]
            else:
                data[f"{metric_name} difference"][total_sample_count] = episode_metrics[
                    metric_name
                ]
        total_sample_count += 1
    df = pd.DataFrame(data)

    # drop invalid samples
    # code adapted from piRSquared's work on
    # https://stackoverflow.com/questions/45745085/python-pandas-how-to-remove-nan-and-inf-values
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    # create box-and-strip plot for each metric
    for metric_name in metric_names:
        fig = plt.figure(figsize=(12.8, 9.6))
        ax = fig.add_subplot(111)
        if diff_in_percentage:
            sns.boxplot(
                x="compared configs", y=f"{metric_name} difference (%)", data=df, ax=ax
            )
            sns.stripplot(
                x="compared configs",
                y=f"{metric_name} difference (%)",
                data=df,
                color=".25",
                size=2,
                ax=ax,
            )
            fig.savefig(
                f"{plot_dir}/{metric_name}-{config_names[1]}_vs_{config_names[0]}_%.png"
            )
        else:
            sns.boxplot(
                x="compared configs", y=f"{metric_name} difference", data=df, ax=ax
            )
            sns.stripplot(
                x="compared configs",
                y=f"{metric_name} difference",
                data=df,
                color=".25",
                size=2,
                ax=ax,
            )
            fig.savefig(
                f"{plot_dir}/{metric_name}-{config_names[1]}_vs_{config_names[0]}.png"
            )
        plt.close(fig)
