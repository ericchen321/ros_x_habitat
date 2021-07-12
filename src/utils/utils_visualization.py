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
        scene_id: scene ID of the episode.
        seeds: seeds used to initialize the agents.
        maps: maps produced by the agents navigating in np.ndarray format. Should be in
            the same order as seeds.
        map_dir: directory to store the map
    """
    plt.clf()
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

    plt.title(f"episode={episode_id}, scene={scene_id}")
    plt.savefig(
        f"{map_dir}/episode={episode_id}-scene={os.path.basename(scene_id)}.png"
    )
    plt.clf()


def generate_box_plots(
    metrics_list: List[List[Dict[str, float]]],
    seeds: List[int],
    plot_dir: str,
):
    r"""
    Generate box plots from metrics and seeds. Requires same metrics collected
    from all seeds. Save the plots to plot_dir/<metric_name>-<n>_seeds.png,
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
    for i in range(num_seeds):
        assert len(metrics_list[i]) == num_samples_per_seed

    # extract metric names
    metric_names = []
    for metric_name, _ in metrics_list[0][0].items():
        metric_names.append(metric_name)

    # build dataframe
    data = {}
    total_num_samples = num_samples_per_seed * num_seeds
    data["seed"] = np.ndarray((total_num_samples,))
    for metric_name in metric_names:
        data[metric_name] = np.ndarray((total_num_samples,))
    # populate each array
    total_sample_count = 0
    for seed_index in range(num_seeds):
        for sample_count in range(num_samples_per_seed):
            # register a new sample
            data["seed"][total_sample_count] = seeds[seed_index]
            for metric_name in metric_names:
                data[metric_name][total_sample_count] = metrics_list[seed_index][
                    sample_count
                ][metric_name]
            total_sample_count += 1
    df = pd.DataFrame(data)

    # create box-and-warm plot for each metric
    for metric_name in metric_names:
        plt.clf()
        plots = sns.boxplot(x="seed", y=metric_name, data=df)
        plots = sns.swarmplot(x="seed", y=metric_name, data=df, color=".25")
        plt.savefig(f"{plot_dir}/{metric_name}-{num_seeds}_seeds.png")

    plt.clf()
