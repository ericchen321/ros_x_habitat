#!/bin/bash

sros
export PYTHONPATH=$PYTHONPATH:/home/lci-user/Desktop/workspace/src/ros_x_habitat/
conda activate habitat2.0
py-spy record --idle --function --native --subprocesses --rate 200 --output pyspy_profiles/pyspy_profile_no_ros.speedscope --format speedscope -- python src/scripts/eval_and_vis_habitat.py --input-type rgbd --model-path data/checkpoints/v2/gibson-rgbd-best.pth --task-config configs/pointnav_rgbd_val.yaml --episode-id 48 --scene-id data/scene_datasets/habitat-test-scenes/van-gogh-room.glb  --seed-file-path seeds/seed=7.csv --log-dir=logs/test_habitat_aug17/ --make-maps --map-dir habitat_maps/test_habitat_aug17/ --make-plots --plot-dir metric_plots/test_habitat_aug17/

