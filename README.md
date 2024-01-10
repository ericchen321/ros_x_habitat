ros_x_habitat
==============================

A ROS package to bridge [AI Habitat](https://aihabitat.org/) with the ROS ecosystem.

https://user-images.githubusercontent.com/22675300/209244722-2655536c-1104-4554-92f5-bb05cec9523c.mp4

## Outline
   1. [Motivation](#motivation)
   1. [System Architecture](#architecture)
   1. [Installation](#installation)
   1. [Examples](#examples)
   1. [Cite Our Work](#cite-our-work)
   1. [License](#license)
   1. [Acknowledgments](#acknowledgments)
   1. [References](#references-and-citation)

## Motivation
The package allows roboticists to
   * Navigate an AI Habitat agent within photorealistic scenes simulated by Habitat Sim through ROS;
   * Connecting a ROS-based planner with Habitat Sim;
   * Connecting an AI Habitat agent with a ROS-bridged simulator ([Gazebo](http://gazebosim.org/) for example) through ROS.
   * Leverage Habitat Sim's photorealistic and physically-realistic simulation capability through ROS.


## System Architecture
``ros_x_habitat`` exists as a collection of ROS nodes, topics and services. For simplicity, we have omitted components not essential to the interface's operation:

![](docs/architecture_enlarged.png)

## Installation
1. Install Ubuntu 20.04 + ROS Noetic.
2. Install [Anaconda](https://www.anaconda.com/) and create a conda environment for this project.
   ```
   conda create -n rosxhab python=3.6.13 cmake=3.14.0
   conda activate rosxhab
   pip install --upgrade pip
   ```
3. Install Pytorch 1.10.2:
   ```
   # if you have an NVIDIA GPU + driver properly installed, install CUDA-compiled torch:
   pip install torch==1.10.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
   # otherwise, install the CPU-compiled torch:
   pip install torch==1.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
   ```
4. Install other dependent packages:
   ```
   pip install -r requirements.txt
   ```
5. Install [Habitat Sim](https://github.com/facebookresearch/habitat-sim) version `0.2.0`. 
    * Here we show how to install it from conda in the environment you just created:
      ```
      cd <path to Habitat Sim's root directory>
      conda install habitat-sim=0.2.0 withbullet -c conda-forge -c aihabitat
      ```
    * If the above conda install for some reasons returns an error, you can download [the .tar.gz](https://anaconda.org/aihabitat/habitat-sim/0.2.0/download/linux-64/habitat-sim-0.2.0-py3.6_bullet_linux_bfafd7934df465d79d807e4698659e2c20daf57d.tar.bz2) file and install it locally:
      ```
      conda install --use-local linux-64_habitat-sim-0.2.0-py3.6_bullet_linux_bfafd7934df465d79d807e4698659e2c20daf57d.tar.bz2
      ```
    * If installing from conda still doesn't work, you can also try building from source with Bullet Physics and CUDA support (if you have an NVIDIA card).
6. Install [Habitat Lab](https://github.com/facebookresearch/habitat-lab) version `0.2.0` along with `habitat-baselines` in the same conda environment.
   ```
   git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
   cd habitat-lab
   git reset --hard adf90
   pip install -r requirements.txt
   python setup.py develop --all # install habitat and habitat_baselines
   ```
7. Install the following ROS packages:
   * `ros-noetic-depthimage-to-laserscan`
   * `ros-noetic-laser-scan-matcher`
   * `ros-noetic-rtabmap-ros`
   * `ros-noetic-hector-slam`
   * `ros-noetic-joy`
   * `ros-noetic-turtlebot3-gazebo`
   * `ros-noetic-turtlebot3-bringup`
   * `ros-noetic-turtlebot3-navigation`
8. Clone the repo to the `src/` directory under your catkin workspace.
9. Compile the package by calling `catkin_make`.

## Examples
Here we outline steps to reproduce experiments from our paper. 

### Environment Setup
To set up your bash environment before you run any experiment, do the following:
1. Activate your conda environment (if you are running any script/node from this codebase).
2. Export the repo's directory to `$PYTHONPATH`:
   ```
   export PYTHONPATH=$PYTHONPATH:<path to the root directory of this repo>
   ```
   In fact for convenience, you can create a command in your `$HOME/.bashrc`:
   ```
   alias erosxhab="export PYTHONPATH=$PYTHONPATH:<path to the root directory of this repo>"
   ```
3. Source ROS-related environment variables. Similarly we can create a command to do this:
   ```
   alias sros="source /opt/ros/noetic/setup.bash && source <path to catkin_ws/devel>/setup.sh"
   ```

### Navigating Habitat Agent in Habitat Sim without ROS (+/-Physics, -ROS)
We can attempt to reproduce experiments from the [Habitat v1 paper](https://arxiv.org/abs/1904.01201) by evaluating a Habitat agent's performance in a MatterPort3D test scenario. Note that unlike what the authors did in the paper, we used the following experimental configurations:
   * Agents: [Habitat v2's RGBD agent](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/habitat_baselines_v2.zip) since the v1 agents are no longer compatible with Habitat v2.
   * Test episodes and scenes: MatterPort3D test episodes only. To get episode definitions, download from [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip); to download the complete MP3D scene assets, please follow instructions under Section "Dataset Download" from [here](https://niessner.github.io/Matterport/).
   * Also we evaluated the agent in physics-enabled Habitat Sim v2.

To run an evaluation, follow these steps:
   1. Create directory `data/` under the project's root directory. Then create the following directories:
      * `data/checkpoints/v2/`,
      * `data/datasets/pointnav/mp3d/v1/`,
      * `data/objects/`,
      * `data/scene_datasets/mp3d/`.
   2. Download [`default.physics_config.json`](https://github.com/facebookresearch/habitat-sim/blob/v0.2.0/data/default.physics_config.json) and place it under `data/`.
   3. Extract the v2 agents (`.pth` files) to `data/checkpoints/v2/`. Extract the MP3D episode definitions to `data/datasets/pointnav/mp3d/v1/` (then under `v1/` you should see directory `test/`, `train/`, etc). Extract the MP3D scene files to `data/scene_datasets/mp3d/`.
   4. Select an experiment configuration file from `configs/`. Our configurations are coded by numbers:
      * Setting 2: -Physics, -ROS;
      * Setting 4: +Physics, -ROS.
   5. Select from `seeds/` a seed file or create one of your own for your experiment. The seed is used for initializing the Habitat agent.
   6. Run the following command to evaluate the agent over the test episodes while producing top-down maps and box plots to visualize metrics:
      ```
      python src/scripts/eval_and_vis_habitat.py \
      --input-type rgbd \
      --model-path data/checkpoints/v2/gibson-rgbd-best.pth \
      --task-config <path to config file> \
      --episode-id <ID of last episode evaluated; -1 to evaluate from start> \
      --seed-file-path <path to seed file> \
      --log-dir <path to dir storing evaluation logs> \
      --make-maps \
      --map-dir <path to dir storing top-down maps> \
      --make-plots \
      --plot-dir <path to dir storing metric plots>
      ```

### Navigating Habitat Agent in Habitat Sim with ROS (+/-Physics, +ROS)
Under this mode we run a Habitat Agent still inside Habitat Sim but through our interface.

   1. Select an experiment configuration file from `configs/`. Our configurations are coded by numbers:
      * Setting 3: -Physics, +ROS;
      * Setting 5: +Physics, +ROS.
   2. Select a seed file (as above).
   3. Run:
      ```
      python src/scripts/eval_habitat_ros.py \
      --input-type rgbd \
      --model-path data/checkpoints/v2/gibson-rgbd-best.pth \
      --task-config configs/setting_5_configs/pointnav_rgbd-mp3d_with_physics.yaml \
      --episode-id <ID of last episode evaluated; -1 to evaluate from start> \
      --seed-file-path <seed file path> \
      --log-dir= <log dir path>
      ```

### Navigating Habitat Agent in Gazebo
Here we demonstrate steps to posit a Habitat agent embodied on a TurtleBot in a Gazebo-simulated environment, and render sensor data with RViz.
   1. Define the model for the TurtleBot (here we use "Waffle"):
      ```
      export TURTLEBOT3_MODEL="waffle" 
      ```
   2. Launch Gazebo and RViz:
      ```
      roslaunch turtlebot3_gazebo turtlebot3_house.launch
      roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch 
      ```
   3. Launch the agent node:
      ```
      python src/nodes/habitat_agent_node.py \
      --node-name agent_node \
      --input-type rgbd \
      --model-path data/checkpoints/v2/gibson-rgbd-best.pth 
      ```
      Then reset the agent in a separate window:
      ```
      rosservice call /ros_x_habitat/agent_node/reset_agent "{reset: 0, seed: 0}" 
      ```
   4. Launch the converter nodes:
      ```
      python src/nodes/habitat_agent_to_gazebo.py
      python src/nodes/gazebo_to_habitat_agent.py \
      --pointgoal-location <coordinates>
      ```
   5. In RViz, subscribe to topic `/camera/depth/image_raw` and `/camera/rgb/image_raw` to render observations from the RGB and the depth sensor.

### Navigating ROS Agent in Habitat Sim
Here we outline steps to 1) control, via a joystick, a ROS agent with RGBD sensors to roam and map a Habitat Sim-simulated scene; 2) control a planner from the `move_base` package to navigate to a manually-set goal position.
   1. Repeat Step 1 and 2 from [here](#Navigating-Habitat-Agent-in-Habitat-Sim-without-ROS-(+/-Physics,--ROS)) to download MP3D scene assets and episode definitions.
   2. Download Habitat's test object assets into `data/objects/` by running this command from Habitat Sim (more instructions from [here](https://github.com/facebookresearch/habitat-sim/tree/v0.2.0#testing)):
      ```
      python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path <path to ros_x_habitat/data/objects/>
      ```
   3. Start `roscore`.
   4. Select a configuration file from `configs/roam_configs/`, e.g. `pointnav_rgbd_roam_mp3d_test_scenes.yaml`. You might want to change `VIDEO_DIR` to another directory of your liking.
   5. Run this command to initialize a Habitat sim environment and a joystick-controlled roamer:
      ```
      python src/scripts/roam_with_joy.py \
      --hab-env-config-path <config file path> \
      --episode-id <ID of episode to roam inside> \
      --scene-id <path to the episode's scene file, e.g. data/scene_datasets/mp3d/2t7WUuJeko7/2t7WUuJeko7.glb> \
      --video-frame-period <number of continuous steps for each frame recorded>
      ```
      Note that the environment node won't initialize until it makes sure some other node is listening to the topics on which it publishes sensor readings: `/rgb`, `/depth`, `/pointgoal_with_gps_compass`. The script will fire up `image_view` nodes to listen to RGB/Depth readings but you need to fire up a dummy node yourself to listen to Pointgoal/GPS info. You can do this with the command
      ```
      rostopic echo /pointgoal_with_gps_compass
      ```
   6. Next, we map the scene with `rtabmap_ros`. Run
      ```
      roslaunch launch/rtabmap_mapping.launch
      ```
      Move the joystick-controlled agent around to map the environment. Save the map to somewhere. We also have some pre-built maps of Habitat test scenes and Matterport 3D environments under `maps/`.
   
Next, we use a planner from the `move_base` package to navigate in the scene with the help of that
map we just built.
   
   7. Shut down every node you launched from Step 1 to 6 above.
   8. Repeat Step 3 to 5.
   9. Start the planner: run
      ```
      roslaunch launch/move_base.launch
      ```
      Make sure `map_file_path` and `map_file_base_name` have been set correctly before you run. The launcher file should also start an `rviz` session which allows you to 1) specify the goal point and 2) visualize RGB/depth sensor readings. 

## Tested Platforms
The experiments were run on a desktop with  i7-10700K CPU, 64 GB of RAM, and an NVIDIA
RTX 3070 GPU. We also tested the experiments on a desktop with 32 GB of RAM and an NVIDIA GT 1030 GPU.

## Cite Our Work
If you are interested in using ``ros_x_habitat`` for your own research, please cite [our CRV 2022 paper](https://arxiv.org/abs/2109.07703):
```
@INPROCEEDINGS{9867069,  
author={Chen, Guanxiong and Yang, Haoyu and Mitchell, Ian M.},  
booktitle={2022 19th Conference on Robots and Vision (CRV)},  
title={ROS-X-Habitat: Bridging the ROS Ecosystem with Embodied AI},  
year={2022},  volume={},  number={},  pages={24-31},  
doi={10.1109/CRV55824.2022.00012}}
```

## License
This work is under the [Creative Commons](https://creativecommons.org/licenses/by/4.0/) CC BY 4.0 License.

## Acknowledgments
We would like to thank [Bruce Cui](https://github.com/brucecui97) from the Department of Mechanical Engineering at UBC for his initial work on ``ros_x_habitat``. Also,  we would like to appreciate the AI Habitat team from Facebook AI Research, including Prof. Dhruv Batra, Alex Clegg, Prof. Manolis Savva, and Erik Wijmans for their generous support throughout our development process.

## References
1. [Habitat: A Platform for Embodied AI Research.](https://arxiv.org/abs/1904.01201) Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, Dhruv Batra. IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
2. [Habitat 2.0: Training Home Assistants to Rearrange their Habitat.](https://arxiv.org/abs/2106.14405) Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, Dhruv Batra. arXiv preprint arXiv:2106.14405, 2021.
3. [ROS-X-Habitat: Bridging the ROS Ecosystem with Embodied AI.](https://arxiv.org/abs/2109.07703) Guanxiong Chen, Haoyu Yang, Ian Mitchell. arXiv preprint arXiv:2109.07703, 2021.
