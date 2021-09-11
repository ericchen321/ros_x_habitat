ros_x_habitat
==============================

A ROS package to bridge [AI Habitat](https://aihabitat.org/) with the ROS ecosystem.

## Outline
   1. [Motivation](#motivation)
   1. [Installation](#installation)
   1. [System Architecture](#architecture)
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

## Installation
1. Install Ubuntu 20.04 + ROS Noetic.
2. Install [Anaconda](https://www.anaconda.com/). 
3. Install [Habitat Sim](https://github.com/facebookresearch/habitat-sim) following the official instructions. Note that
    * We suggest you to use the version tagged `0.2.0`. Any versions below this is not supported;
    * We also suggest install by building from the source;
    * Install with Bullet Physics;
    * If you have an NVIDIA card, we suggest you to install with CUDA.
4. Install [Habitat Lab](https://github.com/facebookresearch/habitat-lab) in the same conda environment following the official instructions. Note that
    * We suggest you to use the version tagged `0.2.0`. Any versions below this is not supported;
    * In addition to the core of Habitat Lab, also install `habitat_baselines` and other required packages.
5. Install the following ROS packages:
   * `ros-noetic-depthimage-to-laserscan`
   * `ros-noetic-laser-scan-matcher`
   * `ros-noetic-rtabmap-ros`
   * `ros-noetic-joy`
6. Clone the repo to your catkin workspace.
7. Compile the package by calling `catkin_make`.
8. Export the repo's directory to `$PYTHONPATH`:
   ```
   export PYTHONPATH=$PYTHONPATH:<path-to-the-root-directory-of-the-repo>
   ```

## System Architecture
Coming soon

## Examples
Coming soon

## Cite Our Work
Coming soon

## License
Coming soon

## Acknowledgments

## References
Coming soon