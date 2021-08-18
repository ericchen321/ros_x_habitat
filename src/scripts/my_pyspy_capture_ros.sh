#!/bin/bash

sros
export PYTHONPATH=$PYTHONPATH:/home/lci-user/Desktop/workspace/src/ros_x_habitat/
conda activate habitat2.0
py-spy record --idle --function --native --subprocesses --rate 200 --output pyspy_profiles/pyspy_profile_ros.speedscope --format speedscope -- python src/test/test_habitat_ros/test_habitat_ros_agent_node_discrete.py

