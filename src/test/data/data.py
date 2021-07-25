import numpy as np
import os


class TestHabitatROSData:
    # for test_habiat_ros_<agent/env>_node_discrete
    test_acts_and_obs_task_config = "configs/pointnav_rgbd_val.yaml"
    test_acts_and_obs_discrete_episode_id = "49"
    test_acts_and_obs_discrete_scene_id = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    test_acts_and_obs_discrete_num_obs = 47
    test_acts_and_obs_discrete_obs_rgb = []
    test_acts_and_obs_discrete_obs_depth = []
    test_acts_and_obs_discrete_obs_ptgoal_with_comp = []
    test_acts_and_obs_discrete_acts = []
    for i in range(0, test_acts_and_obs_discrete_num_obs):
        test_acts_and_obs_discrete_obs_rgb.append(
            np.load(
                f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/test/data/obs/rgb-{test_acts_and_obs_discrete_episode_id}-{os.path.basename(test_acts_and_obs_discrete_scene_id)}-{i}.npy"
            )
        )
        test_acts_and_obs_discrete_obs_depth.append(
            np.load(
                f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/test/data/obs/depth-{test_acts_and_obs_discrete_episode_id}-{os.path.basename(test_acts_and_obs_discrete_scene_id)}-{i}.npy"
            )
        )
        test_acts_and_obs_discrete_obs_ptgoal_with_comp.append(
            np.load(
                f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/test/data/obs/pointgoal_with_gps_compass-{test_acts_and_obs_discrete_episode_id}-{os.path.basename(test_acts_and_obs_discrete_scene_id)}-{i}.npy"
            )
        )
        test_acts_and_obs_discrete_acts.append(
            np.load(
                f"/home/lci-user/Desktop/workspace/src/ros_x_habitat/src/test/data/acts/action-{test_acts_and_obs_discrete_episode_id}-{os.path.basename(test_acts_and_obs_discrete_scene_id)}-{i}.npy"
            )
        )

    # for test_habiat_ros_<agent/env>_node_continuous
    #TODO:

    # for test_habitat_ros_evaluator_<discrete/continuous>
    test_evaluator_num_episodes = 1
    test_evaluator_episode_id_request = "test-episode-id-request"
    test_evaluator_episode_id_response = "test-episode-id-response"
    test_evaluator_scene_id = "test-scene-id"
    test_evaluator_log_dir = "logs/test_habitat_ros_evaluator_discrete/"
    test_evaluator_agent_seed = 7
    test_evaluator_config_paths = "configs/pointnav_rgbd_val.yaml"
    test_evaluator_input_type = "rgbd"
    test_evaluator_model_path = "data/checkpoints/v2/gibson-rgbd-best.pth"
    test_evaluator_distance_to_goal = 0.123456
    test_evaluator_success = 1.0
    test_evaluator_spl = 0.987654