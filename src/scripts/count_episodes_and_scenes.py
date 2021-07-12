# count the number of episodes and scenes in a dataset
# Arguments:
#   Path to the dataset's .json file

import json
import sys

if __name__ == "__main__":
    # extract args
    json_path = sys.argv[1]

    json_file = open(json_path)
    data = json.load(json_file)

    # count episodes and scenes
    count_episodes = 0
    scene_list = []
    for episode in data["episodes"]:
        count_episodes += 1
        scene_id = episode["scene_id"]
        if scene_id not in scene_list:
            scene_list.append(scene_id)
    print(f"Number of episodes: {count_episodes}")
    print(f"Number of scenes: {len(scene_list)}")
