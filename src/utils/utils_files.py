import csv

def load_seeds_from_file(seed_file_path):
    r"""
    Load random seeds from file identified by seed_file_path.
    :param seed_file_path: path to a .csv file of seeds
    :returns: a list of the random seeds from the given file
    """
    seeds = []
    with open(seed_file_path, newline="") as csv_file:
        csv_lines = csv.reader(csv_file)
        for line in csv_lines:
            seeds.append(int(line[0]))
    return seeds

def load_episode_identifiers(episodes_to_visualize_file_path):
    r"""
    Load episode identifiers from the given file. Each episode must be specified
    by an episode ID and a scene ID.
    :param episodes_to_visualize_file_path: path to a .csv file of episode
        identifiers
    :returns: a list of episode ID's and a list of scene ID's. One-to-one
        correspondence
    """
    episode_ids = []
    scene_ids = []
    with open(episodes_to_visualize_file_path, newline="") as csv_file:
        csv_lines = csv.reader(csv_file)
        for line in csv_lines:
            episode_id = str(line[0])
            scene_id = str(line[1])
            assert episode_id is not None and episode_id != ""
            assert scene_id is not None and scene_id != ""
            episode_ids.append(episode_id)
            scene_ids.append(scene_id)

    return episode_ids, scene_ids