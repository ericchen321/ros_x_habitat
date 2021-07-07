# compute metrics from per-episode log files in given directory
# Arguments:
#   Path to directory containing log files

import glob
import sys


def extract_metric(log_filename, lin_num, splitter):
    log_file = open(log_filename, "r")
    metric_line = log_file.readlines()[lin_num]
    metric = float(metric_line.split(splitter)[1])
    log_file.close()
    return metric


if __name__ == "__main__":
    # extract args
    log_dir = sys.argv[1]

    # get log filenames
    log_filenames = []
    for log_filename in glob.glob(f"{log_dir}/*.log"):
        log_filenames.append(log_filename)

    agg_metrics = {
        "distance_to_goal": 0.0,
        "success": 0.0,
        "spl": 0.0,
        "agent_time": 0.0,
        "sim_time": 0.0,
        "num_steps": 0.0,
    }
    for log_filename in log_filenames:
        # extract metrics
        line_index = 2
        for metric_name, _ in agg_metrics.items():
            agg_metrics[metric_name] += extract_metric(
                log_filename, line_index, f"{metric_name},"
            )
            line_index += 1

    num_episodes = len(log_filenames)
    print(f"Computed metrics from {num_episodes} episodes")
    for k, v in agg_metrics.items():
        avg_v = v / num_episodes
        print(f"{k}: {avg_v:.3f}")
