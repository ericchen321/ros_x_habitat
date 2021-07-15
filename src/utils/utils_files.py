import csv

def load_seeds_from_file(seed_file_path):
    seeds = []
    with open(seed_file_path, newline="") as csv_file:
        csv_lines = csv.reader(csv_file)
        for line in csv_lines:
            seeds.append(int(line[0]))
    return seeds