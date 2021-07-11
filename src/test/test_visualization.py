import unittest
from src.utils.utils_visualization import generate_box_plots
import os


class TestVisualization(unittest.TestCase):
    def test_generate_box_plots_multiple_seeds(self):
        seeds = [248, 3999]
        sample_a = {"spl": 0.8, "success": 1.0}
        sample_b = {"spl": 0.2, "success": 1.0}
        sample_c = {"spl": 0.0, "success": 0.0}
        metrics_list = []
        for seed in seeds:
            metrics_list_per_seed = [sample_a, sample_b, sample_c]
            metrics_list.append(metrics_list_per_seed)

        plot_dir = "plots/test_generate_box_plots/"
        try:
            os.mkdir(plot_dir)
        except FileExistsError:
            pass
        generate_box_plots(metrics_list, seeds, plot_dir)


if __name__ == "__main__":
    unittest.main()
