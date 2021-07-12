import unittest
from src.utils.utils_visualization import generate_box_plots
import os


class TestVisualization(unittest.TestCase):
    def test_generate_box_plots_multiple_seeds(self):
        seeds = range(10)
        sample_a = {
            "spl": 0.8,
            "success": 1.0}
        sample_b = {
            "spl": 0.2,
            "success": 1.0
        }
        sample_c = {
            "spl": 0.0,
            "success": 0.0
        }
        metrics_list = []
        for seed in seeds:
            dict_of_metrics = {
                "sample_a": sample_a,
                "sample_b": sample_b,
                "sample_c": sample_c
            }
            metrics_list.append(dict_of_metrics)
        
        plot_dir = "plots/test_generate_box_plots/"
        try:
            os.mkdir(plot_dir)
        except FileExistsError:
            pass
        generate_box_plots(metrics_list, seeds, plot_dir)


if __name__ == "__main__":
    unittest.main()