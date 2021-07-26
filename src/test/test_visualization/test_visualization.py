import unittest
from src.utils.utils_files import load_seeds_from_file
from src.utils.utils_visualization import generate_box_plots
from src.constants.constants import NumericalMetrics
import os


class TestVisualization(unittest.TestCase):
    def test_generate_box_plots_multiple_seeds(self):
        seeds = load_seeds_from_file("seeds/10_seeds.csv")
        sample_a = {NumericalMetrics.SPL: 0.8, NumericalMetrics.SUCCESS: 1.0}
        sample_b = {NumericalMetrics.SPL: 0.2, NumericalMetrics.SUCCESS: 1.0}
        sample_c = {NumericalMetrics.SPL: 0.0, NumericalMetrics.SUCCESS: 0.0}
        metrics_list = []
        for seed in seeds:
            dict_of_metrics = {
                "sample_a": sample_a,
                "sample_b": sample_b,
                "sample_c": sample_c
            }
            metrics_list.append(dict_of_metrics)
        
        plot_dir = "metric_plots/test_generate_box_plots/"
        os.makedirs(name=plot_dir, exist_ok=True)
        generate_box_plots(metrics_list, seeds, plot_dir)


if __name__ == "__main__":
    unittest.main()
