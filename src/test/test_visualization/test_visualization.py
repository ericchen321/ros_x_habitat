import unittest
from src.utils.utils_files import load_seeds_from_file
from src.utils.utils_visualization import (
    visualize_variability_due_to_seed_with_box_plots,
)
from src.constants.constants import NumericalMetrics
import os


class TestVisualization(unittest.TestCase):
    def test_visualize_variability_due_to_seed_with_box_plots_multiple_seeds(self):
        seeds = load_seeds_from_file("seeds/10_seeds.csv")
        sample_a = {NumericalMetrics.SPL: 0.8, NumericalMetrics.SUCCESS: 1.0}
        sample_b = {NumericalMetrics.SPL: 0.2, NumericalMetrics.SUCCESS: 1.0}
        sample_c = {NumericalMetrics.SPL: 0.0, NumericalMetrics.SUCCESS: 0.0}
        sample_d = {NumericalMetrics.SPL: float("nan"), NumericalMetrics.SUCCESS: 0.0}
        sample_e = {NumericalMetrics.SPL: float("inf"), NumericalMetrics.SUCCESS: 0.0}
        metrics_list = []
        for seed in seeds:
            dict_of_metrics = {
                "sample_a": sample_a,
                "sample_b": sample_b,
                "sample_c": sample_c,
                "sample_d": sample_d,
                "sample_e": sample_e,
            }
            metrics_list.append(dict_of_metrics)

        plot_dir = "metric_plots/test_visualize_variability_due_to_seed_with_box_plots/"
        os.makedirs(name=plot_dir, exist_ok=True)
        # we eye-ball check the generated plot for now
        visualize_variability_due_to_seed_with_box_plots(metrics_list, seeds, plot_dir)


if __name__ == "__main__":
    unittest.main()
