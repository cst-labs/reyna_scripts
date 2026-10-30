import importlib.util
import math
import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_ktas_kate_neighbors.py"
SPEC = importlib.util.spec_from_file_location("analyze_ktas_kate_neighbors", MODULE_PATH)
analyze_ktas_kate_neighbors = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analyze_ktas_kate_neighbors
SPEC.loader.exec_module(analyze_ktas_kate_neighbors)


class KtasKateNeighborAnalysisTests(unittest.TestCase):
    def test_rounded_mean_prediction_uses_half_up_rounding(self):
        self.assertEqual(analyze_ktas_kate_neighbors.rounded_mean_prediction([2, 3]), 3)
        self.assertEqual(analyze_ktas_kate_neighbors.rounded_mean_prediction([1, 1, 2]), 1)
        self.assertEqual(analyze_ktas_kate_neighbors.rounded_mean_prediction([4, 5]), 5)

    def test_compute_metrics_for_subset_matches_expected_values(self):
        truth = pd.Series([1, 2, 4])
        predictions = pd.Series([1, 3, 3])
        metrics = analyze_ktas_kate_neighbors.compute_metrics_for_subset(truth, predictions)

        self.assertEqual(metrics["count"], 3)
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertAlmostEqual(metrics["delta"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt((0 ** 2 + 1 ** 2 + (-1) ** 2) / 3))

    def test_build_metrics_records_splits_overall_and_by_true_label(self):
        predictions_df = pd.DataFrame(
            {
                "true_ktas": [1, 1, 2, 3, 4, 5],
                "predicted_k5": [1, 2, 2, 4, 4, 5],
                "predicted_k10": [1, 1, 3, 3, 5, 4],
            }
        )

        records = analyze_ktas_kate_neighbors.build_metrics_records(predictions_df, [5, 10])
        self.assertEqual(len(records), 12)

        by_key = {(record["k"], record["group"]): record for record in records}
        self.assertEqual(by_key[(5, "overall")]["count"], 6)
        self.assertEqual(by_key[(5, "1")]["count"], 2)
        self.assertEqual(by_key[(5, "2")]["count"], 1)
        self.assertAlmostEqual(by_key[(5, "1")]["mae"], 0.5)
        self.assertAlmostEqual(by_key[(10, "4")]["delta"], 1.0)
        self.assertAlmostEqual(by_key[(10, "5")]["delta"], -1.0)
        self.assertAlmostEqual(by_key[(10, "overall")]["rmse"], math.sqrt(3 / 6))


if __name__ == "__main__":
    unittest.main()
