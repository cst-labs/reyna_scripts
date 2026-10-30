import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_ktas_pairwise_errors.py"
SPEC = importlib.util.spec_from_file_location("analyze_ktas_pairwise_errors", MODULE_PATH)
pairwise_error_analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pairwise_error_analysis
SPEC.loader.exec_module(pairwise_error_analysis)


class PairwiseErrorAnalysisTests(unittest.TestCase):
    def test_attach_pairwise_features_computes_gap_and_pair(self):
        df = pd.DataFrame(
            [
                {
                    "Predicted_Order": "A_BEFORE_B",
                    "Pair_Label": "A_BEFORE_B",
                    "A_KTAS_expert": 2,
                    "B_KTAS_expert": 4,
                },
                {
                    "Predicted_Order": "A_BEFORE_B",
                    "Pair_Label": "B_BEFORE_A",
                    "A_KTAS_expert": 4,
                    "B_KTAS_expert": 2,
                },
            ]
        )

        enriched = pairwise_error_analysis.attach_pairwise_features(df)

        self.assertListEqual(enriched["grade_gap"].tolist(), [2, 2])
        self.assertListEqual(enriched["grade_pair"].tolist(), ["2 vs 4", "2 vs 4"])
        self.assertListEqual(enriched["correct"].tolist(), [True, False])

    def test_parse_run_extracts_pairwise_model_name(self):
        path = Path(
            "results/Triage-KTAS-Pairwise/"
            "Triage-KTAS-Pairwise_KATE-PAIRWISE_qwen2.5:7b_6_shots_natural_0_999999_"
            "ktas_balanced12_pairwise.csv"
        )
        run = pairwise_error_analysis.parse_run(path)
        self.assertEqual(run.model, "qwen2.5:7b")

    def test_analyze_run_reports_gap_accuracy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            pd.DataFrame(
                [
                    {
                        "Predicted_Order": "A_BEFORE_B",
                        "Pair_Label": "A_BEFORE_B",
                        "A_KTAS_expert": 2,
                        "B_KTAS_expert": 3,
                    },
                    {
                        "Predicted_Order": "B_BEFORE_A",
                        "Pair_Label": "A_BEFORE_B",
                        "A_KTAS_expert": 2,
                        "B_KTAS_expert": 4,
                    },
                    {
                        "Predicted_Order": "A_BEFORE_B",
                        "Pair_Label": "A_BEFORE_B",
                        "A_KTAS_expert": 1,
                        "B_KTAS_expert": 5,
                    },
                ]
            ).to_csv(path, index=False)

            stats = pairwise_error_analysis.analyze_run(
                pairwise_error_analysis.RunRecord(model="demo", path=path)
            )

        gap_rows = {int(row["grade_gap"]): row for row in stats["by_gap"]}
        self.assertAlmostEqual(stats["overall"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(gap_rows[1]["accuracy"], 1.0)
        self.assertAlmostEqual(gap_rows[2]["accuracy"], 0.0)
        self.assertAlmostEqual(gap_rows[4]["accuracy"], 1.0)

    def test_build_average_grade_pair_table_aggregates_across_models(self):
        analyzed = [
            (
                pairwise_error_analysis.RunRecord(model="a", path=Path("a.csv")),
                {
                    "by_pair": [
                        {"urgent_grade": 1, "less_urgent_grade": 2, "pair": "1 vs 2", "support": 10, "accuracy": 0.8, "error_rate": 0.2},
                        {"urgent_grade": 2, "less_urgent_grade": 3, "pair": "2 vs 3", "support": 10, "accuracy": 0.6, "error_rate": 0.4},
                    ]
                },
            ),
            (
                pairwise_error_analysis.RunRecord(model="b", path=Path("b.csv")),
                {
                    "by_pair": [
                        {"urgent_grade": 1, "less_urgent_grade": 2, "pair": "1 vs 2", "support": 10, "accuracy": 0.7, "error_rate": 0.3},
                        {"urgent_grade": 2, "less_urgent_grade": 3, "pair": "2 vs 3", "support": 10, "accuracy": 0.9, "error_rate": 0.1},
                    ]
                },
            ),
        ]

        table = pairwise_error_analysis.build_average_grade_pair_table(analyzed)

        self.assertIn("| Grade Pair | Models | Support / Model | Average Accuracy | Min Accuracy | Max Accuracy |", table)
        self.assertIn("| 1 vs 2 | 2 | 10 | 75.0% | 70.0% | 80.0% |", table)
        self.assertIn("| 2 vs 3 | 2 | 10 | 75.0% | 60.0% | 90.0% |", table)


if __name__ == "__main__":
    unittest.main()
