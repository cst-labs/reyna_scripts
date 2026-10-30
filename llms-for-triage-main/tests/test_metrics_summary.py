import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_ktas_metrics.py"
SPEC = importlib.util.spec_from_file_location("summarize_ktas_metrics", MODULE_PATH)
summarize_ktas_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = summarize_ktas_metrics
SPEC.loader.exec_module(summarize_ktas_metrics)


class MetricsSummaryTests(unittest.TestCase):
    def test_parse_standard_metrics_filename_extracts_model_and_timestamp(self):
        path = Path(
            "results/Triage-KTAS/"
            "Triage-KTAS_KATE_azure-gpt-5.4_json_40_shots_natural_0_578_2026-07-08_11:21:01_metrics.json"
        )
        model, timestamp, is_embedded = summarize_ktas_metrics.parse_metrics_filename(path)

        self.assertEqual(model, "azure-gpt-5.4")
        self.assertEqual(timestamp.strftime("%Y-%m-%d_%H:%M:%S"), "2026-07-08_11:21:01")
        self.assertTrue(is_embedded)

    def test_parse_pairwise_metrics_filename_extracts_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / (
                "Triage-KTAS-Pairwise_KATE-PAIRWISE_qwen2.5:7b_6_shots_natural_0_999999_"
                "ktas_balanced12_pairwise_metrics.json"
            )
            path.write_text("{}", encoding="utf-8")

            model, _, is_embedded = summarize_ktas_metrics.parse_metrics_filename(path)

        self.assertEqual(model, "qwen2.5:7b")
        self.assertFalse(is_embedded)

    def test_build_report_for_pairwise_metrics_lists_all_models(self):
        record_a = summarize_ktas_metrics.MetricsRecord(
            model="model-a",
            timestamp=summarize_ktas_metrics.datetime(2026, 7, 8, 12, 0, 0),
            filename="a.json",
            timestamp_is_embedded=False,
            payload={
                "overall": {
                    "accuracy": 0.80,
                    "precision": 0.81,
                    "recall": 0.80,
                    "f1_score": 0.79,
                },
                "by_class": {
                    "A_BEFORE_B": {"precision": 0.82, "recall": 0.78, "f1-score": 0.80, "support": 10.0},
                    "B_BEFORE_A": {"precision": 0.80, "recall": 0.82, "f1-score": 0.81, "support": 12.0},
                    "accuracy": 0.80,
                    "macro avg": {"precision": 0.81, "recall": 0.80, "f1-score": 0.80, "support": 22.0},
                    "weighted avg": {"precision": 0.81, "recall": 0.80, "f1-score": 0.80, "support": 22.0},
                },
            },
        )
        record_b = summarize_ktas_metrics.MetricsRecord(
            model="model-b",
            timestamp=summarize_ktas_metrics.datetime(2026, 7, 8, 12, 5, 0),
            filename="b.json",
            timestamp_is_embedded=False,
            payload={
                "overall": {
                    "accuracy": 0.85,
                    "precision": 0.86,
                    "recall": 0.85,
                    "f1_score": 0.84,
                },
                "by_class": {
                    "A_BEFORE_B": {"precision": 0.88, "recall": 0.83, "f1-score": 0.85, "support": 10.0},
                    "B_BEFORE_A": {"precision": 0.84, "recall": 0.87, "f1-score": 0.85, "support": 12.0},
                    "accuracy": 0.85,
                    "macro avg": {"precision": 0.86, "recall": 0.85, "f1-score": 0.85, "support": 22.0},
                    "weighted avg": {"precision": 0.86, "recall": 0.85, "f1-score": 0.85, "support": 22.0},
                },
            },
        )

        report = summarize_ktas_metrics.build_report(
            [record_a, record_b],
            "results/Triage-KTAS-Pairwise/*_metrics.json",
            deduped=True,
        )

        self.assertIn("# KTAS Pairwise Metrics Summary", report)
        self.assertIn("| Model | Accuracy | Precision | Recall | F1 |", report)
        self.assertIn("model-a", report)
        self.assertIn("model-b", report)
        self.assertIn("Best overall by accuracy: `model-b` (0.8500)", report)
        self.assertIn("### A_BEFORE_B", report)
        self.assertIn("### B_BEFORE_A", report)


if __name__ == "__main__":
    unittest.main()
