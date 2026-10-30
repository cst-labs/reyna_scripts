#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KTAS_LABELS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KTAS retrieval experiment runs and compare them.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction CSV paths to evaluate.")
    parser.add_argument("--retrieval-logs", nargs="*", help="Optional retrieval audit JSONL paths aligned to --predictions.")
    parser.add_argument("--output", help="JSON output path. Defaults to a sibling *_evaluation.json beside the first prediction file.")
    parser.add_argument("--markdown-output", help="Optional Markdown comparison output path.")
    return parser.parse_args()


def load_retrieval_audit(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _boundary_error_summary(truth: pd.Series, pred: pd.Series) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    pairs = [
        ("2_to_3", 2, 3),
        ("3_to_2", 3, 2),
        ("3_to_4", 3, 4),
        ("4_to_3", 4, 3),
    ]
    for name, true_label, predicted_label in pairs:
        relevant_mask = truth == true_label
        total_true = int(relevant_mask.sum())
        error_count = int(((truth == true_label) & (pred == predicted_label)).sum())
        summaries[name] = {
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": error_count,
            "rate_within_true_label": (error_count / total_true) if total_true else 0.0,
        }
    return summaries


def build_metrics_payload(predictions_path: Path, retrieval_path: Path | None) -> dict:
    predictions_df = pd.read_csv(predictions_path)
    truth = predictions_df["KTAS_expert"]
    pred = predictions_df["Estimated_Acuity"]

    by_class = classification_report(truth, pred, labels=KTAS_LABELS, output_dict=True, zero_division=0)
    retrieval_summary = {"per_query": [], "aggregate": {}, "score_summary": {}, "run_config": {}}
    if retrieval_path is not None and retrieval_path.exists():
        audit_records = load_retrieval_audit(retrieval_path)
        per_query = []
        aggregate_accumulator = {label: [] for label in KTAS_LABELS}
        disagreement_scores: list[float] = []
        feature_scores: list[float] = []
        for record in audit_records:
            counts = {int(label): int(count) for label, count in record["final_per_class_counts"].items()}
            per_query.append(
                {
                    "query_id": record["query_id"],
                    "retrieval_strategy": record["retrieval_strategy"],
                    "final_per_class_counts": counts,
                    "quota_targets": {int(label): int(count) for label, count in record["quota_targets"].items()},
                }
            )
            for label in KTAS_LABELS:
                aggregate_accumulator[label].append(counts.get(label, 0))
            disagreement_scores.extend(float(score) for score in record.get("selected_disagreement_scores", []))
            feature_scores.extend(float(score) for score in record.get("selected_feature_disagreement_scores", []))
        retrieval_summary["per_query"] = per_query
        retrieval_summary["aggregate"] = {
            str(label): {
                "mean": sum(values) / len(values) if values else 0.0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
            }
            for label, values in aggregate_accumulator.items()
        }
        retrieval_summary["score_summary"] = {
            "mean_selected_disagreement_score": (
                sum(disagreement_scores) / len(disagreement_scores) if disagreement_scores else 0.0
            ),
            "mean_selected_feature_disagreement_score": (
                sum(feature_scores) / len(feature_scores) if feature_scores else 0.0
            ),
        }
        if audit_records:
            first = audit_records[0]
            retrieval_summary["run_config"] = {
                "retrieval_strategy": first.get("retrieval_strategy"),
                "candidate_pool_size": first.get("candidate_pool_size"),
                "final_shot_count": first.get("final_shot_count"),
                "alpha": first.get("alpha"),
                "beta": first.get("beta"),
                "neighborhood_size": first.get("neighborhood_size"),
            }

    return {
        "run_name": predictions_path.stem,
        "predictions_path": str(predictions_path),
        "retrieval_log_path": str(retrieval_path) if retrieval_path is not None else None,
        "overall": {
            "accuracy": accuracy_score(truth, pred),
            "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(truth, pred, average="weighted", zero_division=0),
            "quadratic_kappa": cohen_kappa_score(truth, pred, weights="quadratic"),
            "plus_minus_1_accuracy": accuracy_score(truth, [t if abs(t - p) <= 1 else p for t, p in zip(truth, pred)]),
        },
        "by_class": by_class,
        "confusion_matrix": {
            "labels": KTAS_LABELS,
            "matrix": confusion_matrix(truth, pred, labels=KTAS_LABELS).tolist(),
        },
        "boundary_error_summary": _boundary_error_summary(truth, pred),
        "retrieval_distribution_summary": retrieval_summary,
    }


def make_markdown(summary: dict) -> str:
    lines = [
        "# KTAS Retrieval Evaluation",
        "",
        "| Run | Accuracy | Macro F1 | Weighted F1 | QWK | +/-1 Accuracy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in summary["runs"]:
        overall = run["overall"]
        lines.append(
            f"| {run['run_name']} | {overall['accuracy']:.4f} | {overall['macro_f1']:.4f} | "
            f"{overall['weighted_f1']:.4f} | {overall['quadratic_kappa']:.4f} | {overall['plus_minus_1_accuracy']:.4f} |"
        )

    lines.extend(["", "## Boundary Errors", ""])
    for run in summary["runs"]:
        boundary = run["boundary_error_summary"]
        lines.append(f"### {run['run_name']}")
        lines.append("")
        lines.append(
            f"- 2->3: {boundary['2_to_3']['count']} ({boundary['2_to_3']['rate_within_true_label']:.4f})"
        )
        lines.append(
            f"- 3->2: {boundary['3_to_2']['count']} ({boundary['3_to_2']['rate_within_true_label']:.4f})"
        )
        lines.append(
            f"- 3->4: {boundary['3_to_4']['count']} ({boundary['3_to_4']['rate_within_true_label']:.4f})"
        )
        lines.append(
            f"- 4->3: {boundary['4_to_3']['count']} ({boundary['4_to_3']['rate_within_true_label']:.4f})"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    retrieval_paths: list[Path | None]
    if args.retrieval_logs:
        retrieval_paths = [Path(path) for path in args.retrieval_logs]
        if len(retrieval_paths) != len(args.predictions):
            raise ValueError("--retrieval-logs must match the number of --predictions when provided.")
    else:
        retrieval_paths = []
        for prediction in args.predictions:
            prediction_path = Path(prediction)
            derived = prediction_path.with_name(f"{prediction_path.stem}_retrieval_audit.jsonl")
            retrieval_paths.append(derived if derived.exists() else None)

    runs = [
        build_metrics_payload(Path(prediction_path), retrieval_path)
        for prediction_path, retrieval_path in zip(args.predictions, retrieval_paths)
    ]
    summary = {"runs": runs}

    output_path = Path(args.output) if args.output else Path(args.predictions[0]).with_name(f"{Path(args.predictions[0]).stem}_evaluation.json")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    markdown_path = Path(args.markdown_output) if args.markdown_output else output_path.with_suffix(".md")
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write(make_markdown(summary))

    print(f"Evaluation summary saved to {output_path}")
    print(f"Markdown summary saved to {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
