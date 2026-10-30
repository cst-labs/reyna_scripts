#!/usr/bin/env python3
"""Summarize KTAS metrics JSON files into a single Markdown report."""

from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_INPUT_GLOB = "results/Triage-KTAS/*_metrics.json"
DEFAULT_OUTPUT = "results/Triage-KTAS/KTAS_metrics_summary.md"

STANDARD_CLASS_LABELS = ["1", "2", "3", "4", "5"]
PAIRWISE_CLASS_LABELS = ["A_BEFORE_B", "B_BEFORE_A"]
SUMMARY_ONLY_CLASS_KEYS = {"accuracy", "macro avg", "weighted avg"}

STANDARD_FILENAME_RE = re.compile(
    r"^Triage-KTAS_KATE_(?P<model>.+?)_json_.*_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$"
)
PAIRWISE_FILENAME_RE = re.compile(
    r"^Triage-KTAS-Pairwise_(?P<strategy>.+?)_(?P<model>.+?)_(?P<shots>\d+)_shots_.*$"
)


@dataclass
class MetricsRecord:
    model: str
    timestamp: datetime
    filename: str
    payload: dict
    timestamp_is_embedded: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize KTAS metrics JSON files.")
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB, help="Glob for metrics JSON files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Markdown output path")
    parser.add_argument("--keep-all-runs", action="store_true", help="Keep all runs instead of latest per model")
    return parser.parse_args()


def parse_metrics_filename(path: Path) -> tuple[str, datetime, bool]:
    suffix = "_metrics.json"
    stem = path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem

    standard_match = STANDARD_FILENAME_RE.match(stem)
    if standard_match:
        timestamp = datetime.strptime(standard_match.group("timestamp"), "%Y-%m-%d_%H:%M:%S")
        return standard_match.group("model"), timestamp, True

    pairwise_match = PAIRWISE_FILENAME_RE.match(stem)
    if pairwise_match:
        timestamp = datetime.fromtimestamp(path.stat().st_mtime)
        return pairwise_match.group("model"), timestamp, False

    timestamp = datetime.fromtimestamp(path.stat().st_mtime)
    return stem, timestamp, False


def load_records(input_glob: str) -> list[MetricsRecord]:
    records: list[MetricsRecord] = []
    for filename in sorted(glob.glob(input_glob)):
        path = Path(filename)
        with path.open() as handle:
            payload = json.load(handle)
        model, timestamp, timestamp_is_embedded = parse_metrics_filename(path)
        records.append(
            MetricsRecord(
                model=model,
                timestamp=timestamp,
                filename=path.name,
                payload=payload,
                timestamp_is_embedded=timestamp_is_embedded,
            )
        )
    return records


def dedupe_latest(records: list[MetricsRecord]) -> list[MetricsRecord]:
    latest_by_model: dict[str, MetricsRecord] = {}
    for record in records:
        current = latest_by_model.get(record.model)
        if current is None or record.timestamp > current.timestamp:
            latest_by_model[record.model] = record
    return list(latest_by_model.values())


def format_float(value: float) -> str:
    return f"{value:.4f}"


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def is_pairwise_payload(payload: dict) -> bool:
    return "quadratic_kappa" not in payload["overall"]


def infer_class_labels(records: list[MetricsRecord]) -> list[str]:
    if is_pairwise_payload(records[0].payload):
        preferred = PAIRWISE_CLASS_LABELS
    else:
        preferred = STANDARD_CLASS_LABELS

    present = [
        label for label in preferred
        if label in records[0].payload["by_class"] and label not in SUMMARY_ONLY_CLASS_KEYS
    ]
    if present:
        return present

    return [
        key for key, value in records[0].payload["by_class"].items()
        if key not in SUMMARY_ONLY_CLASS_KEYS and isinstance(value, dict)
    ]


def build_overall_table(records: list[MetricsRecord]) -> str:
    pairwise = is_pairwise_payload(records[0].payload)
    if pairwise:
        ranked = sorted(
            records,
            key=lambda record: (
                record.payload["overall"]["accuracy"],
                record.payload["overall"]["f1_score"],
                record.payload["overall"]["precision"],
            ),
            reverse=True,
        )
        rows = []
        for record in ranked:
            overall = record.payload["overall"]
            rows.append([
                record.model,
                format_float(overall["accuracy"]),
                format_float(overall["precision"]),
                format_float(overall["recall"]),
                format_float(overall["f1_score"]),
            ])
        return make_table(["Model", "Accuracy", "Precision", "Recall", "F1"], rows)

    ranked = sorted(
        records,
        key=lambda record: (
            record.payload["overall"]["quadratic_kappa"],
            record.payload["overall"]["accuracy"],
            record.payload["overall"]["f1_score"],
        ),
        reverse=True,
    )
    rows = []
    for record in ranked:
        overall = record.payload["overall"]
        rows.append([
            record.model,
            record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            format_float(overall["quadratic_kappa"]),
            format_float(overall["accuracy"]),
            format_float(overall["f1_score"]),
            format_float(overall["adjusted_accuracy"]),
            format_float(overall["mae"]),
            format_float(overall["mse"]),
        ])
    return make_table(
        ["Model", "Timestamp", "Quadratic Kappa", "Accuracy", "F1", "Adjusted Accuracy", "MAE", "MSE"],
        rows,
    )


def build_best_summary(records: list[MetricsRecord], class_labels: list[str]) -> str:
    pairwise = is_pairwise_payload(records[0].payload)
    if pairwise:
        ranked = sorted(
            records,
            key=lambda record: (
                record.payload["overall"]["accuracy"],
                record.payload["overall"]["f1_score"],
                record.payload["overall"]["precision"],
            ),
            reverse=True,
        )
        best_accuracy = ranked[0]
        best_f1 = max(records, key=lambda record: record.payload["overall"]["f1_score"])
        lines = [
            f"- Best overall by accuracy: `{best_accuracy.model}` ({format_float(best_accuracy.payload['overall']['accuracy'])})",
            f"- Best overall by F1: `{best_f1.model}` ({format_float(best_f1.payload['overall']['f1_score'])})",
        ]
        for class_label in class_labels:
            best_for_class = max(
                records,
                key=lambda record: record.payload["by_class"][class_label]["recall"],
            )
            lines.append(
                f"- Best recall for {class_label}: `{best_for_class.model}` "
                f"({format_float(best_for_class.payload['by_class'][class_label]['recall'])})"
            )
        return "\n".join(lines)

    ranked = sorted(
        records,
        key=lambda record: (
            record.payload["overall"]["quadratic_kappa"],
            record.payload["overall"]["accuracy"],
            record.payload["overall"]["f1_score"],
        ),
        reverse=True,
    )
    best_kappa = ranked[0]
    best_accuracy = max(records, key=lambda record: record.payload["overall"]["accuracy"])
    lines = [
        f"- Best overall by quadratic kappa: `{best_kappa.model}` ({format_float(best_kappa.payload['overall']['quadratic_kappa'])})",
        f"- Best overall by accuracy: `{best_accuracy.model}` ({format_float(best_accuracy.payload['overall']['accuracy'])})",
    ]
    for class_label in class_labels:
        best_for_class = max(
            records,
            key=lambda record: record.payload["by_class"][class_label]["recall"],
        )
        lines.append(
            f"- Best recall for KTAS {class_label}: `{best_for_class.model}` "
            f"({format_float(best_for_class.payload['by_class'][class_label]['recall'])})"
        )
    return "\n".join(lines)


def build_class_tables(records: list[MetricsRecord], class_labels: list[str]) -> str:
    pairwise = is_pairwise_payload(records[0].payload)
    if pairwise:
        ranked_models = sorted(
            records,
            key=lambda record: (
                record.payload["overall"]["accuracy"],
                record.payload["overall"]["f1_score"],
                record.payload["overall"]["precision"],
            ),
            reverse=True,
        )
    else:
        ranked_models = sorted(
            records,
            key=lambda record: (
                record.payload["overall"]["quadratic_kappa"],
                record.payload["overall"]["accuracy"],
                record.payload["overall"]["f1_score"],
            ),
            reverse=True,
        )

    sections = []
    for class_label in class_labels:
        rows = []
        for record in ranked_models:
            class_metrics = record.payload["by_class"][class_label]
            rows.append([
                record.model,
                format_float(class_metrics["precision"]),
                format_float(class_metrics["recall"]),
                format_float(class_metrics["f1-score"]),
                str(class_metrics["support"]),
            ])
        heading = class_label if pairwise else f"KTAS {class_label}"
        sections.append(
            f"### {heading}\n\n" + make_table(
                ["Model", "Precision", "Recall", "F1", "Support"],
                rows,
            )
        )
    return "\n\n".join(sections)


def infer_report_title(records: list[MetricsRecord]) -> str:
    return "KTAS Pairwise Metrics Summary" if is_pairwise_payload(records[0].payload) else "KTAS Metrics Summary"


def build_report(records: list[MetricsRecord], input_glob: str, deduped: bool) -> str:
    class_labels = infer_class_labels(records)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {infer_report_title(records)}",
        "",
        f"Generated from `{input_glob}` on {generated_at}.",
        "",
        f"Runs included: {len(records)}",
        "",
        f"Deduplication: {'latest run per model' if deduped else 'all runs included'}",
        "",
        "## Overall Leaderboard",
        "",
        build_overall_table(records),
        "",
        "## Highlights",
        "",
        build_best_summary(records, class_labels),
        "",
        "## Per-Class Performance",
        "",
        build_class_tables(records, class_labels),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    records = load_records(args.input_glob)
    if not records:
        raise SystemExit(f"No metrics files matched: {args.input_glob}")

    deduped = not args.keep_all_runs
    selected_records = dedupe_latest(records) if deduped else records
    report = build_report(selected_records, args.input_glob, deduped)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote summary report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
