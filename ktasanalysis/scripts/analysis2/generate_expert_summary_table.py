#!/usr/bin/env python3
"""Generate a compact KTAS_expert comparison markdown table for analysis2."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("scripts/analysis2/output/full_error_analysis_models/tables/model_reference_metrics.csv")
DEFAULT_WIDE_INPUT = Path("scripts/analysis2/output/model_comparison_wide.csv")
DEFAULT_OUTPUT = Path("scripts/analysis2/output/full_error_analysis_models/ktas_expert_summary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compact KTAS_expert summary markdown table.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Model metrics CSV input (default: {DEFAULT_INPUT})")
    parser.add_argument(
        "--wide-input",
        type=Path,
        default=DEFAULT_WIDE_INPUT,
        help=f"Wide comparison CSV used to derive the KTAS_RN row (default: {DEFAULT_WIDE_INPUT})",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Markdown output path (default: {DEFAULT_OUTPUT})")
    return parser.parse_args()


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_count_pct(count: int, total: int) -> str:
    rate = (count / total) if total else 0.0
    return f"{count}/{total} ({format_pct(rate)})"


def prettify_model_name(name: str) -> str:
    text = str(name)
    if text.startswith("pred_"):
        text = text[5:]
    return text.replace("_", ".")


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.exists():
        raise SystemExit(f"{label} not found: {resolved}")
    return resolved


def build_model_rows(metrics_df: pd.DataFrame) -> list[str]:
    required_columns = {
        "model",
        "reference",
        "accuracy",
        "within_1_accuracy",
        "exact_match_count",
        "adjacent_error_count",
        "catastrophic_error_count",
        "under_triage_rate",
        "over_triage_rate",
        "n",
    }
    missing = sorted(required_columns - set(metrics_df.columns))
    if missing:
        raise SystemExit(f"Metrics CSV is missing required columns: {', '.join(missing)}")

    subset = (
        metrics_df[metrics_df["reference"] == "expert"]
        .sort_values(by=["accuracy", "within_1_accuracy"], ascending=False)
        .copy()
    )
    if subset.empty:
        raise SystemExit("Metrics CSV does not contain any rows with reference == 'expert'.")

    lines = []
    for _, row in subset.iterrows():
        total = int(row["n"])
        accuracy_count = int(row["exact_match_count"])
        over_count = int(round(float(row["over_triage_rate"]) * total))
        under_count = int(round(float(row["under_triage_rate"]) * total))
        within_1_count = int(row["exact_match_count"] + row["adjacent_error_count"])
        more_than_1_count = int(row["catastrophic_error_count"])
        lines.append(
            "| {model} | {accuracy} | {over} | {under} | {within_1} | {more_than_1} |".format(
                model=prettify_model_name(row["model"]),
                accuracy=format_count_pct(accuracy_count, total),
                over=format_count_pct(over_count, total),
                under=format_count_pct(under_count, total),
                within_1=format_count_pct(within_1_count, total),
                more_than_1=format_count_pct(more_than_1_count, total),
            )
        )
    return lines


def build_ktas_rn_row(wide_df: pd.DataFrame) -> str:
    required_columns = {"KTAS_expert", "KTAS_RN"}
    missing = sorted(required_columns - set(wide_df.columns))
    if missing:
        raise SystemExit(f"Wide CSV is missing required columns: {', '.join(missing)}")

    valid = pd.DataFrame(
        {
            "expert": pd.to_numeric(wide_df["KTAS_expert"], errors="coerce"),
            "rn": pd.to_numeric(wide_df["KTAS_RN"], errors="coerce"),
        }
    ).dropna()
    total = len(valid)
    if total == 0:
        raise SystemExit("Wide CSV does not contain any valid KTAS_expert/KTAS_RN pairs.")

    signed_deviation = valid["rn"] - valid["expert"]
    absolute_deviation = signed_deviation.abs()
    exact_match_count = int((absolute_deviation == 0).sum())
    over_count = int((signed_deviation < 0).sum())
    under_count = int((signed_deviation > 0).sum())
    within_1_count = int((absolute_deviation <= 1).sum())
    more_than_1_count = int((absolute_deviation > 1).sum())
    return (
        "| KTAS_RN | {accuracy} | {over} | {under} | {within_1} | {more_than_1} |".format(
            accuracy=format_count_pct(exact_match_count, total),
            over=format_count_pct(over_count, total),
            under=format_count_pct(under_count, total),
            within_1=format_count_pct(within_1_count, total),
            more_than_1=format_count_pct(more_than_1_count, total),
        )
    )


def main() -> int:
    args = parse_args()
    metrics_path = require_file(args.input, "Metrics CSV")
    wide_path = require_file(args.wide_input, "Wide CSV")

    metrics_df = pd.read_csv(metrics_path)
    wide_df = pd.read_csv(wide_path)

    lines = [
        "# KTAS Expert Comparison",
        "",
        "| Model | Accuracy | Over-triage | Under-triage | Within 1 | More than 1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *build_model_rows(metrics_df),
        build_ktas_rn_row(wide_df),
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print(f"Markdown summary written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
