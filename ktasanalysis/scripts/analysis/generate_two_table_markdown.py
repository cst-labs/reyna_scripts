#!/usr/bin/env python3
"""Generate a compact two-table markdown summary from KTAS model metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("scripts/analysis/output/full_error_analysis_models/tables/model_reference_metrics.csv")
DEFAULT_OUTPUT = Path("scripts/analysis/output/full_error_analysis_models/ktas_two_table_summary.md")
DEFAULT_WIDE_INPUT = Path("scripts/analysis/output/model_comparison_wide.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a two-table KTAS markdown summary.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Metrics CSV input (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Markdown output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument(
        "--wide-input",
        type=Path,
        default=DEFAULT_WIDE_INPUT,
        help=f"Wide comparison CSV used to derive the KTAS_RN vs KTAS_expert row (default: {DEFAULT_WIDE_INPUT})",
    )
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


def build_table(df: pd.DataFrame, reference_label: str) -> list[str]:
    subset = (
        df[df["reference"] == reference_label]
        .sort_values(by=["accuracy", "within_1_accuracy"], ascending=False)
        .copy()
    )

    lines = [
        "| Model | Accuracy | Under-triage | Over-triage | Within ±1 level | ≥2-level errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in subset.iterrows():
        total = int(row["n"])
        accuracy_count = int(row["exact_match_count"])
        under_count = int(round(float(row["under_triage_rate"]) * total))
        over_count = int(round(float(row["over_triage_rate"]) * total))
        within1_count = int(row["exact_match_count"] + row["adjacent_error_count"])
        catastrophic_count = int(row["catastrophic_error_count"])
        lines.append(
            "| {model} | {accuracy} | {under} | {over} | {within1} | {catastrophic} |".format(
                model=prettify_model_name(row["model"]),
                accuracy=format_count_pct(accuracy_count, total),
                under=format_count_pct(under_count, total),
                over=format_count_pct(over_count, total),
                within1=format_count_pct(within1_count, total),
                catastrophic=format_count_pct(catastrophic_count, total),
            )
        )
    return lines


def build_ktas_rn_row(wide_df: pd.DataFrame) -> str:
    total = len(wide_df)
    expert = pd.to_numeric(wide_df["KTAS_expert"], errors="coerce")
    rn = pd.to_numeric(wide_df["KTAS_RN"], errors="coerce")
    valid = pd.DataFrame({"expert": expert, "rn": rn}).dropna()
    total = len(valid)
    exact_match_count = int((valid["rn"] == valid["expert"]).sum())
    under_count = int((valid["rn"] > valid["expert"]).sum())
    over_count = int((valid["rn"] < valid["expert"]).sum())
    absolute_deviation = (valid["rn"] - valid["expert"]).abs()
    within1_count = int((absolute_deviation <= 1).sum())
    catastrophic_count = int((absolute_deviation >= 2).sum())
    return (
        "| KTAS_RN | {accuracy} | {under} | {over} | {within1} | {catastrophic} |".format(
            accuracy=format_count_pct(exact_match_count, total),
            under=format_count_pct(under_count, total),
            over=format_count_pct(over_count, total),
            within1=format_count_pct(within1_count, total),
            catastrophic=format_count_pct(catastrophic_count, total),
        )
    )


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    wide_df = pd.read_csv(args.wide_input)
    expert_table = build_table(df, "expert")
    expert_table.append(build_ktas_rn_row(wide_df))

    lines = [
        "# KTAS Model Summary",
        "",
        "## Compared With KTAS_expert",
        "",
        *expert_table,
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print(f"Markdown summary written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
