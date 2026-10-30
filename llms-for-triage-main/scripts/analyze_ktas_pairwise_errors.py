#!/usr/bin/env python3
"""Generate richer error analysis for KTAS pairwise prediction runs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_GLOB = "results/Triage-KTAS-Pairwise/Triage-KTAS-Pairwise_*.csv"
DEFAULT_OUTPUT = "results/Triage-KTAS-Pairwise/KTAS_pairwise_error_analysis.md"
TRUE_LABEL = "Pair_Label"
PRED_LABEL = "Predicted_Order"
GRADE_LABEL_A = "A_KTAS_expert"
GRADE_LABEL_B = "B_KTAS_expert"

PAIRWISE_FILENAME_RE = re.compile(
    r"^Triage-KTAS-Pairwise_(?P<strategy>.+?)_(?P<model>.+?)_(?P<shots>\d+)_shots_.*$"
)


@dataclass
class RunRecord:
    model: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze KTAS pairwise prediction errors.")
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB, help="Glob for pairwise CSV prediction files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Markdown output path")
    return parser.parse_args()


def parse_run(path: Path) -> RunRecord:
    match = PAIRWISE_FILENAME_RE.match(path.stem)
    if match:
        return RunRecord(model=match.group("model"), path=path)
    return RunRecord(model=path.stem, path=path)


def load_runs(input_glob: str) -> list[RunRecord]:
    return sorted((parse_run(path) for path in Path().glob(input_glob)), key=lambda run: run.model)


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_float(value: float) -> str:
    return f"{value:.3f}"


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def attach_pairwise_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[PRED_LABEL, TRUE_LABEL, GRADE_LABEL_A, GRADE_LABEL_B]].dropna().copy()
    df[GRADE_LABEL_A] = df[GRADE_LABEL_A].astype(int)
    df[GRADE_LABEL_B] = df[GRADE_LABEL_B].astype(int)
    df["correct"] = df[PRED_LABEL] == df[TRUE_LABEL]
    df["grade_gap"] = (df[GRADE_LABEL_A] - df[GRADE_LABEL_B]).abs()
    df["urgent_grade"] = df[[GRADE_LABEL_A, GRADE_LABEL_B]].min(axis=1)
    df["less_urgent_grade"] = df[[GRADE_LABEL_A, GRADE_LABEL_B]].max(axis=1)
    df["grade_pair"] = (
        df["urgent_grade"].astype(str) + " vs " + df["less_urgent_grade"].astype(str)
    )
    df["predicted_more_urgent_grade"] = df[GRADE_LABEL_A].where(
        df[PRED_LABEL] == "A_BEFORE_B",
        df[GRADE_LABEL_B],
    )
    df["true_more_urgent_grade"] = df[GRADE_LABEL_A].where(
        df[TRUE_LABEL] == "A_BEFORE_B",
        df[GRADE_LABEL_B],
    )
    df["mistake_type"] = "correct"
    wrong_mask = ~df["correct"]
    df.loc[wrong_mask, "mistake_type"] = "favored_less_urgent_patient"
    return df


def summarize_accuracy(group: pd.DataFrame, label_column: str) -> list[dict[str, object]]:
    rows = []
    for label, subset in group.groupby(label_column, sort=True):
        rows.append({
            label_column: label,
            "support": len(subset),
            "accuracy": subset["correct"].mean(),
            "error_rate": 1.0 - subset["correct"].mean(),
        })
    return rows


def analyze_run(run: RunRecord) -> dict[str, object]:
    df = attach_pairwise_features(pd.read_csv(run.path))

    overall = {
        "n": len(df),
        "accuracy": df["correct"].mean(),
        "error_rate": 1.0 - df["correct"].mean(),
        "mean_grade_gap": df["grade_gap"].mean(),
    }

    by_gap = summarize_accuracy(df, "grade_gap")
    by_pair = []
    for pair_key, subset in df.groupby(["urgent_grade", "less_urgent_grade"], sort=True):
        urgent_grade, less_urgent_grade = pair_key
        by_pair.append({
            "urgent_grade": int(urgent_grade),
            "less_urgent_grade": int(less_urgent_grade),
            "pair": f"{urgent_grade} vs {less_urgent_grade}",
            "support": len(subset),
            "accuracy": subset["correct"].mean(),
            "error_rate": 1.0 - subset["correct"].mean(),
        })

    by_urgent_grade = summarize_accuracy(df, "urgent_grade")
    by_less_urgent_grade = summarize_accuracy(df, "less_urgent_grade")

    wrong = df.loc[~df["correct"]].copy()
    hardest_pairs = (
        pd.DataFrame(by_pair)
        .sort_values(["accuracy", "support", "urgent_grade", "less_urgent_grade"], ascending=[True, False, True, True])
        .head(5)
    )

    wrong_pair_counts = pd.DataFrame(columns=["pair", "count", "share"])
    if not wrong.empty:
        wrong_pair_counts = (
            wrong["grade_pair"]
            .value_counts()
            .rename_axis("pair")
            .reset_index(name="count")
        )
        wrong_pair_counts["share"] = wrong_pair_counts["count"] / len(df)

    return {
        "overall": overall,
        "by_gap": by_gap,
        "by_pair": by_pair,
        "by_urgent_grade": by_urgent_grade,
        "by_less_urgent_grade": by_less_urgent_grade,
        "hardest_pairs": hardest_pairs,
        "wrong_pair_counts": wrong_pair_counts,
    }


def accuracy_for_gap(stats: dict[str, object], gap: int) -> str:
    for row in stats["by_gap"]:
        if int(row["grade_gap"]) == gap:
            return format_pct(row["accuracy"])
    return "n/a"


def build_overall_table(analyzed: list[tuple[RunRecord, dict[str, object]]]) -> str:
    ranked = sorted(
        analyzed,
        key=lambda item: (
            item[1]["overall"]["accuracy"],
            item[1]["overall"]["mean_grade_gap"],
        ),
        reverse=True,
    )
    rows = []
    for run, stats in ranked:
        overall = stats["overall"]
        rows.append([
            run.model,
            str(overall["n"]),
            format_pct(overall["accuracy"]),
            accuracy_for_gap(stats, 1),
            accuracy_for_gap(stats, 2),
            accuracy_for_gap(stats, 3),
            accuracy_for_gap(stats, 4),
            format_float(overall["mean_grade_gap"]),
        ])
    return make_table(
        ["Model", "N", "Accuracy", "Gap 1 Acc", "Gap 2 Acc", "Gap 3 Acc", "Gap 4 Acc", "Mean Grade Gap"],
        rows,
    )


def build_average_grade_pair_table(analyzed: list[tuple[RunRecord, dict[str, object]]]) -> str:
    pair_rows = []
    for _, stats in analyzed:
        pair_rows.extend(stats["by_pair"])

    by_pair = pd.DataFrame(pair_rows)
    summary = (
        by_pair.groupby(["urgent_grade", "less_urgent_grade", "pair"], as_index=False)
        .agg(
            models=("accuracy", "size"),
            avg_accuracy=("accuracy", "mean"),
            min_accuracy=("accuracy", "min"),
            max_accuracy=("accuracy", "max"),
            support_per_model=("support", "first"),
        )
        .sort_values(["urgent_grade", "less_urgent_grade"])
    )

    rows = []
    for row in summary.itertuples(index=False):
        rows.append([
            row.pair,
            str(int(row.models)),
            str(int(row.support_per_model)),
            format_pct(row.avg_accuracy),
            format_pct(row.min_accuracy),
            format_pct(row.max_accuracy),
        ])
    return make_table(
        ["Grade Pair", "Models", "Support / Model", "Average Accuracy", "Min Accuracy", "Max Accuracy"],
        rows,
    )


def build_run_section(run: RunRecord, stats: dict[str, object]) -> str:
    overall = stats["overall"]
    by_gap = pd.DataFrame(stats["by_gap"]).sort_values("grade_gap")
    by_pair = pd.DataFrame(stats["by_pair"]).sort_values(["urgent_grade", "less_urgent_grade"])
    by_urgent_grade = pd.DataFrame(stats["by_urgent_grade"]).sort_values("urgent_grade")
    by_less_urgent_grade = pd.DataFrame(stats["by_less_urgent_grade"]).sort_values("less_urgent_grade")
    hardest_pairs = stats["hardest_pairs"]
    wrong_pair_counts = stats["wrong_pair_counts"].head(5)

    easiest_gap = by_gap.sort_values(["accuracy", "support"], ascending=[False, False]).iloc[0]
    hardest_gap = by_gap.sort_values(["accuracy", "support"], ascending=[True, False]).iloc[0]
    hardest_pair = by_pair.sort_values(["accuracy", "support"], ascending=[True, False]).iloc[0]

    lines = [
        f"## {run.model}",
        "",
        f"- Run: `{run.path.name}`",
        f"- Overall pairwise accuracy: {format_pct(overall['accuracy'])} over {overall['n']} comparisons",
        f"- Easiest grade gap: {int(easiest_gap['grade_gap'])} ({format_pct(easiest_gap['accuracy'])} over {int(easiest_gap['support'])} cases)",
        f"- Hardest grade gap: {int(hardest_gap['grade_gap'])} ({format_pct(hardest_gap['accuracy'])} over {int(hardest_gap['support'])} cases)",
        f"- Hardest exact grade pairing: {hardest_pair['pair']} ({format_pct(hardest_pair['accuracy'])} over {int(hardest_pair['support'])} cases)",
        "",
        "Accuracy by absolute KTAS grade difference:",
        "",
        make_table(
            ["Grade Gap", "Support", "Accuracy", "Error Rate"],
            [
                [
                    str(int(row.grade_gap)),
                    str(int(row.support)),
                    format_pct(row.accuracy),
                    format_pct(row.error_rate),
                ]
                for row in by_gap.itertuples(index=False)
            ],
        ),
        "",
        "Accuracy by exact urgent-vs-less-urgent grade pairing:",
        "",
        make_table(
            ["Grade Pair", "Support", "Accuracy", "Error Rate"],
            [
                [
                    row.pair,
                    str(int(row.support)),
                    format_pct(row.accuracy),
                    format_pct(row.error_rate),
                ]
                for row in by_pair.itertuples(index=False)
            ],
        ),
        "",
        "Accuracy conditioned on the more urgent patient's grade:",
        "",
        make_table(
            ["Urgent Grade", "Support", "Accuracy", "Error Rate"],
            [
                [
                    str(int(row.urgent_grade)),
                    str(int(row.support)),
                    format_pct(row.accuracy),
                    format_pct(row.error_rate),
                ]
                for row in by_urgent_grade.itertuples(index=False)
            ],
        ),
        "",
        "Accuracy conditioned on the less urgent patient's grade:",
        "",
        make_table(
            ["Less Urgent Grade", "Support", "Accuracy", "Error Rate"],
            [
                [
                    str(int(row.less_urgent_grade)),
                    str(int(row.support)),
                    format_pct(row.accuracy),
                    format_pct(row.error_rate),
                ]
                for row in by_less_urgent_grade.itertuples(index=False)
            ],
        ),
        "",
        "Hardest exact grade pairs:",
        "",
        make_table(
            ["Grade Pair", "Support", "Accuracy", "Error Rate"],
            [
                [
                    row["pair"],
                    str(int(row["support"])),
                    format_pct(row["accuracy"]),
                    format_pct(row["error_rate"]),
                ]
                for _, row in hardest_pairs.iterrows()
            ],
        ),
        "",
        "Most common wrong grade pairings:",
        "",
        make_table(
            ["Grade Pair", "Wrong Count", "Share of All Comparisons"],
            [
                [
                    row["pair"],
                    str(int(row["count"])),
                    format_pct(row["share"]),
                ]
                for _, row in wrong_pair_counts.iterrows()
            ],
        ) if not wrong_pair_counts.empty else "_No errors._",
        "",
    ]
    return "\n".join(lines)


def build_report(analyzed: list[tuple[RunRecord, dict[str, object]]], input_glob: str) -> str:
    lines = [
        "# KTAS Pairwise Error Analysis",
        "",
        f"Source glob: `{input_glob}`",
        f"Runs included: {len(analyzed)}",
        "",
        "Interpretation:",
        "- KTAS 1 is most urgent and KTAS 5 is least urgent.",
        "- `Grade Gap` is the absolute difference between the two expert KTAS grades in the pair.",
        "- `Grade Pair` is reported as `more urgent grade vs less urgent grade`, ignoring whether that patient appeared as A or B in the prompt.",
        "",
        "## Overall Comparison",
        "",
        build_overall_table(analyzed),
        "",
        "## Average Accuracy By Grade Pair",
        "",
        build_average_grade_pair_table(analyzed),
        "",
    ]
    for run, stats in analyzed:
        lines.append(build_run_section(run, stats))
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    runs = load_runs(args.input_glob)
    if not runs:
        raise SystemExit(f"No runs matched: {args.input_glob}")

    analyzed = [(run, analyze_run(run)) for run in runs]
    output = build_report(analyzed, args.input_glob)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(f"Wrote error analysis to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
