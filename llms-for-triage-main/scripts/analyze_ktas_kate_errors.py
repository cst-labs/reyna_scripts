#!/usr/bin/env python3
"""Generate error analysis for KTAS KATE prediction runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_GLOB = "results/Triage-KTAS/Triage-KTAS_KATE_*.csv"
DEFAULT_OUTPUT = "results/Triage-KTAS/KTAS_kate_error_analysis.md"
TRUE_LABEL = "KTAS_expert"
PRED_LABEL = "Estimated_Acuity"
CLASS_LABELS = [1, 2, 3, 4, 5]


@dataclass
class RunRecord:
    model: str
    timestamp: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze KTAS KATE prediction errors.")
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB, help="Glob for KATE CSV prediction files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Markdown output path")
    parser.add_argument("--keep-all-runs", action="store_true", help="Keep all runs instead of latest per model")
    return parser.parse_args()


def parse_run(path: Path) -> RunRecord:
    stem = path.stem
    model = stem.split("Triage-KTAS_KATE_", 1)[1].split("_json_", 1)[0]
    parts = stem.rsplit("_", 2)
    timestamp = f"{parts[-2]}_{parts[-1]}"
    return RunRecord(model=model, timestamp=timestamp, path=path)


def load_runs(input_glob: str, keep_all_runs: bool) -> list[RunRecord]:
    runs = [parse_run(path) for path in sorted(Path().glob(input_glob))]
    if keep_all_runs:
        return runs

    latest_by_model: dict[str, RunRecord] = {}
    for run in runs:
        current = latest_by_model.get(run.model)
        if current is None or run.timestamp > current.timestamp:
            latest_by_model[run.model] = run
    return sorted(latest_by_model.values(), key=lambda run: run.model)


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


def analyze_run(run: RunRecord) -> dict[str, object]:
    df = pd.read_csv(run.path).copy()
    df = df[[TRUE_LABEL, PRED_LABEL]].dropna()
    df[TRUE_LABEL] = df[TRUE_LABEL].astype(int)
    df[PRED_LABEL] = df[PRED_LABEL].astype(int)
    df["diff"] = df[PRED_LABEL] - df[TRUE_LABEL]
    df["error_direction"] = "correct"
    df.loc[df["diff"] > 0, "error_direction"] = "under"
    df.loc[df["diff"] < 0, "error_direction"] = "over"

    overall = {
        "n": len(df),
        "accuracy": (df["diff"] == 0).mean(),
        "under_rate": (df["diff"] > 0).mean(),
        "over_rate": (df["diff"] < 0).mean(),
        "mean_signed_error": df["diff"].mean(),
        "mean_abs_error": df["diff"].abs().mean(),
    }

    by_true = []
    for label in CLASS_LABELS:
        subset = df[df[TRUE_LABEL] == label]
        if subset.empty:
            continue
        by_true.append({
            "ktas": label,
            "support": len(subset),
            "accuracy": (subset["diff"] == 0).mean(),
            "under_rate": (subset["diff"] > 0).mean(),
            "over_rate": (subset["diff"] < 0).mean(),
            "mean_signed_error": subset["diff"].mean(),
        })

    confusions = (
        df.loc[df["diff"] != 0, [TRUE_LABEL, PRED_LABEL]]
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["count", TRUE_LABEL, PRED_LABEL], ascending=[False, True, True])
    )
    confusions["share"] = confusions["count"] / len(df)

    return {"overall": overall, "by_true": by_true, "confusions": confusions}


def build_overall_table(analyzed: list[tuple[RunRecord, dict[str, object]]]) -> str:
    ranked = sorted(
        analyzed,
        key=lambda item: (
            item[1]["overall"]["accuracy"],
            -item[1]["overall"]["mean_abs_error"],
        ),
        reverse=True,
    )
    rows = []
    for run, stats in ranked:
        overall = stats["overall"]
        rows.append([
            run.model,
            run.timestamp,
            str(overall["n"]),
            format_pct(overall["accuracy"]),
            format_pct(overall["under_rate"]),
            format_pct(overall["over_rate"]),
            format_float(overall["mean_signed_error"]),
            format_float(overall["mean_abs_error"]),
        ])
    return make_table(
        ["Model", "Timestamp", "N", "Accuracy", "Under-triage", "Over-triage", "Mean Signed Error", "Mean Abs Error"],
        rows,
    )


def build_run_section(run: RunRecord, stats: dict[str, object]) -> str:
    overall = stats["overall"]
    by_true = pd.DataFrame(stats["by_true"]).sort_values("ktas")
    confusions = stats["confusions"].head(5)

    highest_accuracy = by_true.sort_values(["accuracy", "support"], ascending=[False, False]).iloc[0]
    lowest_accuracy = by_true.sort_values(["accuracy", "support"], ascending=[True, False]).iloc[0]

    direction_bias = "balanced"
    if overall["mean_signed_error"] > 0.05:
        direction_bias = "leans toward under-triage"
    elif overall["mean_signed_error"] < -0.05:
        direction_bias = "leans toward over-triage"

    lines = [
        f"## {run.model}",
        "",
        f"- Run: `{run.path.name}`",
        f"- Overall: {format_pct(overall['accuracy'])} correct, {format_pct(overall['under_rate'])} under-triage, {format_pct(overall['over_rate'])} over-triage",
        f"- Bias: {direction_bias} (mean signed error {format_float(overall['mean_signed_error'])}; positive means predicted less urgent than truth)",
        f"- Easiest true severity: KTAS {int(highest_accuracy['ktas'])} ({format_pct(highest_accuracy['accuracy'])} correct over {int(highest_accuracy['support'])} cases)",
        f"- Hardest true severity: KTAS {int(lowest_accuracy['ktas'])} ({format_pct(lowest_accuracy['accuracy'])} correct over {int(lowest_accuracy['support'])} cases)",
        "",
        make_table(
            ["True KTAS", "Support", "Correct", "Under-triage", "Over-triage", "Mean Signed Error"],
            [
                [
                    str(int(row.ktas)),
                    str(int(row.support)),
                    format_pct(row.accuracy),
                    format_pct(row.under_rate),
                    format_pct(row.over_rate),
                    format_float(row.mean_signed_error),
                ]
                for row in by_true.itertuples(index=False)
            ],
        ),
        "",
        "Top wrong true->predicted pairs:",
        "",
        make_table(
            ["True", "Pred", "Count", "Share"],
            [
                [
                    str(int(row[TRUE_LABEL])),
                    str(int(row[PRED_LABEL])),
                    str(int(row["count"])),
                    format_pct(row["share"]),
                ]
                for _, row in confusions.iterrows()
            ],
        ) if not confusions.empty else "_No errors._",
        "",
    ]
    return "\n".join(lines)


def build_report(analyzed: list[tuple[RunRecord, dict[str, object]]], input_glob: str, keep_all_runs: bool) -> str:
    lines = [
        "# KTAS KATE Error Analysis",
        "",
        f"Source glob: `{input_glob}`",
        f"Runs included: {len(analyzed)}",
        f"Deduplication: {'all runs included' if keep_all_runs else 'latest run per model'}",
        "",
        "Interpretation:",
        "- KTAS 1 is most urgent and KTAS 5 is least urgent.",
        "- Under-triage means the prediction is numerically higher than the expert label.",
        "- Over-triage means the prediction is numerically lower than the expert label.",
        "",
        "## Overall Comparison",
        "",
        build_overall_table(analyzed),
        "",
    ]
    for run, stats in analyzed:
        lines.append(build_run_section(run, stats))
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    runs = load_runs(args.input_glob, args.keep_all_runs)
    if not runs:
        raise SystemExit(f"No runs matched: {args.input_glob}")

    analyzed = [(run, analyze_run(run)) for run in runs]
    output = build_report(analyzed, args.input_glob, args.keep_all_runs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(f"Wrote error analysis to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
