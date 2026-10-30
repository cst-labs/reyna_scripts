#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small KTAS BQR gamma sweep and summarize the results.")
    parser.add_argument("--model", required=True, help="LLM model to evaluate.")
    parser.add_argument("--gammas", nargs="+", type=float, default=[1.0, 0.8, 0.6], help="Gamma values to sweep.")
    parser.add_argument("--candidate-pool-size", type=int, default=200, help="Candidate pool size.")
    parser.add_argument("--final-shot-count", type=int, default=40, help="Final number of shots.")
    parser.add_argument("--minimum-floor-per-class", type=int, default=2, help="Minimum quota floor.")
    parser.add_argument("--ktas3-cap-fraction", type=float, default=0.30, help="KTAS 3 cap fraction.")
    parser.add_argument("--serialization", default="natural", help="Prompt serialization.")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "Triage-KTAS-BQR"), help="Output directory.")
    parser.add_argument("--start", type=int, default=0, help="Start index for the KTAS test split.")
    parser.add_argument("--end", type=int, default=578, help="End index for the KTAS test split.")
    parser.add_argument("--json", action="store_true", help="Request JSON responses.")
    return parser.parse_args()


def extract_confusion(matrix_payload: dict, truth_label: int, pred_label: int) -> int:
    labels = matrix_payload["labels"]
    matrix = matrix_payload["matrix"]
    truth_idx = labels.index(truth_label)
    pred_idx = labels.index(pred_label)
    return int(matrix[truth_idx][pred_idx])


def main() -> int:
    args = parse_args()
    runs = []
    for gamma in args.gammas:
        run_id = f"sweep_gamma_{gamma}"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_ktas_bqr_experiments.py"),
            "--model",
            args.model,
            "--retrieval-strategy",
            "balanced_quota",
            "--candidate-pool-size",
            str(args.candidate_pool_size),
            "--final-shot-count",
            str(args.final_shot_count),
            "--gamma",
            str(gamma),
            "--minimum-floor-per-class",
            str(args.minimum_floor_per_class),
            "--ktas3-cap-fraction",
            str(args.ktas3_cap_fraction),
            "--serialization",
            args.serialization,
            "--output-dir",
            args.output_dir,
            "--start",
            str(args.start),
            "--end",
            str(args.end),
            "--run-id",
            run_id,
        ]
        if args.json:
            command.append("--json")
        subprocess.run(command, check=True)

        metrics_path = Path(args.output_dir) / (
            f"Triage-KTAS_KATE_balanced_quota_{args.model}_{args.final_shot_count}_shots_{args.serialization}"
            f"_pool{args.candidate_pool_size}_gamma{gamma}_floor{args.minimum_floor_per_class}"
            f"_cap{args.ktas3_cap_fraction}_{args.start}_{args.end}_{run_id}_metrics.json"
        )
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        runs.append(
            {
                "gamma": gamma,
                "accuracy": metrics["overall"]["accuracy"],
                "macro_f1": metrics["by_class"]["macro avg"]["f1-score"],
                "quadratic_kappa": metrics["overall"]["quadratic_kappa"],
                "ktas_1_recall": metrics["by_class"]["1"]["recall"],
                "ktas_5_recall": metrics["by_class"]["5"]["recall"],
                "confusion_2_to_3": extract_confusion(metrics["confusion_matrix"], 2, 3),
                "confusion_3_to_2": extract_confusion(metrics["confusion_matrix"], 3, 2),
                "confusion_3_to_4": extract_confusion(metrics["confusion_matrix"], 3, 4),
                "confusion_4_to_3": extract_confusion(metrics["confusion_matrix"], 4, 3),
                "metrics_path": str(metrics_path),
            }
        )

    summary_path = Path(args.output_dir) / "ktas_bqr_gamma_sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"runs": runs}, handle, indent=2)
    print(f"Sweep summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

