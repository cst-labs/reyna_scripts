#!/usr/bin/env python3
"""Analyze KTAS KATE neighbor retrieval with a rounded-mean label baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.prompts as prompts
import utils.predictors as predictors
import utils.utils as utils


PREDICTORS_SPEC = importlib.util.spec_from_file_location(
    "llm_predict_triage_module",
    ROOT / "llm-predict-triage.py",
)
llm_predict_triage = importlib.util.module_from_spec(PREDICTORS_SPEC)
assert PREDICTORS_SPEC.loader is not None
PREDICTORS_SPEC.loader.exec_module(llm_predict_triage)


DEFAULT_ENCODER = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
DEFAULT_KS = [5, 10, 20, 40]
GROUP_ORDER = ["overall", "1", "2", "3", "4", "5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze KTAS KATE nearest-neighbor label averages.")
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS, help="Neighbor counts to evaluate.")
    parser.add_argument(
        "--output",
        default="results/Triage-KTAS/ktas_kate_neighbor_analysis.json",
        help="JSON output path for aggregated metrics.",
    )
    parser.add_argument(
        "--detailed-output",
        default="results/Triage-KTAS/ktas_kate_neighbor_predictions.csv",
        help="CSV output path for per-test-case predictions and neighbor traces.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Optional Markdown summary path. Defaults to the JSON output with .md suffix.",
    )
    parser.add_argument("--encoder-model", default=DEFAULT_ENCODER, help="SentenceTransformer model used for retrieval.")
    parser.add_argument("--encoder-retries", type=int, default=3, help="Number of retries when loading the sentence encoder.")
    parser.add_argument("--encoder-retry-delay", type=int, default=2, help="Seconds to wait between encoder retries.")
    return parser.parse_args()


def load_sentence_encoder_with_retry(model_name: str, max_retries: int, retry_delay_seconds: int) -> Any:
    return llm_predict_triage.load_sentence_encoder_with_retry(
        model_name,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )


def rounded_mean_prediction(labels: list[int]) -> int:
    if not labels:
        raise ValueError("Cannot predict from an empty neighbor label list.")
    mean_label = sum(labels) / len(labels)
    rounded = int(math.floor(mean_label + 0.5))
    return max(1, min(5, rounded))


def compute_metrics_for_subset(truth: pd.Series, predictions: pd.Series) -> dict[str, float]:
    if len(truth) == 0:
        return {"count": 0, "mae": float("nan"), "rmse": float("nan"), "delta": float("nan")}
    errors = predictions.astype(int) - truth.astype(int)
    abs_errors = errors.abs()
    return {
        "count": int(len(truth)),
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "delta": float(errors.mean()),
    }


def build_metrics_records(predictions_df: pd.DataFrame, ks: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for k in ks:
        pred_col = f"predicted_k{k}"
        overall_metrics = compute_metrics_for_subset(predictions_df["true_ktas"], predictions_df[pred_col])
        records.append({"k": k, "group": "overall", **overall_metrics})
        for label in range(1, 6):
            subset = predictions_df[predictions_df["true_ktas"] == label]
            subset_metrics = compute_metrics_for_subset(subset["true_ktas"], subset[pred_col])
            records.append({"k": k, "group": str(label), **subset_metrics})
    return records


def make_markdown(metrics_records: list[dict[str, Any]], detailed_df: pd.DataFrame, ks: list[int]) -> str:
    lines = [
        "# KTAS KATE Neighbor Analysis",
        "",
        f"- Test cases processed: {len(detailed_df)}",
        f"- Neighbor counts: {', '.join(str(k) for k in ks)}",
        "",
        "| k | Group | Count | MAE | RMSE | Delta |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    ordered_records = sorted(
        metrics_records,
        key=lambda record: (record["k"], GROUP_ORDER.index(record["group"])),
    )
    for record in ordered_records:
        lines.append(
            f"| {record['k']} | {record['group']} | {record['count']} | "
            f"{record['mae']:.4f} | {record['rmse']:.4f} | {record['delta']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def analyze_neighbors(
    *,
    training_df: pd.DataFrame,
    test_df: pd.DataFrame,
    complaint_embeddings: np.ndarray,
    diagnosis_embeddings: np.ndarray,
    encoder: Any,
    ks: list[int],
) -> pd.DataFrame:
    max_k = max(ks)
    retriever = predictors.KTASSingleCaseRetriever(
        training_set=training_df,
        complaint_embeddings=complaint_embeddings,
        diagnosis_embeddings=diagnosis_embeddings,
        symptom_encoder=encoder,
    )
    detailed_rows: list[dict[str, Any]] = []
    for test_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        neighbors_df = retriever.retrieve_top_k_examples(row, max_k).reset_index(names="train_index")
        detailed_row: dict[str, Any] = {
            "test_index": int(test_index),
            "true_ktas": int(row["KTAS_expert"]),
        }
        for k in ks:
            top_k_neighbors = neighbors_df.head(k)
            neighbor_labels = [int(label) for label in top_k_neighbors["KTAS_expert"].tolist()]
            detailed_row[f"predicted_k{k}"] = rounded_mean_prediction(neighbor_labels)
            detailed_row[f"neighbor_labels_k{k}"] = json.dumps(neighbor_labels)
            detailed_row[f"neighbor_indices_k{k}"] = json.dumps([int(idx) for idx in top_k_neighbors["train_index"].tolist()])
            detailed_row[f"neighbor_text_k{k}"] = json.dumps(
                [
                    prompts.format_row(example, dataset="triage-ktas", serialization="natural")
                    for _, example in top_k_neighbors.drop(columns=["train_index"]).iterrows()
                ],
                ensure_ascii=True,
            )
        detailed_rows.append(detailed_row)
    return pd.DataFrame(detailed_rows)


def main() -> int:
    args = parse_args()
    ks = sorted(set(args.ks))
    if not ks:
        raise ValueError("At least one k value is required.")
    if any(k <= 0 for k in ks):
        raise ValueError("All k values must be positive integers.")

    dataset = utils._DATASETS["Triage-KTAS"]
    training_df = utils.load_dataset(dataset["training_set_filepath"], dataset["format"], 0, 1_000_000).reset_index(drop=True)
    test_df = utils.load_dataset(dataset["test_set_filepath"], dataset["format"], 0, 1_000_000).reset_index(drop=True)
    complaint_embeddings = np.load(dataset["training_complaint_embeddings_filepath"], allow_pickle=True)
    diagnosis_embeddings = np.load(dataset["training_diagnosis_embeddings_filepath"], allow_pickle=True)
    encoder = load_sentence_encoder_with_retry(
        args.encoder_model,
        max_retries=args.encoder_retries,
        retry_delay_seconds=args.encoder_retry_delay,
    )

    detailed_df = analyze_neighbors(
        training_df=training_df,
        test_df=test_df,
        complaint_embeddings=complaint_embeddings,
        diagnosis_embeddings=diagnosis_embeddings,
        encoder=encoder,
        ks=ks,
    )
    metrics_records = build_metrics_records(detailed_df, ks)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "ks": ks,
                "test_case_count": int(len(detailed_df)),
                "metrics": metrics_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    detailed_output_path = Path(args.detailed_output)
    detailed_output_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_df.to_csv(detailed_output_path, index=False)

    markdown_path = Path(args.markdown_output) if args.markdown_output else output_path.with_suffix(".md")
    markdown_path.write_text(make_markdown(metrics_records, detailed_df, ks), encoding="utf-8")

    print(f"Wrote metrics to {output_path}")
    print(f"Wrote detailed predictions to {detailed_output_path}")
    print(f"Wrote Markdown summary to {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
