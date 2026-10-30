#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.utils as utils
from utils.ktas_bqr import (
    build_ktas_candidate_pool,
    compute_adjacent_boundary_counts,
    compute_feature_statistics,
    compute_label_entropy,
    compute_local_feature_disagreement,
    normalize_feature_disagreement_scores,
)


PREDICTORS_SPEC = importlib.util.spec_from_file_location(
    "llm_predict_triage_module",
    ROOT / "llm-predict-triage.py",
)
llm_predict_triage = importlib.util.module_from_spec(PREDICTORS_SPEC)
assert PREDICTORS_SPEC.loader is not None
PREDICTORS_SPEC.loader.exec_module(llm_predict_triage)


DEFAULT_ENCODER = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute disagreement-aware KTAS retrieval metadata.")
    parser.add_argument("--neighborhood-size", type=int, default=40, help="Number of local neighbors per training example.")
    parser.add_argument("--output", required=True, help="JSONL output path.")
    parser.add_argument("--encoder-model", default=DEFAULT_ENCODER, help="SentenceTransformer model used for retrieval.")
    parser.add_argument("--encoder-retries", type=int, default=3, help="Number of retries when loading the sentence encoder.")
    parser.add_argument("--encoder-retry-delay", type=int, default=2, help="Seconds to wait between encoder retries.")
    return parser.parse_args()


def load_sentence_encoder_with_retry(model_name: str, max_retries: int, retry_delay_seconds: int):
    return llm_predict_triage.load_sentence_encoder_with_retry(
        model_name,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )


def main() -> int:
    args = parse_args()
    if args.neighborhood_size <= 0:
        raise ValueError("--neighborhood-size must be positive.")

    dataset = utils._DATASETS["Triage-KTAS"]
    training_df = utils.load_dataset(dataset["training_set_filepath"], dataset["format"], 0, 1_000_000).reset_index(drop=True)
    complaint_embeddings = np.load(dataset["training_complaint_embeddings_filepath"], allow_pickle=True)
    diagnosis_embeddings = np.load(dataset["training_diagnosis_embeddings_filepath"], allow_pickle=True)
    encoder = load_sentence_encoder_with_retry(
        args.encoder_model,
        max_retries=args.encoder_retries,
        retry_delay_seconds=args.encoder_retry_delay,
    )

    feature_stats = compute_feature_statistics(training_df)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_feature_scores: dict[int, float] = {}
    intermediate_records: list[dict[str, object]] = []
    for example_id, row in tqdm(training_df.iterrows(), total=len(training_df)):
        candidate_pool = build_ktas_candidate_pool(
            row=row,
            training_set=training_df,
            complaint_embeddings=complaint_embeddings,
            diagnosis_embeddings=diagnosis_embeddings,
            symptom_encoder=encoder,
            candidate_pool_size=args.neighborhood_size + 1,
        )
        neighbor_ids = [
            candidate.example_id
            for candidate in candidate_pool
            if candidate.example_id != int(example_id)
        ][: args.neighborhood_size]
        neighbors_df = training_df.loc[neighbor_ids]
        neighbor_labels = [int(label) for label in neighbors_df["KTAS_expert"].tolist()]
        histogram = {
            int(label): int(count)
            for label, count in Counter(neighbor_labels).items()
        }
        raw_feature_scores[int(example_id)] = compute_local_feature_disagreement(
            row,
            neighbors_df,
            feature_stats,
        )
        intermediate_records.append(
            {
                "example_id": int(example_id),
                "ktas_label": int(row["KTAS_expert"]),
                "neighbor_ids": neighbor_ids,
                "neighbor_label_histogram": histogram,
                "disagreement_score": compute_label_entropy(histogram),
                "adjacent_boundary_counts": compute_adjacent_boundary_counts(neighbor_labels),
                "neighborhood_size": args.neighborhood_size,
            }
        )

    normalized_feature_scores = normalize_feature_disagreement_scores(raw_feature_scores)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in intermediate_records:
            record["feature_disagreement_score"] = normalized_feature_scores[record["example_id"]]
            handle.write(json.dumps(record) + "\n")

    print(f"Disagreement metadata saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
