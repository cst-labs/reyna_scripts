#!/usr/bin/env python3
"""Backfill retrieved KTAS KATE examples into existing prediction CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.prompts as prompts
import utils.utils as utils
from utils.predictors import KTASSingleCaseRetriever, SentenceTransformer


DEFAULT_INPUT_GLOB = "results/Triage-KTAS/Triage-KTAS_KATE_*.csv"
DEFAULT_ENCODER = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
SERIALIZATION_CANDIDATES = [
    "natural_full",
    "natural_sex_race",
    "newline",
    "commas",
    "spaces",
    "natural",
    "json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover KTAS KATE retrieved examples and write them into existing CSV files."
    )
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="Glob for KTAS KATE prediction CSVs to backfill.",
    )
    parser.add_argument(
        "--k-shots",
        type=int,
        help="Override the number of retrieved examples instead of inferring from each filename.",
    )
    parser.add_argument(
        "--serialization",
        choices=SERIALIZATION_CANDIDATES,
        help="Override serialization instead of inferring from each filename.",
    )
    parser.add_argument(
        "--output-suffix",
        default=".with_examples",
        help="Suffix to append before .csv when not writing in place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the matched CSVs in place.",
    )
    parser.add_argument(
        "--encoder-model",
        default=DEFAULT_ENCODER,
        help="SentenceTransformer model used for KTAS retrieval.",
    )
    return parser.parse_args()


def infer_k_shots(path: Path) -> int:
    match = re.search(r"_(\d+)_shots(?:_|$)", path.stem)
    if not match:
        raise ValueError(f"Could not infer k_shots from filename: {path.name}")
    return int(match.group(1))


def infer_serialization(path: Path) -> str:
    stem = path.stem
    for candidate in SERIALIZATION_CANDIDATES:
        if f"_{candidate}_" in stem:
            return candidate
    raise ValueError(f"Could not infer serialization from filename: {path.name}")


def build_retriever(encoder_model: str) -> KTASSingleCaseRetriever:
    dataset = utils._DATASETS["Triage-KTAS"]
    training_df = utils.load_dataset(dataset["training_set_filepath"], dataset["format"], 0, 1000000)
    complaint_embeddings = np.load(dataset["training_complaint_embeddings_filepath"], allow_pickle=True)
    diagnosis_embeddings = np.load(dataset["training_diagnosis_embeddings_filepath"], allow_pickle=True)
    encoder = SentenceTransformer(encoder_model)
    return KTASSingleCaseRetriever(
        training_set=training_df,
        complaint_embeddings=complaint_embeddings,
        diagnosis_embeddings=diagnosis_embeddings,
        symptom_encoder=encoder,
    )


def format_examples_for_row(
    row: pd.Series,
    *,
    retriever: KTASSingleCaseRetriever,
    k_shots: int,
    serialization: str,
) -> str:
    examples = retriever.retrieve_top_k_examples(row, k_shots)
    return "\n\n".join(
        prompts.format_row(example, dataset="triage-ktas", serialization=serialization)
        + f"\nAcuity Level: {example['KTAS_expert']}"
        for _, example in examples.iterrows()
    )


def backfill_dataframe(
    df: pd.DataFrame,
    *,
    retriever: KTASSingleCaseRetriever,
    k_shots: int,
    serialization: str,
) -> pd.DataFrame:
    result = df.copy()
    result["Retrieved_Examples"] = [
        format_examples_for_row(row, retriever=retriever, k_shots=k_shots, serialization=serialization)
        for _, row in result.iterrows()
    ]
    return result


def output_path_for(path: Path, *, in_place: bool, output_suffix: str) -> Path:
    if in_place:
        return path
    return path.with_name(f"{path.stem}{output_suffix}{path.suffix}")


def main() -> int:
    args = parse_args()
    paths = sorted(Path().glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.input_glob}")

    retriever = build_retriever(args.encoder_model)

    for path in paths:
        k_shots = args.k_shots if args.k_shots is not None else infer_k_shots(path)
        serialization = args.serialization or infer_serialization(path)
        df = pd.read_csv(path)
        updated = backfill_dataframe(
            df,
            retriever=retriever,
            k_shots=k_shots,
            serialization=serialization,
        )
        destination = output_path_for(path, in_place=args.in_place, output_suffix=args.output_suffix)
        updated.to_csv(destination, index=False)
        print(
            f"Backfilled {path} -> {destination} "
            f"(k_shots={k_shots}, serialization={serialization}, rows={len(updated)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
