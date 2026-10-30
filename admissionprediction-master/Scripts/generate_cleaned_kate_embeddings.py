#!/usr/bin/env python3
"""Generate embeddings for the cleaned ESI KATE retrieval pool."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="Results/5v_relevant_features_kate_train_50000.csv",
        help="Input retrieval-pool CSV.",
    )
    parser.add_argument(
        "--output",
        default="Results/5v_relevant_features_kate_train_50000_embeddings.npy",
        help="Output embeddings path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="SentenceTransformer batch size.",
    )
    return parser.parse_args()


def build_embedding_text(chief_complaint, past_medical_history) -> str:
    parts = []
    for label, value in (
        ("Chief complaint", chief_complaint),
        ("Past medical history", past_medical_history),
    ):
        if pd.isna(value):
            continue
        text = str(value).replace("|", ", ").strip()
        if text:
            parts.append(f"{label}: {text}.")
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    texts = [
        build_embedding_text(row.get("chief_complaint"), row.get("past_medical_history"))
        for _, row in df.iterrows()
    ]

    model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"rows={len(df)}")
    print(f"embedding_shape={embeddings.shape}")


if __name__ == "__main__":
    main()
