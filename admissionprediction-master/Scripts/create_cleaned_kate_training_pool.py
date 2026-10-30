#!/usr/bin/env python3
"""Create a 50k-row stratified retrieval pool for cleaned ESI KATE runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="Results/5v_relevant_features_train.csv",
        help="Input training CSV.",
    )
    parser.add_argument(
        "--output",
        default="Results/5v_relevant_features_kate_train_50000.csv",
        help="Output retrieval-pool CSV.",
    )
    parser.add_argument(
        "--target",
        default="esi",
        help="Target column used for stratification.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of rows to include in the retrieval pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=args.sample_size, random_state=args.seed)
    sample_idx, _ = next(splitter.split(df, df[args.target]))
    sample_df = df.iloc[sample_idx].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"rows_total={len(df)}")
    print(f"rows_sample={len(sample_df)}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
