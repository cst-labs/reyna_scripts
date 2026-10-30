#!/usr/bin/env python3
"""Create a stratified train/test split for the cleaned ESI dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="Results/5v_relevant_features.csv",
        help="Input cleaned CSV.",
    )
    parser.add_argument(
        "--train-output",
        default="Results/5v_relevant_features_train.csv",
        help="Output training CSV.",
    )
    parser.add_argument(
        "--test-output",
        default="Results/5v_relevant_features_test_500.csv",
        help="Output test CSV.",
    )
    parser.add_argument(
        "--target",
        default="esi",
        help="Target column used for stratification.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Number of rows to place in the test split.",
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
    train_output = Path(args.train_output)
    test_output = Path(args.test_output)

    df = pd.read_csv(input_path)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(df, df[args.target]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"input={input_path}")
    print(f"train_output={train_output}")
    print(f"test_output={test_output}")
    print(f"rows_total={len(df)}")
    print(f"rows_train={len(train_df)}")
    print(f"rows_test={len(test_df)}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
