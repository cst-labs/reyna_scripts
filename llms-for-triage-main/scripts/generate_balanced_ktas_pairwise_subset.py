#!/usr/bin/env python3

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import List

import pandas as pd


def derive_pair_label(ktas_a: int, ktas_b: int) -> str | None:
    if ktas_a < ktas_b:
        return "A_BEFORE_B"
    if ktas_b < ktas_a:
        return "B_BEFORE_A"
    return None


def select_balanced_cases(
    df: pd.DataFrame,
    *,
    per_grade: int,
    seed: int,
) -> pd.DataFrame:
    selected_groups: List[pd.DataFrame] = []
    for grade in [1, 2, 3, 4, 5]:
        group = df[df["KTAS_expert"] == grade].copy()
        if len(group) < per_grade:
            raise ValueError(
                f"KTAS grade {grade} has only {len(group)} rows, fewer than requested {per_grade}."
            )
        if len(group) == per_grade:
            sampled = group.copy()
        else:
            sampled = group.sample(n=per_grade, random_state=seed)
        sampled = sampled.sort_index().reset_index(names="source_index")
        sampled["_slot"] = range(len(sampled))
        selected_groups.append(sampled)
    balanced = pd.concat(selected_groups, ignore_index=True)
    balanced = balanced.sort_values(["_slot", "KTAS_expert", "source_index"]).reset_index(drop=True)
    return balanced.drop(columns=["_slot"])


def build_pairwise_rows(
    balanced_df: pd.DataFrame,
    *,
    include_equal_ktas: bool,
) -> pd.DataFrame:
    rows = []
    for left_idx, right_idx in combinations(range(len(balanced_df)), 2):
        patient_a = balanced_df.iloc[left_idx]
        patient_b = balanced_df.iloc[right_idx]
        label = derive_pair_label(int(patient_a["KTAS_expert"]), int(patient_b["KTAS_expert"]))
        if label is None and not include_equal_ktas:
            continue

        pair_row = {
            "Pair_Label": label,
            "A_source_index": int(patient_a["source_index"]),
            "B_source_index": int(patient_b["source_index"]),
        }
        for column, value in patient_a.drop(labels=["source_index"]).items():
            pair_row[f"A_{column}"] = value
        for column, value in patient_b.drop(labels=["source_index"]).items():
            pair_row[f"B_{column}"] = value
        rows.append(pair_row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a balanced KTAS test subset and unique pairwise comparisons."
    )
    parser.add_argument(
        "--input",
        default="data/KTAS/test.csv",
        help="Source KTAS test CSV.",
    )
    parser.add_argument(
        "--per_grade",
        type=int,
        default=12,
        help="Number of cases to keep for each KTAS_expert level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic downsampling.",
    )
    parser.add_argument(
        "--cases_output",
        default="data/KTAS/test_balanced_12_per_grade.csv",
        help="Output CSV for the balanced case subset.",
    )
    parser.add_argument(
        "--pairs_output",
        default="data/KTAS/test_balanced_12_per_grade_pairwise.csv",
        help="Output CSV for the unique pairwise rows.",
    )
    parser.add_argument(
        "--include_equal_ktas",
        action="store_true",
        help="Include same-KTAS pairs. Their Pair_Label will be empty.",
    )
    args = parser.parse_args()

    source_df = pd.read_csv(args.input)
    balanced_df = select_balanced_cases(
        source_df,
        per_grade=args.per_grade,
        seed=args.seed,
    )
    pairwise_df = build_pairwise_rows(
        balanced_df,
        include_equal_ktas=args.include_equal_ktas,
    )

    cases_output = Path(args.cases_output)
    pairs_output = Path(args.pairs_output)
    cases_output.parent.mkdir(parents=True, exist_ok=True)
    pairs_output.parent.mkdir(parents=True, exist_ok=True)

    balanced_df.to_csv(cases_output, index=False)
    pairwise_df.to_csv(pairs_output, index=False)

    print(f"Saved balanced cases to {cases_output}")
    print(f"Balanced case count: {len(balanced_df)}")
    print("KTAS counts:")
    print(balanced_df["KTAS_expert"].value_counts().sort_index().to_string())
    print(f"Saved pairwise rows to {pairs_output}")
    print(f"Unique pair count: {len(pairwise_df)}")


if __name__ == "__main__":
    main()
