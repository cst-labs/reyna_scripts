#!/usr/bin/env python3
"""
Convert KTAS edtriage split CSVs from numeric-coded categorical fields to the
text-form schema expected by the KTAS prompt formatter.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = {
    "Sex",
    "Arrival mode",
    "Injury",
    "Mental",
}


def normalize_value(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value).strip()


def build_code_map(numeric_df: pd.DataFrame, text_df: pd.DataFrame, column: str) -> dict[object, str]:
    mapping: dict[object, str] = {}
    pairs = pd.DataFrame({"code": numeric_df[column], "label": text_df[column]}).dropna().drop_duplicates()
    for _, pair in pairs.iterrows():
        mapping[normalize_value(pair["code"])] = str(pair["label"]).strip()
    return mapping


def convert_dataframe(df: pd.DataFrame, mappings: dict[str, dict[object, str]]) -> pd.DataFrame:
    converted = df.copy()
    for column, mapping in mappings.items():
        if column not in converted.columns:
            continue
        converted[column] = converted[column].map(lambda value: mapping.get(normalize_value(value), value))
    return converted


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ktas_dir = repo_root / "data" / "KTAS"

    train_numeric_df = pd.read_csv(ktas_dir / "train_numeric.csv")
    train_text_df = pd.read_csv(ktas_dir / "train.csv")

    mappings = {
        column: build_code_map(train_numeric_df, train_text_df, column)
        for column in NUMERIC_COLUMNS
    }

    inputs = {
        ktas_dir / "edtriage_split" / "train_with_reasoning.csv":
            ktas_dir / "edtriage_split" / "train_with_reasoning_text.csv",
        ktas_dir / "edtriage_split" / "test.csv":
            ktas_dir / "edtriage_split" / "test_text.csv",
    }

    for input_path, output_path in inputs.items():
        df = pd.read_csv(input_path)
        converted = convert_dataframe(df, mappings)
        converted.to_csv(output_path, index=False)
        print(f"Wrote {len(converted)} rows to {output_path}")


if __name__ == "__main__":
    main()
