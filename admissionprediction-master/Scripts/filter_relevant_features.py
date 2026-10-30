#!/usr/bin/env python3
"""Keep only the requested admission-prediction feature groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import pyreadr
except ImportError:  # pragma: no cover
    pyreadr = None


RESPONSE_COLUMNS = ["esi"]
DEMOGRAPHIC_COLUMNS = ["age", "gender", "arrivalmode", "previousdispo"]

PMH_START = "2ndarymalig"
PMH_END = "whtblooddx"
TRIAGE_VITAL_COLUMNS = [
    "triage_vital_hr",
    "triage_vital_rr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_o2",
    "triage_vital_temp",
]
CHIEF_COMPLAINT_START = "cc_abdominalcramping"
CHIEF_COMPLAINT_END = "cc_wristpain"
PAST_MEDICAL_HISTORY_COLUMN = "past_medical_history"
CHIEF_COMPLAINT_COLUMN = "chief_complaint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter the admission prediction dataset to the requested "
            "response, demographics, past medical history, triage vital "
            "signs, and chief complaints."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default="Results/5v_cleandf.RData",
        help="Input dataset path (.RData or .csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="Results/5v_relevant_features.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--keep-empty-rows",
        action="store_true",
        help="Keep rows with missing values after feature selection.",
    )
    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".rdata":
        if pyreadr is None:
            raise SystemExit(
                "pyreadr is required to read .RData files. "
                "Install it with: python3 -m pip install pyreadr"
            )
        result = pyreadr.read_r(str(path))
        if not result:
            raise SystemExit(f"No objects found in {path}")
        _, obj = next(iter(result.items()))
        if not isinstance(obj, pd.DataFrame):
            raise SystemExit(f"The first object in {path} is not a data frame.")
        return obj

    raise SystemExit(f"Unsupported input format: {path.suffix}")


def slice_columns(df: pd.DataFrame, start: str, end: str) -> list[str]:
    columns = list(df.columns)
    try:
        start_index = columns.index(start)
        end_index = columns.index(end)
    except ValueError as exc:
        raise SystemExit(f"Missing expected column: {exc}") from exc

    if start_index > end_index:
        raise SystemExit(f"Column order is invalid for range {start} -> {end}")

    return columns[start_index : end_index + 1]


def build_selected_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    expected_columns = RESPONSE_COLUMNS + DEMOGRAPHIC_COLUMNS + TRIAGE_VITAL_COLUMNS
    missing_expected = [col for col in expected_columns if col not in df.columns]
    if missing_expected:
        raise SystemExit(
            "Missing expected columns: " + ", ".join(missing_expected)
        )

    selected = list(RESPONSE_COLUMNS)
    selected.extend(DEMOGRAPHIC_COLUMNS)
    selected.extend(TRIAGE_VITAL_COLUMNS)

    return (
        list(dict.fromkeys(selected)),
        slice_columns(df, PMH_START, PMH_END),
        slice_columns(df, CHIEF_COMPLAINT_START, CHIEF_COMPLAINT_END),
    )


def derive_text_field(
    df: pd.DataFrame,
    source_columns: list[str],
    *,
    prefix_to_strip: str = "",
) -> pd.Series:
    active = df.loc[:, source_columns].fillna(0).eq(1)
    labels = [col.removeprefix(prefix_to_strip) for col in source_columns]

    def join_active(row: pd.Series) -> str | pd.NA:
        values = [label for label, is_active in zip(labels, row.tolist()) if is_active]
        return "|".join(values) if values else pd.NA

    return active.apply(join_active, axis=1)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_dataframe(input_path)
    selected_columns, pmh_columns, chief_complaint_columns = build_selected_columns(df)
    filtered_df = df.loc[:, selected_columns].copy()
    filtered_df[PAST_MEDICAL_HISTORY_COLUMN] = derive_text_field(df, pmh_columns)
    filtered_df[CHIEF_COMPLAINT_COLUMN] = derive_text_field(
        df,
        chief_complaint_columns,
        prefix_to_strip="cc_",
    )

    rows_before = len(filtered_df)
    if not args.keep_empty_rows:
        filtered_df = filtered_df.dropna(axis=0, how="any")
    rows_after = len(filtered_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"columns_kept={len(filtered_df.columns)}")
    print(f"rows_before={rows_before}")
    print(f"rows_after={rows_after}")
    print(f"rows_removed={rows_before - rows_after}")


if __name__ == "__main__":
    main()
