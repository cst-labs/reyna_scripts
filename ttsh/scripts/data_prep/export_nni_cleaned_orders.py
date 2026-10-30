"""
Export the final NNI cleaned-order CSV.

Inputs:
  - merged/ct_orders_cleaned.xlsx
      Canonical cleaned imaging data from clean_ct_orders.py. Its Site column
      is derived from the Ordering Question Responses' Performing Location;
      when a merged row spans conflicting parsed locations, clean_ct_orders.py
      keeps the site from the original row with the earliest Exam Start.
Output:
  - data/nni_cleaned_orders.csv

Final inclusion rules:
  1. Keep rows whose canonical cleaned Site is NNI.
  2. Keep only positive, non-missing scan_duration.

scan_duration is computed from the chronologically ordered retained NNI scans:
  - if this scan's Exam End is before the next scan's Exam Start, use
      Exam End - Exam Start + 17.9 minutes
  - if this scan's Exam End is at or after the next scan's Exam Start, use
      next Exam Start - this Exam Start
  - boundary rows without a valid next scan fall back to
      Exam End - Exam Start
Zero-minute, negative, or missing durations are dropped.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[2]
CSV_PATH = BASE / "data" / "nni_cleaned_orders.csv"
CLEAN_PATH = BASE / "merged" / "ct_orders_cleaned.xlsx"

NNI = "National Neuroscience Institute"
TTSH = "Tan Tock Seng Hospital"
MIXED_NNI_TTSH = f"{NNI} + {TTSH}"

ID_SPLITTER = re.compile(r"\s*[,;]\s*")


def order_ids(value) -> list[str]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [re.sub(r"\.0$", "", item.strip()) for item in ID_SPLITTER.split(text) if item.strip()]


def order_key(row: pd.Series) -> str:
    ids = order_ids(row.get("Merged Order IDs", "")) or order_ids(row.get("Order ID", ""))
    return "|".join(sorted(set(ids), key=lambda item: (len(item), item)))


def datetimes_from_columns(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    times = (
        df[time_col]
        .astype(str)
        .str.strip()
        .str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)", expand=False)
    )
    return pd.to_datetime(dates + " " + times, errors="coerce")


def compute_scan_duration(df: pd.DataFrame) -> pd.Series:
    start = datetimes_from_columns(df, "Exam Start Date", "Exam Start Time")
    end = datetimes_from_columns(df, "Exam End Date", "Exam End Time")

    sort_cols = ["_exam_start_sort"]
    if "_order_key" in df.columns:
        sort_cols.append("_order_key")

    timeline = pd.DataFrame(
        {
            "_exam_start_sort": start,
            "_exam_end_sort": end,
        },
        index=df.index,
    )
    if "_order_key" in df.columns:
        timeline["_order_key"] = df["_order_key"]

    timeline = timeline.sort_values(sort_cols, kind="stable")
    timeline["_prev_exam_end"] = timeline["_exam_end_sort"].shift(1)
    timeline["_next_exam_start"] = timeline["_exam_start_sort"].shift(-1)

    exam_duration = (
        timeline["_exam_end_sort"] - timeline["_exam_start_sort"]
    ).dt.total_seconds() / 60
    next_start_duration = (
        timeline["_next_exam_start"] - timeline["_exam_start_sort"]
    ).dt.total_seconds() / 60

    valid_exam_duration = exam_duration.notna() & (exam_duration >= 0)
    valid_next_start_duration = (
        next_start_duration.notna() & (next_start_duration >= 0)
    )
    ends_before_next_start = (
        timeline["_exam_end_sort"].notna()
        & timeline["_next_exam_start"].notna()
        & (timeline["_exam_end_sort"] < timeline["_next_exam_start"])
    )
    ends_at_or_after_next_start = (
        timeline["_exam_end_sort"].notna()
        & timeline["_next_exam_start"].notna()
        & (timeline["_exam_end_sort"] >= timeline["_next_exam_start"])
    )

    duration = exam_duration.copy()
    add_gap_buffer = ends_before_next_start & valid_exam_duration
    duration.loc[add_gap_buffer] = exam_duration.loc[add_gap_buffer] + 17.9
    duration.loc[ends_at_or_after_next_start] = next_start_duration.loc[
        ends_at_or_after_next_start
    ]
    no_valid_next_scan = ~valid_next_start_duration & valid_exam_duration
    duration.loc[no_valid_next_scan] = exam_duration.loc[no_valid_next_scan]
    return duration.reindex(df.index)


def main() -> None:
    original_columns = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()

    cleaned = pd.read_excel(CLEAN_PATH, sheet_name="Imaging")
    nni = cleaned[cleaned["Site"].astype(str).str.strip().str.upper().eq("NNI")].copy()
    nni["_order_key"] = nni.apply(order_key, axis=1)

    for column in original_columns:
        if column not in nni.columns:
            nni[column] = pd.NA
    out = nni[original_columns + ["_order_key"]].copy()

    keep_location = pd.Series(True, index=out.index)

    out.loc[keep_location, "scan_duration"] = compute_scan_duration(nni.loc[keep_location])
    numeric_duration = pd.to_numeric(out["scan_duration"], errors="coerce")
    keep_duration = numeric_duration.notna() & (numeric_duration > 0)
    keep = keep_location & keep_duration

    filtered = out.loc[keep].copy()
    filtered["_exam_start_sort"] = datetimes_from_columns(
        filtered, "Exam Start Date", "Exam Start Time"
    )
    filtered = (
        filtered.sort_values(["_exam_start_sort", "_order_key"], kind="stable")
        .drop(columns=["_exam_start_sort", "_order_key"])
    )

    backup_path = CSV_PATH.with_suffix(
        CSV_PATH.suffix + f".bak_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if CSV_PATH.exists():
        shutil.copy2(CSV_PATH, backup_path)

    filtered.to_csv(CSV_PATH, index=False)

    removed = out.loc[~keep].copy()
    missing_or_negative_duration = int((keep_location & (numeric_duration.isna() | (numeric_duration < 0))).sum())
    zero_duration = int((keep_location & numeric_duration.eq(0)).sum())

    print(f"Backed up previous CSV to: {backup_path}")
    print(f"Canonical cleaned NNI rows: {len(nni):,}")
    print(f"Rows written: {len(filtered):,}")
    print(f"Rows removed: {len(removed):,}")
    print(f"  missing/negative scan_duration: {missing_or_negative_duration:,}")
    print(f"  zero scan_duration: {zero_duration:,}")


if __name__ == "__main__":
    main()
