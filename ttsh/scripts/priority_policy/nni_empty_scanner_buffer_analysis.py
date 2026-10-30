"""
nni_empty_scanner_buffer_analysis.py

Reconstructs the NNI scanner system from data/nni_cleaned_orders.csv and
identifies periods where the reconstructed system was empty: no order waiting
and no scan in progress. For the next order after each empty period, measures
how long that order had to buffer before entering the scanner.

Reads:
  data/nni_cleaned_orders.csv

Outputs:
  analysis/nni_empty_scanner_buffer_analysis.xlsx
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
ANALYSIS = os.path.join(BASE, "analysis")
os.makedirs(ANALYSIS, exist_ok=True)

IN_CSV = os.path.join(DATA, "nni_cleaned_orders.csv")
OUT_XLSX = os.path.join(ANALYSIS, "nni_empty_scanner_buffer_analysis.xlsx")


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 50)


def datetimes_from_columns(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    times = (
        df[time_col]
        .astype(str)
        .str.strip()
        .str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)", expand=False)
    )
    return pd.to_datetime(dates + " " + times, errors="coerce")


def split_joined(value) -> list[str]:
    if value is None:
        return [""]
    try:
        if isinstance(value, float) and np.isnan(value):
            return [""]
    except Exception:
        pass
    text = str(value).strip()
    return text.split("; ") if "; " in text else [text]


def earliest_joined_datetime(date_val, time_val):
    """Parent Order Date/Time can contain '; '-joined values after order
    merging. Use the earliest candidate as the order's arrival time."""
    date_parts = split_joined(date_val)
    time_parts = split_joined(time_val)
    n_parts = max(len(date_parts), len(time_parts))
    if len(date_parts) < n_parts:
        date_parts = date_parts + [date_parts[-1]] * (n_parts - len(date_parts))
    if len(time_parts) < n_parts:
        time_parts = time_parts + [time_parts[-1]] * (n_parts - len(time_parts))

    candidates = []
    for date_part, time_part in zip(date_parts, time_parts):
        date = pd.to_datetime(date_part, errors="coerce")
        time = (
            pd.Series([time_part])
            .astype(str)
            .str.strip()
            .str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)", expand=False)
            .iloc[0]
        )
        if pd.notna(date) and pd.notna(time):
            dt = pd.to_datetime(f"{date.strftime('%Y-%m-%d')} {time}", errors="coerce")
            if pd.notna(dt):
                candidates.append(dt)
    return min(candidates) if candidates else pd.NaT


def summarize_dist(vals: pd.Series) -> dict:
    vals = vals.dropna()
    if len(vals) == 0:
        return {k: None for k in ["N", "Mean", "Median", "P25", "P75", "P90", "P95", "Max"]}
    return {
        "N": int(len(vals)),
        "Mean": round(float(vals.mean()), 1),
        "Median": round(float(vals.median()), 1),
        "P25": round(float(vals.quantile(0.25)), 1),
        "P75": round(float(vals.quantile(0.75)), 1),
        "P90": round(float(vals.quantile(0.90)), 1),
        "P95": round(float(vals.quantile(0.95)), 1),
        "Max": round(float(vals.max()), 1),
    }


def exclusion_reason(row: pd.Series) -> str | None:
    reasons = []
    if pd.isna(row["parent_order_dt"]):
        reasons.append("missing_parent_order_dt")
    if pd.isna(row["exam_start_dt"]):
        reasons.append("missing_exam_start_dt")
    if pd.isna(row["scan_duration_min"]):
        reasons.append("missing_scan_duration")
    elif row["scan_duration_min"] < 0:
        reasons.append("negative_scan_duration")
    if pd.notna(row["wait_to_scanner_min"]) and row["wait_to_scanner_min"] < 0:
        reasons.append("negative_wait_to_scanner")
    return "; ".join(reasons) if reasons else None


def build_empty_intervals(valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = []
    for _, r in valid.iterrows():
        events.append((r["parent_order_dt"], 1, "arrival", r["_row_id"]))
        events.append((r["scan_end_dt"], -1, "scan_end", r["_row_id"]))

    # Ends before arrivals on exact ties, so a scan ending at the same minute
    # a new order appears is treated as an empty instant before the arrival.
    events.sort(key=lambda e: (e[0], e[1]))

    rows = []
    intervals = []
    running = 0
    empty_start_dt = None

    for dt, delta, event_type, row_id in events:
        if event_type == "arrival" and running == 0 and empty_start_dt is not None:
            intervals.append({
                "empty_start_dt": empty_start_dt,
                "next_order_dt": dt,
                "next_row_id": row_id,
                "empty_gap_before_order_min": (dt - empty_start_dt).total_seconds() / 60,
            })
            empty_start_dt = None

        running += delta
        if running < 0:
            raise AssertionError(f"Active system count went negative at {dt}")

        rows.append({
            "datetime": dt,
            "event_type": event_type,
            "_row_id": row_id,
            "active_system_count": running,
        })

        if event_type == "scan_end" and running == 0:
            empty_start_dt = dt

    return pd.DataFrame(intervals), pd.DataFrame(rows)


print(f"Loading {IN_CSV} ...")
orders = pd.read_csv(IN_CSV)
orders["_row_id"] = range(len(orders))
orders["parent_order_dt"] = orders.apply(
    lambda r: earliest_joined_datetime(r["Parent Order Date"], r["Parent Order Time"]),
    axis=1,
)
orders["exam_start_dt"] = datetimes_from_columns(orders, "Exam Start Date", "Exam Start Time")
orders["scan_duration_min"] = pd.to_numeric(orders["scan_duration"], errors="coerce")
orders["scan_end_dt"] = orders["exam_start_dt"] + pd.to_timedelta(orders["scan_duration_min"], unit="m")
orders["wait_to_scanner_min"] = (
    orders["exam_start_dt"] - orders["parent_order_dt"]
).dt.total_seconds() / 60
orders["exclusion_reason"] = orders.apply(exclusion_reason, axis=1)

valid = orders[orders["exclusion_reason"].isna()].copy()
excluded = orders[orders["exclusion_reason"].notna()].copy()

assert (valid["parent_order_dt"] <= valid["exam_start_dt"]).all()
assert (valid["exam_start_dt"] <= valid["scan_end_dt"]).all()

print(f"  Source rows: {len(orders):,}")
print(f"  Included rows: {len(valid):,}")
print(f"  Excluded rows: {len(excluded):,}")
if len(excluded):
    print("  Exclusion reasons:")
    print(excluded["exclusion_reason"].value_counts().to_string())

empty_intervals, system_events = build_empty_intervals(valid)

context_cols = [
    "_row_id", "month", "CSN", "Order ID", "Order Description", "Scan Priority",
    "Procedure Location", "Performing Location", "parent_order_dt", "exam_start_dt",
    "scan_duration_min", "scan_end_dt", "wait_to_scanner_min",
]
context_cols = [c for c in context_cols if c in valid.columns]
order_context = valid[context_cols].copy()

empty_intervals = empty_intervals.merge(
    order_context,
    left_on="next_row_id",
    right_on="_row_id",
    how="left",
)
empty_intervals["buffer_to_scanner_min"] = (
    empty_intervals["exam_start_dt"] - empty_intervals["next_order_dt"]
).dt.total_seconds() / 60
empty_intervals["buffer_check_delta_min"] = (
    empty_intervals["buffer_to_scanner_min"] - empty_intervals["wait_to_scanner_min"]
).abs()
assert np.isclose(empty_intervals["buffer_check_delta_min"].fillna(0), 0).all()

flagged_orders = empty_intervals[empty_intervals["buffer_to_scanner_min"] > 0].copy()

summary_rows = [
    {"Metric": "Source rows", "Value": len(orders)},
    {"Metric": "Included rows", "Value": len(valid)},
    {"Metric": "Excluded rows", "Value": len(excluded)},
    {"Metric": "Empty intervals followed by a next order", "Value": len(empty_intervals)},
    {"Metric": "Next orders that buffered before scanner entry", "Value": len(flagged_orders)},
]
for reason, count in excluded["exclusion_reason"].value_counts().items():
    summary_rows.append({"Metric": f"Excluded: {reason}", "Value": int(count)})

for metric, col in [
    ("Empty gap before next order (mins)", "empty_gap_before_order_min"),
    ("Buffer to scanner after next order (mins)", "buffer_to_scanner_min"),
]:
    stats = summarize_dist(empty_intervals[col])
    summary_rows.append({"Metric": f"-- {metric} --", "Value": ""})
    for stat_name, value in stats.items():
        summary_rows.append({"Metric": f"{metric}: {stat_name}", "Value": value})

for priority, sub in empty_intervals.groupby("Scan Priority", dropna=False):
    label = "Missing" if pd.isna(priority) else priority
    summary_rows.append({
        "Metric": f"Empty-interval next orders: {label}",
        "Value": int(len(sub)),
    })
    summary_rows.append({
        "Metric": f"Median buffer to scanner (mins): {label}",
        "Value": round(float(sub["buffer_to_scanner_min"].median()), 1),
    })

summary = pd.DataFrame(summary_rows)

empty_out_cols = [
    "empty_start_dt", "next_order_dt", "empty_gap_before_order_min",
    "exam_start_dt", "buffer_to_scanner_min", "scan_duration_min",
    "scan_end_dt", "Order ID", "CSN", "Scan Priority", "Order Description",
    "month", "Procedure Location", "Performing Location",
]
empty_out_cols = [c for c in empty_out_cols if c in empty_intervals.columns]
empty_intervals_out = (
    empty_intervals[empty_out_cols]
    .sort_values(["empty_start_dt", "next_order_dt"], kind="stable")
    .reset_index(drop=True)
)
flagged_orders_out = (
    flagged_orders[empty_out_cols]
    .sort_values(["buffer_to_scanner_min", "next_order_dt"], ascending=[False, True], kind="stable")
    .reset_index(drop=True)
)
excluded_out_cols = [
    "exclusion_reason", "month", "CSN", "Order ID", "Order Description",
    "Parent Order Date", "Parent Order Time", "Exam Start Date", "Exam Start Time",
    "scan_duration", "parent_order_dt", "exam_start_dt", "scan_duration_min",
    "wait_to_scanner_min", "Scan Priority", "Procedure Location", "Performing Location",
]
excluded_out_cols = [c for c in excluded_out_cols if c in excluded.columns]
excluded_out = excluded[excluded_out_cols].reset_index(drop=True)

print(f"  Empty intervals followed by a next order: {len(empty_intervals_out):,}")
print(f"  Flagged buffered next orders: {len(flagged_orders_out):,}")
print(f"Writing {OUT_XLSX} ...")
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    empty_intervals_out.to_excel(writer, sheet_name="Empty_Intervals", index=False)
    flagged_orders_out.to_excel(writer, sheet_name="Flagged_Orders", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)
    excluded_out.to_excel(writer, sheet_name="Excluded_Rows", index=False)
    system_events.to_excel(writer, sheet_name="System_Events", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
