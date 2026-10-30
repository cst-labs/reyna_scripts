"""Estimate NNI waiting times under one non-preemptive CT scanner.

The simulation uses ``scan_duration`` from data/nni_cleaned_orders.csv as
each order's service time. That column is produced by
export_nni_cleaned_orders.py from the chronological NNI scan sequence:

* Exam End < next Exam Start:
    Exam End - Exam Start + 17.9 minutes
* Exam End >= next Exam Start:
    next Exam Start - current Exam Start
* No valid next scan (the boundary row):
    Exam End - Exam Start

Two one-scanner counterfactuals are generated:

1. Historical sequence: preserve the actual Exam Start ordering.
2. Priority + FCFS: at every scanner release, serve P1, then P2, then P3,
   then Unknown; within a tier, serve the earliest Parent Order first.

Both models are non-preemptive. An order arriving during a scan waits until
that scan finishes. If nobody is waiting, the scanner clock advances to the
next Parent Order time.

Output: analysis/nni_one_scanner_estimation.xlsx
"""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter


BASE = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE / "data" / "nni_cleaned_orders.csv"
OUTPUT_PATH = BASE / "analysis" / "nni_one_scanner_estimation.xlsx"

PRIORITY_ORDER = ["P1", "P2", "P3", "Unknown"]
PRIORITY_RANK = {priority: rank for rank, priority in enumerate(PRIORITY_ORDER)}
WAIT_TARGET_MIN = {"P1": 30.0, "P2": 120.0, "P3": 360.0}
EXPECTED_ROWS = 8_115


def datetimes_from_columns(
    df: pd.DataFrame, date_column: str, time_column: str
) -> pd.Series:
    dates = pd.to_datetime(df[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
    times = (
        df[time_column]
        .astype(str)
        .str.strip()
        .str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)", expand=False)
    )
    return pd.to_datetime(dates + " " + times, errors="coerce")


def split_joined(value: object) -> list[str]:
    return str(value).split("; ")


def combine_datetime(date_value: object, time_value: object) -> pd.Timestamp | None:
    date = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(date):
        return None
    combined = pd.to_datetime(
        f"{date.strftime('%Y-%m-%d')} {str(time_value).strip()}", errors="coerce"
    )
    return None if pd.isna(combined) else combined


def earliest_joined_datetime(date_value: object, time_value: object) -> pd.Timestamp:
    """Use the earliest timestamp represented by a cleaned merged-order row."""
    date_parts = split_joined(date_value)
    time_parts = split_joined(time_value)
    count = max(len(date_parts), len(time_parts))
    date_parts.extend([date_parts[-1]] * (count - len(date_parts)))
    time_parts.extend([time_parts[-1]] * (count - len(time_parts)))
    candidates = [
        combined
        for combined in (
            combine_datetime(date, time)
            for date, time in zip(date_parts, time_parts)
        )
        if combined is not None
    ]
    return min(candidates) if candidates else pd.NaT


def normalize_priority(value: object) -> str:
    priority = str(value).strip().upper()
    return priority if priority in PRIORITY_RANK else "Unknown"


def auto_column_width(worksheet, maximum: int = 42) -> None:
    for column in worksheet.columns:
        width = max(
            (len(str(cell.value)) for cell in column if cell.value is not None),
            default=0,
        )
        worksheet.column_dimensions[get_column_letter(column[0].column)].width = min(
            width + 2, maximum
        )


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "Order ID",
        "Scan Priority",
        "Parent Order Date",
        "Parent Order Time",
        "Exam Start Date",
        "Exam Start Time",
        "scan_duration",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} cleaned NNI rows, found {len(df):,}. "
            "Review the cohort before using these estimates."
        )
    if df["arrival_dt"].isna().any() or df["actual_exam_start_dt"].isna().any():
        raise ValueError("Parent Order and Exam Start timestamps must be complete.")
    if df["scan_duration_min"].isna().any():
        raise ValueError("scan_duration contains non-numeric or missing values.")
    if (df["scan_duration_min"] <= 0).any():
        raise ValueError("scan_duration must be positive for every simulated order.")
    if (df["actual_exam_start_dt"] < df["arrival_dt"]).any():
        raise ValueError("At least one historical Exam Start precedes its Parent Order time.")


def simulate_historical_sequence(df: pd.DataFrame) -> pd.DataFrame:
    sequence = df.sort_values(
        ["actual_exam_start_dt", "arrival_dt", "_row_id"], kind="stable"
    )
    scanner_free_at = None
    records = []

    for _, row in sequence.iterrows():
        simulated_start = (
            row["arrival_dt"]
            if scanner_free_at is None
            else max(scanner_free_at, row["arrival_dt"])
        )
        simulated_end = simulated_start + pd.to_timedelta(
            row["scan_duration_min"], unit="min"
        )
        records.append((int(row["_row_id"]), simulated_start, simulated_end))
        scanner_free_at = simulated_end

    return pd.DataFrame(
        records,
        columns=["_row_id", "historical_one_scanner_start", "historical_one_scanner_end"],
    ).set_index("_row_id")


def simulate_priority_fcfs(df: pd.DataFrame) -> pd.DataFrame:
    arrivals = df.sort_values(["arrival_dt", "_row_id"], kind="stable").reset_index(
        drop=True
    )
    waiting: list[tuple[int, pd.Timestamp, int, object]] = []
    records = []
    next_arrival = 0
    scanner_free_at = arrivals.loc[0, "arrival_dt"]

    while next_arrival < len(arrivals) or waiting:
        if not waiting and next_arrival < len(arrivals):
            scanner_free_at = max(
                scanner_free_at, arrivals.loc[next_arrival, "arrival_dt"]
            )

        while (
            next_arrival < len(arrivals)
            and arrivals.loc[next_arrival, "arrival_dt"] <= scanner_free_at
        ):
            row = arrivals.iloc[next_arrival]
            heapq.heappush(
                waiting,
                (
                    int(row["priority_rank"]),
                    row["arrival_dt"],
                    int(row["_row_id"]),
                    row,
                ),
            )
            next_arrival += 1

        _, _, row_id, row = heapq.heappop(waiting)
        simulated_start = scanner_free_at
        simulated_end = simulated_start + pd.to_timedelta(
            row["scan_duration_min"], unit="min"
        )
        records.append((row_id, simulated_start, simulated_end))
        scanner_free_at = simulated_end

    return pd.DataFrame(
        records,
        columns=["_row_id", "priority_fcfs_start", "priority_fcfs_end"],
    ).set_index("_row_id")


def distribution(values: pd.Series) -> dict[str, float | int]:
    return {
        "N": int(values.count()),
        "Mean (min)": values.mean(),
        "SD (min)": values.std(ddof=1),
        "Median (min)": values.median(),
        "P25 (min)": values.quantile(0.25),
        "P75 (min)": values.quantile(0.75),
        "P90 (min)": values.quantile(0.90),
        "P95 (min)": values.quantile(0.95),
        "Max (min)": values.max(),
    }


def summarize_waits(df: pd.DataFrame) -> pd.DataFrame:
    models = {
        "Actual historical timestamps": "actual_wait_min",
        "Historical sequence, one scanner": "historical_one_scanner_wait_min",
        "Priority + FCFS, one scanner": "priority_fcfs_wait_min",
    }
    rows = []
    for model, wait_column in models.items():
        for priority in PRIORITY_ORDER:
            values = df.loc[df["priority"] == priority, wait_column]
            row = {"Model": model, "Priority": priority, **distribution(values)}
            target = WAIT_TARGET_MIN.get(priority)
            row["Target (min)"] = target
            row["% Within target"] = (
                100.0 * values.le(target).mean() if target is not None else np.nan
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    numeric_columns = result.select_dtypes(include="number").columns.difference(
        ["N", "Target (min)"]
    )
    result[numeric_columns] = result[numeric_columns].round(1)
    return result


def priority_fcfs_join_position_breakdown(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize where incoming P1/P2 orders join the simulated one-scanner
    priority-FCFS waiting line.

    The waiting line is derived from the priority_fcfs_start/end timeline:
    at an order's arrival time, already-arrived orders whose priority-FCFS
    start is still in the future are waiting. The scan currently in progress
    is not counted as part of the waiting line.
    """
    records = []
    timeline = df.sort_values(["arrival_dt", "_row_id"], kind="stable").copy()

    for _, row in timeline[timeline["priority"].isin(["P1", "P2"])].iterrows():
        arrival = row["arrival_dt"]
        row_id = int(row["_row_id"])

        arrived_before = (
            (timeline["arrival_dt"] < arrival)
            | ((timeline["arrival_dt"] == arrival) & (timeline["_row_id"] < row_id))
        )
        not_started_yet = timeline["priority_fcfs_start"] > arrival
        waiting = timeline[arrived_before & not_started_yet]
        active_scan = timeline[
            (timeline["priority_fcfs_start"] <= arrival)
            & (timeline["priority_fcfs_end"] > arrival)
            & (timeline["_row_id"] != row_id)
        ]

        baseline_join_position = 1 + int(
            (waiting["priority_rank"] <= row["priority_rank"]).sum()
        )
        scanner_empty = waiting.empty and active_scan.empty
        records.append({
            "_row_id": row_id,
            "baseline_position_bucket": (
                "Scanner empty" if scanner_empty else str(baseline_join_position)
            ),
            "baseline_join_position": baseline_join_position,
            "baseline_waiting_ahead": baseline_join_position - 1,
            "waiting_queue_size_at_arrival": len(waiting),
            "scanner_empty_at_arrival": scanner_empty,
            "active_scan_at_arrival": not active_scan.empty,
            "immediately_scanned_priority_fcfs": (
                row["priority_fcfs_start"] == row["arrival_dt"]
            ),
        })

    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame()

    detail = detail.merge(
        df[[
            "_row_id", "CSN", "Order ID", "Order Description", "priority",
            "arrival_dt", "priority_fcfs_start", "priority_fcfs_end",
            "priority_fcfs_wait_min", "actual_exam_start_dt", "actual_wait_min",
        ]],
        on="_row_id",
        how="left",
    )
    detail = detail[[
        "CSN", "Order ID", "Order Description", "priority", "arrival_dt",
        "priority_fcfs_start", "priority_fcfs_end", "priority_fcfs_wait_min",
        "actual_exam_start_dt", "actual_wait_min",
        "baseline_position_bucket", "baseline_join_position",
        "baseline_waiting_ahead", "waiting_queue_size_at_arrival",
        "scanner_empty_at_arrival", "active_scan_at_arrival",
        "immediately_scanned_priority_fcfs",
    ]]

    summary = (
        detail.groupby(["priority", "baseline_position_bucket"], dropna=False)
        .agg(
            n_orders=("Order ID", "size"),
            n_scanner_empty=("scanner_empty_at_arrival", "sum"),
            n_immediately_scanned_priority_fcfs=("immediately_scanned_priority_fcfs", "sum"),
        )
        .reset_index()
    )
    tier_totals = detail.groupby("priority").size().rename("tier_total")
    summary = summary.merge(tier_totals, on="priority", how="left")
    summary["pct_of_tier"] = (summary["n_orders"] / summary["tier_total"] * 100).round(1)
    summary["sort_key"] = summary["baseline_position_bucket"].map(
        lambda value: 0 if value == "Scanner empty" else int(value)
    )
    summary = summary.sort_values(["priority", "sort_key"]).drop(columns="sort_key")
    return detail, summary


def scanner_state_time_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-weighted scanner/queue state from the one-scanner priority-FCFS
    simulation.

    Queue length is the number of arrived orders that have not started yet.
    The active scan, if any, is counted as scanner occupied and is not counted
    in the queue length.
    """
    horizon_start = df["arrival_dt"].min()
    horizon_end = df["priority_fcfs_end"].max()
    event_times = pd.Series(
        pd.concat([
            df["arrival_dt"],
            df["priority_fcfs_start"],
            df["priority_fcfs_end"],
        ]).dropna().unique()
    ).sort_values(kind="stable").reset_index(drop=True)

    intervals = []
    for t, next_t in zip(event_times.iloc[:-1], event_times.iloc[1:]):
        duration_min = (next_t - t).total_seconds() / 60
        if duration_min <= 0:
            continue
        active = df[(df["priority_fcfs_start"] <= t) & (df["priority_fcfs_end"] > t)]
        waiting = df[(df["arrival_dt"] <= t) & (df["priority_fcfs_start"] > t)]
        scanner_occupied = len(active) > 0
        queue_len = len(waiting)
        if not scanner_occupied and queue_len == 0:
            state = "Scanner empty, queue empty"
        elif scanner_occupied and queue_len == 0:
            state = "Scanner occupied, queue empty"
        elif scanner_occupied:
            state = f"Scanner occupied, queue {queue_len}"
        else:
            state = f"Scanner empty, queue {queue_len}"
        intervals.append({
            "interval_start": t,
            "interval_end": next_t,
            "duration_min": duration_min,
            "scanner_occupied": scanner_occupied,
            "queue_len": queue_len,
            "state": state,
        })

    interval_df = pd.DataFrame(intervals)
    total_min = (horizon_end - horizon_start).total_seconds() / 60
    state_summary = (
        interval_df.groupby(["state", "scanner_occupied", "queue_len"], dropna=False)["duration_min"]
        .sum()
        .reset_index()
    )
    state_summary["pct_of_total_time"] = (state_summary["duration_min"] / total_min * 100).round(2)
    state_summary["duration_hours"] = (state_summary["duration_min"] / 60).round(2)
    state_summary = state_summary.sort_values(
        ["scanner_occupied", "queue_len"], ascending=[True, True]
    )

    monthly_rows = []
    month_starts = pd.date_range(
        horizon_start.normalize().replace(day=1),
        horizon_end.normalize().replace(day=1),
        freq="MS",
    )
    for month_start in month_starts:
        month_end = month_start + pd.offsets.MonthBegin(1)
        clipped_start = max(horizon_start, month_start)
        clipped_end = min(horizon_end, month_end)
        if clipped_start >= clipped_end:
            continue
        month_total_min = (clipped_end - clipped_start).total_seconds() / 60
        overlapping = interval_df[
            (interval_df["interval_start"] < clipped_end)
            & (interval_df["interval_end"] > clipped_start)
        ].copy()
        overlapping["clipped_start"] = overlapping["interval_start"].clip(lower=clipped_start)
        overlapping["clipped_end"] = overlapping["interval_end"].clip(upper=clipped_end)
        overlapping["clipped_duration_min"] = (
            overlapping["clipped_end"] - overlapping["clipped_start"]
        ).dt.total_seconds() / 60
        by_state = (
            overlapping.groupby(["state", "scanner_occupied", "queue_len"], dropna=False)["clipped_duration_min"]
            .sum()
            .reset_index()
        )
        by_state["month"] = month_start.strftime("%Y-%m")
        by_state["month_total_min"] = month_total_min
        by_state["pct_of_month"] = (
            by_state["clipped_duration_min"] / month_total_min * 100
        ).round(2)
        by_state["duration_hours"] = (by_state["clipped_duration_min"] / 60).round(2)
        monthly_rows.append(by_state)
    monthly_summary = pd.concat(monthly_rows, ignore_index=True)
    monthly_summary = monthly_summary[[
        "month", "state", "scanner_occupied", "queue_len", "clipped_duration_min",
        "duration_hours", "month_total_min", "pct_of_month",
    ]].sort_values(["month", "scanner_occupied", "queue_len"])

    interval_df["duration_min"] = interval_df["duration_min"].round(4)
    return interval_df, state_summary, monthly_summary


def main() -> None:
    df = pd.read_csv(INPUT_PATH, dtype={"CSN": str, "Order ID": str})
    df["_row_id"] = np.arange(len(df), dtype=int)
    df["arrival_dt"] = df.apply(
        lambda row: earliest_joined_datetime(
            row["Parent Order Date"], row["Parent Order Time"]
        ),
        axis=1,
    )
    df["actual_exam_start_dt"] = datetimes_from_columns(
        df, "Exam Start Date", "Exam Start Time"
    )
    df["scan_duration_min"] = pd.to_numeric(df["scan_duration"], errors="coerce")
    df["priority"] = df["Scan Priority"].apply(normalize_priority)
    df["priority_rank"] = df["priority"].map(PRIORITY_RANK)
    validate_input(df)

    historical = simulate_historical_sequence(df)
    priority_fcfs = simulate_priority_fcfs(df)
    df = df.join(historical, on="_row_id").join(priority_fcfs, on="_row_id")

    df["actual_wait_min"] = (
        df["actual_exam_start_dt"] - df["arrival_dt"]
    ).dt.total_seconds() / 60
    df["historical_one_scanner_wait_min"] = (
        df["historical_one_scanner_start"] - df["arrival_dt"]
    ).dt.total_seconds() / 60
    df["priority_fcfs_wait_min"] = (
        df["priority_fcfs_start"] - df["arrival_dt"]
    ).dt.total_seconds() / 60

    wait_columns = [
        "actual_wait_min",
        "historical_one_scanner_wait_min",
        "priority_fcfs_wait_min",
    ]
    if df[wait_columns].isna().any().any() or (df[wait_columns] < 0).any().any():
        raise AssertionError("Every order must receive one nonnegative waiting time.")

    wait_summary = summarize_waits(df)
    p1_p2_join_detail, p1_p2_join_summary = priority_fcfs_join_position_breakdown(df)
    state_intervals, state_summary, state_monthly = scanner_state_time_summary(df)
    duration_summary = pd.DataFrame([distribution(df["scan_duration_min"])])
    duration_summary.insert(0, "Metric", "Updated scan_duration")
    duration_summary.iloc[:, 2:] = duration_summary.iloc[:, 2:].round(1)

    notes = pd.DataFrame(
        {
            "Note": [
                f"Input: {INPUT_PATH.relative_to(BASE)}; validated {len(df):,} rows and positive scan_duration for every row.",
                "Service time is the existing scan_duration column. The simulation does not recalculate or modify it.",
                "Upstream duration rule: if Exam End < next Exam Start, duration = Exam End - Exam Start + 17.9 minutes.",
                "Upstream duration rule: if Exam End >= next Exam Start, duration = next Exam Start - current Exam Start.",
                "For the final boundary row with no valid next scan, duration = Exam End - Exam Start.",
                "The 17.9-minute value is the rounded mean nonzero gap for immediate scan pairs where the next order was already queued by the previous computed scan end.",
                "Historical one scanner preserves ascending actual Exam Start order. Each simulated start is max(previous simulated end, this order's Parent Order time).",
                "Priority + FCFS one scanner is non-preemptive. Whenever the scanner becomes free, it selects the best arrived order by P1, P2, P3, Unknown, then earliest Parent Order time.",
                "If the Priority + FCFS queue is empty, the scanner clock advances to the next Parent Order time.",
                "Actual historical wait = actual Exam Start - Parent Order time. Simulated wait = simulated start - Parent Order time.",
                "Wait targets: P1 <= 30 minutes, P2 <= 120 minutes, P3 <= 360 minutes. Unknown has no target.",
                "P1_P2_Position_Summary and P1_P2_Position_Detail use the Priority + FCFS one-scanner simulated timeline. At each P1/P2 arrival, the waiting line is all earlier-arrived orders whose simulated priority_fcfs_start is still in the future; the scan in progress is not counted as a waiting queue position. Scanner empty is separated from numeric position 1 when there is no active scan and no waiting order at arrival.",
                "Scanner_State_Summary is a time-weighted breakdown over the Priority + FCFS one-scanner simulated timeline, from first Parent Order arrival to final simulated scan end. Queue length means arrived-but-not-started orders; the active scan is not counted in the queue. Percentages are based on exact interval durations between state-changing events, not rounded minute sampling.",
            ]
        }
    )

    order_level_columns = [
        "month",
        "CSN",
        "Order ID",
        "Order Description",
        "priority",
        "arrival_dt",
        "actual_exam_start_dt",
        "scan_duration_min",
        "actual_wait_min",
        "historical_one_scanner_start",
        "historical_one_scanner_end",
        "historical_one_scanner_wait_min",
        "priority_fcfs_start",
        "priority_fcfs_end",
        "priority_fcfs_wait_min",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="Notes", index=False)
        duration_summary.to_excel(writer, sheet_name="Scan_Duration_Summary", index=False)
        wait_summary.to_excel(writer, sheet_name="Wait_Summary", index=False)
        p1_p2_join_summary.to_excel(writer, sheet_name="P1_P2_Position_Summary", index=False)
        p1_p2_join_detail.to_excel(writer, sheet_name="P1_P2_Position_Detail", index=False)
        state_summary.to_excel(writer, sheet_name="Scanner_State_Summary", index=False)
        state_monthly.to_excel(writer, sheet_name="Scanner_State_By_Month", index=False)
        state_intervals.to_excel(writer, sheet_name="Scanner_State_Intervals", index=False)
        df[order_level_columns].sort_values("arrival_dt").to_excel(
            writer, sheet_name="Order_Level", index=False
        )
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            auto_column_width(worksheet)

    print("Scan duration source and derivation:")
    print("  Exam End < next Exam Start: Exam End - Exam Start + 17.9 min")
    print("  Exam End >= next Exam Start: next Exam Start - current Exam Start")
    print("  No valid next scan: Exam End - Exam Start")
    print(f"Validated input rows: {len(df):,}")
    print(f"Wrote: {OUTPUT_PATH}")
    print("\nWaiting-time summary by priority:")
    print(wait_summary.to_string(index=False))
    print("\nP1/P2 one-scanner priority-FCFS join-position summary:")
    print(p1_p2_join_summary.to_string(index=False))
    print("\nOne-scanner priority-FCFS scanner-state summary:")
    print(state_summary.to_string(index=False))


if __name__ == "__main__":
    main()
