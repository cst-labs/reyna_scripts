"""
ddr_emergency_one_scanner_estimation.py
DDR-Emergency queue only (new data, Jul/Aug/Sep 2026). Estimates waiting
times under one non-preemptive CT scanner, using scan_duration_min (already
present in ct_ddr_emergency_orders_newdata.xlsx -- an overrun-aware duration
computed by ct_queue_analysis_newdata.py) as each order's actual service
time on the scanner.

Same methodology as nni_one_scanner_estimation.py, the prior script this one
is modelled on, adapted for DDR-Emergency's new-data columns:

Two one-scanner counterfactuals:
1. Historical sequence: preserve the actual Exam Start ordering.
2. Priority + FCFS: at every scanner release, serve P1, then P2, then P3,
   then Unknown; within a tier, serve the earliest Parent Order first.

Both models are non-preemptive. An order arriving during a scan waits until
that scan finishes. If nobody is waiting, the scanner clock advances to the
next Parent Order time.

Unlike the NNI script, DDR-Emergency has 15 rows with scan_duration_min <= 0
(a known artifact of the overrun-aware calc for back-to-back orders sharing
the exact same Exam Start timestamp -- not missing/bad data, per the
project's "flag, don't drop" convention). These are kept and treated as a
legitimate instantaneous (0-minute) scan rather than hard-failing like the
NNI script's validate_input() does.

Reads:
  analysis/ct_ddr_emergency_orders_newdata.xlsx (sheet DDR_Emergency_Orders)

Outputs: analysis/ddr_emergency_one_scanner_estimation.xlsx
"""
from __future__ import annotations

import heapq
import os

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS = os.path.join(BASE, "analysis")

TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
WAIT_TARGET_MIN = {"P1": 30.0, "P2": 120.0, "P3": 360.0}


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 50)


def simulate_historical_sequence(df):
    """Non-preemptive single scanner, preserving the actual Exam Start
    ordering. Each order's simulated start = max(scanner_free_at, its own
    arrival); simulated end = start + scan_duration_min."""
    sequence = df.sort_values(["exam_start_dt", "_row_id"], kind="stable")
    scanner_free_at = None
    records = []
    for _, row in sequence.iterrows():
        start = row["parent_order_dt"] if scanner_free_at is None else max(scanner_free_at, row["parent_order_dt"])
        end = start + pd.to_timedelta(row["scan_duration_min"], unit="min")
        records.append((int(row["_row_id"]), start, end))
        scanner_free_at = end
    return pd.DataFrame(records, columns=["_row_id", "historical_one_scanner_start", "historical_one_scanner_end"]).set_index("_row_id")


def simulate_priority_fcfs(df):
    """Non-preemptive single scanner. Whenever the scanner frees up, serves
    the best arrived-but-not-yet-started order by (tier_rank, parent_order_dt).
    If nobody has arrived yet, the scanner clock jumps to the next arrival."""
    arrivals = df.sort_values(["parent_order_dt", "_row_id"], kind="stable").reset_index(drop=True)
    waiting = []
    records = []
    next_arrival = 0
    scanner_free_at = arrivals.loc[0, "parent_order_dt"]

    while next_arrival < len(arrivals) or waiting:
        if not waiting and next_arrival < len(arrivals):
            scanner_free_at = max(scanner_free_at, arrivals.loc[next_arrival, "parent_order_dt"])

        while next_arrival < len(arrivals) and arrivals.loc[next_arrival, "parent_order_dt"] <= scanner_free_at:
            row = arrivals.iloc[next_arrival]
            heapq.heappush(waiting, (int(row["tier_rank"]), row["parent_order_dt"], int(row["_row_id"]), row))
            next_arrival += 1

        _, _, row_id, row = heapq.heappop(waiting)
        start = scanner_free_at
        end = start + pd.to_timedelta(row["scan_duration_min"], unit="min")
        records.append((row_id, start, end))
        scanner_free_at = end

    return pd.DataFrame(records, columns=["_row_id", "priority_fcfs_start", "priority_fcfs_end"]).set_index("_row_id")


def distribution(values):
    return {
        "N": int(values.count()), "Mean (min)": values.mean(), "SD (min)": values.std(ddof=1),
        "Median (min)": values.median(), "P25 (min)": values.quantile(0.25),
        "P75 (min)": values.quantile(0.75), "P90 (min)": values.quantile(0.90),
        "P95 (min)": values.quantile(0.95), "Max (min)": values.max(),
    }


def summarize_waits(df):
    models = {
        "Actual historical timestamps": "wait_to_start_min",
        "Historical sequence, one scanner": "historical_one_scanner_wait_min",
        "Priority + FCFS, one scanner": "priority_fcfs_wait_min",
    }
    rows = []
    for model, wait_col in models.items():
        for tier in TIER_ORDER:
            values = df.loc[df["priority_tier"] == tier, wait_col]
            target = WAIT_TARGET_MIN.get(tier)
            row = {"Model": model, "Priority": tier, **distribution(values)}
            row["Target (min)"] = target
            row["% Within target"] = 100.0 * values.le(target).mean() if target is not None else np.nan
            rows.append(row)
    result = pd.DataFrame(rows)
    numeric_cols = result.select_dtypes(include="number").columns.difference(["N", "Target (min)"])
    result[numeric_cols] = result[numeric_cols].round(1)
    return result


def priority_fcfs_join_position_breakdown(df):
    """For each real P1/P2 arrival, where it would join the waiting line of
    the simulated one-scanner priority-FCFS timeline (same method as
    nni_one_scanner_estimation.py's version)."""
    records = []
    timeline = df.sort_values(["parent_order_dt", "_row_id"], kind="stable").copy()
    for _, row in timeline[timeline["priority_tier"].isin(["P1", "P2"])].iterrows():
        arrival = row["parent_order_dt"]
        row_id = int(row["_row_id"])
        arrived_before = (
            (timeline["parent_order_dt"] < arrival)
            | ((timeline["parent_order_dt"] == arrival) & (timeline["_row_id"] < row_id))
        )
        not_started_yet = timeline["priority_fcfs_start"] > arrival
        waiting = timeline[arrived_before & not_started_yet]
        active_scan = timeline[
            (timeline["priority_fcfs_start"] <= arrival)
            & (timeline["priority_fcfs_end"] > arrival)
            & (timeline["_row_id"] != row_id)
        ]
        baseline_join_position = 1 + int((waiting["tier_rank"] <= row["tier_rank"]).sum())
        scanner_empty = waiting.empty and active_scan.empty
        records.append({
            "_row_id": row_id,
            "baseline_position_bucket": "Scanner empty" if scanner_empty else str(baseline_join_position),
            "baseline_join_position": baseline_join_position,
            "baseline_waiting_ahead": baseline_join_position - 1,
            "waiting_queue_size_at_arrival": len(waiting),
            "scanner_empty_at_arrival": scanner_empty,
            "active_scan_at_arrival": not active_scan.empty,
            "immediately_scanned_priority_fcfs": row["priority_fcfs_start"] == row["parent_order_dt"],
        })

    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame()

    detail = detail.merge(
        df[["_row_id", "CSN", "Order ID", "priority_tier", "parent_order_dt",
            "priority_fcfs_start", "priority_fcfs_end", "priority_fcfs_wait_min",
            "exam_start_dt", "wait_to_start_min"]],
        on="_row_id", how="left",
    )
    detail = detail[[
        "CSN", "Order ID", "priority_tier", "parent_order_dt",
        "priority_fcfs_start", "priority_fcfs_end", "priority_fcfs_wait_min",
        "exam_start_dt", "wait_to_start_min",
        "baseline_position_bucket", "baseline_join_position", "baseline_waiting_ahead",
        "waiting_queue_size_at_arrival", "scanner_empty_at_arrival", "active_scan_at_arrival",
        "immediately_scanned_priority_fcfs",
    ]]

    summary = (
        detail.groupby(["priority_tier", "baseline_position_bucket"], dropna=False)
        .agg(n_orders=("Order ID", "size"), n_scanner_empty=("scanner_empty_at_arrival", "sum"),
             n_immediately_scanned_priority_fcfs=("immediately_scanned_priority_fcfs", "sum"))
        .reset_index()
    )
    tier_totals = detail.groupby("priority_tier").size().rename("tier_total")
    summary = summary.merge(tier_totals, on="priority_tier", how="left")
    summary["pct_of_tier"] = (summary["n_orders"] / summary["tier_total"] * 100).round(1)
    summary["sort_key"] = summary["baseline_position_bucket"].map(lambda v: 0 if v == "Scanner empty" else int(v))
    summary = summary.sort_values(["priority_tier", "sort_key"]).drop(columns="sort_key")
    return detail, summary


def scanner_state_time_summary(df):
    """Time-weighted scanner/queue state from the one-scanner priority-FCFS
    simulation (same method as nni_one_scanner_estimation.py)."""
    horizon_start = df["parent_order_dt"].min()
    horizon_end = df["priority_fcfs_end"].max()
    event_times = pd.Series(
        pd.concat([df["parent_order_dt"], df["priority_fcfs_start"], df["priority_fcfs_end"]]).dropna().unique()
    ).sort_values(kind="stable").reset_index(drop=True)

    intervals = []
    for t, next_t in zip(event_times.iloc[:-1], event_times.iloc[1:]):
        duration_min = (next_t - t).total_seconds() / 60
        if duration_min <= 0:
            continue
        active = df[(df["priority_fcfs_start"] <= t) & (df["priority_fcfs_end"] > t)]
        waiting = df[(df["parent_order_dt"] <= t) & (df["priority_fcfs_start"] > t)]
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
        intervals.append({"interval_start": t, "interval_end": next_t, "duration_min": duration_min,
                           "scanner_occupied": scanner_occupied, "queue_len": queue_len, "state": state})

    interval_df = pd.DataFrame(intervals)
    total_min = (horizon_end - horizon_start).total_seconds() / 60
    state_summary = (
        interval_df.groupby(["state", "scanner_occupied", "queue_len"], dropna=False)["duration_min"]
        .sum().reset_index()
    )
    state_summary["pct_of_total_time"] = (state_summary["duration_min"] / total_min * 100).round(2)
    state_summary["duration_hours"] = (state_summary["duration_min"] / 60).round(2)
    state_summary = state_summary.sort_values(["scanner_occupied", "queue_len"], ascending=[True, True])
    interval_df["duration_min"] = interval_df["duration_min"].round(4)
    return interval_df, state_summary


# ---------------------------------------------------------------------------
# Load -- DDR-Emergency only
# ---------------------------------------------------------------------------
print("Loading analysis/ct_ddr_emergency_orders_newdata.xlsx ...")
ddr = pd.read_excel(os.path.join(ANALYSIS, "ct_ddr_emergency_orders_newdata.xlsx"), sheet_name="DDR_Emergency_Orders")
print(f"  DDR-Emergency orders: {len(ddr):,}")

n_before = len(ddr)
ddr = ddr[ddr["wait_to_start_min"] >= 0].copy()
print(f"  Valid orders with a well-defined service-start time (wait_to_start_min >= 0): "
      f"{len(ddr):,} of {n_before:,} ({n_before - len(ddr)} dropped)")

n_zero_or_neg_duration = int((ddr["scan_duration_min"] <= 0).sum())
print(f"  scan_duration_min <= 0: {n_zero_or_neg_duration} row(s) -- kept as a legitimate 0-minute scan "
      f"(known artifact of back-to-back orders sharing the same Exam Start timestamp; not bad data).")
ddr["scan_duration_min"] = ddr["scan_duration_min"].clip(lower=0)

ddr["tier_rank"] = ddr["priority_tier"].map(TIER_RANK)
ddr["_row_id"] = range(len(ddr))

assert ddr["scan_duration_min"].notna().all(), "scan_duration_min must be complete for every row"
assert (ddr["exam_start_dt"] >= ddr["parent_order_dt"]).all(), "Exam Start must not precede Parent Order"

# ---------------------------------------------------------------------------
# Run both one-scanner simulations
# ---------------------------------------------------------------------------
print("\nSimulating one non-preemptive scanner (historical sequence) ...")
historical = simulate_historical_sequence(ddr)
print("Simulating one non-preemptive scanner (priority + FCFS) ...")
priority_fcfs = simulate_priority_fcfs(ddr)
ddr = ddr.join(historical, on="_row_id").join(priority_fcfs, on="_row_id")

ddr["historical_one_scanner_wait_min"] = (ddr["historical_one_scanner_start"] - ddr["parent_order_dt"]).dt.total_seconds() / 60
ddr["priority_fcfs_wait_min"] = (ddr["priority_fcfs_start"] - ddr["parent_order_dt"]).dt.total_seconds() / 60

wait_cols = ["wait_to_start_min", "historical_one_scanner_wait_min", "priority_fcfs_wait_min"]
if ddr[wait_cols].isna().any().any() or (ddr[wait_cols] < 0).any().any():
    raise AssertionError("Every order must receive one nonnegative waiting time.")

wait_summary = summarize_waits(ddr)
print("\nWaiting-time summary by priority:")
print(wait_summary.to_string(index=False))

p1_p2_detail, p1_p2_summary = priority_fcfs_join_position_breakdown(ddr)
print("\nP1/P2 one-scanner priority-FCFS join-position summary:")
print(p1_p2_summary.to_string(index=False))

state_intervals, state_summary = scanner_state_time_summary(ddr)
print("\nOne-scanner priority-FCFS scanner-state summary:")
print(state_summary.to_string(index=False))

# ---------------------------------------------------------------------------
# Output workbook
# ---------------------------------------------------------------------------
notes = pd.DataFrame([
    {"Note": "Scope: DDR-Emergency CT queue only, from analysis/ct_ddr_emergency_orders_newdata.xlsx "
             "(new data, Jul/Aug/Sep 2026)."},
    {"Note": f"{n_before - len(ddr)} row(s) with wait_to_start_min < 0 (a data-quality / rounding issue, "
             "not a real negative wait) excluded before simulating."},
    {"Note": f"{n_zero_or_neg_duration} row(s) had scan_duration_min <= 0 -- a known artifact of the "
             "overrun-aware duration calc (ct_queue_analysis_newdata.py) for back-to-back orders sharing "
             "the exact same Exam Start timestamp, not missing/bad data. Kept (clipped to 0) and treated "
             "as a legitimate instantaneous scan, rather than dropped."},
    {"Note": "Service time = scan_duration_min, already computed upstream by ct_queue_analysis_newdata.py "
             "(DDR-Emergency uses an overrun-aware override of the generic shorter-of-own-duration-or-"
             "gap-to-next formula). This simulation does not recalculate it."},
    {"Note": "No documented DDR-Emergency scanner count exists in this project (unlike NNI=2, DR=1 "
             "elsewhere), so both counterfactuals here assume exactly ONE non-preemptive scanner, per "
             "explicit request -- a deliberate simplification, not a measured capacity."},
    {"Note": "Historical sequence, one scanner: preserves the actual ascending Exam Start order. Each "
             "simulated start = max(previous simulated end, this order's Parent Order time); simulated "
             "end = start + scan_duration_min."},
    {"Note": "Priority + FCFS, one scanner: non-preemptive. Whenever the scanner becomes free, it selects "
             "the best arrived-but-not-started order by (priority tier: P1>P2>P3>Unknown, then earliest "
             "Parent Order time). If the queue is empty, the scanner clock advances to the next arrival."},
    {"Note": "Wait = simulated start - Parent Order time (or, for Actual, real Exam Start - Parent Order "
             "time). Wait targets: P1<=30min, P2<=120min, P3<=360min; Unknown has no target."},
    {"Note": "P1_P2_Position_Summary / Detail use the Priority+FCFS one-scanner simulated timeline: at "
             "each P1/P2 arrival, the waiting line is all earlier-arrived orders whose simulated "
             "priority_fcfs_start is still in the future; a scan in progress is not counted as a queue "
             "position. Scanner empty is separated from numeric position 1."},
    {"Note": "Scanner_State_Summary is a time-weighted breakdown over the Priority+FCFS one-scanner "
             "simulated timeline, from first Parent Order arrival to final simulated scan end. Queue "
             "length = arrived-but-not-started orders; the active scan is not counted in the queue."},
    {"Note": "This single-scanner model is unavoidably more congested than reality (DDR-Emergency's real "
             "exam-start/exam-end overlap reaches several concurrent exams at peak), so absolute wait "
             "times here will run higher than the real historical ones -- the useful comparison is "
             "Historical-sequence-one-scanner vs. Priority+FCFS-one-scanner (same assumed capacity), not "
             "either one vs. Actual."},
])

order_level_cols = [
    "CSN", "Order ID", "priority_tier", "parent_order_dt", "exam_start_dt", "scan_duration_min",
    "wait_to_start_min",
    "historical_one_scanner_start", "historical_one_scanner_end", "historical_one_scanner_wait_min",
    "priority_fcfs_start", "priority_fcfs_end", "priority_fcfs_wait_min",
]

out_path = os.path.join(ANALYSIS, "ddr_emergency_one_scanner_estimation.xlsx")
print(f"\nWriting {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    notes.to_excel(writer, sheet_name="Notes", index=False)
    wait_summary.to_excel(writer, sheet_name="Wait_Summary", index=False)
    p1_p2_summary.to_excel(writer, sheet_name="P1_P2_Position_Summary", index=False)
    p1_p2_detail.to_excel(writer, sheet_name="P1_P2_Position_Detail", index=False)
    state_summary.to_excel(writer, sheet_name="Scanner_State_Summary", index=False)
    state_intervals.to_excel(writer, sheet_name="Scanner_State_Intervals", index=False)
    ddr[order_level_cols].sort_values("parent_order_dt").to_excel(writer, sheet_name="Order_Level", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
