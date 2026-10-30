"""
build_ct_queue_newdata.py
Reconstructs ONE unified CT scanner queue (not split by Performing Location/
site) from merged/ct_orders_newdata_merged.xlsx (Jul/Aug/Sep 2026 new data).
Deliberate deviation from build_ct_queue.py: that script reconstructs two
independent queues (NNI, DR); this one sweeps all CT orders together as a
single queue, per request. Performing Location and Scan Priority are kept as
columns for downstream per-dimension analysis, not as separate queues.

Every order enters the queue at Parent Order time. Two departure notions,
both deviating from build_ct_queue.py's gap-to-next approximation (per
request -- real Exam End timestamps have no nulls/backwards-ordering issues
in this data, so there's no need to approximate):
  - queue_len_at_arrival:      departs at this order's own real exam_end_dt
                                (scanner-occupancy count).
  - queue_position_at_arrival: departs at Exam Start time (waiting-line
                                count), restricted to orders with a valid
                                wait_to_start_min.

scan_duration_min still uses the gap-to-next-Exam-Start approximation
(exam_end_dt_approx) alongside the real exam_duration_min, taking whichever
is shorter -- that's a separate question (how long the scan itself likely
took) from occupancy, so it's untouched here.

Outputs: analysis/ct_queue_reconstruction_newdata.xlsx
"""
import os
import re

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

PRIORITY_PATTERN = re.compile(r"P([1-3])")


def is_blank(v):
    if v is None:
        return True
    try:
        if isinstance(v, float) and np.isnan(v):
            return True
    except Exception:
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def parse_priority(val):
    """Scan Priority is normally a single 'P1 - Urgent scan' etc. string; a
    handful of merged rows have '; '-joined conflicting values (e.g. 'P2 -
    Expedited scan; P3 - Normal scan') -- the first captured P-value wins,
    same first-match convention used elsewhere in this project."""
    if is_blank(val):
        return "Unknown"
    m = PRIORITY_PATTERN.search(str(val))
    return f"P{m.group(1)}" if m else "Unknown"


def split_joined(val):
    if is_blank(val):
        return [None]
    s = str(val)
    return s.split("; ") if "; " in s else [s]


def parse_date_only(date_val):
    if isinstance(date_val, pd.Timestamp):
        return date_val
    s = str(date_val)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        if pd.notna(d):
            return d
    return pd.to_datetime(date_val, errors="coerce", dayfirst=True)


def combine_dt(date_val, time_val):
    if is_blank(date_val) or is_blank(time_val):
        return None
    d = parse_date_only(date_val)
    if pd.isna(d):
        return None
    dt = pd.to_datetime(f"{d.strftime('%Y-%m-%d')} {time_val}", errors="coerce")
    return None if pd.isna(dt) else dt


def earliest_parent_order_dt(date_val, time_val):
    d_parts = split_joined(date_val)
    t_parts = split_joined(time_val)
    n = max(len(d_parts), len(t_parts))
    if len(d_parts) < n:
        d_parts = d_parts + [d_parts[-1]] * (n - len(d_parts))
    if len(t_parts) < n:
        t_parts = t_parts + [t_parts[-1]] * (n - len(t_parts))
    dts = [combine_dt(d, t) for d, t in zip(d_parts, t_parts)]
    dts = [dt for dt in dts if dt is not None]
    return min(dts) if dts else None


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 50)


def time_weighted_mean(events_df, col="running_queue_len"):
    if len(events_df) < 2:
        return None
    df = events_df.sort_values("datetime").reset_index(drop=True)
    total_time = 0.0
    weighted_sum = 0.0
    for i in range(len(df) - 1):
        dt_delta = (df.loc[i + 1, "datetime"] - df.loc[i, "datetime"]).total_seconds()
        weighted_sum += df.loc[i, col] * dt_delta
        total_time += dt_delta
    return weighted_sum / total_time if total_time > 0 else None


def build_queue(df, departure_col):
    """Single unified sweep: arrival (+1) at parent_order_dt, departure (-1)
    at departure_col. Departures sort before arrivals on an exact tie (so a
    slot freed by one order isn't double-counted as still occupied when
    another order arrives at that same instant).

    A handful of orders have zero wait (parent_order_dt == departure instant
    for THAT SAME order, e.g. a flagged zero-wait row kept per the "don't
    drop, flag instead" rule) -- for those, the departure-before-arrival tie
    rule would process the order's own departure before its own arrival ever
    happened, driving the running count negative. Nudge a same-row departure
    forward by 1us so an order's own arrival always precedes its own
    departure, while ties between DIFFERENT orders are untouched."""
    events = []
    for _, r in df.iterrows():
        arrival_dt = r["parent_order_dt"]
        departure_dt = r[departure_col]
        if pd.notna(arrival_dt) and pd.notna(departure_dt) and departure_dt <= arrival_dt:
            departure_dt = arrival_dt + pd.Timedelta(microseconds=1)
        events.append((arrival_dt, 1, "arrival", r["Order ID"], r["CSN"], r["_row_id"]))
        events.append((departure_dt, -1, "departure", r["Order ID"], r["CSN"], r["_row_id"]))
    events.sort(key=lambda e: (e[0], e[1]))

    running = 0
    rows = []
    arrival_running = {}
    for dt, delta, etype, order_id, csn, row_id in events:
        running += delta
        assert running >= 0, f"Queue length went negative at {dt}"
        rows.append({
            "datetime": dt, "event_type": etype, "Order ID": order_id,
            "CSN": csn, "running_queue_len": running,
        })
        if etype == "arrival":
            arrival_running[row_id] = running
    return pd.DataFrame(rows), arrival_running


def build_rank_clusters(df):
    df = df.sort_values("parent_order_dt").reset_index(drop=True)
    cluster_ids = []
    current_cluster = 0
    cluster_end = None
    for _, r in df.iterrows():
        if cluster_end is None or r["parent_order_dt"] >= cluster_end:
            current_cluster += 1
            cluster_end = r["exam_start_dt"]
        else:
            cluster_end = max(cluster_end, r["exam_start_dt"])
        cluster_ids.append(current_cluster)
    df["wait_cluster_id"] = cluster_ids
    df["wait_cluster_size"] = df.groupby("wait_cluster_id")["wait_cluster_id"].transform("size")
    df["arrival_rank"] = df.groupby("wait_cluster_id")["parent_order_dt"].rank(method="first").astype(int)
    df["service_rank"] = df.groupby("wait_cluster_id")["exam_start_dt"].rank(method="first").astype(int)
    df["rank_gap"] = df["service_rank"] - df["arrival_rank"]
    return df[["_row_id", "wait_cluster_id", "wait_cluster_size", "arrival_rank", "service_rank", "rank_gap"]]


# ---------------------------------------------------------------------------
# Load (already pure CT -- Procedure Category is 100% IMG CT ORDERABLES)
# ---------------------------------------------------------------------------
print("Loading merged/ct_orders_newdata_merged.xlsx ...")
ct = pd.read_excel(os.path.join(MERGED, "ct_orders_newdata_merged.xlsx"), sheet_name="Imaging")
print(f"  CT rows: {len(ct):,}")

ct["procedure_location"] = ct["Performing Location"].fillna("Unknown")
ct["priority_tier"] = ct["Scan Priority"].apply(parse_priority)
print("  Procedure location split:")
print(ct["procedure_location"].value_counts().to_string())
print("  Scan priority split:")
print(ct["priority_tier"].value_counts().to_string())

ct["parent_order_dt"] = ct.apply(
    lambda r: earliest_parent_order_dt(r["Parent Order Date"], r["Parent Order Time"]), axis=1
)
ct["exam_start_dt"] = ct.apply(lambda r: combine_dt(r["Exam Start Date"], r["Exam Start Time"]), axis=1)
ct["exam_end_dt"] = ct.apply(lambda r: combine_dt(r["Exam End Date"], r["Exam End Time"]), axis=1)

ct["wait_to_start_min"] = ct.apply(
    lambda r: (r["exam_start_dt"] - r["parent_order_dt"]).total_seconds() / 60
    if pd.notna(r["exam_start_dt"]) and pd.notna(r["parent_order_dt"]) else np.nan,
    axis=1,
)
ct["exam_duration_min"] = ct.apply(
    lambda r: (r["exam_end_dt"] - r["exam_start_dt"]).total_seconds() / 60
    if pd.notna(r["exam_end_dt"]) and pd.notna(r["exam_start_dt"]) else np.nan,
    axis=1,
)

# exam_end_dt_approx: next order's real Exam Start, globally -- single
# unified queue, so no groupby("site") here (that's the deliberate deviation
# from build_ct_queue.py).
ct = ct.sort_values("exam_start_dt").reset_index(drop=True)
ct["exam_end_dt_approx"] = ct["exam_start_dt"].shift(-1)
ct["exam_duration_min_approx"] = ct.apply(
    lambda r: (r["exam_end_dt_approx"] - r["exam_start_dt"]).total_seconds() / 60
    if pd.notna(r["exam_end_dt_approx"]) and pd.notna(r["exam_start_dt"]) else np.nan,
    axis=1,
)
# scan_duration_min: shorter of this exam's own duration and the gap to the
# next exam's start on the single unified queue (ct_scan_time_by_type.py's
# concept, reinterpreted for one queue instead of per-site).
ct["scan_duration_min"] = ct[["exam_duration_min_approx", "exam_duration_min"]].min(axis=1, skipna=True)


def dq_flag(row):
    if pd.isna(row["parent_order_dt"]):
        return "missing_timestamp"
    return None


ct["data_quality_flag"] = ct.apply(dq_flag, axis=1)
n_flagged = ct["data_quality_flag"].notna().sum()
print(f"  Data-quality flagged (excluded from queue sweep): {n_flagged:,} of {len(ct):,}")

ct["_row_id"] = range(len(ct))
valid = ct[ct["data_quality_flag"].isna()].copy()

# ---------------------------------------------------------------------------
# Single unified sweep: scanner-occupancy (queue_len_at_arrival). Departs at
# this order's own real exam_end_dt (not the next order's Exam Start approx)
# -- exam_end_dt has no nulls, so every order gets a queue_len_at_arrival.
# ---------------------------------------------------------------------------
events, arrival_running = build_queue(valid, "exam_end_dt")
ct["queue_len_at_arrival"] = ct["_row_id"].map(arrival_running) - 1
print(f"  Queue sweep: {len(events) // 2:,} orders, max concurrent {events['running_queue_len'].max()}, "
      f"time-weighted mean concurrent {time_weighted_mean(events):.2f}")

# ---------------------------------------------------------------------------
# Single unified sweep: waiting-line (queue_position_at_arrival)
# ---------------------------------------------------------------------------
wait_valid = valid[valid["wait_to_start_min"] >= 0].copy()
print(f"  Waiting-line sweep population: {len(wait_valid):,} of {len(valid):,} valid orders")

wait_events, wait_arrival = build_queue(wait_valid, "exam_start_dt")
ct["queue_position_at_arrival"] = ct["_row_id"].map(wait_arrival) - 1
print(f"  Waiting-line sweep: {len(wait_events) // 2:,} orders, max concurrent waiting "
      f"{wait_events['running_queue_len'].max()}, time-weighted mean waiting "
      f"{time_weighted_mean(wait_events):.2f}")

# ---------------------------------------------------------------------------
# Rank: arrival order vs. actual service order, within each waiting cluster
# ---------------------------------------------------------------------------
ranks = build_rank_clusters(wait_valid).set_index("_row_id")
ct = ct.merge(
    ranks[["wait_cluster_id", "wait_cluster_size", "arrival_rank", "service_rank", "rank_gap"]],
    left_on="_row_id", right_index=True, how="left",
)
contested = ranks[ranks["wait_cluster_size"] > 1]
n_jumped = int((contested["rank_gap"] < 0).sum())
print(f"  {len(contested):,} orders in a contested cluster (2+ overlapping), "
      f"{n_jumped:,} ({n_jumped / len(contested):.1%}) scanned ahead of their arrival-order position")

# ---------------------------------------------------------------------------
# Output workbook
# ---------------------------------------------------------------------------
ct_orders_out = ct[[
    "CSN", "Order ID", "Order Description", "procedure_location", "priority_tier",
    "parent_order_dt", "exam_start_dt", "exam_end_dt", "exam_end_dt_approx",
    "wait_to_start_min", "exam_duration_min", "scan_duration_min",
    "queue_len_at_arrival", "queue_position_at_arrival",
    "wait_cluster_id", "wait_cluster_size", "arrival_rank", "service_rank", "rank_gap",
    "data_quality_flag",
]].sort_values("parent_order_dt").reset_index(drop=True)

summary_rows = [
    {"Metric": "Total CT orders", "Value": len(ct)},
    {"Metric": "Data quality flagged (excluded from queue sweep)", "Value": int(n_flagged)},
]
for flag, cnt in ct["data_quality_flag"].value_counts(dropna=True).items():
    summary_rows.append({"Metric": f"  {flag}", "Value": int(cnt)})
summary_rows.append({"Metric": "-- queue_len_at_arrival: scanner-occupancy count (single unified queue) --", "Value": ""})
summary_rows.append({"Metric": "Max concurrent queue length", "Value": int(events["running_queue_len"].max())})
summary_rows.append({"Metric": "Time-weighted mean concurrent queue length", "Value": round(time_weighted_mean(events), 2)})
summary_rows.append({"Metric": "-- queue_position_at_arrival: waiting-line count --", "Value": ""})
summary_rows.append({"Metric": "Orders excluded from waiting-line sweep (missing/negative wait_to_start_min)", "Value": int(len(valid) - len(wait_valid))})
summary_rows.append({"Metric": "Max concurrent waiting", "Value": int(wait_events["running_queue_len"].max())})
summary_rows.append({"Metric": "Time-weighted mean waiting", "Value": round(time_weighted_mean(wait_events), 2)})
summary_rows.append({"Metric": "-- arrival_rank vs. service_rank: queue-jumping within contested waiting clusters --", "Value": ""})
summary_rows.append({"Metric": "Orders in a contested cluster (2+ overlapping)", "Value": len(contested)})
summary_rows.append({"Metric": "Scanned ahead of arrival-order position (rank_gap < 0)", "Value": f"{n_jumped:,} ({n_jumped / len(contested):.1%})" if len(contested) else n_jumped})
n_fifo = int((contested["rank_gap"] == 0).sum())
n_delayed = int((contested["rank_gap"] > 0).sum())
summary_rows.append({"Metric": "Scanned in strict arrival (FIFO) order (rank_gap = 0)", "Value": f"{n_fifo:,} ({n_fifo / len(contested):.1%})" if len(contested) else n_fifo})
summary_rows.append({"Metric": "Scanned later than arrival-order position (rank_gap > 0)", "Value": f"{n_delayed:,} ({n_delayed / len(contested):.1%})" if len(contested) else n_delayed})
summary_out = pd.DataFrame(summary_rows)

out_path = os.path.join(OUT, "ct_queue_reconstruction_newdata.xlsx")
print(f"Writing {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    ct_orders_out.to_excel(writer, sheet_name="CT_Orders", index=False)
    events.to_excel(writer, sheet_name="Queue_Events", index=False)
    wait_events.to_excel(writer, sheet_name="Waiting_Events", index=False)
    summary_out.to_excel(writer, sheet_name="Summary", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
