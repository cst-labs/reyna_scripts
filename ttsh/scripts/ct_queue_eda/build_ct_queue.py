"""
build_ct_queue.py
Reconstructs the CT scanner queue at each physical site from
merged/ct_orders_cleaned.xlsx. NNI (neuro CT scanner) and DR (general
diagnostic radiology CT scanner) are reconstructed as two independent
queues, using the canonical "Site" column assigned upstream by
clean_ct_orders.py (from Order Description via brainbodysep.tsv, with
cross-row NNI/DR merge conflicts already resolved there).

Every order enters its site's queue at Parent Order time, but two different
notions of "leaves the queue" are tracked side by side, since they answer
different questions:
  - queue_len_at_arrival:      departs at exam_end_dt_approx -- an
                                approximated scanner-occupancy count ("how
                                many orders currently have the scanner
                                busy"). exam_end_dt_approx approximates each
                                order's exam end as the next order's real
                                Exam Start time at the same site (real Exam
                                End is kept in the output as exam_end_dt for
                                reference, but is not what drives this
                                sweep) -- same technique
                                nni_priority_sequence_rearrangement.py uses
                                (gap_to_next_min). Under this approximation
                                consecutive scans at a site never overlap,
                                so there's no separate scanner-capacity
                                reasoning needed here.
  - queue_position_at_arrival: departs at Exam Start time -- waiting-line
                                count ("how many other orders are still
                                waiting for their turn"), restricted to
                                orders with a valid wait_to_start_min.

The chronologically-last order at each site has no "next" order to
approximate an end time from, so it's excluded from the reconstruction
(data_quality_flag = "last_scan_in_site_excluded").

Outputs: analysis/ct_queue_reconstruction.xlsx
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

TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
PRIORITY_PATTERN = re.compile(r"(?:ED )?Scan Priority:\s*(P[1-3])")


# ---------------------------------------------------------------------------
# Helpers (same pattern as ct_wait_time_by_consult_acuity.py)
# ---------------------------------------------------------------------------
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
    if is_blank(val):
        return "Unknown"
    m = PRIORITY_PATTERN.search(str(val))
    return m.group(1) if m else "Unknown"


def split_joined(val):
    """clean_ct_orders.py joins values from merged rows with '; '."""
    if is_blank(val):
        return [None]
    s = str(val)
    return s.split("; ") if "; " in s else [s]


def parse_date_only(date_val):
    if isinstance(date_val, (pd.Timestamp,)):
        return date_val
    s = str(date_val)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        if pd.notna(d):
            return d
    return pd.to_datetime(date_val, errors="coerce", dayfirst=True)


def combine_dt(date_val, time_val):
    """Combine a single (non-joined) date value + time value into a Timestamp."""
    if is_blank(date_val) or is_blank(time_val):
        return None
    d = parse_date_only(date_val)
    if pd.isna(d):
        return None
    dt = pd.to_datetime(f"{d.strftime('%Y-%m-%d')} {time_val}", errors="coerce")
    return None if pd.isna(dt) else dt


def earliest_parent_order_dt(date_val, time_val):
    """Parent Order Date/Time can be a '; '-joined multi-value string on rows
    clean_ct_orders.py merged from originally-separate orders placed at
    different times but with overlapping exam windows. Returns the earliest
    combined timestamp across all joined candidates."""
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


def build_waiting_queue(site_df):
    """Same event-sweep as build_queue, but the departure event fires at
    exam_start_dt instead of exam_end_dt: once an order's exam actually
    starts, it's no longer waiting in line (it's being scanned), even though
    the scanner itself stays busy until exam_end_dt. This answers 'how many
    other orders are still waiting for their turn' rather than 'how many
    orders currently have the scanner occupied' (that's what build_queue /
    queue_len_at_arrival measures). Only orders with a valid (non-negative)
    wait_to_start_min are included -- an order with a missing/bad exam_start
    timestamp has no well-defined departure-from-queue moment."""
    events = []
    for _, r in site_df.iterrows():
        events.append((r["parent_order_dt"], 1, "arrival", r["Order ID"], r["CSN"], r["_row_id"]))
        events.append((r["exam_start_dt"], -1, "departure", r["Order ID"], r["CSN"], r["_row_id"]))
    events.sort(key=lambda e: (e[0], e[1]))  # delta=-1 (departure) sorts before +1 on ties

    running = 0
    rows = []
    arrival_running = {}
    for dt, delta, etype, order_id, csn, row_id in events:
        running += delta
        assert running >= 0, f"Waiting queue length went negative at {dt}"
        rows.append({
            "datetime": dt, "event_type": etype, "Order ID": order_id,
            "CSN": csn, "running_waiting_len": running,
        })
        if etype == "arrival":
            arrival_running[row_id] = running
    return pd.DataFrame(rows), arrival_running


def build_rank_clusters(site_df):
    """Groups orders into 'waiting clusters': maximal runs of orders whose
    [parent_order_dt, exam_start_dt) waiting intervals overlap transitively
    (order A overlaps B, B overlaps C -> A, B, C share a cluster even if A
    and C don't directly overlap -- the queue moves through them as one
    continuous knot of people). Within each cluster:
      - arrival_rank: this order's rank by parent_order_dt (1 = arrived first)
      - service_rank: this order's rank by exam_start_dt (1 = scanned first)
      - rank_gap = service_rank - arrival_rank: 0 under strict FIFO service;
        negative means this order got scanned earlier than its arrival
        position alone would predict (jumped ahead of others in its
        cluster); positive means it got scanned later (was jumped over).
    Only meaningful within a cluster of size > 1 -- a lone order has no one
    to jump ahead of or fall behind."""
    df = site_df.sort_values("parent_order_dt").reset_index(drop=True)
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


def time_weighted_mean(events_df, col="running_queue_len"):
    """Average concurrent queue length, weighted by how long each level held."""
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


# ---------------------------------------------------------------------------
# Load & assign each CT order to its physical queue (NNI or DR)
# ---------------------------------------------------------------------------
print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")

ct = img[img["Site"].notna()].copy()
ct["site"] = ct["Site"]
ct["priority_tier"] = ct["Ordering Question Responses"].apply(parse_priority)
print(f"  CT rows: {len(ct):,} of {len(img):,} total imaging rows")
print("  Site split:")
print(ct["site"].value_counts().to_string())


# ---------------------------------------------------------------------------
# Arrival (Parent Order time) / departure (Exam End time) per order
# ---------------------------------------------------------------------------
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

# exam_end_dt_approx: this order's exam end approximated as the next order's
# real Exam Start at the same site (gap-to-next-start -- same technique
# nni_priority_sequence_rearrangement.py uses for gap_to_next_min), instead
# of this order's own real exam_end_dt. Requires sorting by exam_start_dt
# first so groupby(...).shift(-1) picks up the chronologically-next order.
ct = ct.sort_values("exam_start_dt").reset_index(drop=True)
ct["exam_end_dt_approx"] = ct.groupby("site")["exam_start_dt"].shift(-1)
ct["exam_duration_min_approx"] = ct.apply(
    lambda r: (r["exam_end_dt_approx"] - r["exam_start_dt"]).total_seconds() / 60
    if pd.notna(r["exam_end_dt_approx"]) and pd.notna(r["exam_start_dt"]) else np.nan,
    axis=1,
)


def dq_flag(row):
    if pd.isna(row["parent_order_dt"]):
        return "missing_timestamp"
    if pd.isna(row["exam_end_dt_approx"]):
        # Either no real exam_start_dt to approximate from, or (the common
        # case) this is the chronologically-last order at its site with no
        # "next" order to derive an approximated end time from.
        return "last_scan_in_site_excluded"
    return None


ct["data_quality_flag"] = ct.apply(dq_flag, axis=1)
n_flagged = ct["data_quality_flag"].notna().sum()
print(f"  Data-quality flagged (excluded from queue sweep): {n_flagged:,} of {len(ct):,}")


# ---------------------------------------------------------------------------
# Per-site queue sweep: arrival (+1) at parent_order_dt, departure (-1) at
# exam_end_dt_approx. On an exact-tie timestamp, departures process before
# arrivals so a slot freed at that instant isn't double-counted as occupied.
# ---------------------------------------------------------------------------
ct["_row_id"] = range(len(ct))
valid = ct[ct["data_quality_flag"].isna()].copy()


def build_queue(site_df):
    events = []
    for _, r in site_df.iterrows():
        events.append((r["parent_order_dt"], 1, "arrival", r["Order ID"], r["CSN"], r["_row_id"]))
        events.append((r["exam_end_dt_approx"], -1, "departure", r["Order ID"], r["CSN"], r["_row_id"]))
    events.sort(key=lambda e: (e[0], e[1]))  # delta=-1 (departure) sorts before +1 on ties

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


nni_events, nni_arrival = build_queue(valid[valid["site"] == "NNI"])
dr_events, dr_arrival = build_queue(valid[valid["site"] == "DR"])

arrival_running_all = {**nni_arrival, **dr_arrival}
ct["queue_len_at_arrival"] = ct["_row_id"].map(arrival_running_all) - 1

for site, events in [("NNI", nni_events), ("DR", dr_events)]:
    print(f"  {site}: {len(events) // 2:,} orders in queue sweep, "
          f"max concurrent {events['running_queue_len'].max()}, "
          f"time-weighted mean concurrent {time_weighted_mean(events):.2f}")


# ---------------------------------------------------------------------------
# Waiting-line sweep: same idea, but an order departs the line at
# exam_start_dt (when it actually gets called in) instead of exam_end_dt
# (when the scanner frees up). Only orders with a valid wait_to_start_min
# have a well-defined "left the line" moment, so this sweep runs on a
# narrower population than the occupancy sweep above.
# ---------------------------------------------------------------------------
wait_valid = valid[valid["wait_to_start_min"] >= 0].copy()
print(f"  Waiting-line sweep population: {len(wait_valid):,} of {len(valid):,} valid orders "
      f"(excludes {len(valid) - len(wait_valid):,} with missing/negative wait_to_start_min)")

nni_wait_events, nni_wait_arrival = build_waiting_queue(wait_valid[wait_valid["site"] == "NNI"])
dr_wait_events, dr_wait_arrival = build_waiting_queue(wait_valid[wait_valid["site"] == "DR"])

wait_arrival_all = {**nni_wait_arrival, **dr_wait_arrival}
ct["queue_position_at_arrival"] = ct["_row_id"].map(wait_arrival_all) - 1

for site, events in [("NNI", nni_wait_events), ("DR", dr_wait_events)]:
    print(f"  {site}: {len(events) // 2:,} orders in waiting-line sweep, "
          f"max concurrent waiting {events['running_waiting_len'].max()}, "
          f"time-weighted mean waiting {time_weighted_mean(events, col='running_waiting_len'):.2f}")


# ---------------------------------------------------------------------------
# Rank: arrival order vs. actual service order, within each waiting cluster
# -- did this order get scanned ahead of / behind where its arrival time
# alone would place it?
# ---------------------------------------------------------------------------
nni_ranks = build_rank_clusters(wait_valid[wait_valid["site"] == "NNI"])
dr_ranks = build_rank_clusters(wait_valid[wait_valid["site"] == "DR"])
ranks_all = pd.concat([nni_ranks, dr_ranks], ignore_index=True).set_index("_row_id")
ct = ct.merge(
    ranks_all[["wait_cluster_id", "wait_cluster_size", "arrival_rank", "service_rank", "rank_gap"]],
    left_on="_row_id", right_index=True, how="left",
)

for site, ranks in [("NNI", nni_ranks), ("DR", dr_ranks)]:
    contested = ranks[ranks["wait_cluster_size"] > 1]
    n_jumped = int((contested["rank_gap"] < 0).sum())
    print(f"  {site}: {len(contested):,} orders in a contested cluster (2+ overlapping), "
          f"{n_jumped:,} ({n_jumped / len(contested):.1%}) scanned ahead of their arrival-order position")


# ---------------------------------------------------------------------------
# Output workbook
# ---------------------------------------------------------------------------
ct_orders_out = ct[[
    "CSN", "Order ID", "Order Description", "site",
    "parent_order_dt", "exam_start_dt", "exam_end_dt", "exam_end_dt_approx",
    "wait_to_start_min", "exam_duration_min", "exam_duration_min_approx",
    "queue_len_at_arrival", "queue_position_at_arrival",
    "wait_cluster_id", "wait_cluster_size", "arrival_rank", "service_rank", "rank_gap",
    "data_quality_flag", "priority_tier",
]].sort_values(["site", "parent_order_dt"]).reset_index(drop=True)

summary_rows = [{"Metric": "Total CT orders", "Value": len(ct)}]
for site in ["NNI", "DR"]:
    summary_rows.append({"Metric": f"{site} orders", "Value": int((ct["site"] == site).sum())})
summary_rows.append({"Metric": "Data quality flagged (excluded from queue sweep)", "Value": int(n_flagged)})
for flag, cnt in ct["data_quality_flag"].value_counts(dropna=True).items():
    summary_rows.append({"Metric": f"  {flag}", "Value": int(cnt)})
summary_rows.append({"Metric": "-- queue_len_at_arrival: scanner-occupancy count (departs at exam_end_dt_approx, i.e. next order's real exam start at the same site) --", "Value": ""})
for site, events in [("NNI", nni_events), ("DR", dr_events)]:
    if len(events):
        summary_rows.append({"Metric": f"{site} max concurrent queue length", "Value": int(events["running_queue_len"].max())})
        summary_rows.append({"Metric": f"{site} time-weighted mean concurrent queue length", "Value": round(time_weighted_mean(events), 2)})
summary_rows.append({"Metric": "-- queue_position_at_arrival: waiting-line count (departs at exam_start) --", "Value": ""})
summary_rows.append({"Metric": "Orders excluded from waiting-line sweep (missing/negative wait_to_start_min)", "Value": int(len(valid) - len(wait_valid))})
for site, events in [("NNI", nni_wait_events), ("DR", dr_wait_events)]:
    if len(events):
        summary_rows.append({"Metric": f"{site} max concurrent waiting", "Value": int(events["running_waiting_len"].max())})
        summary_rows.append({"Metric": f"{site} time-weighted mean waiting", "Value": round(time_weighted_mean(events, col="running_waiting_len"), 2)})
summary_rows.append({"Metric": "-- arrival_rank vs. service_rank: queue-jumping within contested waiting clusters --", "Value": ""})
for site, ranks in [("NNI", nni_ranks), ("DR", dr_ranks)]:
    contested = ranks[ranks["wait_cluster_size"] > 1]
    n_jumped = int((contested["rank_gap"] < 0).sum())
    n_fifo = int((contested["rank_gap"] == 0).sum())
    n_delayed = int((contested["rank_gap"] > 0).sum())
    summary_rows.append({"Metric": f"{site} orders in a contested cluster (2+ overlapping)", "Value": len(contested)})
    summary_rows.append({"Metric": f"{site} scanned ahead of arrival-order position (rank_gap < 0)", "Value": f"{n_jumped:,} ({n_jumped / len(contested):.1%})" if len(contested) else n_jumped})
    summary_rows.append({"Metric": f"{site} scanned in strict arrival (FIFO) order (rank_gap = 0)", "Value": f"{n_fifo:,} ({n_fifo / len(contested):.1%})" if len(contested) else n_fifo})
    summary_rows.append({"Metric": f"{site} scanned later than arrival-order position (rank_gap > 0)", "Value": f"{n_delayed:,} ({n_delayed / len(contested):.1%})" if len(contested) else n_delayed})
summary_out = pd.DataFrame(summary_rows)

out_path = os.path.join(OUT, "ct_queue_reconstruction.xlsx")
print(f"Writing {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    ct_orders_out.to_excel(writer, sheet_name="CT_Orders", index=False)
    nni_events.to_excel(writer, sheet_name="NNI_Queue_Events", index=False)
    dr_events.to_excel(writer, sheet_name="DR_Queue_Events", index=False)
    nni_wait_events.to_excel(writer, sheet_name="NNI_Waiting_Events", index=False)
    dr_wait_events.to_excel(writer, sheet_name="DR_Waiting_Events", index=False)
    summary_out.to_excel(writer, sheet_name="Summary", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
