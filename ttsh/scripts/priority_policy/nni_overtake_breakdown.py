"""
nni_overtake_breakdown.py
NNI queue only, one-off follow-up to ct_queue_priority_policy_check.py:
for each priority tier (P1/P2/P3), how many -- and what % -- of that tier's
orders were "overtaken" at least once: i.e. while still waiting (arrived,
not yet started), a STRICTLY LOWER priority order got called in ahead of
them. Also breaks each tier's overtakes down by which lower tier did the
overtaking.

This is the mirror image of ct_queue_priority_policy_check.py's
n_skipped_higher_priority (which counts, from the server's perspective, how
many higher-tier orders were skipped over) -- here we count, from the
victim's perspective, how many orders of a given tier ever got skipped over
by someone lower.

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)

Prints results; no file output.
"""
import os
import re

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
ANALYSIS = os.path.join(BASE, "analysis")

TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
PRIORITY_PATTERN = re.compile(r"(?:ED )?Scan Priority:\s*(P[1-3])")


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


print("Loading analysis/ct_queue_reconstruction.xlsx ...")
q = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")
nni = q[(q["site"] == "NNI") & q["data_quality_flag"].isna() & (q["wait_to_start_min"] >= 0)].copy()

print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
nni = nni.merge(img[["Order ID", "Ordering Question Responses"]], on="Order ID", how="left")

nni["priority_tier"] = nni["Ordering Question Responses"].apply(parse_priority)
nni["tier_rank"] = nni["priority_tier"].map(TIER_RANK)
nni["_row_id"] = range(len(nni))

meta = nni.set_index("_row_id")[["tier_rank", "parent_order_dt", "priority_tier"]].to_dict("index")

events = []
for row_id, m in meta.items():
    r = nni.loc[nni["_row_id"] == row_id].iloc[0]
    zero_wait = r["parent_order_dt"] == r["exam_start_dt"]
    events.append((r["parent_order_dt"], -1 if zero_wait else 1, "arrival", row_id))
    events.append((r["exam_start_dt"], 0, "departure", row_id))
events.sort(key=lambda e: (e[0], e[1]))

waiting = {}
# Per departing (served) order: which higher-tier orders were still waiting
# and got skipped over -- i.e. this served order's own "took over" stats.
skipped_by_server = {rid: [] for rid in meta}  # row_id (server) -> list of skipped victims' tiers
for dt, delta, etype, row_id in events:
    if etype == "arrival":
        waiting[row_id] = meta[row_id]["tier_rank"]
    else:
        this_tier = meta[row_id]["tier_rank"]
        for rid, t_rank in waiting.items():
            if rid != row_id and t_rank < this_tier:
                skipped_by_server[row_id].append(meta[rid]["priority_tier"])
        del waiting[row_id]

rows = []
for tier in ["P2", "P3", "Unknown"]:
    tier_rows = [rid for rid, m in meta.items() if m["priority_tier"] == tier]
    n_total = len(tier_rows)
    servers_that_skipped = [rid for rid in tier_rows if len(skipped_by_server[rid]) > 0]
    n_skipped_ge1 = len(servers_that_skipped)
    total_events = sum(len(skipped_by_server[rid]) for rid in tier_rows)
    by_victim_tier = {}
    for rid in tier_rows:
        for t in skipped_by_server[rid]:
            by_victim_tier[t] = by_victim_tier.get(t, 0) + 1
    rows.append({
        "Serving Tier": tier, "N Served": n_total,
        "N That Took Over >=1 Higher Order": n_skipped_ge1,
        "% That Took Over >=1": round(100 * n_skipped_ge1 / n_total, 1),
        "Total Higher-Priority Orders Taken Over": total_events,
        "Broken down by victim tier": ", ".join(f"{t}: {c}" for t, c in sorted(by_victim_tier.items())),
    })

result = pd.DataFrame(rows)
print("\nNNI: served orders that took over (skipped ahead of) a higher-priority order still waiting")
print(result.to_string(index=False))
