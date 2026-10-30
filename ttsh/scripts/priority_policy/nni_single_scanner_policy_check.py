"""
nni_single_scanner_policy_check.py
NNI queue only. Same policy-compliance replay as
ct_queue_priority_policy_check.py (P1 before P2 before P3 before Unknown,
FCFS within tier), but under a counterfactual single-scanner assumption
(SITE_CAPACITY = 1 instead of NNI's real 2 physical scanners) -- i.e. how
much of the raw violation rate is actually explained away by having 2
scanners free at once, versus how much would still be a genuine skip even
with just 1 scanner.

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders -- site, parent_order_dt,
                                           exam_start_dt, wait_to_start_min)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)

Outputs: analysis/nni_single_scanner_policy_check.xlsx + 1 PNG chart.
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
ANALYSIS = os.path.join(BASE, "analysis")

TIER_COLORS = {"P1": "#2a78d6", "P2": "#eb6834", "P3": "#1baf7a", "Unknown": "#898781"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

SITE = "NNI"
TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
PRIORITY_PATTERN = re.compile(r"(?:ED )?Scan Priority:\s*(P[1-3])")

# Counterfactual: treat NNI as if it only had 1 scanner (real capacity is 2 --
# see ct_queue_priority_policy_check.py). With capacity=1, a start can only
# happen when the single scanner is free, so capacity_bound reduces to the
# plain `compliant` check (same as DR's real capacity=1 case there).
CAPACITY = 1

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE, "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY, "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY, "grid.color": GRIDLINE,
})


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


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 50)


def check_policy_compliance(site_df, capacity):
    """Same replay as ct_queue_priority_policy_check.py -- see that module for
    the full explanation of the event sweep and capacity-aware refinement."""
    meta = site_df.set_index("_row_id")[["tier_rank", "parent_order_dt"]].to_dict("index")

    events = []
    for row_id, m in meta.items():
        r = site_df.loc[site_df["_row_id"] == row_id].iloc[0]
        zero_wait = r["parent_order_dt"] == r["exam_start_dt"]
        events.append((r["parent_order_dt"], -1 if zero_wait else 1, "arrival", row_id))
        events.append((r["exam_start_dt"], 0, "departure", row_id))
        events.append((r["exam_end_dt"], -2, "end", row_id))
    events.sort(key=lambda e: (e[0], e[1]))

    occupied = 0
    waiting = {}
    results = []
    for dt, delta, etype, row_id in events:
        if etype == "arrival":
            waiting[row_id] = meta[row_id]
        elif etype == "end":
            occupied -= 1
        else:
            this_tier = meta[row_id]["tier_rank"]
            this_arrival = meta[row_id]["parent_order_dt"]
            best_row = min(waiting, key=lambda rid: (waiting[rid]["tier_rank"], waiting[rid]["parent_order_dt"]))
            compliant = (best_row == row_id)
            n_skip_priority = sum(
                1 for rid, m in waiting.items()
                if rid != row_id and m["tier_rank"] < this_tier
            )
            n_skip_fcfs = sum(
                1 for rid, m in waiting.items()
                if rid != row_id and m["tier_rank"] == this_tier and m["parent_order_dt"] < this_arrival
            )
            occupied_before_start = occupied
            free_before_start = capacity - occupied_before_start
            rank_position_among_waiting = 1 + n_skip_priority + n_skip_fcfs
            capacity_bound = rank_position_among_waiting > free_before_start
            compliant_capacity_aware = compliant or (not capacity_bound)
            occupied += 1
            results.append({
                "_row_id": row_id,
                "waiting_at_service": len(waiting) - 1,
                "compliant": compliant,
                "n_skipped_higher_priority": n_skip_priority,
                "n_skipped_same_tier_earlier": n_skip_fcfs,
                "occupied_before_start": occupied_before_start,
                "free_before_start": free_before_start,
                "rank_position_among_waiting": rank_position_among_waiting,
                "capacity_bound": capacity_bound,
                "compliant_capacity_aware": compliant_capacity_aware,
            })
            del waiting[row_id]
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Load & join -- NNI only
# ---------------------------------------------------------------------------
print("Loading analysis/ct_queue_reconstruction.xlsx ...")
q = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")
valid = q[(q["site"] == SITE) & q["data_quality_flag"].isna() & (q["wait_to_start_min"] >= 0)].copy()
print(f"  NNI valid orders with a well-defined service-start time: {len(valid):,}")

print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
valid = valid.merge(img[["Order ID", "Ordering Question Responses"]], on="Order ID", how="left")

valid["priority_tier"] = valid["Ordering Question Responses"].apply(parse_priority)
valid["tier_rank"] = valid["priority_tier"].map(TIER_RANK)
valid["_row_id"] = range(len(valid))

print("\nNNI priority tier composition:")
print(valid["priority_tier"].value_counts().reindex(TIER_ORDER, fill_value=0).to_string())

# ---------------------------------------------------------------------------
# Run the compliance check, capacity = 1 (counterfactual single scanner)
# ---------------------------------------------------------------------------
res = check_policy_compliance(valid, CAPACITY)
res = res.merge(valid[["_row_id", "priority_tier", "wait_to_start_min", "parent_order_dt"]], on="_row_id")

n_compliant = int(res["compliant"].sum())
print(f"\nNNI (single-scanner counterfactual, capacity={CAPACITY}): {len(res):,} served orders | "
      f"{n_compliant:,} ({n_compliant / len(res):.1%}) served exactly as policy dictates "
      f"(lowest-tier, earliest-arrival among concurrently waiting orders)")
viol = res[~res["compliant"]]
n_priority_viol = int((viol["n_skipped_higher_priority"] > 0).sum())
n_fcfs_only_viol = int(((viol["n_skipped_higher_priority"] == 0) & (viol["n_skipped_same_tier_earlier"] > 0)).sum())
print(f"  Of {len(viol):,} violations: {n_priority_viol:,} skipped a higher-priority-tier order still waiting, "
      f"{n_fcfs_only_viol:,} were same-tier FCFS-only violations (skipped an earlier arrival in the same tier)")

n_compliant_ca = int(res["compliant_capacity_aware"].sum())
n_capacity_bound = int(res["capacity_bound"].sum())
print(f"  Capacity-aware (with only {CAPACITY} scanner, this collapses to the plain compliant check -- "
      f"a start can only happen when the sole scanner is free, so any better-ranked order still waiting "
      f"is a real skip): {n_capacity_bound:,} of {len(res):,} services were a genuine capacity-bound skip; "
      f"{n_compliant_ca:,} ({n_compliant_ca / len(res):.1%}) compliant_capacity_aware")

# ---------------------------------------------------------------------------
# Compliance rate by priority tier of the SERVED order
# ---------------------------------------------------------------------------
tier_compliance_rows = []
for tier in TIER_ORDER:
    sub = res[res["priority_tier"] == tier]
    if len(sub) == 0:
        continue
    tier_compliance_rows.append({
        "Priority Tier": tier, "N Served": len(sub),
        "N Policy-Compliant": int(sub["compliant"].sum()),
        "% Policy-Compliant": round(100 * sub["compliant"].mean(), 1),
        "Mean # higher-priority orders skipped (when non-compliant)":
            round(sub.loc[~sub["compliant"], "n_skipped_higher_priority"].mean(), 2) if (~sub["compliant"]).any() else 0,
    })
tier_compliance = pd.DataFrame(tier_compliance_rows)
print("\nNNI policy compliance by priority tier of the served order (single-scanner counterfactual):")
print(tier_compliance.to_string(index=False))

# ---------------------------------------------------------------------------
# Chart: compliance rate by tier -- single-scanner counterfactual vs. real
# 2-scanner capacity-aware rate, for reference
# ---------------------------------------------------------------------------
real_res = check_policy_compliance(valid, 2)
real_res = real_res.merge(valid[["_row_id", "priority_tier"]], on="_row_id")

compare_rows = []
for tier in TIER_ORDER:
    sub1 = res[res["priority_tier"] == tier]
    sub2 = real_res[real_res["priority_tier"] == tier]
    if len(sub1) == 0:
        continue
    compare_rows.append({
        "Priority Tier": tier,
        "N": len(sub1),
        "% Compliant (raw, tier+FCFS)": round(100 * sub1["compliant"].mean(), 1),
        "% Compliant (capacity-aware, 1 scanner)": round(100 * sub1["compliant_capacity_aware"].mean(), 1),
        "% Compliant (capacity-aware, 2 scanners -- real)": round(100 * sub2["compliant_capacity_aware"].mean(), 1),
    })
compare_df = pd.DataFrame(compare_rows)
print("\nSingle-scanner counterfactual vs. real 2-scanner capacity-aware compliance, by tier:")
print(compare_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8.5, 5.5))
x = np.arange(len(compare_df))
w = 0.27
ax.bar(x - w, compare_df["% Compliant (raw, tier+FCFS)"], width=w, color="#c0392b", label="Raw (tier+FCFS)")
ax.bar(x, compare_df["% Compliant (capacity-aware, 1 scanner)"], width=w, color="#eb6834", label="Capacity-aware, 1 scanner (counterfactual)")
ax.bar(x + w, compare_df["% Compliant (capacity-aware, 2 scanners -- real)"], width=w, color="#1baf7a", label="Capacity-aware, 2 scanners (real)")
for i, row in compare_df.iterrows():
    for off, col in zip([-w, 0, w], ["% Compliant (raw, tier+FCFS)", "% Compliant (capacity-aware, 1 scanner)", "% Compliant (capacity-aware, 2 scanners -- real)"]):
        ax.annotate(f"{row[col]:.0f}%", (i + off, row[col]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels(compare_df["Priority Tier"])
ax.set_ylim(0, 100)
ax.set_ylabel("% served exactly as policy dictates")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("NNI: policy compliance under a single-scanner counterfactual vs. real 2-scanner capacity", color=INK_PRIMARY, fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "nni_single_scanner_policy_compliance.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved nni_single_scanner_policy_compliance.png")

# ---------------------------------------------------------------------------
# Output workbook
# ---------------------------------------------------------------------------
overall = pd.DataFrame([{
    "Site": "NNI", "Capacity Assumption": f"{CAPACITY} scanner (counterfactual)",
    "N Served": len(res),
    "N Policy-Compliant": int(res["compliant"].sum()),
    "% Policy-Compliant": round(100 * res["compliant"].mean(), 1),
    "N Violations": int((~res["compliant"]).sum()),
    "  of which involve a skipped higher-priority order": int((res["n_skipped_higher_priority"] > 0).sum()),
    "  of which are same-tier FCFS-only violations": int(
        ((res["n_skipped_higher_priority"] == 0) & (res["n_skipped_same_tier_earlier"] > 0)).sum()
    ),
    f"N Services at Capacity ({CAPACITY} scanner busy)": int(res["capacity_bound"].sum()),
    "N Policy-Compliant (capacity-aware)": int(res["compliant_capacity_aware"].sum()),
    "% Policy-Compliant (capacity-aware)": round(100 * res["compliant_capacity_aware"].mean(), 1),
}])

worst_violations = res[res["n_skipped_higher_priority"] > 0].sort_values("n_skipped_higher_priority", ascending=False)
worst_violations = worst_violations[["priority_tier", "parent_order_dt", "wait_to_start_min",
                                      "waiting_at_service", "n_skipped_higher_priority", "n_skipped_same_tier_earlier"]]

notes = pd.DataFrame([
    {"Note": "Scope: NNI only. This is a counterfactual re-run of ct_queue_priority_policy_check.py's "
             "compliance replay with SITE_CAPACITY forced to 1 instead of NNI's real 2 physical scanners -- "
             "i.e. 'if NNI only had 1 scanner, how much of the raw priority/FCFS violation rate would still "
             "be a genuine skip, versus explained away by having a second scanner free at that moment?'"},
    {"Note": "Method: identical event-sweep replay as ct_queue_priority_policy_check.py (arrival at "
             "Parent Order time, departure at Exam Start, scanner freed at Exam End). See that module's "
             "Notes sheet for the full model explanation."},
    {"Note": "With capacity=1, compliant_capacity_aware collapses to the plain compliant check -- a start "
             "can only happen when the sole scanner is free, so any strictly better-ranked order still "
             "waiting is a real, capacity-bound skip. This mirrors how DR (real capacity=1) already behaves "
             "in ct_queue_priority_policy_check.py."},
    {"Note": "Compare_1_vs_2_Scanner sheet shows, by tier, the raw (tier+FCFS) rate alongside the "
             "capacity-aware rate under this 1-scanner counterfactual and NNI's real 2-scanner capacity-aware "
             "rate side by side."},
])

out_path = os.path.join(ANALYSIS, "nni_single_scanner_policy_check.xlsx")
print(f"\nWriting {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    notes.to_excel(writer, sheet_name="Notes", index=False)
    overall.to_excel(writer, sheet_name="Overall_Compliance", index=False)
    tier_compliance.to_excel(writer, sheet_name="Compliance_by_Tier", index=False)
    compare_df.to_excel(writer, sheet_name="Compare_1_vs_2_Scanner", index=False)
    worst_violations.to_excel(writer, sheet_name="Worst_Violations", index=False)
    res.drop(columns=["_row_id"]).to_excel(writer, sheet_name="All_Served_Orders", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
