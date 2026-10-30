"""
nni_queue_priority_policy_check.py
NNI queue only. Tests whether the reconstructed NNI queue (build_ct_queue.py)
actually follows the stated service policy: P1 first, then P2, then P3, then
unknown-priority orders -- and first-come-first-served within each tier.

Method: replay the arrival (Parent Order time) / departure (Exam Start time)
event sweep, but at every departure ("this order's exam starts now"), check
whether the served order was actually the policy-optimal pick -- lowest
priority tier, then earliest arrival -- among every order currently in the
queue (arrived, not yet started), including the one leaving. If some other
still-waiting order should have gone first under the policy, that departure
is a violation, and we log who got skipped and how many.

Exam end time / scanner capacity is deliberately NOT modelled here -- this
is a pure sequencing check (does the order that leaves the queue rank best
among everyone in the queue at that moment?), not a capacity-aware one.

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders -- site, parent_order_dt,
                                           exam_start_dt, wait_to_start_min)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)

Outputs: analysis/nni_queue_priority_policy_check.xlsx + 1 PNG chart.
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


def check_policy_compliance(site_df):
    """Replays the arrival/departure sweep (arrival at parent_order_dt,
    departure at exam_start_dt; exam_end_dt is not used). At each departure,
    the leaving order should be whoever is currently in the queue -- arrived,
    not yet started, including itself -- with the lowest (priority_tier,
    parent_order_dt). If it isn't, that's a policy violation: we log how many
    strictly-higher-tier orders, and how many same-tier-but-earlier orders,
    were still in the queue at that moment. Returns one row per served order.
    """
    meta = site_df.set_index("_row_id")[["tier_rank", "parent_order_dt"]].to_dict("index")

    events = []
    for row_id, m in meta.items():
        r = site_df.loc[site_df["_row_id"] == row_id].iloc[0]
        zero_wait = r["parent_order_dt"] == r["exam_start_dt"]
        # Normally a departure sorts before an arrival on an exact-tie
        # timestamp (so a just-freed spot isn't double-counted against the
        # newcomer). But a zero-wait order's OWN arrival must still precede
        # its OWN departure, so that pair gets the opposite priority.
        events.append((r["parent_order_dt"], -1 if zero_wait else 1, "arrival", row_id))
        events.append((r["exam_start_dt"], 0, "departure", row_id))
    events.sort(key=lambda e: (e[0], e[1]))

    waiting = {}
    results = []
    for dt, delta, etype, row_id in events:
        if etype == "arrival":
            waiting[row_id] = meta[row_id]
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
            results.append({
                "_row_id": row_id,
                "waiting_at_service": len(waiting) - 1,
                "compliant": compliant,
                "n_skipped_higher_priority": n_skip_priority,
                "n_skipped_same_tier_earlier": n_skip_fcfs,
            })
            del waiting[row_id]
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Load & join -- NNI only
# ---------------------------------------------------------------------------
print("Loading analysis/ct_queue_reconstruction.xlsx ...")
q = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")
valid = q[(q["site"] == SITE) & (q["wait_to_start_min"] >= 0)].copy()
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
# Run the compliance check
# ---------------------------------------------------------------------------
res = check_policy_compliance(valid)
res = res.merge(valid[["_row_id", "priority_tier", "wait_to_start_min", "parent_order_dt"]], on="_row_id")

n_compliant = int(res["compliant"].sum())
print(f"\nNNI: {len(res):,} served orders | "
      f"{n_compliant:,} ({n_compliant / len(res):.1%}) served exactly as policy dictates "
      f"(lowest-tier, earliest-arrival among everyone in the queue at that moment)")
viol = res[~res["compliant"]]
n_priority_viol = int((viol["n_skipped_higher_priority"] > 0).sum())
n_fcfs_only_viol = int(((viol["n_skipped_higher_priority"] == 0) & (viol["n_skipped_same_tier_earlier"] > 0)).sum())
print(f"  Of {len(viol):,} violations: {n_priority_viol:,} skipped a higher-priority-tier order still in "
      f"the queue, {n_fcfs_only_viol:,} were same-tier FCFS-only violations (skipped an earlier arrival "
      f"in the same tier)")

# ---------------------------------------------------------------------------
# Compliance rate by priority tier of the served order
# ---------------------------------------------------------------------------
def tier_compliance_row(label, sub):
    sub_viol = sub[~sub["compliant"]]
    return {
        "Priority Tier": label, "N Served": len(sub),
        "N Policy-Compliant": int(sub["compliant"].sum()),
        "% Policy-Compliant": round(100 * sub["compliant"].mean(), 1),
        "N Violations": len(sub_viol),
        "  of which involve a skipped higher-priority order": int((sub_viol["n_skipped_higher_priority"] > 0).sum()),
        "  of which are same-tier FCFS-only violations": int(
            ((sub_viol["n_skipped_higher_priority"] == 0) & (sub_viol["n_skipped_same_tier_earlier"] > 0)).sum()
        ),
        "Mean # higher-priority orders skipped (when non-compliant)":
            round(sub_viol["n_skipped_higher_priority"].mean(), 2) if len(sub_viol) else 0,
    }


tier_compliance_rows = []
for tier in TIER_ORDER:
    sub = res[res["priority_tier"] == tier]
    if len(sub) == 0:
        continue
    tier_compliance_rows.append(tier_compliance_row(tier, sub))
tier_compliance_rows.append(tier_compliance_row("Overall", res))
tier_compliance = pd.DataFrame(tier_compliance_rows)
print("\nNNI policy compliance by priority tier of the served order:")
print(tier_compliance.to_string(index=False))

# ---------------------------------------------------------------------------
# Chart: compliance rate by tier
# ---------------------------------------------------------------------------
chart_df = tier_compliance[tier_compliance["Priority Tier"] != "Overall"]
fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = [TIER_COLORS[t] for t in chart_df["Priority Tier"]]
bars = ax.bar(chart_df["Priority Tier"], chart_df["% Policy-Compliant"], color=colors, width=0.5)
for b, n in zip(bars, chart_df["N Served"]):
    ax.annotate(f"n={n:,}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=9, color=INK_SECONDARY)
ax.set_ylim(0, 100)
ax.set_ylabel("% served exactly as policy dictates")
ax.axhline(100, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_title("NNI: does the queue actually serve P1 > P2 > P3 > Unknown, FCFS within tier?",
             color=INK_PRIMARY, fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "nni_queue_priority_policy_compliance.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved nni_queue_priority_policy_compliance.png")

# ---------------------------------------------------------------------------
# Detail: worst violations (highest-priority orders skipped over)
# ---------------------------------------------------------------------------
worst_violations = res[res["n_skipped_higher_priority"] > 0].sort_values("n_skipped_higher_priority", ascending=False)
worst_violations = worst_violations[["priority_tier", "parent_order_dt", "wait_to_start_min",
                                      "waiting_at_service", "n_skipped_higher_priority", "n_skipped_same_tier_earlier"]]

# ---------------------------------------------------------------------------
# Output workbook
# ---------------------------------------------------------------------------
overall = pd.DataFrame([{
    "Site": "NNI", "N Served": len(res),
    "N Policy-Compliant": int(res["compliant"].sum()),
    "% Policy-Compliant": round(100 * res["compliant"].mean(), 1),
    "N Violations": int((~res["compliant"]).sum()),
    "  of which involve a skipped higher-priority order": int((res["n_skipped_higher_priority"] > 0).sum()),
    "  of which are same-tier FCFS-only violations": int(
        ((res["n_skipped_higher_priority"] == 0) & (res["n_skipped_same_tier_earlier"] > 0)).sum()
    ),
}])

notes = pd.DataFrame([
    {"Note": "Scope: NNI queue only."},
    {"Note": "The chronologically last NNI scan is included because this service-sequence check uses Parent Order "
             "time and Exam Start time only; it does not require a next scan for queue-duration approximation."},
    {"Note": "Stated policy under test: P1 served before P2, P2 before P3, P3 before unknown-priority "
             "orders; within the same tier, first-come-first-served (by Parent Order time)."},
    {"Note": "Method: replays the arrival (Parent Order time) / departure (Exam Start time) sweep from "
             "build_ct_queue.py. At each departure (an order's exam actually starting), checks whether "
             "that order was the policy-optimal pick -- lowest priority tier, then earliest arrival -- "
             "among every order currently in the queue (arrived but not yet started), including itself. "
             "If a different order in the queue should have gone first, this is logged as a violation."},
    {"Note": "n_skipped_higher_priority = how many orders of a strictly higher priority tier were still "
             "in the queue (already arrived, not yet started) at the moment this order was served instead. "
             "n_skipped_same_tier_earlier = how many same-tier orders that arrived earlier were still in "
             "the queue at that moment."},
    {"Note": "Exam end time and scanner capacity are NOT modelled here -- this is a pure sequencing check "
             "against everyone in the queue at the moment of departure, not a capacity-aware one."},
])

out_path = os.path.join(ANALYSIS, "nni_queue_priority_policy_check.xlsx")
print(f"\nWriting {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    notes.to_excel(writer, sheet_name="Notes", index=False)
    overall.to_excel(writer, sheet_name="Overall_Compliance", index=False)
    tier_compliance.to_excel(writer, sheet_name="Compliance_by_Tier", index=False)
    worst_violations.to_excel(writer, sheet_name="Worst_Violations", index=False)
    res.drop(columns=["_row_id"]).to_excel(writer, sheet_name="All_Served_Orders", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
