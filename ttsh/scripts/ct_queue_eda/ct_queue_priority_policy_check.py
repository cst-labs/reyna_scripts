"""
ct_queue_priority_policy_check.py
Tests whether the reconstructed queue (build_ct_queue.py) at each site
actually follows the stated service policy: P1 first, then P2, then P3, then
unknown-priority orders -- and first-come-first-served within each tier.

Method: replay the same arrival/departure event sweep used for
queue_position_at_arrival (arrival at parent_order_dt, departure at
exam_start_dt), but at every departure ("this order starts its exam now"),
check whether the served order was actually the policy-optimal pick among
everyone currently waiting (lowest priority tier, then earliest arrival). If
some other still-waiting order should have gone first under the policy, that
departure is a violation, and we log who got skipped and how many.

Note: DR has a Scan Priority answer captured for only ~6% of orders (see
Notes sheet) -- DR's result is therefore mostly a test of plain FCFS, not
priority-based triage. NNI (~94% coverage) is the meaningful test of the
stated priority policy.

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders -- site, parent_order_dt,
                                           exam_start_dt, wait_to_start_min)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)

Outputs: analysis/ct_queue_priority_policy_check.xlsx + 1 PNG chart.
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

SITES = ["NNI", "DR"]
TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
PRIORITY_PATTERN = re.compile(r"(?:ED )?Scan Priority:\s*(P[1-3])")

# NNI has 2 physical CT scanners running in parallel; DR has 1. A "violation"
# under the raw priority+FCFS check below can be a false positive if it
# happened while a scanner was actually sitting free -- two orders can
# legitimately start moments apart on two different machines with zero real
# unfairness. compliant_capacity_aware (added below) only counts it as a
# genuine violation when every scanner was actually busy at that moment.
SITE_CAPACITY = {"NNI": 2, "DR": 1}

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
    """Replays the arrival/departure sweep. At each departure, the served
    order should be whoever is currently waiting with the lowest
    (priority_tier, parent_order_dt) -- if it isn't, that's a policy
    violation. Returns one row per served order.

    Also interleaves 'end' events (exam_end_dt) to track how many of the
    site's `capacity` scanners were free immediately before this departure's
    own exam started (free_before_start = capacity - occupied_before_start).

    A raw violation (compliant=False) is only a genuine queue-jump once
    capacity is taken into account: with `free_before_start` scanners free,
    the top `free_before_start` ranked waiting candidates can all start
    around now without displacing each other -- e.g. with 2 free scanners,
    the #1 and #2 ranked candidates both starting (in either order) is not a
    skip. Only when this order's own rank among the waiting set
    (rank_position_among_waiting, 1 = best) exceeds free_before_start did it
    take a slot that should have gone to a strictly better-ranked order still
    waiting -- that is a genuine, capacity-bound violation.
    """
    meta = site_df.set_index("_row_id")[["tier_rank", "parent_order_dt"]].to_dict("index")

    events = []
    for row_id, m in meta.items():
        r = site_df.loc[site_df["_row_id"] == row_id].iloc[0]
        zero_wait = r["parent_order_dt"] == r["exam_start_dt"]
        # Normally a departure sorts before an arrival on an exact-tie
        # timestamp (a freed slot isn't double-counted). But a zero-wait
        # order's OWN arrival must still precede its OWN departure, so that
        # pair gets the opposite priority. 'end' events sort before all of
        # these (delta=-2) so a scanner freed at this exact instant is
        # already available when a same-timestamp departure is counted.
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
            # Not enough free scanners for everyone strictly better-ranked
            # who's still waiting -> this order's slot should have gone to
            # one of them instead. (compliant, i.e. rank_position == 1,
            # always satisfies this too, since free_before_start >= 1 given
            # this order did start -- so the "or compliant" is implied, kept
            # for symmetry with the plain `compliant` column above.)
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
# Load & join -- both sites
# ---------------------------------------------------------------------------
print("Loading analysis/ct_queue_reconstruction.xlsx ...")
q = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")
valid = q[q["data_quality_flag"].isna() & (q["wait_to_start_min"] >= 0)].copy()
print(f"  Valid orders with a well-defined service-start time: {len(valid):,}")

print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
valid = valid.merge(img[["Order ID", "Ordering Question Responses"]], on="Order ID", how="left")

valid["priority_tier"] = valid["Ordering Question Responses"].apply(parse_priority)
valid["tier_rank"] = valid["priority_tier"].map(TIER_RANK)
valid["_row_id"] = range(len(valid))

print("\nPriority tier composition by site:")
print(valid.groupby("site")["priority_tier"].value_counts().unstack(fill_value=0).reindex(columns=TIER_ORDER).to_string())

# ---------------------------------------------------------------------------
# Run the compliance check per site
# ---------------------------------------------------------------------------
results = {}
for site in SITES:
    site_df = valid[valid["site"] == site].copy()
    res = check_policy_compliance(site_df, SITE_CAPACITY[site])
    res = res.merge(site_df[["_row_id", "priority_tier", "wait_to_start_min", "parent_order_dt"]], on="_row_id")
    results[site] = res

    n_compliant = int(res["compliant"].sum())
    print(f"\n{site}: {len(res):,} served orders | "
          f"{n_compliant:,} ({n_compliant / len(res):.1%}) served exactly as policy dictates "
          f"(lowest-tier, earliest-arrival among concurrently waiting orders)")
    viol = res[~res["compliant"]]
    n_priority_viol = int((viol["n_skipped_higher_priority"] > 0).sum())
    n_fcfs_only_viol = int(((viol["n_skipped_higher_priority"] == 0) & (viol["n_skipped_same_tier_earlier"] > 0)).sum())
    print(f"  Of {len(viol):,} violations: {n_priority_viol:,} skipped a higher-priority-tier order still waiting, "
          f"{n_fcfs_only_viol:,} were same-tier FCFS-only violations (skipped an earlier arrival in the same tier)")

    n_compliant_ca = int(res["compliant_capacity_aware"].sum())
    n_capacity_bound = int(res["capacity_bound"].sum())
    print(f"  Capacity-aware (only counts a violation when there weren't enough of the {SITE_CAPACITY[site]} "
          f"{site} scanner(s) free for every strictly-better-ranked order still waiting): "
          f"{n_capacity_bound:,} of {len(res):,} services were a genuine capacity-bound skip; "
          f"{n_compliant_ca:,} ({n_compliant_ca / len(res):.1%}) compliant_capacity_aware")

# ---------------------------------------------------------------------------
# Compliance rate by priority tier of the SERVED order (does P1 get honored
# more reliably than P3/Unknown?)
# ---------------------------------------------------------------------------
tier_compliance_rows = []
for site in SITES:
    res = results[site]
    for tier in TIER_ORDER:
        sub = res[res["priority_tier"] == tier]
        if len(sub) == 0:
            continue
        tier_compliance_rows.append({
            "Site": site, "Priority Tier": tier, "N Served": len(sub),
            "N Policy-Compliant": int(sub["compliant"].sum()),
            "% Policy-Compliant": round(100 * sub["compliant"].mean(), 1),
            "Mean # higher-priority orders skipped (when non-compliant)":
                round(sub.loc[~sub["compliant"], "n_skipped_higher_priority"].mean(), 2) if (~sub["compliant"]).any() else 0,
        })
tier_compliance = pd.DataFrame(tier_compliance_rows)
print("\nPolicy compliance by priority tier of the served order:")
print(tier_compliance.to_string(index=False))

# ---------------------------------------------------------------------------
# Chart: compliance rate by tier, one panel per site
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, site in zip(axes, SITES):
    sub = tier_compliance[tier_compliance["Site"] == site]
    colors = [TIER_COLORS[t] for t in sub["Priority Tier"]]
    bars = ax.bar(sub["Priority Tier"], sub["% Policy-Compliant"], color=colors, width=0.5)
    for b, n in zip(bars, sub["N Served"]):
        ax.annotate(f"n={n:,}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% served exactly as policy dictates" if site == "NNI" else "")
    ax.axhline(100, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_title(site, color=INK_PRIMARY, fontsize=12, loc="left")
fig.suptitle("Does the queue actually serve P1 > P2 > P3 > Unknown, FCFS within tier?",
             color=INK_PRIMARY, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_queue_priority_policy_compliance.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved ct_queue_priority_policy_compliance.png")

# ---------------------------------------------------------------------------
# Detail: worst violations (highest-priority orders skipped over)
# ---------------------------------------------------------------------------
worst_violations = {}
for site in SITES:
    res = results[site]
    w = res[res["n_skipped_higher_priority"] > 0].sort_values("n_skipped_higher_priority", ascending=False)
    worst_violations[site] = w[["priority_tier", "parent_order_dt", "wait_to_start_min",
                                 "waiting_at_service", "n_skipped_higher_priority", "n_skipped_same_tier_earlier"]]

# ---------------------------------------------------------------------------
# Output workbook
# ---------------------------------------------------------------------------
overall_rows = []
for site in SITES:
    res = results[site]
    overall_rows.append({
        "Site": site, "N Served": len(res),
        "N Policy-Compliant": int(res["compliant"].sum()),
        "% Policy-Compliant": round(100 * res["compliant"].mean(), 1),
        "N Violations": int((~res["compliant"]).sum()),
        "  of which involve a skipped higher-priority order": int((res["n_skipped_higher_priority"] > 0).sum()),
        "  of which are same-tier FCFS-only violations": int(
            ((res["n_skipped_higher_priority"] == 0) & (res["n_skipped_same_tier_earlier"] > 0)).sum()
        ),
        f"N Services at Capacity ({SITE_CAPACITY[site]} scanner(s) busy)": int(res["capacity_bound"].sum()),
        "N Policy-Compliant (capacity-aware)": int(res["compliant_capacity_aware"].sum()),
        "% Policy-Compliant (capacity-aware)": round(100 * res["compliant_capacity_aware"].mean(), 1),
    })
overall = pd.DataFrame(overall_rows)

notes = pd.DataFrame([
    {"Note": "Sites: NNI and DR, analysed independently (each queue is served by its own scanner(s))."},
    {"Note": "Stated policy under test: P1 served before P2, P2 before P3, P3 before unknown-priority "
             "orders; within the same tier, first-come-first-served (by Parent Order time)."},
    {"Note": "Method: replays the arrival (Parent Order time) / departure (Exam Start time) sweep from "
             "build_ct_queue.py. At each departure (an order's exam actually starting), checks whether "
             "that order was the policy-optimal pick -- lowest priority tier, then earliest arrival -- "
             "among every order currently waiting (arrived but not yet started). If a different waiting "
             "order should have gone first, this is logged as a violation."},
    {"Note": "DR has a Scan Priority answer captured for only ~6% of orders (61 P3 + 45 P2 + 39 P1 of "
             "2,401) -- the rest are 'Unknown' tier there. DR's compliance rate is therefore mostly a "
             "test of plain FCFS, not priority-based triage; NNI has ~94% priority coverage and is the "
             "meaningful test of the stated priority policy."},
    {"Note": "n_skipped_higher_priority = how many orders of a strictly higher priority tier were still "
             "waiting (already arrived, not yet started) at the moment this order was served instead. "
             "n_skipped_same_tier_earlier = how many same-tier orders that arrived earlier were still "
             "waiting at that moment."},
    {"Note": f"SITE_CAPACITY = {SITE_CAPACITY} (NNI has 2 physical CT scanners, DR has 1). "
             "compliant_capacity_aware sits alongside compliant, not replacing it. Model: "
             "free_before_start = capacity - occupied_before_start (how many scanners were free the instant "
             "this order started); rank_position_among_waiting = this order's own rank (1=best) among "
             "everyone currently waiting. With free_before_start scanners free, the top free_before_start "
             "ranked waiting orders can all start around now with no real skip -- e.g. on NNI's 2 scanners, "
             "the #1 and #2 ranked candidates both starting (in either order) is not unfair. Only when "
             "rank_position_among_waiting > free_before_start did this order take a slot that should have "
             "gone to a strictly better-ranked order still waiting (capacity_bound=True) -- that is a "
             "genuine, capacity-bound violation; compliant_capacity_aware = compliant OR NOT capacity_bound. "
             "For DR (capacity=1) this reduces to exactly the plain compliant check, since a start can only "
             "happen when the one scanner is free, so any 1+ better-ranked order still waiting is a real skip."},
])

out_path = os.path.join(ANALYSIS, "ct_queue_priority_policy_check.xlsx")
print(f"\nWriting {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    notes.to_excel(writer, sheet_name="Notes", index=False)
    overall.to_excel(writer, sheet_name="Overall_Compliance", index=False)
    tier_compliance.to_excel(writer, sheet_name="Compliance_by_Tier", index=False)
    for site in SITES:
        worst_violations[site].to_excel(writer, sheet_name=f"{site}_Worst_Violations", index=False)
        results[site].drop(columns=["_row_id"]).to_excel(writer, sheet_name=f"{site}_All_Served_Orders", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
