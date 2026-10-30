"""
nni_priority_policy_simulation.py
NNI queue only. Simulates what wait_to_start_min would have looked like if
the stated priority policy (P1 > P2 > P3 > Unknown, FCFS within tier) had
actually been followed exactly -- and compares that counterfactual against
what really happened.

Key idea: the historical Exam Start timestamps aren't arbitrary -- they mark
the real moments the scanner became free to start a new exam, so they
already encode NNI's real throughput/capacity over time. Rather than
guessing how many scanners run in parallel and re-simulating service times
from scratch, this keeps that exact timeline of "slot open" moments fixed,
and only changes WHO fills each slot: instead of whoever historically got
it, the slot goes to whoever is policy-optimal (lowest priority tier, then
earliest arrival) among everyone who has arrived (Parent Order time) but
not yet been assigned a slot as of that moment.

Each order's simulated wait = (the slot time it was assigned) - (its own
arrival time). Compared against its real wait_to_start_min, tier by tier,
and against the SLA targets (P1<=30, P2<=120, P3<=360 min).

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders)
  data/nni_cleaned_orders.csv            (final NNI cohort)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)

Outputs: analysis/nni_priority_policy_simulation.xlsx + PNG charts.
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
DATA = os.path.join(BASE, "data")

TIER_COLORS = {"P1": "#2a78d6", "P2": "#eb6834", "P3": "#1baf7a", "Unknown": "#898781"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
ACTUAL_COLOR = "#898781"
SIM_COLOR = "#2a78d6"

TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
SLA_TARGETS = {"P1": 30, "P2": 120, "P3": 360}
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


def simulate_priority_order(df):
    """Keeps the real timeline of slot-open moments (each order's actual
    exam_start_dt) fixed, but reassigns each slot to whoever is
    policy-optimal (lowest tier_rank, then earliest parent_order_dt) among
    everyone who has arrived and is not yet assigned. Returns a Series of
    simulated exam-start timestamps, indexed by _row_id, in the ORIGINAL
    row's identity -- i.e. result.loc[row_id] = the slot time that order
    would have been given under strict policy order."""
    df = df.sort_values(["exam_start_dt", "_row_id"], kind="stable").copy()
    slot_rank = dict(zip(df["_row_id"], range(1, len(df) + 1)))
    meta = df.set_index("_row_id")[["tier_rank", "parent_order_dt"]].to_dict("index")

    events = []
    for row_id, m in meta.items():
        events.append((m["parent_order_dt"], 0, "arrival", row_id))
    for row_id, m in meta.items():
        r = df.loc[df["_row_id"] == row_id].iloc[0]
        events.append((r["exam_start_dt"], 1, "slot", row_id))  # row_id here just carries the slot's real timestamp
    # Arrivals sort before slots at an exact tie, so an order arriving at
    # the exact instant a slot opens is immediately eligible for it.
    events.sort(key=lambda e: (e[0], e[1]))

    waiting = {}
    assigned_start = {}
    assigned_slot_rank = {}
    for dt, delta, etype, row_id in events:
        if etype == "arrival":
            waiting[row_id] = meta[row_id]
        else:
            if not waiting:
                raise RuntimeError(f"No waiting order available for slot at {dt} -- timeline mismatch")
            best_row = min(waiting, key=lambda rid: (waiting[rid]["tier_rank"], waiting[rid]["parent_order_dt"]))
            assigned_start[best_row] = dt
            assigned_slot_rank[best_row] = slot_rank[row_id]
            del waiting[best_row]
    return pd.DataFrame(
        {
            "sim_exam_start_dt": pd.Series(assigned_start),
            "sim_service_rank": pd.Series(assigned_slot_rank),
        }
    )


def baseline_join_position_breakdown(df):
    """For each real P1/P2 arrival, compute where it would join the waiting
    line under the baseline priority/FCFS rule at that instant.

    Position is 1-indexed relative to the current waiting queue: for example,
    a P2 arrives after all waiting P1s and all earlier-arrived waiting P2s.
    immediate_scanned flags orders whose own real exam started exactly at
    arrival, so they can be separated from queued position-1 arrivals.
    """
    rows = []
    for _, r in df[df["priority_tier"].isin(["P1", "P2"])].iterrows():
        snapshot = df[
            (df["parent_order_dt"] < r["parent_order_dt"])
            & (df["exam_start_dt"] > r["parent_order_dt"])
            & (df["_row_id"] != r["_row_id"])
        ].copy()
        snapshot["waiting_so_far_min"] = (
            r["parent_order_dt"] - snapshot["parent_order_dt"]
        ).dt.total_seconds() / 60
        snapshot = snapshot.sort_values(
            ["tier_rank", "waiting_so_far_min"], ascending=[True, False]
        )
        baseline_position = 1 + int((snapshot["tier_rank"] <= r["tier_rank"]).sum())
        rows.append({
            "_row_id": r["_row_id"],
            "baseline_join_position": baseline_position,
            "baseline_waiting_ahead": baseline_position - 1,
            "waiting_queue_size_at_arrival": len(snapshot),
            "immediately_scanned": r["parent_order_dt"] == r["exam_start_dt"],
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    detail = detail.merge(
        df[["_row_id", "CSN", "Order ID", "priority_tier", "parent_order_dt",
            "exam_start_dt", "wait_to_start_min", "queue_position_at_arrival"]],
        on="_row_id",
        how="left",
    )
    detail = detail[[
        "CSN", "Order ID", "priority_tier", "parent_order_dt", "exam_start_dt",
        "wait_to_start_min", "queue_position_at_arrival",
        "waiting_queue_size_at_arrival", "baseline_join_position",
        "baseline_waiting_ahead", "immediately_scanned",
    ]]
    detail["baseline_position_bucket"] = np.where(
        detail["waiting_queue_size_at_arrival"] == 0,
        "Scanner empty",
        detail["baseline_join_position"].astype(int).astype(str),
    )

    summary = (
        detail.groupby(["priority_tier", "baseline_position_bucket"], dropna=False)
        .agg(
            n_orders=("Order ID", "size"),
            n_immediately_scanned=("immediately_scanned", "sum"),
            n_scanner_empty=("waiting_queue_size_at_arrival", lambda s: int((s == 0).sum())),
        )
        .reset_index()
    )
    tier_totals = detail.groupby("priority_tier").size().rename("tier_total")
    summary = summary.merge(tier_totals, on="priority_tier", how="left")
    summary["n_queued_not_immediate"] = summary["n_orders"] - summary["n_immediately_scanned"]
    summary["pct_of_tier"] = (summary["n_orders"] / summary["tier_total"] * 100).round(1)
    summary["pct_immediately_scanned_within_position"] = (
        summary["n_immediately_scanned"] / summary["n_orders"] * 100
    ).round(1)
    summary["sort_key"] = summary["baseline_position_bucket"].map(
        lambda v: 0 if v == "Scanner empty" else int(v)
    )
    summary = summary.sort_values(["priority_tier", "sort_key"]).drop(columns="sort_key")

    return detail, summary


# ---------------------------------------------------------------------------
# Load & join -- NNI only
# ---------------------------------------------------------------------------
print("Loading analysis/ct_queue_reconstruction.xlsx ...")
q = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")
nni = q[(q["site"] == "NNI") & (q["wait_to_start_min"] >= 0)].copy()
print(f"  NNI valid orders with a well-defined service-start time: {len(nni):,}")

print("Loading data/nni_cleaned_orders.csv ...")
final_nni = pd.read_csv(os.path.join(DATA, "nni_cleaned_orders.csv"), dtype={"CSN": str, "Order ID": str})
final_keys = set(zip(final_nni["CSN"].astype(str), final_nni["Order ID"].astype(str)))
nni["_final_key"] = list(zip(nni["CSN"].astype(str), nni["Order ID"].astype(str)))
n_before_final_filter = len(nni)
nni = nni[nni["_final_key"].isin(final_keys)].drop(columns="_final_key").copy()
print(f"  Keeping final exported NNI cohort: {len(nni):,} of {n_before_final_filter:,} rows")

print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
nni = nni.merge(img[["Order ID", "Ordering Question Responses"]], on="Order ID", how="left")

nni["priority_tier"] = nni["Ordering Question Responses"].apply(parse_priority)
nni["tier_rank"] = nni["priority_tier"].map(TIER_RANK)
nni["_row_id"] = range(len(nni))

# ---------------------------------------------------------------------------
# Run the simulation
# ---------------------------------------------------------------------------
print("\nSimulating strict priority-FCFS assignment onto the real slot timeline ...")
sim_start = simulate_priority_order(nni)
nni = nni.merge(sim_start, left_on="_row_id", right_index=True, how="left")
assert nni["sim_exam_start_dt"].notna().all(), "Every order should get exactly one simulated slot"
assert nni["sim_service_rank"].notna().all(), "Every order should get exactly one simulated slot rank"

nni["sim_wait_to_start_min"] = (nni["sim_exam_start_dt"] - nni["parent_order_dt"]).dt.total_seconds() / 60
nni["wait_delta_min"] = nni["sim_wait_to_start_min"] - nni["wait_to_start_min"]  # negative = sim is faster
nni = nni.sort_values(["exam_start_dt", "_row_id"], kind="stable").reset_index(drop=True)
nni["actual_service_rank"] = np.arange(1, len(nni) + 1)
nni["service_rank_shift"] = nni["sim_service_rank"] - nni["actual_service_rank"]

print(f"  Sanity check -- sim wait can't be negative: min = {nni['sim_wait_to_start_min'].min():.2f}")

p1_p2_baseline_detail, p1_p2_baseline_summary = baseline_join_position_breakdown(nni)
print("\nP1/P2 baseline join-position breakdown:")
print(p1_p2_baseline_summary.to_string(index=False))

# ---------------------------------------------------------------------------
# Compare actual vs simulated, by tier
# ---------------------------------------------------------------------------
summary_rows = []
for tier in TIER_ORDER:
    sub = nni[nni["priority_tier"] == tier]
    target = SLA_TARGETS.get(tier)
    actual_median = sub["wait_to_start_min"].median()
    simulated_median = sub["sim_wait_to_start_min"].median()
    row = {
        "Priority Tier": tier, "N": len(sub),
        "Actual median wait (min)": round(actual_median, 1),
        "Simulated median wait (min)": round(simulated_median, 1),
        "Median shift (sim median - actual median, min)": round(simulated_median - actual_median, 1),
        "% moved earlier": round(100 * (sub["wait_delta_min"] < 0).mean(), 1),
        "% unchanged": round(100 * (sub["wait_delta_min"] == 0).mean(), 1),
        "% moved later": round(100 * (sub["wait_delta_min"] > 0).mean(), 1),
        "Median service-rank shift (sim - actual)": round(sub["service_rank_shift"].median(), 1),
        "Mean service-rank shift (sim - actual)": round(sub["service_rank_shift"].mean(), 1),
        "% position moved earlier": round(100 * (sub["service_rank_shift"] < 0).mean(), 1),
        "% position unchanged": round(100 * (sub["service_rank_shift"] == 0).mean(), 1),
        "% position moved later": round(100 * (sub["service_rank_shift"] > 0).mean(), 1),
    }
    if target is not None:
        row["SLA target (min)"] = target
        row["Actual % exceeding target"] = round(100 * (sub["wait_to_start_min"] > target).mean(), 1)
        row["Simulated % exceeding target"] = round(100 * (sub["sim_wait_to_start_min"] > target).mean(), 1)
    summary_rows.append(row)
tier_summary = pd.DataFrame(summary_rows)
print("\nActual vs. simulated (strict priority order) wait, by tier:")
print(tier_summary.to_string(index=False))

n_faster = int((nni["wait_delta_min"] < 0).sum())
n_same = int((nni["wait_delta_min"] == 0).sum())
n_slower = int((nni["wait_delta_min"] > 0).sum())
print(f"\nUnder strict policy order: {n_faster:,} orders would wait less, {n_same:,} unchanged, "
      f"{n_slower:,} would wait more (zero-sum -- total scanner throughput is unchanged, "
      f"just redistributed by tier).")

# ---------------------------------------------------------------------------
# Chart: actual vs. simulated % exceeding SLA target, by tier (P1-P3 only --
# Unknown has no SLA target)
# ---------------------------------------------------------------------------
chart_df = tier_summary[tier_summary["Priority Tier"] != "Unknown"]
fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(chart_df))
w = 0.35
bars_a = ax.bar(x - w / 2, chart_df["Actual % exceeding target"], width=w, color=ACTUAL_COLOR, label="Actual")
bars_s = ax.bar(x + w / 2, chart_df["Simulated % exceeding target"], width=w, color=SIM_COLOR, label="Simulated (strict priority order)")
for bars in (bars_a, bars_s):
    for b in bars:
        ax.annotate(f"{b.get_height():.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9, color=INK_SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels([f"{t}\n(target {SLA_TARGETS[t]}m)" for t in chart_df["Priority Tier"]])
ax.set_ylabel("% of orders exceeding SLA target")
ax.set_ylim(top=chart_df[["Actual % exceeding target", "Simulated % exceeding target"]].max().max() * 1.25)
ax.legend(frameon=False, loc="upper right")
ax.set_title("NNI: would strict priority order fix the SLA misses?", color=INK_PRIMARY, fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "nni_priority_policy_simulation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved nni_priority_policy_simulation.png")

# ---------------------------------------------------------------------------
# Chart: 100%-stacked queue-position shift direction, by tier.
# Negative rank shift = earlier slot under simulated rule; positive = later.
# ---------------------------------------------------------------------------
rank_shift_df = tier_summary.set_index("Priority Tier").loc[TIER_ORDER]
MOVED_LATER_COLOR = TIER_COLORS["P2"]
UNCHANGED_COLOR = "#c9c7c0"

fig, ax = plt.subplots(figsize=(8.5, 4.5))
y = np.arange(len(TIER_ORDER))
earlier = rank_shift_df["% position moved earlier"]
unchanged = rank_shift_df["% position unchanged"]
later = rank_shift_df["% position moved later"]

bars_e = ax.barh(y, earlier, color=SIM_COLOR, label="Moved earlier")
bars_u = ax.barh(y, unchanged, left=earlier, color=UNCHANGED_COLOR, label="Unchanged")
bars_l = ax.barh(y, later, left=earlier + unchanged, color=MOVED_LATER_COLOR, label="Moved later")

for bars, vals in ((bars_e, earlier), (bars_u, unchanged), (bars_l, later)):
    for b, v in zip(bars, vals):
        if v >= 3:
            text_color = INK_PRIMARY if tuple(b.get_facecolor()) == tuple(plt.matplotlib.colors.to_rgba(UNCHANGED_COLOR)) else "white"
            ax.annotate(
                f"{v:.0f}%",
                (b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2),
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )
ax.set_yticks(y)
ax.set_yticklabels(TIER_ORDER)
ax.set_xlim(0, 100)
ax.set_xlabel("% of orders in tier")
ax.invert_yaxis()
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.set_title("NNI: how much would each tier's queue position shift?", color=INK_PRIMARY, fontsize=12, loc="left")
fig.tight_layout()
rank_shift_path = os.path.join(ANALYSIS, "nni_priority_policy_queue_position_shift.png")
fig.savefig(rank_shift_path, dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(ANALYSIS, "nni_priority_sequence_shift.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {os.path.basename(rank_shift_path)}")

# ---------------------------------------------------------------------------
# Output workbook
# ---------------------------------------------------------------------------
notes = pd.DataFrame([
    {"Note": "Scope: final exported NNI cohort from data/nni_cleaned_orders.csv, joined back to "
             "analysis/ct_queue_reconstruction.xlsx for the actual queue timestamps. This means rows already "
             "dropped from the final CSV, including zero scan-duration rows, are excluded from this simulation."},
    {"Note": "Method: the real Exam Start timestamps mark the actual moments the scanner became free to "
             "start a new exam -- i.e. they already encode NNI's real historical throughput/capacity over "
             "time. This simulation keeps that exact set of 'slot open' moments fixed, and only changes "
             "WHO is assigned to each slot: instead of whoever historically got it, each slot goes to "
             "whoever is policy-optimal (lowest priority tier, then earliest Parent Order time) among "
             "everyone who has arrived but not yet been assigned a slot as of that moment. No assumption "
             "about the number of physical scanners is needed, since the slot timeline is taken as-is "
             "from what actually happened."},
    {"Note": "sim_wait_to_start_min = the slot time assigned to that order under strict policy order, "
             "minus its own real Parent Order (arrival) time. This is a like-for-like counterfactual: "
             "total scanner throughput over the period is unchanged (same number of slots, same slot "
             "times) -- only the order in which patients fill those slots changes."},
    {"Note": "wait_delta_min = simulated wait - actual wait. Negative means that order would have waited "
             "LESS under strict policy order; positive means it would have waited MORE. Since total "
             "throughput is fixed, these deltas are zero-sum across the whole queue -- gains for "
             "higher-priority orders come at the direct expense of lower-priority ones."},
    {"Note": "SLA targets (P1<=30min, P2<=120min, P3<=360min) don't apply to Unknown-priority orders, so "
             "Unknown has no SLA columns in Tier_Summary and is excluded from the chart."},
    {"Note": "P1_P2_Baseline_Position shows, for each real incoming P1/P2 order, the position it would "
             "join in the waiting queue under the baseline priority/FCFS insertion rule at its Parent "
             "Order time. baseline_join_position is 1-indexed; baseline_waiting_ahead is that position "
             "minus 1. In P1_P2_Position_Summary, Scanner empty is broken out from numeric position 1 "
             "when no other order was waiting at arrival. immediately_scanned is true when the order's "
             "real Parent Order time exactly equals its real Exam Start time."},
])

out_path = os.path.join(ANALYSIS, "nni_priority_policy_simulation.xlsx")
print(f"\nWriting {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    notes.to_excel(writer, sheet_name="Notes", index=False)
    tier_summary.to_excel(writer, sheet_name="Tier_Summary", index=False)
    p1_p2_baseline_summary.to_excel(writer, sheet_name="P1_P2_Position_Summary", index=False)
    p1_p2_baseline_detail.to_excel(writer, sheet_name="P1_P2_Baseline_Position", index=False)
    nni[["CSN", "Order ID", "priority_tier", "parent_order_dt", "exam_start_dt", "sim_exam_start_dt",
         "wait_to_start_min", "sim_wait_to_start_min", "wait_delta_min",
         "actual_service_rank", "sim_service_rank", "service_rank_shift"]] \
        .sort_values("wait_delta_min").to_excel(writer, sheet_name="Order_Level", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
