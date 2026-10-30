"""
nni_priority_sequence_rearrangement.py
NNI queue only. Models the QUEUE SEQUENCE (not just wait times) if the
stated priority policy (P1 > P2 > P3 > Unknown, FCFS within tier) had
actually been followed exactly.

Method: the real Exam Start timestamps mark the actual moments the scanner
became free to start a new exam. This script explicitly derives the
inter-slot gap duration between each order's real exam start and the next
order's real exam start (gap_to_next_min), then reconstructs the slot-
opening timeline as a cumulative sum of those gaps starting from the first
real exam start -- and asserts that this reconstruction exactly reproduces
the real exam_start_dt sequence. That reconstructed timeline (not the raw
column) is what gets fed into the reassignment sweep below, so the
derive-gaps -> cumsum -> reassign data flow is literally what the code
does, not just a description of something mathematically equivalent.

Reassignment: each reconstructed slot is filled, in chronological order, by
whoever is policy-optimal (lowest priority tier, then earliest Parent Order
time, then Order ID as a final deterministic tiebreak) among everyone who
has arrived (Parent Order time has passed) but not yet been assigned a
slot. No order can be assigned a slot before its own real arrival time
(arrival-gating enforced).

For every order this produces both a wait-time delta (as in
nni_priority_policy_simulation.py) AND a sequence-position delta: where in
the real chronological order it actually was vs. where it would be under
strict policy order.

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)

Outputs: analysis/nni_priority_sequence_rearrangement.xlsx + 2 PNG charts.
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


def rearrange_by_priority(seq_df):
    """Event-sweep: arrivals at parent_order_dt, slot-opens at
    slot_time_reconstructed (arrivals sort before slots on an exact tie, so
    an order arriving at the exact instant a slot opens is immediately
    eligible for it). At each slot, assigns to whoever is waiting with the
    lowest (tier_rank, parent_order_dt, Order ID) -- Order ID is a final
    deterministic tiebreak for the rare exact (tier_rank, parent_order_dt)
    duplicate. Returns a Series of simulated exam-start timestamps, indexed
    by Order ID."""
    meta = seq_df.set_index("Order ID")[["tier_rank", "parent_order_dt"]].to_dict("index")

    events = [(m["parent_order_dt"], 0, "arrival", oid) for oid, m in meta.items()]
    events += [(t, 1, "slot", None) for t in seq_df["slot_time_reconstructed"]]
    events.sort(key=lambda e: (e[0], e[1]))

    waiting = {}
    assigned_start = {}
    for dt, _, etype, oid in events:
        if etype == "arrival":
            waiting[oid] = meta[oid]
        else:
            if not waiting:
                raise RuntimeError(f"No waiting order available for slot at {dt} -- timeline mismatch")
            best = min(waiting, key=lambda k: (waiting[k]["tier_rank"], waiting[k]["parent_order_dt"], str(k)))
            assigned_start[best] = dt
            del waiting[best]
    return pd.Series(assigned_start, name="sim_exam_start_dt")


# ---------------------------------------------------------------------------
# Load & join -- NNI only
# ---------------------------------------------------------------------------
print("Loading analysis/ct_queue_reconstruction.xlsx ...")
q = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")
nni = q[(q["site"] == "NNI") & q["data_quality_flag"].isna() & (q["wait_to_start_min"] >= 0)].copy()
print(f"  NNI valid orders with a well-defined service-start time: {len(nni):,}")
assert nni["Order ID"].is_unique, "Order ID must be unique within this scope to use as a row key"

print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
nni = nni.merge(img[["Order ID", "Ordering Question Responses"]], on="Order ID", how="left")

nni["priority_tier"] = nni["Ordering Question Responses"].apply(parse_priority)
nni["tier_rank"] = nni["priority_tier"].map(TIER_RANK)

n_before = len(nni)

# ---------------------------------------------------------------------------
# Real chronological sequence + derived inter-slot gaps
# ---------------------------------------------------------------------------
seq = nni.sort_values(["exam_start_dt", "parent_order_dt", "Order ID"]).reset_index(drop=True)
seq["actual_sequence_position"] = np.arange(1, len(seq) + 1)
seq["gap_to_next_min"] = (seq["exam_start_dt"].shift(-1) - seq["exam_start_dt"]).dt.total_seconds() / 60

n_ties = int((seq["gap_to_next_min"] == 0).sum())
print(f"  Exact exam_start_dt ties in the real sequence: {n_ties:,} (broken by earliest parent_order_dt, then Order ID)")

# ---------------------------------------------------------------------------
# Reconstruct the slot-opening timeline from the derived gaps, and prove it
# exactly reproduces the real exam_start_dt sequence.
# ---------------------------------------------------------------------------
gap_prev_min = seq["gap_to_next_min"].shift(1).fillna(0.0)
seq["slot_time_reconstructed"] = seq["exam_start_dt"].iloc[0] + pd.to_timedelta(gap_prev_min.cumsum(), unit="min")
assert (seq["slot_time_reconstructed"] == seq["exam_start_dt"]).all(), \
    "cumsum of derived inter-slot gaps must exactly reproduce the real exam_start_dt sequence"
print("  Sanity check -- cumsum of derived gaps reproduces the real exam_start_dt sequence exactly: OK")

# ---------------------------------------------------------------------------
# Rearrange onto the reconstructed slot timeline under strict priority order
# ---------------------------------------------------------------------------
print("\nRearranging orders onto the reconstructed slot timeline under strict priority order ...")
sim_start = rearrange_by_priority(seq)
seq = seq.merge(sim_start, left_on="Order ID", right_index=True, how="left")
assert seq["sim_exam_start_dt"].notna().all(), "Every order should get exactly one simulated slot"

seq["sim_sequence_position"] = seq["sim_exam_start_dt"].rank(method="first").astype(int)
seq["sequence_position_shift"] = seq["sim_sequence_position"] - seq["actual_sequence_position"]
seq["sim_wait_to_start_min"] = (seq["sim_exam_start_dt"] - seq["parent_order_dt"]).dt.total_seconds() / 60
seq["wait_delta_min"] = seq["sim_wait_to_start_min"] - seq["wait_to_start_min"]

assert seq["sim_wait_to_start_min"].min() >= 0, "No order should be assigned a slot before its own arrival"
assert sorted(seq["actual_sequence_position"]) == list(range(1, len(seq) + 1))
assert sorted(seq["sim_sequence_position"]) == list(range(1, len(seq) + 1))
assert len(seq) == seq["Order ID"].nunique() == n_before, "Order count must be conserved through the pipeline"
print(f"  Sanity check -- sim wait can't be negative: min = {seq['sim_wait_to_start_min'].min():.2f}")
print("  Sanity check -- actual_sequence_position and sim_sequence_position are both clean 1..N permutations: OK")

n_faster = int((seq["wait_delta_min"] < 0).sum())
n_same = int((seq["wait_delta_min"] == 0).sum())
n_slower = int((seq["wait_delta_min"] > 0).sum())
print(f"\nUnder strict policy order: {n_faster:,} orders would wait less, {n_same:,} unchanged, "
      f"{n_slower:,} would wait more (zero-sum -- total scanner throughput is unchanged, "
      f"just redistributed by tier).")

# ---------------------------------------------------------------------------
# Tier_Summary -- wait-time stats (as in nni_priority_policy_simulation.py)
# plus new sequence-position-shift stats
# ---------------------------------------------------------------------------
summary_rows = []
for tier in TIER_ORDER:
    sub = seq[seq["priority_tier"] == tier]
    target = SLA_TARGETS.get(tier)
    row = {
        "Priority Tier": tier, "N": len(sub),
        "Actual median wait (min)": round(sub["wait_to_start_min"].median(), 1),
        "Simulated median wait (min)": round(sub["sim_wait_to_start_min"].median(), 1),
        "Median shift (sim - actual, min)": round(sub["wait_delta_min"].median(), 1),
    }
    if target is not None:
        row["SLA target (min)"] = target
        row["Actual % exceeding target"] = round(100 * (sub["wait_to_start_min"] > target).mean(), 1)
        row["Simulated % exceeding target"] = round(100 * (sub["sim_wait_to_start_min"] > target).mean(), 1)
    row["Mean position shift"] = round(sub["sequence_position_shift"].mean(), 1)
    row["Median position shift"] = round(sub["sequence_position_shift"].median(), 1)
    row["% moved earlier"] = round(100 * (sub["sequence_position_shift"] < 0).mean(), 1)
    row["% unchanged"] = round(100 * (sub["sequence_position_shift"] == 0).mean(), 1)
    row["% moved later"] = round(100 * (sub["sequence_position_shift"] > 0).mean(), 1)
    summary_rows.append(row)
tier_summary = pd.DataFrame(summary_rows)
print("\nActual vs. simulated (strict priority order) wait and sequence position, by tier:")
print(tier_summary.to_string(index=False))

# ---------------------------------------------------------------------------
# Chart A: actual vs. simulated % exceeding SLA target, by tier (P1-P3 only)
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
ax.set_title("NNI: SLA impact of a priority-rearranged queue", color=INK_PRIMARY, fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "nni_priority_sequence_rearrangement.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved nni_priority_sequence_rearrangement.png")

# ---------------------------------------------------------------------------
# Chart B: 100%-stacked horizontal bar of position-shift direction, by tier
# (all 4 tiers -- this metric isn't SLA-gated)
# ---------------------------------------------------------------------------
MOVED_LATER_COLOR = TIER_COLORS["P2"]
UNCHANGED_COLOR = "#c9c7c0"

fig, ax = plt.subplots(figsize=(8.5, 4.5))
y = np.arange(len(TIER_ORDER))
earlier = tier_summary.set_index("Priority Tier").loc[TIER_ORDER, "% moved earlier"]
unchanged = tier_summary.set_index("Priority Tier").loc[TIER_ORDER, "% unchanged"]
later = tier_summary.set_index("Priority Tier").loc[TIER_ORDER, "% moved later"]

bars_e = ax.barh(y, earlier, color=SIM_COLOR, label="Moved earlier")
bars_u = ax.barh(y, unchanged, left=earlier, color=UNCHANGED_COLOR, label="Unchanged")
bars_l = ax.barh(y, later, left=earlier + unchanged, color=MOVED_LATER_COLOR, label="Moved later")
for bars, vals in ((bars_e, earlier), (bars_u, unchanged), (bars_l, later)):
    for b, v in zip(bars, vals):
        if v >= 3:
            ax.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2),
                        ha="center", va="center", fontsize=9, color="white" if b.get_facecolor() != UNCHANGED_COLOR else INK_PRIMARY)
ax.set_yticks(y)
ax.set_yticklabels(TIER_ORDER)
ax.set_xlim(0, 100)
ax.set_xlabel("% of orders in tier")
ax.invert_yaxis()
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.set_title("NNI: how much would each tier's queue position shift?", color=INK_PRIMARY, fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "nni_priority_sequence_shift.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved nni_priority_sequence_shift.png")

# ---------------------------------------------------------------------------
# Output workbook
# ---------------------------------------------------------------------------
notes = pd.DataFrame([
    {"Note": "Scope: NNI queue only (data_quality_flag isna, wait_to_start_min >= 0)."},
    {"Note": "Method: gap_to_next_min is the duration from each order's real Exam Start to the next "
             "order's real Exam Start, in the real chronological sequence. The slot-opening timeline is "
             "reconstructed as a cumulative sum of these derived gaps, starting from the first real Exam "
             "Start -- and is asserted to exactly reproduce the real Exam Start sequence (a mechanical "
             "proof that 'derive gaps then rearrange' and 'reuse the real timestamps directly' are the "
             "same model, not two different ones)."},
    {"Note": "Reassignment rule: each reconstructed slot is filled, in chronological order, by whoever is "
             "policy-optimal (lowest priority tier, then earliest Parent Order time, then Order ID as a "
             "final deterministic tiebreak) among everyone who has arrived (own Parent Order time has "
             "passed) but not yet been assigned a slot. No order can be assigned a slot before its own "
             "real arrival time -- arrival-gating is enforced."},
    {"Note": "sequence_position_shift = sim_sequence_position - actual_sequence_position. NEGATIVE means "
             "that order would move to an EARLIER position (jump ahead) under strict policy order; "
             "POSITIVE means it would move to a LATER position (be pushed back); zero means unchanged. "
             "This is the same sign convention as wait_delta_min (sim - actual), so both metrics point "
             "the same direction for the same order."},
    {"Note": "gap_to_next_min is blank for the last order in the real sequence -- there is no following "
             "order to measure a gap to."},
    {"Note": "Ties in exam_start_dt (multiple orders with the identical real Exam Start timestamp) are "
             "broken deterministically by earliest Parent Order time, then by Order ID, so that "
             "actual_sequence_position and sim_sequence_position are always clean 1..N permutations, not "
             "tied/fractional ranks."},
    {"Note": "Gaps are structurally non-negative (a diff over a column already sorted ascending by "
             "itself); zero gaps correspond exactly to exam_start_dt ties and are expected, not a "
             "data-quality problem."},
    {"Note": "SLA targets (P1<=30min, P2<=120min, P3<=360min) don't apply to Unknown-priority orders, so "
             "Unknown has no SLA columns in Tier_Summary and is excluded from Chart A -- but position-"
             "shift stats and Chart B DO include Unknown, since arrival-gating and rearrangement apply to "
             "it too (tier_rank=3, always last in line)."},
    {"Note": "wait_delta_min and sequence_position_shift are both zero-sum across the whole queue: total "
             "scanner throughput over the period is unchanged (same number of slots, same slot times) -- "
             "only the order in which patients fill those slots, and their resulting wait, changes."},
])

out_path = os.path.join(ANALYSIS, "nni_priority_sequence_rearrangement.xlsx")
print(f"\nWriting {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    notes.to_excel(writer, sheet_name="Notes", index=False)
    tier_summary.to_excel(writer, sheet_name="Tier_Summary", index=False)
    seq[["CSN", "Order ID", "priority_tier", "parent_order_dt", "exam_start_dt",
         "actual_sequence_position", "gap_to_next_min",
         "sim_exam_start_dt", "sim_sequence_position", "sequence_position_shift",
         "wait_to_start_min", "sim_wait_to_start_min", "wait_delta_min"]] \
        .sort_values("actual_sequence_position").to_excel(writer, sheet_name="Sequence", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
