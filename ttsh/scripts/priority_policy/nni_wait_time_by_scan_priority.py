"""
nni_wait_time_by_scan_priority.py
Wait time (Parent Order -> Exam Start) against the scan-level "ED Scan
Priority" (P1/P2/P3) captured in Ordering Question Responses, for both NNI
and DR queues, plus an NNI-specific data-driven look at what separates P1
orders that blow past the 30-min target from those that don't.

Note: DR has a Scan Priority answer captured for only ~6% of orders, so
DR's distribution/SLA/box-plot slices are mostly "Unknown" and are not a
meaningful test of priority-based triage there -- NNI (~94% coverage) is.

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders -- site, wait_to_start_min,
                                           queue_len_at_arrival, exam_duration_min)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)

Outputs: analysis/nni_wait_time_by_scan_priority.xlsx + 2 PNG charts.
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
import seaborn as sns
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
ANALYSIS = os.path.join(BASE, "analysis")

CAT_COLORS = {"P1": "#2a78d6", "P2": "#eb6834", "P3": "#1baf7a", "Unknown": "#898781"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
ONTIME_COLOR = "#1baf7a"
LATE_COLOR = "#c0392b"

SITES = ["NNI", "DR"]
PRIO_ORDER = ["P1", "P2", "P3"]
PRIO_ORDER_PLOT = ["P1", "P2", "P3", "Unknown"]
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


def parse_scan_priority(val):
    """Every NNI row with >1 captured value has them all identical (checked --
    no P1/P2/P3 conflicts within a row), so the first match is the priority."""
    if is_blank(val):
        return None
    m = PRIORITY_PATTERN.search(str(val))
    return m.group(1) if m else None


def summarize_dist(vals):
    vals = vals.dropna()
    if len(vals) == 0:
        return {k: None for k in ["N", "Mean", "Median", "P25", "P75", "P90", "P95", "Max"]}
    return {
        "N": len(vals), "Mean": round(vals.mean(), 1), "Median": round(vals.median(), 1),
        "P25": round(vals.quantile(0.25), 1), "P75": round(vals.quantile(0.75), 1),
        "P90": round(vals.quantile(0.90), 1), "P95": round(vals.quantile(0.95), 1),
        "Max": round(vals.max(), 1),
    }


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 50)


# ---------------------------------------------------------------------------
# Load & join: queue orders (canonical wait/queue fields) + Scan Priority,
# both sites
# ---------------------------------------------------------------------------
print("Loading analysis/ct_queue_reconstruction.xlsx ...")
q = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")
all_sites = q[q["site"].isin(SITES)].copy()

print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
all_sites = all_sites.merge(img[["Order ID", "Ordering Question Responses"]], on="Order ID", how="left")

all_sites["Scan Priority"] = all_sites["Ordering Question Responses"].apply(parse_scan_priority)

# Same wait-time validity rule as ct_queue_analysis.py / build_ct_queue.py:
# negative wait (bad timestamps) and zero-duration placeholder exam windows
# aren't real turnaround times.
all_sites["wait_valid"] = (all_sites["wait_to_start_min"] >= 0) & (all_sites["exam_duration_min"] > 0)

site_data = {}
for site in SITES:
    sub = all_sites[all_sites["site"] == site].copy()
    n_no_prio = sub["Scan Priority"].isna().sum()
    print(f"  {site} valid orders: {len(sub):,} | no Scan Priority captured: "
          f"{n_no_prio:,} ({n_no_prio / len(sub):.1%}) -- excluded from priority-split analysis")

    wait_clean = sub[sub["wait_valid"] & sub["Scan Priority"].notna()].copy()
    # Box-plot-only view: same validity rule, but keeps rows with no captured
    # priority as their own "Unknown" category instead of excluding them.
    wait_clean_plot = sub[sub["wait_valid"]].copy()
    wait_clean_plot["Scan Priority"] = wait_clean_plot["Scan Priority"].fillna("Unknown")
    site_data[site] = {"all": sub, "wait_clean": wait_clean, "wait_clean_plot": wait_clean_plot}

nni = site_data["NNI"]["all"]
wait_clean = site_data["NNI"]["wait_clean"]
print(f"  NNI usable for wait-by-priority analysis: {len(wait_clean):,}")

# ---------------------------------------------------------------------------
# 1. Distribution summary + SLA compliance, by site x scan priority
# ---------------------------------------------------------------------------
dist_rows = []
for site in SITES:
    wcp = site_data[site]["wait_clean_plot"]
    for prio in PRIO_ORDER_PLOT:
        row = {"Site": site, "Scan Priority": prio}
        row.update(summarize_dist(wcp.loc[wcp["Scan Priority"] == prio, "wait_to_start_min"]))
        dist_rows.append(row)
dist_summary = pd.DataFrame(dist_rows)
print("\nWait-to-start distribution by Site x Scan Priority (mins):")
print(dist_summary.to_string(index=False))

sla_rows = []
for site in SITES:
    sub = site_data[site]["all"]
    wc = site_data[site]["wait_clean"]
    for prio, target in SLA_TARGETS.items():
        n_captured = int((sub["Scan Priority"] == prio).sum())  # all rows w/ this priority, incl. bad-timestamp
        vals = wc.loc[wc["Scan Priority"] == prio, "wait_to_start_min"]
        n_valid = len(vals)
        n_within = int((vals <= target).sum())
        sla_rows.append({
            "Site": site, "Scan Priority": prio, "Target (mins)": target,
            "N Captured (this priority)": n_captured,
            "N Valid (usable wait time)": n_valid,
            "N Within Target": n_within,
            "% of Valid Orders": round(100 * n_within / n_valid, 1) if n_valid else None,
            "% of All Captured Orders": round(100 * n_within / n_captured, 1) if n_captured else None,
        })
sla_compliance = pd.DataFrame(sla_rows)
print("\nSLA compliance (order -> exam start):")
for r in sla_rows:
    print(f"  {r['Site']} {r['Scan Priority']} (<= {r['Target (mins)']} min): {r['N Within Target']:,} / {r['N Valid (usable wait time)']:,} "
          f"valid orders within target ({r['% of Valid Orders']}% of valid, {r['% of All Captured Orders']}% of all {r['Scan Priority']} orders)")

# ---------------------------------------------------------------------------
# 2. Box plot: wait to start by priority, with SLA target lines -- one panel
# per site, same image
# ---------------------------------------------------------------------------
def draw_priority_boxplot(ax, data, title):
    sns.boxplot(
        data=data, x="Scan Priority", y="wait_to_start_min", order=PRIO_ORDER_PLOT,
        hue="Scan Priority", palette=CAT_COLORS, legend=False, width=0.5, fliersize=2, ax=ax,
    )
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(bottom=-1)
    ax.tick_params(axis="both", labelsize=11)
    tick_labels = []
    for i, prio in enumerate(PRIO_ORDER_PLOT):
        vals = data.loc[data["Scan Priority"] == prio, "wait_to_start_min"]
        n = len(vals)
        if n == 0:
            tick_labels.append(f"{prio}\nmedian: n/a\nn=0")
            continue
        med = vals.median()
        tick_labels.append(f"{prio}\nmedian {med:.0f}m\nn={n:,}")
    ax.set_xticks(range(len(PRIO_ORDER_PLOT)))
    ax.set_xticklabels(tick_labels, fontsize=11, fontweight="bold", color=INK_PRIMARY)
    blended = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for prio, target in SLA_TARGETS.items():
        ax.axhline(target, color=CAT_COLORS[prio], linewidth=1.2, linestyle="--", alpha=0.55, zorder=1)
        ax.text(
            1.015,
            target,
            f"{prio} target: {target}m",
            transform=blended,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=CAT_COLORS[prio],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=CAT_COLORS[prio], alpha=0.9),
        )
    ax.set_xlabel("")
    ax.set_title(title, color=INK_PRIMARY, fontsize=14, fontweight="bold", loc="left")


fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
for ax, site in zip(axes, SITES):
    draw_priority_boxplot(ax, site_data[site]["wait_clean_plot"], f"{site}: wait to scan start by ED Scan Priority")
axes[0].set_ylabel("Wait to exam start (mins, symlog scale)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("")
fig.tight_layout()
fig.subplots_adjust(right=0.92, wspace=0.32)
fig.savefig(os.path.join(ANALYSIS, "nni_wait_boxplot_by_priority.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved nni_wait_boxplot_by_priority.png")

# ---------------------------------------------------------------------------
# 3. Why do P1 orders miss the 30-min target? Compare on-time vs late P1s
# across candidate factors -- all derived from data already on hand, not
# assumed.
# ---------------------------------------------------------------------------
p1 = wait_clean[wait_clean["Scan Priority"] == "P1"].copy()
p1["on_time"] = p1["wait_to_start_min"] <= 30
p1["hour"] = pd.to_datetime(p1["parent_order_dt"]).dt.hour
p1["dow"] = pd.to_datetime(p1["parent_order_dt"]).dt.day_name()
p1["is_weekend"] = p1["dow"].isin(["Saturday", "Sunday"])
p1["is_night"] = p1["hour"].between(0, 6, inclusive="both")  # 12am-6:59am, historically thinnest staffing
p1["is_combo_order"] = p1["Order Description"].str.contains(";", na=False)  # bundled multi-study order (e.g. stroke protocol)

on_time = p1[p1["on_time"]]
late = p1[~p1["on_time"]]
print(f"\nP1 orders: {len(p1):,} total | on-time (<=30min): {len(on_time):,} ({len(on_time)/len(p1):.1%}) | "
      f"late (>30min): {len(late):,} ({len(late)/len(p1):.1%})")

factor_rows = []
def add_factor(label, ontime_val, late_val, unit=""):
    factor_rows.append({"Factor": label, "On-time P1 (<=30min)": ontime_val, "Late P1 (>30min)": late_val, "Unit": unit})

add_factor("N orders", len(on_time), len(late), "orders")
add_factor("Queue length at arrival (mean)", round(on_time["queue_len_at_arrival"].mean(), 2),
           round(late["queue_len_at_arrival"].mean(), 2), "patients ahead")
add_factor("Queue length at arrival (median)", on_time["queue_len_at_arrival"].median(),
           late["queue_len_at_arrival"].median(), "patients ahead")
add_factor("Exam duration (mean)", round(on_time["exam_duration_min"].mean(), 1),
           round(late["exam_duration_min"].mean(), 1), "mins")
add_factor("Exam duration (median)", on_time["exam_duration_min"].median(),
           late["exam_duration_min"].median(), "mins")
add_factor("Bundled multi-study order (e.g. stroke protocol)", round(100 * on_time["is_combo_order"].mean(), 1),
           round(100 * late["is_combo_order"].mean(), 1), "%")
add_factor("Arrived overnight (00:00-06:59)", round(100 * on_time["is_night"].mean(), 1),
           round(100 * late["is_night"].mean(), 1), "%")
add_factor("Arrived on a weekend", round(100 * on_time["is_weekend"].mean(), 1),
           round(100 * late["is_weekend"].mean(), 1), "%")
p1_delay_factors = pd.DataFrame(factor_rows)
print("\nP1 on-time vs late -- candidate factors:")
print(p1_delay_factors.to_string(index=False))

# Hour-of-day arrival profile for each group (where in the day do late P1s cluster?)
hour_profile = pd.DataFrame({
    "On-time P1 (%)": on_time["hour"].value_counts(normalize=True).reindex(range(24), fill_value=0) * 100,
    "Late P1 (%)": late["hour"].value_counts(normalize=True).reindex(range(24), fill_value=0) * 100,
}).round(1)
hour_profile.index.name = "hour"

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
sns.boxplot(data=p1, x="on_time", y="queue_len_at_arrival", order=[True, False],
            hue="on_time", palette={True: ONTIME_COLOR, False: LATE_COLOR}, legend=False, width=0.5, fliersize=2, ax=ax)
ax.set_xticks([0, 1]); ax.set_xticklabels(["On-time (<=30min)", "Late (>30min)"])
ax.set_xlabel(""); ax.set_ylabel("Patients ahead at arrival")
ax.set_title("Queue depth at arrival", color=INK_PRIMARY, fontsize=11, loc="left")

ax = axes[0, 1]
sns.boxplot(data=p1, x="on_time", y="exam_duration_min", order=[True, False],
            hue="on_time", palette={True: ONTIME_COLOR, False: LATE_COLOR}, legend=False, width=0.5, fliersize=2, ax=ax)
ax.set_yscale("symlog", linthresh=10)
ax.set_ylim(bottom=-1)
ax.set_xticks([0, 1]); ax.set_xticklabels(["On-time (<=30min)", "Late (>30min)"])
ax.set_xlabel(""); ax.set_ylabel("Exam duration (mins, symlog)")
ax.set_title("Scan complexity (exam duration)", color=INK_PRIMARY, fontsize=11, loc="left")

ax = axes[1, 0]
bar_df = p1_delay_factors[p1_delay_factors["Factor"].isin(
    ["Bundled multi-study order (e.g. stroke protocol)", "Arrived overnight (00:00-06:59)", "Arrived on a weekend"]
)]
x = np.arange(len(bar_df))
w = 0.35
ax.bar(x - w / 2, bar_df["On-time P1 (<=30min)"], width=w, color=ONTIME_COLOR, label="On-time (<=30min)")
ax.bar(x + w / 2, bar_df["Late P1 (>30min)"], width=w, color=LATE_COLOR, label="Late (>30min)")
ax.set_xticks(x); ax.set_xticklabels(["Bundled\nmulti-study order", "Overnight\narrival", "Weekend\narrival"], fontsize=9)
ax.set_ylabel("% of group")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Order/timing characteristics", color=INK_PRIMARY, fontsize=11, loc="left")

ax = axes[1, 1]
ax.plot(hour_profile.index, hour_profile["On-time P1 (%)"], color=ONTIME_COLOR, marker="o", markersize=3, label="On-time (<=30min)")
ax.plot(hour_profile.index, hour_profile["Late P1 (%)"], color=LATE_COLOR, marker="o", markersize=3, label="Late (>30min)")
ax.set_xlabel("Hour of arrival"); ax.set_ylabel("% of group's P1 orders")
ax.set_xticks(range(0, 24, 3))
ax.legend(frameon=False, fontsize=8)
ax.set_title("When each group arrives", color=INK_PRIMARY, fontsize=11, loc="left")

fig.suptitle("NNI P1 orders: what separates on-time from late (>30 min) scans?", color=INK_PRIMARY, fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "nni_p1_delay_factors.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved nni_p1_delay_factors.png")

# ---------------------------------------------------------------------------
# Output workbook
# ---------------------------------------------------------------------------
notes = pd.DataFrame([
    {"Note": "Scope: NNI and DR CT orders with valid wait_to_start_min >= 0 and exam_duration_min > 0. "
             "The chronologically last scan at each site is included here because wait-time and target-compliance "
             "analysis do not require a next scan for queue-duration approximation."},
    {"Note": "Scan Priority is parsed from the 'ED Scan Priority' / 'Scan Priority' answer in Ordering "
             "Question Responses (P1/P2/P3). Rows where no priority was ever captured are grouped as "
             "'Unknown' in Distribution_by_Priority and the box-plot chart, but are excluded from "
             "SLA_Compliance since none of the stated SLA targets (P1/P2/P3) apply to an unlabeled order."},
    {"Note": "DR has a Scan Priority answer captured for only ~6% of orders -- its Distribution/SLA rows "
             "and box-plot P1/P2/P3 boxes are built from a small, likely non-representative slice; the "
             "bulk of DR orders fall into the chart's 'Unknown' box. NNI has ~94% coverage."},
    {"Note": "SLA targets: P1 <= 30 min, P2 <= 120 min (2h), P3 <= 360 min (6h), measured Parent Order "
             "time -> Exam Start time."},
    {"Note": "P1_Delay_Factors and P1_Hour_Profile are NNI-only: they compare NNI P1 orders that met the "
             "30-min target against those that didn't, across queue depth at arrival, exam duration, "
             "bundled multi-study orders (e.g. stroke protocol: angiogram+perfusion+brain in one order), "
             "overnight/weekend arrival, and hour-of-day arrival profile. These are candidate explanatory "
             "factors, not a causal test."},
])

out_path = os.path.join(ANALYSIS, "nni_wait_time_by_scan_priority.xlsx")
print(f"\nWriting {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    notes.to_excel(writer, sheet_name="Notes", index=False)
    dist_summary.to_excel(writer, sheet_name="Distribution_by_Priority", index=False)
    sla_compliance.to_excel(writer, sheet_name="SLA_Compliance", index=False)
    p1_delay_factors.to_excel(writer, sheet_name="P1_Delay_Factors", index=False)
    hour_profile.reset_index().to_excel(writer, sheet_name="P1_Hour_Profile", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
