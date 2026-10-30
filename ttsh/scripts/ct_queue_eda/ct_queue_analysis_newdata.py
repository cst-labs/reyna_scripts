"""
ct_queue_analysis_newdata.py
Analyses the single unified CT queue (analysis/ct_queue_reconstruction_
newdata.xlsx, Jul/Aug/Sep 2026 new data) -- how busy it gets, when it peaks,
how wait time breaks down by Scan Priority (P1/P2/P3) and by Procedure
Location (Performing Location), and scan duration. Deliberate deviation
from ct_queue_analysis.py / nni_wait_time_by_scan_priority.py / ct_scan_
time_by_type.py: this is ONE queue, not split by site -- Scan Priority and
Procedure Location are cross-cutting dimensions on that single queue's data,
not separate queues.

"How busy" is sampled at each new order's arrival (queue_len_at_arrival --
how many orders were already ahead of it), one data point per arrival, same
convention as ct_queue_analysis.py.

Outputs: analysis/ct_queue_analysis_newdata.xlsx + 6 PNG charts.
"""
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms
import seaborn as sns
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS = os.path.join(BASE, "analysis")

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
QUEUE_COLOR = "#2a78d6"

PRIO_ORDER_PLOT = ["P1", "P2", "P3", "Unknown"]
PRIO_COLORS = {"P1": "#2a78d6", "P2": "#eb6834", "P3": "#1baf7a", "Unknown": "#898781"}
SLA_TARGETS = {"P1": 30, "P2": 120, "P3": 360}

LOC_SHORT = {
    "TTSH DDR-EMERGENCY [106003002]": "DDR-Emergency",
    "NNI NEURORADIOLOGY [106003044]": "NNI Neuroradiology",
    "TTSH MAIN DIAGNOSTIC RADIOLOGY [106003037]": "TTSH Diagnostic Radiology",
    "NCID DIAGNOSTIC RADIOLOGY [106003139]": "NCID Diagnostic",
    "Unknown": "Unknown",
}
LOC_ORDER_PLOT = ["DDR-Emergency", "NNI Neuroradiology", "TTSH Diagnostic Radiology", "NCID Diagnostic", "Unknown"]
LOC_COLORS = {
    "DDR-Emergency": "#eb6834", "NNI Neuroradiology": "#2a78d6",
    "TTSH Diagnostic Radiology": "#1baf7a", "NCID Diagnostic": "#8e44ad", "Unknown": "#898781",
}

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

WITHIN_HOURS_START = 8 * 60 + 30   # 08:30
WITHIN_HOURS_END = 17 * 60 + 30    # 17:30
AFTER_HOURS_LABEL = "After-Hours (17:30-08:30)"
WITHIN_HOURS_LABEL = "Within Hours (08:30-17:30)"


def is_within_hours(dt_series):
    """08:30-17:30 business hours (minute-precision); everything else is
    after-hours."""
    minute_of_day = dt_series.dt.hour * 60 + dt_series.dt.minute
    return (minute_of_day >= WITHIN_HOURS_START) & (minute_of_day < WITHIN_HOURS_END)

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE, "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY, "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY, "grid.color": GRIDLINE,
})


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 50)


def spearman(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return None
    return x[m].rank().corr(y[m].rank())


def summarize_dist(vals):
    vals = vals.dropna()
    if len(vals) == 0:
        return {k: None for k in ["N", "Mean", "Median", "P25", "P75", "P90", "P95", "Max"]}
    return {
        "N": len(vals), "Mean": round(vals.mean(), 2), "Median": round(vals.median(), 1),
        "P25": round(vals.quantile(0.25), 1), "P75": round(vals.quantile(0.75), 1),
        "P90": round(vals.quantile(0.90), 1), "P95": round(vals.quantile(0.95), 1),
        "Max": round(vals.max(), 1),
    }


def draw_boxplot(ax, data, x, y, order, palette, ylabel, title, symlog=False, sla=None,
                  box_width=0.5, tick_fontsize=10, label_fontsize=11, title_fontsize=12):
    sns.boxplot(
        data=data, x=x, y=y, order=order, hue=x, palette=palette, legend=False,
        width=box_width, fliersize=2, ax=ax,
    )
    if symlog:
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(bottom=-1)
    tick_labels = []
    for cat in order:
        vals = data.loc[data[x] == cat, y]
        n = len(vals)
        tick_labels.append(f"{cat}\nmedian: n/a\nn=0" if n == 0 else f"{cat}\nmedian {vals.median():.0f}\nn={n:,}")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(tick_labels, fontsize=tick_fontsize, fontweight="bold", color=INK_PRIMARY)
    ax.tick_params(axis="y", labelsize=max(tick_fontsize - 1, 9))
    if sla:
        blended = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for cat, target in sla.items():
            ax.axhline(target, color=palette[cat], linewidth=1.2, linestyle="--", alpha=0.55, zorder=1)
            ax.text(1.015, target, f"{cat} target: {target}m", transform=blended, ha="left", va="center",
                    fontsize=9, fontweight="bold", color=palette[cat],
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=palette[cat], alpha=0.9))
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(title, color=INK_PRIMARY, fontsize=title_fontsize, fontweight="bold", loc="left")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading analysis/ct_queue_reconstruction_newdata.xlsx ...")
orders = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction_newdata.xlsx"), sheet_name="CT_Orders")
orders["procedure_location"] = orders["procedure_location"].map(LOC_SHORT).fillna("Unknown")

# Primary ED Dx lives on the merged Imaging sheet, not the queue-reconstruction
# output -- pull it in by Order ID for the location/dx trend below.
dx = pd.read_excel(os.path.join(BASE, "merged", "ct_orders_newdata_merged.xlsx"), sheet_name="Imaging")[["Order ID", "Primary ED Dx", "ED Dx"]]
orders = orders.merge(dx, on="Order ID", how="left")

# ---------------------------------------------------------------------------
# DDR-Emergency: replace scan_duration_min with an overrun-aware method.
# gap_mean = mean gap (current Exam Start - previous order's Exam Start +
# scan_duration_min) when the current order was already queued before that
# end (Parent Order <= that end), excluding exact back-to-back (0 min).
#   next Exam Start > Exam End  -> Exam End - Exam Start + gap_mean
#   next Exam Start <= Exam End -> next Exam Start - Exam Start
#   no next scan                -> Exam End - Exam Start
# ---------------------------------------------------------------------------
ddr_sorted = orders[orders["procedure_location"] == "DDR-Emergency"].sort_values("exam_start_dt").copy()
_prev_end = ddr_sorted["exam_start_dt"].shift(1) + pd.to_timedelta(ddr_sorted["scan_duration_min"].shift(1), unit="min")
_gap_min = (ddr_sorted["exam_start_dt"] - _prev_end).dt.total_seconds() / 60
_already_queued = ddr_sorted["parent_order_dt"] <= _prev_end
gap_mean = _gap_min[_already_queued & (_gap_min != 0)].mean()

next_start = ddr_sorted["exam_start_dt"].shift(-1)
old_scan_duration = ddr_sorted["scan_duration_min"]
new_scan_duration = pd.Series(
    np.select(
        [ddr_sorted["exam_end_dt"] < next_start, ddr_sorted["exam_end_dt"] >= next_start],
        [ddr_sorted["exam_duration_min"] + gap_mean, (next_start - ddr_sorted["exam_start_dt"]).dt.total_seconds() / 60],
        default=ddr_sorted["exam_duration_min"],
    ),
    index=ddr_sorted.index,
)

scan_duration_comparison = pd.DataFrame([
    {"Method": "Old (gap-to-next-start / real duration, whichever shorter)", **summarize_dist(old_scan_duration)},
    {"Method": "New (overrun-aware, gap_mean fill)", **summarize_dist(new_scan_duration)},
])
print(f"\nDDR-Emergency scan duration, old vs. new method (gap_mean={gap_mean:.2f} min):\n",
      scan_duration_comparison.to_string(index=False))

orders["scan_duration_min"] = orders["scan_duration_min"].astype(float)
orders.loc[ddr_sorted.index, "scan_duration_min"] = new_scan_duration

valid = orders.copy()  # data_quality_flag only voids queue_len_at_arrival/queue_position_at_arrival (NaN there); other fields on that 1 row are real and should count
wait_clean = valid.copy()  # negative/zero wait_to_start_min and exam_duration_min are kept, not filtered out
wait_clean["priority_tier"] = wait_clean["priority_tier"].fillna("Unknown")
print(f"  Valid CT orders: {len(valid):,}  |  usable for wait analysis: {len(wait_clean):,}")

# No ramp-up/partial-day trimming needed: daily order volume is roughly even
# (47-86/day) across the full Jul 8 - Sep 30, 2026 window, unlike the old
# pipeline's pre-Nov-2025 sparse period.
arrivals = valid.copy()
arrivals["hour"] = arrivals["parent_order_dt"].dt.hour
arrivals["dow"] = arrivals["parent_order_dt"].dt.day_name()

# ---------------------------------------------------------------------------
# 1. Distribution summaries
# ---------------------------------------------------------------------------
overall_dist = pd.DataFrame([
    {"Metric": "Queue length at arrival (orders ahead)", **summarize_dist(valid["queue_len_at_arrival"])},
    {"Metric": "Wait to exam start (mins)", **summarize_dist(wait_clean["wait_to_start_min"])},
    {"Metric": "Scan duration (mins)", **summarize_dist(valid["scan_duration_min"])},
])
print(overall_dist.to_string(index=False))

dist_by_priority = pd.DataFrame([
    {"Scan Priority": p, **summarize_dist(wait_clean.loc[wait_clean["priority_tier"] == p, "wait_to_start_min"])}
    for p in PRIO_ORDER_PLOT
])
dist_by_location = pd.DataFrame([
    {"Procedure Location": loc, **summarize_dist(wait_clean.loc[wait_clean["procedure_location"] == loc, "wait_to_start_min"])}
    for loc in LOC_ORDER_PLOT
])
print("\nWait-to-start by Scan Priority:\n", dist_by_priority.to_string(index=False))
print("\nWait-to-start by Procedure Location:\n", dist_by_location.to_string(index=False))

sla_rows = []
for prio, target in SLA_TARGETS.items():
    n_captured = int((valid["priority_tier"] == prio).sum())
    vals = wait_clean.loc[wait_clean["priority_tier"] == prio, "wait_to_start_min"]
    n_valid = len(vals)
    n_within = int((vals <= target).sum())
    sla_rows.append({
        "Scan Priority": prio, "Target (mins)": target,
        "N Captured": n_captured, "N Valid (usable wait time)": n_valid, "N Within Target": n_within,
        "% of Valid Orders": round(100 * n_within / n_valid, 1) if n_valid else None,
    })
sla_compliance = pd.DataFrame(sla_rows)
print("\nSLA compliance:\n", sla_compliance.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Hour-of-day x day-of-week heatmap: avg. queue length at arrival, single
# unified queue (no per-site split, unlike ct_queue_analysis.py)
# ---------------------------------------------------------------------------
heatmap = arrivals.pivot_table(index="dow", columns="hour", values="queue_len_at_arrival", aggfunc="mean").reindex(DOW_ORDER)

fig, ax = plt.subplots(figsize=(11, 5.5))
sns.heatmap(heatmap, ax=ax, cmap=sns.color_palette(SEQ_BLUE, as_cmap=True), cbar_kws={"label": "Avg. orders ahead at arrival"},
            linewidths=0.5, linecolor=SURFACE)
ax.set_title("When the CT queue is busiest (avg. orders ahead at arrival, by hour & day)", color=INK_PRIMARY, fontsize=13, loc="left")
ax.set_xlabel("Hour of day")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_queue_occupancy_heatmap_newdata.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved ct_queue_occupancy_heatmap_newdata.png")

# ---------------------------------------------------------------------------
# 3. Daily trend over the data period
# ---------------------------------------------------------------------------
daily = arrivals.groupby(arrivals["parent_order_dt"].dt.date)["queue_len_at_arrival"].mean()
daily.index = pd.to_datetime(daily.index)
overall_avg = daily.mean()

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(daily.index, daily.values, color=QUEUE_COLOR, linewidth=1, alpha=0.35)
ax.plot(daily.index, daily.rolling(7, min_periods=1).mean(), color=QUEUE_COLOR, linewidth=2, label="7-day avg.")
ax.axhline(overall_avg, color=INK_MUTED, linewidth=1, linestyle="--", zorder=0)
ax.annotate(f"period avg. {overall_avg:.1f}", (daily.index[-1], overall_avg), xytext=(6, 0),
            textcoords="offset points", fontsize=9, color=INK_SECONDARY, va="center", ha="left")
ax.set_ylabel("Avg. orders\nahead at arrival")
ax.legend(loc="upper left", frameon=False)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.suptitle(f"Daily average queue length seen at arrival, {daily.index.min().date()} to {daily.index.max().date()}",
             color=INK_PRIMARY, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_queue_daily_trend_newdata.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_queue_daily_trend_newdata.png")

# ---------------------------------------------------------------------------
# 4. Box plot: queue length at arrival (single distribution, whole queue)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5, 5.5))
sns.boxplot(data=valid, y="queue_len_at_arrival", color=QUEUE_COLOR, width=0.4, fliersize=2, ax=ax)
med = valid["queue_len_at_arrival"].median()
ax.annotate(f"median {med:.0f}", (0, med), xytext=(10, 0), textcoords="offset points", fontsize=9, color=INK_SECONDARY)
ax.set_xticks([])
ax.set_ylabel("Orders already in queue at arrival")
ax.set_title("CT queue depth at arrival", color=INK_PRIMARY, fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_queue_length_boxplot_newdata.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_queue_length_boxplot_newdata.png")

# ---------------------------------------------------------------------------
# 5. Box plot: wait to start by Scan Priority, with SLA target lines
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
draw_boxplot(ax, wait_clean, "priority_tier", "wait_to_start_min", PRIO_ORDER_PLOT, PRIO_COLORS,
             "Wait to exam start (mins, symlog scale)", "Wait to scan start by Scan Priority",
             symlog=True, sla=SLA_TARGETS)
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_wait_boxplot_by_priority_newdata.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_wait_boxplot_by_priority_newdata.png")

# ---------------------------------------------------------------------------
# 6. Box plot: wait to start by Procedure Location
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
draw_boxplot(ax, wait_clean, "procedure_location", "wait_to_start_min", LOC_ORDER_PLOT, LOC_COLORS,
             "Wait to exam start (mins, symlog scale)", "Wait to scan start by Procedure Location", symlog=True,
             box_width=0.75, tick_fontsize=12, label_fontsize=12, title_fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_wait_boxplot_by_location_newdata.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_wait_boxplot_by_location_newdata.png")

# ---------------------------------------------------------------------------
# 7. Box plot: scan duration, by Procedure Location (duration profile varies
# by what's actually being scanned at each location)
# ---------------------------------------------------------------------------
dur_clean = valid.dropna(subset=["scan_duration_min"]).copy()
fig, ax = plt.subplots(figsize=(10, 6))
draw_boxplot(ax, dur_clean, "procedure_location", "scan_duration_min", LOC_ORDER_PLOT, LOC_COLORS,
             "Scan duration (mins, symlog scale)", "Scan duration by Procedure Location", symlog=True)
ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_scan_duration_boxplot_by_location_newdata.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_scan_duration_boxplot_by_location_newdata.png")

# ---------------------------------------------------------------------------
# 8. Queue length at arrival vs wait-to-start relationship
# ---------------------------------------------------------------------------
bw = 1
edges = np.arange(0, wait_clean["queue_len_at_arrival"].max() + bw * 2, bw)
sub = wait_clean.copy()
sub["bin"] = pd.cut(sub["queue_len_at_arrival"], edges, right=False)
bin_table = sub.groupby("bin", observed=True)["wait_to_start_min"].agg(
    median="median", p25=lambda v: v.quantile(0.25), p75=lambda v: v.quantile(0.75), n="count",
)
bin_table = bin_table[bin_table["n"] >= 5]
bin_table["bin_mid"] = [iv.left + bw / 2 for iv in bin_table.index]
bin_table = bin_table.reset_index(drop=True)
corr = spearman(sub["queue_len_at_arrival"], sub["wait_to_start_min"])
print(f"\n  Spearman corr(queue_len_at_arrival, wait_to_start_min) = {corr:.3f} (n={len(sub):,})")

fig, ax = plt.subplots(figsize=(7, 5))
ax.fill_between(bin_table["bin_mid"], bin_table["p25"], bin_table["p75"], color=QUEUE_COLOR, alpha=0.15, linewidth=0)
ax.plot(bin_table["bin_mid"], bin_table["median"], color=QUEUE_COLOR, linewidth=2, marker="o", markersize=5)
ax.set_yscale("symlog", linthresh=10)
ax.set_xlabel("Orders ahead at arrival")
ax.set_ylabel("Wait to exam start (mins)")
ax.set_title(f"Does arriving to a deeper queue predict a longer wait?  (Spearman ρ = {corr:.2f})",
             color=INK_PRIMARY, fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_queue_wait_relationship_newdata.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_queue_wait_relationship_newdata.png")

# ---------------------------------------------------------------------------
# 9. Scan-priority mix + temporal pattern by Procedure Location, and the raw
# rows with no Performing Location at all (follow-up investigation)
# ---------------------------------------------------------------------------
priority_by_location = pd.crosstab(orders["procedure_location"], orders["priority_tier"], normalize="index").mul(100).round(1).reindex(columns=PRIO_ORDER_PLOT)
priority_by_location.insert(0, "N", orders.groupby("procedure_location").size())
priority_by_location = priority_by_location.reindex(LOC_ORDER_PLOT).reset_index().rename(columns={"procedure_location": "Procedure Location"})

temporal_rows = []
for loc in LOC_ORDER_PLOT:
    grp = orders[orders["procedure_location"] == loc]
    if len(grp) == 0:
        continue
    hour = grp["exam_start_dt"].dt.hour
    dow_pct = grp["exam_start_dt"].dt.day_name().value_counts(normalize=True).mul(100).round(1).reindex(DOW_ORDER).fillna(0)
    peak = hour.value_counts(normalize=True).mul(100).round(1).sort_values(ascending=False).head(3)
    temporal_rows.append({
        "Procedure Location": loc, "N": len(grp),
        "Exam Start Range": f"{grp['exam_start_dt'].min()} to {grp['exam_start_dt'].max()}",
        **{f"% {d}": dow_pct[d] for d in DOW_ORDER},
        "Peak Hour 1": f"{peak.index[0]:02d}:00 ({peak.iloc[0]}%)",
        "Peak Hour 2": f"{peak.index[1]:02d}:00 ({peak.iloc[1]}%)" if len(peak) > 1 else None,
        "Peak Hour 3": f"{peak.index[2]:02d}:00 ({peak.iloc[2]}%)" if len(peak) > 2 else None,
        f"% {AFTER_HOURS_LABEL}": round(100 * (~is_within_hours(grp["exam_start_dt"])).mean(), 1),
    })
temporal_by_location = pd.DataFrame(temporal_rows)

dx_by_location_rows = []
for loc in LOC_ORDER_PLOT:
    grp = orders[orders["procedure_location"] == loc]
    top5 = grp["Primary ED Dx"].value_counts(normalize=True).mul(100).round(1).head(5)
    for rank, (dx_name, pct) in enumerate(top5.items(), start=1):
        dx_by_location_rows.append({
            "Procedure Location": loc, "Rank": rank, "Primary ED Dx": dx_name,
            "N": int((grp["Primary ED Dx"] == dx_name).sum()), "% of Location": pct,
        })
dx_by_location = pd.DataFrame(dx_by_location_rows)

# Diagram A: scan-priority mix by location (stacked bars, % labeled per segment),
# plus an "Overall" bar for the all-locations-combined distribution.
plot_locs = [loc for loc in LOC_ORDER_PLOT if (orders["procedure_location"] == loc).sum() > 0]
overall_pct = orders["priority_tier"].value_counts(normalize=True).mul(100).round(1).reindex(PRIO_ORDER_PLOT).fillna(0)
pbl = priority_by_location.set_index("Procedure Location").reindex(plot_locs)
pbl.loc["Overall"] = [len(orders)] + overall_pct.tolist()
plot_rows = plot_locs + ["Overall"]

fig, ax = plt.subplots(figsize=(12, 6))
left = np.zeros(len(plot_rows))
for prio in PRIO_ORDER_PLOT:
    vals = pbl.loc[plot_rows, prio].values
    ax.barh(plot_rows, vals, left=left, color=PRIO_COLORS[prio], label=prio)
    for y, (v, l) in enumerate(zip(vals, left)):
        if v >= 4:
            ax.text(l + v / 2, y, f"{v:.0f}%", va="center", ha="center", fontsize=11, color="white", fontweight="bold")
    left += vals
for i, loc in enumerate(plot_rows):
    ax.text(102, i, f"N={pbl.loc[loc, 'N']:,.0f}", va="center", fontsize=12, color=INK_SECONDARY)
ax.axhline(len(plot_locs) - 0.5, color=INK_PRIMARY, linewidth=1.2, linestyle="--", alpha=0.5)
ax.set_xlim(0, 118)
ax.set_xlabel("% of orders at that location", fontsize=13)
ax.set_title("Scan priority mix by Procedure Location", color=INK_PRIMARY, fontsize=16, loc="left")
ax.tick_params(axis="y", labelsize=13)
ax.tick_params(axis="x", labelsize=11)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=False, fontsize=12)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_priority_by_location_newdata.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_priority_by_location_newdata.png")

# Diagram B: top-3 ED Dx per location, as 5 separate single-axis plots sharing
# one global x-scale so bar lengths stay visually comparable across files;
# each plot uses that location's own color from LOC_COLORS.
global_xmax = dx_by_location[dx_by_location["Procedure Location"].isin(plot_locs)].groupby(
    "Procedure Location")["% of Location"].apply(lambda s: s.head(3).max()).max()
xlim = global_xmax * 1.3
xticks = np.arange(0, xlim, 10)
PLOT_RECT = [0.3, 0.15, 0.65, 0.68]  # fixed axes position (in figure fraction) -> identical
                                     # plot-area width/tick spacing on every figure, however
                                     # long the dx label text wrapped onto the left margin gets
for loc in plot_locs:
    top3 = dx_by_location[dx_by_location["Procedure Location"] == loc].head(3).iloc[::-1]
    wrapped_labels = ["\n".join(textwrap.wrap(name, width=18, break_long_words=False)) for name in top3["Primary ED Dx"]]
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_axes(PLOT_RECT)
    ax.barh(wrapped_labels, top3["% of Location"], color=LOC_COLORS[loc])
    ax.set_xlim(0, xlim)
    ax.set_xticks(xticks)
    for y, (pct, n) in enumerate(zip(top3["% of Location"], top3["N"])):
        ax.text(pct + global_xmax * 0.03, y, f"{pct}% (n={n:,})", va="center", ha="left", fontsize=14, color=INK_SECONDARY, fontweight="bold")
    ax.set_title(loc, fontsize=16, color=INK_PRIMARY, loc="left")
    ax.set_xlabel("% of location", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    slug = loc.lower().replace(" ", "_").replace("-", "_")
    fname = f"ct_dx_by_location_newdata_{slug}.png"
    fig.savefig(os.path.join(ANALYSIS, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

# NNI queue by Dx and time-of-day: Stroke vs. non-Stroke, 1h buckets of Exam
# Start hour (histogram, not a table).
nni = orders[orders["procedure_location"] == "NNI Neuroradiology"].copy()
nni["dx_group"] = np.where(nni["Primary ED Dx"] == "Stroke", "Stroke", "Non-Stroke")
nni["hour"] = nni["exam_start_dt"].dt.hour
nni_dx_by_hour = pd.crosstab(nni["hour"], nni["dx_group"]).reindex(range(24), fill_value=0)
nni_dx_by_hour = nni_dx_by_hour.reset_index().rename(columns={"hour": "Exam Start Hour"})

fig, ax = plt.subplots(figsize=(11, 5.5))
hours = nni_dx_by_hour["Exam Start Hour"]
ax.bar(hours, nni_dx_by_hour["Stroke"], color=PRIO_COLORS["P1"], label="Stroke", width=0.8)
ax.bar(hours, nni_dx_by_hour["Non-Stroke"], bottom=nni_dx_by_hour["Stroke"], color=INK_MUTED, label="Non-Stroke", width=0.8)
ax.set_xticks(range(24))
ax.set_xlabel("Exam start hour (1h buckets)")
ax.set_ylabel("NNI orders")
ax.set_title("NNI Neuroradiology: Stroke vs. non-Stroke Primary ED Dx by hour of day", color=INK_PRIMARY, fontsize=13, loc="left")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_nni_stroke_by_hour_newdata.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_nni_stroke_by_hour_newdata.png")

# ---------------------------------------------------------------------------
# 10. DDR-Emergency only: gap distribution when the current order was already
# queued before the previous scan finished (previous end = previous exam
# start + OLD-method scan_duration_min), excluding exact back-to-back (gap ==
# 0 min). Reuses _gap_min/_already_queued computed above (pre-override).
# ---------------------------------------------------------------------------
ddr_gap = _gap_min[_already_queued & (_gap_min != 0)]

ddr_gap_dist = pd.DataFrame([
    {"Metric": "DDR-Emergency: gap when already queued, excl. 0 min (mins)", **summarize_dist(ddr_gap)}
])
print(f"\nDDR-Emergency: {int(_already_queued.sum()):,} orders already queued before previous scan ended, "
      f"{len(ddr_gap):,} after excluding exact back-to-back (0 min):\n", ddr_gap_dist.to_string(index=False))

unknown_location_rows = orders.loc[orders["procedure_location"] == "Unknown",
    ["Order ID", "Order Description", "parent_order_dt", "exam_start_dt", "exam_end_dt", "priority_tier", "Primary ED Dx", "ED Dx"]
].rename(columns={
    "Order Description": "Order Description (exact)", "parent_order_dt": "Parent Order Timestamp",
    "exam_start_dt": "Exam Start Timestamp", "exam_end_dt": "Exam End Timestamp", "priority_tier": "Scan Priority",
})
print("\nPriority mix by location:\n", priority_by_location.to_string(index=False))
print("\nTop Primary ED Dx by location:\n", dx_by_location.to_string(index=False))
print("\nNNI Stroke vs. non-Stroke by hour:\n", nni_dx_by_hour.to_string(index=False))
print("\nRows with no Performing Location:\n", unknown_location_rows.to_string(index=False))

# ---------------------------------------------------------------------------
# Output workbook
# ---------------------------------------------------------------------------
dur_by_location = pd.DataFrame([
    {"Procedure Location": loc, **summarize_dist(dur_clean.loc[dur_clean["procedure_location"] == loc, "scan_duration_min"])}
    for loc in LOC_ORDER_PLOT
])
dur_by_priority = pd.DataFrame([
    {"Scan Priority": p, **summarize_dist(valid.loc[valid["priority_tier"].fillna("Unknown") == p, "scan_duration_min"])}
    for p in PRIO_ORDER_PLOT
])

corr_summary = pd.DataFrame([{"Spearman corr (queue_len_at_arrival vs wait_to_start_min)": round(corr, 3), "N": len(sub)}])

notes = pd.DataFrame([
    {"Note": "Single unified CT queue (not split by Performing Location/site) -- all 5,233 CT orders in "
             "merged/ct_orders_newdata_merged.xlsx, Jul 8 - Sep 30, 2026. Scan Priority and Procedure "
             "Location are breakdown dimensions on this one queue, not separate queues."},
    {"Note": "Distribution_Overall/By_Priority/By_Location, SLA_Compliance, boxplots and the queue-vs-wait "
             "relationship use every one of the 5,233 CT orders -- no wait_to_start_min/exam_duration_min "
             "validity filter (see the zero/negative-wait note below for how those are handled instead)."},
    {"Note": "Hourly_DOW_Occupancy and Daily_Occupancy sample queue_len_at_arrival (orders already ahead "
             "at the moment of arrival), one value per arriving order -- not a time-weighted continuous "
             "occupancy curve. No ramp-up/partial-day trimming was applied: daily order volume is roughly "
             "even (47-86/day) across the full window, unlike the old data's sparse pre-rollout period."},
    {"Note": "Scan duration = the shorter of this order's own Exam Start-to-End duration and the gap to "
             "the next order's Exam Start on the single unified queue (reinterpreting ct_scan_time_by_"
             "type.py's per-site concept for one combined queue). Queue_len_at_arrival (scanner-occupancy) "
             "is a separate calc and uses this order's own real Exam End time as its departure point, not "
             "the next-order-start approximation."},
    {"Note": "Scan Priority is parsed from the native 'Scan Priority' column (first captured P1/P2/P3 "
             "token); 314 orders have no priority captured and a handful of merged rows have genuinely "
             "conflicting priorities across their merged member orders -- both are grouped as 'Unknown'."},
    {"Note": "SLA targets: P1 <= 30 min, P2 <= 120 min (2h), P3 <= 360 min (6h), measured Parent Order "
             "time -> Exam Start time."},
    {"Note": "Zero/negative-wait and zero/negative-duration cases are NOT dropped anywhere, including in "
             "this script's own calcs. clean_ct_orders_newdata.py flags them upstream (2 negative-wait, "
             "4 zero-wait, 10 zero-duration; 16 total, logged in merged/ct_orders_newdata_merged.xlsx's "
             "'Flagged Rows' sheet) but keeps them in the main Imaging sheet; build_ct_queue_newdata.py's "
             "queue sweep counts all 5,233 (queue_len_at_arrival now departs at each order's own real Exam "
             "End time, so every order gets a value -- no more 1-row 'last_scan_excluded' gap). THIS script "
             "also includes all of them: wait_clean (feeding Distribution_By_Priority/By_Location, SLA_"
             "Compliance, the wait boxplots, and the queue-vs-wait relationship) and dur_clean "
             "(ScanDuration_By_Location boxplot) apply no wait/duration validity filter -- every sheet is "
             "keyed to the full 5,233 orders. The 2 negative-wait orders (-0.02, -0.42 min) and 10 zero/"
             "negative-duration orders are counted at face value, not clipped to zero."},
])

out_path = os.path.join(ANALYSIS, "ct_queue_analysis_newdata.xlsx")
print(f"\nWriting {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    notes.to_excel(writer, sheet_name="Notes", index=False)
    overall_dist.to_excel(writer, sheet_name="Distribution_Overall", index=False)
    dist_by_priority.to_excel(writer, sheet_name="Distribution_By_Priority", index=False)
    dist_by_location.to_excel(writer, sheet_name="Distribution_By_Location", index=False)
    sla_compliance.to_excel(writer, sheet_name="SLA_Compliance", index=False)
    dur_by_priority.to_excel(writer, sheet_name="ScanDuration_By_Priority", index=False)
    dur_by_location.to_excel(writer, sheet_name="ScanDuration_By_Location", index=False)
    heatmap.to_excel(writer, sheet_name="Hourly_DOW_Occupancy")
    daily.rename("avg_queue_len_at_arrival").reset_index().rename(columns={"parent_order_dt": "date"}).to_excel(
        writer, sheet_name="Daily_Occupancy", index=False)
    bin_table.to_excel(writer, sheet_name="Queue_vs_Wait_Bins", index=False)
    corr_summary.to_excel(writer, sheet_name="Correlations", index=False)
    priority_by_location.to_excel(writer, sheet_name="Priority_By_Location", index=False)
    temporal_by_location.to_excel(writer, sheet_name="Temporal_By_Location", index=False)
    dx_by_location.to_excel(writer, sheet_name="Dx_By_Location", index=False)
    nni_dx_by_hour.to_excel(writer, sheet_name="NNI_Stroke_By_Hour", index=False)
    ddr_gap_dist.to_excel(writer, sheet_name="DDR_Queued_Gap", index=False)
    scan_duration_comparison.to_excel(writer, sheet_name="DDR_ScanDuration_OldVsNew", index=False)
    unknown_location_rows.to_excel(writer, sheet_name="Unknown_Location_Rows", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")

# ---------------------------------------------------------------------------
# DDR-Emergency-only orders (with the new overrun-aware scan_duration_min)
# ---------------------------------------------------------------------------
ddr_out_path = os.path.join(ANALYSIS, "ct_ddr_emergency_orders_newdata.xlsx")
with pd.ExcelWriter(ddr_out_path, engine="openpyxl") as writer:
    orders[orders["procedure_location"] == "DDR-Emergency"].to_excel(writer, sheet_name="DDR_Emergency_Orders", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)
print(f"Wrote {ddr_out_path} ({int((orders['procedure_location'] == 'DDR-Emergency').sum()):,} rows)")
