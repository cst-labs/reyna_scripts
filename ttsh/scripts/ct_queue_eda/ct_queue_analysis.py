"""
ct_queue_analysis.py
Analyses the reconstructed CT queues (NNI / DR) from
analysis/ct_queue_reconstruction.xlsx: how busy each queue actually gets,
when it peaks, how it's trended over the data period, and whether a
patient's position in the queue at arrival predicts how long they wait
for their scan to start.

"How busy" is sampled at each new order's arrival (queue_len_at_arrival --
how many patients were already ahead of it), not on a uniform time grid:
every arriving order contributes exactly one data point, valued at the
depth of the queue it actually walked into, rather than time-weighting a
continuously-sampled occupancy curve.

Outputs: analysis/ct_queue_analysis.xlsx + 4 PNG charts.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS = os.path.join(BASE, "analysis")

# ── Palette (matches ct_wait_time_by_consult_acuity.py for visual consistency;
# NNI/DR pair validated with the dataviz skill's palette checker: all PASS,
# worst-adjacent CVD ΔE 24.7 / normal-vision ΔE 33.6) ──
SITE_COLORS = {"NNI": "#2a78d6", "DR": "#eb6834"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

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
    """Rank-based correlation without scipy: Pearson correlation of ranks."""
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


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading analysis/ct_queue_reconstruction.xlsx ...")
orders = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")

valid = orders[orders["data_quality_flag"].isna()].copy()
# Same exclusions as ct_wait_time_by_consult_acuity.py: negative wait (bad
# timestamps) and zero-duration placeholder exam windows aren't real
# turnaround times.
wait_clean = valid[(valid["wait_to_start_min"] >= 0) & (valid["exam_duration_min"] > 0)].copy()
print(f"  Valid CT orders: {len(valid):,}  |  usable for wait analysis: {len(wait_clean):,}")

# Order volume before Nov 2025 is negligible (38 DR + 12 NNI orders across
# Jun-Oct 2025 combined, vs 400+/1300+ per month from Nov onward) -- almost
# certainly a data-extract ramp-up, not a real near-empty queue, so it would
# drag the hour/day-of-week and daily averages down with an artifact.
# The final calendar day is also partial (data-extraction cutoff mid-day),
# which would understate that one day's/hour's average, so it's trimmed too.
# Per-order metrics (wait times, queue length at arrival) have no such
# boundary effect and are NOT filtered.
OCCUPANCY_WINDOW_START = pd.Timestamp("2025-11-01")
OCCUPANCY_WINDOW_END = orders["parent_order_dt"].max().normalize()
print(f"  Occupancy analysis window: {OCCUPANCY_WINDOW_START.date()} to {OCCUPANCY_WINDOW_END.date()} "
      f"(excludes the sparse pre-rollout period and the final partial day)")

arrivals = valid[
    (valid["parent_order_dt"] >= OCCUPANCY_WINDOW_START) & (valid["parent_order_dt"] < OCCUPANCY_WINDOW_END)
].copy()
arrivals["hour"] = arrivals["parent_order_dt"].dt.hour
arrivals["dow"] = arrivals["parent_order_dt"].dt.day_name()

# ---------------------------------------------------------------------------
# 1. Distribution summary: queue length at arrival, wait to start, by site
# ---------------------------------------------------------------------------
dist_rows = []
for site in ["NNI", "DR"]:
    row = {"Site": site, "Metric": "Queue length at arrival (patients ahead)"}
    row.update(summarize_dist(valid.loc[valid["site"] == site, "queue_len_at_arrival"]))
    dist_rows.append(row)
for site in ["NNI", "DR"]:
    row = {"Site": site, "Metric": "Wait to exam start (mins)"}
    row.update(summarize_dist(wait_clean.loc[wait_clean["site"] == site, "wait_to_start_min"]))
    dist_rows.append(row)
dist_summary = pd.DataFrame(dist_rows)
print(dist_summary.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Hour-of-day x day-of-week heatmap: avg. queue length seen AT ARRIVAL
# (shared scale across sites) -- one sample per arriving order
# (queue_len_at_arrival), not a continuously time-weighted occupancy curve.
# ---------------------------------------------------------------------------
heatmaps = {}
for site in ["NNI", "DR"]:
    sub = arrivals[arrivals["site"] == site]
    pivot = sub.pivot_table(index="dow", columns="hour", values="queue_len_at_arrival", aggfunc="mean").reindex(DOW_ORDER)
    heatmaps[site] = pivot

# vmin/vmax fixed to the combined NNI+DR range so a given shade means the
# same queue depth in both panels, sharing a single colorbar/legend.
occ_vmin = min(heatmaps[s].min().min() for s in ["NNI", "DR"])
occ_vmax = max(heatmaps[s].max().max() for s in ["NNI", "DR"])

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
for ax, site in zip(axes, ["NNI", "DR"]):
    sns.heatmap(
        heatmaps[site], ax=ax, cmap=sns.color_palette(SEQ_BLUE, as_cmap=True),
        vmin=occ_vmin, vmax=occ_vmax,
        cbar=False, linewidths=0.5, linecolor=SURFACE,
    )
    ax.set_title(f"{site} queue length seen at arrival, by hour & day", color=INK_PRIMARY, fontsize=12, loc="left")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
fig.colorbar(axes[0].collections[0], ax=axes, label="Avg. patients ahead at arrival", shrink=0.85)
fig.suptitle(f"When each CT queue is busiest (avg. patients ahead at arrival, "
             f"{OCCUPANCY_WINDOW_START.date()} to {OCCUPANCY_WINDOW_END.date()})",
             color=INK_PRIMARY, fontsize=13, y=1.03)
fig.savefig(os.path.join(ANALYSIS, "ct_queue_occupancy_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_queue_occupancy_heatmap.png")

# ---------------------------------------------------------------------------
# 3. Daily trend over the data period (own panel per site — different scales)
# Avg. queue_len_at_arrival of orders arriving that calendar date -- one
# sample per arrival, not a time-weighted occupancy curve.
# ---------------------------------------------------------------------------
daily = {}
for site in ["NNI", "DR"]:
    sub = arrivals[arrivals["site"] == site].copy()
    d = sub.groupby(sub["parent_order_dt"].dt.date)["queue_len_at_arrival"].mean()
    d.index = pd.to_datetime(d.index)
    daily[site] = d


fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
for ax, site in zip(axes, ["NNI", "DR"]):
    d = daily[site]
    overall_avg = d.mean()
    ax.plot(d.index, d.values, color=SITE_COLORS[site], linewidth=1, alpha=0.35)
    ax.plot(d.index, d.rolling(7, min_periods=1).mean(), color=SITE_COLORS[site], linewidth=2,
             label=f"{site} (7-day avg.)")
    ax.axhline(overall_avg, color=INK_MUTED, linewidth=1, linestyle="--", zorder=0)
    ax.annotate(
        f"period avg. {overall_avg:.1f}",
        (d.index[-1], overall_avg), xytext=(6, 0), textcoords="offset points",
        fontsize=9, color=INK_SECONDARY, va="center", ha="left",
    )
    ax.set_ylabel("Avg. patients\nahead at arrival")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(bottom=0)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.suptitle(f"Daily average queue length seen at arrival, {OCCUPANCY_WINDOW_START.date()} to {OCCUPANCY_WINDOW_END.date()}",
             color=INK_PRIMARY, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_queue_daily_trend.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_queue_daily_trend.png")

# ---------------------------------------------------------------------------
# 4. Distribution comparison: queue length at arrival, NNI vs DR
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5.5))
sns.boxplot(
    data=valid, x="site", y="queue_len_at_arrival", order=["NNI", "DR"],
    hue="site", palette=SITE_COLORS, legend=False, width=0.5, fliersize=2, ax=ax,
)
for i, site in enumerate(["NNI", "DR"]):
    med = valid.loc[valid["site"] == site, "queue_len_at_arrival"].median()
    ax.annotate(f"median {med:.0f}", (i, med), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color=INK_SECONDARY)
ax.set_xlabel("")
ax.set_ylabel("Patients already in queue at arrival")
ax.set_title("DR's queue runs far deeper than NNI's", color=INK_PRIMARY, fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_queue_length_by_site.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_queue_length_by_site.png")

# ---------------------------------------------------------------------------
# 4b. Distribution comparison: wait to exam start, NNI vs DR
# Symlog scale: both sites have a heavy right tail (P95 already 3-8x the
# median, and a handful of multi-day-to-multi-month outliers) that would
# crush the boxes to invisibility on a linear axis.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5.5))
sns.boxplot(
    data=wait_clean, x="site", y="wait_to_start_min", order=["NNI", "DR"],
    hue="site", palette=SITE_COLORS, legend=False, width=0.5, fliersize=2, ax=ax,
)
ax.set_yscale("symlog", linthresh=10)
ax.set_ylim(bottom=-1)  # data is non-negative; hide the unused negative half of the symlog axis
for i, site in enumerate(["NNI", "DR"]):
    med = wait_clean.loc[wait_clean["site"] == site, "wait_to_start_min"].median()
    ax.annotate(f"median {med:.0f} min", (i, med), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color=INK_SECONDARY)
ax.set_xlabel("")
ax.set_ylabel("Wait to exam start (mins, symlog scale)")
ax.set_title("DR patients wait far longer to start than NNI's", color=INK_PRIMARY, fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_queue_wait_by_site.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_queue_wait_by_site.png")

# ---------------------------------------------------------------------------
# 5. Queue length at arrival vs wait-to-start relationship
# ---------------------------------------------------------------------------
BIN_WIDTH = {"NNI": 1, "DR": 5}
bin_tables = {}
correlations = {}
for site in ["NNI", "DR"]:
    sub = wait_clean[wait_clean["site"] == site].copy()
    bw = BIN_WIDTH[site]
    edges = np.arange(0, sub["queue_len_at_arrival"].max() + bw * 2, bw)
    sub["bin"] = pd.cut(sub["queue_len_at_arrival"], edges, right=False)
    grp = sub.groupby("bin", observed=True)["wait_to_start_min"].agg(
        median="median", p25=lambda v: v.quantile(0.25), p75=lambda v: v.quantile(0.75), n="count",
    )
    grp = grp[grp["n"] >= 5]  # drop sparse bins
    grp["bin_mid"] = [iv.left + bw / 2 for iv in grp.index]
    bin_tables[site] = grp.reset_index(drop=True)
    correlations[site] = spearman(sub["queue_len_at_arrival"], sub["wait_to_start_min"])
    print(f"  {site}: Spearman corr(queue_len_at_arrival, wait_to_start_min) = {correlations[site]:.3f} (n={len(sub):,})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, site in zip(axes, ["NNI", "DR"]):
    g = bin_tables[site]
    ax.fill_between(g["bin_mid"], g["p25"], g["p75"], color=SITE_COLORS[site], alpha=0.15, linewidth=0)
    ax.plot(g["bin_mid"], g["median"], color=SITE_COLORS[site], linewidth=2, marker="o", markersize=5)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Patients ahead at arrival")
    ax.set_ylabel("Wait to exam start (mins)")
    ax.set_title(f"{site}  (Spearman ρ = {correlations[site]:.2f})", color=INK_PRIMARY, fontsize=12, loc="left")
fig.suptitle("Does arriving to a deeper queue predict a longer wait?", color=INK_PRIMARY, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ANALYSIS, "ct_queue_wait_relationship.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved ct_queue_wait_relationship.png")

# ---------------------------------------------------------------------------
# Output workbook
# ---------------------------------------------------------------------------
corr_summary = pd.DataFrame([
    {"Site": site, "Spearman corr (queue_len_at_arrival vs wait_to_start_min)": round(correlations[site], 3),
     "N": len(wait_clean[wait_clean["site"] == site])}
    for site in ["NNI", "DR"]
])

daily_out = pd.concat(
    [daily[site].rename(site) for site in ["NNI", "DR"]], axis=1, sort=True
).reset_index().rename(columns={"index": "date"})

notes = pd.DataFrame([
    {"Note": "Distribution_Summary, Correlations, Queue_vs_Wait_Bins use every valid CT order "
             "(per-order metrics have no time-boundary artifact)."},
    {"Note": "Hourly_DOW_Occupancy_* and Daily_Occupancy sample queue_len_at_arrival -- how many "
             "patients were already ahead of an order the moment it entered the queue -- one value "
             "per arriving order, averaged by hour-of-day/day-of-week or by calendar date. This is "
             "NOT a time-weighted continuous occupancy curve; a queue that stays deep between "
             "arrivals but has few new orders land in a given hour/day will read lower here than a "
             "continuous-occupancy measure would show."},
    {"Note": f"Hourly_DOW_Occupancy_* and Daily_Occupancy are restricted to "
             f"{OCCUPANCY_WINDOW_START.date()} - {OCCUPANCY_WINDOW_END.date()}: order volume before "
             f"Nov 2025 is negligible (likely an incomplete data extract, not a real near-empty queue), "
             f"and the final day is trimmed because it's a partial day (extraction cutoff mid-day)."},
])

out_path = os.path.join(ANALYSIS, "ct_queue_analysis.xlsx")
print(f"Writing {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    notes.to_excel(writer, sheet_name="Notes", index=False)
    dist_summary.to_excel(writer, sheet_name="Distribution_Summary", index=False)
    heatmaps["NNI"].to_excel(writer, sheet_name="Hourly_DOW_Occupancy_NNI")
    heatmaps["DR"].to_excel(writer, sheet_name="Hourly_DOW_Occupancy_DR")
    daily_out.to_excel(writer, sheet_name="Daily_Occupancy", index=False)
    bin_tables["NNI"].to_excel(writer, sheet_name="Queue_vs_Wait_Bins_NNI", index=False)
    bin_tables["DR"].to_excel(writer, sheet_name="Queue_vs_Wait_Bins_DR", index=False)
    corr_summary.to_excel(writer, sheet_name="Correlations", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
