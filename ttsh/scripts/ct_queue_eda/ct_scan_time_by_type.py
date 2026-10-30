"""
ct_scan_time_by_type.py
For each CT type, computes distribution stats of:
  1. exam_duration_min (Exam End - Exam Start)               -> Duration_By_Type_*
  2. scan_duration_min (shorter of this exam's start-to-end duration
     and the time from this exam's start to the next exam's start on the
     same physical scanner queue)                            -> NNI/DR_ScanTime_By_Type_*

Both analyses are run twice, on two different row granularities, so the two
can be compared side by side:

  - *_Split:  merged/ct_orders_combined.xlsx -- the pre-merge, one-row-per-
    original-order Imaging data. clean_ct_orders.py's non-merge cleaning
    steps (Order Completion fallback, Site assignment, validity filter) are
    re-applied here at the individual-row level (no row-merging).
  - *_Merged: merged/ct_orders_cleaned.xlsx -- the post-merge rows, where
    clean_ct_orders.py has already collapsed same-CSN overlapping/identical
    exam windows into one row (Exam Start/End = earliest start / latest end
    across the merged group, Order Description "; "-joined if the merged
    session covered 2+ distinct types).

Outputs:
  - analysis/ct_scan_time_by_type.xlsx
  - analysis/scan_duration_boxplot_nni_dr.png
"""
import os
import math
from datetime import datetime, timedelta, date, time as time_type

import pandas as pd
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

SITE_COLORS = {"NNI": "#2a78d6", "DR": "#eb6834"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def is_blank(v):
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 50)


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


def with_scan_duration(site_df):
    """Sort a site's rows by exam_start_dt and compute the capped scan duration.

    time_to_next_start_min is retained for auditability. scan_duration_min is
    the shorter of that next-start gap and the row's own start-to-end exam
    duration. If one side is unavailable, the available side is used.
    """
    s = site_df.sort_values("exam_start_dt", kind="stable").reset_index(drop=True)
    next_start = s["exam_start_dt"].shift(-1)
    s["time_to_next_start_min"] = (next_start - s["exam_start_dt"]).dt.total_seconds() / 60
    s["scan_duration_min"] = s[["time_to_next_start_min", "exam_duration_min"]].min(axis=1, skipna=True)
    return s


def scan_time_by_type(site_df):
    rows = []
    for desc, grp in site_df.groupby("Order Description"):
        rows.append({"Order Description": desc, **summarize_dist(grp["scan_duration_min"])})
    return pd.DataFrame(rows).sort_values("N", ascending=False).reset_index(drop=True)


# =============================================================================
# SPLIT pipeline: merged/ct_orders_combined.xlsx (one row per original order)
# =============================================================================
def load_sheet(path, sheet):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return pd.DataFrame()
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    seen = {}; uh = []
    for h in headers:
        if h in seen: seen[h] += 1; uh.append(f"{h}_{seen[h]}")
        else: seen[h] = 0; uh.append(h)
    return pd.DataFrame(rows[1:], columns=uh)


def to_dt(val):
    if val is None:
        return None
    try:
        if isinstance(val, float) and math.isnan(val):
            return None
    except Exception:
        pass
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        return datetime(val.year, val.month, val.day)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
                    "%d/%m/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return datetime(1899, 12, 30) + timedelta(days=f)
    except Exception:
        return None


def to_time(val):
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.time()
    if isinstance(val, time_type):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"):
            try:
                return datetime.strptime(s, fmt).time()
            except ValueError:
                continue
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        if 0 <= f < 2:
            ts = round(f * 86400); h, r = divmod(ts, 3600); m, s = divmod(r, 60)
            return time_type(h % 24, m, s)
    except Exception:
        pass
    return None


def combine_dt_raw(d_val, t_val):
    d = to_dt(d_val)
    if d is None:
        return None
    if t_val is None:
        return d
    t = to_time(t_val)
    if t is None:
        return d
    return datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)


def fallback_exam_time(row):
    """Same logic as clean_ct_orders.py's fallback_exam_time, applied per
    individual row here instead of per merged group."""
    start_missing = is_blank(row.get("Exam Start Date")) or is_blank(row.get("Exam Start Time"))
    end_missing   = is_blank(row.get("Exam End Date"))   or is_blank(row.get("Exam End Time"))
    oc_date, oc_time = row.get("Order Completion Date"), row.get("Order Completion Time")
    oc_available = not is_blank(oc_date) and not is_blank(oc_time)

    source = "Exam"
    if start_missing:
        if oc_available:
            row["Exam Start Date"] = oc_date
            row["Exam Start Time"] = oc_time
            source = "Order Completion"
        else:
            source = "Missing" if end_missing else source
    if end_missing:
        if oc_available:
            row["Exam End Date"] = oc_date
            row["Exam End Time"] = oc_time
            if source == "Exam":
                source = "Order Completion"
        else:
            if start_missing:
                source = "Missing"
    row["Exam Time Source"] = source
    return row


print("Loading merged/ct_orders_combined.xlsx ...")
df = load_sheet(os.path.join(MERGED, "ct_orders_combined.xlsx"), "Imaging")
n_loaded_split = len(df)
print(f"  Loaded {n_loaded_split:,} imaging rows")

print("Applying Order Completion fallback for missing Exam Start/End...")
df = df.apply(fallback_exam_time, axis=1)
n_missing_split = (df["Exam Time Source"] == "Missing").sum()
n_oc_split = (df["Exam Time Source"] == "Order Completion").sum()
print(f"  Exam Time Source: Exam={n_loaded_split - n_missing_split - n_oc_split:,}  "
      f"Order Completion={n_oc_split:,}  Missing={n_missing_split:,}")

df["exam_start_dt"] = df.apply(lambda r: combine_dt_raw(r["Exam Start Date"], r["Exam Start Time"]), axis=1)
df["exam_end_dt"] = df.apply(lambda r: combine_dt_raw(r["Exam End Date"], r["Exam End Time"]), axis=1)
df["parent_order_dt"] = df.apply(lambda r: combine_dt_raw(r["Parent Order Date"], r["Parent Order Time"]), axis=1)

print("Loading brainbodysep.tsv for CT site mapping...")
site_map_df = pd.read_csv(os.path.join(BASE, "brainbodysep.tsv"), sep="\t")
SITE_MAP = dict(zip(site_map_df["Order Type"].str.strip(), site_map_df["Site of Scan"].str.strip()))

df["Site"] = df["Order Description"].apply(
    lambda v: SITE_MAP.get(str(v).strip()) if not is_blank(v) else None
)
ct = df[df["Site"].notna()].copy()
print(f"  CT rows (Site assigned): {len(ct):,} of {len(df):,}")
print(f"  Site split: {ct['Site'].value_counts().to_dict()}")

print("Dropping rows with no Parent Order time / no Exam Start / non-positive exam time window / negative wait time...")
no_parent_order = ct["parent_order_dt"].isna()
no_exam_start = ct["exam_start_dt"].isna()
non_positive_duration = (
    ct["exam_start_dt"].notna()
    & ct["exam_end_dt"].notna()
    & (ct["exam_end_dt"] <= ct["exam_start_dt"])
)
wait_to_start_min = (ct["exam_start_dt"] - ct["parent_order_dt"]).dt.total_seconds() / 60
negative_wait = wait_to_start_min.notna() & (wait_to_start_min < 0)
zero_wait = wait_to_start_min.notna() & (wait_to_start_min == 0)
drop_mask_split = (no_parent_order | no_exam_start | non_positive_duration | negative_wait)

print(f"  no Parent Order time: {int(no_parent_order.sum()):,}")
print(f"  no Exam Start: {int(no_exam_start.sum()):,}")
print(f"  non-positive exam time window: {int(non_positive_duration.sum()):,}")
print(f"  negative wait time: {int(negative_wait.sum()):,}")
print(f"  zero wait time retained: {int(zero_wait.sum()):,}")
print(f"  Total dropped (a row can meet more than one condition): {int(drop_mask_split.sum()):,} of {len(ct):,}")

ct_split = ct[~drop_mask_split].reset_index(drop=True)
print(f"  ct_split (final individual CT orders): {len(ct_split):,}")
print(f"  Site split: {ct_split['Site'].value_counts().to_dict()}")

print("Split pipeline -- Part 1: exam duration distribution per CT type...")
ct_split["exam_duration_min"] = (ct_split["exam_end_dt"] - ct_split["exam_start_dt"]).dt.total_seconds() / 60
ct_split["wait_to_start_min"] = (ct_split["exam_start_dt"] - ct_split["parent_order_dt"]).dt.total_seconds() / 60

duration_rows = []
for (site, desc), grp in ct_split.groupby(["Site", "Order Description"]):
    duration_rows.append({"Site": site, "Order Description": desc, **summarize_dist(grp["exam_duration_min"])})
duration_by_type_split = pd.DataFrame(duration_rows).sort_values(["Site", "N"], ascending=[True, False]).reset_index(drop=True)

print("Split pipeline -- Part 2: capped scan duration per CT type, per site queue...")
n_ties_split = {}
site_splits = {}
for site in ["NNI", "DR"]:
    sdf = with_scan_duration(ct_split[ct_split["Site"] == site])
    site_splits[site] = sdf
    n_ties_split[site] = int((sdf["time_to_next_start_min"] == 0).sum())
    print(f"  {site}: {len(sdf):,} orders, {n_ties_split[site]:,} exact-tie (0-minute) next-start gaps")

nni_scan_time_split = scan_time_by_type(site_splits["NNI"])
dr_scan_time_split = scan_time_by_type(site_splits["DR"])
ct_split_seq = pd.concat([site_splits["NNI"], site_splits["DR"]], ignore_index=True)


# =============================================================================
# MERGED pipeline: merged/ct_orders_cleaned.xlsx (post-merge rows -- Site
# already assigned, Exam Start/End already collapsed to one min-start/
# max-end value per merged session, no fallback/validity re-filtering
# needed since clean_ct_orders.py already applied those before merging)
# =============================================================================
def parse_date_only_pd(date_val):
    if isinstance(date_val, pd.Timestamp):
        return date_val
    s = str(date_val)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        if pd.notna(d):
            return d
    return pd.to_datetime(date_val, errors="coerce", dayfirst=True)


def combine_dt_pd(date_val, time_val):
    """Combine a single (non-joined) date value + time value into a
    Timestamp -- Exam Start/End on a merged row is always a single value
    (min-start/max-end), never '; '-joined, so no split_joined needed here."""
    if is_blank(date_val) or is_blank(time_val):
        return None
    d = parse_date_only_pd(date_val)
    if pd.isna(d):
        return None
    dt = pd.to_datetime(f"{d.strftime('%Y-%m-%d')} {time_val}", errors="coerce")
    return None if pd.isna(dt) else dt


print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
n_loaded_merged = len(img)
print(f"  Loaded {n_loaded_merged:,} imaging rows")

ct_merged = img[img["Site"].notna()].copy()
print(f"  CT rows (Site already assigned): {len(ct_merged):,} of {len(img):,}")
print(f"  Site split: {ct_merged['Site'].value_counts().to_dict()}")

ct_merged["exam_start_dt"] = ct_merged.apply(lambda r: combine_dt_pd(r["Exam Start Date"], r["Exam Start Time"]), axis=1)
ct_merged["exam_end_dt"] = ct_merged.apply(lambda r: combine_dt_pd(r["Exam End Date"], r["Exam End Time"]), axis=1)
ct_merged["wait_to_start_min"] = pd.to_numeric(ct_merged.get("Wait Time (min)"), errors="coerce")
negative_wait_m = ct_merged["wait_to_start_min"].notna() & (ct_merged["wait_to_start_min"] < 0)
zero_wait_m = ct_merged["wait_to_start_min"].notna() & (ct_merged["wait_to_start_min"] == 0)
if negative_wait_m.any():
    print(f"  Dropping {int(negative_wait_m.sum()):,} merged CT rows with negative wait time")
    ct_merged = ct_merged[~negative_wait_m].copy()
print(f"  Zero wait time retained, merged CT rows: {int(zero_wait_m.sum()):,}")

print("Merged pipeline -- Part 1: exam duration distribution per CT type...")
ct_merged["exam_duration_min"] = (ct_merged["exam_end_dt"] - ct_merged["exam_start_dt"]).dt.total_seconds() / 60
non_positive_duration_m = ct_merged["exam_duration_min"].notna() & (ct_merged["exam_duration_min"] <= 0)
if non_positive_duration_m.any():
    print(f"  Dropping {int(non_positive_duration_m.sum()):,} merged CT rows with non-positive exam time windows")
    ct_merged = ct_merged[~non_positive_duration_m].copy()

duration_rows_m = []
for (site, desc), grp in ct_merged.groupby(["Site", "Order Description"]):
    duration_rows_m.append({"Site": site, "Order Description": desc, **summarize_dist(grp["exam_duration_min"])})
duration_by_type_merged = pd.DataFrame(duration_rows_m).sort_values(["Site", "N"], ascending=[True, False]).reset_index(drop=True)

print("Merged pipeline -- Part 2: capped scan duration per CT type, per site queue...")
n_ties_merged = {}
site_splits_m = {}
for site in ["NNI", "DR"]:
    sdf = with_scan_duration(ct_merged[ct_merged["Site"] == site])
    site_splits_m[site] = sdf
    n_ties_merged[site] = int((sdf["time_to_next_start_min"] == 0).sum())
    print(f"  {site}: {len(sdf):,} orders, {n_ties_merged[site]:,} exact-tie (0-minute) next-start gaps")

nni_scan_time_merged = scan_time_by_type(site_splits_m["NNI"])
dr_scan_time_merged = scan_time_by_type(site_splits_m["DR"])
ct_merged_seq = pd.concat([site_splits_m["NNI"], site_splits_m["DR"]], ignore_index=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
summary_rows = [
    {"Metric": "-- Split pipeline (ct_orders_combined.xlsx, one row per original order) --", "Value": ""},
    {"Metric": "Imaging rows loaded", "Value": n_loaded_split},
    {"Metric": "CT rows (Site assigned via brainbodysep.tsv)", "Value": len(ct)},
    {"Metric": "  Dropped: no Parent Order time", "Value": int(no_parent_order.sum())},
    {"Metric": "  Dropped: no Exam Start", "Value": int(no_exam_start.sum())},
    {"Metric": "  Dropped: non-positive exam time window", "Value": int(non_positive_duration.sum())},
    {"Metric": "  Dropped: negative wait time", "Value": int(negative_wait.sum())},
    {"Metric": "  Retained: zero wait time", "Value": int(zero_wait.sum())},
    {"Metric": "  Dropped total (a row can meet more than one condition)", "Value": int(drop_mask_split.sum())},
    {"Metric": "Final individual CT orders (ct_split)", "Value": len(ct_split)},
]
for site in ["NNI", "DR"]:
    summary_rows.append({"Metric": f"{site} orders", "Value": int((ct_split["Site"] == site).sum())})
    summary_rows.append({"Metric": f"{site} exact-tie (0-minute) next-start gaps", "Value": n_ties_split[site]})

summary_rows.append({"Metric": "-- Merged pipeline (ct_orders_cleaned.xlsx, post-merge rows) --", "Value": ""})
summary_rows.append({"Metric": "Imaging rows loaded", "Value": n_loaded_merged})
summary_rows.append({"Metric": "CT rows (Site already assigned)", "Value": len(ct_merged)})
summary_rows.append({"Metric": "  Retained: zero wait time", "Value": int((ct_merged["wait_to_start_min"] == 0).sum())})
for site in ["NNI", "DR"]:
    summary_rows.append({"Metric": f"{site} orders", "Value": int((ct_merged["Site"] == site).sum())})
    summary_rows.append({"Metric": f"{site} exact-tie (0-minute) next-start gaps", "Value": n_ties_merged[site]})
summary_out = pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Plot: merged capped scan-duration distribution by site
# ---------------------------------------------------------------------------
plot_df = ct_merged_seq[["Site", "scan_duration_min"]].dropna().copy()
plot_df = plot_df[plot_df["scan_duration_min"] >= 0]

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, ax = plt.subplots(figsize=(6, 5.5))
sns.boxplot(
    data=plot_df, x="Site", y="scan_duration_min", order=["NNI", "DR"],
    hue="Site", palette=SITE_COLORS, legend=False, width=0.5, fliersize=2, ax=ax,
)
for i, site in enumerate(["NNI", "DR"]):
    med = plot_df.loc[plot_df["Site"] == site, "scan_duration_min"].median()
    ax.annotate(f"median {med:.1f} min", (i, med), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color="#52514e")
ax.set_xlabel("")
ax.set_ylabel("Scan duration (mins)")
ax.set_title("Capped CT scan duration by site", fontsize=12, loc="left")
fig.tight_layout()
plot_path = os.path.join(OUT, "scan_duration_boxplot_nni_dr.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {plot_path}")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
out_path = os.path.join(OUT, "ct_scan_time_by_type.xlsx")
print(f"Writing {out_path} ...")

split_out = ct_split_seq[[
    "CSN", "Order ID", "Order Description", "Site",
    "parent_order_dt", "exam_start_dt", "exam_end_dt", "Exam Time Source",
    "wait_to_start_min", "exam_duration_min", "time_to_next_start_min", "scan_duration_min",
]].sort_values(["Site", "exam_start_dt"]).reset_index(drop=True)

merged_out = ct_merged_seq[[
    "CSN", "Order ID", "Order Description", "Site", "N Merged Rows", "Merged Order IDs",
    "Exam Time Source", "exam_start_dt", "exam_end_dt",
    "wait_to_start_min", "exam_duration_min", "time_to_next_start_min", "scan_duration_min",
]].sort_values(["Site", "exam_start_dt"]).reset_index(drop=True)

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    split_out.to_excel(writer, sheet_name="Split_CT_Orders", index=False)
    merged_out.to_excel(writer, sheet_name="Merged_CT_Orders", index=False)
    duration_by_type_split.to_excel(writer, sheet_name="Duration_By_Type_Split", index=False)
    duration_by_type_merged.to_excel(writer, sheet_name="Duration_By_Type_Merged", index=False)
    nni_scan_time_split.to_excel(writer, sheet_name="NNI_ScanTime_By_Type_Split", index=False)
    dr_scan_time_split.to_excel(writer, sheet_name="DR_ScanTime_By_Type_Split", index=False)
    nni_scan_time_merged.to_excel(writer, sheet_name="NNI_ScanTime_By_Type_Merged", index=False)
    dr_scan_time_merged.to_excel(writer, sheet_name="DR_ScanTime_By_Type_Merged", index=False)
    summary_out.to_excel(writer, sheet_name="Summary", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
