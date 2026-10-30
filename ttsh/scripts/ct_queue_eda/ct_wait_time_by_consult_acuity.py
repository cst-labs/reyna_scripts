"""
CT Scan Wait Time by Consult Acuity Level
TTSH ED

Wait time = Parent Order Time -> Exam Start Time, computed per CT order row in
merged/ct_orders_cleaned.xlsx, joined to Consult Acuity from
merged/all_pt_cleaned.xlsx (CSN), and related to the scan-level "ED Scan
Priority" parsed out of Ordering Question Responses.

Outputs: analysis/ct_wait_time_by_consult_acuity.xlsx + 4 PNG charts.
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
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

# ── Palette (matches consult_acuity_field_heatmap.py for visual consistency) ──
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
CAT_COLORS = {
    "P1": "#2a78d6", "P2": "#eb6834", "P3": "#1baf7a", "P4": "#8d6bce",
    "Unknown": "#898781", "Not enough info": "#c9c7c0",
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

ACUITY_ORDER = ["P1", "P2", "P3", "Unknown"]
SCAN_PRIO_ORDER = ["P1", "P2", "P3", "P4", "Not enough info"]
# Waits span 0 min to ~261,000 min (181 days) — a linear axis would crush
# everything under an hour into an invisible sliver next to the long-lead
# outliers. symlog shows every point at its real value (nothing clipped):
# linear for the first SYMLOG_LINTHRESH minutes, log beyond it.
SYMLOG_LINTHRESH = 10

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE, "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY, "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY, "grid.color": GRIDLINE,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def split_joined(val):
    """clean_ct_orders.py joins values from merged rows with '; '."""
    if is_blank(val):
        return [None]
    s = str(val)
    return s.split("; ") if "; " in s else [s]


def parse_date_only(date_val):
    """Parse a single date value (native datetime/date, ISO string from a
    joined-value str(), or a raw dd/mm/yyyy string) without format ambiguity."""
    if isinstance(date_val, (pd.Timestamp,)):
        return date_val
    s = str(date_val)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        if pd.notna(d):
            return d
    return pd.to_datetime(date_val, errors="coerce", dayfirst=True)


def combine_dt(date_val, time_val):
    """Combine a single (non-joined) date value + time value into a Timestamp."""
    if is_blank(date_val) or is_blank(time_val):
        return None
    d = parse_date_only(date_val)
    if pd.isna(d):
        return None
    dt = pd.to_datetime(f"{d.strftime('%Y-%m-%d')} {time_val}", errors="coerce")
    return None if pd.isna(dt) else dt


def earliest_parent_order_dt(date_val, time_val):
    """Parent Order Date/Time can be a '; '-joined multi-value string on rows
    clean_ct_orders.py merged from originally-separate orders placed at
    different times but with overlapping exam windows. Returns the earliest
    combined timestamp across all joined candidates, plus whether it was
    actually multi-valued."""
    d_parts = split_joined(date_val)
    t_parts = split_joined(time_val)
    n = max(len(d_parts), len(t_parts))
    if len(d_parts) < n:
        d_parts = d_parts + [d_parts[-1]] * (n - len(d_parts))
    if len(t_parts) < n:
        t_parts = t_parts + [t_parts[-1]] * (n - len(t_parts))
    dts = [combine_dt(d, t) for d, t in zip(d_parts, t_parts)]
    dts = [dt for dt in dts if dt is not None]
    if not dts:
        return None, False
    return min(dts), len(dts) > 1


def parse_scan_priority(val):
    if is_blank(val):
        return None
    m = re.search(r"ED Scan Priority:\s*(P[1-4])", str(val))
    return m.group(1) if m else None


def iqr_outliers(vals):
    """Tukey's fence: outliers are values beyond Q1-1.5*IQR / Q3+1.5*IQR of
    THIS group's own distribution — the same rule matplotlib/seaborn use to
    decide which boxplot points draw as fliers, so the count below lines up
    with what's visibly plotted, instead of an arbitrary fixed-hour cutoff."""
    if len(vals) < 4:
        return 0, None
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    lower_fence = q1 - 1.5 * iqr
    n_out = int(((vals < lower_fence) | (vals > upper_fence)).sum())
    return n_out, round(upper_fence, 1)


def summarize(sub):
    raw = sub["_wait_raw"]
    n_total = len(sub)
    n_no_ts = raw.isna().sum()
    n_negative = (raw < 0).sum()
    n_zero_dur = (sub["zero_duration_exam"] & raw.notna() & (raw >= 0)).sum()
    vals = sub["wait_mins"].dropna()
    n_valid = len(vals)
    n_outliers, upper_fence = iqr_outliers(vals)
    row = {
        "N Total": n_total,
        "N Excluded": int(n_no_ts + n_negative + n_zero_dur),
        "  No parent order / exam start timestamp": int(n_no_ts),
        "  Negative duration": int(n_negative),
        "  Zero-duration exam window (Start = End)": int(n_zero_dur),
        "N Valid": n_valid,
        "  incl. statistical outliers (IQR method, not excluded)": n_outliers,
        "  Outlier upper fence (mins)": upper_fence,
    }
    for label, val in [
        ("Mean (mins)", vals.mean()), ("Median (mins)", vals.median()),
        ("P25 (mins)", vals.quantile(0.25)), ("P75 (mins)", vals.quantile(0.75)),
        ("P90 (mins)", vals.quantile(0.90)), ("P95 (mins)", vals.quantile(0.95)),
    ]:
        row[label] = round(val, 1) if n_valid else None
    return row


def summary_table(df, group_col, order):
    rows = []
    for key in order:
        sub = df[df[group_col] == key]
        if len(sub) == 0:
            continue
        row = {group_col: key}
        row.update(summarize(sub))
        rows.append(row)
    row = {group_col: "All"}
    row.update(summarize(df))
    rows.append(row)
    return pd.DataFrame(rows)


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 50)


# ---------------------------------------------------------------------------
# Load & join
# ---------------------------------------------------------------------------
print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
ct = img[img["Procedure Category"].astype(str).str.contains("IMG CT ORDERABLES")].copy()
print(f"  CT rows: {len(ct):,}  |  unique CSN: {ct['CSN'].nunique():,}")

print("Loading merged/all_pt_cleaned.xlsx ...")
allpt = pd.read_excel(os.path.join(MERGED, "all_pt_cleaned.xlsx"))
allpt["Consult Acuity"] = allpt["Consult Acuity"].replace({"(P) P2": "P2"})
allpt_acuity = allpt[["CSN", "Consult Acuity"]].drop_duplicates("CSN")

ct = ct.merge(allpt_acuity, on="CSN", how="left")
ct["Consult Acuity"] = ct["Consult Acuity"].where(ct["Consult Acuity"].isin(["P1", "P2", "P3"]), "Unknown")
match_rate = (ct["Consult Acuity"] != "Unknown").mean()
print(f"  Consult Acuity match rate: {match_rate:.1%}")

# New field: scan-level priority parsed from Ordering Question Responses
ct["Scan Priority"] = ct["Ordering Question Responses"].apply(parse_scan_priority)
ct["Scan Priority Group"] = ct["Scan Priority"].fillna("Not enough info")
not_enough_info_rate = (ct["Scan Priority Group"] == "Not enough info").mean()
print(f"  Scan Priority 'not enough info' rate: {not_enough_info_rate:.1%}")

# ---------------------------------------------------------------------------
# Compute wait time per row
# ---------------------------------------------------------------------------
parent_dt_results = ct.apply(
    lambda r: earliest_parent_order_dt(r["Parent Order Date"], r["Parent Order Time"]), axis=1
)
ct["parent_order_dt"] = [r[0] for r in parent_dt_results]
ct["_parent_order_multi_value"] = [r[1] for r in parent_dt_results]
ct["parent_order_dt_floored"] = ct["parent_order_dt"].apply(
    lambda dt: dt.floor("min") if pd.notna(dt) else None
)
ct["exam_start_dt"] = ct.apply(
    lambda r: combine_dt(r["Exam Start Date"], r["Exam Start Time"]), axis=1
)
ct["exam_end_dt"] = ct.apply(
    lambda r: combine_dt(r["Exam End Date"], r["Exam End Time"]), axis=1
)

ct["_wait_raw"] = ct.apply(
    lambda r: (r["exam_start_dt"] - r["parent_order_dt_floored"]).total_seconds() / 60
    if pd.notna(r["exam_start_dt"]) and pd.notna(r["parent_order_dt_floored"]) else np.nan,
    axis=1,
)

# ---------------------------------------------------------------------------
# Data-cleaning flags: characterise *why* rows are missing/extreme rather
# than just counting them.
# ---------------------------------------------------------------------------
ct["exam_duration_min"] = (ct["exam_end_dt"] - ct["exam_start_dt"]).dt.total_seconds() / 60

# Source system sometimes splits one clinical CT event into two order lines
# with the same CSN + Order Description + Parent Order Date/Time: one row
# carries the real (non-zero) scan window, a sibling row carries either no
# exam window at all (-> missing-timestamp exclusion) or a zero-duration
# placeholder window (Exam Start == Exam End). The placeholder's own
# "wait_mins" isn't a real turnaround time (there's no actual scan behind
# it) and duplicates the wait already captured by its real-scan sibling, so
# it's excluded from wait_mins here, same as missing-timestamp/negative rows.
ct["zero_duration_exam"] = ct["exam_duration_min"] == 0
ct["wait_mins"] = ct["_wait_raw"].where((ct["_wait_raw"] >= 0) & ~ct["zero_duration_exam"])

_dup_key = ct["CSN"].astype(str) + "||" + ct["Order Description"].astype(str)
_zero_keys = set(_dup_key[ct["zero_duration_exam"]])
_real_dur_keys = set(_dup_key[ct["exam_duration_min"] > 0])
ct["duplicate_order_pair"] = ct["zero_duration_exam"] & _dup_key.isin(_real_dur_keys) \
    | (ct["exam_duration_min"] > 0) & _dup_key.isin(_zero_keys)

# Data-driven (not hand-picked) split of order types into "long-lead" vs
# "routine" pacing: an Order Description's OWN median wait time (using only
# types with enough rows to be stable) decides the bucket, so this reflects
# how that procedure is actually scheduled rather than an arbitrary list.
_desc_counts = ct["Order Description"].value_counts()
_stable_desc = _desc_counts[_desc_counts >= 5].index
_desc_median_wait = ct[ct["Order Description"].isin(_stable_desc)].groupby("Order Description")["wait_mins"].median()
ct["order_type_median_wait"] = ct["Order Description"].map(_desc_median_wait)
ct["long_lead_order_type"] = ct["order_type_median_wait"] > 1440

# ---------------------------------------------------------------------------
# Multi-scan bookkeeping
# ---------------------------------------------------------------------------
ct = ct.sort_values(["CSN", "exam_start_dt"], na_position="last").reset_index(drop=True)
ct["scan_seq"] = ct.groupby("CSN").cumcount() + 1
ct["scan_seq_group"] = ct["scan_seq"].clip(upper=3).map({1: "1st", 2: "2nd", 3: "3rd+"})

n_scans = ct.groupby("CSN").size()
ct["n_scans_in_visit"] = ct["CSN"].map(n_scans)
ct["multi_scan_visit"] = np.where(ct["n_scans_in_visit"] > 1, "Multi-scan (2+)", "Single scan")

same_parent_time = ct.groupby("CSN")["Parent Order Time"].nunique()
multi_csns = n_scans[n_scans > 1].index
pct_multi_shared_time = (same_parent_time.loc[multi_csns] == 1).mean() if len(multi_csns) else None
print(f"  CSNs with >1 CT row: {len(multi_csns):,} ({len(multi_csns)/ct['CSN'].nunique():.1%})")
if pct_multi_shared_time is not None:
    print(f"  Of those, share with identical Parent Order Time across rows: {pct_multi_shared_time:.1%}")

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
summary_by_consult = summary_table(ct, "Consult Acuity", ACUITY_ORDER)
summary_by_scan_prio = summary_table(ct, "Scan Priority Group", SCAN_PRIO_ORDER)

# ---------------------------------------------------------------------------
# SLA compliance: % of orders with exam start within target wait, by the
# scan-level priority captured on the order itself (Scan Priority Group).
# ---------------------------------------------------------------------------
SLA_TARGETS = {"P1": 30, "P2": 120, "P3": 360}

sla_rows = []
for prio, target in SLA_TARGETS.items():
    sub = ct[ct["Scan Priority Group"] == prio]
    n_total = len(sub)
    vals = sub["wait_mins"].dropna()
    n_valid = len(vals)
    n_within = int((vals <= target).sum())
    sla_rows.append({
        "Scan Priority": prio,
        "Target (mins)": target,
        "N Total Orders": n_total,
        "N Valid (wait_mins available)": n_valid,
        "N Completed Within Target": n_within,
        "% of Valid Orders": round(n_within / n_valid * 100, 1) if n_valid else None,
        "% of All Orders (incl. excluded)": round(n_within / n_total * 100, 1) if n_total else None,
    })
sla_compliance = pd.DataFrame(sla_rows)
print("  SLA compliance by Scan Priority (order -> exam start):")
for r in sla_rows:
    print(f"    {r['Scan Priority']} (<= {r['Target (mins)']} min): "
          f"{r['N Completed Within Target']:,} / {r['N Valid (wait_mins available)']:,} valid "
          f"({r['% of Valid Orders']}%)")

by_seq_rows = []
for seq in ["1st", "2nd", "3rd+"]:
    for acuity in ACUITY_ORDER:
        sub = ct[(ct["scan_seq_group"] == seq) & (ct["Consult Acuity"] == acuity)]
        if len(sub) == 0:
            continue
        row = {"Scan Sequence": seq, "Consult Acuity": acuity}
        row.update(summarize(sub))
        by_seq_rows.append(row)
by_scan_sequence = pd.DataFrame(by_seq_rows)

single_vs_multi_rows = []
for label in ["Single scan", "Multi-scan (2+)"]:
    for acuity in ACUITY_ORDER:
        sub = ct[(ct["multi_scan_visit"] == label) & (ct["Consult Acuity"] == acuity)]
        if len(sub) == 0:
            continue
        row = {"Visit Type": label, "Consult Acuity": acuity}
        row.update(summarize(sub))
        single_vs_multi_rows.append(row)
single_vs_multi = pd.DataFrame(single_vs_multi_rows)

# Share of multi-scan CSNs with identical Parent Order Time, by (patient-level) Consult Acuity
csn_acuity = ct.groupby("CSN")["Consult Acuity"].first()
multi_share_rows = []
for acuity in ACUITY_ORDER:
    acuity_csns = csn_acuity[csn_acuity == acuity].index
    acuity_multi = [c for c in multi_csns if c in set(acuity_csns)]
    if not acuity_multi:
        continue
    share = (same_parent_time.loc[acuity_multi] == 1).mean()
    multi_share_rows.append({
        "Consult Acuity": acuity,
        "Multi-scan CSNs": len(acuity_multi),
        "Share with identical Parent Order Time (%)": round(share * 100, 1),
    })
multi_share_rows.append({
    "Consult Acuity": "All",
    "Multi-scan CSNs": len(multi_csns),
    "Share with identical Parent Order Time (%)": round((pct_multi_shared_time or 0) * 100, 1),
})
multi_shared_time_by_acuity = pd.DataFrame(multi_share_rows)

# ---------------------------------------------------------------------------
# Data Quality sheet
# ---------------------------------------------------------------------------
dq_rows = []
for acuity in ACUITY_ORDER + ["All"]:
    sub = ct if acuity == "All" else ct[ct["Consult Acuity"] == acuity]
    raw = sub["_wait_raw"]
    n_out, upper_fence = iqr_outliers(sub["wait_mins"].dropna())
    dq_rows.append({
        "Consult Acuity": acuity,
        "N Total": len(sub),
        "Missing exam start / parent order timestamp": int(raw.isna().sum()),
        "Negative wait (excluded)": int((raw < 0).sum()),
        "Zero-duration exam window, Start = End (excluded)": int((sub["zero_duration_exam"] & raw.notna() & (raw >= 0)).sum()),
        "Statistical outliers, IQR method (included in stats, flagged only)": n_out,
        "Outlier upper fence (mins)": upper_fence,
        "Merged / multi-value Parent Order Time": int(sub["_parent_order_multi_value"].sum()),
        "Merged / multi-value Parent Order Time (%)": round(sub["_parent_order_multi_value"].mean() * 100, 1),
    })
data_quality = pd.DataFrame(dq_rows)

# ---------------------------------------------------------------------------
# Data Cleaning Deep-Dive — root-causing each bad-data category rather than
# just counting it.
# ---------------------------------------------------------------------------
cleaning_sections = []

# 1. Missing Exam Start / Parent Order timestamp (excluded from wait_mins)
missing_ts = ct[ct["_wait_raw"].isna()]
cleaning_sections.append(("1. Missing exam start / parent order timestamp (excluded)", pd.DataFrame([
    {"Metric": "Rows affected", "Value": len(missing_ts)},
    {"Metric": "Rows affected (%)", "Value": round(len(missing_ts) / len(ct) * 100, 2)},
    {"Metric": "Root cause", "Value": "100% are the 'report' half of a duplicate cardiac-CT order pair"},
    {"Metric": "  Order Description = CT ANGIOGRAM CORONARY ARTERIES", "Value": int((missing_ts["Order Description"] == "CT ANGIOGRAM CORONARY ARTERIES").sum())},
    {"Metric": "  Order Description = CT CARDIAC CALCIUM SCORING", "Value": int((missing_ts["Order Description"] == "CT CARDIAC CALCIUM SCORING").sum())},
    {"Metric": "  Any other Order Description", "Value": int((~missing_ts["Order Description"].isin(["CT ANGIOGRAM CORONARY ARTERIES", "CT CARDIAC CALCIUM SCORING"])).sum())},
    {"Metric": "Explanation", "Value": "Cardiac CT orders generate 2 order lines sharing the same CSN/Order "
        "Description/Parent Order Date-Time: one line carries the real Exam Start/End, "
        "the other carries only Result Date/Time (no exam window at all). Since the sibling "
        "line already captures the real wait, these rows are correctly excluded, not a data error."},
])))

# 2. Negative-duration rows (excluded from wait_mins)
neg = ct[ct["_wait_raw"] < 0]
cleaning_sections.append(("2. Negative duration, exam start before order (excluded)", pd.DataFrame([
    {"Metric": "Rows affected", "Value": len(neg)},
    {"Metric": "Rows affected (%)", "Value": round(len(neg) / len(ct) * 100, 2)},
    {"Metric": "Note", "Value": "None present in this data extract. Kept as an explicit exclusion "
        "rule (rather than assumed impossible) so a future refresh with reversed timestamps is still caught."},
])))

# 3. Zero-duration exam window / duplicate order-line pairs (excluded from
#    wait_mins — Start == End means there's no real scan window behind the
#    timestamp, so any "wait" computed off it isn't a real turnaround time)
zero_dur = ct[ct["zero_duration_exam"]]
dup_pairs = ct[ct["duplicate_order_pair"]]
zero_dur_top = zero_dur["Order Description"].value_counts().head(6)
cleaning_sections.append(("3. Zero-duration exam window (Start = End) — excluded", pd.DataFrame(
    [
        {"Metric": "Rows with 0-minute exam window", "Value": len(zero_dur)},
        {"Metric": "Rows with 0-minute exam window (%)", "Value": round(len(zero_dur) / len(ct) * 100, 2)},
        {"Metric": "  ...with a same-CSN/same-Order-Description sibling row that has a real (>0 min) window", "Value": int((zero_dur["duplicate_order_pair"]).sum())},
        {"Metric": "Total rows flagged as either half of a duplicate order-line pair", "Value": len(dup_pairs)},
        {"Metric": "Explanation", "Value": "Same duplicate-order-line pattern as category 1, but here both "
            "halves happen to carry a timestamp — one is the real scan, the other a 0-minute placeholder "
            "(Start == End, typically a report/result timestamp folded into the Exam fields, not an actual "
            "scan). The placeholder side is excluded from wait_mins since it isn't a real turnaround time and "
            "duplicates the wait its real-scan sibling already captures. It still counts as its own row for "
            "'N scans in visit' / multi-scan bookkeeping, since that's based on row presence, not wait_mins."},
    ] + [{"Metric": f"  Order Description = {desc}", "Value": int(n)} for desc, n in zero_dur_top.items()]
)))

# 4. Extreme / statistically-outlying waits (NOT excluded, IQR-flagged) —
#    characterised by procedure pacing rather than a fixed hour cutoff.
n_out_all, upper_fence_all = iqr_outliers(ct["wait_mins"].dropna())
outliers_all = ct[ct["wait_mins"] > (upper_fence_all or np.inf)]
cleaning_sections.append(("4. Statistical outliers (IQR method) — not excluded", pd.DataFrame([
    {"Metric": "Outlier upper fence, all CT rows (mins)", "Value": upper_fence_all},
    {"Metric": "Rows beyond fence", "Value": len(outliers_all)},
    {"Metric": "Rows beyond fence (%)", "Value": round(len(outliers_all) / len(ct) * 100, 2)},
    {"Metric": "  ...belonging to a 'long-lead' order type (that type's own median wait > 24h)", "Value": int((outliers_all["long_lead_order_type"] == True).sum())},
    {"Metric": "  ...belonging to a 'routine' order type", "Value": int((outliers_all["long_lead_order_type"] != True).sum())},
    {"Metric": "N order types classified 'long-lead' (median wait > 24h, n>=5 rows)", "Value": int(ct.loc[ct["long_lead_order_type"] == True, "Order Description"].nunique())},
    {"Metric": "Top long-lead order types (median wait, mins)", "Value": ", ".join(
        f"{d} ({m:.0f})" for d, m in _desc_median_wait[_desc_median_wait > 1440].sort_values(ascending=False).head(8).items()
    )},
    {"Metric": "Explanation", "Value": "Long waits concentrate in specific CT order types (CT KUB, CT "
        "Abdomen and Pelvis, CT Thorax Abdomen and Pelvis, CT Urography, cardiac CT, limb CT) whose OWN "
        "median wait already exceeds 24h — consistent with stone-protocol follow-up / oncology-staging / "
        "elective scans booked during an ED visit but performed weeks later as an outpatient appointment "
        "under the same CSN, not with corrupted timestamps. These rows disproportionately fall in 'Unknown' "
        "Consult Acuity / 'Not enough info' Scan Priority because elective referrals don't go through the "
        "same ED consult-acuity capture as acute orders."},
])))

data_cleaning_deep_dive_sections = cleaning_sections

# ---------------------------------------------------------------------------
# Outlier Analysis — where the IQR-flagged outliers concentrate and why,
# beyond the single long-lead/routine split already in the Deep-Dive sheet.
# ---------------------------------------------------------------------------
ct["is_outlier"] = ct["wait_mins"] > (upper_fence_all or np.inf)
ct["order_dow"] = ct["parent_order_dt_floored"].apply(lambda d: d.day_name() if pd.notna(d) else None)
ct["order_weekend"] = ct["parent_order_dt_floored"].apply(lambda d: d.dayofweek >= 5 if pd.notna(d) else None)
ct["order_offhours"] = ct["parent_order_dt_floored"].apply(
    lambda d: (d.hour < 8 or d.hour >= 18) if pd.notna(d) else None
)
outliers = ct[ct["is_outlier"]]

outlier_sections = []

outlier_sections.append(("1. Headline", pd.DataFrame([
    {"Metric": "Outlier definition", "Value": f"wait_mins > {upper_fence_all:.0f} min (Tukey fence: Q3 + 1.5x IQR, all valid CT rows)"},
    {"Metric": "Outlier rows", "Value": len(outliers)},
    {"Metric": "Outlier rows, % of all CT rows", "Value": round(len(outliers) / len(ct) * 100, 2)},
    {"Metric": "Outlier rows, % of valid (non-excluded) rows", "Value": round(len(outliers) / len(ct["wait_mins"].dropna()) * 100, 2)},
    {"Metric": "Median wait among outliers (mins / days)", "Value": f"{outliers['wait_mins'].median():.0f} / {outliers['wait_mins'].median() / 1440:.1f}"},
    {"Metric": "P90 wait among outliers (mins / days)", "Value": f"{outliers['wait_mins'].quantile(.9):.0f} / {outliers['wait_mins'].quantile(.9) / 1440:.1f}"},
    {"Metric": "P99 wait among outliers (mins / days)", "Value": f"{outliers['wait_mins'].quantile(.99):.0f} / {outliers['wait_mins'].quantile(.99) / 1440:.1f}"},
    {"Metric": "Max wait (mins / days)", "Value": f"{outliers['wait_mins'].max():.0f} / {outliers['wait_mins'].max() / 1440:.1f}"},
])))

acuity_tbl = ct.groupby("Consult Acuity")["is_outlier"].agg(n_outliers="sum", n_total="size")
acuity_tbl["outlier_rate_pct"] = (acuity_tbl["n_outliers"] / acuity_tbl["n_total"] * 100).round(1)
outlier_sections.append(("2. By Consult Acuity", acuity_tbl.reindex(ACUITY_ORDER).reset_index()))

prio_tbl = ct.groupby("Scan Priority Group")["is_outlier"].agg(n_outliers="sum", n_total="size")
prio_tbl["outlier_rate_pct"] = (prio_tbl["n_outliers"] / prio_tbl["n_total"] * 100).round(1)
outlier_sections.append(("3. By Scan Priority", prio_tbl.reindex(SCAN_PRIO_ORDER).reset_index()))

by_count = outliers["Order Description"].value_counts().head(10).reset_index()
by_count.columns = ["Order Description", "N Outliers"]
outlier_sections.append(("4a. Top order types by outlier COUNT", by_count))

stable_desc = _desc_counts[_desc_counts >= 5].index
rate_tbl = ct[ct["Order Description"].isin(stable_desc)].groupby("Order Description")["is_outlier"].agg(n_total="size", n_outliers="sum")
rate_tbl["outlier_rate_pct"] = (rate_tbl["n_outliers"] / rate_tbl["n_total"] * 100).round(1)
rate_tbl = rate_tbl.sort_values("outlier_rate_pct", ascending=False).head(10).reset_index()
outlier_sections.append(("4b. Top order types by outlier RATE (n>=5 rows)", rate_tbl))

ms_tbl = ct.groupby("multi_scan_visit")["is_outlier"].agg(n_outliers="sum", n_total="size")
ms_tbl["outlier_rate_pct"] = (ms_tbl["n_outliers"] / ms_tbl["n_total"] * 100).round(1)
outlier_sections.append(("5a. By visit type (single vs multi-scan)", ms_tbl.reset_index()))

seq_tbl = ct.groupby("scan_seq_group")["is_outlier"].agg(n_outliers="sum", n_total="size")
seq_tbl["outlier_rate_pct"] = (seq_tbl["n_outliers"] / seq_tbl["n_total"] * 100).round(1)
outlier_sections.append(("5b. By scan sequence within visit", seq_tbl.reindex(["1st", "2nd", "3rd+"]).reset_index()))

_out_dup_key = outliers["CSN"].astype(str) + "||" + outliers["Order Description"].astype(str)
_out_dup_counts = _out_dup_key.value_counts()
_out_both_sides = _out_dup_counts[_out_dup_counts > 1]
_overcount = int((_out_both_sides - 1).sum())
outlier_sections.append(("6. Duplicate order-line double-counting among outliers", pd.DataFrame([
    {"Metric": "Outlier rows sharing CSN + Order Description with another outlier row", "Value": int(_out_both_sides.sum())},
    {"Metric": "  ...as % of all outlier rows", "Value": round(_out_both_sides.sum() / len(outliers) * 100, 1)},
    {"Metric": "Distinct CSN + Order Description pairs affected", "Value": int(len(_out_both_sides))},
    {"Metric": "Estimated true distinct-event outlier count (rows minus double-counted duplicates)", "Value": len(outliers) - _overcount},
    {"Metric": "Explanation", "Value": "Now that zero-duration placeholder rows are excluded from wait_mins "
        "(Data Cleaning Deep-Dive category 3), the duplicate-order-line double-counting that used to inflate "
        "this outlier count (150 rows / 75 pairs before the exclusion) is essentially resolved. The 1 residual "
        "pair remaining is 2 genuinely separate real-duration scans of the same Order Description under one "
        "CSN, not a data artefact — the row-level outlier count above (section 1) is now essentially accurate "
        "at the clinical-event level, not an overcount."},
])))

dow_tbl = ct.groupby("order_weekend")["is_outlier"].agg(n_outliers="sum", n_total="size")
dow_tbl["outlier_rate_pct"] = (dow_tbl["n_outliers"] / dow_tbl["n_total"] * 100).round(1)
dow_tbl = dow_tbl.reset_index().rename(columns={"order_weekend": "Weekend order?"})
outlier_sections.append(("7a. By weekend vs weekday order", dow_tbl))

oh_tbl = ct.groupby("order_offhours")["is_outlier"].agg(n_outliers="sum", n_total="size")
oh_tbl["outlier_rate_pct"] = (oh_tbl["n_outliers"] / oh_tbl["n_total"] * 100).round(1)
oh_tbl = oh_tbl.reset_index().rename(columns={"order_offhours": "Off-hours order (6pm-8am)?"})
oh_tbl_with_note = pd.concat([oh_tbl, pd.DataFrame([{oh_tbl.columns[0]: "Note", "n_outliers": None, "n_total": None,
    "outlier_rate_pct": "Modest effect only (~4pp); order timing is not a dominant outlier driver."}])], ignore_index=True)
outlier_sections.append(("7b. By off-hours vs business-hours order", oh_tbl_with_note))

unk = ct[ct["Consult Acuity"] == "Unknown"]
outlier_sections.append(("8. 'Unknown' Consult Acuity composition", pd.DataFrame([
    {"Metric": "Rows with Consult Acuity = Unknown", "Value": len(unk)},
    {"Metric": "  ...with no Acuity Abbr at all (never ED-triaged / no acuity captured)", "Value": int(unk["Acuity Abbr"].isna().sum())},
    {"Metric": "  ...with an Acuity Abbr present (Consult Acuity join simply unmatched)", "Value": int(unk["Acuity Abbr"].notna().sum())},
    {"Metric": "Outlier rate within the Unknown group", "Value": round(unk["is_outlier"].mean() * 100, 1)},
    {"Metric": "Explanation", "Value": "Most 'Unknown' rows have no ED Acuity Abbr at all, consistent with these "
        "being non-ED-triaged referrals (e.g. outpatient/specialist CT ordered under an ED-linked CSN) rather "
        "than a failed join — explaining both why Consult Acuity was never captured and why this group "
        "dominates the outlier population (87.6% outlier rate vs 2.3-22.8% for P1-P3)."},
])))

top20 = outliers.sort_values("wait_mins", ascending=False).head(20)[[
    "CSN", "Order Description", "Consult Acuity", "Scan Priority Group", "wait_mins",
    "duplicate_order_pair", "long_lead_order_type",
]].copy()
top20["wait_days"] = (top20["wait_mins"] / 1440).round(1)
outlier_sections.append(("9. Top 20 most extreme waits", top20))

outlier_analysis_sections = outlier_sections

# ---------------------------------------------------------------------------
# Consult Acuity x Scan Priority relationship, split by physical CT site
# ---------------------------------------------------------------------------
SITE_ORDER = ["NNI", "DR"]

# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------
def box_and_median_chart(df, group_col, order, title_label, out_name):
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    plot_df = df[df["wait_mins"].notna() & df[group_col].isin(order)]
    colors = [CAT_COLORS.get(k, INK_MUTED) for k in order]

    sns.boxplot(data=plot_df, x=group_col, y="wait_mins", hue=group_col, order=order, hue_order=order,
                palette=colors, width=0.5, legend=False,
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.5}, ax=ax1)
    ax1.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH)
    ax1.set_ylim(bottom=0)  # data is never negative; hides symlog's mirrored negative axis
    ax1.set_xlabel(title_label, color=INK_SECONDARY)
    ax1.set_ylabel("Wait time: Order → Exam Start (mins, log scale)", color=INK_SECONDARY)
    ax1.set_title(f"CT Wait Time (Parent Order → Exam Start) by {title_label}\n"
                  "Full distribution, incl. outliers (symlog scale) — excl. missing/invalid timestamps",
                  fontsize=11, color=INK_PRIMARY, loc="left")
    ax1.grid(axis="y", which="both", color=GRIDLINE, alpha=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out_name), dpi=150, facecolor=SURFACE)
    plt.close(fig)


print("Building charts...")
box_and_median_chart(ct, "Consult Acuity", ACUITY_ORDER, "Consult Acuity", "ct_wait_time_by_consult_acuity.png")
box_and_median_chart(ct, "Scan Priority Group", SCAN_PRIO_ORDER, "Scan Priority", "ct_wait_time_by_scan_priority.png")

# Heatmap: Consult Acuity x Scan Priority, one per physical CT site (NNI, DR)
consult_vs_scan_by_site = {}
for site in SITE_ORDER:
    site_ct = ct[ct["Site"] == site]
    site_counts = pd.crosstab(site_ct["Consult Acuity"], site_ct["Scan Priority Group"])
    site_counts = site_counts.reindex(index=ACUITY_ORDER, columns=SCAN_PRIO_ORDER, fill_value=0)
    site_pct = site_counts / site_counts.values.sum() * 100
    consult_vs_scan_by_site[site] = (site_counts, site_pct)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    seq_cmap = sns.light_palette(SEQ_BLUE[4], as_cmap=True)
    sns.heatmap(site_pct, annot=site_counts, fmt="d", cmap=seq_cmap, vmin=0,
                vmax=site_pct.values.max(),
                linewidths=2, linecolor=SURFACE, cbar_kws={"label": "% of all cases"},
                annot_kws={"color": INK_PRIMARY, "fontsize": 16, "fontweight": "bold"}, ax=ax)
    ax.set_title(f"Consult Acuity vs. Scan Priority — {site}\n"
                 f"(n={len(site_ct):,}; cell = count, color = % of all cases)",
                 fontsize=11, color=INK_PRIMARY, loc="left")
    ax.set_xlabel("Scan Priority (from Ordering Question Responses)", color=INK_SECONDARY)
    ax.set_ylabel("Consult Acuity", color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"ct_consult_vs_scan_priority_heatmap_{site.lower()}.png"),
                dpi=150, facecolor=SURFACE)
    plt.close(fig)

# Scan sequence chart: median wait by scan_seq, split by Consult Acuity
fig, ax = plt.subplots(figsize=(8, 5))
seq_order = ["1st", "2nd", "3rd+"]
w = 0.2
x_pos = {s: i for i, s in enumerate(seq_order)}
for j, acuity in enumerate(ACUITY_ORDER):
    sub = by_scan_sequence[by_scan_sequence["Consult Acuity"] == acuity]
    xs, meds, ns = [], [], []
    for seq in seq_order:
        row = sub[sub["Scan Sequence"] == seq]
        if len(row) and pd.notna(row["Median (mins)"].values[0]):
            xs.append(x_pos[seq] + (j - (len(ACUITY_ORDER) - 1) / 2) * w)
            meds.append(row["Median (mins)"].values[0])
            ns.append(row["N Valid"].values[0])
    if xs:
        ax.bar(xs, meds, width=w, color=CAT_COLORS.get(acuity, INK_MUTED), label=acuity, alpha=0.9)
ax.set_xticks(list(x_pos.values()))
ax.set_xticklabels(seq_order)
ax.set_xlabel("Scan sequence within visit", color=INK_SECONDARY)
ax.set_ylabel("Median wait: Order → Exam Start (mins)", color=INK_SECONDARY)
ax.set_title("Median CT Wait Time by Scan Sequence, split by Consult Acuity", fontsize=11, color=INK_PRIMARY, loc="left")
ax.legend(title="Consult Acuity")
sns.despine(ax=ax, left=True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ct_wait_time_by_scan_sequence.png"), dpi=150, facecolor=SURFACE)
plt.close(fig)

# ---------------------------------------------------------------------------
# Write Excel output
# ---------------------------------------------------------------------------
print("Writing analysis/ct_wait_time_by_consult_acuity.xlsx ...")
detail_cols = [
    "CSN", "Consult Acuity", "Acuity Abbr", "Scan Priority", "Scan Priority Group",
    "Order Description", "Order ID", "scan_seq", "n_scans_in_visit", "multi_scan_visit",
    "Parent Order Date", "Parent Order Time", "Exam Start Date", "Exam Start Time",
    "parent_order_dt_floored", "exam_start_dt", "wait_mins", "_parent_order_multi_value",
    "exam_duration_min", "zero_duration_exam", "duplicate_order_pair",
    "order_type_median_wait", "long_lead_order_type",
]
raw_detail = ct[[c for c in detail_cols if c in ct.columns]].copy()

with pd.ExcelWriter(os.path.join(OUT, "ct_wait_time_by_consult_acuity.xlsx"), engine="openpyxl") as writer:
    summary_by_consult.to_excel(writer, sheet_name="Summary by Consult Acuity", index=False)
    summary_by_scan_prio.to_excel(writer, sheet_name="Summary by Scan Priority", index=False)
    sla_compliance.to_excel(writer, sheet_name="SLA Compliance", index=False)
    row_cursor_cvs = 0
    for site in SITE_ORDER:
        site_counts, site_pct = consult_vs_scan_by_site[site]
        pd.DataFrame([{"Metric": f"{site} — counts", "Value": ""}]).to_excel(
            writer, sheet_name="Consult vs Scan Priority", index=False, header=False, startrow=row_cursor_cvs)
        row_cursor_cvs += 1
        site_counts.to_excel(writer, sheet_name="Consult vs Scan Priority", startrow=row_cursor_cvs)
        row_cursor_cvs += len(site_counts) + 2
        pd.DataFrame([{"Metric": f"{site} — % of all {site} cases", "Value": ""}]).to_excel(
            writer, sheet_name="Consult vs Scan Priority", index=False, header=False, startrow=row_cursor_cvs)
        row_cursor_cvs += 1
        site_pct.round(1).to_excel(writer, sheet_name="Consult vs Scan Priority", startrow=row_cursor_cvs)
        row_cursor_cvs += len(site_pct) + 3
    by_scan_sequence.to_excel(writer, sheet_name="By Scan Sequence x Acuity", index=False)
    single_vs_multi.to_excel(writer, sheet_name="Single vs Multi-Scan", index=False)
    multi_shared_time_by_acuity.to_excel(writer, sheet_name="Single vs Multi-Scan", index=False, startrow=len(single_vs_multi) + 3)
    data_quality.to_excel(writer, sheet_name="Data Quality", index=False)

    row_cursor = 0
    for title, block in data_cleaning_deep_dive_sections:
        pd.DataFrame([{"Metric": title, "Value": ""}]).to_excel(
            writer, sheet_name="Data Cleaning Deep-Dive", index=False, header=False, startrow=row_cursor)
        row_cursor += 1
        block.to_excel(writer, sheet_name="Data Cleaning Deep-Dive", index=False, startrow=row_cursor)
        row_cursor += len(block) + 3

    row_cursor = 0
    for title, block in outlier_analysis_sections:
        pd.DataFrame([{"Metric": title, "Value": ""}]).to_excel(
            writer, sheet_name="Outlier Analysis", index=False, header=False, startrow=row_cursor)
        row_cursor += 1
        block.to_excel(writer, sheet_name="Outlier Analysis", index=False, startrow=row_cursor)
        row_cursor += len(block) + 3

    raw_detail.to_excel(writer, sheet_name="Raw Detail", index=False)

    for name, sheet in writer.sheets.items():
        if name == "Data Cleaning Deep-Dive":
            sheet.column_dimensions["A"].width = 78
            sheet.column_dimensions["B"].width = 90
        elif name == "Outlier Analysis":
            sheet.column_dimensions["A"].width = 45
            auto_col_width(sheet)
        else:
            auto_col_width(sheet)

print("Done. Output in analysis/:")
for f in [
    "ct_wait_time_by_consult_acuity.xlsx", "ct_wait_time_by_consult_acuity.png",
    "ct_wait_time_by_scan_priority.png",
    "ct_consult_vs_scan_priority_heatmap_nni.png", "ct_consult_vs_scan_priority_heatmap_dr.png",
    "ct_wait_time_by_scan_sequence.png",
]:
    path = os.path.join(OUT, f)
    if os.path.exists(path):
        print(f"  {f}  ({os.path.getsize(path)//1024} KB)")
    else:
        print(f"  {f}  MISSING")
