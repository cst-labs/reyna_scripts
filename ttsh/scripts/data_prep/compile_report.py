"""
compile_report.py
Compiles all analysis into a single master_report.xlsx (7 sheets).
Reads from already-built merged/ and analysis/ outputs only.
"""

import os, math
from datetime import datetime, timedelta, date

import openpyxl
import pandas as pd

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
ANA    = os.path.join(BASE, "analysis")
OUT    = os.path.join(ANA, "master_report.xlsx")

MONTH_ORDER = ["Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def csn_set(df, month=None):
    sub = df if month is None else df[df["month"] == month]
    return set(
        sub["CSN"].dropna().apply(str).str.strip()
        .loc[lambda s: s.str.match(r"^\d+$")]
    )


def to_dt(val):
    if val is None: return None
    try:
        if isinstance(val, float) and math.isnan(val): return None
    except: pass
    if isinstance(val, datetime): return val
    if isinstance(val, date): return datetime(val.year, val.month, val.day)
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f): return None
        return datetime(1899, 12, 30) + timedelta(days=f)
    except: return None


def combine_dt(d_val, t_val):
    d = to_dt(d_val)
    if d is None: return None
    if t_val is None: return d
    # t_val may be a time object, datetime, or OLE fraction
    if isinstance(t_val, datetime):
        return datetime(d.year, d.month, d.day, t_val.hour, t_val.minute, t_val.second)
    import datetime as dt_mod
    if isinstance(t_val, dt_mod.time):
        return datetime(d.year, d.month, d.day, t_val.hour, t_val.minute, t_val.second)
    try:
        f = float(t_val)
        if 0 <= f < 2:  # OLE fraction
            total_secs = round(f * 86400)
            h, rem = divmod(total_secs, 3600)
            m, s = divmod(rem, 60)
            return datetime(d.year, d.month, d.day, h % 24, m, s)
    except: pass
    return d


def blank_row(ncols):
    return pd.DataFrame([[""] * ncols], columns=[f"_b{i}" for i in range(ncols)])


def separator(label, ncols):
    row = [""] * ncols
    row[0] = f"--- {label} ---"
    return pd.DataFrame([row], columns=[f"_s{i}" for i in range(ncols)])


def stack(*frames, labels=None):
    """Stack dataframes with blank rows between them, aligning on a shared column superset."""
    all_cols = []
    for df in frames:
        for c in df.columns:
            if c not in all_cols:
                all_cols.append(c)
    pieces = []
    for i, df in enumerate(frames):
        if i > 0:
            pieces.append(pd.DataFrame([{c: "" for c in all_cols}]))
        if labels:
            pieces.append(pd.DataFrame([{all_cols[0]: f"=== {labels[i]} ==="}]))
        pieces.append(df.reindex(columns=all_cols))
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# Load source data
# ---------------------------------------------------------------------------
print("Loading merged files...")
df_ap  = load_sheet(os.path.join(MERGED, "all_pt_combined.xlsx"),    "All Patients")
df_img = load_sheet(os.path.join(MERGED, "ct_orders_combined.xlsx"), "Imaging")
df_lab = load_sheet(os.path.join(MERGED, "ct_orders_combined.xlsx"), "Lab")

print("Loading analysis outputs...")
df_agreement = load_sheet(os.path.join(ANA, "merged_analysis.xlsx"), "Duplicate Field Agreement")
df_chrono    = load_sheet(os.path.join(ANA, "merged_analysis.xlsx"), "Chronology Checks")
df_spillover = load_sheet(os.path.join(ANA, "merged_analysis.xlsx"), "Spillover CSNs")
df_img_nolab = load_sheet(os.path.join(ANA, "merged_analysis.xlsx"), "Imaging — No Lab")
df_img_dist  = load_sheet(os.path.join(ANA, "merged_analysis.xlsx"), "Img — Orders per Patient")
df_lab_dist  = load_sheet(os.path.join(ANA, "merged_analysis.xlsx"), "Lab — Orders per Patient")
df_order_mix = load_sheet(os.path.join(ANA, "merged_analysis.xlsx"), "Img — Order Type Mix")
df_img_rpt   = load_sheet(os.path.join(ANA, "merged_analysis.xlsx"), "Img — Repeated Scan Types")
df_lab_top   = load_sheet(os.path.join(ANA, "merged_analysis.xlsx"), "Lab — Top Tests")

# ---------------------------------------------------------------------------
# Sheet 1 — Overview
# ---------------------------------------------------------------------------
print("Building Sheet 1 — Overview...")

# Table 1: row counts
vol_rows = []
for m in MONTH_ORDER:
    ap_m  = df_ap [df_ap ["month"] == m]
    img_m = df_img[df_img["month"] == m]
    lab_m = df_lab[df_lab["month"] == m]
    vol_rows.append({
        "Month":                m,
        "All Pt Rows":          len(ap_m),
        "All Pt Unique CSNs":   csn_set(ap_m).__len__(),
        "Imaging Rows":         len(img_m),
        "Imaging Unique CSNs":  csn_set(img_m).__len__(),
        "Lab Rows":             len(lab_m),
        "Lab Unique CSNs":      csn_set(lab_m).__len__(),
    })
# Totals row
vol_rows.append({
    "Month":               "ALL MONTHS",
    "All Pt Rows":         len(df_ap),
    "All Pt Unique CSNs":  len(csn_set(df_ap)),
    "Imaging Rows":        len(df_img),
    "Imaging Unique CSNs": len(csn_set(df_img)),
    "Lab Rows":            len(df_lab),
    "Lab Unique CSNs":     len(csn_set(df_lab)),
})
df_vol = pd.DataFrame(vol_rows)

# Table 2: Venn by month
def venn_row(label, ap, im, lb):
    all3        = ap & im & lb
    ap_im_only  = (ap & im) - lb
    ap_lb_only  = (ap & lb) - im
    im_lb_noap  = (im & lb) - ap
    ap_only     = ap - im - lb
    im_only     = im - lb - ap
    lb_only     = lb - ap - im
    union       = ap | im | lb
    return {
        "Month":                               label,
        "All 3 (All Pt + Img + Lab)":          len(all3),
        "All Pt + Img only (no Lab)":          len(ap_im_only),
        "All Pt + Lab only (no Img)":          len(ap_lb_only),
        "Img + Lab only — no All Pt (spillover)": len(im_lb_noap),
        "All Pt only (no orders)":             len(ap_only),
        "Img only (no Lab, no All Pt)":        len(im_only),
        "Lab only (no Img, no All Pt)":        len(lb_only),
        "Total Union":                         len(union),
    }

venn_rows = []
for m in MONTH_ORDER:
    venn_rows.append(venn_row(m, csn_set(df_ap, m), csn_set(df_img, m), csn_set(df_lab, m)))
venn_rows.append(venn_row("ALL MONTHS", csn_set(df_ap), csn_set(df_img), csn_set(df_lab)))
df_venn = pd.DataFrame(venn_rows)

df_sheet1 = stack(df_vol, df_venn, labels=["Row Counts by Month", "Coverage Venn by Month"])

# ---------------------------------------------------------------------------
# Sheet 2 — Coverage Detail
# ---------------------------------------------------------------------------
print("Building Sheet 2 — Coverage Detail...")

# Spillover: tidy up columns
spill_cols = ["CSN", "month", "arrival_date_img", "earliest_order_date", "n_img_orders", "n_lab_orders"]
spill_cols = [c for c in spill_cols if c in df_spillover.columns]
df_spill_out = df_spillover[spill_cols].copy().rename(columns={
    "month": "Month", "arrival_date_img": "Arrival Date (Imaging)",
    "earliest_order_date": "Earliest Order Date",
    "n_img_orders": "N Imaging Orders", "n_lab_orders": "N Lab Orders",
})

# Imaging no lab: tidy up columns
nolab_cols = ["CSN", "month", "acuity", "ed_disposition", "n_img_orders"]
nolab_cols = [c for c in nolab_cols if c in df_img_nolab.columns]
df_nolab_out = df_img_nolab[nolab_cols].copy().rename(columns={
    "month": "Month", "acuity": "Acuity", "ed_disposition": "ED Disposition",
    "n_img_orders": "N Imaging Orders",
})

df_sheet2 = stack(df_spill_out, df_nolab_out,
                  labels=["Spillover CSNs (in CT Orders, not in All Pt)",
                          "Imaging — No Lab (in All Pt + Imaging, no lab order)"])

# ---------------------------------------------------------------------------
# Sheet 3 — Orders per Patient
# ---------------------------------------------------------------------------
print("Building Sheet 3 — Orders per Patient...")

# Imaging distribution (pull from existing, keep key columns)
img_dist_cols = [c for c in df_img_dist.columns if c in [
    "N Orders", "Patient Count", "% of Patients", "Median", "P75", "P90", "P95", "Mean", "Max"]]
df_img_dist_out = df_img_dist[img_dist_cols] if img_dist_cols else df_img_dist

# Lab distribution
lab_dist_cols = [c for c in df_lab_dist.columns if c in [
    "N Orders", "Patient Count", "% of Patients", "Median", "P75", "P90", "P95", "Mean", "Max"]]
df_lab_dist_out = df_lab_dist[lab_dist_cols] if lab_dist_cols else df_lab_dist

# CT vs XR mix by month
mix_month = df_order_mix[df_order_mix["Month"] != "All"].copy() if "Month" in df_order_mix.columns else df_order_mix
mix_pivot = mix_month.pivot_table(
    index="Month", columns="Order Type", values="N Orders", aggfunc="sum", fill_value=0
).reset_index()
mix_pivot.columns.name = None
# Add totals and percentages
for ot in ["CT","XR","MRI","Other","Unknown"]:
    if ot not in mix_pivot.columns:
        mix_pivot[ot] = 0
mix_pivot["Total"] = mix_pivot[["CT","XR","MRI","Other","Unknown"]].sum(axis=1)
mix_pivot["% CT"]  = (mix_pivot["CT"]  / mix_pivot["Total"] * 100).round(1)
mix_pivot["% XR"]  = (mix_pivot["XR"]  / mix_pivot["Total"] * 100).round(1)
# Sort by month order
mix_pivot["_sort"] = mix_pivot["Month"].map({m: i for i, m in enumerate(MONTH_ORDER)})
mix_pivot = mix_pivot.sort_values("_sort").drop(columns=["_sort"])

df_sheet3 = stack(df_img_dist_out, df_lab_dist_out, mix_pivot,
                  labels=["Imaging Orders per Patient",
                          "Lab Orders per Patient",
                          "CT vs XR vs MRI Mix by Month"])

# ---------------------------------------------------------------------------
# Sheet 4 — CT Order Time vs Lab Result Time
# ---------------------------------------------------------------------------
print("Building Sheet 4 — CT Order vs Lab Result Timing...")

def classify_order(desc):
    v = str(desc).upper().strip() if desc else ""
    if v.startswith("NGEMR CT") or (v.startswith("CT ") and "XR" not in v): return "CT"
    if v.startswith("XR ") or v.startswith("NGEMR XR"): return "XR"
    if "MRI" in v: return "MRI"
    return "Other"

df_img["order_type"] = df_img["Order Description"].apply(classify_order)
df_img["CSN_key"]    = df_img["CSN"].apply(str).str.strip()
df_lab["CSN_key"]    = df_lab["CSN"].apply(str).str.strip()

# Per patient: first CT order dt
df_ct_only = df_img[df_img["order_type"] == "CT"].copy()
df_ct_only["order_dt"] = df_ct_only.apply(
    lambda r: combine_dt(r["Order Date"], r["Order Time"]), axis=1)

print("  Computing first CT order per patient...")
ct_first = (df_ct_only.dropna(subset=["order_dt"])
    .groupby("CSN_key")["order_dt"].min()
    .reset_index().rename(columns={"order_dt": "first_ct_order_dt"}))
ct_month = df_ct_only.groupby("CSN_key")["month"].first().reset_index()
ct_first = ct_first.merge(ct_month, on="CSN_key", how="left")

# Per patient: last lab result dt (latest result available at time of CT order)
# and first lab result dt
print("  Computing lab result times per patient...")
df_lab["result_dt"] = df_lab.apply(
    lambda r: combine_dt(r["Result Date"], r["Result Time"]), axis=1)

lab_results = (df_lab.dropna(subset=["result_dt"])
    .groupby("CSN_key")["result_dt"]
    .agg(last_lab_result_dt="max", first_lab_result_dt="min")
    .reset_index())

timing_pt = ct_first.merge(lab_results, on="CSN_key", how="inner")
timing_pt["diff_mins"] = (
    timing_pt["first_ct_order_dt"] - timing_pt["last_lab_result_dt"]
).dt.total_seconds() / 60

def classify_timing(row):
    ct = row["first_ct_order_dt"]
    last_res = row["last_lab_result_dt"]
    if ct is None or last_res is None: return "Unknown"
    diff_m = (ct - last_res).total_seconds() / 60
    if diff_m > 0:   return "CT ordered after last lab result"
    if diff_m == 0:  return "CT ordered same minute as last lab result"
    # CT before last result — check if ANY result was before CT
    first_res = row["first_lab_result_dt"]
    if first_res and ct > first_res:
        return "CT ordered after some (not all) lab results"
    return "CT ordered before any lab result"

timing_pt["category"] = timing_pt.apply(classify_timing, axis=1)

# Summary by month
summary_rows = []
for scope, sub in [("ALL MONTHS", timing_pt)] + [(m, timing_pt[timing_pt["month"] == m]) for m in MONTH_ORDER]:
    total = len(sub)
    if total == 0: continue
    cats = sub["category"].value_counts()
    valid_diff = sub["diff_mins"].dropna()

    # Stats for CT-after-result group only
    after = sub[sub["category"] == "CT ordered after last lab result"]["diff_mins"].dropna()
    some  = sub[sub["category"] == "CT ordered after some (not all) lab results"]["diff_mins"].dropna()

    row = {
        "Month":                                          scope,
        "N Patients (CT + lab)":                         total,
        "CT after ALL lab results":                      cats.get("CT ordered after last lab result", 0),
        "% CT after all":                                round(cats.get("CT ordered after last lab result", 0) / total * 100, 1),
        "CT after SOME lab results":                     cats.get("CT ordered after some (not all) lab results", 0),
        "% CT after some":                               round(cats.get("CT ordered after some (not all) lab results", 0) / total * 100, 1),
        "CT before any lab result":                      cats.get("CT ordered before any lab result", 0),
        "% CT before any":                               round(cats.get("CT ordered before any lab result", 0) / total * 100, 1),
        "CT same minute as last result":                 cats.get("CT ordered same minute as last lab result", 0),
        "Unknown (missing timestamps)":                  cats.get("Unknown", 0),
        "Median mins CT−last result (when CT after all)":  round(after.median(), 0) if len(after) else None,
        "P25 (mins)":                                    round(after.quantile(.25), 0) if len(after) else None,
        "P75 (mins)":                                    round(after.quantile(.75), 0) if len(after) else None,
        "P90 (mins)":                                    round(after.quantile(.90), 0) if len(after) else None,
    }
    summary_rows.append(row)
df_timing_summary = pd.DataFrame(summary_rows)

df_sheet4 = df_timing_summary.copy()

# ---------------------------------------------------------------------------
# Sheet 8 — CT After Lab Result — Per Patient Detail
# ---------------------------------------------------------------------------
print("Building Sheet 8 — CT After Lab Result detail...")

ct_after = timing_pt[
    timing_pt["category"].isin([
        "CT ordered after last lab result",
        "CT ordered after some (not all) lab results",
    ])
].copy()

# Enrich with acuity, disposition from imaging
img_meta = df_ct_only.groupby("CSN_key").agg(
    acuity=("Acuity Abbr", "first"),
    ed_disposition=("ED Disposition", "first"),
    primary_dx=("Primary ED Dx", "first"),
    ct_descriptions=("Order Description", lambda x: "; ".join(sorted(set(x.dropna())))),
).reset_index()

ct_after = ct_after.merge(img_meta, on="CSN_key", how="left")

# For each patient, find the specific lab result(s) that came before the CT order
print("  Finding which lab results preceded the CT order...")
# Build a lookup: CSN → list of (result_dt, test_name, result_text)
lab_before_ct = []
for _, row in ct_after.iterrows():
    csn = row["CSN_key"]
    ct_dt = row["first_ct_order_dt"]
    labs = df_lab[(df_lab["CSN_key"] == csn) & df_lab["result_dt"].notna()]
    before = labs[labs["result_dt"] <= ct_dt].sort_values("result_dt", ascending=False)
    n_before = len(before)
    last_result = before.iloc[0]["result_dt"] if len(before) else None
    last_test   = before.iloc[0]["Order Description"] if len(before) else ""
    last_value  = str(before.iloc[0]["Last Lab Test Results"])[:100] if len(before) else ""
    tests_before = "; ".join(before["Order Description"].dropna().unique()[:5])
    lab_before_ct.append({
        "CSN":                     csn,
        "Month":                   row["month"],
        "Category":                row["category"],
        "First CT Order DT":       row["first_ct_order_dt"],
        "Last Lab Result DT":      row["last_lab_result_dt"],
        "Mins: CT Order − Last Result": round(row["diff_mins"], 0) if pd.notna(row["diff_mins"]) else None,
        "N Lab Results Before CT": n_before,
        "Last Result Before CT — Test":  last_test,
        "Last Result Before CT — Value": last_value,
        "All Tests With Results Before CT": tests_before,
        "Acuity":                  row.get("acuity", ""),
        "ED Disposition":          row.get("ed_disposition", ""),
        "Primary ED Dx":           row.get("primary_dx", ""),
        "CT Descriptions":         row.get("ct_descriptions", ""),
    })

df_sheet8 = pd.DataFrame(lab_before_ct).sort_values(["Month","Mins: CT Order − Last Result"])

# ---------------------------------------------------------------------------
# Sheet 5 — Repeated Orders
# ---------------------------------------------------------------------------
print("Building Sheet 5 — Repeated Orders...")

# Repeated CT scans (from existing)
rpt_cols = ["CSN", "Month", "Order Description", "N Repeats", "Order Dates", "Same Date", "Acuity"]
rpt_cols = [c for c in rpt_cols if c in df_img_rpt.columns]
df_rpt_ct = df_img_rpt[rpt_cols].copy() if rpt_cols else df_img_rpt

# Repeated lab tests summary
if "Test Name" not in df_lab_top.columns and "Order Description" in df_lab_top.columns:
    df_lab_top = df_lab_top.rename(columns={"Order Description": "Test Name"})

# Compute repeated lab tests from raw data
lab_repeats = (
    df_lab.groupby(["CSN_key","Order Description"])
    .size().reset_index(name="n_per_pt")
)
lab_repeats_only = lab_repeats[lab_repeats["n_per_pt"] > 1]
lab_rpt_summary = (
    lab_repeats_only.groupby("Order Description")
    .agg(
        N_Patients_With_Repeat=("CSN_key","nunique"),
        Max_Repeats=("n_per_pt","max"),
    ).reset_index().rename(columns={"Order Description":"Test Name"})
)
# Total patients who had each test (to compute %)
lab_total_by_test = df_lab.groupby("Order Description")["CSN_key"].nunique().reset_index(
    name="N_Total_Patients").rename(columns={"Order Description":"Test Name"})
lab_rpt_summary = lab_rpt_summary.merge(lab_total_by_test, on="Test Name", how="left")
lab_rpt_summary["% Patients with Repeat"] = (
    lab_rpt_summary["N_Patients_With_Repeat"] / lab_rpt_summary["N_Total_Patients"] * 100
).round(1)
lab_rpt_summary = lab_rpt_summary.sort_values("N_Patients_With_Repeat", ascending=False)

df_sheet5 = stack(df_rpt_ct, lab_rpt_summary,
                  labels=["Repeated CT Scan Types (same description >1 for same patient)",
                          "Repeated Lab Tests — Summary"])

# ---------------------------------------------------------------------------
# Sheet 6 — Duplicate Field Agreement
# ---------------------------------------------------------------------------
print("Building Sheet 6 — Duplicate Field Agreement...")

INTERPRETATIONS = {
    "Arrival Date":     "0% — identical in both sources; reliable join key",
    "Primary ED Dx":    "0.1–0.2% minor disagreement — diagnosis is sometimes updated after the CT order is placed; CT Orders captures the dx at time of ordering",
    "ED Dx":            "0.1–0.2% minor disagreement — same reason as Primary ED Dx; final dx may differ from ordering dx",
    "ED Disposition":   "0% — identical; same system field",
    "Pt. Doc Type":     "0.2–0.5% minor disagreement — document type occasionally differs between patient registration and the value recorded at order entry",
    "Specialty":        "0% — identical",
    "Admit IP Room":    "0% — identical",
    "Admit IP Bed":     "0% — identical",
    "Bed Assigned Date":"0% — identical after coercing OLE serial (Apr 2026 All Pt) to date",
    "Bed Assigned Time":"0% after ±1 min tolerance — OLE decimal fraction (All Pt) vs HH:MM:SS string (orders) differ by rounding; values represent the same time",
    "Admit Dispo Date": "0% — identical after OLE serial coercion",
    "Admit Dispo Time": "0% after ±1 min tolerance — same OLE vs string precision issue",
}

df_agree_out = df_agreement.copy()
df_agree_out["Interpretation"] = df_agree_out["Field"].map(INTERPRETATIONS).fillna("")
# Drop internal Notes column if present, tidy column order
keep = ["Field","Source Pair","N Both Non-null","N Agree","N Disagree","Disagree %","Interpretation"]
keep = [c for c in keep if c in df_agree_out.columns]
df_sheet6 = df_agree_out[keep]

# ---------------------------------------------------------------------------
# Sheet 7 — Chronology Checks
# ---------------------------------------------------------------------------
print("Building Sheet 7 — Chronology Checks...")
df_sheet7 = df_chrono.copy()

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
print("Writing master_report.xlsx...")

with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
    df_sheet1.to_excel(writer, sheet_name="1 — Overview",              index=False)
    df_sheet2.to_excel(writer, sheet_name="2 — Coverage Detail",       index=False)
    df_sheet3.to_excel(writer, sheet_name="3 — Orders per Patient",    index=False)
    df_sheet4.to_excel(writer, sheet_name="4 — CT vs Lab Result Time", index=False)
    df_sheet5.to_excel(writer, sheet_name="5 — Repeated Orders",       index=False)
    df_sheet6.to_excel(writer, sheet_name="6 — Field Agreement",       index=False)
    df_sheet7.to_excel(writer, sheet_name="7 — Chronology Checks",     index=False)
    df_sheet8.to_excel(writer, sheet_name="8 — CT After Lab (Detail)", index=False)

size_kb = os.path.getsize(OUT) // 1024
print(f"  -> master_report.xlsx  ({size_kb} KB)")

# ---------------------------------------------------------------------------
# Verification summary
# ---------------------------------------------------------------------------
print("\n=== Verification ===")

# Sheet 1: Venn all-months totals
all_row = df_venn[df_venn["Month"] == "ALL MONTHS"].iloc[0]
month_rows = df_venn[df_venn["Month"] != "ALL MONTHS"]
cols_to_sum = [c for c in df_venn.columns if c not in ("Month",)]
print("\nCoverage Venn — All Months:")
for c in cols_to_sum:
    print(f"  {c:<45} {int(all_row[c]):>6,}")

# Sheet 4: CT vs lab result timing summary
all_row = df_timing_summary[df_timing_summary["Month"] == "ALL MONTHS"].iloc[0]
n = int(all_row["N Patients (CT + lab)"])
print(f"\nCT Order vs Lab Result Timing (all months, n={n:,}):")
for col in ["CT after ALL lab results", "CT after SOME lab results",
            "CT before any lab result", "CT same minute as last result",
            "Unknown (missing timestamps)"]:
    v = int(all_row.get(col, 0) or 0)
    print(f"  {col:<45} {v:>5,}  ({v/n*100:.1f}%)")
print(f"\nSheet 8 — CT after lab detail rows: {len(df_sheet8):,}")

print("\nDone.")
