"""
consolidate_orders.py
Adds order-consolidation sheets to analysis/merged_analysis.xlsx:

  Imaging — Orders per Patient   : distribution of CT/XR orders per CSN
  Imaging — Per Patient Summary  : one row per CSN, all orders collapsed
  Imaging — Repeated Scan Types  : patients who got same scan description >1x
  Imaging — Order Type Mix       : CT vs XR vs Other breakdown
  Lab — Orders per Patient       : distribution of lab orders per CSN
  Lab — Per Patient Summary      : one row per CSN, all lab tests collapsed
  Lab — Top Tests                : frequency table of individual lab tests
"""

import os, math
from datetime import datetime, timedelta

import openpyxl
import pandas as pd

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
OUT    = os.path.join(BASE, "analysis")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_merged(path, sheet):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    seen = {}; uh = []
    for h in headers:
        if h in seen: seen[h] += 1; uh.append(f"{h}_{seen[h]}")
        else: seen[h] = 0; uh.append(h)
    return pd.DataFrame(rows[1:], columns=uh)

def to_dt(val):
    if val is None: return None
    try:
        if isinstance(val, float) and math.isnan(val): return None
    except: pass
    if isinstance(val, datetime): return val
    if isinstance(val, type(datetime.today().date())): return datetime(val.year, val.month, val.day)
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f): return None
        return datetime(1899, 12, 30) + timedelta(days=f)
    except: return None

print("Loading...")
ct_path = os.path.join(MERGED, "ct_orders_combined.xlsx")
df_img  = load_merged(ct_path, "Imaging")
df_lab  = load_merged(ct_path, "Lab")

df_img["CSN_key"] = df_img["CSN"].apply(str).str.strip()
df_lab["CSN_key"] = df_lab["CSN"].apply(str).str.strip()

print(f"  Imaging: {len(df_img):,} rows  |  Lab: {len(df_lab):,} rows")

# ---------------------------------------------------------------------------
# Classify order type
# ---------------------------------------------------------------------------
def classify_order(desc):
    v = str(desc).upper().strip() if desc else ""
    if v.startswith("NGEMR CT") or v.startswith("CT "):
        return "CT"
    if v.startswith("XR ") or v.startswith("NGEMR XR") or "X-RAY" in v:
        return "XR"
    if "MRI" in v:
        return "MRI"
    if v == "" or v == "NONE":
        return "Unknown"
    return "Other"

df_img["order_type"] = df_img["Order Description"].apply(classify_order)

# ---------------------------------------------------------------------------
# Sheet 1 — Imaging: orders-per-patient distribution
# ---------------------------------------------------------------------------
print("Building imaging per-patient distribution...")

img_per_pt = df_img.groupby("CSN_key").size().rename("total_orders")
ct_per_pt  = df_img[df_img["order_type"] == "CT"].groupby("CSN_key").size().rename("ct_orders")
xr_per_pt  = df_img[df_img["order_type"] == "XR"].groupby("CSN_key").size().rename("xr_orders")
other_per_pt = df_img[df_img["order_type"].isin(["MRI","Other","Unknown"])].groupby("CSN_key").size().rename("other_orders")

orders_per_pt = pd.concat([img_per_pt, ct_per_pt, xr_per_pt, other_per_pt], axis=1).fillna(0).astype(int)

# Distribution table
def dist_table(series, label):
    rows = []
    vc = series.value_counts().sort_index()
    for n, c in vc.items():
        rows.append({"N Orders": n, "Patient Count": c,
                     "% of Patients": round(c / len(series) * 100, 1)})
    df = pd.DataFrame(rows)
    # Summary stats row
    stats = {"N Orders": "Summary",
             "Patient Count": len(series),
             "% of Patients": None}
    for pct, q in [(50,"Median"),(75,"P75"),(90,"P90"),(95,"P95"),(99,"P99")]:
        stats[q] = series.quantile(pct/100)
    stats["Mean"] = round(series.mean(), 1)
    stats["Max"]  = series.max()
    return df, pd.DataFrame([stats])

img_dist, img_summary = dist_table(orders_per_pt["total_orders"], "Total Imaging")
ct_dist,  ct_summary  = dist_table(orders_per_pt["ct_orders"],    "CT only")

df_img_dist = pd.concat([
    pd.DataFrame([{"N Orders": "=== ALL IMAGING ORDERS ==="}]),
    img_dist, img_summary,
    pd.DataFrame([{"N Orders": "=== CT ORDERS ONLY ==="}]),
    ct_dist, ct_summary,
], ignore_index=True)

# ---------------------------------------------------------------------------
# Sheet 2 — Imaging: per-patient summary (one row per CSN)
# ---------------------------------------------------------------------------
print("Building imaging per-patient summary...")

def join_unique(series, sep="; "):
    vals = series.dropna().apply(str).str.strip()
    vals = vals[vals.isin(["","None","nan"]) == False]
    return sep.join(sorted(set(vals))) if len(vals) else ""

img_summary_rows = []
for csn, grp in df_img.groupby("CSN_key"):
    grp = grp.sort_values("Order Date", na_position="last")
    ct_grp = grp[grp["order_type"] == "CT"]
    xr_grp = grp[grp["order_type"] == "XR"]

    # Repeated same CT description?
    ct_descs = ct_grp["Order Description"].apply(lambda v: str(v).upper().strip() if v else "").tolist()
    ct_desc_counts = pd.Series(ct_descs).value_counts()
    repeated_ct = ct_desc_counts[ct_desc_counts > 1]

    row = {
        "CSN":                   csn,
        "Month":                 grp["month"].iloc[0],
        "N Total Imaging Orders": len(grp),
        "N CT Orders":           len(ct_grp),
        "N XR Orders":           len(xr_grp),
        "N Other Orders":        len(grp) - len(ct_grp) - len(xr_grp),
        "CT Descriptions":       join_unique(ct_grp["Order Description"]),
        "XR Descriptions":       join_unique(xr_grp["Order Description"]),
        "Repeated CT Descriptions": "; ".join(f"{d}(x{n})" for d, n in repeated_ct.items()) if len(repeated_ct) else "",
        "Has Repeated CT":       "Yes" if len(repeated_ct) > 0 else "No",
        "Acuity":                grp["Acuity Abbr"].dropna().iloc[0] if grp["Acuity Abbr"].notna().any() else "",
        "ED Disposition":        grp["ED Disposition"].dropna().iloc[0] if grp["ED Disposition"].notna().any() else "",
        "First Order Date":      str(grp["Order Date"].iloc[0])[:10] if grp["Order Date"].notna().any() else "",
        "Last Order Date":       str(grp["Order Date"].iloc[-1])[:10] if grp["Order Date"].notna().any() else "",
        "Abnormal Results":      (grp["Abnormal"] == "Abnormal").sum(),
        "Critical Results":      grp["Critical Result"].notna().sum(),
    }
    img_summary_rows.append(row)

df_img_per_pt = pd.DataFrame(img_summary_rows).sort_values(["Month","CSN"])

# ---------------------------------------------------------------------------
# Sheet 3 — Imaging: repeated scan types detail
# ---------------------------------------------------------------------------
print("Building repeated scan types sheet...")

repeat_rows = []
for csn, grp in df_img[df_img["order_type"] == "CT"].groupby("CSN_key"):
    desc_counts = grp["Order Description"].apply(
        lambda v: str(v).upper().strip() if v else "").value_counts()
    for desc, cnt in desc_counts[desc_counts > 1].items():
        sub = grp[grp["Order Description"].apply(
            lambda v: str(v).upper().strip() if v else "") == desc]
        # Check if they are on different dates (serial scans) or same date (potential dup)
        order_dates = sub["Order Date"].dropna().apply(
            lambda v: str(v)[:10]).unique().tolist()
        repeat_rows.append({
            "CSN":              csn,
            "Month":            grp["month"].iloc[0],
            "Order Description": desc,
            "N Repeats":        int(cnt),
            "Order Dates":      "; ".join(sorted(order_dates)),
            "Same Date":        "Yes" if len(order_dates) == 1 else "No",
            "Acuity":           grp["Acuity Abbr"].dropna().iloc[0] if grp["Acuity Abbr"].notna().any() else "",
        })

df_repeats = pd.DataFrame(repeat_rows).sort_values(
    ["Same Date","N Repeats","Order Description"], ascending=[False, False, True])

# ---------------------------------------------------------------------------
# Sheet 4 — Imaging: order type mix
# ---------------------------------------------------------------------------
print("Building order type mix sheet...")

mix_rows = []
for month in ["All"] + sorted(df_img["month"].dropna().unique().tolist()):
    sub = df_img if month == "All" else df_img[df_img["month"] == month]
    total = len(sub)
    for otype in ["CT","XR","MRI","Other","Unknown"]:
        n = (sub["order_type"] == otype).sum()
        if n > 0 or otype in ["CT","XR"]:
            mix_rows.append({
                "Month": month, "Order Type": otype,
                "N Orders": int(n),
                "% of Orders": round(n / total * 100, 1) if total else 0,
                "Unique Patients": sub[sub["order_type"] == otype]["CSN_key"].nunique(),
            })

df_mix = pd.DataFrame(mix_rows)

# ---------------------------------------------------------------------------
# Sheet 5 — Lab: orders-per-patient distribution
# ---------------------------------------------------------------------------
print("Building lab per-patient distribution...")

lab_per_pt = df_lab.groupby("CSN_key").size()
lab_dist, lab_summary = dist_table(lab_per_pt, "Lab")

df_lab_dist = pd.concat([lab_dist, lab_summary], ignore_index=True)

# ---------------------------------------------------------------------------
# Sheet 6 — Lab: per-patient summary
# ---------------------------------------------------------------------------
print("Building lab per-patient summary...")

lab_summary_rows = []
for csn, grp in df_lab.groupby("CSN_key"):
    grp = grp.sort_values("Order Date", na_position="last")
    test_counts = grp["Order Description"].value_counts()
    repeated = test_counts[test_counts > 1]

    row = {
        "CSN":                  csn,
        "Month":                grp["month"].iloc[0],
        "N Lab Orders":         len(grp),
        "N Unique Test Types":  grp["Order Description"].nunique(),
        "Test Types":           join_unique(grp["Order Description"]),
        "Repeated Tests":       "; ".join(f"{t}(x{n})" for t, n in repeated.items()) if len(repeated) else "",
        "N Repeated Tests":     int((test_counts > 1).sum()),
        "Has Repeated Test":    "Yes" if len(repeated) > 0 else "No",
        "Abnormal Results":     (grp["Abnormal"] == "Abnormal").sum(),
        "Critical Results":     grp["Critical Result"].notna().sum(),
        "Acuity":               grp["Acuity Abbr"].dropna().iloc[0] if grp["Acuity Abbr"].notna().any() else "",
        "First Order Date":     str(grp["Order Date"].iloc[0])[:10] if grp["Order Date"].notna().any() else "",
        "Last Order Date":      str(grp["Order Date"].iloc[-1])[:10] if grp["Order Date"].notna().any() else "",
    }
    lab_summary_rows.append(row)

df_lab_per_pt = pd.DataFrame(lab_summary_rows).sort_values(["Month","CSN"])

# ---------------------------------------------------------------------------
# Sheet 7 — Lab: top tests frequency table
# ---------------------------------------------------------------------------
print("Building lab top tests table...")

test_freq = df_lab.groupby("Order Description").agg(
    N_Orders=("Order ID","count"),
    N_Patients=("CSN_key","nunique"),
).reset_index().rename(columns={"Order Description":"Test Name"})
test_freq["Orders per Patient"] = (test_freq["N_Orders"] / test_freq["N_Patients"]).round(2)
test_freq = test_freq.sort_values("N_Orders", ascending=False).reset_index(drop=True)

# Add % of patients who had this test
total_pts = df_lab["CSN_key"].nunique()
test_freq["% Patients"] = (test_freq["N_Patients"] / total_pts * 100).round(1)

# ---------------------------------------------------------------------------
# Write — append to existing merged_analysis.xlsx
# ---------------------------------------------------------------------------
print("Writing sheets to merged_analysis.xlsx...")
out_path = os.path.join(OUT, "merged_analysis.xlsx")

# Load existing workbook and append
with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_img_dist.to_excel(  writer, sheet_name="Img — Orders per Patient",   index=False)
    df_img_per_pt.to_excel(writer, sheet_name="Img — Per Patient Summary",  index=False)
    df_repeats.to_excel(   writer, sheet_name="Img — Repeated Scan Types",  index=False)
    df_mix.to_excel(       writer, sheet_name="Img — Order Type Mix",       index=False)
    df_lab_dist.to_excel(  writer, sheet_name="Lab — Orders per Patient",   index=False)
    df_lab_per_pt.to_excel(writer, sheet_name="Lab — Per Patient Summary",  index=False)
    test_freq.to_excel(    writer, sheet_name="Lab — Top Tests",            index=False)

size_kb = os.path.getsize(out_path) // 1024
print(f"  -> merged_analysis.xlsx  ({size_kb} KB)")

# Print summary
print("\n=== Imaging order type mix (all months) ===")
print(df_mix[df_mix["Month"]=="All"].to_string(index=False))

print("\n=== CT repeated scans: same-day vs different-day ===")
print(f"  Total CSN+description repeats:   {len(df_repeats)}")
print(f"  Same-day (potential duplicates):  {(df_repeats['Same Date']=='Yes').sum()}")
print(f"  Different-day (serial scans):     {(df_repeats['Same Date']=='No').sum()}")

print("\n=== Lab top 10 tests ===")
print(test_freq.head(10)[["Test Name","N_Orders","N_Patients","% Patients","Orders per Patient"]].to_string(index=False))

print("\nDone.")
