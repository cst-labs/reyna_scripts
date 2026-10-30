"""
imaging_vs_lab_timing.py
For each patient (CSN), compares the CT imaging Order Date/Time against
four lab timestamps:
  - Collection Date/Time
  - Received Date/Time
  - Result Date/Time
  - Read/Ackn on (datetime)

For each comparison, classifies the imaging order as:
  Before   : imaging order strictly before the lab timestamp
  Same Min : within the same minute (±1 min tolerance)
  After    : imaging order strictly after the lab timestamp
  No Data  : one or both timestamps missing

Produces analysis/imaging_vs_lab_timing.xlsx with:
  Sheet 1 — Summary            : overall + by month, counts + % for each category
  Sheet 2 — Per Patient Detail : one row per CSN, all four comparisons
"""

import os, math
from datetime import datetime, timedelta, date
import openpyxl
import pandas as pd

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
OUT    = os.path.join(BASE, "analysis", "imaging_vs_lab_timing.xlsx")

MONTH_ORDER = ["Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sheet(path, sheet):
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
    if isinstance(val, date): return datetime(val.year, val.month, val.day)
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f): return None
        return datetime(1899, 12, 30) + timedelta(days=f)
    except: return None

import datetime as dt_mod
def combine_dt(d_val, t_val):
    d = to_dt(d_val)
    if d is None: return None
    if t_val is None: return d
    if isinstance(t_val, datetime):
        return datetime(d.year, d.month, d.day, t_val.hour, t_val.minute, t_val.second)
    if isinstance(t_val, dt_mod.time):
        return datetime(d.year, d.month, d.day, t_val.hour, t_val.minute, t_val.second)
    try:
        f = float(t_val)
        if 0 <= f < 2:
            ts = round(f * 86400); h, r = divmod(ts, 3600); m, s = divmod(r, 60)
            return datetime(d.year, d.month, d.day, h % 24, m, s)
    except: pass
    return d

def classify(img_dt, lab_dt, tol_mins=1):
    """Compare imaging order time to a lab timestamp."""
    if img_dt is None or lab_dt is None:
        return "No Data"
    diff = (img_dt - lab_dt).total_seconds() / 60
    if abs(diff) <= tol_mins:
        return "Same Minute"
    if diff < 0:
        return "Before"   # imaging order is before lab timestamp
    return "After"        # imaging order is after lab timestamp

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading...")
ct_path = os.path.join(MERGED, "ct_orders_combined.xlsx")
df_img  = load_sheet(ct_path, "Imaging")
df_lab  = load_sheet(ct_path, "Lab")

df_img["CSN_key"] = df_img["CSN"].apply(lambda v: str(v).strip())
df_lab["CSN_key"] = df_lab["CSN"].apply(lambda v: str(v).strip())

print(f"  Imaging: {len(df_img):,} rows")
print(f"  Lab:     {len(df_lab):,} rows")

# ---------------------------------------------------------------------------
# Compute imaging order datetime (CT orders only)
# ---------------------------------------------------------------------------
def is_ct(desc):
    v = str(desc).upper().strip() if desc else ""
    return v.startswith("NGEMR CT") or (v.startswith("CT ") and "XR" not in v)

df_ct = df_img[df_img["Order Description"].apply(is_ct)].copy()
df_ct["img_order_dt"] = df_ct.apply(
    lambda r: combine_dt(r["Order Date"], r["Order Time"]), axis=1)

print(f"  CT imaging orders: {len(df_ct):,} (of {len(df_img):,} total imaging orders)")

# Per patient: earliest CT order datetime
print("  Computing earliest CT order per patient...")
ct_first = (df_ct.dropna(subset=["img_order_dt"])
    .groupby("CSN_key")
    .agg(img_order_dt=("img_order_dt", "min"),
         month=("month", "first"),
         acuity=("Acuity Abbr", "first"),
         ed_disposition=("ED Disposition", "first"),
         primary_dx=("Primary ED Dx", "first"),
         n_ct_orders=("Order ID", "count"),
         ct_descriptions=("Order Description",
                          lambda x: "; ".join(sorted(set(x.dropna())))))
    .reset_index())

# ---------------------------------------------------------------------------
# Compute lab timestamps per patient (aggregate: first collection, first result, etc.)
# "First" is most conservative — if imaging is before the FIRST lab timestamp, it's
# clearly before all of them. Using FIRST gives the most meaningful clinical comparison.
# ---------------------------------------------------------------------------
print("  Computing lab timestamps per patient...")

df_lab["_collection_dt"] = df_lab.apply(
    lambda r: combine_dt(r["Collection Date"], r["Collection Time"]), axis=1)
df_lab["_received_dt"] = df_lab.apply(
    lambda r: combine_dt(r["Received Date"], r["Received Time"]), axis=1)
df_lab["_result_dt"] = df_lab.apply(
    lambda r: combine_dt(r["Result Date"], r["Result Time"]), axis=1)
df_lab["_readack_dt"] = df_lab["Read/Ackn on"].apply(to_dt)

lab_times = (df_lab.groupby("CSN_key")
    .agg(
        first_collection_dt=("_collection_dt", "min"),
        first_received_dt=("_received_dt",    "min"),
        first_result_dt=("_result_dt",        "min"),
        first_readack_dt=("_readack_dt",      "min"),
    )
    .reset_index())

# ---------------------------------------------------------------------------
# Merge and classify
# ---------------------------------------------------------------------------
print("  Merging and classifying...")
merged = ct_first.merge(lab_times, on="CSN_key", how="inner")
print(f"  Patients with both CT order and lab data: {len(merged):,}")

COMPARISONS = [
    ("vs Collection",  "img_order_dt", "first_collection_dt"),
    ("vs Received",    "img_order_dt", "first_received_dt"),
    ("vs Result",      "img_order_dt", "first_result_dt"),
    ("vs Read/Ack",    "img_order_dt", "first_readack_dt"),
]

for label, col_a, col_b in COMPARISONS:
    merged[f"_cls_{label}"] = merged.apply(
        lambda r: classify(r[col_a], r[col_b]), axis=1)
    merged[f"_diff_mins_{label}"] = merged.apply(
        lambda r: (r[col_a] - r[col_b]).total_seconds() / 60
        if r[col_a] is not None and r[col_b] is not None else None, axis=1)

# ---------------------------------------------------------------------------
# Sheet 1 — Summary table
# ---------------------------------------------------------------------------
print("  Building summary table...")

CATS = ["Before", "Same Minute", "After", "No Data"]

def summary_block(df, scope_label):
    n = len(df)
    rows = []
    for label, _, _ in COMPARISONS:
        col = f"_cls_{label}"
        diff_col = f"_diff_mins_{label}"
        vc = df[col].value_counts()
        valid = df[diff_col].dropna()
        row = {
            "Scope":               scope_label,
            "Comparison":          f"CT Order {label}",
            "N Patients":          n,
        }
        for cat in CATS:
            cnt = int(vc.get(cat, 0))
            row[cat]           = cnt
            row[f"% {cat}"]    = round(cnt / n * 100, 1) if n else 0
        # Median diff for Before/After (excluding No Data and Same Minute)
        before_vals = df.loc[df[col]=="Before",  diff_col].dropna()
        after_vals  = df.loc[df[col]=="After",   diff_col].dropna()
        row["Median mins (Before)"] = round(before_vals.median(), 0) if len(before_vals) else None
        row["Median mins (After)"]  = round(after_vals.median(), 0)  if len(after_vals) else None
        rows.append(row)
    return rows

summary_rows = []
# Overall
summary_rows += summary_block(merged, "All Months")
# Per month
for m in MONTH_ORDER:
    sub = merged[merged["month"] == m]
    if len(sub): summary_rows += summary_block(sub, m)

df_summary = pd.DataFrame(summary_rows)

# Reorder columns
col_order = ["Scope", "Comparison", "N Patients",
             "Before", "% Before", "Same Minute", "% Same Minute",
             "After",  "% After",  "No Data", "% No Data",
             "Median mins (Before)", "Median mins (After)"]
col_order = [c for c in col_order if c in df_summary.columns]
df_summary = df_summary[col_order]

# ---------------------------------------------------------------------------
# Sheet 2 — Per-patient detail
# ---------------------------------------------------------------------------
detail_cols = [
    "CSN_key", "month", "acuity", "ed_disposition", "primary_dx",
    "n_ct_orders", "ct_descriptions",
    "img_order_dt",
    "first_collection_dt", "first_received_dt",
    "first_result_dt",     "first_readack_dt",
]
for label, _, _ in COMPARISONS:
    detail_cols.append(f"_cls_{label}")
    detail_cols.append(f"_diff_mins_{label}")

df_detail = merged[detail_cols].rename(columns={
    "CSN_key":               "CSN",
    "month":                 "Month",
    "acuity":                "Acuity",
    "ed_disposition":        "ED Disposition",
    "primary_dx":            "Primary ED Dx",
    "n_ct_orders":           "N CT Orders",
    "ct_descriptions":       "CT Descriptions",
    "img_order_dt":          "CT Order DT",
    "first_collection_dt":   "First Lab Collection DT",
    "first_received_dt":     "First Lab Received DT",
    "first_result_dt":       "First Lab Result DT",
    "first_readack_dt":      "First Lab Read/Ack DT",
    "_cls_vs Collection":    "Class vs Collection",
    "_cls_vs Received":      "Class vs Received",
    "_cls_vs Result":        "Class vs Result",
    "_cls_vs Read/Ack":      "Class vs Read/Ack",
    "_diff_mins_vs Collection": "Diff mins vs Collection",
    "_diff_mins_vs Received":   "Diff mins vs Received",
    "_diff_mins_vs Result":     "Diff mins vs Result",
    "_diff_mins_vs Read/Ack":   "Diff mins vs Read/Ack",
}).sort_values(["Month","CSN"])

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
print("Writing imaging_vs_lab_timing.xlsx...")
with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Summary",            index=False)
    df_detail.to_excel( writer, sheet_name="Per Patient Detail", index=False)

size_kb = os.path.getsize(OUT) // 1024
print(f"  -> imaging_vs_lab_timing.xlsx  ({size_kb} KB)")

# Quick terminal preview
print("\n=== Summary (All Months) ===")
all_m = df_summary[df_summary["Scope"] == "All Months"]
print(all_m[["Comparison","N Patients","Before","% Before","Same Minute","% Same Minute",
             "After","% After","No Data","% No Data"]].to_string(index=False))
print("\nDone.")
