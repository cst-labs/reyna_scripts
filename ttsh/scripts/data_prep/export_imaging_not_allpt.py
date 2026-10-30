"""
export_imaging_not_allpt.py
Generates analysis/imaging_not_in_allpt.xlsx

Single sheet: all 172 patients in CT Imaging but absent from combined All Pt,
with every available imaging field plus a Category and Explanation column,
sorted by category then arrival date.

Categories:
  A — ED patient, arrival before dataset (Oct 2025)
  B — ED patient, arrived last days of month, ED case straddled boundary
  C — ED patient, null arrival date (Dec 2025 imaging export missing Arrival col)
  D — Non-ED, arrived prior month (inpatient/outpatient, no triage)
  E — Non-ED, arrived in scan month (inpatient/outpatient, no triage)
"""

import os, math
from datetime import datetime, timedelta, date
import openpyxl
import pandas as pd

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
OUT    = os.path.join(BASE, "analysis", "imaging_not_in_allpt.xlsx")

MONTH_RANGES = {
    "Nov 2025": (datetime(2025,11,1),  datetime(2025,11,30)),
    "Dec 2025": (datetime(2025,12,1),  datetime(2025,12,31)),
    "Jan 2026": (datetime(2026,1,1),   datetime(2026,1,31)),
    "Feb 2026": (datetime(2026,2,1),   datetime(2026,2,28)),
    "Mar 2026": (datetime(2026,3,1),   datetime(2026,3,31)),
    "Apr 2026": (datetime(2026,4,1),   datetime(2026,4,30)),
}
DATASET_START = datetime(2025, 11, 1)

CATEGORIES = {
    "A": "A — ED patient: arrived Oct 2025, before dataset start. No Oct All Pt file.",
    "B": "B — ED patient: arrived last days of prior month, ED case straddled month boundary. Absent from both arrival-month and scan-month All Pt.",
    "C": "C — ED patient: arrival date null (Dec 2025 imaging export missing Arrival Date/Time columns). Has acuity and disposition.",
    "D": "D — Non-ED: arrived prior month, no triage acuity or ED disposition. Likely inpatient/outpatient order.",
    "E": "E — Non-ED: arrived within scan month, no triage acuity or ED disposition. Likely inpatient/outpatient order (mainly CT KUB).",
}

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

def normalise_csn(val):
    s = str(val).strip() if val is not None else ""
    if s.endswith(".0"): s = s[:-2]
    return s

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
    if isinstance(t_val, datetime):
        return datetime(d.year, d.month, d.day, t_val.hour, t_val.minute, t_val.second)
    import datetime as dt_mod
    if isinstance(t_val, dt_mod.time):
        return datetime(d.year, d.month, d.day, t_val.hour, t_val.minute, t_val.second)
    try:
        f = float(t_val)
        if 0 <= f < 2:
            total_secs = round(f * 86400)
            h, rem = divmod(total_secs, 3600)
            m, s = divmod(rem, 60)
            return datetime(d.year, d.month, d.day, h % 24, m, s)
    except: pass
    return d

def fmt(d_val, t_val=None):
    try:
        dt = combine_dt(d_val, t_val) if t_val is not None else to_dt(d_val)
        if dt is None: return ""
        if t_val is None or (dt.hour == 0 and dt.minute == 0 and dt.second == 0):
            return dt.strftime("%Y-%m-%d")
        return dt.strftime("%Y-%m-%d %H:%M")
    except: return ""

def s(val, n=None):
    v = str(val).strip() if val is not None and str(val) not in ("nan","None","") else ""
    return v[:n] if n else v

def has_acuity(val):
    return s(val) not in ("", "nan", "None")

def arrival_month(arr_dt):
    if arr_dt is None: return None
    for label, (st, en) in MONTH_RANGES.items():
        if st <= arr_dt <= en: return label
    return None

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading...")
df_ap  = load_sheet(f"{MERGED}/all_pt_combined.xlsx",    "All Patients")
df_img = load_sheet(f"{MERGED}/ct_orders_combined.xlsx", "Imaging")
df_lab = load_sheet(f"{MERGED}/ct_orders_combined.xlsx", "Lab")

df_ap ["CSN_key"] = df_ap["CSN"].apply(normalise_csn)
df_img["CSN_key"] = df_img["CSN"].apply(normalise_csn)
df_lab["CSN_key"] = df_lab["CSN"].apply(normalise_csn)

ap_csns  = set(df_ap ["CSN_key"])
lab_csns = set(df_lab["CSN_key"])

img_not_ap = df_img[~df_img["CSN_key"].isin(ap_csns)].copy()
img_not_ap["arrival_dt"] = img_not_ap["Arrival Date"].apply(to_dt)

def categorise(row):
    arr       = row["arrival_dt"]
    scan_m    = row["month"]
    scan_start = MONTH_RANGES.get(scan_m, (None,None))[0]
    acuity_ok = has_acuity(row.get("Acuity Abbr"))

    if arr is None:
        return "C" if acuity_ok else "E"
    if arr < DATASET_START:
        return "A"
    if scan_start and arr >= scan_start:
        return "E"   # arrived in scan month, no acuity = non-ED
    # arrived in a prior month within dataset range
    if acuity_ok:
        return "B"
    else:
        return "D"

img_not_ap["_cat"] = img_not_ap.apply(categorise, axis=1)

# ---------------------------------------------------------------------------
# Build one row per ORDER (not per patient) — all imaging fields + category info
# ---------------------------------------------------------------------------

# All columns from the imaging sheet, in their original order
IMG_COLS = [c for c in df_img.columns if c not in ("CSN_key","arrival_dt","_cat")]

# Compute per-CSN summaries for extra context columns
def classify_order_type(desc):
    v = s(desc).upper()
    if v.startswith("NGEMR CT") or (v.startswith("CT ") and "XR" not in v): return "CT"
    if v.startswith("XR ") or v.startswith("NGEMR XR"): return "XR"
    if "MRI" in v: return "MRI"
    return "Other"

img_not_ap["_order_type"] = img_not_ap["Order Description"].apply(classify_order_type)

# Per-CSN: N orders, CT descriptions, has lab
csn_summary = {}
for csn, grp in img_not_ap.groupby("CSN_key"):
    ct_descs = grp[grp["_order_type"]=="CT"]["Order Description"].dropna().unique().tolist()
    csn_summary[csn] = {
        "N Imaging Orders (this CSN)": len(grp),
        "N CT Orders":  (grp["_order_type"]=="CT").sum(),
        "N XR Orders":  (grp["_order_type"]=="XR").sum(),
        "CT Descriptions (all)": "; ".join(sorted(set(ct_descs))),
        "Has Lab Orders": "Yes" if csn in lab_csns else "No",
    }

# Build output dataframe: category header rows + data rows
output_rows = []
cat_order = ["A","B","C","D","E"]

for cat in cat_order:
    subset = img_not_ap[img_not_ap["_cat"] == cat].sort_values(
        ["arrival_dt","CSN_key","Order Date"], na_position="last")
    if len(subset) == 0:
        continue

    # Section separator row
    n_csns = subset["CSN_key"].nunique()
    sep = {c: "" for c in IMG_COLS}
    sep["CSN"]            = f"▶ CATEGORY {cat} — {CATEGORIES[cat]}"
    sep["month"]          = f"({n_csns} patients, {len(subset)} orders)"
    sep["_is_separator"]  = True
    output_rows.append(sep)

    for _, r in subset.iterrows():
        row = {c: r[c] for c in IMG_COLS if c in r.index}
        csn = r["CSN_key"]
        row["CSN"] = csn

        # Extra context columns at the front
        row["Category"]       = cat
        row["Explanation"]    = CATEGORIES[cat]
        row["Arrival Month"]  = arrival_month(r["arrival_dt"]) or ("Pre-Nov 2025" if r["arrival_dt"] and r["arrival_dt"] < DATASET_START else "Unknown/Null")
        row["Order Type"]     = r["_order_type"]

        sm = csn_summary.get(csn, {})
        row["N Imaging Orders (this CSN)"] = sm.get("N Imaging Orders (this CSN)", "")
        row["N CT Orders"]                 = sm.get("N CT Orders", "")
        row["N XR Orders"]                 = sm.get("N XR Orders", "")
        row["CT Descriptions (all)"]       = sm.get("CT Descriptions (all)", "")
        row["Has Lab Orders"]              = sm.get("Has Lab Orders", "")
        row["_is_separator"]               = False
        output_rows.append(row)

# Build the final column order: context cols first, then all imaging cols
CONTEXT_COLS = [
    "Category", "Explanation", "Arrival Month",
    "N Imaging Orders (this CSN)", "N CT Orders", "N XR Orders",
    "CT Descriptions (all)", "Has Lab Orders", "Order Type",
]
# Imaging cols: month + CSN first, then the rest
IMG_ORDERED = ["month","CSN"] + [c for c in IMG_COLS if c not in ("month","CSN")]
ALL_COLS = CONTEXT_COLS + IMG_ORDERED

df_out = pd.DataFrame(output_rows)
# Fill missing columns
for c in ALL_COLS:
    if c not in df_out.columns:
        df_out[c] = ""
df_out = df_out[ALL_COLS + ["_is_separator"]]

# Strip the internal separator flag col before writing
df_write = df_out.drop(columns=["_is_separator"])

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
print("Writing imaging_not_in_allpt.xlsx...")
with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
    df_write.to_excel(writer, sheet_name="Imaging Not In All Pt", index=False)

    # Apply bold styling to separator rows
    wb = writer.book
    ws = wb["Imaging Not In All Pt"]
    from openpyxl.styles import Font, PatternFill
    sep_fill  = PatternFill("solid", fgColor="D9E1F2")
    sep_font  = Font(bold=True)
    # Row 1 is header, data starts at row 2
    for i, is_sep in enumerate(df_out["_is_separator"], start=2):
        if is_sep:
            for cell in ws[i]:
                cell.font  = sep_font
                cell.fill  = sep_fill

size_kb = os.path.getsize(OUT) // 1024
print(f"  -> imaging_not_in_allpt.xlsx  ({size_kb} KB)")

# Summary
print("\nCategory breakdown:")
for cat in cat_order:
    n_csns   = img_not_ap[img_not_ap["_cat"]==cat]["CSN_key"].nunique()
    n_orders = (img_not_ap["_cat"]==cat).sum()
    if n_csns:
        print(f"  {cat}: {n_csns} patients, {n_orders} orders — {CATEGORIES[cat][:70]}")
print(f"\n  Total: {img_not_ap['CSN_key'].nunique()} patients, {len(img_not_ap)} orders")
