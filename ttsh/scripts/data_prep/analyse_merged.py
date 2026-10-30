"""
analyse_merged.py
Merges All Pt + CT Imaging + CT Lab on CSN, then produces:
  analysis/merged_analysis.xlsx with 5 sheets:
    1. Duplicate Field Agreement  — do shared fields agree across sources?
    2. Chronology Checks          — do timestamps follow clinical logic?
    3. Coverage Summary           — Venn breakdown of CSN coverage
    4. All Pt Only — No Orders    — patients with no CT imaging or lab
    5. Spillover CSNs             — CT/lab orders not matched to All Pt
"""

import os, math
from datetime import datetime, timedelta, date, time as time_type

import openpyxl
import pandas as pd

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
OUT    = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

MONTH_ORDER = ["Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026"]

# ---------------------------------------------------------------------------
# Load helpers
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


def normalise_csn(series):
    return series.apply(str).str.strip().where(
        series.apply(str).str.strip().str.match(r'^\d+$'), other=None
    )


def to_dt(val):
    """Coerce a value to datetime — handles Python datetime, date, Excel OLE serial."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        return datetime(val.year, val.month, val.day)
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return datetime(1899, 12, 30) + timedelta(days=f)
    except Exception:
        return None


def coerce_dt_col(series):
    return series.apply(to_dt)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading merged files...")
allpt_path = os.path.join(MERGED, "all_pt_combined.xlsx")
ct_path    = os.path.join(MERGED, "ct_orders_combined.xlsx")

df_ap  = load_merged(allpt_path, "All Patients")
df_img = load_merged(ct_path,    "Imaging")
df_lab = load_merged(ct_path,    "Lab")

print(f"  All Pt:  {len(df_ap):,} rows")
print(f"  Imaging: {len(df_img):,} rows")
print(f"  Lab:     {len(df_lab):,} rows")

# Normalise CSN
df_ap ["CSN_key"] = normalise_csn(df_ap ["CSN"])
df_img["CSN_key"] = normalise_csn(df_img["CSN"])
df_lab["CSN_key"] = normalise_csn(df_lab["CSN"])

# Coerce All Pt Arrival Date (Apr 2026 has OLE serials)
df_ap["Arrival Date_dt"] = coerce_dt_col(df_ap["Arrival Date"])
df_ap["FCT_dt"] = df_ap["FCT"].apply(
    lambda v: (datetime.strptime(str(v).strip(), "%d/%m/%Y %I:%M:%S %p")
               if v and str(v).strip() not in ("", "nan", "None") else None)
)


# ---------------------------------------------------------------------------
# Step 1 — Merge (left join from orders onto All Pt)
# ---------------------------------------------------------------------------
print("Merging...")

# Pull only the fields we need from All Pt for the join
AP_SHARED = [
    "CSN_key", "month",
    "Arrival Date_dt", "FCT_dt",
    "Primary ED Dx", "ED Dx", "ED Disposition",
    "Pt. Doc Type", "Specialty",
    "Admit Dispo Date", "Admit Dispo Time",
    "Bed Assigned Date", "Bed Assigned Time",
    "Admit IP Room", "Admit IP Bed",
    "Triage Acuity", "ED Primary Dx ICD10",
]
AP_SHARED = [c for c in AP_SHARED if c in df_ap.columns]
ap_slim = df_ap[AP_SHARED].copy()

# De-duplicate All Pt by CSN_key (should already be 1 per CSN, but guard)
ap_dedup = ap_slim.drop_duplicates(subset="CSN_key")

df_img_merged = df_img.merge(ap_dedup, on="CSN_key", how="left", suffixes=("", "_ap"))
df_lab_merged = df_lab.merge(ap_dedup, on="CSN_key", how="left", suffixes=("", "_ap"))

print(f"  Img merged: {len(df_img_merged):,} rows")
print(f"  Lab merged: {len(df_lab_merged):,} rows")


# ---------------------------------------------------------------------------
# Step 2 — Duplicate field agreement analysis
# ---------------------------------------------------------------------------
print("Analysing duplicate fields...")

# Fields to compare: (field_in_orders, field_from_ap_after_merge, label)
# Date-only fields — compared as YYYY-MM-DD strings
SHARED_DATE_FIELDS = [
    ("Bed Assigned Date","Bed Assigned Date_ap","Bed Assigned Date"),
    ("Admit Dispo Date", "Admit Dispo Date_ap", "Admit Dispo Date"),
]
# Text/categorical fields
SHARED_FIELDS = [
    ("Primary ED Dx",    "Primary ED Dx_ap",    "Primary ED Dx"),
    ("ED Dx",            "ED Dx_ap",            "ED Dx"),
    ("ED Disposition",   "ED Disposition_ap",   "ED Disposition"),
    ("Pt. Doc Type",     "Pt. Doc Type_ap",     "Pt. Doc Type"),
    ("Specialty",        "Specialty_ap",        "Specialty"),
    ("Admit IP Room",    "Admit IP Room_ap",    "Admit IP Room"),
    ("Admit IP Bed",     "Admit IP Bed_ap",     "Admit IP Bed"),
]
# Time fields — OLE fraction vs HH:MM:SS string, ±1 min rounding artefact expected
SHARED_TIME_FIELDS = [
    ("Bed Assigned Time","Bed Assigned Time_ap","Bed Assigned Time"),
    ("Admit Dispo Time", "Admit Dispo Time_ap", "Admit Dispo Time"),
]

def _ole_frac_to_hhmm(f):
    """Convert OLE fractional-day float to HH:MM string."""
    try:
        total_mins = round(f * 1440)
        h, m = divmod(total_mins, 60)
        return f"{h % 24:02d}:{m:02d}"
    except Exception:
        return None


def _normalise_val(v):
    """Coerce to comparable string; handles datetimes, OLE serials/fractions, time strings."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass
    if hasattr(v, 'isnull') and v.isnull():
        return None
    # Pure time object
    if isinstance(v, time_type):
        return f"{v.hour:02d}:{v.minute:02d}"
    # Datetime → date string
    dt = to_dt(v)
    if dt is not None:
        try:
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None
    # OLE fractional-day (0 < f < 1) — time value
    try:
        f = float(v)
        if 0 <= f < 1:
            return _ole_frac_to_hhmm(f)
    except Exception:
        pass
    # HH:MM or HH:MM:SS string → HH:MM
    s = str(v).strip()
    if len(s) >= 5 and s[2] == ':':
        return s[:5]
    return s.upper() if s not in ("", "NONE", "NAN", "NAT") else None
    s = str(v).strip().upper()
    return s if s not in ("", "NONE", "NAT", "NAT", "NAN") else None


def compare_field(df, col_orders, col_ap, label, source_label):
    if col_orders not in df.columns or col_ap not in df.columns:
        return None
    a = df[col_orders].apply(_normalise_val)
    b = df[col_ap].apply(_normalise_val)
    both = a.notna() & b.notna()
    n_both = both.sum()
    if n_both == 0:
        return {"Field": label, "Source Pair": source_label,
                "N Both Non-null": 0, "N Agree": 0, "N Disagree": 0,
                "Disagree %": None, "Sample Disagreements": ""}
    agree = (a[both] == b[both]).sum()
    disagree = n_both - agree
    pct = round(disagree / n_both * 100, 1)
    # Sample disagreements
    bad_mask = both & (a != b)
    samples = df[bad_mask][["CSN_key", "month_ap" if "month_ap" in df.columns else "month",
                             col_orders, col_ap]].head(3)
    sample_str = "; ".join(
        f"CSN {r['CSN_key']}: orders='{r[col_orders]}' vs ap='{r[col_ap]}'"
        for _, r in samples.iterrows()
    )
    return {"Field": label, "Source Pair": source_label,
            "N Both Non-null": int(n_both), "N Agree": int(agree),
            "N Disagree": int(disagree), "Disagree %": pct,
            "Sample Disagreements": sample_str[:300]}

def compare_time_field(df, col_orders, col_ap, label, source_label):
    """Compare time fields, allowing ±1 min rounding tolerance (OLE fraction vs HH:MM string)."""
    if col_orders not in df.columns or col_ap not in df.columns:
        return None
    def to_mins(v):
        if v is None: return None
        try:
            if isinstance(v, float) and math.isnan(v): return None
            if isinstance(v, time_type): return v.hour * 60 + v.minute
            f = float(v)
            if 0 <= f < 1: return round(f * 1440)
        except Exception: pass
        s = str(v).strip()
        if len(s) >= 5 and s[2] == ':':
            try: return int(s[:2]) * 60 + int(s[3:5])
            except Exception: pass
        return None
    a = df[col_orders].apply(to_mins)
    b = df[col_ap].apply(to_mins)
    both = a.notna() & b.notna()
    n_both = both.sum()
    if n_both == 0:
        return {"Field": label, "Source Pair": source_label,
                "N Both Non-null": 0, "N Agree": 0, "N Disagree": 0,
                "Disagree %": None, "Sample Disagreements": "",
                "Notes": ""}
    agree = (abs(a[both] - b[both]) <= 1).sum()  # ±1 min tolerance
    disagree = n_both - agree
    pct = round(disagree / n_both * 100, 1)
    bad_mask = both & (abs(a - b) > 1)
    samples = df[bad_mask][["CSN_key", col_orders, col_ap]].head(3)
    sample_str = "; ".join(
        f"CSN {r['CSN_key']}: orders={r[col_orders]!r} vs ap={r[col_ap]!r}"
        for _, r in samples.iterrows()
    )
    return {"Field": label, "Source Pair": source_label,
            "N Both Non-null": int(n_both), "N Agree": int(agree),
            "N Disagree": int(disagree), "Disagree %": pct,
            "Sample Disagreements": sample_str[:300],
            "Notes": "Time comparison uses ±1 min tolerance (OLE fraction vs HH:MM:SS rounding)"}

agreement_rows = []
for col_o, col_a, lbl in SHARED_FIELDS + SHARED_DATE_FIELDS:
    row = compare_field(df_img_merged, col_o, col_a, lbl, "Imaging vs All Pt")
    if row:
        row["Notes"] = ""
        agreement_rows.append(row)
for col_o, col_a, lbl in SHARED_FIELDS + SHARED_DATE_FIELDS:
    row = compare_field(df_lab_merged, col_o, col_a, lbl, "Lab vs All Pt")
    if row:
        row["Notes"] = ""
        agreement_rows.append(row)
for col_o, col_a, lbl in SHARED_TIME_FIELDS:
    row = compare_time_field(df_img_merged, col_o, col_a, lbl, "Imaging vs All Pt")
    if row: agreement_rows.append(row)
for col_o, col_a, lbl in SHARED_TIME_FIELDS:
    row = compare_time_field(df_lab_merged, col_o, col_a, lbl, "Lab vs All Pt")
    if row: agreement_rows.append(row)

# Arrival Date: Imaging vs All Pt (special — one is already a datetime, compare date only)
def compare_arrival(df, source_label):
    if "Arrival Date" not in df.columns or "Arrival Date_dt" not in df.columns:
        return None
    a = coerce_dt_col(df["Arrival Date"]).apply(lambda d: d.date() if d else None)
    b = df["Arrival Date_dt"].apply(lambda d: d.date() if d else None)
    both = a.notna() & b.notna()
    n_both = both.sum()
    if n_both == 0:
        return {"Field": "Arrival Date", "Source Pair": source_label,
                "N Both Non-null": 0, "N Agree": 0, "N Disagree": 0,
                "Disagree %": None, "Sample Disagreements": ""}
    agree = (a[both] == b[both]).sum()
    disagree = n_both - agree
    pct = round(disagree / n_both * 100, 1)
    bad_mask = both & (a != b)
    samples = df[bad_mask][["CSN_key"]].assign(
        arrival_img=a[bad_mask], arrival_ap=b[bad_mask]
    ).head(3)
    sample_str = "; ".join(
        f"CSN {r['CSN_key']}: orders={r['arrival_img']} vs ap={r['arrival_ap']}"
        for _, r in samples.iterrows()
    )
    return {"Field": "Arrival Date", "Source Pair": source_label,
            "N Both Non-null": int(n_both), "N Agree": int(agree),
            "N Disagree": int(disagree), "Disagree %": pct,
            "Sample Disagreements": sample_str[:300]}

row = compare_arrival(df_img_merged, "Imaging vs All Pt")
if row:
    row["Notes"] = ""
    agreement_rows.append(row)

df_agreement = pd.DataFrame(agreement_rows)
df_agreement = df_agreement.sort_values(["Field", "Source Pair"])


# ---------------------------------------------------------------------------
# Step 3 — Chronological sense check
# ---------------------------------------------------------------------------
print("Running chronology checks...")

def dt_col(df, col):
    if col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)
    return coerce_dt_col(df[col])

def check_order(df, col_a, col_b, label, source):
    a = dt_col(df, col_a)
    b = dt_col(df, col_b)
    both = a.notna() & b.notna()
    n_eval = both.sum()
    if n_eval == 0:
        return {"Check": label, "Source": source, "N Evaluable": 0,
                "N Pass": 0, "N Fail": 0, "Fail %": None, "Sample Fail CSNs": ""}
    ok = (a[both] <= b[both]).sum()
    fail = n_eval - ok
    pct = round(fail / n_eval * 100, 1)
    fail_csns = df.loc[both & (a > b), "CSN_key"].dropna().unique()[:5]
    return {"Check": label, "Source": source,
            "N Evaluable": int(n_eval), "N Pass": int(ok), "N Fail": int(fail),
            "Fail %": pct, "Sample Fail CSNs": ", ".join(fail_csns)}

chrono_rows = []

# Imaging chronology checks
img = df_img_merged.copy()
img["_order_dt"]  = dt_col(img, "Order Date").combine(dt_col(img, "Order Time"),
    lambda d, t: datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)
    if d and t else (d if d else None))
img["_exam_s_dt"] = dt_col(img, "Exam Start Date").combine(dt_col(img, "Exam Start Time"),
    lambda d, t: datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)
    if d and t else (d if d else None))
img["_exam_e_dt"] = dt_col(img, "Exam End Date").combine(dt_col(img, "Exam End Time"),
    lambda d, t: datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)
    if d and t else (d if d else None))
img["_result_dt"] = dt_col(img, "Result Date").combine(dt_col(img, "Result Time"),
    lambda d, t: datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)
    if d and t else (d if d else None))
img["_arrival_dt"] = img["Arrival Date_dt"]

chrono_rows.append(check_order(img, "_arrival_dt", "_order_dt",   "Arrival ≤ CT Order Date",      "Imaging"))
chrono_rows.append(check_order(img, "_order_dt",   "_exam_s_dt",  "CT Order ≤ Exam Start",         "Imaging"))
chrono_rows.append(check_order(img, "_exam_s_dt",  "_exam_e_dt",  "Exam Start ≤ Exam End",         "Imaging"))
chrono_rows.append(check_order(img, "_exam_e_dt",  "_result_dt",  "Exam End ≤ Result",             "Imaging"))
chrono_rows.append(check_order(img, "_arrival_dt", "_result_dt",  "Arrival ≤ Result",              "Imaging"))
# FCT is a full datetime; Order Date is date-only (time = midnight). Compare date parts only.
def _to_date_midnight(v):
    if v is None: return None
    try:
        if isinstance(v, float) and math.isnan(v): return None
        return datetime(int(v.year), int(v.month), int(v.day))
    except Exception: return None

img["_fct_date_dt"]     = img["FCT_dt"].apply(_to_date_midnight)
img["_order_date_only"] = img["_order_dt"].apply(_to_date_midnight)
chrono_rows.append(check_order(img, "_fct_date_dt", "_order_date_only", "FCT (Triage) date ≤ CT Order date", "Imaging"))

# Lab chronology checks
lab = df_lab_merged.copy()
lab["_order_dt"]  = dt_col(lab, "Order Date").combine(dt_col(lab, "Order Time"),
    lambda d, t: datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)
    if d and t else (d if d else None))
lab["_coll_dt"]   = dt_col(lab, "Collection Date").combine(dt_col(lab, "Collection Time"),
    lambda d, t: datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)
    if d and t else (d if d else None))
lab["_result_dt"] = dt_col(lab, "Result Date").combine(dt_col(lab, "Result Time"),
    lambda d, t: datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)
    if d and t else (d if d else None))
lab["_arrival_dt"] = lab["Arrival Date_dt"]

chrono_rows.append(check_order(lab, "_arrival_dt", "_order_dt",  "Arrival ≤ Lab Order Date",   "Lab"))
chrono_rows.append(check_order(lab, "_order_dt",   "_coll_dt",   "Lab Order ≤ Collection",     "Lab"))
chrono_rows.append(check_order(lab, "_coll_dt",    "_result_dt", "Collection ≤ Result",        "Lab"))
chrono_rows.append(check_order(lab, "_arrival_dt", "_result_dt", "Arrival ≤ Lab Result",       "Lab"))
lab["_fct_date_dt"]     = lab["FCT_dt"].apply(_to_date_midnight)
lab["_order_date_only"] = lab["_order_dt"].apply(_to_date_midnight)
chrono_rows.append(check_order(lab, "_fct_date_dt", "_order_date_only", "FCT (Triage) date ≤ Lab Order date", "Lab"))

df_chrono = pd.DataFrame(chrono_rows)


# ---------------------------------------------------------------------------
# Step 4 — Coverage gaps
# ---------------------------------------------------------------------------
print("Computing coverage...")

ap_csns  = set(df_ap ["CSN_key"].dropna())
img_csns = set(df_img["CSN_key"].dropna())
lab_csns = set(df_lab["CSN_key"].dropna())

all3          = ap_csns  & img_csns & lab_csns
ap_img_only   = (ap_csns  & img_csns) - lab_csns
ap_lab_only   = (ap_csns  & lab_csns) - img_csns
img_lab_noap  = (img_csns & lab_csns) - ap_csns
ap_only       = ap_csns  - img_csns - lab_csns
img_only      = img_csns - lab_csns  - ap_csns
lab_only      = lab_csns - ap_csns   - img_csns

coverage_rows = [
    {"Group": "All 3 (All Pt + Imaging + Lab)",           "Unique CSNs": len(all3)},
    {"Group": "All Pt + Imaging only (no Lab)",           "Unique CSNs": len(ap_img_only)},
    {"Group": "All Pt + Lab only (no Imaging)",           "Unique CSNs": len(ap_lab_only)},
    {"Group": "Imaging + Lab only (no All Pt — spillover)","Unique CSNs": len(img_lab_noap)},
    {"Group": "All Pt only (no Imaging, no Lab)",         "Unique CSNs": len(ap_only)},
    {"Group": "Imaging only (no Lab, no All Pt)",         "Unique CSNs": len(img_only)},
    {"Group": "Lab only (no Imaging, no All Pt)",         "Unique CSNs": len(lab_only)},
    {"Group": "TOTAL (union)",                            "Unique CSNs": len(ap_csns | img_csns | lab_csns)},
]
df_coverage = pd.DataFrame(coverage_rows)

# Detail sheet: All Pt patients with NO CT imaging and NO lab
no_orders_csns = ap_only
df_no_orders = df_ap[df_ap["CSN_key"].isin(no_orders_csns)][
    ["month", "CSN", "Triage Acuity", "Age Group", "Sex",
     "ED Disposition", "Chief Complaint", "Primary ED Dx"]
].sort_values(["month", "CSN"]).reset_index(drop=True)

# Detail sheet: spillover CSNs (in CT orders but not All Pt)
spillover_csns = (img_csns | lab_csns) - ap_csns
spill_img = df_img[df_img["CSN_key"].isin(spillover_csns)].groupby("CSN_key").agg(
    month=("month", "first"),
    earliest_order_date=("Order Date", "min"),
    arrival_date_img=("Arrival Date", "first"),
    n_img_orders=("Order ID", "count"),
).reset_index()
spill_lab = df_lab[df_lab["CSN_key"].isin(spillover_csns)].groupby("CSN_key").agg(
    n_lab_orders=("Order ID", "count"),
).reset_index()
df_spillover = spill_img.merge(spill_lab, on="CSN_key", how="left")
df_spillover["n_lab_orders"] = df_spillover["n_lab_orders"].fillna(0).astype(int)
df_spillover = df_spillover.rename(columns={"CSN_key": "CSN"}).sort_values(["month", "CSN"])

# Detail sheet: All Pt + Imaging only (no lab) — listing
no_lab_csns = ap_img_only
df_img_no_lab = df_img[df_img["CSN_key"].isin(no_lab_csns)].groupby("CSN_key").agg(
    month=("month", "first"),
    n_img_orders=("Order ID", "count"),
    acuity=("Acuity Abbr", "first"),
    ed_disposition=("ED Disposition", "first"),
).reset_index().rename(columns={"CSN_key": "CSN"}).sort_values(["month", "CSN"])


# ---------------------------------------------------------------------------
# Step 5 — Write output
# ---------------------------------------------------------------------------
print("Writing merged_analysis.xlsx...")

out_path = os.path.join(OUT, "merged_analysis.xlsx")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df_agreement.to_excel(writer,    sheet_name="Duplicate Field Agreement", index=False)
    df_chrono.to_excel(writer,       sheet_name="Chronology Checks",         index=False)
    df_coverage.to_excel(writer,     sheet_name="Coverage Summary",          index=False)
    df_no_orders.to_excel(writer,    sheet_name="All Pt Only — No Orders",   index=False)
    df_spillover.to_excel(writer,    sheet_name="Spillover CSNs",            index=False)
    df_img_no_lab.to_excel(writer,   sheet_name="Imaging — No Lab",         index=False)

size_kb = os.path.getsize(out_path) // 1024
print(f"  -> merged_analysis.xlsx  ({size_kb} KB)")

# Print summary to terminal
print("\n=== Coverage Summary ===")
for _, r in df_coverage.iterrows():
    print(f"  {r['Group']:<45} {r['Unique CSNs']:>6,}")

print("\n=== Duplicate Field Agreement (Disagree % > 0) ===")
notable = df_agreement[df_agreement["Disagree %"].notna() & (df_agreement["Disagree %"] > 0)]
for _, r in notable.iterrows():
    print(f"  {r['Field']:<25} {r['Source Pair']:<25} {r['Disagree %']:>6.1f}%  (n={r['N Disagree']})")

print("\n=== Chronology Checks (Fail % > 0) ===")
fails = df_chrono[df_chrono["Fail %"].notna() & (df_chrono["Fail %"] > 0)]
for _, r in fails.iterrows():
    print(f"  [{r['Source']:<8}] {r['Check']:<35} {r['Fail %']:>6.1f}%  (n={r['N Fail']})")

print("\nDone.")
