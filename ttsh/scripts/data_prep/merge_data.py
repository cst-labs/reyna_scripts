"""
merge_data.py
Decrypts new monthly files, merges all months into two combined files:
  - merged/all_pt_combined.xlsx       (one sheet: All Patients)
  - merged/ct_orders_combined.xlsx    (two sheets: Imaging, Lab)
Logs all structural anomalies to merged/merge_log.txt
"""

import os
import math
from datetime import datetime, timedelta, date, time as time_type

import openpyxl
import pandas as pd

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA   = os.path.join(BASE, "data")
OUT    = os.path.join(BASE, "merged")
os.makedirs(OUT, exist_ok=True)

LOG_LINES = []

def log(msg):
    print(msg)
    LOG_LINES.append(msg)


# ---------------------------------------------------------------------------
# File manifest  (month_label, allpt_file/sheet, imaging_file/sheet, lab_sheet)
# ---------------------------------------------------------------------------
MONTHS = [
    # (label,       allpt_fname,                          allpt_sheet,
    #               ct_fname,                             img_sheet,  lab_sheet)
    ("Nov 2025", "2025 11 All Pt (decrypted).xlsx",    "Sheet2",
                 "2025 11 CT Orders (decrypted).xlsx", "Nov Imaging", "Nov Lab"),
    ("Dec 2025", "2025 12 All Pt (decrypted).xlsx",    "Sheet2",
                 "2025 12 CT Orders (decrypted).xlsx", "scans",       "lab"),
    ("Jan 2026", "2026 1 All Pt (decrypted).xlsx",     "Sheet2",
                 "2026 1 CT Orders (decrypted).xlsx",  "Jan26",       "lab"),
    ("Feb 2026", "2026 2 All Pt (decrypted).xlsx",     "Sheet2",
                 "2026 2 CT Orders (decrypted).xlsx",  "Feb26",       "lab"),
    ("Mar 2026", "2026 3 All Pt (decrypted).xlsx",     "Sheet2",
                 "2026 3 CT Orders (decrypted).xlsx",  "Mar26",       "lab"),
    ("Apr 2026", "2026 4 All Pt (decrypted).xlsx",     "Sheet2",
                 "2026 4 CT Orders (decrypted).xlsx",  "Apr26",       "lab"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wb_to_df(path, sheet):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return pd.DataFrame()
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    seen = {}
    unique_headers = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            unique_headers.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            unique_headers.append(h)
    return pd.DataFrame(rows[1:], columns=unique_headers)


def csn_set(df):
    return set(
        df["CSN"].dropna().apply(str).str.strip()
        .loc[lambda s: s.str.match(r"^\d+$")]
    )


# Reference column sets (Nov 2025 as baseline)
REF_ALLPT = None
REF_IMG   = None
REF_LAB   = None

# Dec 2025 imaging has analyst-added computed columns — strip them
DEC_IMG_DROP = [
    "[A] Parent Order date time", "[B] Exam start date time",
    "[C] Exam End date time",     "[D] Result date time",
    "x (A to D) - 14.96 (2024)", "A to D (h)",
    "x (B to C) =  0.78 (2024)", "B to C (h)",
    "x (A to B) = 3.53 (2024)",  "A to B (h)",
    "Unique CSN (1 visit = 1 CSN)", "helper",
]

# Columns present in Apr–Mar imaging but not Nov (added from Apr onwards)
EXTRA_IMG_COLS = [
    "Admit Dispo Date", "Admit Dispo Time",
    "Admit EDTC/U Date", "Admit EDTC/U Time",
    "EDTC/U IP Time", "EDTC/U IP Date",
]


# ---------------------------------------------------------------------------
# Load, validate, and tag each month
# ---------------------------------------------------------------------------
all_allpt  = []
all_img    = []
all_lab    = []

for (label, ap_fname, ap_sheet, ct_fname, img_sheet, lab_sheet) in MONTHS:
    ap_path  = os.path.join(DATA, ap_fname)
    ct_path  = os.path.join(DATA, ct_fname)

    log(f"\n{'='*60}")
    log(f"  Loading {label}")
    log(f"{'='*60}")

    # ---- All Pt ----
    df_ap = wb_to_df(ap_path, ap_sheet)
    df_ap["month"] = label
    log(f"  All Pt:  {len(df_ap):>5,} rows  |  {len(df_ap.columns)} cols")

    if REF_ALLPT is None:
        REF_ALLPT = [c for c in df_ap.columns if c != "month"]
    else:
        raw_cols = [c for c in df_ap.columns if c != "month"]
        added   = [c for c in raw_cols    if c not in REF_ALLPT]
        removed = [c for c in REF_ALLPT   if c not in raw_cols]
        if added:
            log(f"  [ANOMALY] All Pt {label}: unexpected columns added: {added}")
        if removed:
            log(f"  [ANOMALY] All Pt {label}: expected columns missing: {removed}")

    # Drop stray embedded header rows
    stray_ap = df_ap["CSN"].astype(str).str.strip() == "CSN"
    if stray_ap.any():
        log(f"  [ANOMALY] All Pt {label}: {stray_ap.sum()} stray embedded header row(s) found and removed")
        df_ap = df_ap[~stray_ap].reset_index(drop=True)

    # Check for duplicate CSNs within the month
    dup_ap = df_ap["CSN"].dropna()[df_ap["CSN"].duplicated(keep=False)]
    if len(dup_ap):
        log(f"  [ANOMALY] All Pt {label}: {dup_ap.nunique()} CSNs appear more than once ({len(dup_ap)} rows)")

    all_allpt.append(df_ap)

    # ---- CT Imaging ----
    df_img = wb_to_df(ct_path, img_sheet)
    df_img["month"] = label

    # Strip Dec 2025 analyst-added computed columns
    dec_dropped = [c for c in DEC_IMG_DROP if c in df_img.columns]
    if dec_dropped:
        log(f"  [ANOMALY] CT Imaging {label} (sheet '{img_sheet}'): dropping {len(dec_dropped)} analyst-added computed columns: {dec_dropped}")
        df_img = df_img.drop(columns=dec_dropped)

    # Dec 2025 is also missing Arrival Date/Time — note it
    for col in ["Arrival Date", "Arrival Time"]:
        if col not in df_img.columns:
            log(f"  [ANOMALY] CT Imaging {label}: column '{col}' is absent (not present in this month's export)")

    # Drop stray embedded header rows (row where CSN cell == "CSN")
    stray_img = df_img["CSN"].astype(str).str.strip() == "CSN"
    if stray_img.any():
        log(f"  [ANOMALY] CT Imaging {label}: {stray_img.sum()} stray embedded header row(s) found and removed")
        df_img = df_img[~stray_img].reset_index(drop=True)

    log(f"  CT Img:  {len(df_img):>5,} rows  |  {len(df_img.columns)} cols  (sheet: {img_sheet})")

    if REF_IMG is None:
        REF_IMG = [c for c in df_img.columns if c != "month"]
    else:
        raw_cols = [c for c in df_img.columns if c != "month"]
        added   = [c for c in raw_cols  if c not in REF_IMG]
        removed = [c for c in REF_IMG   if c not in raw_cols]
        if added:
            log(f"  [NOTE]    CT Imaging {label}: extra columns (present from Apr 2026 onwards): {added}")
        if removed:
            log(f"  [ANOMALY] CT Imaging {label}: expected columns missing: {removed}")

    all_img.append(df_img)

    # ---- CT Lab ----
    df_lab = wb_to_df(ct_path, lab_sheet)
    df_lab["month"] = label

    # Drop stray embedded header rows
    stray_lab = df_lab["CSN"].astype(str).str.strip() == "CSN"
    if stray_lab.any():
        log(f"  [ANOMALY] CT Lab {label}: {stray_lab.sum()} stray embedded header row(s) found and removed")
        df_lab = df_lab[~stray_lab].reset_index(drop=True)

    log(f"  CT Lab:  {len(df_lab):>5,} rows  |  {len(df_lab.columns)} cols")

    if REF_LAB is None:
        REF_LAB = [c for c in df_lab.columns if c != "month"]
    else:
        raw_cols = [c for c in df_lab.columns if c != "month"]
        added   = [c for c in raw_cols  if c not in REF_LAB]
        removed = [c for c in REF_LAB   if c not in raw_cols]
        if added:
            log(f"  [ANOMALY] CT Lab {label}: unexpected columns added: {added}")
        if removed:
            log(f"  [NOTE]    CT Lab {label}: column absent (present in some months only): {removed}")

    all_lab.append(df_lab)

    # ---- Cross-table CSN checks ----
    ap_csns  = csn_set(df_ap)
    img_csns = csn_set(df_img)
    lab_csns = csn_set(df_lab)

    img_not_ap = img_csns - ap_csns
    lab_not_ap = lab_csns - ap_csns
    ap_not_img = ap_csns  - img_csns

    if img_not_ap:
        log(f"  [NOTE]    CT Imaging {label}: {len(img_not_ap)} CSNs not in All Pt (likely prior-month spillover)")
    if lab_not_ap:
        log(f"  [NOTE]    CT Lab {label}: {len(lab_not_ap)} CSNs not in All Pt")
    if ap_not_img:
        log(f"  [ANOMALY] All Pt {label}: {len(ap_not_img)} CSNs have no CT imaging order")


# ---------------------------------------------------------------------------
# Concatenate
# ---------------------------------------------------------------------------
log(f"\n{'='*60}")
log("  Concatenating all months")
log(f"{'='*60}")

combined_ap  = pd.concat(all_allpt, ignore_index=True)
combined_img = pd.concat(all_img,   ignore_index=True)
combined_lab = pd.concat(all_lab,   ignore_index=True)

log(f"  All Pt combined:  {len(combined_ap):>6,} rows  |  {combined_ap['month'].nunique()} months")
log(f"  CT Img combined:  {len(combined_img):>6,} rows  |  {combined_img['month'].nunique()} months")
log(f"  CT Lab combined:  {len(combined_lab):>6,} rows  |  {combined_lab['month'].nunique()} months")

# Duplicate CSN check across the full combined dataset (same CSN in multiple months = expected; same CSN twice within same month = anomaly)
for label_str, df, id_cols in [
    ("All Pt combined",  combined_ap,  ["CSN", "month"]),
    ("CT Img combined",  combined_img, ["CSN", "Order ID", "month"]),
    ("CT Lab combined",  combined_lab, ["CSN", "Order ID", "month"]),
]:
    existing = [c for c in id_cols if c in df.columns]
    dups = df[df.duplicated(subset=existing, keep=False)]
    if len(dups):
        log(f"  [ANOMALY] {label_str}: {len(dups)} duplicate rows on {existing}")

# All Pt: check for any CSN appearing in more than one month (patients who re-attended)
ap_csn_months = combined_ap.groupby("CSN")["month"].nunique()
multi_month_ap = ap_csn_months[ap_csn_months > 1]
if len(multi_month_ap):
    log(f"  [NOTE]    All Pt: {len(multi_month_ap)} CSNs appear in more than one month (re-attenders or duplicate export)")


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------
log(f"\n{'='*60}")
log("  Writing output files")
log(f"{'='*60}")

# Put month column first
def reorder_month_first(df):
    cols = ["month"] + [c for c in df.columns if c != "month"]
    return df[cols]

combined_ap  = reorder_month_first(combined_ap)
combined_img = reorder_month_first(combined_img)
combined_lab = reorder_month_first(combined_lab)

allpt_path = os.path.join(OUT, "all_pt_combined.xlsx")
with pd.ExcelWriter(allpt_path, engine="openpyxl") as writer:
    combined_ap.to_excel(writer, sheet_name="All Patients", index=False)
log(f"  -> all_pt_combined.xlsx  ({len(combined_ap):,} rows)")

ct_path = os.path.join(OUT, "ct_orders_combined.xlsx")
with pd.ExcelWriter(ct_path, engine="openpyxl") as writer:
    combined_img.to_excel(writer, sheet_name="Imaging", index=False)
    combined_lab.to_excel(writer, sheet_name="Lab", index=False)
log(f"  -> ct_orders_combined.xlsx  (Imaging: {len(combined_img):,} rows  |  Lab: {len(combined_lab):,} rows)")

# Row counts by month summary
log(f"\n{'='*60}")
log("  Row counts by month")
log(f"{'='*60}")
log(f"  {'Month':<12} {'All Pt':>8} {'CT Img':>8} {'CT Lab':>8}")
log(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
for m in combined_ap["month"].unique():
    n_ap  = (combined_ap["month"]  == m).sum()
    n_img = (combined_img["month"] == m).sum()
    n_lab = (combined_lab["month"] == m).sum()
    log(f"  {m:<12} {n_ap:>8,} {n_img:>8,} {n_lab:>8,}")
n_ap  = len(combined_ap)
n_img = len(combined_img)
n_lab = len(combined_lab)
log(f"  {'TOTAL':<12} {n_ap:>8,} {n_img:>8,} {n_lab:>8,}")

# Write log
log_path = os.path.join(OUT, "merge_log.txt")
with open(log_path, "w") as f:
    f.write("\n".join(LOG_LINES))
print(f"\nLog written to: {log_path}")
