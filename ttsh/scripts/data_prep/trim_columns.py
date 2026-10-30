"""
trim_columns.py
Drops all columns except an explicit keep-list (CSN always kept) from:
  - merged/all_pt_combined.xlsx  (All Patients sheet)  -> merged/all_pt_cleaned.xlsx
  - merged/ct_orders_cleaned.xlsx (Imaging + Lab sheets) -> overwritten in place

Lab uses the same keep-list as Imaging; columns not present in a given
sheet are silently skipped (e.g. Imaging-only fields like Exam Start/End
don't exist in Lab).
"""

import os
import openpyxl
import pandas as pd

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")

ALL_PT_IN  = os.path.join(MERGED, "all_pt_combined.xlsx")
ALL_PT_OUT = os.path.join(MERGED, "all_pt_cleaned.xlsx")
CT_PATH    = os.path.join(MERGED, "ct_orders_cleaned.xlsx")


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


def keep_columns(df, keep_list):
    keep = ["CSN"] + [c for c in keep_list if c != "CSN"]
    present = [c for c in keep if c in df.columns]
    missing = [c for c in keep if c not in df.columns]
    return df[present], missing


ALL_PT_KEEP = [
    "CSN", "Age at Visit", "Arrival Date", "Arrival Time", "Arrival Method",
    "Chief Complaint", "KIV Uptriage", "Primary ED Dx", "ED Dx",
    "ID Template", "Triage Acuity", "Consult Acuity", "ED Disposition",
    "Reatt Same ED", "Reatt Reason", "72h Reatt NGEMR ED",
    "72h Reatt NGEMR ED then Admitted", "Referral Source",
    "Referral Source Type", "Ref Src Add", "HD/ICU from GW within 24H",
    "First ED Care Area", "Primary Care Area", "Age Group",
    "Classification Type", "CFS (ED)",
    "FCT", "ED Primary Dx ICD10", "Admit Dispo Date",
    "Admit Dispo Time", "Sex", "BP first",
    "First BP Flowsheet & Timestamp", "BP before Consult",
]

# Imaging keep-list (source column is "Nationality " with a trailing space).
IMG_KEEP = [
    "CSN", "Nationality ", "Acuity Abbr", "Order ID", "Order Description",
    "Ordering Question Responses", "Order Type", "Procedure Category",
    "Result Acuity", "Order Status", "Exam Start Date", "Exam Start Time",
    "Exam End Date", "Exam End Time", "Result Date", "Result Time",
    "Abnormal", "Critical Result", "Parent Order Date", "Parent Order Time",
]

# Lab keep-list (explicit, distinct from Imaging).
LAB_KEEP = [
    "CSN", "Acuity Abbr", "Order ID", "Order Description",
    "Ordering Question Responses", "Order Type", "Procedure Category",
    "Order Status", "Collection Date", "Collection Time", "Received Date",
    "Received Time", "Result Date", "Result Time", "Abnormal",
    "Critical Result", "Last Lab Test Results", "Parent Order Date",
    "Parent Order Time",
]


# ---------------------------------------------------------------------------
# All Pt
# ---------------------------------------------------------------------------
print("Loading merged/all_pt_combined.xlsx ...")
df_ap = load_sheet(ALL_PT_IN, "All Patients")
print(f"  All Pt: {len(df_ap):,} rows, {len(df_ap.columns)} columns")

df_ap_trimmed, missing = keep_columns(df_ap, ALL_PT_KEEP)
print(f"  -> keeping {len(df_ap_trimmed.columns)} columns")
if missing:
    print(f"  NOTE: {len(missing)} requested columns not found in source: {missing}")

print("Writing merged/all_pt_cleaned.xlsx...")
with pd.ExcelWriter(ALL_PT_OUT, engine="openpyxl") as writer:
    df_ap_trimmed.to_excel(writer, sheet_name="All Patients", index=False)
print(f"  -> all_pt_cleaned.xlsx  [{os.path.getsize(ALL_PT_OUT)//1024} KB]")


# ---------------------------------------------------------------------------
# CT Orders (Imaging + Lab) — separate keep-lists, overwrite ct_orders_cleaned.xlsx
# ---------------------------------------------------------------------------
print("\nLoading merged/ct_orders_cleaned.xlsx ...")
df_img = load_sheet(CT_PATH, "Imaging")
df_lab = load_sheet(CT_PATH, "Lab")
print(f"  Imaging: {len(df_img):,} rows, {len(df_img.columns)} columns")
print(f"  Lab:     {len(df_lab):,} rows, {len(df_lab.columns)} columns")

df_img_trimmed, img_missing = keep_columns(df_img, IMG_KEEP)
df_lab_trimmed, lab_missing = keep_columns(df_lab, LAB_KEEP)

print(f"  Imaging -> keeping {len(df_img_trimmed.columns)} columns")
if img_missing:
    print(f"  NOTE: {len(img_missing)} requested columns not found in Imaging: {img_missing}")
print(f"  Lab -> keeping {len(df_lab_trimmed.columns)} columns")
if lab_missing:
    print(f"  NOTE: {len(lab_missing)} requested columns not found in Lab: {lab_missing}")

print("Writing merged/ct_orders_cleaned.xlsx (overwrite)...")
with pd.ExcelWriter(CT_PATH, engine="openpyxl") as writer:
    df_img_trimmed.to_excel(writer, sheet_name="Imaging", index=False)
    df_lab_trimmed.to_excel(writer, sheet_name="Lab", index=False)
print(f"  -> ct_orders_cleaned.xlsx  [{os.path.getsize(CT_PATH)//1024} KB]")

print("\nDone.")
