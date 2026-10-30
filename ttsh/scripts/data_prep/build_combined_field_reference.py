"""
build_combined_field_reference.py
Generates field_reference_combined.xlsx with three sheets:
  - All Pt (Combined)
  - CT Imaging (Combined)
  - CT Lab (Combined)

Each sheet contains: Field Name, Example Value, Data Type, Field Type,
Range / Categories, Null % (overall), Null % by month, Notes
(including cross-month inconsistencies and data quality observations).
"""

import os
import pandas as pd
import openpyxl

MERGED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "merged")
OUT    = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "analysis")

MONTH_ORDER = ["Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026"]

# ---------------------------------------------------------------------------
# Known cross-month inconsistencies (col -> note string)
# ---------------------------------------------------------------------------
ALLPT_NOTES = {
    "match":       "Identical to CSN in all rows across all months — redundant duplicate column.",
    "Age at Visit": "Stored as string e.g. ' 73 y.o.' — numeric extraction requires regex \\d+.",
    "FCT":         "First Contact Time stored as formatted string 'DD/MM/YYYY HH:MM:SS AM/PM' — requires strptime parsing.",
    "BP last":     "Duplicate column header in source files (indices 77 and 78); deduplicated as 'BP last' and 'BP last_1'.",
    "BP last_1":   "Second instance of 'BP last' (original index 78); contains value with timestamp suffix.",
    "Triage Acuity":  "P1=most urgent, P2=urgent, P3=semi-urgent, P4=non-urgent.",
    "Consult Acuity": "Re-assessed acuity at physician contact — not known at triage; distinct from Triage Acuity.",
    "CFS (ED)":    "Clinical Frailty Scale; may be assessed at triage for elderly patients.",
    "Reatt Same ED": "Known if patient is a prior attender; lookup possible at triage.",
    "Arrival Date": "Apr 2026 export stores this as raw Excel OLE serial number (integer) rather than a parsed date — requires conversion: datetime(1899,12,30) + timedelta(days=serial).",
    "Arrival Time": "Apr 2026 export stores this as raw Excel OLE decimal fraction — requires conversion to time.",
    "ED Dispo Last Date": "Apr 2026 export: raw OLE serial. Same conversion needed as Arrival Date.",
    "ED Dispo Last Time": "Apr 2026 export: raw OLE fraction. Same conversion needed as Arrival Time.",
    "Depart Date":  "Apr 2026 export: raw OLE serial.",
    "Depart Time":  "Apr 2026 export: raw OLE fraction.",
    "Admit Dispo Date": "Apr 2026 export: raw OLE serial.",
    "Admit Dispo Time": "Apr 2026 export: raw OLE fraction.",
    "Bed Assigned Date": "Apr 2026 export: raw OLE serial.",
    "Bed Assigned Time": "Apr 2026 export: raw OLE fraction.",
    "Hospital Discharge Date": "Apr 2026 export: raw OLE serial.",
    "Hospital Discharge Time": "Apr 2026 export: raw OLE fraction.",
}

IMG_NOTES = {
    "Arrival Date":      "ABSENT in Dec 2025 — column not exported for that month; null for all Dec 2025 imaging rows.",
    "Arrival Time":      "ABSENT in Dec 2025 — column not exported for that month; null for all Dec 2025 imaging rows.",
    "Admit Dispo Date":  "Present from Jan 2026 onwards only (absent in Nov and Dec 2025); null for earlier months.",
    "Admit Dispo Time":  "Present from Jan 2026 onwards only (absent in Nov and Dec 2025); null for earlier months.",
    "Admit EDTC/U Date": "Present from Jan 2026 onwards only (absent in Nov and Dec 2025); null for earlier months.",
    "Admit EDTC/U Time": "Present from Jan 2026 onwards only (absent in Nov and Dec 2025); null for earlier months.",
    "EDTC/U IP Date":    "Present from Jan 2026 onwards only (absent in Nov and Dec 2025); null for earlier months.",
    "EDTC/U IP Time":    "Present from Jan 2026 onwards only (absent in Nov and Dec 2025); null for earlier months.",
    "Ordering Question Responses": "Scan-level CT priority (P1/P2/P3) embedded in free text — requires regex parsing. ~77% null in lab sheet.",
    "Acuity Abbr":       "Patient-level ED triage acuity (P1-P3); distinct from CT scan priority in Ordering Question Responses.",
    "Order Completion Duration (mins)": "Pre-calculated field: order placed → order completed (mins). ~99-100% null across all months.",
    "EDTC/U Order":      "Flag: 1 = order placed while patient in EDTC/EDTU observation area.",
    "Result Acuity":     "Radiologist's assessment of result severity (Abnormal, Normal, etc.). 100% null in lab sheet.",
    "Last Lab Test Results": "Always null in imaging sheet; contains actual lab result text only in lab sheet.",
    "Received Date":     "Always null in imaging sheet; populated in lab sheet (specimen receipt timestamp).",
    "Received Time":     "Always null in imaging sheet; populated in lab sheet.",
    "Scan Schedule Time": "100% null in both imaging and lab sheets across all months — field present but never populated.",
    "Exam Start Date":   "Recorded to minute precision only (no seconds) — Order Time has seconds precision. Floor Order Time to minute before computing order-to-exam-start duration to avoid artefactual negatives.",
    "Exam Start Time":   "Recorded to minute precision only. See Exam Start Date note.",
    "CSN":               "One stray embedded header row per month (Dec 2025–Apr 2026) had 'CSN' as the cell value — these 5 rows were removed during merge.",
}

LAB_NOTES = {
    "match":        "Identical to CSN in all rows — redundant column. ABSENT in Mar 2026 and Apr 2026 exports.",
    "ED/OP Case#":  "ED/outpatient encounter number; may be None for inpatient-only orders.",
    "IP Case#":     "Inpatient encounter number; may be None for ED-only orders.",
    "Ordering Question Responses": "~77% null in lab sheet; occasionally populated for special lab requests.",
    "Result Acuity": "100% null in lab sheet — radiologist severity flag is not assigned to lab results.",
    "Last Lab Test Results": "Free-text blob containing all component results (e.g. 'Sodium: 139 mmol/L\\nPotassium: 3.5 mmol/L'). ~4% null.",
    "Received Date": "Lab specimen receipt timestamp — always null in imaging sheet, populated here.",
    "Received Time": "Lab specimen receipt timestamp — always null in imaging sheet, populated here.",
    "Order Completion Duration (mins)": "100% null across all months in lab sheet.",
    "Linked Chargeable": "More consistently filled for labs (~2% null) vs imaging (~37% null).",
    "Abnormal":      "Less consistently flagged for labs (~60% null) vs imaging (~10% null).",
    "Scan Schedule Time": "100% null — imaging-only field included in lab template but never populated.",
}

# ---------------------------------------------------------------------------
# Load merged files
# ---------------------------------------------------------------------------
print("Loading merged files...")

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

allpt_path = os.path.join(MERGED, "all_pt_combined.xlsx")
ct_path    = os.path.join(MERGED, "ct_orders_combined.xlsx")

df_ap  = load_merged(allpt_path, "All Patients")
df_img = load_merged(ct_path, "Imaging")
df_lab = load_merged(ct_path, "Lab")

print(f"  All Pt:  {len(df_ap):,} rows")
print(f"  Imaging: {len(df_img):,} rows")
print(f"  Lab:     {len(df_lab):,} rows")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_example(df, col):
    vals = df[col].dropna()
    if vals.empty:
        return ""
    return str(vals.iloc[0])[:80]


def classify_field(series):
    non_null = series.dropna()
    if len(non_null) == 0:
        return "Unknown", ""
    if pd.api.types.is_datetime64_any_dtype(series):
        try:
            return "DateTime", f"{non_null.min().date()} to {non_null.max().date()}"
        except Exception:
            return "DateTime", ""
    if pd.api.types.is_numeric_dtype(series):
        uniq = set(non_null.unique())
        if uniq <= {0, 1}:
            return "Binary (0/1)", "0, 1"
        lo, hi, med = non_null.min(), non_null.max(), non_null.median()
        return "Numerical", f"min={lo:.1f}, max={hi:.1f}, median={med:.1f}"
    str_vals = non_null.astype(str).str.strip()
    uniq_str = set(str_vals.unique())
    if uniq_str <= {"0", "1"}:      return "Binary (0/1)",    "0, 1"
    if uniq_str <= {"Yes", "No"}:   return "Binary (Yes/No)", "No, Yes"
    if uniq_str <= {"True","False"}: return "Binary (True/False)", "False, True"
    n_unique = str_vals.nunique()
    if n_unique <= 30:
        return "Categorical", ", ".join(sorted(str_vals.unique()))[:300]
    return "Text / ID", f"{n_unique} unique values"


def null_pct(series):
    if len(series) == 0:
        return 100.0
    return round(series.isna().mean() * 100, 1)


def build_ref(df, notes_map, skip_cols=("month",)):
    """Build field reference rows for df, one row per column."""
    rows = []
    months_present = [m for m in MONTH_ORDER if m in df["month"].unique()]

    for col in df.columns:
        if col in skip_cols:
            continue

        ex    = get_example(df, col)
        dtype = str(df[col].dtype)
        ftype, frange = classify_field(df[col])
        null_overall  = null_pct(df[col])

        # Null % per month
        month_nulls = {}
        for m in months_present:
            sub = df.loc[df["month"] == m, col]
            month_nulls[m] = null_pct(sub)

        # Data quality flags
        dq_flags = []
        if null_overall == 100.0:
            dq_flags.append("100% null across all months")
        elif null_overall > 90:
            dq_flags.append(f"{null_overall}% null overall")

        # Flag months where null rate differs substantially from the others
        if len(months_present) > 1:
            mn_vals = list(month_nulls.values())
            mn_mean = sum(mn_vals) / len(mn_vals)
            outliers = [
                f"{m}: {month_nulls[m]}%"
                for m in months_present
                if abs(month_nulls[m] - mn_mean) > 25
            ]
            if outliers:
                dq_flags.append("Null rate varies by month — " + ", ".join(outliers))

        dq_note = "; ".join(dq_flags)
        field_note = notes_map.get(col, "")

        row = {
            "Field Name":       col,
            "Example Value":    ex,
            "Data Type":        dtype,
            "Field Type":       ftype,
            "Range / Categories": frange,
            "Null % (Overall)": null_overall,
        }
        for m in months_present:
            row[f"Null % {m}"] = month_nulls[m]

        row["Data Quality Notes"] = dq_note
        row["Notes"] = field_note
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build the three sheets
# ---------------------------------------------------------------------------
print("Building field reference sheets...")

df_ref_ap  = build_ref(df_ap,  ALLPT_NOTES)
df_ref_img = build_ref(df_img, IMG_NOTES)
df_ref_lab = build_ref(df_lab, LAB_NOTES)

print(f"  All Pt:  {len(df_ref_ap)} fields")
print(f"  Imaging: {len(df_ref_img)} fields")
print(f"  Lab:     {len(df_ref_lab)} fields")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
out_path = os.path.join(OUT, "field_reference_combined.xlsx")
print(f"Writing {out_path}...")

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df_ref_ap.to_excel(writer,  sheet_name="All Pt (Combined)",      index=False)
    df_ref_img.to_excel(writer, sheet_name="CT Imaging (Combined)",  index=False)
    df_ref_lab.to_excel(writer, sheet_name="CT Lab (Combined)",      index=False)

size_kb = os.path.getsize(out_path) // 1024
print(f"  -> field_reference_combined.xlsx  ({size_kb} KB)")
print("Done.")
