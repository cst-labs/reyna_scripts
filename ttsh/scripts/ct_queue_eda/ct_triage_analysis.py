"""
CT Triage Analysis Script
TTSH ED — Nov 2025 & Apr 2026
Outputs: field_reference.xlsx, summary_stats.xlsx, ct_turnaround.xlsx, data_quality.xlsx
"""

import re
import os
from datetime import datetime, timedelta, date, time

import pandas as pd
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

DATA = os.path.join(BASE, "data")
FILES = {
    "allpt_nov": os.path.join(DATA, "2025 11 All Pt (decrypted).xlsx"),
    "allpt_apr": os.path.join(DATA, "2026 4 All Pt (decrypted).xlsx"),
    "ct_nov":    os.path.join(DATA, "2025 11 CT Orders (decrypted).xlsx"),
    "ct_apr":    os.path.join(DATA, "2026 4 CT Orders (decrypted).xlsx"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def excel_serial_to_date(val):
    if val is None:
        return None
    if isinstance(val, (datetime, date)):
        return val
    try:
        f = float(val)
        import math
        if math.isnan(f) or math.isinf(f):
            return None
        return datetime(1899, 12, 30) + timedelta(days=f)
    except Exception:
        return None


def excel_serial_to_time(val):
    if val is None:
        return None
    if isinstance(val, time):
        return val
    if isinstance(val, datetime):
        return val.time()
    try:
        frac = float(val) % 1
        total_secs = round(frac * 86400)
        return (datetime(1900, 1, 1) + timedelta(seconds=total_secs)).time()
    except Exception:
        return None


def parse_fct(val):
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.strptime(str(val).strip(), "%d/%m/%Y %I:%M:%S %p")
    except Exception:
        return None


def extract_age(val):
    if val is None:
        return None
    m = re.search(r"\d+", str(val))
    return int(m.group()) if m else None


def parse_scan_priority(val):
    if not val:
        return None
    m = re.search(r"(P[123])", str(val))
    return m.group(1) if m else None


def to_datetime_combined(d, t):
    """Combine date and time values (may be raw Excel serials or Python objects) into a datetime."""
    if d is None:
        return None
    d_conv = excel_serial_to_date(d)
    if d_conv is None:
        return None
    # Normalise to a date object regardless of whether we got datetime or date
    try:
        if isinstance(d_conv, datetime):
            d_only = d_conv.date()
        elif isinstance(d_conv, date):
            d_only = d_conv
        else:
            return None
    except Exception:
        return None
    try:
        if t is None:
            return datetime(d_only.year, d_only.month, d_only.day)
        t_conv = excel_serial_to_time(t)
        if t_conv is None:
            return datetime(d_only.year, d_only.month, d_only.day)
        return datetime(d_only.year, d_only.month, d_only.day,
                        t_conv.hour, t_conv.minute, t_conv.second)
    except Exception:
        return None


def minutes_between(dt1, dt2):
    if dt1 is None or dt2 is None:
        return None
    delta = (dt2 - dt1).total_seconds() / 60
    # Reject negative or implausibly large values (>24h = 1440 mins)
    if delta < 0 or delta > 1440:
        return None
    return round(delta, 1)


def wb_to_df(path, sheet):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return pd.DataFrame()
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    # Deduplicate headers
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


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading All Pt files...")
allpt_nov = wb_to_df(FILES["allpt_nov"], "Sheet2")
allpt_apr = wb_to_df(FILES["allpt_apr"], "Sheet2")

allpt_nov["month"] = "Nov 2025"
allpt_apr["month"] = "Apr 2026"

# Fix 2026 date/time serial numbers
for col in ["Arrival Date", "ED Dispo Last Date", "Depart Date", "Admit Dispo Date",
            "Bed Assigned Date", "Hospital Discharge Date"]:
    if col in allpt_apr.columns:
        allpt_apr[col] = allpt_apr[col].apply(excel_serial_to_date)

for col in ["Arrival Time", "ED Dispo Last Time", "Depart Time", "Admit Dispo Time",
            "Bed Assigned Time", "Hospital Discharge Time"]:
    if col in allpt_apr.columns:
        allpt_apr[col] = allpt_apr[col].apply(excel_serial_to_time)

allpt_nov["FCT_dt"] = allpt_nov["FCT"].apply(parse_fct)
allpt_apr["FCT_dt"] = allpt_apr["FCT"].apply(parse_fct)

allpt_nov["Age_num"] = allpt_nov["Age at Visit"].apply(extract_age)
allpt_apr["Age_num"] = allpt_apr["Age at Visit"].apply(extract_age)

allpt = pd.concat([allpt_nov, allpt_apr], ignore_index=True)

print("Loading CT Orders files...")
ct_nov_img = wb_to_df(FILES["ct_nov"], "Nov Imaging")
ct_apr_img = wb_to_df(FILES["ct_apr"], "Apr26")

ct_nov_lab = wb_to_df(FILES["ct_nov"], "Nov Lab")
ct_apr_lab = wb_to_df(FILES["ct_apr"], "lab")

ct_nov_img["month"] = "Nov 2025"
ct_apr_img["month"] = "Apr 2026"

ct_img = pd.concat([ct_nov_img, ct_apr_img], ignore_index=True)

# Parse CT scan priority from free text
ct_img["CT_Scan_Priority"] = ct_img["Ordering Question Responses"].apply(parse_scan_priority)

# Build datetimes for turnaround calculations
print("Computing turnaround times...")
for df, label in [(ct_nov_img, "Nov"), (ct_apr_img, "Apr")]:
    pass  # handled below on combined df

def build_dt(df, date_col, time_col):
    return df.apply(lambda r: to_datetime_combined(r.get(date_col), r.get(time_col)), axis=1)

ct_img["dt_arrival"]   = build_dt(ct_img, "Arrival Date", "Arrival Time")
ct_img["dt_order"]     = build_dt(ct_img, "Order Date", "Order Time")
ct_img["dt_exam_start"]= build_dt(ct_img, "Exam Start Date", "Exam Start Time")
ct_img["dt_exam_end"]  = build_dt(ct_img, "Exam End Date", "Exam End Time")
ct_img["dt_result"]    = build_dt(ct_img, "Result Date", "Result Time")

# Exam Start Time is recorded only to the minute (no seconds) while Order Time has
# full seconds precision.  Floor dt_order to the minute before computing the
# order→exam-start delta to avoid artefactual negatives caused by sub-minute
# differences (e.g. order at 10:22:59, exam start stored as 10:22:00 → raw delta
# = −59 s).  This correction reduces spurious negatives from ~1527 to ~1.
ct_img["dt_order_floored"] = ct_img["dt_order"].apply(
    lambda dt: dt.replace(second=0, microsecond=0) if isinstance(dt, datetime) else None
)

ct_img["mins_arrival_to_order"]      = ct_img.apply(lambda r: minutes_between(r["dt_arrival"], r["dt_order"]), axis=1)
ct_img["mins_order_to_exam_start"]   = ct_img.apply(lambda r: minutes_between(r["dt_order_floored"], r["dt_exam_start"]), axis=1)

# Raw (unfiltered) delta using the floored order time — for exclusion accounting
ct_img["_order_exam_raw"] = ct_img.apply(
    lambda r: (r["dt_exam_start"] - r["dt_order_floored"]).total_seconds() / 60
              if isinstance(r["dt_exam_start"], datetime) and isinstance(r["dt_order_floored"], datetime)
              else None, axis=1
)
ct_img["mins_exam_duration"]         = ct_img.apply(lambda r: minutes_between(r["dt_exam_start"], r["dt_exam_end"]), axis=1)
ct_img["mins_exam_end_to_result"]    = ct_img.apply(lambda r: minutes_between(r["dt_exam_end"], r["dt_result"]), axis=1)
ct_img["mins_arrival_to_result"]     = ct_img.apply(lambda r: minutes_between(r["dt_arrival"], r["dt_result"]), axis=1)

# Simplify CT type from Order Description
def simplify_ct_type(val):
    if not val:
        return "Other"
    v = str(val).upper()
    if "BRAIN" in v:
        return "CT Brain"
    if "CERVICAL" in v or "C-SPINE" in v or "NECK" in v:
        return "CT C-Spine/Neck"
    if "THORAX" in v and "ABDOMEN" in v:
        return "CT TAP"
    if "THORAX" in v or "CHEST" in v:
        return "CT Thorax"
    if "ABDOMEN" in v or "PELVIS" in v:
        return "CT Abdomen/Pelvis"
    if "SPINE" in v or "LUMBAR" in v or "THORACIC" in v:
        return "CT Spine"
    if "AORTA" in v or "ANGIO" in v or "CTA" in v:
        return "CT Angio/Aorta"
    if "KUB" in v or "UROGRAM" in v:
        return "CT KUB"
    if "FACE" in v or "ORBIT" in v or "SINUS" in v:
        return "CT Face/Orbit/Sinus"
    if "EXTREMIT" in v or "LIMB" in v or "KNEE" in v or "HIP" in v or "SHOULDER" in v:
        return "CT Extremity"
    return "Other CT"

ct_img["CT_Type"] = ct_img["Order Description"].apply(simplify_ct_type)


# ---------------------------------------------------------------------------
# 1. field_reference.xlsx
# ---------------------------------------------------------------------------
print("Writing field_reference.xlsx...")

TRIAGE_AVAIL = {
    # All Pt fields
    "match": "No", "CSN": "No",
    "Age at Visit": "Yes", "Age Group": "Yes", "Sex": "Yes",
    "Arrival Date": "Yes", "Arrival Time": "Yes",
    "Arrival Method": "Yes",
    "Chief Complaint": "Yes",
    "KIV Uptriage": "Partial",
    "Primary ED Dx": "No", "ED Dx": "No",
    "FYI Text": "Partial",
    "ID Template": "No",
    "Triage Acuity": "Yes",
    "Consult Acuity": "No",
    "ED Disposition": "No", "Last Dispo B4 FCT": "No",
    "Referred/Transferred Hospital": "No",
    "Discipline": "No",
    "First ED Primary Physician ": "No", "Attending Clinicians (All)": "No",
    "LOS (mins)": "No", "Arrival to Doc": "No", "Dispo to Admit": "No",
    "ED Dispo Last Date": "No", "ED Dispo Last Time": "No",
    "Depart Date": "No", "Depart Time": "No",
    "IP Bed Request to Discharge Duration": "No", "IP Admit to Discharge Duration": "No",
    "Discharge Suboptions": "No", "Admission Type": "No",
    "Reatt Same ED": "Partial", "Reatt Reason": "Partial",
    "72h Reatt NGEMR ED": "No", "72h Reatt NGEMR ED then Admitted": "No",
    "Postal Sector": "Yes",
    "Patient Seen by GEM Tx Team": "No",
    "IPED Pt": "Partial",
    "Referral Source": "Yes", "Referral Source Type": "Yes", "Ref Src Add": "Yes",
    "HD/ICU from GW within 24H": "No",
    "First ED Care Area": "Yes", "Primary Care Area": "No",
    "Pt. Doc Type": "Yes",
    "Emergency/Non Emergency": "Yes", "Classification Type": "Yes",
    "CFS (ED)": "Partial", "CFS": "Partial",
    "COVID Protocol Indicator": "Partial",
    "Admit IP Ward": "No", "Admit IP Room": "No", "Admit IP Bed": "No",
    "Direct Triage Referral": "Yes",
    "Frontload": "Partial",
    "IDC": "No", "Specialty": "No",
    "Payment (with comments)": "Partial",
    "FCT": "Yes",
    "ED Primary Dx ICD10": "No",
    "Admit Dispo Date": "No", "Admit Dispo Time": "No",
    "TOSP": "No",
    "Bed Assigned Date": "No", "Bed Assigned": "No", "Bed Assigned Time": "No",
    "Hospital Discharge Date": "No", "Hospital Discharge Time": "No",
    "Disch Date/Time": "No", "Discharge Method": "No",
    "BP first": "No", "First BP Flowsheet & Timestamp": "No",
    "BP before Consult": "No", "BP before ETT": "No",
    "Last BP Flowsheet & Timestamp (ED)": "No",
    "BP last": "No", "BP last_1": "No",
}

CT_TIMING = {
    "Arrival Date": "Pre-scan", "Arrival Time": "Pre-scan",
    "Parent Order Date": "Pre-scan", "Parent Order Time": "Pre-scan",
    "Order Date": "Pre-scan", "Order Time": "Pre-scan",
    "Scan Schedule Time": "Pre-scan",
    "Exam Start Date": "During scan", "Exam Start Time": "During scan",
    "Exam End Date": "Post-scan", "Exam End Time": "Post-scan",
    "Result Date": "Post-scan", "Result Time": "Post-scan",
    "Read/Ackn on": "Post-scan", "Read/Ackn by": "Post-scan",
    "Order Completion Date": "Post-scan", "Order Completion Time": "Post-scan",
    "Order Completion Duration (mins)": "Post-scan",
    "Acuity Abbr": "Pre-scan",
    "Primary ED Dx": "Post-scan",
    "ED Dx": "Post-scan",
    "ED Disposition": "Post-scan",
    "Result Acuity": "Post-scan",
    "Abnormal": "Post-scan",
    "Critical Result": "Post-scan",
    "Ordering Question Responses": "Pre-scan",
    "Order Description": "Pre-scan",
    "Ordering Clinician": "Pre-scan",
}

def get_example(df, col):
    vals = df[col].dropna()
    if vals.empty:
        return ""
    v = vals.iloc[0]
    return str(v)[:80]


def classify_field(series):
    """Return (field_type, range_or_categories) for a pandas Series."""
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

    # Object / string columns
    str_vals = non_null.astype(str).str.strip()
    uniq_str = set(str_vals.unique())

    # Check binary-like strings
    if uniq_str <= {"0", "1"}:
        return "Binary (0/1)", "0, 1"
    if uniq_str <= {"Yes", "No"}:
        return "Binary (Yes/No)", "No, Yes"
    if uniq_str <= {"True", "False"}:
        return "Binary (True/False)", "False, True"

    n_unique = str_vals.nunique()
    if n_unique <= 30:
        cats = ", ".join(sorted(str_vals.unique()))
        return "Categorical", cats[:300]

    return "Text / ID", f"{n_unique} unique values"

# All Pt field reference
allpt_cols = [c for c in allpt_nov.columns if not c.endswith("_dt") and c not in ("Age_num", "month")]
allpt_ref_rows = []
for col in allpt_cols:
    ex = get_example(allpt_nov, col)
    dtype = str(allpt_nov[col].dtype)
    avail = TRIAGE_AVAIL.get(col, "Unknown")
    note = ""
    if col == "Triage Acuity":
        note = "P1=most urgent, P2=urgent, P3=semi-urgent, P4=non-urgent"
    elif col == "Consult Acuity":
        note = "Re-assessed acuity at physician contact — not known at triage"
    elif col == "FCT":
        note = "First Contact Time = triage time; stored as string, needs parsing"
    elif col == "CFS (ED)":
        note = "Clinical Frailty Scale; may be assessed at triage for elderly"
    elif col == "Reatt Same ED":
        note = "Known if patient is a prior attender (lookup possible at triage)"
    elif col in ("BP last", "BP last_1"):
        note = "Duplicate column header in source file"
    elif col == "Age at Visit":
        note = "Stored as string e.g. ' 73 y.o.' — numeric value needs regex extraction"
    ftype, frange = classify_field(allpt_nov[col])
    allpt_ref_rows.append({
        "Field Name": col,
        "Example Value": ex,
        "Data Type": dtype,
        "Field Type": ftype,
        "Range / Categories": frange,
        "Available at Triage?": avail,
        "Notes": note,
    })

df_allpt_ref = pd.DataFrame(allpt_ref_rows)

# CT Orders field reference
ct_cols = [c for c in ct_nov_img.columns if not c.startswith("dt_") and not c.startswith("mins_")
           and c not in ("CT_Type", "CT_Scan_Priority", "month")]
ct_ref_rows = []
for col in ct_cols:
    ex = get_example(ct_nov_img, col)
    dtype = str(ct_nov_img[col].dtype)
    timing = CT_TIMING.get(col, "")
    note = ""
    if col == "Ordering Question Responses":
        note = "Scan-level CT priority (P1/P2/P3) embedded in free text — needs regex parsing"
    elif col == "Acuity Abbr":
        note = "Patient-level ED triage acuity (P1-P3); distinct from CT scan priority"
    elif col == "Order Completion Duration (mins)":
        note = "Pre-calculated: order placed → order completed (mins)"
    elif col == "EDTC/U Order":
        note = "Flag: 1 = order placed while patient in EDTC/EDTU observation area"
    elif col == "Result Acuity":
        note = "Radiologist's assessment of result severity (Abnormal, Normal, etc.)"
    ftype, frange = classify_field(ct_nov_img[col])
    ct_ref_rows.append({
        "Field Name": col,
        "Example Value": ex,
        "Data Type": dtype,
        "Field Type": ftype,
        "Range / Categories": frange,
        "Pre/During/Post Scan": timing,
        "Notes": note,
    })

df_ct_ref = pd.DataFrame(ct_ref_rows)

# Lab sheet field reference
# Nov Lab has an extra "match" column not present in Apr lab; use Nov as reference
# and note Apr differences where they exist.
LAB_TIMING = {
    "Order Date": "Pre-result", "Order Time": "Pre-result",
    "Parent Order Date": "Pre-result", "Parent Order Time": "Pre-result",
    "Collection Date": "Pre-result", "Collection Time": "Pre-result",
    "Received Date": "Pre-result", "Received Time": "Pre-result",
    "Result Date": "Post-result", "Result Time": "Post-result",
    "Read/Ackn on": "Post-result", "Read/Ackn by": "Post-result",
    "Order Completion Date": "Post-result", "Order Completion Time": "Post-result",
    "Order Completion Duration (mins)": "Post-result",
    "Admit Dispo Date": "Post-result", "Admit Dispo Time": "Post-result",
    "Acuity Abbr": "Pre-result",
    "Primary ED Dx": "Post-result",
    "ED Dx": "Post-result",
    "ED Disposition": "Post-result",
    "Result Acuity": "Post-result",
    "Abnormal": "Post-result",
    "Critical Result": "Post-result",
    "Last Lab Test Results": "Post-result",
    "Order Description": "Pre-result",
    "Ordering Clinician": "Pre-result",
    "Ordering Question Responses": "Pre-result",
    "Scan Schedule Time": "Pre-result",
}

# Columns present in Apr lab but absent in Nov Lab (and vice versa)
nov_lab_only = set(ct_nov_lab.columns) - set(ct_apr_lab.columns)
apr_lab_only = set(ct_apr_lab.columns) - set(ct_nov_lab.columns)

lab_cols = [c for c in ct_nov_lab.columns if c not in ("month",)]
lab_ref_rows = []
for col in lab_cols:
    ex = get_example(ct_nov_lab, col)
    dtype = str(ct_nov_lab[col].dtype)
    timing = LAB_TIMING.get(col, "")
    note = ""
    if col in nov_lab_only:
        note = "Present in Nov Lab only (absent in Apr lab sheet)"
    if col in apr_lab_only:
        note = "Present in Apr lab only (absent in Nov Lab sheet)"
    if col == "Last Lab Test Results":
        note = "Free-text blob containing all component results for the order (e.g. 'Sodium: 139 mmol/L\\nPotassium: 3.5 mmol/L')"
    elif col == "Order Description":
        note = "Lab test name as ordered (e.g. 'NGEMR LAB RENAL PANEL', 'FULL BLOOD COUNT - NON BEAKER')"
    elif col == "Linked Chargeable":
        note = "Billing code / chargeable item linked to this order"
    elif col == "Acuity Abbr":
        note = "Patient-level ED triage acuity (P1-P3)"
    elif col == "Order Completion Duration (mins)":
        note = "Pre-calculated: order placed → order completed (mins)"
    elif col == "EDTC/U Order":
        note = "Flag: 1 = order placed while patient in EDTC/EDTU observation area"
    elif col == "Result Acuity":
        note = "Lab result severity flag set by the lab system"
    elif col == "ED/OP Case#":
        note = "ED/outpatient encounter number; may be None for inpatient-only orders"
    elif col == "IP Case#":
        note = "Inpatient encounter number; may be None for ED-only orders"
    elif col == "match":
        note = "Internal match/dedup key; present in Nov Lab only"
    elif col == "Scan Schedule Time":
        note = "Not applicable to labs; populated only for imaging orders — will be null here"
    ftype, frange = classify_field(ct_nov_lab[col])
    lab_ref_rows.append({
        "Field Name": col,
        "Example Value": ex,
        "Data Type": dtype,
        "Field Type": ftype,
        "Range / Categories": frange,
        "Pre/Post Result": timing,
        "Notes": note,
    })

# Also flag columns that appear in Apr lab but not Nov Lab
for col in sorted(apr_lab_only):
    ex = get_example(ct_apr_lab, col)
    dtype = str(ct_apr_lab[col].dtype)
    ftype, frange = classify_field(ct_apr_lab[col])
    lab_ref_rows.append({
        "Field Name": col,
        "Example Value": ex,
        "Data Type": dtype,
        "Field Type": ftype,
        "Range / Categories": frange,
        "Pre/Post Result": LAB_TIMING.get(col, ""),
        "Notes": "Present in Apr lab only (absent in Nov Lab sheet)",
    })

df_lab_ref = pd.DataFrame(lab_ref_rows)

with pd.ExcelWriter(os.path.join(OUT, "field_reference.xlsx"), engine="openpyxl") as writer:
    df_allpt_ref.to_excel(writer, sheet_name="All Pt Fields", index=False)
    df_ct_ref.to_excel(writer, sheet_name="CT Orders Fields", index=False)
    df_lab_ref.to_excel(writer, sheet_name="Lab Fields", index=False)

print("  -> field_reference.xlsx done")


# ---------------------------------------------------------------------------
# 2. summary_stats.xlsx
# ---------------------------------------------------------------------------
print("Writing summary_stats.xlsx...")

def pct(df, col, grp=None):
    if grp:
        counts = df.groupby([grp, col]).size().reset_index(name="Count")
        totals = df.groupby(grp).size().reset_index(name="Total")
        merged = counts.merge(totals, on=grp)
        merged["Pct"] = (merged["Count"] / merged["Total"] * 100).round(1)
        return merged
    counts = df[col].value_counts(dropna=False).reset_index()
    counts.columns = [col, "Count"]
    counts["Pct"] = (counts["Count"] / len(df) * 100).round(1)
    return counts

# Patient volume
vol = allpt.groupby("month").size().reset_index(name="Patient Count")

# Triage acuity
acuity_allpt = pct(allpt, "Triage Acuity", "month")

# CT volume by month
ct_vol = ct_img.groupby("month").size().reset_index(name="CT Orders")

# CT by acuity
ct_acuity = pct(ct_img, "Acuity Abbr", "month")

# CT scan priority
ct_priority = pct(ct_img, "CT_Scan_Priority", "month")

# CT type
ct_type = ct_img.groupby(["month", "CT_Type"]).size().reset_index(name="Count")
ct_type_total = ct_img.groupby("month").size().reset_index(name="Total")
ct_type = ct_type.merge(ct_type_total, on="month")
ct_type["Pct"] = (ct_type["Count"] / ct_type["Total"] * 100).round(1)
ct_type = ct_type.drop(columns=["Total"]).sort_values(["month", "Count"], ascending=[True, False])

# CT type x acuity
ct_type_acuity = ct_img.groupby(["CT_Type", "Acuity Abbr"]).size().unstack(fill_value=0).reset_index()

# Disposition
dispo = pct(allpt, "ED Disposition", "month")

# Arrival method
arrival = pct(allpt, "Arrival Method", "month")

# Age group
age_grp = pct(allpt, "Age Group", "month")

with pd.ExcelWriter(os.path.join(OUT, "summary_stats.xlsx"), engine="openpyxl") as writer:
    vol.to_excel(writer, sheet_name="Patient Volume", index=False)
    acuity_allpt.to_excel(writer, sheet_name="Triage Acuity", index=False)
    ct_vol.to_excel(writer, sheet_name="CT Volume", index=False)
    ct_acuity.to_excel(writer, sheet_name="CT by Acuity", index=False)
    ct_priority.to_excel(writer, sheet_name="CT Scan Priority", index=False)
    ct_type.to_excel(writer, sheet_name="CT Type", index=False)
    ct_type_acuity.to_excel(writer, sheet_name="CT Type x Acuity", index=False)
    dispo.to_excel(writer, sheet_name="Disposition", index=False)
    arrival.to_excel(writer, sheet_name="Arrival Method", index=False)
    age_grp.to_excel(writer, sheet_name="Age Group", index=False)

print("  -> summary_stats.xlsx done")


# ---------------------------------------------------------------------------
# 3. ct_turnaround.xlsx
# ---------------------------------------------------------------------------
print("Writing ct_turnaround.xlsx...")

time_cols = [
    "mins_arrival_to_order",
    "mins_order_to_exam_start",
    "mins_exam_duration",
    "mins_exam_end_to_result",
    "mins_arrival_to_result",
]
time_labels = {
    "mins_arrival_to_order": "Arrival → CT Order (mins)",
    "mins_order_to_exam_start": "CT Order → Exam Start (mins)",
    "mins_exam_duration": "Exam Duration (mins)",
    "mins_exam_end_to_result": "Exam End → Result (mins)",
    "mins_arrival_to_result": "Arrival → Result (mins)",
}

def turnaround_summary(df, group_cols):
    rows = []
    grp = df.groupby(group_cols)
    for key, sub in grp:
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["N"] = len(sub)
        for col in time_cols:
            vals = sub[col].dropna()
            if len(vals) > 0:
                row[f"{time_labels[col]} — Median"] = round(vals.median(), 1)
                row[f"{time_labels[col]} — P25"] = round(vals.quantile(0.25), 1)
                row[f"{time_labels[col]} — P75"] = round(vals.quantile(0.75), 1)
                row[f"{time_labels[col]} — Mean"] = round(vals.mean(), 1)
                row[f"{time_labels[col]} — N valid"] = len(vals)
            else:
                for suffix in ("Median", "P25", "P75", "Mean", "N valid"):
                    row[f"{time_labels[col]} — {suffix}"] = None
        rows.append(row)
    return pd.DataFrame(rows)

by_acuity = turnaround_summary(ct_img, ["month", "Acuity Abbr"])
by_ct_type = turnaround_summary(ct_img, ["month", "CT_Type"])
by_dispo = turnaround_summary(ct_img, ["month", "ED Disposition"])
by_scan_priority = turnaround_summary(ct_img, ["month", "CT_Scan_Priority"])

# Raw detail rows (one row per CT order) — all key fields
detail_cols = [
    "month", "CSN", "Acuity Abbr", "CT_Scan_Priority", "CT_Type", "Order Description",
    "Primary ED Dx", "ED Disposition",
    "Arrival Date", "Arrival Time",
    "Order Date", "Order Time",
    "Exam Start Date", "Exam Start Time",
    "Exam End Date", "Exam End Time",
    "Result Date", "Result Time",
    "mins_arrival_to_order", "mins_order_to_exam_start",
    "mins_exam_duration", "mins_exam_end_to_result", "mins_arrival_to_result",
    "Order Completion Duration (mins)",
]
detail_cols = [c for c in detail_cols if c in ct_img.columns]
ct_detail = ct_img[detail_cols].copy()

with pd.ExcelWriter(os.path.join(OUT, "ct_turnaround.xlsx"), engine="openpyxl") as writer:
    by_acuity.to_excel(writer, sheet_name="By Acuity", index=False)
    by_ct_type.to_excel(writer, sheet_name="By CT Type", index=False)
    by_dispo.to_excel(writer, sheet_name="By Disposition", index=False)
    by_scan_priority.to_excel(writer, sheet_name="By Scan Priority", index=False)
    ct_detail.to_excel(writer, sheet_name="Raw Detail", index=False)

print("  -> ct_turnaround.xlsx done")


# ---------------------------------------------------------------------------
# 4. data_quality.xlsx
# ---------------------------------------------------------------------------
print("Writing data_quality.xlsx...")

def null_report(df, label):
    rows = []
    for col in df.columns:
        total = len(df)
        nulls = df[col].isna().sum()
        pct_null = round(nulls / total * 100, 1) if total > 0 else 0
        rows.append({
            "Dataset": label,
            "Field": col,
            "Total Rows": total,
            "Null Count": nulls,
            "Null %": pct_null,
        })
    return pd.DataFrame(rows)

null_allpt_nov = null_report(allpt_nov[[c for c in allpt_nov.columns if not c.endswith("_dt") and c != "Age_num"]], "All Pt Nov 2025")
null_allpt_apr = null_report(allpt_apr[[c for c in allpt_apr.columns if not c.endswith("_dt") and c != "Age_num"]], "All Pt Apr 2026")
null_ct_nov = null_report(ct_nov_img[[c for c in ct_nov_img.columns if not c.startswith("dt_") and not c.startswith("mins_") and c not in ("CT_Type","CT_Scan_Priority")]], "CT Nov 2025")
null_ct_apr = null_report(ct_apr_img[[c for c in ct_apr_img.columns if not c.startswith("dt_") and not c.startswith("mins_") and c not in ("CT_Type","CT_Scan_Priority")]], "CT Apr 2026")

null_all = pd.concat([null_allpt_nov, null_allpt_apr, null_ct_nov, null_ct_apr], ignore_index=True)

known_issues = pd.DataFrame([
    {"Issue": "Date serialization mismatch",
     "Affected File": "2026 4 All Pt (decrypted).xlsx",
     "Detail": "Arrival Date/Time and other date fields are raw Excel OLE serial numbers (integer + decimal fraction) instead of parsed datetime objects. Requires manual conversion: datetime(1899,12,30) + timedelta(days=serial)."},
    {"Issue": "Duplicate column header",
     "Affected File": "Both All Pt files",
     "Detail": "'BP last' appears twice (indices 77 and 78). Index 77 = bare value, index 78 = value with timestamp. Deduplicated as 'BP last' and 'BP last_1' in this analysis."},
    {"Issue": "Age stored as string",
     "Affected File": "All All Pt files",
     "Detail": "Age at Visit is stored as e.g. ' 73 y.o.' — numeric extraction requires regex r'\\d+'."},
    {"Issue": "Flag fields as strings",
     "Affected File": "All All Pt files",
     "Detail": "Several 0/1 flag fields (Patient Seen by GEM Tx Team, HD/ICU from GW within 24H) are stored as strings '0'/'1' rather than integers."},
    {"Issue": "CT scan priority in free text",
     "Affected File": "Both CT Orders files",
     "Detail": "The scan-level CT priority (P1 Urgent / P3 Normal) is embedded inside 'Ordering Question Responses' free-text field, not a separate column. Requires regex parsing."},
    {"Issue": "FCT as formatted string",
     "Affected File": "All All Pt files",
     "Detail": "First Contact Time (FCT) is stored as '01/11/2025 01:00:29 AM' string. Requires datetime.strptime(s, '%d/%m/%Y %I:%M:%S %p')."},
    {"Issue": "Consult Acuity vs Triage Acuity",
     "Affected File": "All All Pt files",
     "Detail": "Both fields exist but serve different purposes. Triage Acuity = assigned at triage. Consult Acuity = re-assessed at physician contact. Use Triage Acuity for triage analysis."},
    {"Issue": "Patient Acuity vs CT Scan Priority may differ",
     "Affected File": "Both CT Orders files",
     "Detail": "Acuity Abbr (patient-level triage) and CT scan priority (from Ordering Question Responses) are different and can disagree. E.g., a P2 patient may have a P3 scan priority."},
    {"Issue": "Apr26 has 6 extra columns vs Nov Imaging",
     "Affected File": "2026 4 CT Orders (decrypted).xlsx",
     "Detail": "Apr26 sheet adds: Admit Dispo Date/Time, Admit EDTC/U Date/Time, EDTC/U IP Date/Time. These are absent in Nov Imaging sheet."},
    {"Issue": "Exam Start Time recorded to minute only; Order Time has seconds precision",
     "Affected File": "Both CT Orders imaging sheets",
     "Detail": "Exam Start Time is stored as HH:MM (no seconds), while Order Time includes seconds. Computing the raw delta produces ~1,527/1,565 artefactual negative values per month (exam start appears before order) for cases where the order was placed in the same minute the scan started (e.g. order at 10:22:59, exam start at 10:22:00). Fix applied: Order Time is floored to the minute before computing Order→Exam Start duration."},
])

# Triage availability summary
triage_avail_summary = df_allpt_ref[["Field Name", "Available at Triage?", "Notes"]].copy()
triage_avail_counts = triage_avail_summary["Available at Triage?"].value_counts().reset_index()
triage_avail_counts.columns = ["Available at Triage?", "Field Count"]

with pd.ExcelWriter(os.path.join(OUT, "data_quality.xlsx"), engine="openpyxl") as writer:
    null_all.to_excel(writer, sheet_name="Null Rates", index=False)
    known_issues.to_excel(writer, sheet_name="Known Issues", index=False)
    triage_avail_summary.to_excel(writer, sheet_name="Triage Availability", index=False)
    triage_avail_counts.to_excel(writer, sheet_name="Availability Summary", index=False)

print("  -> data_quality.xlsx done")


# ---------------------------------------------------------------------------
# 5. ct_orders_analysis.xlsx + charts
# ---------------------------------------------------------------------------
print("Writing ct_orders_analysis.xlsx and charts...")

sns.set_theme(style="whitegrid", font_scale=1.05)
MONTH_ORDER = ["Nov 2025", "Apr 2026"]
ACUITY_ORDER = ["P1", "P2", "P3"]
PALETTE = {"Nov 2025": "#4C72B0", "Apr 2026": "#DD8452"}


def timing_stats(series, raw_series, label=None):
    """
    series     — filtered column (mins_order_to_exam_start): negatives and >1440 already nulled
    raw_series — unfiltered delta in minutes (_order_exam_raw): NaN only when a timestamp is missing
    """
    vals = series.dropna()
    n_total    = len(series)
    n_no_exam  = raw_series.isna().sum()
    n_negative = (raw_series < 0).sum()
    n_over_24h = (raw_series > 1440).sum()
    n_excluded = n_no_exam + n_negative + n_over_24h
    n_valid    = len(vals)
    row = {
        "Group": label if label else "",
        "N Total": n_total,
        "N Excluded": n_excluded,
        "  No exam timestamp": n_no_exam,
        "  Negative duration": n_negative,
        "  Duration > 24 h":   n_over_24h,
        "N Valid": n_valid,
        "Mean (mins)": round(vals.mean(), 1) if n_valid else None,
        "Median (mins)": round(vals.median(), 1) if n_valid else None,
        "P25 (mins)": round(vals.quantile(0.25), 1) if n_valid else None,
        "P75 (mins)": round(vals.quantile(0.75), 1) if n_valid else None,
        "P90 (mins)": round(vals.quantile(0.90), 1) if n_valid else None,
        "P95 (mins)": round(vals.quantile(0.95), 1) if n_valid else None,
    }
    return row


# --- 5a. Timing Overall (per month + combined) ---
timing_overall_rows = []
for m in MONTH_ORDER:
    mask = ct_img["month"] == m
    row = timing_stats(ct_img.loc[mask, "mins_order_to_exam_start"],
                       ct_img.loc[mask, "_order_exam_raw"], label=m)
    timing_overall_rows.append(row)
row_all = timing_stats(ct_img["mins_order_to_exam_start"],
                       ct_img["_order_exam_raw"], label="Both Months")
timing_overall_rows.append(row_all)
df_timing_overall = pd.DataFrame(timing_overall_rows)

# --- 5b. Timing by Acuity (per month × acuity + combined × acuity) ---
timing_acuity_rows = []
for m in MONTH_ORDER:
    for ac in ACUITY_ORDER:
        mask = (ct_img["month"] == m) & (ct_img["Acuity Abbr"] == ac)
        row = timing_stats(ct_img.loc[mask, "mins_order_to_exam_start"],
                           ct_img.loc[mask, "_order_exam_raw"],
                           label=f"{m} — {ac}")
        row["Month"] = m
        row["Acuity"] = ac
        timing_acuity_rows.append(row)
for ac in ACUITY_ORDER:
    mask = ct_img["Acuity Abbr"] == ac
    row = timing_stats(ct_img.loc[mask, "mins_order_to_exam_start"],
                       ct_img.loc[mask, "_order_exam_raw"],
                       label=f"Both Months — {ac}")
    row["Month"] = "Both Months"
    row["Acuity"] = ac
    timing_acuity_rows.append(row)
df_timing_acuity = pd.DataFrame(timing_acuity_rows)
# Reorder columns for readability
ta_cols = ["Month", "Acuity", "N Total", "N Excluded", "  No exam timestamp",
           "  Negative duration", "  Duration > 24 h", "N Valid",
           "Mean (mins)", "Median (mins)", "P25 (mins)", "P75 (mins)", "P90 (mins)", "P95 (mins)"]
df_timing_acuity = df_timing_acuity[[c for c in ta_cols if c in df_timing_acuity.columns]]

# --- 5c. Volume by Day ---
ct_img["order_day"] = ct_img["dt_order"].dt.day
vol_day = ct_img.groupby(["order_day", "month"]).size().unstack(fill_value=0).reset_index()
vol_day.columns.name = None
vol_day = vol_day.rename(columns={"order_day": "Day of Month"})
# Ensure both month columns exist
for m in MONTH_ORDER:
    if m not in vol_day.columns:
        vol_day[m] = 0
vol_day = vol_day[["Day of Month"] + MONTH_ORDER].sort_values("Day of Month")

# --- 5d. Volume by Month ---
vol_month = ct_img.groupby("month").size().reset_index(name="CT Orders")
vol_month = vol_month.rename(columns={"month": "Month"})

# --- 5e. Volume by Hour ---
ct_img["order_hour"] = ct_img["dt_order"].dt.hour
vol_hour = ct_img.groupby("order_hour").size().reset_index(name="CT Orders (Combined)")
vol_hour = vol_hour.rename(columns={"order_hour": "Hour of Day (0–23)"})
all_hours = pd.DataFrame({"Hour of Day (0–23)": range(24)})
vol_hour = all_hours.merge(vol_hour, on="Hour of Day (0–23)", how="left").fillna(0)
vol_hour["CT Orders (Combined)"] = vol_hour["CT Orders (Combined)"].astype(int)

# Write Excel
with pd.ExcelWriter(os.path.join(OUT, "ct_orders_analysis.xlsx"), engine="openpyxl") as writer:
    df_timing_overall.to_excel(writer, sheet_name="Timing Overall", index=False)
    df_timing_acuity.to_excel(writer, sheet_name="Timing by Acuity", index=False)
    vol_day.to_excel(writer, sheet_name="Volume by Day", index=False)
    vol_month.to_excel(writer, sheet_name="Volume by Month", index=False)
    vol_hour.to_excel(writer, sheet_name="Volume by Hour", index=False)

print("  -> ct_orders_analysis.xlsx done")

# --- Charts ---

# Chart 1: CT Orders by Day of Month (grouped bar, Nov vs Apr)
fig, ax = plt.subplots(figsize=(13, 5))
days = vol_day["Day of Month"].values
x = range(len(days))
w = 0.4
bars_nov = ax.bar([i - w/2 for i in x], vol_day.get("Nov 2025", 0), width=w,
                  label="Nov 2025", color=PALETTE["Nov 2025"], alpha=0.85)
bars_apr = ax.bar([i + w/2 for i in x], vol_day.get("Apr 2026", 0), width=w,
                  label="Apr 2026", color=PALETTE["Apr 2026"], alpha=0.85)
ax.set_xticks(list(x))
ax.set_xticklabels(days, fontsize=8)
ax.set_xlabel("Day of Month")
ax.set_ylabel("CT Orders")
ax.set_title("CT Orders by Day of Month")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ct_volume_by_day.png"), dpi=150)
plt.close(fig)

# Chart 2: CT Orders by Hour of Day (combined, bar + mean line)
fig, ax = plt.subplots(figsize=(12, 5))
hours = vol_hour["Hour of Day (0–23)"].values
counts = vol_hour["CT Orders (Combined)"].values
ax.bar(hours, counts, color="#4C72B0", alpha=0.8, label="CT Orders")
mean_val = counts.mean()
ax.axhline(mean_val, color="#C44E52", linewidth=1.5, linestyle="--",
           label=f"Mean = {mean_val:.1f}")
ax.set_xticks(list(range(24)))
ax.set_xlabel("Hour of Day (0–23)")
ax.set_ylabel("CT Orders")
ax.set_title("CT Orders by Hour of Day (Both Months Combined)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ct_volume_by_hour.png"), dpi=150)
plt.close(fig)

# Chart 3: Median order→exam-start by Acuity (grouped bar, Nov vs Apr, with IQR error bars)
fig, ax = plt.subplots(figsize=(8, 5))
both_months_only = df_timing_acuity[df_timing_acuity["Month"].isin(MONTH_ORDER)].copy()
x_pos = {ac: i for i, ac in enumerate(ACUITY_ORDER)}
w = 0.35
for j, m in enumerate(MONTH_ORDER):
    sub = both_months_only[both_months_only["Month"] == m]
    xs = [x_pos[ac] + (j - 0.5) * w for ac in ACUITY_ORDER]
    meds = [sub[sub["Acuity"] == ac]["Median (mins)"].values[0]
            if len(sub[sub["Acuity"] == ac]) > 0 else 0 for ac in ACUITY_ORDER]
    p25s = [sub[sub["Acuity"] == ac]["P25 (mins)"].values[0]
            if len(sub[sub["Acuity"] == ac]) > 0 else 0 for ac in ACUITY_ORDER]
    p75s = [sub[sub["Acuity"] == ac]["P75 (mins)"].values[0]
            if len(sub[sub["Acuity"] == ac]) > 0 else 0 for ac in ACUITY_ORDER]
    yerr_lo = [max(0, m_ - p) if m_ is not None and p is not None else 0
               for m_, p in zip(meds, p25s)]
    yerr_hi = [max(0, p - m_) if m_ is not None and p is not None else 0
               for m_, p in zip(meds, p75s)]
    bars = ax.bar(xs, meds, width=w, label=m, color=PALETTE[m], alpha=0.85,
                  yerr=[yerr_lo, yerr_hi], capsize=4, error_kw={"elinewidth": 1.2})
    for xi, med in zip(xs, meds):
        if med is not None:
            ax.text(xi, med + max(yerr_hi) * 0.05 + 0.5, f"{med:.0f}",
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(list(x_pos.values()))
ax.set_xticklabels(ACUITY_ORDER)
ax.set_xlabel("Acuity")
ax.set_ylabel("Median Order → Exam Start (mins)")
ax.set_title("Median Order → Exam Start by Acuity\n(error bars = P25–P75)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ct_timing_by_acuity.png"), dpi=150)
plt.close(fig)

# Chart 4: Box plot of order→exam-start by Acuity (combined months, capped at 120 min)
CAP = 120
plot_data = ct_img[ct_img["mins_order_to_exam_start"].notna() &
                   (ct_img["mins_order_to_exam_start"] <= CAP) &
                   ct_img["Acuity Abbr"].isin(ACUITY_ORDER)].copy()
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=plot_data, x="Acuity Abbr", y="mins_order_to_exam_start",
            hue="Acuity Abbr", order=ACUITY_ORDER, palette=["#4C72B0", "#DD8452", "#55A868"],
            width=0.45, flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
            legend=False, ax=ax)
ax.set_xlabel("Acuity")
ax.set_ylabel("Order → Exam Start (mins)")
ax.set_title(f"Order → Exam Start Distribution by Acuity\n(both months combined, capped at {CAP} min)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ct_timing_overall_box.png"), dpi=150)
plt.close(fig)

print("  -> charts done")

print("\nAll done. Output files in:", OUT)
for f in sorted(os.listdir(OUT)):
    path = os.path.join(OUT, f)
    size_kb = os.path.getsize(path) // 1024
    print(f"  {f}  ({size_kb} KB)")
