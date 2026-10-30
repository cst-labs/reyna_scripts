"""
Per-feature distribution by Scan Priority, for ALL NNI CT orders (every
Consult Acuity, not just the P2-consult-acuity cohort in
nni_p2_scan_priority_rf.py / nni_p2_scan_priority_feature_distributions.py).

Same feature engineering as the P2-only version, applied to the full NNI
cohort, so the two workbooks are directly comparable -- this one answers
"does this feature/Scan Priority relationship hold across all consult
acuities, or is it specific to the P2 cohort?"

Cohort/feature-engineering logic is duplicated from nni_p2_scan_priority_rf.py
(same join, same derived columns, but no Consult Acuity filter) so this can
be run independently; keep the two in sync if that script's feature
engineering changes.

Reads:
  data/nni_cleaned_orders.csv
  merged/all_pt_cleaned.xlsx

Outputs: analysis/nni_scan_priority_feature_distributions_allcases.xlsx
  - class_balance      : Scan Priority counts, overall and by Consult Acuity
  - numeric_summary    : count/mean/median/std/min/p25/p75/p90/max per feature x Scan Priority
  - ordinal_summary    : category-level counts/% per feature x Scan Priority (ordered);
                         percentages are within category rows
  - one sheet per nominal feature: category-level counts/% x Scan Priority;
                         percentages are within category rows
"""
import os
import re

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

PRIO_ORDER = ["P1", "P2", "P3"]
CFS_ORDER = [
    "1 - Very Fit", "2 - Well", "3 - Managing Well", "4 - Very Mild Frailty",
    "5 - Mild Frailty", "6 - Moderate Frailty", "7 - Severe Frailty",
    "8 - Very Severe Frailty", "9 - Terminally Ill",
]
BP_CATEGORY_ORDER = [
    "Normal BP", "High-Normal BP",
    "Grade 1 Hypertension", "Grade 2 Hypertension", "Isolated Systolic Hypertension",
]


# ---------------------------------------------------------------------------
# Helpers (duplicated from nni_p2_scan_priority_rf.py for consistency)
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
    if is_blank(val):
        return [None]
    s = str(val)
    return s.split("; ") if "; " in s else [s]


def parse_date_only(date_val):
    if isinstance(date_val, pd.Timestamp):
        return date_val
    s = str(date_val)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        if pd.notna(d):
            return d
    return pd.to_datetime(date_val, errors="coerce", dayfirst=True)


def combine_dt(date_val, time_val):
    if is_blank(date_val) or is_blank(time_val):
        return None
    d = parse_date_only(date_val)
    if pd.isna(d):
        return None
    dt = pd.to_datetime(f"{d.strftime('%Y-%m-%d')} {time_val}", errors="coerce")
    return None if pd.isna(dt) else dt


def earliest_parent_order_dt(date_val, time_val):
    d_parts = split_joined(date_val)
    t_parts = split_joined(time_val)
    n = max(len(d_parts), len(t_parts))
    if len(d_parts) < n:
        d_parts = d_parts + [d_parts[-1]] * (n - len(d_parts))
    if len(t_parts) < n:
        t_parts = t_parts + [t_parts[-1]] * (n - len(t_parts))
    dts = [combine_dt(d, t) for d, t in zip(d_parts, t_parts)]
    dts = [dt for dt in dts if dt is not None]
    return min(dts) if dts else None


def classify_bp(series):
    parsed = series.astype(str).str.replace(r"^\(!\)\s*", "", regex=True).str.extract(r"(\d+)\s*/\s*(\d+)")
    sysbp = parsed[0].astype(float)
    diabp = parsed[1].astype(float)
    mask = sysbp.notna() & diabp.notna()
    conditions = [
        (sysbp >= 140) & (diabp < 90),
        (sysbp >= 160) | (diabp >= 100),
        (sysbp >= 140) | (diabp >= 90),
        (sysbp >= 130) | (diabp >= 80),
    ]
    choices = [
        "Isolated Systolic Hypertension", "Grade 2 Hypertension", "Grade 1 Hypertension",
        "High-Normal BP",
    ]
    category = pd.Series(np.select(conditions, choices, default="Normal BP"), index=series.index)
    category[~mask] = pd.NA
    return pd.Categorical(category, categories=BP_CATEGORY_ORDER, ordered=True)


def parse_block(text):
    answers = {}
    for line in str(text).split("\n"):
        parts = re.split(r"::|:", line, maxsplit=1)
        if len(parts) == 2:
            answers[parts[0].strip()] = parts[1].strip()
    return answers


def proc_type(desc):
    d = str(desc).upper()
    parts = []
    if "BRAIN" in d and "ANGIOGRAM" not in d and "VENOGRAM" not in d and "PERFUSION" not in d:
        parts.append("Brain")
    if "FACE" in d:
        parts.append("Face")
    if "CERVICAL" in d:
        parts.append("C-Spine")
    if "ANGIOGRAM" in d or "VENOGRAM" in d or "PERFUSION" in d:
        parts.append("Angio/Perfusion")
    return "+".join(parts) if parts else "Other"


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 50)


# ---------------------------------------------------------------------------
# Load, join patient fields, filter to the P2-consult-acuity cohort
# ---------------------------------------------------------------------------
print("Loading data/nni_cleaned_orders.csv ...")
nni = pd.read_csv(os.path.join(DATA, "nni_cleaned_orders.csv"))

print("Loading merged/all_pt_cleaned.xlsx ...")
allpt = pd.read_excel(os.path.join(MERGED, "all_pt_cleaned.xlsx"))
allpt["Consult Acuity"] = allpt["Consult Acuity"].replace({"(P) P2": "P2"})
patient_cols = [
    "CSN", "Consult Acuity", "Age at Visit", "Sex", "CFS (ED)", "Triage Acuity", "BP first",
    "Arrival Method", "Classification Type", "First ED Care Area", "Primary Care Area",
    "Reatt Same ED", "KIV Uptriage", "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted",
    "HD/ICU from GW within 24H",
]
nni = nni.merge(allpt[patient_cols].drop_duplicates("CSN"), on="CSN", how="left")

cohort = nni[nni["Scan Priority"].isin(PRIO_ORDER)].copy()
print(f"  All NNI rows with known Scan Priority (any Consult Acuity): {len(cohort):,}")

# ---------------------------------------------------------------------------
# Feature engineering (identical to nni_p2_scan_priority_rf.py)
# ---------------------------------------------------------------------------
cohort["proc_type"] = cohort["Order Description"].apply(proc_type)

dx_top = cohort["Primary ED Dx"].value_counts().head(14).index
cohort["primary_dx"] = cohort["Primary ED Dx"].where(cohort["Primary ED Dx"].isin(dx_top), "Other")

first_block = cohort["Ordering Question Responses"].astype(str).str.split("; ").str[0]
parsed = first_block.apply(parse_block)
cohort["mode_of_transport"] = parsed.apply(lambda d: d.get("Mode of transport"))
cohort["renal_risk_factor"] = parsed.apply(lambda d: d.get(
    "Does the patient have risk factors for renal disease eg. hypertension, diabetes, "
    "is age above 60 years old (based on respective hospitals' guidelines)?"
))
cohort["contrast_allergy"] = parsed.apply(lambda d: d.get("Does patient have allergy to iodinated/ CT contrast?"))
cohort["polypharmacy_allergy"] = parsed.apply(
    lambda d: d.get("Is the patient allergic to 3 or more different classes of drugs?")
)
cohort["pregnant"] = parsed.apply(lambda d: d.get("Is the patient pregnant at the point of imaging request?"))

parent_dt = cohort.apply(
    lambda r: earliest_parent_order_dt(r["Parent Order Date"], r["Parent Order Time"]), axis=1
)
cohort["order_hour"] = parent_dt.apply(lambda dt: dt.hour if pd.notna(dt) else np.nan)

cohort["Age at Visit"] = cohort["Age at Visit"].astype(str).str.extract(r"(\d+)").astype(float)

cohort["Triage Acuity"] = cohort["Triage Acuity"].astype(str).str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
cohort.loc[~cohort["Triage Acuity"].isin(PRIO_ORDER), "Triage Acuity"] = np.nan

cohort["CFS (ED)"] = pd.Categorical(cohort["CFS (ED)"], categories=CFS_ORDER, ordered=True)
cohort["BP first Category"] = classify_bp(cohort["BP first"])

for flag in ["SOB", "Chest pain", "One-sided weakness", "Major bleeding"]:
    cohort[f"kiv_{flag.lower().replace(' ', '_').replace('-', '_')}"] = (
        cohort["KIV Uptriage"].astype(str).str.contains(flag, case=False, regex=False).astype(int)
    )

ORDINAL_FEATURES = {
    "CFS (ED)": CFS_ORDER,
    "BP first Category": BP_CATEGORY_ORDER,
}
NOMINAL_FEATURES = [
    "proc_type", "primary_dx", "mode_of_transport", "renal_risk_factor",
    "contrast_allergy", "polypharmacy_allergy", "pregnant", "Sex",
    "Triage Acuity", "Arrival Method", "Classification Type", "First ED Care Area",
    "Primary Care Area", "Order Type", "Reatt Same ED",
]
NUMERIC_FEATURES = [
    "Age at Visit", "order_hour",
    "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted", "HD/ICU from GW within 24H",
    "kiv_sob", "kiv_chest_pain", "kiv_one_sided_weakness", "kiv_major_bleeding",
]

# ---------------------------------------------------------------------------
# Distribution tables
# ---------------------------------------------------------------------------
PCTS = [0.25, 0.75, 0.90]


def numeric_distribution(col):
    rows = []
    for prio in PRIO_ORDER:
        vals = cohort.loc[cohort["Scan Priority"] == prio, col].dropna()
        row = {"feature": col, "Scan Priority": prio, "n": len(vals)}
        if len(vals):
            row.update({
                "mean": round(vals.mean(), 3), "median": round(vals.median(), 3),
                "std": round(vals.std(), 3), "min": round(vals.min(), 3),
                "p25": round(vals.quantile(0.25), 3), "p75": round(vals.quantile(0.75), 3),
                "p90": round(vals.quantile(0.90), 3), "max": round(vals.max(), 3),
            })
        else:
            row.update({k: None for k in ["mean", "median", "std", "min", "p25", "p75", "p90", "max"]})
        rows.append(row)
    return rows


def category_distribution(col, categories=None):
    """count + % within each category row, for every observed (or given
    ordered) category of col, including missing as its own row."""
    series = cohort[col].astype("object")
    series = series.where(series.notna(), "Missing").astype(str)
    if categories is not None:
        cats = list(categories) + (["Missing"] if (series == "Missing").any() else [])
    else:
        cats = series.value_counts().index.tolist()

    counts = pd.crosstab(series, cohort["Scan Priority"])
    counts = counts.reindex(index=cats, columns=PRIO_ORDER, fill_value=0)
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100

    out = pd.DataFrame(index=counts.index)
    for prio in PRIO_ORDER:
        out[f"{prio} n"] = counts[prio]
        out[f"{prio} %"] = pct[prio].round(1)
    out["Total n"] = counts.sum(axis=1)
    out = out.reset_index().rename(columns={"index": col})
    return out


numeric_rows = []
for col in NUMERIC_FEATURES:
    numeric_rows.extend(numeric_distribution(col))
numeric_summary = pd.DataFrame(numeric_rows)

ordinal_tables = {col: category_distribution(col, categories=order) for col, order in ORDINAL_FEATURES.items()}
nominal_tables = {col: category_distribution(col) for col in NOMINAL_FEATURES}
# Consult Acuity itself varies in this all-cases cohort (unlike the P2-only
# version, where it's constant) -- break it out same as any other nominal.
nominal_tables["Consult Acuity"] = category_distribution("Consult Acuity")

# ---------------------------------------------------------------------------
# Write workbook
# ---------------------------------------------------------------------------
SAFE_NAME = re.compile(r"[\\/*?:\[\]]")


def sheet_name(col):
    return SAFE_NAME.sub("_", col)[:31]


out_path = os.path.join(OUT, "nni_scan_priority_feature_distributions_allcases.xlsx")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    class_balance = cohort["Scan Priority"].value_counts().reindex(PRIO_ORDER).rename("n").reset_index()
    class_balance.rename(columns={"index": "Scan Priority"}).to_excel(writer, sheet_name="class_balance", index=False)

    by_consult = pd.crosstab(cohort["Consult Acuity"].fillna("Missing"), cohort["Scan Priority"])
    by_consult = by_consult.reindex(columns=PRIO_ORDER, fill_value=0)
    by_consult.reset_index().to_excel(writer, sheet_name="class_balance", index=False, startrow=len(class_balance) + 2)

    numeric_summary.to_excel(writer, sheet_name="numeric_summary", index=False)

    ordinal_rows = []
    for col, tbl in ordinal_tables.items():
        tbl.insert(0, "feature", col)
        ordinal_rows.append(tbl.rename(columns={col: "category"}))
    pd.concat(ordinal_rows, ignore_index=True).to_excel(writer, sheet_name="ordinal_summary", index=False)

    for col, tbl in nominal_tables.items():
        tbl.to_excel(writer, sheet_name=sheet_name(col), index=False)

    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print(f"Wrote {out_path}")
