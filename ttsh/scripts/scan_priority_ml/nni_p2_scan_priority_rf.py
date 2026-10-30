"""
What predicts Scan Priority among P2-consult-acuity NNI orders
================================================================
Among NNI CT orders where the ED consult acuity (Consult Acuity, from
all_pt_cleaned.xlsx -- not the same field as the order-level "Acuity Abbr",
which is ED triage acuity, though the two agree ~99.9% of the time) is P2,
the radiology scan priority (parsed from Ordering Question Responses) often
diverges: some get upgraded to P1, some stay P2, most get downgraded to P3.
This fits a shallow RandomForest + a single shallow DecisionTree (for
readable splits) on order- and patient-level fields to see what's
associated with that divergence. Exploratory, not predictive deployment.

Reads:
  data/nni_cleaned_orders.csv   (NNI CT orders; Scan Priority already parsed)
  merged/all_pt_cleaned.xlsx    (patient-level fields, joined on CSN)

Outputs: analysis/nni_p2_scan_priority_rf.xlsx + 3 PNGs + 1 TXT.
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl.utils import get_column_letter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

# ── Palette (matches nni_wait_time_by_scan_priority.py) ────────────────────
CAT_COLORS = {"P1": "#2a78d6", "P2": "#eb6834", "P3": "#1baf7a"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

PRIO_ORDER = ["P1", "P2", "P3"]
AGE_GROUP_ORDER = ["<19", "19 to <40", "40 to <65", "65 to <75", "75 to <85", ">/= 85"]
CFS_ORDER = [
    "1 - Very Fit", "2 - Well", "3 - Managing Well", "4 - Very Mild Frailty",
    "5 - Mild Frailty", "6 - Moderate Frailty", "7 - Severe Frailty",
    "8 - Very Severe Frailty", "9 - Terminally Ill",
]
BP_CATEGORY_ORDER = [
    "Normal BP", "High-Normal BP",
    "Grade 1 Hypertension", "Grade 2 Hypertension", "Isolated Systolic Hypertension",
]
RNG = 0

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE, "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY, "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY, "grid.color": GRIDLINE,
})


# ---------------------------------------------------------------------------
# Helpers (adapted from ct_wait_time_by_consult_acuity.py / consult_acuity_
# field_heatmap.py / checklist_question_analysis.py for consistency)
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
    """Parent Order Date/Time can be a '; '-joined multi-value string on rows
    clean_ct_orders.py merged from originally-separate orders. Returns the
    earliest combined timestamp across all joined candidates."""
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
    """First '; '-joined block of Ordering Question Responses only (later
    blocks are amendments of the same order, not new info)."""
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


def build_p2_cohort_features():
    """Load NNI orders + All PT, filter to P2-consult-acuity rows with a known
    Scan Priority, and build the one-hot model matrix. Shared by
    nni_p2_scan_priority_rf.py and tune_nni_p2_scan_priority_rf_v2.py so both
    scripts see the same cohort/feature definition."""
    print("Loading data/nni_cleaned_orders.csv ...")
    nni = pd.read_csv(os.path.join(DATA, "nni_cleaned_orders.csv"), dtype={"CSN": str})
    print(f"  All NNI rows: {len(nni):,}")

    print("Loading merged/all_pt_cleaned.xlsx ...")
    allpt = pd.read_excel(os.path.join(MERGED, "all_pt_cleaned.xlsx"))
    allpt["CSN"] = allpt["CSN"].astype(str)
    # Consult Acuity (join key: CSN) is the ED consult acuity, distinct from the
    # order-level "Acuity Abbr" field (which is ED triage acuity). Same cleaning
    # as ct_wait_time_by_consult_acuity.py.
    allpt["Consult Acuity"] = allpt["Consult Acuity"].replace({"(P) P2": "P2"})
    patient_cols = [
        "CSN", "Consult Acuity", "Age at Visit", "Sex", "CFS (ED)", "Triage Acuity", "BP first",
        "Arrival Method", "Classification Type", "First ED Care Area", "Primary Care Area",
        "Reatt Same ED", "KIV Uptriage", "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted",
        "HD/ICU from GW within 24H",
    ]
    nni = nni.merge(allpt[patient_cols].drop_duplicates("CSN"), on="CSN", how="left")
    match_rate = nni["Age at Visit"].notna().mean()
    print(f"  Patient-field join match rate: {match_rate:.1%}")

    cohort = nni[(nni["Consult Acuity"] == "P2") & (nni["Scan Priority"].isin(PRIO_ORDER))].copy()
    print(f"  P2-consult-acuity rows with known Scan Priority: {len(cohort):,}")
    print(f"  Scan Priority class balance:\n{cohort['Scan Priority'].value_counts()}")

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

    # KIV Uptriage is a ";"-joined multi-select of red-flag symptoms (e.g.
    # "Chest pain;SOB") -- split into individual binary flags rather than
    # one-hot the raw combo string, since combos are rare and would fragment
    # the tree's splits.
    for flag in ["SOB", "Chest pain", "One-sided weakness", "Major bleeding"]:
        cohort[f"kiv_{flag.lower().replace(' ', '_').replace('-', '_')}"] = (
            cohort["KIV Uptriage"].astype(str).str.contains(flag, case=False, regex=False).astype(int)
        )

    feature_df = pd.DataFrame(index=cohort.index)
    for col in NUMERIC_FEATURES:
        feature_df[col] = cohort[col]
    for col, order in ORDINAL_FEATURES.items():
        feature_df[col] = cohort[col].cat.codes.replace(-1, np.nan)
    nominal_filled = cohort[NOMINAL_FEATURES].fillna("Missing").astype(str)
    feature_df = pd.concat([feature_df, pd.get_dummies(nominal_filled, prefix=NOMINAL_FEATURES)], axis=1)

    X = feature_df.fillna(-1)
    y = cohort["Scan Priority"]
    print(f"\nModel matrix: {X.shape[0]:,} rows x {X.shape[1]:,} features")
    return cohort, feature_df, X, y


def main() -> None:
    cohort, feature_df, X, y = build_p2_cohort_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RNG, stratify=y
    )

    # ---------------------------------------------------------------------------
    # Shallow RandomForest -- feature importance
    # ---------------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=20,
        class_weight="balanced", random_state=RNG,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print("\nRandomForest holdout report:")
    report = classification_report(y_test, rf_pred, labels=PRIO_ORDER)
    print(report)

    importance = (
        pd.Series(rf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(importance.index[::-1], importance.values[::-1], color=CAT_COLORS["P2"])
    ax.set_title("RandomForest feature importance — P2-consult orders, Scan Priority outcome",
                 fontsize=11, color=INK_PRIMARY, loc="left")
    ax.set_xlabel("Gini importance")
    fig.subplots_adjust(left=0.42, right=0.95, top=0.93, bottom=0.08)
    fig.savefig(os.path.join(OUT, "nni_p2_scan_priority_rf_importance.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---------------------------------------------------------------------------
    # Single shallow DecisionTree -- readable splits
    # ---------------------------------------------------------------------------
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=RNG)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    print("\nDecisionTree holdout report:")
    dt_report = classification_report(y_test, dt_pred, labels=PRIO_ORDER)
    print(dt_report)

    tree_text = export_text(dt, feature_names=list(X.columns), class_names=list(dt.classes_))
    print("\nDecision tree splits:")
    print(tree_text)
    with open(os.path.join(OUT, "nni_p2_scan_priority_rf_tree.txt"), "w") as f:
        f.write(tree_text)

    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        dt, feature_names=list(X.columns), class_names=list(dt.classes_),
        filled=True, fontsize=8, ax=ax, impurity=False,
    )
    fig.savefig(os.path.join(OUT, "nni_p2_scan_priority_rf_tree.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---------------------------------------------------------------------------
    # Excel summary
    # ---------------------------------------------------------------------------
    cm = confusion_matrix(y_test, rf_pred, labels=PRIO_ORDER)
    cm_df = pd.DataFrame(cm, index=[f"actual {p}" for p in PRIO_ORDER], columns=[f"pred {p}" for p in PRIO_ORDER])

    with pd.ExcelWriter(os.path.join(OUT, "nni_p2_scan_priority_rf.xlsx"), engine="openpyxl") as writer:
        importance.rename("importance").reset_index().rename(columns={"index": "feature"}).to_excel(
            writer, sheet_name="feature_importance", index=False
        )
        cm_df.to_excel(writer, sheet_name="confusion_matrix")
        y.value_counts().rename("n").reset_index().rename(columns={"index": "Scan Priority"}).to_excel(
            writer, sheet_name="class_balance", index=False
        )
        cohort.assign(**{c: feature_df[c] for c in NUMERIC_FEATURES}).to_excel(
            writer, sheet_name="cohort_raw", index=False
        )
        for sheet in writer.sheets.values():
            auto_col_width(sheet)

    print(f"\nWrote {OUT}/nni_p2_scan_priority_rf.xlsx")
    print(f"Wrote {OUT}/nni_p2_scan_priority_rf_importance.png")
    print(f"Wrote {OUT}/nni_p2_scan_priority_rf_tree.png")
    print(f"Wrote {OUT}/nni_p2_scan_priority_rf_tree.txt")


if __name__ == "__main__":
    main()
