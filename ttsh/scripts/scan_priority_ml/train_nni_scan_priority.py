"""Train and evaluate scan-priority models for the NNI CT queue.

The decision point is the moment the imaging order is placed.  The script
joins NNI orders to the full All Patients extract by CSN and uses the fields
confirmed by the project owner as visible before the scan, including the
72-hour reattendance and HD/ICU flags. It parses only discrete order-form
answers and excludes free-text response fields; ``ED Scan Priority`` is the
target itself.

Models
------
* Triage-acuity baseline
* Regularised multinomial logistic regression using structured tabular fields
* Histogram gradient boosting on the structured features

The split is chronological: Nov 2025--Feb 2026 train, Mar validate, and Apr
2026 is an untouched final test month.  Outputs are written below analysis/
and the fitted models are saved below models/.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


BASE = Path(__file__).resolve().parents[2]
ORDERS_PATH = BASE / "data" / "nni_cleaned_orders.csv"
ALL_PT_PATH = BASE / "merged" / "all_pt_combined.xlsx"
OUT = BASE / "analysis"
MODELS = BASE / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

PRIORITIES = ["P1", "P2", "P3"]
TARGET = "Scan Priority"
RNG = 42

# These are deliberately restricted to fields intended to exist at the order
# decision point.  The three flags below are included at the owner's request.
NUMERIC = [
    "age_years", "order_hour", "arrival_to_order_mins",
    "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted",
    "HD/ICU from GW within 24H", "kiv_sob", "kiv_chest_pain",
    "kiv_one_sided_weakness", "kiv_major_bleeding",
]
CATEGORICAL = [
    "Triage Acuity", "Consult Acuity", "Sex", "Arrival Method",
    "First ED Care Area", "Primary Care Area", "Classification Type", "CFS (ED)",
    "Order Type", "Ordering Mode", "proc_group",
    "primary_dx_group", "mode_of_transport", "parental_consent", "renal_risk_factor",
    "contrast_allergy", "polypharmacy_allergy", "metformin", "asthma_history",
    "egfr_pathway", "pregnancy_status", "pregnancy_detail", "contrast_guideline_ack",
]


def earliest_datetime(date_value: object, time_value: object) -> pd.Timestamp:
    """Return earliest timestamp from order rows merged with '; '-separated values."""
    dates = str(date_value).split("; ")
    times = str(time_value).split("; ")
    candidates = []
    for i in range(max(len(dates), len(times))):
        d = dates[min(i, len(dates) - 1)]
        t = times[min(i, len(times) - 1)]
        value = pd.to_datetime(f"{d} {t}", errors="coerce", dayfirst=True)
        if pd.notna(value):
            candidates.append(value)
    return min(candidates) if candidates else pd.NaT


def order_datetime_series(data: pd.DataFrame) -> pd.Series:
    """Vectorised timestamp from the first (chronologically earliest) merged order."""
    dates = data["Parent Order Date"].astype(str).str.split("; ").str[0]
    times = data["Parent Order Time"].astype(str).str.split("; ").str[0]
    return pd.to_datetime(dates + " " + times, errors="coerce", dayfirst=True)


def parse_answers(value: object) -> dict[str, str]:
    """Parse only the first order's response block, excluding the label later."""
    answers: dict[str, str] = {}
    for line in str(value).split("; ")[0].splitlines():
        key_value = re.split(r"::|:", line, maxsplit=1)
        if len(key_value) == 2:
            answers[key_value[0].strip()] = key_value[1].strip()
    return answers


def add_parsed_response_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add only discrete order-form answers; intentionally exclude free text."""
    parsed = data["Ordering Question Responses"].map(parse_answers)
    questions = {
        "mode_of_transport": "Mode of transport",
        "parental_consent": "Is parental/ 3rd party consent required for the study?",
        "renal_risk_factor": "Does the patient have risk factors for renal disease eg. hypertension, diabetes, is age above 60 years old (based on respective hospitals' guidelines)?",
        "contrast_allergy": "Does patient have allergy to iodinated/ CT contrast?",
        "polypharmacy_allergy": "Is the patient allergic to 3 or more different classes of drugs?",
        "metformin": "Is the patient on medication containing Metformin?",
        "asthma_history": "Does the patient have history of asthma (exclude childhood asthma and COPD)?",
        "egfr_pathway": "Please select",
        "pregnancy_status": "Is the patient pregnant at the point of imaging request?",
        "pregnancy_detail": "Please provide additional detail",
        "contrast_guideline_ack": "Kindly refer to respective hospital guidelines.",
    }
    for feature, question in questions.items():
        data[feature] = parsed.map(lambda values: values.get(question, "Missing"))
    return data


def procedure_group(description: object) -> str:
    text = str(description).upper()
    parts = []
    if "BRAIN" in text and not any(x in text for x in ("ANGIOGRAM", "VENOGRAM", "PERFUSION")):
        parts.append("brain")
    if "FACE" in text:
        parts.append("face")
    if "CERVICAL" in text or "C-SPINE" in text:
        parts.append("c_spine")
    if any(x in text for x in ("ANGIOGRAM", "VENOGRAM", "PERFUSION")):
        parts.append("angio_perfusion")
    return "+".join(parts) if parts else "other"


def top_group(series: pd.Series, n: int = 30) -> pd.Series:
    top = series.value_counts().head(n).index
    return series.where(series.isin(top), "Other").fillna("Missing")


def metric_rows(name: str, y_true: pd.Series, prediction: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    rows = [{
        "model": name,
        "metric": "accuracy",
        "value": accuracy_score(y_true, prediction),
    }, {
        "model": name,
        "metric": "balanced_accuracy",
        "value": balanced_accuracy_score(y_true, prediction),
    }, {
        "model": name,
        "metric": "macro_f1",
        "value": f1_score(y_true, prediction, labels=PRIORITIES, average="macro", zero_division=0),
    }]
    for i, priority in enumerate(PRIORITIES):
        rows.append({
            "model": name, "metric": f"brier_{priority}",
            "value": brier_score_loss((y_true == priority).astype(int), probabilities[:, i]),
        })
    return rows


def main() -> None:
    print("Loading NNI orders and All Patients...")
    orders = pd.read_csv(ORDERS_PATH, dtype={"CSN": str})
    all_pt = pd.read_excel(ALL_PT_PATH, sheet_name="All Patients", dtype={"CSN": str})
    all_pt["Consult Acuity"] = all_pt["Consult Acuity"].replace({"(P) P2": "P2"})

    # Only take listed columns, preventing duplicate clinical-order fields.
    patient_columns = [
        "CSN", "Age at Visit", "Sex", "Arrival Date", "Arrival Time", "Arrival Method",
        "Chief Complaint", "KIV Uptriage", "Triage Acuity", "Consult Acuity",
        "First ED Care Area", "Primary Care Area", "Classification Type", "CFS (ED)",
        "ED Primary Dx ICD10", "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted",
        "HD/ICU from GW within 24H",
    ]
    # Arrival fields exist in both extracts.  Keep the NNI value as the
    # canonical value and suffix the duplicate All PT copy for traceability.
    data = orders.merge(
        all_pt[patient_columns].drop_duplicates("CSN"), on="CSN", how="left",
        validate="m:1", suffixes=("", "_allpt"),
    )
    data = data[data[TARGET].isin(PRIORITIES)].copy()
    data["order_datetime"] = order_datetime_series(data)
    data = data[data["order_datetime"].notna()].copy()
    data["month_period"] = data["order_datetime"].dt.to_period("M").astype(str)
    data["age_years"] = pd.to_numeric(data["Age at Visit"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    data["order_hour"] = data["order_datetime"].dt.hour
    arrival = pd.to_datetime(data["Arrival Date"].astype(str) + " " + data["Arrival Time"].astype(str), errors="coerce", dayfirst=True)
    data["arrival_to_order_mins"] = (data["order_datetime"] - arrival).dt.total_seconds().div(60).clip(-60, 24 * 60)
    data["proc_group"] = data["Order Description"].map(procedure_group)
    data["primary_dx_group"] = top_group(data["Primary ED Dx"])
    data = add_parsed_response_features(data)
    for flag, label in [("sob", "SOB"), ("chest_pain", "Chest pain"), ("one_sided_weakness", "One-sided weakness"), ("major_bleeding", "Major bleeding")]:
        data[f"kiv_{flag}"] = data["KIV Uptriage"].fillna("").str.contains(label, case=False, regex=False).astype(int)
    for col in ["72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted", "HD/ICU from GW within 24H"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    train = data[data["month_period"].isin(["2025-11", "2025-12", "2026-01", "2026-02"])].copy()
    valid = data[data["month_period"] == "2026-03"].copy()
    test = data[data["month_period"] == "2026-04"].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise RuntimeError(f"Chronological split failed: train={len(train)}, validation={len(valid)}, test={len(test)}")
    print(f"Rows: train={len(train):,}, validation={len(valid):,}, test={len(test):,}")

    # The baseline represents a simple current workflow mapping.
    baseline_pred = test["Triage Acuity"].where(test["Triage Acuity"].isin(PRIORITIES), "P2").to_numpy()
    baseline_prob = np.full((len(test), 3), 0.1)
    baseline_prob[np.arange(len(test)), [PRIORITIES.index(x) for x in baseline_pred]] = 0.8

    tab_preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=8))]), CATEGORICAL),
    ])
    tab_train = tab_preprocessor.fit_transform(train)
    logit = LogisticRegression(C=0.4, class_weight="balanced", max_iter=3000, random_state=RNG)
    logit.fit(tab_train, train[TARGET])
    logit_test = tab_preprocessor.transform(test)
    logit_pred, logit_prob = logit.predict(logit_test), logit.predict_proba(logit_test)

    # Native categorical handling is the strongest available structured model
    # in the local environment; CatBoost can be added later without changing
    # the cohort or evaluation contract.
    tree_preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan))]), CATEGORICAL),
    ])
    tree_train = tree_preprocessor.fit_transform(train)
    tree_test = tree_preprocessor.transform(test)
    categorical_mask = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)
    tree = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, categorical_features=categorical_mask, random_state=RNG)
    tree.fit(tree_train, train[TARGET], sample_weight=compute_sample_weight("balanced", train[TARGET]))
    tree_pred, tree_prob = tree.predict(tree_test), tree.predict_proba(tree_test)

    metric_data = []
    reports = []
    outcomes = [("triage_baseline", baseline_pred, baseline_prob), ("logistic_structured", logit_pred, logit_prob), ("hist_gradient_tabular", tree_pred, tree_prob)]
    for name, pred, prob in outcomes:
        metric_data.extend(metric_rows(name, test[TARGET], pred, prob))
        report = classification_report(test[TARGET], pred, labels=PRIORITIES, output_dict=True, zero_division=0)
        for priority in PRIORITIES:
            reports.append({"model": name, "priority": priority, **report[priority]})

    test_predictions = test[["CSN", "month", "order_datetime", TARGET, "Triage Acuity", "Consult Acuity", "Order Description", "Primary ED Dx"]].copy()
    for name, pred, prob in outcomes:
        test_predictions[f"{name}_prediction"] = pred
        for i, priority in enumerate(PRIORITIES):
            test_predictions[f"{name}_{priority}_probability"] = prob[:, i]
    feature_catalogue = pd.DataFrame([
        {"feature": col, "type": "numeric", "source": "NNI / All PT", "included": "yes"} for col in NUMERIC
    ] + [
        {"feature": col, "type": "categorical", "source": "NNI / All PT", "included": "yes"} for col in CATEGORICAL
    ])
    split_table = pd.DataFrame([
        {"split": "train", "months": "2025-11 to 2026-02", "rows": len(train), "unique_csn": train["CSN"].nunique()},
        {"split": "validation", "months": "2026-03", "rows": len(valid), "unique_csn": valid["CSN"].nunique()},
        {"split": "final_test", "months": "2026-04", "rows": len(test), "unique_csn": test["CSN"].nunique()},
    ])
    class_balance = pd.concat([x[TARGET].value_counts().rename(name) for name, x in [("train", train), ("validation", valid), ("final_test", test)]], axis=1).reindex(PRIORITIES).fillna(0).astype(int)

    output_path = OUT / "nni_scan_priority_model_training.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(metric_data).to_excel(writer, sheet_name="test_metrics", index=False)
        pd.DataFrame(reports).to_excel(writer, sheet_name="test_per_class", index=False)
        class_balance.to_excel(writer, sheet_name="class_balance")
        split_table.to_excel(writer, sheet_name="split_definition", index=False)
        feature_catalogue.to_excel(writer, sheet_name="features", index=False)
        test_predictions.to_excel(writer, sheet_name="test_predictions", index=False)
        for name, pred, _ in outcomes:
            pd.DataFrame(confusion_matrix(test[TARGET], pred, labels=PRIORITIES), index=[f"actual_{x}" for x in PRIORITIES], columns=[f"predicted_{x}" for x in PRIORITIES]).to_excel(writer, sheet_name=f"cm_{name[:25]}")
    joblib.dump({"tab_preprocessor": tab_preprocessor, "model": logit, "priorities": PRIORITIES}, MODELS / "nni_scan_priority_logistic.joblib")
    joblib.dump({"preprocessor": tree_preprocessor, "model": tree, "priorities": PRIORITIES}, MODELS / "nni_scan_priority_hist_gradient.joblib")
    print(f"Wrote {output_path}")
    print(f"Saved models under {MODELS}")
    print(pd.DataFrame(metric_data).pivot(index="model", columns="metric", values="value").round(3))


if __name__ == "__main__":
    main()
