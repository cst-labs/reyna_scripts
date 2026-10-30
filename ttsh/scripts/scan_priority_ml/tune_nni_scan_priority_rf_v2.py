"""Hyperparameter tuning (v2) for RandomForest on all NNI orders with a known
Scan Priority (not filtered to P2 consult acuity).

Same experimental setup as tune_nni_scan_priority_models.py: all Nov 2025--
Apr 2026 rows, grouped+stratified 80/20 train/test split by CSN, with a
second grouped+stratified split inside the 80% development portion for
hyperparameter selection. Only RandomForest is tuned here, with a real grid
(the shared script's current RF entry is a single placeholder trial).

Outputs: analysis/nni_scan_priority_rf_tune_v2.xlsx
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_nni_scan_priority import (
    ALL_PT_PATH, CATEGORICAL, MODELS, NUMERIC, ORDERS_PATH, OUT, PRIORITIES,
    RNG, TARGET, add_parsed_response_features, order_datetime_series,
    procedure_group, top_group,
)
import joblib

GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
    for n in [200, 500]
    for d in [8, 16, None]
    for leaf in [5, 20]
]


def cohort() -> pd.DataFrame:
    orders = pd.read_csv(ORDERS_PATH, dtype={"CSN": str})
    patient = pd.read_excel(ALL_PT_PATH, sheet_name="All Patients", dtype={"CSN": str})
    patient["Consult Acuity"] = patient["Consult Acuity"].replace({"(P) P2": "P2"})
    cols = [
        "CSN", "Age at Visit", "Sex", "Arrival Date", "Arrival Time", "Arrival Method",
        "KIV Uptriage", "Triage Acuity", "Consult Acuity", "First ED Care Area",
        "Primary Care Area", "Classification Type", "CFS (ED)", "ED Primary Dx ICD10",
        "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted", "HD/ICU from GW within 24H",
    ]
    data = orders.merge(patient[cols].drop_duplicates("CSN"), on="CSN", how="left", validate="m:1", suffixes=("", "_allpt"))
    data = data[data[TARGET].isin(PRIORITIES)].copy()
    data["order_datetime"] = order_datetime_series(data)
    data = data[data["order_datetime"].notna()].copy()
    data["age_years"] = pd.to_numeric(data["Age at Visit"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    data["order_hour"] = data["order_datetime"].dt.hour
    arrival = pd.to_datetime(data["Arrival Date"].astype(str) + " " + data["Arrival Time"].astype(str), errors="coerce", dayfirst=True)
    data["arrival_to_order_mins"] = (data["order_datetime"] - arrival).dt.total_seconds().div(60).clip(-60, 1440)
    data["proc_group"] = data["Order Description"].map(procedure_group)
    data["primary_dx_group"] = top_group(data["Primary ED Dx"])
    data = add_parsed_response_features(data)
    for key, phrase in [("sob", "SOB"), ("chest_pain", "Chest pain"), ("one_sided_weakness", "One-sided weakness"), ("major_bleeding", "Major bleeding")]:
        data[f"kiv_{key}"] = data["KIV Uptriage"].fillna("").str.contains(phrase, case=False, regex=False).astype(int)
    for c in ["72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted", "HD/ICU from GW within 24H"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


def tab_onehot():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=8))]), CATEGORICAL),
    ])


def main() -> None:
    print("Preparing structured cohort (all Scan Priority cases)...", flush=True)
    data = cohort()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
    dev_i, test_i = next(outer.split(data, data[TARGET], groups=data["CSN"]))
    dev, test = data.iloc[dev_i].copy(), data.iloc[test_i].copy()
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RNG + 1)
    train_i, valid_i = next(inner.split(dev, dev[TARGET], groups=dev["CSN"]))
    train, valid = dev.iloc[train_i].copy(), dev.iloc[valid_i].copy()
    print(f"Split: train={len(train)}, validation={len(valid)}, test={len(test)}", flush=True)

    prep = tab_onehot()
    xtr = prep.fit_transform(train)
    xva = prep.transform(valid)

    trials = []
    best_params, best_f1 = None, -1.0
    for params in GRID:
        model = RandomForestClassifier(**params, class_weight="balanced_subsample", n_jobs=-1, random_state=RNG)
        model.fit(xtr, train[TARGET])
        pred = model.predict(xva)
        macro_f1 = f1_score(valid[TARGET], pred, labels=PRIORITIES, average="macro", zero_division=0)
        trials.append({"parameters": str(params), "valid_macro_f1": macro_f1})
        print(f"  {params} -> valid macro_f1={macro_f1:.3f}", flush=True)
        if macro_f1 > best_f1:
            best_f1, best_params = macro_f1, params
    print(f"Best params on inner validation: {best_params} (macro_f1={best_f1:.3f})")

    # Refit on all development rows with the selected hyperparameters, score
    # the untouched test set exactly once.
    prep = tab_onehot()
    xdev = prep.fit_transform(dev)
    xte = prep.transform(test)
    final_model = RandomForestClassifier(**best_params, class_weight="balanced_subsample", n_jobs=-1, random_state=RNG)
    final_model.fit(xdev, dev[TARGET])
    test_pred = final_model.predict(xte)
    joblib.dump({"preprocessor": prep, "model": final_model, "priorities": PRIORITIES}, MODELS / "nni_scan_priority_tuned_random_forest_v2.joblib")

    report = classification_report(test[TARGET], test_pred, labels=PRIORITIES, output_dict=True, zero_division=0)
    print("\nTest-set classification report (tune v2 RandomForest, all cases):")
    print(classification_report(test[TARGET], test_pred, labels=PRIORITIES, zero_division=0))

    per_class = pd.DataFrame([
        {"priority": p, "precision": report[p]["precision"], "recall": report[p]["recall"],
         "f1_score": report[p]["f1-score"], "support": report[p]["support"]}
        for p in PRIORITIES
    ])
    overall = pd.DataFrame([
        {"metric": "accuracy", "value": report["accuracy"]},
        {"metric": "macro_f1", "value": report["macro avg"]["f1-score"]},
        {"metric": "weighted_f1", "value": report["weighted avg"]["f1-score"]},
    ])
    cm_df = pd.DataFrame(
        confusion_matrix(test[TARGET], test_pred, labels=PRIORITIES),
        index=[f"actual_{p}" for p in PRIORITIES], columns=[f"predicted_{p}" for p in PRIORITIES],
    )

    output = OUT / "nni_scan_priority_rf_tune_v2.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        per_class.to_excel(writer, sheet_name="test_precision_recall_f1", index=False)
        overall.to_excel(writer, sheet_name="test_overall", index=False)
        cm_df.to_excel(writer, sheet_name="test_confusion_matrix")
        pd.DataFrame(trials).sort_values("valid_macro_f1", ascending=False).to_excel(writer, sheet_name="grid_search_trials", index=False)
        pd.DataFrame([{
            "cohort_months": "2025-11 to 2026-04", "outer_split": "StratifiedGroupKFold 80/20 by CSN",
            "inner_tuning_split": "StratifiedGroupKFold 75/25 of development by CSN", "selection_metric": "macro_f1",
            "selected_parameters": str(best_params), "development_rows": len(dev), "test_rows": len(test),
            "development_unique_csn": dev.CSN.nunique(), "test_unique_csn": test.CSN.nunique(),
        }]).to_excel(writer, sheet_name="protocol", index=False)

    print(f"\nWrote {output}")
    print("\nPer-class precision / recall / f1 (test set):")
    print(per_class.to_string(index=False))


if __name__ == "__main__":
    main()
