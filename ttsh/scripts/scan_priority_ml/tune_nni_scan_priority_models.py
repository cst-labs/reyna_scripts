"""Six-month hyperparameter search for NNI scan-priority candidates.

All Nov 2025--Apr 2026 rows contribute to a grouped, stratified 80/20
train/test split.  A second grouped split inside the 80% development portion
selects hyperparameters.  This avoids having one patient's repeated orders in
both sides while ensuring every month is represented in both train and test.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from train_nni_scan_priority import (
    ALL_PT_PATH, CATEGORICAL, MODELS, NUMERIC, ORDERS_PATH, OUT, PRIORITIES,
    RNG, TARGET, add_parsed_response_features, order_datetime_series, procedure_group, top_group,
)


def cohort() -> pd.DataFrame:
    orders = pd.read_csv(ORDERS_PATH, dtype={"CSN": str})
    patient = pd.read_excel(ALL_PT_PATH, sheet_name="All Patients", dtype={"CSN": str})
    print("  loaded source files", flush=True)
    patient["Consult Acuity"] = patient["Consult Acuity"].replace({"(P) P2": "P2"})
    cols = [
        "CSN", "Age at Visit", "Sex", "Arrival Date", "Arrival Time", "Arrival Method",
        "KIV Uptriage", "Triage Acuity", "Consult Acuity", "First ED Care Area",
        "Primary Care Area", "Classification Type", "CFS (ED)", "ED Primary Dx ICD10",
        "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted", "HD/ICU from GW within 24H",
    ]
    data = orders.merge(patient[cols].drop_duplicates("CSN"), on="CSN", how="left", validate="m:1", suffixes=("", "_allpt"))
    data = data[data[TARGET].isin(PRIORITIES)].copy()
    print("  merged patient fields", flush=True)
    data["order_datetime"] = order_datetime_series(data)
    print("  parsed order timestamps", flush=True)
    data = data[data["order_datetime"].notna()].copy()
    data["month_period"] = data["order_datetime"].dt.to_period("M").astype(str)
    data["age_years"] = pd.to_numeric(data["Age at Visit"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    data["order_hour"] = data["order_datetime"].dt.hour
    arrival = pd.to_datetime(data["Arrival Date"].astype(str) + " " + data["Arrival Time"].astype(str), errors="coerce", dayfirst=True)
    data["arrival_to_order_mins"] = (data["order_datetime"] - arrival).dt.total_seconds().div(60).clip(-60, 1440)
    data["proc_group"] = data["Order Description"].map(procedure_group)
    data["primary_dx_group"] = top_group(data["Primary ED Dx"])
    data = add_parsed_response_features(data)
    print("  parsed order response fields", flush=True)
    for key, phrase in [("sob", "SOB"), ("chest_pain", "Chest pain"), ("one_sided_weakness", "One-sided weakness"), ("major_bleeding", "Major bleeding")]:
        data[f"kiv_{key}"] = data["KIV Uptriage"].fillna("").str.contains(phrase, case=False, regex=False).astype(int)
    for c in ["72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted", "HD/ICU from GW within 24H"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


def score(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    return {"accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p), "macro_f1": f1_score(y, p, labels=PRIORITIES, average="macro", zero_division=0), "p1_recall": recall_score(y, p, labels=PRIORITIES, average=None, zero_division=0)[0]}


def tab_onehot():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=8))]), CATEGORICAL),
    ])


def tab_ordinal():
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan))]), CATEGORICAL),
    ])


def main() -> None:
    print("Preparing structured cohort...", flush=True)
    data = cohort()
    # Outer split: final test uses 20% of all six months, grouped by patient.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
    dev_i, test_i = next(outer.split(data, data[TARGET], groups=data["CSN"]))
    dev, test = data.iloc[dev_i].copy(), data.iloc[test_i].copy()
    # Inner split: tune inside development data only.
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RNG + 1)
    train_i, valid_i = next(inner.split(dev, dev[TARGET], groups=dev["CSN"]))
    train, valid = dev.iloc[train_i].copy(), dev.iloc[valid_i].copy()
    print(f"Split: train={len(train)}, validation={len(valid)}, test={len(test)}", flush=True)
    trials: list[dict] = []
    finalists: dict[str, tuple[dict, object]] = {}

    # Structured multinomial logistic regression.
    prep = tab_onehot(); xtr_tab = prep.fit_transform(train); xva_tab = prep.transform(valid)
    best = None
    print("Tuning logistic regression...", flush=True)
    for c in [0.05, 0.15, 0.4, 1.0, 2.0]:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=RNG)
        model.fit(xtr_tab, train[TARGET]); result = score(valid[TARGET], model.predict(xva_tab))
        trials.append({"family": "logistic_structured", "parameters": f"C={c}", **result})
        if best is None or result["macro_f1"] > best[0]["macro_f1"]: best = (result, {"C": c})
    finalists["logistic_structured"] = (best[1], None)

    # Linear SVM on the same structured, one-hot feature matrix. It has no
    # probability output; this requires a separate calibration step.
    best = None
    print("Tuning linear SVM...", flush=True)
    for c in [0.01, 0.03, 0.1, 0.3, 1.0]:
        model = LinearSVC(C=c, class_weight="balanced", max_iter=10000, random_state=RNG)
        model.fit(xtr_tab, train[TARGET]); result = score(valid[TARGET], model.predict(xva_tab))
        trials.append({"family": "linear_svm_structured", "parameters": f"C={c}; class_weight=balanced", **result})
        if best is None or result["macro_f1"] > best[0]["macro_f1"]: best = (result, {"C": c})
    finalists["linear_svm_structured"] = (best[1], None)

    # Bagged trees on one-hot structured data provide a non-linear, no-text
    # comparator that is distinct from gradient boosting.
    for family, estimator, grid in [
        ("extra_trees_tabular", ExtraTreesClassifier, [{"n_estimators": 10, "max_features": "sqrt", "min_samples_leaf": 6}]),
        ("random_forest_tabular", RandomForestClassifier, [{"n_estimators": 10, "max_features": "sqrt", "min_samples_leaf": 6}]),
    ]:
        print(f"Tuning {family}...", flush=True)
        best = None
        for params in grid:
            model = estimator(**params, class_weight="balanced_subsample", n_jobs=1, random_state=RNG)
            model.fit(xtr_tab, train[TARGET]); result = score(valid[TARGET], model.predict(xva_tab))
            trials.append({"family": family, "parameters": str(params), **result})
            if best is None or result["macro_f1"] > best[0]["macro_f1"]: best = (result, params)
        finalists[family] = (best[1], None)

    # Histogram gradient boosting uses native categorical columns.
    ordinal = tab_ordinal(); xtr_ord = ordinal.fit_transform(train); xva_ord = ordinal.transform(valid)
    cat_mask = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)
    best = None
    print("Tuning histogram gradient boosting...", flush=True)
    for leaves, lr, regularization in [(8, .05, .5)]:
        params = {"max_iter": 30, "max_leaf_nodes": leaves, "learning_rate": lr, "l2_regularization": regularization}
        model = HistGradientBoostingClassifier(**params, categorical_features=cat_mask, random_state=RNG)
        model.fit(xtr_ord, train[TARGET], sample_weight=compute_sample_weight("balanced", train[TARGET]))
        result = score(valid[TARGET], model.predict(xva_ord))
        trials.append({"family": "hist_gradient_tabular", "parameters": str(params), **result})
        if best is None or result["macro_f1"] > best[0]["macro_f1"]: best = (result, params)
    finalists["hist_gradient_tabular"] = (best[1], None)

    # Refit each selected model on all development rows and score the unseen
    # 20% test cohort.
    final_rows = []
    print("Refitting selected models...", flush=True)
    per_priority_rows = []
    confusion_tables: dict[str, pd.DataFrame] = {}
    predictions = test[["CSN", "order_datetime", TARGET, "Order Description", "Primary ED Dx"]].copy()
    prep = tab_onehot(); xdev_tab = prep.fit_transform(dev); xte_tab = prep.transform(test)
    params = finalists["logistic_structured"][0]; model = LogisticRegression(**params, class_weight="balanced", max_iter=3000, random_state=RNG)
    model.fit(xdev_tab, dev[TARGET]); pred = model.predict(xte_tab); final_rows.append({"family": "logistic_structured", "selected_parameters": str(params), **score(test[TARGET], pred)})
    predictions["logistic_structured_prediction"] = pred
    joblib.dump({"tab_preprocessor": prep, "model": model, "priorities": PRIORITIES}, MODELS / "nni_scan_priority_tuned_logistic.joblib")
    params = finalists["linear_svm_structured"][0]; model = LinearSVC(**params, class_weight="balanced", max_iter=10000, random_state=RNG)
    model.fit(xdev_tab, dev[TARGET]); pred = model.predict(xte_tab); final_rows.append({"family": "linear_svm_structured", "selected_parameters": str(params), **score(test[TARGET], pred)})
    predictions["linear_svm_structured_prediction"] = pred
    joblib.dump({"tab_preprocessor": prep, "model": model, "priorities": PRIORITIES, "probabilities": "Not available; calibrate separately before confidence-based use."}, MODELS / "nni_scan_priority_tuned_linear_svm.joblib")
    for family, estimator in [("extra_trees_tabular", ExtraTreesClassifier), ("random_forest_tabular", RandomForestClassifier)]:
        params = finalists[family][0]; model = estimator(**params, class_weight="balanced_subsample", n_jobs=1, random_state=RNG)
        model.fit(xdev_tab, dev[TARGET]); pred = model.predict(xte_tab); final_rows.append({"family": family, "selected_parameters": str(params), **score(test[TARGET], pred)})
        predictions[f"{family}_prediction"] = pred
        joblib.dump({"preprocessor": prep, "model": model, "priorities": PRIORITIES}, MODELS / f"nni_scan_priority_tuned_{family}.joblib")
    ordinal = tab_ordinal(); xdev_ord = ordinal.fit_transform(dev); xte_ord = ordinal.transform(test)
    params = finalists["hist_gradient_tabular"][0]; model = HistGradientBoostingClassifier(**params, categorical_features=cat_mask, random_state=RNG)
    model.fit(xdev_ord, dev[TARGET], sample_weight=compute_sample_weight("balanced", dev[TARGET])); pred = model.predict(xte_ord)
    final_rows.append({"family": "hist_gradient_tabular", "selected_parameters": str(params), **score(test[TARGET], pred)})
    predictions["hist_gradient_tabular_prediction"] = pred
    joblib.dump({"preprocessor": ordinal, "model": model, "priorities": PRIORITIES}, MODELS / "nni_scan_priority_tuned_hist_gradient.joblib")

    # Per-priority performance makes P1/P2/P3 trade-offs visible rather than
    # hiding them behind one aggregate score.
    for row in final_rows:
        family = row["family"]
        predicted = predictions[f"{family}_prediction"]
        report = classification_report(test[TARGET], predicted, labels=PRIORITIES, output_dict=True, zero_division=0)
        for priority in PRIORITIES:
            per_priority_rows.append({"family": family, "actual_scan_priority": priority, **report[priority]})
        confusion_tables[family] = pd.DataFrame(
            confusion_matrix(test[TARGET], predicted, labels=PRIORITIES),
            index=[f"actual_{p}" for p in PRIORITIES], columns=[f"predicted_{p}" for p in PRIORITIES],
        )

    output = OUT / "nni_scan_priority_hyperparameter_tuning.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(trials).sort_values(["family", "macro_f1"], ascending=[True, False]).to_excel(writer, sheet_name="inner_tuning_trials", index=False)
        pd.DataFrame(final_rows).sort_values("macro_f1", ascending=False).to_excel(writer, sheet_name="six_month_test", index=False)
        pd.DataFrame(per_priority_rows).to_excel(writer, sheet_name="test_by_actual_priority", index=False)
        predictions.to_excel(writer, sheet_name="test_predictions", index=False)
        for family, table in confusion_tables.items():
            table.to_excel(writer, sheet_name=f"cm_{family[:25]}")
        pd.DataFrame([{
            "cohort_months": "2025-11 to 2026-04", "outer_split": "StratifiedGroupKFold 80/20 by CSN",
            "inner_tuning_split": "StratifiedGroupKFold 75/25 of development by CSN", "selection_metric": "macro_f1",
            "development_rows": len(dev), "test_rows": len(test), "development_unique_csn": dev.CSN.nunique(), "test_unique_csn": test.CSN.nunique(),
        }]).to_excel(writer, sheet_name="protocol", index=False)
    print(pd.DataFrame(final_rows).sort_values("macro_f1", ascending=False).to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
