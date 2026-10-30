"""Hyperparameter tuning (v2) for RandomForest on the P2-consult-acuity NNI
cohort, Scan Priority target.

Reuses the exact cohort/feature matrix from nni_p2_scan_priority_rf.py
(build_p2_cohort_features) instead of the fixed-depth RF in that script.
Grid search picks hyperparameters on an inner validation split carved out of
an 80% development set; the remaining 20% (grouped by CSN, stratified by
Scan Priority) is an untouched test set scored only once at the end.

Outputs: analysis/nni_p2_scan_priority_rf_tune_v2.xlsx
"""
from __future__ import annotations

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from nni_p2_scan_priority_rf import OUT, PRIO_ORDER, RNG, build_p2_cohort_features

GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
    for n in [200, 500]
    for d in [4, 8, None]
    for leaf in [5, 20]
]


def split_dev_test(cohort, X, y):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
    dev_i, test_i = next(outer.split(X, y, groups=cohort["CSN"]))
    return dev_i, test_i


def main() -> None:
    cohort, feature_df, X, y = build_p2_cohort_features()

    dev_i, test_i = split_dev_test(cohort, X, y)
    X_dev, y_dev, csn_dev = X.iloc[dev_i], y.iloc[dev_i], cohort["CSN"].iloc[dev_i]
    X_test, y_test = X.iloc[test_i], y.iloc[test_i]

    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RNG + 1)
    train_i, valid_i = next(inner.split(X_dev, y_dev, groups=csn_dev))
    X_train, y_train = X_dev.iloc[train_i], y_dev.iloc[train_i]
    X_valid, y_valid = X_dev.iloc[valid_i], y_dev.iloc[valid_i]

    trials = []
    best_params, best_f1 = None, -1.0
    for params in GRID:
        model = RandomForestClassifier(
            **params, class_weight="balanced_subsample", n_jobs=-1, random_state=RNG,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        macro_f1 = f1_score(y_valid, pred, labels=PRIO_ORDER, average="macro", zero_division=0)
        trials.append({"parameters": str(params), "valid_macro_f1": macro_f1})
        if macro_f1 > best_f1:
            best_f1, best_params = macro_f1, params
    print(f"Best params on inner validation: {best_params} (macro_f1={best_f1:.3f})")

    # Refit on the full development set with the selected hyperparameters,
    # then score the untouched test set exactly once.
    final_model = RandomForestClassifier(
        **best_params, class_weight="balanced_subsample", n_jobs=-1, random_state=RNG,
    )
    final_model.fit(X_dev, y_dev)
    test_pred = final_model.predict(X_test)

    report = classification_report(y_test, test_pred, labels=PRIO_ORDER, output_dict=True, zero_division=0)
    print("\nTest-set classification report (tune v2 RandomForest):")
    print(classification_report(y_test, test_pred, labels=PRIO_ORDER, zero_division=0))

    per_class = pd.DataFrame([
        {"priority": p, "precision": report[p]["precision"], "recall": report[p]["recall"],
         "f1_score": report[p]["f1-score"], "support": report[p]["support"]}
        for p in PRIO_ORDER
    ])
    overall = pd.DataFrame([{
        "metric": "accuracy", "value": report["accuracy"],
    }, {
        "metric": "macro_f1", "value": report["macro avg"]["f1-score"],
    }, {
        "metric": "weighted_f1", "value": report["weighted avg"]["f1-score"],
    }])
    cm = confusion_matrix(y_test, test_pred, labels=PRIO_ORDER)
    cm_df = pd.DataFrame(cm, index=[f"actual_{p}" for p in PRIO_ORDER], columns=[f"predicted_{p}" for p in PRIO_ORDER])

    output = os.path.join(OUT, "nni_p2_scan_priority_rf_tune_v2.xlsx")
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        per_class.to_excel(writer, sheet_name="test_precision_recall_f1", index=False)
        overall.to_excel(writer, sheet_name="test_overall", index=False)
        cm_df.to_excel(writer, sheet_name="test_confusion_matrix")
        pd.DataFrame(trials).sort_values("valid_macro_f1", ascending=False).to_excel(
            writer, sheet_name="grid_search_trials", index=False
        )
        pd.DataFrame([{"selected_parameters": str(best_params), "development_rows": len(X_dev),
                        "test_rows": len(X_test), "development_unique_csn": csn_dev.nunique(),
                        "test_unique_csn": cohort["CSN"].iloc[test_i].nunique()}]).to_excel(
            writer, sheet_name="protocol", index=False
        )

    print(f"\nWrote {output}")
    print("\nPer-class precision / recall / f1 (test set):")
    print(per_class.to_string(index=False))


if __name__ == "__main__":
    main()
