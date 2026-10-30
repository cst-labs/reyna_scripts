"""Proper grid search for RandomForest and ExtraTrees on the six-month NNI
scan-priority cohort.

The earlier tune_nni_scan_priority_models.py only compared two
min_samples_leaf values per family with n_estimators fixed at 30. This
widens the grid (n_estimators, max_depth, min_samples_leaf) while reusing
the same cohort, one-hot preprocessor, and grouped/stratified split so
results are comparable.

Outputs: analysis/nni_rf_et_scan_priority_tuning.xlsx
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from train_nni_scan_priority import MODELS, OUT, PRIORITIES, RNG, TARGET
from tune_nni_scan_priority_models import cohort, score, tab_onehot

GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf, "max_features": "sqrt"}
    for n in [100, 300]
    for d in [8, 20, None]
    for leaf in [2, 6, 12]
]


def main() -> None:
    print("Preparing structured cohort...", flush=True)
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

    trials: list[dict] = []
    finalists: dict[str, dict] = {}
    for family, estimator in [("extra_trees_tabular", ExtraTreesClassifier), ("random_forest_tabular", RandomForestClassifier)]:
        print(f"Tuning {family} over {len(GRID)} combinations...", flush=True)
        best, best_f1 = None, -1.0
        for params in GRID:
            model = estimator(**params, class_weight="balanced_subsample", n_jobs=-1, random_state=RNG)
            model.fit(xtr, train[TARGET])
            result = score(valid[TARGET], model.predict(xva))
            trials.append({"family": family, "parameters": str(params), **result})
            if result["macro_f1"] > best_f1:
                best_f1, best = result["macro_f1"], params
        finalists[family] = best
        print(f"  best {family}: {best} (valid macro_f1={best_f1:.3f})", flush=True)

    # Refit each winner on the full development set and score the untouched test set once.
    prep = tab_onehot()
    xdev = prep.fit_transform(dev)
    xte = prep.transform(test)

    final_rows, per_priority_rows = [], []
    confusion_tables: dict[str, pd.DataFrame] = {}
    predictions = test[["CSN", "order_datetime", TARGET, "Order Description", "Primary ED Dx"]].copy()
    for family, estimator in [("extra_trees_tabular", ExtraTreesClassifier), ("random_forest_tabular", RandomForestClassifier)]:
        params = finalists[family]
        model = estimator(**params, class_weight="balanced_subsample", n_jobs=-1, random_state=RNG)
        model.fit(xdev, dev[TARGET])
        pred = model.predict(xte)
        final_rows.append({"family": family, "selected_parameters": str(params), **score(test[TARGET], pred)})
        predictions[f"{family}_prediction"] = pred
        report = classification_report(test[TARGET], pred, labels=PRIORITIES, output_dict=True, zero_division=0)
        for priority in PRIORITIES:
            per_priority_rows.append({"family": family, "scan_priority": priority, **report[priority]})
        confusion_tables[family] = pd.DataFrame(
            confusion_matrix(test[TARGET], pred, labels=PRIORITIES),
            index=[f"actual_{p}" for p in PRIORITIES], columns=[f"predicted_{p}" for p in PRIORITIES],
        )
        joblib.dump({"preprocessor": prep, "model": model, "priorities": PRIORITIES}, MODELS / f"nni_scan_priority_regridded_{family}.joblib")

    output = OUT / "nni_rf_et_scan_priority_tuning.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(trials).sort_values(["family", "macro_f1"], ascending=[True, False]).to_excel(writer, sheet_name="inner_tuning_trials", index=False)
        pd.DataFrame(final_rows).sort_values("macro_f1", ascending=False).to_excel(writer, sheet_name="six_month_test", index=False)
        pd.DataFrame(per_priority_rows).to_excel(writer, sheet_name="test_by_scan_priority", index=False)
        predictions.to_excel(writer, sheet_name="test_predictions", index=False)
        for family, table in confusion_tables.items():
            table.to_excel(writer, sheet_name=f"cm_{family[:25]}")
        pd.DataFrame([{
            "cohort_months": "2025-11 to 2026-04", "outer_split": "StratifiedGroupKFold 80/20 by CSN",
            "inner_tuning_split": "StratifiedGroupKFold 75/25 of development by CSN", "selection_metric": "macro_f1",
            "grid_size_per_family": len(GRID), "development_rows": len(dev), "test_rows": len(test),
        }]).to_excel(writer, sheet_name="protocol", index=False)

    print()
    print("Final test-set metrics:")
    print(pd.DataFrame(final_rows).sort_values("macro_f1", ascending=False).to_string(index=False))
    print()
    print("Precision / recall / f1 by scan priority (test set):")
    print(pd.DataFrame(per_priority_rows)[["family", "scan_priority", "precision", "recall", "f1-score", "support"]].to_string(index=False))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
