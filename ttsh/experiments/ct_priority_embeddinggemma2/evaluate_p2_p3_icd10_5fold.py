"""Full k/vote/adjustment search for ICD10-category templates built on top of
base_plus_care_location_acuity_diagnosis.

EDA on historical_cases.xlsx found ED Primary Dx ICD10 (truncated to its
3-character category) has by far the highest mutual information with the
binary P2-vs-P3 target of any available field (0.0365 vs next-best
Classification Type at 0.0045), and was never evaluated as a model input in
STUDY_REPORT.md. This script adds that derived field (and, in one template,
Classification Type/CFS (ED) alongside it) on top of
base_plus_care_location_acuity_diagnosis (the user's current best-P2-recall
template, see results/care_acuity_diagnosis_5fold/FIVE_FOLD_RESULTS.md).

Unlike every fixed-k/vote/adjustment P2/P3 script so far, this one reopens
the full grid (same K_VALUES/weightings/adjustments as
evaluate_care_acuity_diagnosis_5fold.py) and selects, per template, the
config with the best mean (P2_recall + P3_recall)/2 -- not macro_f1/P1_recall
first -- since the user explicitly said P1 may drop in exchange for P2/P3
separation. The existing base_plus_care_location_acuity_diagnosis grid is
reused from results/care_acuity_diagnosis_5fold/run_metadata.json as the
comparison baseline row, not recomputed.

Per the user's instruction, Primary Care Area is excluded from every new
template's input here -- only First ED Care Area represents "Care location".
The reused baseline row still includes Primary Care Area (it is a frozen
prior result, not recomputed); this asymmetry is called out in the report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import evaluate_care_acuity_diagnosis as focused
import run_experiment as exp
from evaluate_care_acuity_diagnosis_5fold import label_metrics

OUT = exp.RESULTS_DIR / "p2_p3_icd10_5fold"
OUT.mkdir(parents=True, exist_ok=True)
OUT.chmod(0o700)
SEED = 42
WEIGHTINGS = ["uniform", "similarity_weighted"]
ADJUSTMENTS = list(exp.CLASS_ADJUSTMENTS)

# Per user instruction: Primary Care Area is excluded from all new templates.
CARE_LOCATION_FIELDS = [f for f in focused.GROUPS["Care location"] if f != "Primary Care Area"]
DIAGNOSIS_BASE = focused.BASE_FIELDS + CARE_LOCATION_FIELDS + focused.GROUPS["Acuity"] + focused.GROUPS["Diagnosis"]
CARE_ACUITY_BASE = focused.BASE_FIELDS + CARE_LOCATION_FIELDS + focused.GROUPS["Acuity"]

TEMPLATES: dict[str, list[str]] = {
    "diagnosis_base_plus_icd3": DIAGNOSIS_BASE + ["ICD10 Category"],
    "diagnosis_base_plus_icd3_classification_type": DIAGNOSIS_BASE + ["ICD10 Category", "Classification Type"],
    "care_location_acuity_plus_icd3": CARE_ACUITY_BASE + ["ICD10 Category"],
    "diagnosis_base_plus_icd3_cfs": DIAGNOSIS_BASE + ["ICD10 Category", "CFS (ED)"],
}

BASELINE_TEMPLATE = "base_plus_care_location_acuity_diagnosis"
BASELINE_META_PATH = exp.RESULTS_DIR / "care_acuity_diagnosis_5fold" / "run_metadata.json"
BASELINE_WORKBOOK_PATH = exp.RESULTS_DIR / "care_acuity_diagnosis_5fold" / "five_fold_results.xlsx"


def add_icd10_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ICD10 Category"] = df["ED Primary Dx ICD10"].astype(str).str[:3]
    return df


def load_baseline_rows() -> tuple[dict, dict]:
    """Baseline CV row (mean five-fold grid) and test row, reused as-is."""
    meta = json.loads(BASELINE_META_PATH.read_text())
    cv_row = next(r for r in meta["top_five"] if r["template"] == BASELINE_TEMPLATE)
    test_row = next(r for r in meta["test_results"] if r["template"] == BASELINE_TEMPLATE)
    return cv_row, test_row


def get_vectors(model, revision, template: str, fields: list[str], df: pd.DataFrame,
                cache_name: str | None = None) -> np.ndarray:
    ids = df["Case ID"].astype(str).tolist()
    texts = exp.render_cases(df, fields)
    return exp.get_embeddings(model, texts, cache_name or template, ids, revision, argparse.Namespace(
        device="cpu", batch_size=32, local_files_only=True,
    ))


def main() -> None:
    hist = pd.read_excel(exp.DATA_DIR / "historical_cases.xlsx")
    hist = hist[hist["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    hist = add_icd10_category(hist)
    y = hist["Priority Label"].to_numpy()

    test_raw = pd.read_excel(exp.DATA_DIR / "newdata_test_cases.xlsx")
    test = test_raw[test_raw["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    test = add_icd10_category(test)
    y_test = test["Priority Label"].to_numpy()

    if set(hist["Case ID"].astype(str)) & set(test["Case ID"].astype(str)):
        raise RuntimeError("Historical and test Case IDs overlap")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(hist, y, groups=hist["Patient Group ID"]))
    for train_idx, val_idx in splits:
        if set(hist.iloc[train_idx]["Patient Group ID"]) & set(hist.iloc[val_idx]["Patient Group ID"]):
            raise RuntimeError("Patient group leakage between fold train and validation")

    class Args:
        device = "cpu"
        batch_size = 32
        local_files_only = True
    model = exp.embedder(Args())
    revision = exp.model_revision(model)

    fold_rows: list[dict] = []
    hist_vectors: dict[str, np.ndarray] = {}
    for template, fields in TEMPLATES.items():
        vectors = get_vectors(model, revision, template, fields, hist)
        hist_vectors[template] = vectors
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            similarity = vectors[val_idx] @ vectors[train_idx].T
            neighbor_order = np.argsort(-similarity, axis=1)[:, :max(exp.K_VALUES)]
            neighbor_sims = np.take_along_axis(similarity, neighbor_order, axis=1)
            y_train, y_val = y[train_idx], y[val_idx]
            for k in exp.K_VALUES:
                neighbor_labels = y_train[neighbor_order[:, :k]]
                top_sims = neighbor_sims[:, :k]
                for weighting in WEIGHTINGS:
                    for adjustment in ADJUSTMENTS:
                        pred, *_ = exp.knn_from_sorted_neighbors(
                            neighbor_labels, top_sims, y_train, k, weighting, adjustment
                        )
                        scores = label_metrics(y_val, pred)
                        fold_rows.append({
                            "template": template, "fold": fold, "k": k,
                            "weighting": weighting, "class_adjustment": adjustment,
                            **scores,
                        })
        print(f"Completed full grid: {template}", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in fold_df.columns if c not in ("template", "fold", "k", "weighting", "class_adjustment")]
    group_cols = ["template", "k", "weighting", "class_adjustment"]
    mean_df = fold_df.groupby(group_cols, as_index=False)[metric_cols].mean()
    sd_df = fold_df.groupby(group_cols, as_index=False)[metric_cols].std().rename(
        columns={c: f"sd_{c}" for c in metric_cols}
    )
    mean_df = mean_df.merge(sd_df, on=group_cols)

    # Selection rule: best mean (P2_recall + P3_recall)/2 per template, macro_f1
    # as tiebreak -- P1 recall is reported but not part of the ranking, per the
    # user's explicit permission for P1 to drop.
    mean_df["p2_p3_recall_mean"] = (mean_df["P2_recall"] + mean_df["P3_recall"]) / 2
    best_per_template = (mean_df.sort_values(
        ["p2_p3_recall_mean", "macro_f1"], ascending=[False, False], kind="stable")
        .drop_duplicates("template").reset_index(drop=True))

    # Score each template's selected config retrospectively on the test file,
    # refitting the k-NN reference on all historical labeled rows.
    test_rows: list[dict] = []
    for row in best_per_template.to_dict("records"):
        template = row["template"]
        test_vec = get_vectors(model, revision, template, TEMPLATES[template], test)
        similarity = test_vec @ hist_vectors[template].T
        k = int(row["k"])
        neighbor_order = np.argsort(-similarity, axis=1)[:, :k]
        neighbor_sims = np.take_along_axis(similarity, neighbor_order, axis=1)
        neighbor_labels = y[neighbor_order]
        pred, *_ = exp.knn_from_sorted_neighbors(
            neighbor_labels, neighbor_sims, y, k, row["weighting"], row["class_adjustment"]
        )
        scores = label_metrics(y_test, pred)
        test_rows.append({
            "template": template, "k": k, "weighting": row["weighting"],
            "class_adjustment": row["class_adjustment"],
            "cv_macro_f1_mean": row["macro_f1"], "cv_p2_p3_recall_mean": row["p2_p3_recall_mean"],
            **scores,
        })
        print(f"Scored on retrospective test set: {template}", flush=True)

    baseline_cv, baseline_test = load_baseline_rows()
    # top_five in run_metadata.json omits per-label recall; pull the matching
    # mean-grid row (same template/k/weighting/adjustment) from the workbook.
    baseline_grid_df = pd.read_excel(BASELINE_WORKBOOK_PATH, sheet_name="All config means SD")
    baseline_full = baseline_grid_df[
        (baseline_grid_df["template"] == BASELINE_TEMPLATE)
        & (baseline_grid_df["k"] == baseline_cv["k"])
        & (baseline_grid_df["weighting"] == baseline_cv["weighting"])
        & (baseline_grid_df["class_adjustment"] == baseline_cv["class_adjustment"])
    ].iloc[0]
    cv_baseline_row = {
        "template": BASELINE_TEMPLATE, "k": baseline_cv["k"], "weighting": baseline_cv["weighting"],
        "class_adjustment": baseline_cv["class_adjustment"],
        "macro_f1": baseline_cv["cv_macro_f1_mean"],
    }
    for key in ("P1_recall", "P2_recall", "P3_recall", "balanced_accuracy", "accuracy"):
        cv_baseline_row[key] = float(baseline_full[f"validation_{key}"])
    cv_baseline_row["p2_p3_recall_mean"] = (cv_baseline_row["P2_recall"] + cv_baseline_row["P3_recall"]) / 2
    cv_baseline_row["reused_from"] = str(BASELINE_META_PATH.relative_to(exp.RESULTS_DIR.parent))

    best_per_template["reused_from"] = ""
    cv_combined = pd.concat([pd.DataFrame([cv_baseline_row]), best_per_template], ignore_index=True, sort=False)

    test_baseline_row = {
        "template": BASELINE_TEMPLATE, "k": baseline_test.get("k"), "weighting": baseline_test.get("weighting"),
        "class_adjustment": baseline_test.get("class_adjustment"),
        **{k: v for k, v in baseline_test.items() if k not in ("rank", "model", "template", "k", "weighting", "class_adjustment")},
        "reused_from": str(BASELINE_META_PATH.relative_to(exp.RESULTS_DIR.parent)),
    }
    test_df = pd.DataFrame(test_rows)
    test_df["reused_from"] = ""
    test_combined = pd.concat([pd.DataFrame([test_baseline_row]), test_df], ignore_index=True, sort=False)

    lead_cols = ["template", "k", "weighting", "class_adjustment", "macro_f1", "P2_recall", "P3_recall",
                 "P1_recall", "balanced_accuracy", "accuracy", "reused_from"]
    cv_combined = cv_combined[[c for c in lead_cols if c in cv_combined.columns] +
                              [c for c in cv_combined.columns if c not in lead_cols]]
    test_combined = test_combined[[c for c in lead_cols if c in test_combined.columns] +
                                  [c for c in test_combined.columns if c not in lead_cols]]

    baseline_cv_s = cv_combined.set_index("template").loc[BASELINE_TEMPLATE]
    baseline_test_s = test_combined.set_index("template").loc[BASELINE_TEMPLATE]
    deltas = {}
    for template in TEMPLATES:
        cv_row = cv_combined.set_index("template").loc[template]
        test_row = test_combined.set_index("template").loc[template]
        deltas[template] = {
            "cv_macro_f1_delta": float(cv_row["macro_f1"] - baseline_cv_s["macro_f1"]),
            "cv_P2_recall_delta": float(cv_row["P2_recall"] - baseline_cv_s["P2_recall"]),
            "cv_P3_recall_delta": float(cv_row["P3_recall"] - baseline_cv_s["P3_recall"]),
            "cv_P1_recall_delta": float(cv_row["P1_recall"] - baseline_cv_s["P1_recall"]),
            "test_macro_f1_delta": float(test_row["macro_f1"] - baseline_test_s["macro_f1"]),
            "test_P2_recall_delta": float(test_row["P2_recall"] - baseline_test_s["P2_recall"]),
            "test_P3_recall_delta": float(test_row["P3_recall"] - baseline_test_s["P3_recall"]),
            "test_P1_recall_delta": float(test_row["P1_recall"] - baseline_test_s["P1_recall"]),
        }

    workbook = OUT / "results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        fold_df.to_excel(writer, sheet_name="Per-fold full grid (new)", index=False)
        mean_df.to_excel(writer, sheet_name="Mean grid by template config", index=False)
        cv_combined.to_excel(writer, sheet_name="Selected CV config (+baseline)", index=False)
        test_combined.to_excel(writer, sheet_name="Test set results (+baseline)", index=False)
    workbook.chmod(0o600)
    print(f"Workbook written: {workbook}", flush=True)

    meta = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Full k/vote/adjustment grid search for ICD10-category (ED Primary Dx ICD10, "
                   "3-char prefix) templates built on base_plus_care_location_acuity_diagnosis, "
                   "selecting per-template config by mean (P2_recall+P3_recall)/2 rather than "
                   "macro_f1/P1_recall, since the user explicitly authorized a P1 recall drop.",
        "model_id": exp.MODEL_ID, "model_revision": revision,
        "model_snapshot_commit": exp.model_snapshot_commit(),
        "task_prefix": exp.TASK_PREFIX, "dimension": 768,
        "seed": SEED, "k_values": exp.K_VALUES, "weightings": WEIGHTINGS,
        "class_adjustments": exp.CLASS_ADJUSTMENTS,
        "selection_rule": "max mean (P2_recall+P3_recall)/2, macro_f1 tiebreak",
        "derived_field": "ICD10 Category = ED Primary Dx ICD10[:3]",
        "eda_summary": {
            "method": "mutual_info_classif on P2-vs-P3 binary target, historical P2/P3 rows only (n=5381)",
            "icd3_mutual_information": 0.0365,
            "next_best_field": "Classification Type",
            "next_best_mutual_information": 0.0045,
            "test_icd3_category_coverage_vs_historical": 0.947,
        },
        "baseline_template": BASELINE_TEMPLATE,
        "baseline_reused_from": str(BASELINE_META_PATH.relative_to(exp.RESULTS_DIR.parent)),
        "new_templates": TEMPLATES,
        "historical_labeled_n": len(hist),
        "test_labeled_n": len(test),
        "selected_cv_config_by_template": cv_combined.to_dict("records"),
        "test_results_by_template": test_combined.to_dict("records"),
        "deltas_vs_baseline": deltas,
        "artifact_workbook": workbook.name,
    }
    meta_path = OUT / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    meta_path.chmod(0o600)
    print(f"Metadata written: {meta_path}", flush=True)

    print("\nSelected CV config (mean five-fold) + baseline:")
    print(cv_combined[["template", "k", "weighting", "class_adjustment", "macro_f1",
                       "P2_recall", "P3_recall", "P1_recall"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nRetrospective test-set results + baseline:")
    print(test_combined[["template", "k", "weighting", "class_adjustment", "macro_f1",
                         "P2_recall", "P3_recall", "P1_recall"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
