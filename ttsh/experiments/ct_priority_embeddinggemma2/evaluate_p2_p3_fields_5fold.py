"""Fixed-config 5-fold check: do P2/P3-differentiating fields help the best template?

Reuses the exact 5 patient-grouped folds and the rank-1 k-NN setting
(k=9, uniform, inverse_sqrt) from evaluate_care_acuity_diagnosis_5fold.py.
No k/vote/adjustment search here; only the input template changes.
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

OUT = exp.RESULTS_DIR / "p2_p3_fields_5fold"
OUT.mkdir(parents=True, exist_ok=True)
OUT.chmod(0o700)
SEED = 42
K = 9
WEIGHTING = "uniform"
ADJUSTMENT = "inverse_sqrt"

BASE = focused.BASE_FIELDS + focused.GROUPS["Care location"] + focused.GROUPS["Acuity"]
TEMPLATES = {
    "best_baseline": BASE,
    "best_plus_classification_type": BASE + ["Classification Type"],
    "best_plus_chief_complaint": BASE + ["Chief Complaint"],
    "best_plus_cfs": BASE + ["CFS (ED)"],
    "best_plus_all_three": BASE + ["Classification Type", "Chief Complaint", "CFS (ED)"],
}
# best_baseline is field-for-field identical to the already-cached
# base_plus_care_location_acuity template; reuse that cache instead of re-encoding.
CACHE_ALIAS = {"best_baseline": "base_plus_care_location_acuity"}


def get_vectors(model, revision, template: str, fields: list[str], hist: pd.DataFrame) -> np.ndarray:
    cache_name = CACHE_ALIAS.get(template, template)
    ids = hist["Case ID"].astype(str).tolist()
    texts = exp.render_cases(hist, fields)
    return exp.get_embeddings(model, texts, cache_name, ids, revision, argparse.Namespace(
        device="cpu", batch_size=32, local_files_only=True,
    ))


def main() -> None:
    hist = pd.read_excel(exp.DATA_DIR / "historical_cases.xlsx")
    hist = hist[hist["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    y = hist["Priority Label"].to_numpy()

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
    for template, fields in TEMPLATES.items():
        vectors = get_vectors(model, revision, template, fields, hist)
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            similarity = vectors[val_idx] @ vectors[train_idx].T
            neighbor_order = np.argsort(-similarity, axis=1)[:, :K]
            neighbor_sims = np.take_along_axis(similarity, neighbor_order, axis=1)
            y_train, y_val = y[train_idx], y[val_idx]
            neighbor_labels = y_train[neighbor_order]
            pred, *_ = exp.knn_from_sorted_neighbors(
                neighbor_labels, neighbor_sims, y_train, K, WEIGHTING, ADJUSTMENT
            )
            scores = label_metrics(y_val, pred)
            fold_rows.append({"template": template, "fold": fold, **scores})
        print(f"Completed five folds: {template}", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in fold_df.columns if c not in ("template", "fold")]
    mean_df = fold_df.groupby("template", as_index=False)[metric_cols].mean()
    sd_df = fold_df.groupby("template", as_index=False)[metric_cols].std().rename(
        columns={c: f"sd_{c}" for c in metric_cols}
    )
    mean_df = mean_df.merge(sd_df, on="template")
    template_order = pd.Categorical(mean_df["template"], categories=list(TEMPLATES), ordered=True)
    mean_df = mean_df.assign(_order=template_order).sort_values("_order").drop(columns="_order")
    lead_cols = ["template", "macro_f1", "P2_recall", "P3_recall", "P2_f1", "P3_f1",
                 "balanced_accuracy", "P1_recall", "accuracy"]
    other_cols = [c for c in mean_df.columns if c not in lead_cols]
    mean_df = mean_df[[c for c in lead_cols if c in mean_df.columns] + other_cols]

    workbook = OUT / "results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        fold_df.to_excel(writer, sheet_name="Per-fold results", index=False)
        mean_df.to_excel(writer, sheet_name="Mean SD by template", index=False)
    workbook.chmod(0o600)
    print(f"Workbook written: {workbook}", flush=True)

    baseline = mean_df.set_index("template").loc["best_baseline"]
    deltas = {}
    for template in TEMPLATES:
        if template == "best_baseline":
            continue
        row = mean_df.set_index("template").loc[template]
        deltas[template] = {
            "macro_f1_delta": float(row["macro_f1"] - baseline["macro_f1"]),
            "P2_recall_delta": float(row["P2_recall"] - baseline["P2_recall"]),
            "P3_recall_delta": float(row["P3_recall"] - baseline["P3_recall"]),
        }

    meta = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed-k/vote/adjustment check of whether P2-vs-P3 differentiating fields "
                   "(Classification Type, Chief Complaint, CFS (ED)) improve on the rank-1 "
                   "5-fold template (base_plus_care_location_acuity).",
        "model_id": exp.MODEL_ID, "model_revision": revision,
        "model_snapshot_commit": exp.model_snapshot_commit(),
        "task_prefix": exp.TASK_PREFIX, "dimension": 768,
        "seed": SEED, "k": K, "weighting": WEIGHTING, "class_adjustment": ADJUSTMENT,
        "fixed_config_source": "results/care_acuity_diagnosis_5fold/run_metadata.json top_five[0]",
        "templates": TEMPLATES,
        "historical_labeled_n": len(hist),
        "mean_cv_by_template": mean_df.to_dict("records"),
        "deltas_vs_best_baseline": deltas,
        "artifact_workbook": workbook.name,
    }
    meta_path = OUT / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    meta_path.chmod(0o600)
    print(f"Metadata written: {meta_path}", flush=True)

    print("\nSummary (mean five-fold):")
    print(mean_df[["template", "macro_f1", "P2_recall", "P3_recall", "balanced_accuracy", "P1_recall"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
