"""Retrospective test-set scoring for the 13 templates compared in
evaluate_p2_p3_combinations_5fold.py.

Same fixed k-NN setting (k=9, uniform, inverse_sqrt). For each template,
refits the retrieval reference on *all* historical labeled rows and scores
the retrospective Jul-Sep 2026 test file (newdata_test_cases.xlsx), per the
methodology in STUDY_REPORT.md / run_experiment.evaluate_embeddings. This is
a retrospective re-examination of already-collected test data, not a blind
test.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import evaluate_care_acuity_diagnosis as focused
import run_experiment as exp
from evaluate_care_acuity_diagnosis_5fold import label_metrics
from evaluate_p2_p3_combinations_5fold import OUT, TEMPLATES, REUSED_TEMPLATES, PRIOR_RUN_PATH

K = 9
WEIGHTING = "uniform"
ADJUSTMENT = "inverse_sqrt"

ALL_TEMPLATE_FIELDS = dict(TEMPLATES)
PRIOR_TEMPLATE_FIELDS = json.loads(PRIOR_RUN_PATH.read_text())["templates"]
for name in REUSED_TEMPLATES:
    ALL_TEMPLATE_FIELDS[name] = PRIOR_TEMPLATE_FIELDS[name]
CACHE_ALIAS = {"best_baseline": "base_plus_care_location_acuity"}


def get_combined_vectors(model, revision, template: str, fields: list[str],
                         hist: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Combined historical+test embeddings, cached under the historical-only cache name."""
    cache_name = CACHE_ALIAS.get(template, template)
    hist_ids = hist["Case ID"].astype(str).tolist()
    test_ids = test["Case ID"].astype(str).tolist()
    combined_ids = hist_ids + test_ids
    combined_texts = exp.render_cases(hist, fields) + exp.render_cases(test, fields)
    args = argparse.Namespace(device="cpu", batch_size=32, local_files_only=True)
    vectors = exp.get_embeddings(model, combined_texts, cache_name, combined_ids, revision, args)
    return vectors[:len(hist)], vectors[len(hist):]


def main() -> None:
    hist = pd.read_excel(exp.DATA_DIR / "historical_cases.xlsx")
    hist = hist[hist["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    y_hist = hist["Priority Label"].to_numpy()

    test_raw = pd.read_excel(exp.DATA_DIR / "newdata_test_cases.xlsx")
    test = test_raw[test_raw["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    y_test = test["Priority Label"].to_numpy()

    if set(hist["Case ID"].astype(str)) & set(test["Case ID"].astype(str)):
        raise RuntimeError("Historical and test Case IDs overlap")

    class Args:
        device = "cpu"
        batch_size = 32
        local_files_only = True
    model = exp.embedder(Args())
    revision = exp.model_revision(model)

    rows: list[dict] = []
    for template, fields in ALL_TEMPLATE_FIELDS.items():
        hist_vec, test_vec = get_combined_vectors(model, revision, template, fields, hist, test)
        similarity = test_vec @ hist_vec.T
        neighbor_order = np.argsort(-similarity, axis=1)[:, :K]
        neighbor_sims = np.take_along_axis(similarity, neighbor_order, axis=1)
        neighbor_labels = y_hist[neighbor_order]
        pred, *_ = exp.knn_from_sorted_neighbors(
            neighbor_labels, neighbor_sims, y_hist, K, WEIGHTING, ADJUSTMENT
        )
        scores = label_metrics(y_test, pred)
        rows.append({"template": template, **scores})
        print(f"Scored on retrospective test set: {template}", flush=True)

    test_df = pd.DataFrame(rows)
    template_order = ["best_baseline"] + list(TEMPLATES) + REUSED_TEMPLATES[1:]
    test_df["template"] = pd.Categorical(test_df["template"], categories=template_order, ordered=True)
    test_df = test_df.sort_values("template").reset_index(drop=True)
    lead_cols = ["template", "macro_f1", "P2_recall", "P3_recall", "P2_f1", "P3_f1",
                 "balanced_accuracy", "P1_recall", "accuracy"]
    other_cols = [c for c in test_df.columns if c not in lead_cols]
    test_df = test_df[[c for c in lead_cols if c in test_df.columns] + other_cols]

    baseline = test_df.set_index("template").loc["best_baseline"]
    deltas = {}
    for template in test_df["template"]:
        if template == "best_baseline":
            continue
        row = test_df.set_index("template").loc[template]
        deltas[template] = {
            "macro_f1_delta": float(row["macro_f1"] - baseline["macro_f1"]),
            "P2_recall_delta": float(row["P2_recall"] - baseline["P2_recall"]),
            "P3_recall_delta": float(row["P3_recall"] - baseline["P3_recall"]),
        }

    workbook = OUT / "results.xlsx"
    existing_sheets = pd.read_excel(workbook, sheet_name=None)
    existing_sheets["Test set results (retrospective)"] = test_df
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for sheet_name, df in existing_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    workbook.chmod(0o600)
    print(f"Workbook updated: {workbook}", flush=True)

    meta_path = OUT / "run_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["test_set"] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "caveat": "Retrospective re-examination of the Jul-Sep 2026 newdata_test_cases.xlsx file "
                  "already used throughout this study, not a blind/held-out test. Reference set for "
                  "k-NN is all historical labeled rows (not a CV fold).",
        "source_file": "data/newdata_test_cases.xlsx",
        "test_labeled_n": len(test),
        "k": K, "weighting": WEIGHTING, "class_adjustment": ADJUSTMENT,
        "results_by_template": test_df.to_dict("records"),
        "deltas_vs_best_baseline": deltas,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    meta_path.chmod(0o600)
    print(f"Metadata updated: {meta_path}", flush=True)

    print("\nRetrospective test-set summary:")
    print(test_df[["template", "macro_f1", "P2_recall", "P3_recall", "balanced_accuracy", "P1_recall"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
