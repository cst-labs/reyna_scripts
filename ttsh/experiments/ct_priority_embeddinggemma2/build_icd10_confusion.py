"""Recompute test-set confusion matrices for the 4 new ICD10 templates at
their already-selected best config (from results/p2_p3_icd10_5fold/run_metadata.json).

Needed because evaluate_p2_p3_icd10_5fold.py's get_vectors() used the same
cache filename for historical and test embedding calls, so encoding the test
set overwrote the on-disk historical embedding cache for these 4 templates.
This re-embeds historical data under distinct cache names (no effect on the
already-reported numbers, which were computed in-memory before the overwrite).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import run_experiment as exp
from evaluate_p2_p3_icd10_5fold import TEMPLATES, add_icd10_category, get_vectors

OUT = exp.RESULTS_DIR / "p2_p3_icd10_5fold"


def main() -> None:
    meta = json.loads((OUT / "run_metadata.json").read_text())
    selected = {r["template"]: r for r in meta["selected_cv_config_by_template"]
                if r["template"] in TEMPLATES}

    hist = pd.read_excel(exp.DATA_DIR / "historical_cases.xlsx")
    hist = hist[hist["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    hist = add_icd10_category(hist)
    y = hist["Priority Label"].to_numpy()

    test_raw = pd.read_excel(exp.DATA_DIR / "newdata_test_cases.xlsx")
    test = test_raw[test_raw["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    test = add_icd10_category(test)
    y_test = test["Priority Label"].to_numpy()

    class Args:
        device = "cpu"
        batch_size = 32
        local_files_only = True
    model = exp.embedder(Args())
    revision = exp.model_revision(model)

    cms: list[dict] = []
    for template, fields in TEMPLATES.items():
        row = selected[template]
        k, weighting, adjustment = int(row["k"]), row["weighting"], row["class_adjustment"]

        hist_vec = get_vectors(model, revision, template, fields, hist, cache_name=f"{template}__hist_fix")
        test_vec = get_vectors(model, revision, template, fields, test, cache_name=f"{template}__test_fix")

        similarity = test_vec @ hist_vec.T
        neighbor_order = np.argsort(-similarity, axis=1)[:, :k]
        neighbor_sims = np.take_along_axis(similarity, neighbor_order, axis=1)
        neighbor_labels = y[neighbor_order]
        pred, *_ = exp.knn_from_sorted_neighbors(neighbor_labels, neighbor_sims, y, k, weighting, adjustment)

        cm = confusion_matrix(y_test, pred, labels=exp.LABELS)
        for i, actual in enumerate(exp.LABELS):
            for j, predicted in enumerate(exp.LABELS):
                cms.append({"template": template, "actual": actual, "predicted": predicted,
                            "count": int(cm[i, j])})
        print(f"Confusion matrix done: {template}", flush=True)

    cm_df = pd.DataFrame(cms)
    out_path = OUT / "test_confusion_matrices.xlsx"
    cm_df.to_excel(out_path, index=False)
    out_path.chmod(0o600)
    print(f"Written: {out_path}", flush=True)
    print(cm_df.to_string(index=False))


if __name__ == "__main__":
    main()
