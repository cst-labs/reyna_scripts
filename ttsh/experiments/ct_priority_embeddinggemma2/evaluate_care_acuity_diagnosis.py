"""Initial-fold EmbeddingGemma search for care-location/acuity/diagnosis combos.

The three candidate groups are switched on in all 2^3 combinations, both
with a fixed order-information text base and, for non-empty group subsets,
without that base. The first seed-42, patient-grouped historical fold selects
the input combination and k-NN settings; the selected configuration is then
scored on the existing retrospective new-data file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import run_experiment as experiment


BASE_FIELDS = [
    "Order Description",
    "Working Diagnosis / Reason",
    "Relevant Clinical Information",
]
GROUPS = {
    "Care location": ["First ED Care Area", "Primary Care Area"],
    "Acuity": ["Triage Acuity", "Consult Acuity", "KIV Uptriage"],
    "Diagnosis": ["Primary ED Dx", "ED Dx"],
}


def candidate_templates() -> dict[str, list[str]]:
    names = {
        (): "base_order_info",
        ("Care location",): "base_plus_care_location",
        ("Acuity",): "base_plus_acuity",
        ("Diagnosis",): "base_plus_diagnosis",
        ("Care location", "Acuity"): "base_plus_care_location_acuity",
        ("Care location", "Diagnosis"): "base_plus_care_location_diagnosis",
        ("Acuity", "Diagnosis"): "base_plus_acuity_diagnosis",
        ("Care location", "Acuity", "Diagnosis"): "base_plus_care_location_acuity_diagnosis",
    }
    templates = {}
    for selected_groups, name in names.items():
        fields = list(BASE_FIELDS)
        for group_name in selected_groups:
            fields.extend(GROUPS[group_name])
        templates[name] = fields

    # Matched ablations: for each non-empty combination, also omit the fixed
    # order-information base entirely. There is no meaningful text-only
    # embedding when neither the base nor any candidate group is present.
    no_base_names = {
        ("Care location",): "care_location_only",
        ("Acuity",): "acuity_only",
        ("Diagnosis",): "diagnosis_only",
        ("Care location", "Acuity"): "care_location_acuity_only",
        ("Care location", "Diagnosis"): "care_location_diagnosis_only",
        ("Acuity", "Diagnosis"): "acuity_diagnosis_only",
        ("Care location", "Acuity", "Diagnosis"): "care_location_acuity_diagnosis_only",
    }
    for selected_groups, name in no_base_names.items():
        fields = []
        for group_name in selected_groups:
            fields.extend(GROUPS[group_name])
        templates[name] = fields
    return templates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    # Reuse the validated embedding, cosine k-NN, abstention, and artifact code,
    # with a dedicated candidate set and output files so prior results persist.
    experiment.TEMPLATES = candidate_templates()
    data_dir = experiment.DATA_DIR
    results_dir = experiment.RESULTS_DIR / "care_acuity_diagnosis_single_split"
    results_dir.mkdir(parents=True, exist_ok=True)
    historical = pd.read_excel(data_dir / "historical_cases.xlsx")
    test = pd.read_excel(data_dir / "newdata_test_cases.xlsx")
    labeled_test = test[test["Priority Label"].isin(experiment.LABELS)].reset_index(drop=True)
    hist = historical[historical["Priority Label"].isin(experiment.LABELS)].reset_index(drop=True)
    y = hist["Priority Label"].to_numpy()
    y_test = labeled_test["Priority Label"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=experiment.RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(hist, y, groups=hist["Patient Group ID"]))
    train_groups = set(hist.iloc[train_idx]["Patient Group ID"])
    val_groups = set(hist.iloc[val_idx]["Patient Group ID"])
    test_groups = set(labeled_test["Patient Group ID"])
    if train_groups & val_groups:
        raise RuntimeError("Patient leakage across historical train and validation")
    if (train_groups | val_groups) & test_groups:
        raise RuntimeError("Patient overlap between historical and new-data cohorts")

    split_path = results_dir / "care_acuity_diagnosis_splits.json"
    experiment.json_write(split_path, {
        "seed": experiment.RANDOM_SEED,
        "split": "first fold of 5-fold StratifiedGroupKFold; grouped by Patient Group ID",
        "train_case_ids": hist.iloc[train_idx]["Case ID"].tolist(),
        "validation_case_ids": hist.iloc[val_idx]["Case ID"].tolist(),
        "test_case_ids": test["Case ID"].tolist(),
        "train_patient_group_ids": sorted(train_groups),
        "validation_patient_group_ids": sorted(val_groups),
        "test_patient_group_ids": sorted(set(test["Patient Group ID"])),
    })

    y_train, y_val = y[train_idx], y[val_idx]
    majority = pd.Series(y_train).value_counts().idxmax()
    metric_rows, confusion_rows = experiment.metrics_for(
        "majority_class", y_val, np.repeat(majority, len(y_val)), "validation"
    )
    test_majority = pd.Series(y).value_counts().idxmax()
    rows, cms = experiment.metrics_for(
        "majority_class", y_test, np.repeat(test_majority, len(y_test)), "newdata_test"
    )
    metric_rows.extend(rows)
    confusion_rows.extend(cms)

    model = experiment.embedder(args)
    revision = experiment.model_revision(model)
    selected, configs, abst_val, abst_test, diagnostics, review_rows, neighbor_rows, _ = (
        experiment.evaluate_embeddings(
            model, revision, hist, test, train_idx, val_idx, y, y_test, args,
            metric_rows, confusion_rows,
        )
    )

    template_hash = hashlib.sha256(
        json.dumps(experiment.TEMPLATES, sort_keys=True).encode()
    ).hexdigest()
    metadata = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Compare all combinations of care location, acuity, and diagnosis groups with and without the fixed order-information text on the initial patient-grouped validation fold, then score the selected setup on previously examined new data.",
        "model_id": experiment.MODEL_ID,
        "model_revision": revision,
        "model_snapshot_commit": experiment.model_snapshot_commit(),
        "task_prefix": experiment.TASK_PREFIX,
        "dimension": 768,
        "normalized_embeddings": True,
        "cosine": "dot product of L2-normalized vectors",
        "device": args.device,
        "batch_size": args.batch_size,
        "precision": "float32",
        "seed": experiment.RANDOM_SEED,
        "base_fields_in_matched_templates": BASE_FIELDS,
        "matched_ablation": "For each non-empty group subset, compare a template with BASE_FIELDS against the same group fields without BASE_FIELDS. The empty-group, no-base case is omitted because it contains no meaningful case text.",
        "candidate_groups": GROUPS,
        "templates": experiment.TEMPLATES,
        "template_count": len(experiment.TEMPLATES),
        "settings_per_template": len(experiment.K_VALUES) * 2 * len(experiment.CLASS_ADJUSTMENTS),
        "total_knn_configurations": len(configs),
        "selection_rule": "highest first-fold macro F1; ties by P1 recall, balanced accuracy, smaller k, uniform vote, then less class adjustment",
        "selected_configuration": selected,
        "test_status": "retrospective; new-data results were examined in prior analyses, not blind",
        "template_hash": template_hash,
        "historical_labeled_cases": len(hist),
        "newdata_labeled_cases": len(labeled_test),
        "split_manifest": split_path.name,
    }
    experiment.json_write(results_dir / "care_acuity_diagnosis_metadata.json", metadata)

    out = results_dir / "care_acuity_diagnosis_combinations.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        configs.to_excel(writer, sheet_name="Validation configurations", index=False)
        pd.DataFrame(metric_rows).to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame(confusion_rows).to_excel(writer, sheet_name="Confusion matrices", index=False)
        abst_val.to_excel(writer, sheet_name="Abstention validation", index=False)
        abst_test.to_excel(writer, sheet_name="Abstention new data", index=False)
        diagnostics.to_excel(writer, sheet_name="Embedding diagnostics", index=False)
        pd.DataFrame(review_rows).to_excel(writer, sheet_name="New data suggestions", index=False)
        pd.DataFrame(neighbor_rows).to_excel(writer, sheet_name="Supporting neighbors", index=False)
    out.chmod(0o600)
    print(json.dumps(selected, indent=2, default=str))
    print(f"Scored {len(configs)} k-NN settings across {len(experiment.TEMPLATES)} templates.")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
