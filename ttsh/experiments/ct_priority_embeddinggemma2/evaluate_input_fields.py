"""Patient-grouped field ablations; intentionally never opens newdata_test_cases.xlsx."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_experiment import (
    CLASS_ADJUSTMENTS,
    K_VALUES,
    LABELS,
    MODEL_ID,
    TASK_PREFIX,
    TEMPLATES,
    knn_from_sorted_neighbors,
    model_snapshot_commit,
    render_cases,
    structured_frame,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EMBEDDING_DIR = ROOT / "artifacts" / "embeddings"
RESULTS_DIR = ROOT / "results" / "field_selection"
SEED = 2026

CORE_FIELDS = [
    "Order Description", "Working Diagnosis / Reason",
    "Relevant Clinical Information", "Primary ED Dx", "ED Dx",
]
FIELD_GROUPS = {
    "Age at Visit": ["Age at Visit Numeric"],
    "Sex": ["Sex"],
    "Arrival Method": ["Arrival Method"],
    "Chief Complaint": ["Chief Complaint"],
    "KIV Uptriage": ["KIV Uptriage"],
    "Triage Acuity": ["Triage Acuity"],
    "Consult Acuity": ["Consult Acuity"],
    "First ED Care Area": ["First ED Care Area"],
    "Primary Care Area": ["Primary Care Area"],
    "CFS (ED)": ["CFS (ED)"],
    "BP first": ["BP first systolic", "BP first diastolic"],
    "BP before clinician": [
        "BP before clinician systolic", "BP before clinician diastolic",
    ],
    "Reatt Same ED": ["Reatt Same ED"],
    "Reatt Reason": ["Reatt Reason"],
    "72h Reatt NGEMR ED": ["72h Reatt NGEMR ED"],
    "72h Reatt then Admitted": ["72h Reatt NGEMR ED then Admitted"],
    "HD/ICU within 24H": ["HD/ICU from GW within 24H"],
    "three timing-confirmed flags": [
        "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted",
        "HD/ICU from GW within 24H",
    ],
}
NUMERIC_FIELDS = {
    "Age at Visit Numeric", "BP first systolic", "BP first diastolic",
    "BP before clinician systolic", "BP before clinician diastolic",
}


def build_structured_model(columns: list[str]) -> Pipeline:
    numeric = [column for column in columns if column in NUMERIC_FIELDS]
    categorical = [column for column in columns if column not in NUMERIC_FIELDS]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(
                handle_unknown="infrequent_if_exist", min_frequency=5,
            )),
        ]), categorical))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(
            max_iter=2000, C=0.03, class_weight="balanced", solver="lbfgs",
        )),
    ])


def score_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0,
        ),
        "P1_precision": precision[0], "P1_recall": recall[0],
        "P2_precision": precision[1], "P2_recall": recall[1],
        "P3_precision": precision[2], "P3_recall": recall[2],
    }


def structured_ablation(frame: pd.DataFrame, y: np.ndarray,
                        groups: np.ndarray, folds: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_fields = list(dict.fromkeys(CORE_FIELDS + sum(FIELD_GROUPS.values(), [])))
    variants = {"core only": CORE_FIELDS}
    for name, fields in FIELD_GROUPS.items():
        variants[f"core + {name}"] = CORE_FIELDS + fields
    variants["all fields"] = full_fields
    for name, fields in FIELD_GROUPS.items():
        variants[f"all minus {name}"] = [field for field in full_fields if field not in fields]

    rows = []
    for variant, columns in variants.items():
        for fold_number, (train_idx, validation_idx) in enumerate(folds, 1):
            if set(groups[train_idx]) & set(groups[validation_idx]):
                raise RuntimeError("Patient group crossed a fold boundary")
            model = build_structured_model(columns)
            model.fit(frame.iloc[train_idx][columns], y[train_idx])
            prediction = model.predict(frame.iloc[validation_idx][columns])
            rows.append({
                "variant": variant, "fold": fold_number,
                "n_validation": len(validation_idx),
                **score_row(y[validation_idx], prediction),
            })
    raw = pd.DataFrame(rows)
    metrics = [column for column in raw.columns
               if column not in {"variant", "fold", "n_validation"}]
    summary = raw.groupby("variant")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return raw, summary.reset_index().sort_values("macro_f1_mean", ascending=False)


def embedding_ablation(history: pd.DataFrame, y: np.ndarray,
                       groups: np.ndarray, folds: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for template, fields in TEMPLATES.items():
        metadata_path = EMBEDDING_DIR / f"{template}.json"
        vector_path = EMBEDDING_DIR / f"{template}.npz"
        if not (metadata_path.exists() and vector_path.exists()):
            continue
        texts = render_cases(history, fields)
        expected_text_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()
        expected_ids_hash = hashlib.sha256(
            "\n".join(history["Case ID"].astype(str)).encode()
        ).hexdigest()
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("text_hash") != expected_text_hash
                or metadata.get("ids_hash") != expected_ids_hash):
            # Some older cache files also contain the new-data queries; do not
            # use a vector matrix unless its historical rows are verified.
            continue
        vectors = np.load(vector_path)["embeddings"]
        if vectors.shape != (len(history), 768):
            continue
        for fold_number, (train_idx, validation_idx) in enumerate(folds, 1):
            if set(groups[train_idx]) & set(groups[validation_idx]):
                raise RuntimeError("Patient group crossed a fold boundary")
            similarities = vectors[validation_idx] @ vectors[train_idx].T
            top_indices = np.argpartition(-similarities, 24, axis=1)[:, :25]
            top_similarities = np.take_along_axis(similarities, top_indices, axis=1)
            order = np.argsort(-top_similarities, axis=1)
            top_indices = np.take_along_axis(top_indices, order, axis=1)
            top_similarities = np.take_along_axis(top_similarities, order, axis=1)
            neighbor_labels = y[train_idx][top_indices]
            for k in K_VALUES:
                for weighting in ("uniform", "similarity_weighted"):
                    for adjustment in CLASS_ADJUSTMENTS:
                        prediction, *_ = knn_from_sorted_neighbors(
                            neighbor_labels[:, :k], top_similarities[:, :k],
                            y[train_idx], k, weighting, adjustment,
                        )
                        fold_rows.append({
                            "template": template, "fold": fold_number, "k": k,
                            "weighting": weighting,
                            "class_adjustment": adjustment,
                            "n_validation": len(validation_idx),
                            **score_row(y[validation_idx], prediction),
                        })
    raw = pd.DataFrame(fold_rows)
    if raw.empty:
        return raw, raw
    metric_columns = ["accuracy", "balanced_accuracy", "macro_f1",
                      "P1_recall", "P2_recall", "P3_recall"]
    means = raw.groupby(
        ["template", "k", "weighting", "class_adjustment"], as_index=False,
    )[metric_columns].mean()
    best = means.sort_values(
        ["macro_f1", "P1_recall", "balanced_accuracy"], ascending=False,
    ).groupby("template", as_index=False).head(1)
    return raw, best.sort_values(["macro_f1", "P1_recall"], ascending=False)


def main() -> None:
    # Deliberately load only the historical cases file. No held-out file or
    # held-out labels are read by this field-selection utility.
    history = pd.read_excel(DATA_DIR / "historical_cases.xlsx")
    history = history[history["Priority Label"].isin(LABELS)].reset_index(drop=True)
    y = history["Priority Label"].to_numpy()
    groups = history["Patient Group ID"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(history, y, groups))
    split_manifest = {
        "seed": SEED,
        "splitter": "StratifiedGroupKFold(n_splits=5, shuffle=True)",
        "data": "historical labeled cases only; new-data file not read",
        "folds": [],
    }
    for fold_number, (train_idx, validation_idx) in enumerate(folds, 1):
        train_groups = set(groups[train_idx])
        validation_groups = set(groups[validation_idx])
        if train_groups & validation_groups:
            raise RuntimeError("Patient group crossed a field-selection fold boundary")
        split_manifest["folds"].append({
            "fold": fold_number,
            "train_case_ids": history.iloc[train_idx]["Case ID"].astype(str).tolist(),
            "validation_case_ids": history.iloc[validation_idx]["Case ID"].astype(str).tolist(),
            "train_patient_group_ids": sorted(train_groups),
            "validation_patient_group_ids": sorted(validation_groups),
        })
    frame = structured_frame(history)

    structured_folds, structured_summary = structured_ablation(
        frame, y, groups, folds,
    )
    embedding_folds, embedding_best = embedding_ablation(
        history, y, groups, folds,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "field_selection_validation.xlsx"
    split_path = RESULTS_DIR / "field_selection_splits.json"
    split_path.write_text(json.dumps(split_manifest, indent=2) + "\n")
    split_path.chmod(0o600)
    embedding_metadata = {}
    for template in embedding_best.get("template", pd.Series(dtype=str)).tolist():
        metadata_path = EMBEDDING_DIR / f"{template}.json"
        if metadata_path.exists():
            embedding_metadata[template] = json.loads(metadata_path.read_text())
    metadata = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": "data/historical_cases.xlsx",
        "data_sha256": hashlib.sha256((DATA_DIR / "historical_cases.xlsx").read_bytes()).hexdigest(),
        "labeled_rows": len(history),
        "split_manifest": split_path.name,
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "splitter": split_manifest["splitter"],
        "seed": SEED,
        "fold_count": len(folds),
        "patient_grouped": True,
        "new_data_file_read": False,
        "structured_model": "one-hot categorical fields + numeric age/BP; multinomial LogisticRegression C=0.03, class_weight=balanced",
        "embedding_model_id": MODEL_ID,
        "embedding_model_snapshot_commit": model_snapshot_commit(),
        "embedding_task_prefix": TASK_PREFIX,
        "embedding_dimension": 768,
        "embedding_normalized": True,
        "cosine": "dot product of L2-normalized vectors",
        "embedding_precision": "float32",
        "embedding_device": "cpu",
        "k_values": K_VALUES,
        "voting": ["uniform", "similarity_weighted"],
        "class_adjustments": CLASS_ADJUSTMENTS,
        "embedding_selection": "For each template, best mean five-fold macro F1; ties broken by P1 recall then balanced accuracy. These are selection scores, not independent performance estimates.",
        "embedding_templates_scored": embedding_best.get("template", pd.Series(dtype=str)).tolist(),
        "embedding_cache_metadata": embedding_metadata,
        "user_confirmed_pre_order_fields": [
            "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted",
            "HD/ICU from GW within 24H",
        ],
        "timing_provenance_note": "User confirmed these fields are available before CT order placement; verify source timestamps before prospective use.",
    }
    metadata_path = RESULTS_DIR / "field_selection_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    metadata_path.chmod(0o600)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        structured_summary.to_excel(writer, sheet_name="Structured summary", index=False)
        structured_folds.to_excel(writer, sheet_name="Structured fold scores", index=False)
        embedding_folds.to_excel(writer, sheet_name="Embedding fold scores", index=False)
        embedding_best.to_excel(writer, sheet_name="Embedding best per template", index=False)
        pd.DataFrame({
            "setting": ["data", "split", "seed", "structured model", "EmbeddingGemma", "embedding dimensions/prefix", "k/voting/class adjustments", "blind test data used"],
            "value": [
                "historical labeled cases only",
                "5-fold StratifiedGroupKFold grouped by patient",
                SEED,
                "one-hot fields + numeric age/BP, C=0.03, class_weight=balanced",
                f"{MODEL_ID} @ {metadata['embedding_model_snapshot_commit']}",
                f"768; normalized; {TASK_PREFIX}",
                f"k={K_VALUES}; uniform/similarity-weighted; {CLASS_ADJUSTMENTS}",
                "No",
            ],
        }).to_excel(writer, sheet_name="Run settings", index=False)
    output.chmod(0o600)
    print(f"Wrote {output}")
    print(f"Wrote {metadata_path} and {split_path}")
    print("Embedding templates skipped if their cache identity did not match the historical cases.")


if __name__ == "__main__":
    main()
