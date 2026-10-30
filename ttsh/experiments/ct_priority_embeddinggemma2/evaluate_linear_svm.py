"""Nested patient-grouped LinearSVC comparison on historical cases only."""

from __future__ import annotations

import hashlib
import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from evaluate_input_fields import CORE_FIELDS, FIELD_GROUPS, NUMERIC_FIELDS, score_row
from run_experiment import LABELS


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "linear_svm"
SEED = 2026
C_VALUES = [0.01, 0.1, 1.0, 10.0]
CLASS_WEIGHTS = [None, "balanced"]
DEFAULT_TOLERANCE = 1e-4
FULL_FIELDS_TOLERANCE = 1e-3
TEXT_FIELDS = {
    "Order Description", "Working Diagnosis / Reason",
    "Relevant Clinical Information", "Primary ED Dx", "ED Dx",
    "Chief Complaint", "Reatt Reason",
}
TIMING_FIELDS = [
    "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted",
    "HD/ICU from GW within 24H",
]


def dedupe(fields: list[str]) -> list[str]:
    return list(dict.fromkeys(fields))


def field_variants() -> dict[str, list[str]]:
    base = dedupe(CORE_FIELDS + ["Age at Visit Numeric", "Triage Acuity", "Sex"])
    all_fields = dedupe(CORE_FIELDS + sum(FIELD_GROUPS.values(), []))
    without_timing = [field for field in all_fields if field not in TIMING_FIELDS]
    return {
        "core only": list(CORE_FIELDS),
        "matched base": base,
        "matched base + 72h reattendance": base + [TIMING_FIELDS[0]],
        "matched base + 72h reattendance then admitted": base + [TIMING_FIELDS[1]],
        "matched base + HD/ICU within 24H": base + [TIMING_FIELDS[2]],
        "matched base + all three timing fields": base + TIMING_FIELDS,
        "all fields without three timing fields": without_timing,
        "all fields including three timing fields": all_fields,
    }


def tolerance_for_variant(variant: str) -> float:
    if variant in {
        "all fields without three timing fields",
        "all fields including three timing fields",
    }:
        return FULL_FIELDS_TOLERANCE
    return DEFAULT_TOLERANCE


def build_matrices(train: pd.DataFrame, validation: pd.DataFrame,
                   columns: list[str]):
    """Fit every text/category transform on train only; apply it to validation."""
    text_columns = [column for column in columns if column in TEXT_FIELDS]
    numeric_columns = [column for column in columns if column in NUMERIC_FIELDS]
    categorical_columns = [
        column for column in columns
        if column not in TEXT_FIELDS and column not in NUMERIC_FIELDS
    ]
    matrices_train = []
    matrices_validation = []

    if text_columns:
        def join_text(frame: pd.DataFrame) -> list[str]:
            texts = []
            for _, row in frame[text_columns].iterrows():
                texts.append("\n".join(
                    f"{column}: {('Unknown' if pd.isna(row[column]) or not str(row[column]).strip() else str(row[column]).strip())}"
                    for column in text_columns
                ))
            return texts

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=50000,
            sublinear_tf=True, strip_accents="unicode",
        )
        matrices_train.append(vectorizer.fit_transform(join_text(train)))
        matrices_validation.append(vectorizer.transform(join_text(validation)))

    if categorical_columns:
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=5,
        )
        train_categories = train[categorical_columns].astype("string").fillna("Unknown").astype(object)
        validation_categories = validation[categorical_columns].astype("string").fillna("Unknown").astype(object)
        matrices_train.append(encoder.fit_transform(train_categories))
        matrices_validation.append(encoder.transform(validation_categories))

    if numeric_columns:
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        scaler = StandardScaler(with_mean=False)
        train_numeric = imputer.fit_transform(train[numeric_columns])
        validation_numeric = imputer.transform(validation[numeric_columns])
        matrices_train.append(csr_matrix(scaler.fit_transform(train_numeric)))
        matrices_validation.append(csr_matrix(scaler.transform(validation_numeric)))

    return hstack(matrices_train, format="csr"), hstack(matrices_validation, format="csr")


def tune_on_inner_folds(outer_train: pd.DataFrame, y_outer: np.ndarray,
                        groups_outer: np.ndarray, columns: list[str],
                        outer_fold: int, tolerance: float):
    inner = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=SEED + outer_fold,
    )
    inner_splits = list(inner.split(outer_train, y_outer, groups_outer))
    scores: dict[tuple[float, str], list[tuple[float, float]]] = {
        (c, "None" if weight is None else weight): []
        for c in C_VALUES for weight in CLASS_WEIGHTS
    }
    fit_warnings: dict[str, int] = {f"C={c};class_weight={w}": 0
                                    for c in C_VALUES for w in CLASS_WEIGHTS}
    for inner_train_idx, inner_validation_idx in inner_splits:
        inner_train = outer_train.iloc[inner_train_idx]
        inner_validation = outer_train.iloc[inner_validation_idx]
        if set(groups_outer[inner_train_idx]) & set(groups_outer[inner_validation_idx]):
            raise RuntimeError("Patient group crossed an inner fold boundary")
        x_train, x_validation = build_matrices(
            inner_train, inner_validation, columns,
        )
        y_train, y_validation = y_outer[inner_train_idx], y_outer[inner_validation_idx]
        for c_value in C_VALUES:
            for class_weight in CLASS_WEIGHTS:
                weight_key = "None" if class_weight is None else class_weight
                key = (c_value, weight_key)
                model = LinearSVC(
                    C=c_value, class_weight=class_weight, dual="auto",
                    max_iter=10000, tol=tolerance, random_state=SEED,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(x_train, y_train)
                fit_warnings[f"C={c_value};class_weight={weight_key}"] += sum(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                )
                prediction = model.predict(x_validation)
                scores[key].append((
                    f1_score(y_validation, prediction, labels=LABELS,
                             average="macro", zero_division=0),
                    balanced_accuracy_score(y_validation, prediction),
                ))

    tuning_rows = []
    for (c_value, weight_key), fold_scores in scores.items():
        tuning_rows.append({
            "C": c_value,
            "class_weight": weight_key,
            "inner_macro_f1_mean": float(np.mean([item[0] for item in fold_scores])),
            "inner_balanced_accuracy_mean": float(np.mean([item[1] for item in fold_scores])),
            "convergence_warnings": fit_warnings[f"C={c_value};class_weight={weight_key}"],
        })
    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["inner_macro_f1_mean", "inner_balanced_accuracy_mean", "C"],
        ascending=[False, False, True], kind="stable",
    )
    return tuning.iloc[0].to_dict(), tuning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-fields-only", action="store_true",
        help="Rerun the all-fields comparison at the relaxed convergence tolerance.",
    )
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history = pd.read_excel(DATA_DIR / "historical_cases.xlsx")
    history = history[history["Priority Label"].isin(LABELS)].reset_index(drop=True)
    history["Age at Visit Numeric"] = pd.to_numeric(
        history["Age at Visit"].astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    for source, stem in [("BP first", "BP first"),
                         ("BP before clinician", "BP before clinician")]:
        value = (history[source].astype("string") if source in history
                 else pd.Series(pd.NA, index=history.index, dtype="string"))
        parts = value.str.extract(r"(?<!\d)(\d{2,3})\s*/\s*(\d{2,3})(?!\d)")
        history[f"{stem} systolic"] = pd.to_numeric(parts[0], errors="coerce")
        history[f"{stem} diastolic"] = pd.to_numeric(parts[1], errors="coerce")

    y = history["Priority Label"].to_numpy()
    groups = history["Patient Group ID"].to_numpy()
    outer_splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED,
    )
    outer_folds = list(outer_splitter.split(history, y, groups))
    variants = field_variants()
    if args.full_fields_only:
        variants = {
            name: columns for name, columns in variants.items()
            if name in {
                "all fields without three timing fields",
                "all fields including three timing fields",
            }
        }
    fold_results = []
    tuning_results = []
    split_manifest = {"seed": SEED, "outer_folds": []}

    for fold_number, (train_idx, validation_idx) in enumerate(outer_folds, 1):
        if set(groups[train_idx]) & set(groups[validation_idx]):
            raise RuntimeError("Patient group crossed an outer fold boundary")
        split_manifest["outer_folds"].append({
            "fold": fold_number,
            "train_case_ids": history.iloc[train_idx]["Case ID"].astype(str).tolist(),
            "validation_case_ids": history.iloc[validation_idx]["Case ID"].astype(str).tolist(),
            "train_patient_group_ids": sorted(set(groups[train_idx])),
            "validation_patient_group_ids": sorted(set(groups[validation_idx])),
        })
        outer_train = history.iloc[train_idx].reset_index(drop=True)
        outer_validation = history.iloc[validation_idx].reset_index(drop=True)
        y_outer = y[train_idx]
        groups_outer = groups[train_idx]
        for variant, columns in variants.items():
            best, tuning = tune_on_inner_folds(
                outer_train, y_outer, groups_outer, columns, fold_number,
                tolerance_for_variant(variant),
            )
            tuning.insert(0, "variant", variant)
            tuning.insert(1, "outer_fold", fold_number)
            tuning["tol"] = tolerance_for_variant(variant)
            tuning_results.append(tuning)

            x_train, x_validation = build_matrices(
                outer_train, outer_validation, columns,
            )
            selected_weight = None if best["class_weight"] == "None" else best["class_weight"]
            model = LinearSVC(
                C=float(best["C"]), class_weight=selected_weight,
                dual="auto", max_iter=10000, tol=tolerance_for_variant(variant),
                random_state=SEED,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_train, y_outer)
            prediction = model.predict(x_validation)
            fold_results.append({
                "variant": variant, "outer_fold": fold_number,
                "n_validation": len(validation_idx),
                "selected_C": float(best["C"]),
                "selected_class_weight": best["class_weight"],
                "tol": tolerance_for_variant(variant),
                "outer_convergence_warnings": sum(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                ),
                **score_row(y[validation_idx], prediction),
            })
            print(f"Finished fold {fold_number}: {variant}; C={best['C']}, class_weight={best['class_weight']}", flush=True)

    raw = pd.DataFrame(fold_results)
    metric_columns = [
        "accuracy", "balanced_accuracy", "macro_f1",
        "P1_precision", "P1_recall", "P2_precision", "P2_recall",
        "P3_precision", "P3_recall",
    ]
    summary = raw.groupby("variant")[metric_columns].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index().sort_values("macro_f1_mean", ascending=False)

    suffix = "full_fields_" if args.full_fields_only else ""
    split_path = RESULTS_DIR / f"linear_svm_{suffix}validation_splits.json"
    split_path.write_text(json.dumps(split_manifest, indent=2) + "\n")
    split_path.chmod(0o600)
    settings = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": "data/historical_cases.xlsx",
        "data_sha256": hashlib.sha256((DATA_DIR / "historical_cases.xlsx").read_bytes()).hexdigest(),
        "labeled_rows": len(history),
        "new_data_used_for_tuning": False,
        "outer_split": "5-fold StratifiedGroupKFold by patient, shuffle=True, seed=2026",
        "inner_tuning": "3-fold StratifiedGroupKFold by patient within each outer training partition; mean inner macro F1, tie-break balanced accuracy then smaller C",
        "split_file": split_path.name,
        "split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "feature_representation": "Field-labeled text with TF-IDF word unigrams/bigrams (min_df=2, max_features=50000, sublinear_tf), one-hot categoricals (min_frequency=5), median-imputed/scaled numeric age and BP with missingness indicators; transforms fit within each training fold",
        "classifier": "LinearSVC, squared hinge, dual=auto, max_iter=10000",
        "tolerance_by_variant": {
            name: tolerance_for_variant(name) for name in variants
        },
        "hyperparameter_grid": {"C": C_VALUES, "class_weight": ["None", "balanced"]},
        "variants": variants,
        "limitations": "Nested patient-grouped internal validation; outer fold scores are not a blind external evaluation.",
    }
    metadata_path = RESULTS_DIR / f"linear_svm_{suffix}validation_metadata.json"
    metadata_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    metadata_path.chmod(0o600)

    output = RESULTS_DIR / f"linear_svm_{suffix}validation.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Outer fold summary", index=False)
        raw.to_excel(writer, sheet_name="Outer fold scores", index=False)
        pd.concat(tuning_results, ignore_index=True).to_excel(
            writer, sheet_name="Inner tuning scores", index=False,
        )
        pd.DataFrame([{"setting": key, "value": json.dumps(value, ensure_ascii=False)}
                      for key, value in settings.items()]).to_excel(
            writer, sheet_name="Run settings", index=False,
        )
    output.chmod(0o600)
    print("Wrote", output)
    print("Wrote", metadata_path, "and", split_path)


if __name__ == "__main__":
    main()
