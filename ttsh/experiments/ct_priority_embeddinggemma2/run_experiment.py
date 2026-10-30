"""Run the EmbeddingGemma 2 CT priority retrieval prototype locally.

Embeddings use the symmetric SentenceSimilarity instruction for both the
historical reference set and new orders. The script creates a patient-grouped
historical train/validation split, tunes only on validation, and evaluates the
chosen configuration on the separate new-data set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
BASELINE_DIR = RESULTS_DIR / "baseline_knn"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_ID = "google/embeddinggemma-2"
TASK_PREFIX = "task: sentence similarity | query: "
LABELS = ["P1", "P2", "P3"]
RANDOM_SEED = 42
K_VALUES = [1, 3, 5, 9, 15, 25]

TEMPLATES: dict[str, list[str]] = {
    "description": ["Order Description"],
    "plus_indication": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information"],
    "plus_diagnosis": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx"],
    "plus_age": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit"],
    "plus_triage_acuity": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity"],
    "plus_sex": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity", "Sex"],
    # Ablations append the two additional fields confirmed available before
    # scan-priority assignment, so their marginal contribution is measurable.
    "plus_kiv_consult_acuity": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity", "Sex", "KIV Uptriage", "Consult Acuity"],
    "plus_arrival_chief": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity", "Sex", "Arrival Method", "Chief Complaint"],
    "plus_care_areas_cfs": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity", "Sex", "First ED Care Area", "Primary Care Area", "CFS (ED)"],
    "plus_early_bp": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity", "Sex", "BP first", "BP before clinician"],
    "plus_reattendance": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity", "Sex", "Reatt Same ED", "Reatt Reason"],
    "plus_72h_reattendance": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity", "Sex", "72h Reatt NGEMR ED"],
    "plus_72h_reattendance_admitted": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity", "Sex", "72h Reatt NGEMR ED then Admitted"],
    "plus_hd_icu_24h": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity", "Sex", "HD/ICU from GW within 24H"],
    "plus_three_timing_fields": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity", "Sex", "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted", "HD/ICU from GW within 24H"],
    "plus_all_confirmed_context": ["Order Description", "Working Diagnosis / Reason", "Relevant Clinical Information", "Primary ED Dx", "ED Dx", "Age at Visit", "Triage Acuity", "Sex", "KIV Uptriage", "Consult Acuity", "Arrival Method", "Chief Complaint", "First ED Care Area", "Primary Care Area", "CFS (ED)", "BP first", "BP before clinician", "Reatt Same ED", "Reatt Reason", "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted", "HD/ICU from GW within 24H"],
}
# The user confirmed these three fields are populated from information
# available before the CT order. Treat them as eligible inputs; the claim is
# an explicit provenance assumption, not inferred from the field names.
USER_CONFIRMED_PRE_ORDER_FIELDS = [
    "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted",
    "HD/ICU from GW within 24H",
]

CLASS_ADJUSTMENTS = {"none": 0.0, "inverse_sqrt": 0.5, "inverse": 1.0}


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    path.chmod(0o600)


def stable_text(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value or "Unknown"


def render_cases(df: pd.DataFrame, field_names: list[str]) -> list[str]:
    return ["\n".join(f"{field}: {stable_text(row.get(field))}" for field in field_names) for _, row in df.iterrows()]


def metrics_for(name: str, y_true: np.ndarray, y_pred: np.ndarray, split: str) -> tuple[list[dict], list[dict]]:
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    rows = [
        {"split": split, "model": name, "metric": "accuracy", "value": accuracy_score(y_true, y_pred), "support": len(y_true)},
        {"split": split, "model": name, "metric": "balanced_accuracy", "value": balanced_accuracy_score(y_true, y_pred), "support": len(y_true)},
        {"split": split, "model": name, "metric": "macro_f1", "value": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), "support": len(y_true)},
    ]
    rows.extend([
        {"split": split, "model": name, "metric": f"{label}_precision", "value": p[i], "support": int(support[i])}
        for i, label in enumerate(LABELS)
    ])
    rows.extend([
        {"split": split, "model": name, "metric": f"{label}_recall", "value": r[i], "support": int(support[i])}
        for i, label in enumerate(LABELS)
    ])
    rows.extend([
        {"split": split, "model": name, "metric": f"{label}_f1", "value": f[i], "support": int(support[i])}
        for i, label in enumerate(LABELS)
    ])
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_rows = [
        {"split": split, "model": name, "actual": LABELS[i], "predicted": LABELS[j], "count": int(matrix[i, j])}
        for i in range(len(LABELS)) for j in range(len(LABELS))
    ]
    return rows, cm_rows


def knn_from_sorted_neighbors(labels_sorted: np.ndarray, sims_sorted: np.ndarray,
                              y_ref: np.ndarray, k: int, weighting: str,
                              class_adjustment: str = "none") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = labels_sorted[:, :k]
    sims = sims_sorted[:, :k]
    scores = np.zeros((len(labels), len(LABELS)), dtype=np.float64)
    for i, label in enumerate(LABELS):
        mask = labels == label
        if weighting == "uniform":
            scores[:, i] = mask.sum(axis=1)
        else:
            scores[:, i] = (np.maximum(sims, 0.0) * mask).sum(axis=1)
    all_zero = scores.sum(axis=1) == 0
    if all_zero.any():
        for i, label in enumerate(LABELS):
            scores[all_zero, i] = (labels[all_zero] == label).sum(axis=1)
    alpha = CLASS_ADJUSTMENTS[class_adjustment]
    if alpha:
        counts = np.asarray([(y_ref == label).sum() for label in LABELS], dtype=np.float64)
        priors = np.maximum(counts / counts.sum(), 1e-12)
        scores *= priors[None, :] ** (-alpha)
    probs = scores / scores.sum(axis=1, keepdims=True)
    # np.argmax follows LABELS order for deterministic ties (P1, then P2, then P3).
    pred = np.asarray(LABELS)[np.argmax(probs, axis=1)]
    top_label_count = np.max(np.stack([(labels == label).sum(axis=1) for label in LABELS], axis=1), axis=1)
    agreement = top_label_count / k
    similarity_top1 = sims[:, 0]
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return pred, probs, agreement, similarity_top1, margin


def knn_scores(similarity: np.ndarray, y_ref: np.ndarray, k: int, weighting: str,
               class_adjustment: str = "none") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-similarity, axis=1)[:, :k]
    sims = np.take_along_axis(similarity, order, axis=1)
    labels = y_ref[order]
    pred, probs, agreement, similarity_top1, margin = knn_from_sorted_neighbors(
        labels, sims, y_ref, k, weighting, class_adjustment
    )
    return pred, probs, agreement, similarity_top1, margin, order


def validation_abstention(y_true: np.ndarray, pred: np.ndarray, agreement: np.ndarray,
                          similarity: np.ndarray, margin: np.ndarray) -> tuple[dict, pd.DataFrame]:
    similarity_cuts = np.unique(np.quantile(similarity, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]))
    agreement_cuts = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    margin_cuts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rows: list[dict] = []
    for min_agreement in agreement_cuts:
        for min_similarity in similarity_cuts:
            for min_margin in margin_cuts:
                accepted = (agreement >= min_agreement) & (similarity >= min_similarity) & (margin >= min_margin)
                coverage = float(accepted.mean()) if len(accepted) else 0.0
                if accepted.any():
                    bal_acc = balanced_accuracy_score(y_true[accepted], pred[accepted])
                    macro_f1 = f1_score(y_true[accepted], pred[accepted], labels=LABELS, average="macro", zero_division=0)
                    p1_recall = precision_recall_fscore_support(y_true[accepted], pred[accepted], labels=LABELS, zero_division=0)[1][0]
                else:
                    bal_acc = macro_f1 = p1_recall = 0.0
                rows.append({
                    "min_neighbor_agreement": min_agreement,
                    "min_top1_cosine": float(min_similarity),
                    "min_vote_margin": min_margin,
                    "coverage": coverage,
                    "balanced_accuracy_accepted": bal_acc,
                    "macro_f1_accepted": macro_f1,
                    "P1_recall_accepted": p1_recall,
                    "validation_utility_coverage_x_balanced_accuracy": coverage * bal_acc,
                })
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["validation_utility_coverage_x_balanced_accuracy", "balanced_accuracy_accepted", "coverage"],
        ascending=False,
        kind="stable",
    ).iloc[0].to_dict()
    best["selection_note"] = "Validation-only heuristic: maximize coverage × selective balanced accuracy; thresholds require clinical agreement."
    return best, table.sort_values("validation_utility_coverage_x_balanced_accuracy", ascending=False)


def make_structured_model(class_weight: str | dict | None = "balanced", c_value: float = 1.0) -> Pipeline:
    numeric = ["Age at Visit Numeric", "BP first systolic", "BP first diastolic",
               "BP before clinician systolic", "BP before clinician diastolic"]
    categorical = [
        "Sex", "Arrival Method", "Chief Complaint", "KIV Uptriage",
        "Reatt Same ED", "Reatt Reason",
        "72h Reatt NGEMR ED", "72h Reatt NGEMR ED then Admitted",
        "HD/ICU from GW within 24H",
        "Triage Acuity", "Consult Acuity", "Order Description",
        "Working Diagnosis / Reason", "Relevant Clinical Information",
        "Primary ED Dx", "ED Dx", "First ED Care Area", "Primary Care Area", "CFS (ED)",
    ]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=2000, C=c_value, class_weight=class_weight, solver="lbfgs")),
    ])


def structured_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["Age at Visit Numeric"] = pd.to_numeric(
        frame["Age at Visit"].astype("string").str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    for source, stem in [("BP first", "BP first"),
                         ("BP before clinician", "BP before clinician")]:
        value = frame[source].astype("string") if source in frame else pd.Series(pd.NA, index=frame.index, dtype="string")
        parts = value.str.extract(r"(?<!\d)(\d{2,3})\s*/\s*(\d{2,3})(?!\d)")
        frame[f"{stem} systolic"] = pd.to_numeric(parts[0], errors="coerce")
        frame[f"{stem} diastolic"] = pd.to_numeric(parts[1], errors="coerce")
    return frame


def run_structured(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> tuple[list[dict], list[dict], np.ndarray, pd.DataFrame, dict]:
    weights = {
        "balanced": "balanced",
        "none": None,
        "P1_x1.5": {"P1": 1.5, "P2": 1.0, "P3": 1.0},
        "P1_x2": {"P1": 2.0, "P2": 1.0, "P3": 1.0},
    }
    candidates = []
    best_key = None
    best_config = None
    for c_value in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        for weight_name, class_weight in weights.items():
            model = make_structured_model(class_weight=class_weight, c_value=c_value)
            model.fit(structured_frame(train), y_train)
            val_pred = model.predict(structured_frame(val))
            macro = f1_score(y_val, val_pred, labels=LABELS, average="macro", zero_division=0)
            p1_recall = precision_recall_fscore_support(y_val, val_pred, labels=LABELS, zero_division=0)[1][0]
            bal = balanced_accuracy_score(y_val, val_pred)
            candidates.append({"C": c_value, "class_weight": weight_name,
                               "validation_macro_f1": macro,
                               "validation_balanced_accuracy": bal,
                               "validation_P1_recall": p1_recall})
            key = (macro, p1_recall, bal, -c_value, weight_name == "balanced")
            if best_key is None or key > best_key:
                best_key = key
                best_config = {"C": c_value, "class_weight": class_weight,
                               "class_weight_name": weight_name, "val_pred": val_pred}

    metric_rows, confusion_rows = metrics_for("structured_logistic_selected", y_val, best_config["val_pred"], "validation")
    final = make_structured_model(class_weight=best_config["class_weight"], c_value=best_config["C"])
    final.fit(structured_frame(pd.concat([train, val])), np.concatenate([y_train, y_val]))
    test_pred = final.predict(structured_frame(test))
    rows, cms = metrics_for("structured_logistic_selected", y_test, test_pred, "newdata_test")
    metric_rows.extend(rows); confusion_rows.extend(cms)
    selected = {"C": best_config["C"], "class_weight": best_config["class_weight_name"],
                "validation_macro_f1": best_key[0], "validation_P1_recall": best_key[1],
                "validation_balanced_accuracy": best_key[2]}
    return metric_rows, confusion_rows, test_pred, pd.DataFrame(candidates).sort_values(
        ["validation_macro_f1", "validation_P1_recall", "validation_balanced_accuracy"], ascending=False
    ), selected


def model_revision(model: object) -> str | None:
    try:
        return model[0].auto_model.config._commit_hash
    except Exception:
        try:
            return model[0].auto_model.config._name_or_path
        except Exception:
            return None


def model_snapshot_commit() -> str | None:
    """Record the cached Hub snapshot commit even when config omits it."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        ref = Path(HF_HUB_CACHE) / "models--google--embeddinggemma-2" / "refs" / "main"
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    except Exception:
        return None


def exploratory_embedding_diagnostics(vectors: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Descriptive geometry only; these are not measures of clinical safety."""
    rows: list[dict] = []
    means: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for label in LABELS:
        subset = vectors[labels == label]
        counts[label] = len(subset)
        mean = subset.mean(axis=0)
        norm = np.linalg.norm(mean)
        means[label] = mean / norm if norm else mean
        intra = 1.0 - (subset @ means[label])
        rows.append({"diagnostic": "mean_intra_class_cosine_distance", "group": label, "value": float(intra.mean()), "note": "Exploratory embedding geometry only"})
    for i, label_a in enumerate(LABELS):
        for label_b in LABELS[i + 1:]:
            rows.append({
                "diagnostic": "class_centroid_cosine_distance",
                "group": f"{label_a}-{label_b}",
                "value": float(1.0 - np.dot(means[label_a], means[label_b])),
                "note": "Exploratory embedding geometry only",
            })
    if len(vectors) >= 3 and len(np.unique(labels)) > 1:
        n_sample = min(2000, len(vectors))
        rows.append({
            "diagnostic": "silhouette_cosine_sampled",
            "group": f"n={n_sample}",
            "value": float(silhouette_score(vectors, labels, metric="cosine", sample_size=n_sample, random_state=RANDOM_SEED)),
            "note": "Exploratory diagnostic; not evidence of clinical reliability",
        })
    global_mean = vectors.mean(axis=0)
    between = 0.0
    within = 0.0
    for label in LABELS:
        subset = vectors[labels == label]
        if len(subset):
            center = subset.mean(axis=0)
            between += len(subset) * float(np.sum((center - global_mean) ** 2))
            within += float(np.sum((subset - center) ** 2))
    rows.append({
        "diagnostic": "fisher_between_within_scatter_ratio",
        "group": "all classes",
        "value": between / within if within else np.nan,
        "note": "Exploratory diagnostic; not evidence of clinical reliability",
    })
    return pd.DataFrame(rows)


def embedder(args: argparse.Namespace):
    from sentence_transformers import SentenceTransformer
    import torch

    dtype = torch.float32  # CPU inference; model card advises against float16.
    model = SentenceTransformer(
        MODEL_ID,
        device=args.device,
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"vision_config": None, "audio_config": None},
        local_files_only=args.local_files_only,
    )
    return model


def get_embeddings(model: object, texts: list[str], template_name: str, ids: list[str],
                   revision: str | None, args: argparse.Namespace) -> np.ndarray:
    cache_dir = ARTIFACTS_DIR / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.chmod(0o700)
    identity = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    text_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()
    cache_path = cache_dir / f"{template_name}.npz"
    metadata_path = cache_dir / f"{template_name}.json"
    if cache_path.exists() and metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if meta.get("model_id") == MODEL_ID and meta.get("model_revision") == revision and meta.get("ids_hash") == identity and meta.get("text_hash") == text_hash:
            return np.load(cache_path)["embeddings"]
    vectors = model.encode(
        texts,
        prompt=TASK_PREFIX,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 768:
        raise ValueError(f"Expected 768-dimensional embeddings; got {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding output contains non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Expected normalized embeddings for cosine dot products")
    np.savez_compressed(cache_path, embeddings=vectors)
    cache_path.chmod(0o600)
    json_write(metadata_path, {
        "model_id": MODEL_ID, "model_revision": revision, "dimension": 768,
        "task_prefix": TASK_PREFIX, "normalized": True,
        "precision": "float32", "device": args.device,
        "template": template_name, "ids_hash": identity, "text_hash": text_hash,
    })
    return vectors


def evaluate_embeddings(model: object, revision: str | None, hist: pd.DataFrame,
                        test: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
                        y: np.ndarray, y_test: np.ndarray, args: argparse.Namespace,
                        metric_rows: list[dict], confusion_rows: list[dict]) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict], list[dict], list[dict]]:
    hist = hist.reset_index(drop=True)
    test = test.reset_index(drop=True)
    y_train, y_val = y[train_idx], y[val_idx]
    y_test_scored = y_test
    val_result_rows: list[dict] = []
    configurations: list[dict] = []
    selected_payload = None
    best_key = None

    for template_name, fields in TEMPLATES.items():
        all_hist_ids = hist["Case ID"].astype(str).tolist()
        all_test_ids = test["Case ID"].astype(str).tolist()
        historical_texts = render_cases(hist, fields)
        test_texts = render_cases(test, fields)
        # Reuse pre-existing combined caches when available. For newly added
        # templates, encode only historical cases during model selection; the
        # test cases are encoded only once for the selected template.
        combined_texts = historical_texts + test_texts
        combined_ids = all_hist_ids + all_test_ids
        cache_meta_path = ARTIFACTS_DIR / "embeddings" / f"{template_name}.json"
        combined_cache_is_valid = False
        if cache_meta_path.exists():
            cache_meta = json.loads(cache_meta_path.read_text())
            combined_cache_is_valid = (
                cache_meta.get("model_id") == MODEL_ID
                and cache_meta.get("model_revision") == revision
                and cache_meta.get("ids_hash") == hashlib.sha256("\n".join(combined_ids).encode()).hexdigest()
                and cache_meta.get("text_hash") == hashlib.sha256("\n".join(combined_texts).encode()).hexdigest()
            )
        if combined_cache_is_valid:
            vectors = get_embeddings(model, combined_texts, template_name, combined_ids, revision, args)[:len(hist)]
        else:
            vectors = get_embeddings(model, historical_texts, template_name, all_hist_ids, revision, args)
        train_vec, val_vec = vectors[train_idx], vectors[val_idx]
        similarity_val = val_vec @ train_vec.T
        val_order = np.argsort(-similarity_val, axis=1)[:, :max(K_VALUES)]
        val_sims = np.take_along_axis(similarity_val, val_order, axis=1)
        val_neighbor_labels = y_train[val_order]

        for k in K_VALUES:
            labels_k = val_neighbor_labels[:, :k]
            sims_k = val_sims[:, :k]
            for weighting in ("uniform", "similarity_weighted"):
                for adjustment in CLASS_ADJUSTMENTS:
                    pred, probs, agreement, top_similarity, margin = knn_from_sorted_neighbors(
                        val_neighbor_labels, val_sims, y_train, k, weighting, adjustment
                    )
                    metrics, cms = metrics_for(
                        f"embedding_knn:{template_name}:k{k}:{weighting}:{adjustment}", y_val, pred, "validation"
                    )
                    metric_rows.extend(metrics); confusion_rows.extend(cms)
                    macro = f1_score(y_val, pred, labels=LABELS, average="macro", zero_division=0)
                    class_recall = precision_recall_fscore_support(y_val, pred, labels=LABELS, zero_division=0)[1]
                    p1_recall = class_recall[0]
                    bal = balanced_accuracy_score(y_val, pred)
                    configurations.append({
                        "template": template_name, "k": k, "weighting": weighting,
                        "class_adjustment": adjustment,
                        "validation_macro_f1": macro, "validation_balanced_accuracy": bal,
                        "validation_P1_recall": p1_recall,
                        "validation_P2_recall": class_recall[1],
                        "validation_P3_recall": class_recall[2],
                    })
                    val_result_rows.append({
                        "template": template_name, "k": k, "weighting": weighting,
                        "class_adjustment": adjustment,
                        "macro_f1": macro, "balanced_accuracy": bal, "P1_recall": p1_recall,
                    })
                    key = (macro, p1_recall, bal, -k, weighting == "uniform", -CLASS_ADJUSTMENTS[adjustment])
                    if best_key is None or key > best_key:
                        best_key = key
                        selected_payload = {
                            "template": template_name, "fields": fields, "k": k,
                            "weighting": weighting, "class_adjustment": adjustment,
                            "val_pred": pred, "val_probs": probs, "agreement": agreement,
                            "similarity": top_similarity, "margin": margin,
                        }

        # Only retain vectors while processing one template.
        del vectors, train_vec, val_vec, similarity_val

    if selected_payload is None:
        raise RuntimeError("No embedding k-NN configuration was evaluated")
    selected = selected_payload
    abstention_rule, abstention_val = validation_abstention(
        y_val, selected["val_pred"], selected["agreement"],
        selected["similarity"], selected["margin"],
    )

    # Encode the separate test orders only after model selection is complete.
    # Reuse the historical matrix created for tuning and encode test orders
    # separately; this avoids recomputing all historical vectors.
    selected_hist_texts = render_cases(hist, selected["fields"])
    selected_test_texts = render_cases(test, selected["fields"])
    selected_hist_ids = hist["Case ID"].astype(str).tolist()
    selected_test_ids = test["Case ID"].astype(str).tolist()
    combined_texts = selected_hist_texts + selected_test_texts
    combined_ids = selected_hist_ids + selected_test_ids
    combined_meta_path = ARTIFACTS_DIR / "embeddings" / f"{selected['template']}.json"
    combined_cache_is_valid = False
    if combined_meta_path.exists():
        combined_meta = json.loads(combined_meta_path.read_text())
        combined_cache_is_valid = (
            combined_meta.get("model_id") == MODEL_ID
            and combined_meta.get("model_revision") == revision
            and combined_meta.get("ids_hash") == hashlib.sha256("\n".join(combined_ids).encode()).hexdigest()
            and combined_meta.get("text_hash") == hashlib.sha256("\n".join(combined_texts).encode()).hexdigest()
        )
    if combined_cache_is_valid:
        vectors = get_embeddings(model, combined_texts, selected["template"], combined_ids, revision, args)
    else:
        hist_vectors = get_embeddings(model, selected_hist_texts, selected["template"], selected_hist_ids, revision, args)
        test_vectors = get_embeddings(model, selected_test_texts, selected["template"] + "_test", selected_test_ids, revision, args)
        vectors = np.concatenate([hist_vectors, test_vectors], axis=0)
    n_hist = len(hist)
    diagnostics = exploratory_embedding_diagnostics(vectors[:n_hist], hist["Priority Label"].to_numpy())
    test_mask = pd.Series(test["Priority Label"]).isin(LABELS).to_numpy()
    test_indices = np.flatnonzero(test_mask)
    final_similarity = vectors[n_hist:] @ vectors[:n_hist].T
    final_order = np.argsort(-final_similarity, axis=1)[:, :selected["k"]]
    final_sims = np.take_along_axis(final_similarity, final_order, axis=1)
    ref_labels = hist["Priority Label"].to_numpy()
    neighbor_labels = ref_labels[final_order]
    test_pred, probs, agreement, similarity_top1, margin, final_order = knn_scores(
        final_similarity, ref_labels, selected["k"], selected["weighting"], selected["class_adjustment"]
    )

    scored_pred = test_pred[test_mask]
    rows, cms = metrics_for(f"embedding_knn:{selected['template']}:k{selected['k']}:{selected['weighting']}:{selected['class_adjustment']}", y_test_scored, scored_pred, "newdata_test")
    metric_rows.extend(rows); confusion_rows.extend(cms)

    accepted = (
        (agreement >= float(abstention_rule["min_neighbor_agreement"])) &
        (similarity_top1 >= float(abstention_rule["min_top1_cosine"])) &
        (margin >= float(abstention_rule["min_vote_margin"]))
    )
    test_abst_rows = []
    accepted_scored = accepted & test_mask
    coverage = float(accepted.mean())
    scored_coverage = float(accepted_scored.sum() / max(1, test_mask.sum()))
    test_abst_rows.append({
        "split": "newdata_test", "coverage_all_orders": coverage,
        "coverage_labeled_orders": scored_coverage,
        "accepted_labeled": int(accepted_scored.sum()),
        "deferred_labeled": int((test_mask & ~accepted).sum()),
        "accepted_balanced_accuracy": balanced_accuracy_score(test["Priority Label"].to_numpy()[accepted_scored], test_pred[accepted_scored]) if accepted_scored.any() else 0.0,
        "accepted_macro_f1": f1_score(test["Priority Label"].to_numpy()[accepted_scored], test_pred[accepted_scored], labels=LABELS, average="macro", zero_division=0) if accepted_scored.any() else 0.0,
        "accepted_P1_recall": precision_recall_fscore_support(test["Priority Label"].to_numpy()[accepted_scored], test_pred[accepted_scored], labels=LABELS, zero_division=0)[1][0] if accepted_scored.any() else 0.0,
        "threshold_rule_status": "research candidate only; requires clinical agreement",
    })

    # Export de-identified explanatory neighbors and case-level suggestion or
    # defer state. Ground-truth priorities are marked evaluation-only.
    result_rows = []
    neighbor_rows = []
    for i, row in test.iterrows():
        result_rows.append({
            "Case ID": row["Case ID"], "month": row.get("month"),
            "Suggested Priority (research candidate)": test_pred[i],
            "Status": "SUGGEST (research candidate)" if accepted[i] else "DEFER to clinician",
            "Reference Priority (evaluation only)": row.get("Priority Label"),
            "Label Status": row.get("Label Status"),
            "Neighbor Agreement": agreement[i], "Top Cosine Similarity": similarity_top1[i],
            "Vote Margin": margin[i],
            **{f"P({lab})": probs[i, li] for li, lab in enumerate(LABELS)},
            "Order Description": row.get("Order Description"),
            "Working Diagnosis / Reason": row.get("Working Diagnosis / Reason"),
            "Relevant Clinical Information": row.get("Relevant Clinical Information"),
            "Primary ED Dx": row.get("Primary ED Dx"), "ED Dx": row.get("ED Dx"),
            **{f"Input: {field}": row.get(field) for field in selected["fields"]},
        })
        for rank, (idx, sim) in enumerate(zip(final_order[i], final_sims[i]), start=1):
            ref = hist.iloc[idx]
            neighbor_rows.append({
                "Query Case ID": row["Case ID"], "Rank": rank,
                "Neighbor Case ID": ref["Case ID"], "Cosine Similarity": float(sim),
                "Neighbor Priority": ref["Priority Label"],
                "Neighbor Order Description": ref.get("Order Description"),
                "Neighbor Working Diagnosis / Reason": ref.get("Working Diagnosis / Reason"),
                "Neighbor Primary ED Dx": ref.get("Primary ED Dx"),
                **{f"Neighbor {field}": ref.get(field) for field in selected["fields"]},
            })

    selected_meta = {
        "template": selected["template"], "fields": selected["fields"], "k": selected["k"],
        "weighting": selected["weighting"], "class_adjustment": selected["class_adjustment"],
        "validation_macro_f1": best_key[0],
        "validation_P1_recall": best_key[1], "validation_balanced_accuracy": best_key[2],
        "abstention_rule": abstention_rule,
    }
    configs = pd.DataFrame(configurations).sort_values(
        ["validation_macro_f1", "validation_P1_recall", "validation_balanced_accuracy"], ascending=False
    )
    return selected_meta, configs, abstention_val, pd.DataFrame(test_abst_rows), diagnostics, result_rows, neighbor_rows, test_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--structured-only", action="store_true", help="Run majority and structured baselines without loading embeddings")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.chmod(0o700)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.chmod(0o700)
    historical = pd.read_excel(DATA_DIR / "historical_cases.xlsx")
    test = pd.read_excel(DATA_DIR / "newdata_test_cases.xlsx")
    labeled_test = test[test["Priority Label"].isin(LABELS)].reset_index(drop=True)
    hist = historical[historical["Priority Label"].isin(LABELS)].reset_index(drop=True)
    y = hist["Priority Label"].to_numpy()
    y_test = labeled_test["Priority Label"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(hist, y, groups=hist["Patient Group ID"]))
    train_groups = set(hist.iloc[train_idx]["Patient Group ID"])
    val_groups = set(hist.iloc[val_idx]["Patient Group ID"])
    if train_groups & val_groups:
        raise RuntimeError("Patient leakage across train/validation split")
    test_groups = set(labeled_test["Patient Group ID"])
    if (train_groups | val_groups) & test_groups:
        raise RuntimeError("Patient overlap between historical and new-data test")

    split_manifest = {
        "random_seed": RANDOM_SEED,
        "splitter": "StratifiedGroupKFold(n_splits=5, shuffle=True), first fold held out as validation",
        "train_case_ids": hist.iloc[train_idx]["Case ID"].tolist(),
        "validation_case_ids": hist.iloc[val_idx]["Case ID"].tolist(),
        "test_case_ids": test["Case ID"].tolist(),
        "train_patient_group_ids": sorted(train_groups),
        "validation_patient_group_ids": sorted(val_groups),
        "test_patient_group_ids": sorted(set(test["Patient Group ID"])),
        "historical_labeled_rows": len(hist),
        "train_labeled_rows": len(train_idx),
        "validation_labeled_rows": len(val_idx),
        "test_all_rows": len(test),
        "test_labeled_rows": len(labeled_test),
    }
    json_write(BASELINE_DIR / "split_manifest.json", split_manifest)

    y_train, y_val = y[train_idx], y[val_idx]
    majority = pd.Series(y_train).value_counts().idxmax()
    y_val_majority = np.repeat(majority, len(y_val))
    y_test_majority = np.repeat(pd.Series(y).value_counts().idxmax(), len(y_test))
    metric_rows, confusion_rows = metrics_for("majority_class", y_val, y_val_majority, "validation")
    rows, cms = metrics_for("majority_class", y_test, y_test_majority, "newdata_test")
    metric_rows.extend(rows); confusion_rows.extend(cms)

    structured_rows, structured_cms, structured_test_pred, structured_configs, structured_selected = run_structured(
        hist.iloc[train_idx].reset_index(drop=True), hist.iloc[val_idx].reset_index(drop=True),
        labeled_test, y_train, y_val, y_test,
    )
    metric_rows.extend(structured_rows); confusion_rows.extend(structured_cms)
    structured_summary = {
        "majority_class": majority,
        "majority_test_class": pd.Series(y).value_counts().idxmax(),
        "structured_classifier": "validation-tuned multinomial logistic regression; numeric age + one-hot decision-time fields",
        "structured_selected_configuration": structured_selected,
    }
    import importlib.metadata
    package_versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "sentence-transformers", "transformers", "torch"]:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    prepared_sha256 = {}
    for name in ["historical_cases.xlsx", "newdata_test_cases.xlsx"]:
        prepared_sha256[name] = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()

    if args.structured_only:
        json_write(BASELINE_DIR / "run_metadata.json", {
            "run_utc": datetime.now(timezone.utc).isoformat(), "mode": "structured_only",
            "python": __import__("sys").version, "seed": RANDOM_SEED,
            "package_versions": package_versions, "prepared_data_sha256": prepared_sha256,
            "historical_labeled_class_counts": pd.Series(y).value_counts().to_dict(),
            "test_labeled_class_counts": pd.Series(y_test).value_counts().to_dict(),
            **structured_summary,
        })
        with pd.ExcelWriter(BASELINE_DIR / "evaluation_results.xlsx", engine="openpyxl") as writer:
            pd.DataFrame(metric_rows).to_excel(writer, sheet_name="Metrics", index=False)
            pd.DataFrame(confusion_rows).to_excel(writer, sheet_name="Confusion matrices", index=False)
            structured_configs.to_excel(writer, sheet_name="Structured validation configs", index=False)
        (BASELINE_DIR / "evaluation_results.xlsx").chmod(0o600)
        print("Structured-only results written to", BASELINE_DIR / "evaluation_results.xlsx")
        return

    model = embedder(args)
    revision = model_revision(model)
    selected, configs, abst_val, abst_test, diagnostics, review_rows, neighbor_rows, _ = evaluate_embeddings(
        model, revision, hist, test, train_idx, val_idx, y, y_test, args, metric_rows, confusion_rows
    )
    json_write(BASELINE_DIR / "run_metadata.json", {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "model_revision": revision,
        "model_snapshot_commit": model_snapshot_commit(),
        "task_prefix": TASK_PREFIX, "embedding_dimension": 768,
        "normalized_embeddings": True, "cosine_implementation": "dot product of L2-normalized vectors",
        "device": args.device, "precision": "float32", "batch_size": args.batch_size,
        "seed": RANDOM_SEED, "k_values": K_VALUES, "class_adjustments": CLASS_ADJUSTMENTS,
        "templates": TEMPLATES,
        "user_confirmed_pre_order_fields": USER_CONFIRMED_PRE_ORDER_FIELDS,
        "timing_provenance_assumption": "User confirmed the three listed utilization/outcome flags are available before the CT order is placed.",
        "package_versions": package_versions, "prepared_data_sha256": prepared_sha256,
        "historical_labeled_class_counts": pd.Series(y).value_counts().to_dict(),
        "test_labeled_class_counts": pd.Series(y_test).value_counts().to_dict(),
        "model_selection": "validation macro-F1; ties by P1 recall, balanced accuracy, smaller k, uniform voting, less class adjustment",
        "selected_configuration": selected, **structured_summary,
        "safety_note": "Research evaluation only. Abstention thresholds are validation-selected candidates and require clinical agreement; this output is not for clinical use.",
    })
    with pd.ExcelWriter(BASELINE_DIR / "evaluation_results.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(metric_rows).to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame(confusion_rows).to_excel(writer, sheet_name="Confusion matrices", index=False)
        configs.to_excel(writer, sheet_name="kNN validation configs", index=False)
        structured_configs.to_excel(writer, sheet_name="Structured validation configs", index=False)
        abst_val.to_excel(writer, sheet_name="Abstention validation", index=False)
        abst_test.to_excel(writer, sheet_name="Abstention test", index=False)
        diagnostics.to_excel(writer, sheet_name="Exploratory diagnostics", index=False)
        pd.DataFrame(review_rows).to_excel(writer, sheet_name="Review output", index=False)
        pd.DataFrame(neighbor_rows).to_excel(writer, sheet_name="Supporting neighbors", index=False)
        review_frame = pd.DataFrame(review_rows)
        p1_errors = review_frame[(review_frame["Reference Priority (evaluation only)"] == "P1") &
                                 (review_frame["Suggested Priority (research candidate)"] != "P1")]
        p1_errors.to_excel(writer, sheet_name="P1 false negatives", index=False)
    (BASELINE_DIR / "evaluation_results.xlsx").chmod(0o600)
    print("Selected configuration:", selected)
    print("Results written to", BASELINE_DIR / "evaluation_results.xlsx")


if __name__ == "__main__":
    main()
