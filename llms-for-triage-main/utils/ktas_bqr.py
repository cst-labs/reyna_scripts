from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KTAS_LABELS = [1, 2, 3, 4, 5]
KTAS_VITAL_COLUMNS = ["NRS_pain", "SBP", "DBP", "HR", "RR", "BT"]
KTAS_CONTINUOUS_FEATURES = ["Age", "SBP", "DBP", "HR", "RR", "BT", "NRS_pain"]
KTAS_CATEGORICAL_FEATURES = ["Sex", "Mental", "Injury", "Arrival mode"]
KTAS_CATEGORICAL_MISMATCH_PENALTIES = {
    "Sex": 0.25,
    "Mental": 1.0,
    "Injury": 0.75,
    "Arrival mode": 0.5,
}


@dataclass(frozen=True)
class RetrievedCandidate:
    example_id: int
    ktas_label: int
    similarity_score: float
    row: dict[str, Any]
    text_similarity: float | None = None
    vital_similarity: float | None = None
    disagreement_score: float = 0.0
    feature_disagreement_score: float = 0.0
    final_score: float | None = None


@dataclass(frozen=True)
class RetrievalSelection:
    selected_candidates: list[RetrievedCandidate]
    quota_targets: dict[int, int]
    final_per_class_counts: dict[int, int]
    candidate_pool_per_class_counts: dict[int, int]


@dataclass(frozen=True)
class DisagreementMetadata:
    example_id: int
    ktas_label: int
    neighbor_ids: list[int]
    neighbor_label_histogram: dict[int, int]
    disagreement_score: float
    feature_disagreement_score: float
    adjacent_boundary_counts: dict[str, int]
    neighborhood_size: int


def _normalize_vector(values: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm == 0:
        return None
    return values / norm


def _top_k_indices(similarities: np.ndarray, k: int) -> np.ndarray:
    k = max(0, min(k, len(similarities)))
    if k == 0:
        return np.array([], dtype=int)
    return np.argsort(similarities)[-k:][::-1]


def _score_for_ordering(candidate: RetrievedCandidate) -> float:
    return candidate.final_score if candidate.final_score is not None else candidate.similarity_score


def _candidate_vital_rank_key(candidate: RetrievedCandidate) -> tuple[bool, float, float, float]:
    vital_similarity = candidate.vital_similarity
    return (
        vital_similarity is not None,
        vital_similarity if vital_similarity is not None else float("-inf"),
        candidate.text_similarity if candidate.text_similarity is not None else float("-inf"),
        candidate.similarity_score,
    )


def _get_text_similarity_ranking(
    row: pd.Series,
    complaint_embeddings: np.ndarray,
    diagnosis_embeddings: np.ndarray,
    symptom_encoder: Any,
    shortlist_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    symptom = row["Chief_complain"]
    diagnosis = row["Diagnosis in ED"]

    if pd.isna(symptom) or symptom == "":
        query_embedding = np.asarray(symptom_encoder.encode(diagnosis))
        similarities = np.dot(diagnosis_embeddings, query_embedding)
    elif pd.isna(diagnosis) or diagnosis == "":
        query_embedding = np.asarray(symptom_encoder.encode(symptom))
        similarities = np.dot(complaint_embeddings, query_embedding)
    else:
        symptom_embedding = np.asarray(symptom_encoder.encode(symptom))
        diagnosis_embedding = np.asarray(symptom_encoder.encode(diagnosis))
        query_embedding = np.concatenate([symptom_embedding, diagnosis_embedding])
        similarities = np.dot(
            np.concatenate([complaint_embeddings, diagnosis_embeddings], axis=1),
            query_embedding,
        )

    top_indices = _top_k_indices(similarities, shortlist_size)
    return top_indices, similarities[top_indices]


def _compute_vital_similarity_candidates(
    row: pd.Series,
    candidate_indices: np.ndarray,
    candidate_text_similarities: np.ndarray,
    training_set: pd.DataFrame,
) -> list[RetrievedCandidate]:
    vital_columns = [column for column in KTAS_VITAL_COLUMNS if pd.notna(row.get(column))]
    if not vital_columns:
        return []

    query_vitals = pd.to_numeric(row[vital_columns], errors="coerce")
    if query_vitals.isna().any():
        return []

    query_vector = _normalize_vector(query_vitals.to_numpy(dtype=float))
    if query_vector is None:
        return []

    candidate_frame = training_set.loc[candidate_indices, vital_columns].apply(pd.to_numeric, errors="coerce")
    valid_mask = ~candidate_frame.isna().any(axis=1)
    valid_candidates = candidate_frame.loc[valid_mask]
    if valid_candidates.empty:
        return []

    candidate_values = valid_candidates.to_numpy(dtype=float)
    norms = np.linalg.norm(candidate_values, axis=1, keepdims=True)
    finite_mask = np.isfinite(norms).ravel() & (norms.ravel() != 0)
    valid_candidates = valid_candidates.iloc[finite_mask]
    if valid_candidates.empty:
        return []

    candidate_values = valid_candidates.to_numpy(dtype=float)
    normalized_values = candidate_values / np.linalg.norm(candidate_values, axis=1, keepdims=True)
    vital_similarities = np.dot(normalized_values, query_vector)
    order = np.argsort(vital_similarities)[::-1]

    return [
        RetrievedCandidate(
            example_id=int(valid_candidates.index[position]),
            ktas_label=int(training_set.loc[int(valid_candidates.index[position]), "KTAS_expert"]),
            similarity_score=float(vital_similarities[position]),
            row=training_set.loc[int(valid_candidates.index[position])].to_dict(),
            text_similarity=float(
                candidate_text_similarities[
                    int(np.where(candidate_indices == int(valid_candidates.index[position]))[0][0])
                ]
            ),
            vital_similarity=float(vital_similarities[position]),
        )
        for position in order
    ]


def build_ktas_candidate_pool(
    *,
    row: pd.Series,
    training_set: pd.DataFrame,
    complaint_embeddings: np.ndarray,
    diagnosis_embeddings: np.ndarray,
    symptom_encoder: Any,
    candidate_pool_size: int = 200,
    text_shortlist_factor: int = 3,
) -> list[RetrievedCandidate]:
    shortlist_size = max(candidate_pool_size * text_shortlist_factor, candidate_pool_size)
    candidate_indices, candidate_text_similarities = _get_text_similarity_ranking(
        row=row,
        complaint_embeddings=np.asarray(complaint_embeddings),
        diagnosis_embeddings=np.asarray(diagnosis_embeddings),
        symptom_encoder=symptom_encoder,
        shortlist_size=shortlist_size,
    )

    ranked_by_vitals = _compute_vital_similarity_candidates(
        row=row,
        candidate_indices=candidate_indices,
        candidate_text_similarities=candidate_text_similarities,
        training_set=training_set,
    )

    selected_ids = {candidate.example_id for candidate in ranked_by_vitals}
    fallback_candidates: list[RetrievedCandidate] = []
    if len(ranked_by_vitals) < candidate_pool_size:
        for position, example_id in enumerate(candidate_indices):
            example_id = int(example_id)
            if example_id in selected_ids:
                continue
            text_similarity = float(candidate_text_similarities[position])
            fallback_candidates.append(
                RetrievedCandidate(
                    example_id=example_id,
                    ktas_label=int(training_set.loc[example_id, "KTAS_expert"]),
                    similarity_score=text_similarity,
                    row=training_set.loc[example_id].to_dict(),
                    text_similarity=text_similarity,
                    vital_similarity=None,
                )
            )

    ranked_candidates = ranked_by_vitals + fallback_candidates
    return ranked_candidates[:candidate_pool_size]


def build_ktas_balanced_quota_candidates(
    *,
    row: pd.Series,
    training_set: pd.DataFrame,
    complaint_embeddings: np.ndarray,
    diagnosis_embeddings: np.ndarray,
    symptom_encoder: Any,
    shortlist_size: int = 120,
) -> list[RetrievedCandidate]:
    candidate_indices, candidate_text_similarities = _get_text_similarity_ranking(
        row=row,
        complaint_embeddings=np.asarray(complaint_embeddings),
        diagnosis_embeddings=np.asarray(diagnosis_embeddings),
        symptom_encoder=symptom_encoder,
        shortlist_size=shortlist_size,
    )

    vital_columns = [column for column in KTAS_VITAL_COLUMNS if pd.notna(row.get(column))]
    query_vector: np.ndarray | None = None
    if vital_columns:
        query_vitals = pd.to_numeric(row[vital_columns], errors="coerce")
        if not query_vitals.isna().any():
            query_vector = _normalize_vector(query_vitals.to_numpy(dtype=float))

    candidates: list[RetrievedCandidate] = []
    for position, example_id in enumerate(candidate_indices):
        example_id = int(example_id)
        vital_similarity: float | None = None
        if query_vector is not None and vital_columns:
            candidate_vitals = pd.to_numeric(training_set.loc[example_id, vital_columns], errors="coerce")
            if not candidate_vitals.isna().any():
                candidate_vector = _normalize_vector(candidate_vitals.to_numpy(dtype=float))
                if candidate_vector is not None:
                    vital_similarity = float(np.dot(candidate_vector, query_vector))

        text_similarity = float(candidate_text_similarities[position])
        candidates.append(
            RetrievedCandidate(
                example_id=example_id,
                ktas_label=int(training_set.loc[example_id, "KTAS_expert"]),
                similarity_score=vital_similarity if vital_similarity is not None else text_similarity,
                row=training_set.loc[example_id].to_dict(),
                text_similarity=text_similarity,
                vital_similarity=vital_similarity,
            )
        )
    return candidates


def compute_quota_targets(
    training_set: pd.DataFrame,
    *,
    final_shot_count: int = 40,
    gamma: float = 0.8,
    minimum_floor_per_class: int = 2,
    ktas3_cap_fraction: float = 0.30,
) -> dict[int, int]:
    if final_shot_count <= 0:
        raise ValueError("final_shot_count must be positive.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if minimum_floor_per_class < 0:
        raise ValueError("minimum_floor_per_class must be non-negative.")
    if not 0 < ktas3_cap_fraction <= 1:
        raise ValueError("ktas3_cap_fraction must be in the interval (0, 1].")
    if minimum_floor_per_class * len(KTAS_LABELS) > final_shot_count:
        raise ValueError("minimum floor is infeasible for the requested final shot count.")

    class_counts = training_set["KTAS_expert"].value_counts().to_dict()
    weights = {
        label: float(class_counts.get(label, 0)) ** gamma
        for label in KTAS_LABELS
    }
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        raise ValueError("Training set has no KTAS labels to derive quotas from.")

    raw = {
        label: (weights[label] / weight_sum) * final_shot_count
        for label in KTAS_LABELS
    }

    quotas = {label: int(math.floor(raw[label])) for label in KTAS_LABELS}
    allocated = sum(quotas.values())
    remainders = sorted(
        KTAS_LABELS,
        key=lambda label: (raw[label] - quotas[label], raw[label]),
        reverse=True,
    )
    for label in remainders[: final_shot_count - allocated]:
        quotas[label] += 1

    minimums = {label: minimum_floor_per_class for label in KTAS_LABELS}
    maximums = {label: final_shot_count for label in KTAS_LABELS}
    maximums[3] = min(final_shot_count, int(math.floor(final_shot_count * ktas3_cap_fraction)))
    if maximums[3] < minimums[3]:
        maximums[3] = minimums[3]

    for label in KTAS_LABELS:
        quotas[label] = min(max(quotas[label], minimums[label]), maximums[label])

    def add_one() -> None:
        candidates = [label for label in KTAS_LABELS if quotas[label] < maximums[label]]
        if not candidates:
            raise ValueError("Unable to allocate quota while respecting class caps.")
        label = max(candidates, key=lambda item: (raw[item] - quotas[item], raw[item], -item))
        quotas[label] += 1

    def remove_one() -> None:
        candidates = [label for label in KTAS_LABELS if quotas[label] > minimums[label]]
        if not candidates:
            raise ValueError("Unable to reduce quota while respecting class floors.")
        label = max(candidates, key=lambda item: (quotas[item] - raw[item], quotas[item], item))
        quotas[label] -= 1

    while sum(quotas.values()) < final_shot_count:
        add_one()
    while sum(quotas.values()) > final_shot_count:
        remove_one()

    return quotas


def compute_label_entropy(label_histogram: dict[int | str, int]) -> float:
    counts = [count for count in label_histogram.values() if count > 0]
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        probability = count / total
        entropy -= probability * math.log(probability)
    max_entropy = math.log(len(KTAS_LABELS))
    if max_entropy == 0:
        return 0.0
    return float(entropy / max_entropy)


def compute_feature_statistics(training_set: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in KTAS_CONTINUOUS_FEATURES:
        series = pd.to_numeric(training_set[column], errors="coerce")
        mean = float(series.mean()) if not series.dropna().empty else 0.0
        std = float(series.std(ddof=0)) if not series.dropna().empty else 1.0
        if not np.isfinite(std) or std == 0:
            std = 1.0
        stats[column] = {"mean": mean, "std": std}
    return stats


def compute_feature_difference_score(
    anchor_row: pd.Series | dict[str, Any],
    neighbor_row: pd.Series | dict[str, Any],
    feature_stats: dict[str, dict[str, float]],
) -> float:
    continuous_scores: list[float] = []
    for column in KTAS_CONTINUOUS_FEATURES:
        anchor_value = pd.to_numeric(pd.Series([anchor_row.get(column)]), errors="coerce").iloc[0]
        neighbor_value = pd.to_numeric(pd.Series([neighbor_row.get(column)]), errors="coerce").iloc[0]
        if pd.isna(anchor_value) or pd.isna(neighbor_value):
            continue
        std = feature_stats.get(column, {}).get("std", 1.0) or 1.0
        continuous_scores.append(abs(float(anchor_value) - float(neighbor_value)) / std)

    categorical_penalty = 0.0
    for column in KTAS_CATEGORICAL_FEATURES:
        anchor_value = anchor_row.get(column)
        neighbor_value = neighbor_row.get(column)
        if pd.isna(anchor_value) or pd.isna(neighbor_value):
            continue
        if str(anchor_value) != str(neighbor_value):
            categorical_penalty += KTAS_CATEGORICAL_MISMATCH_PENALTIES[column]

    continuous_component = float(sum(continuous_scores) / len(continuous_scores)) if continuous_scores else 0.0
    return continuous_component + categorical_penalty


def compute_local_feature_disagreement(
    anchor_row: pd.Series,
    neighbor_rows: pd.DataFrame,
    feature_stats: dict[str, dict[str, float]],
) -> float:
    if neighbor_rows.empty:
        return 0.0
    scores = [
        compute_feature_difference_score(anchor_row, neighbor_row, feature_stats)
        for _, neighbor_row in neighbor_rows.iterrows()
    ]
    return float(sum(scores) / len(scores)) if scores else 0.0


def normalize_feature_disagreement_scores(raw_scores: dict[int, float]) -> dict[int, float]:
    if not raw_scores:
        return {}
    values = np.asarray(list(raw_scores.values()), dtype=float)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return {example_id: 0.0 for example_id in raw_scores}
    return {
        example_id: float((score - minimum) / (maximum - minimum))
        for example_id, score in raw_scores.items()
    }


def compute_adjacent_boundary_counts(labels: list[int]) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {
        "2_3": min(counts.get(2, 0), counts.get(3, 0)),
        "3_4": min(counts.get(3, 0), counts.get(4, 0)),
    }


def load_disagreement_metadata(path: str | Path) -> dict[int, DisagreementMetadata]:
    metadata_by_id: dict[int, DisagreementMetadata] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            metadata = DisagreementMetadata(
                example_id=int(payload["example_id"]),
                ktas_label=int(payload["ktas_label"]),
                neighbor_ids=[int(example_id) for example_id in payload["neighbor_ids"]],
                neighbor_label_histogram={
                    int(label): int(count)
                    for label, count in payload["neighbor_label_histogram"].items()
                },
                disagreement_score=float(payload["disagreement_score"]),
                feature_disagreement_score=float(payload["feature_disagreement_score"]),
                adjacent_boundary_counts={
                    str(boundary): int(count)
                    for boundary, count in payload["adjacent_boundary_counts"].items()
                },
                neighborhood_size=int(payload["neighborhood_size"]),
            )
            metadata_by_id[metadata.example_id] = metadata
    return metadata_by_id


def select_similarity_examples(
    candidates: list[RetrievedCandidate],
    *,
    final_shot_count: int,
) -> RetrievalSelection:
    selected = candidates[:final_shot_count]
    per_class_counts = Counter(candidate.ktas_label for candidate in selected)
    candidate_pool_counts = Counter(candidate.ktas_label for candidate in candidates)
    return RetrievalSelection(
        selected_candidates=selected,
        quota_targets=dict(per_class_counts),
        final_per_class_counts={label: per_class_counts.get(label, 0) for label in KTAS_LABELS},
        candidate_pool_per_class_counts={label: candidate_pool_counts.get(label, 0) for label in KTAS_LABELS},
    )


def select_balanced_quota_examples(
    candidates: list[RetrievedCandidate],
    quota_targets: dict[int, int],
    *,
    final_shot_count: int,
    use_per_class_vital_ranking: bool = False,
) -> RetrievalSelection:
    by_class: dict[int, list[RetrievedCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_class[candidate.ktas_label].append(candidate)
    for label in KTAS_LABELS:
        if use_per_class_vital_ranking:
            by_class[label].sort(key=_candidate_vital_rank_key, reverse=True)
        else:
            by_class[label].sort(key=_score_for_ordering, reverse=True)

    selected: list[RetrievedCandidate] = []
    selected_ids: set[int] = set()
    class_positions = {label: 0 for label in KTAS_LABELS}

    unmet_slots = 0
    for label in KTAS_LABELS:
        quota = quota_targets.get(label, 0)
        class_candidates = by_class[label]
        take_count = min(quota, len(class_candidates))
        for candidate in class_candidates[:take_count]:
            selected.append(candidate)
            selected_ids.add(candidate.example_id)
        class_positions[label] = take_count
        unmet_slots += quota - take_count

    def next_unused_candidates() -> list[RetrievedCandidate]:
        available: list[RetrievedCandidate] = []
        for label in KTAS_LABELS:
            position = class_positions[label]
            class_candidates = by_class[label]
            while position < len(class_candidates) and class_candidates[position].example_id in selected_ids:
                position += 1
            class_positions[label] = position
            if position < len(class_candidates):
                available.append(class_candidates[position])
        return available

    while unmet_slots > 0:
        available = next_unused_candidates()
        if not available:
            break
        candidate = (
            max(available, key=_candidate_vital_rank_key)
            if use_per_class_vital_ranking
            else max(available, key=_score_for_ordering)
        )
        selected.append(candidate)
        selected_ids.add(candidate.example_id)
        class_positions[candidate.ktas_label] += 1
        unmet_slots -= 1

    if len(selected) < final_shot_count:
        remaining = [candidate for candidate in candidates if candidate.example_id not in selected_ids]
        remaining.sort(
            key=_candidate_vital_rank_key if use_per_class_vital_ranking else _score_for_ordering,
            reverse=True,
        )
        for candidate in remaining[: final_shot_count - len(selected)]:
            selected.append(candidate)
            selected_ids.add(candidate.example_id)

    selected = selected[:final_shot_count]
    per_class_counts = Counter(candidate.ktas_label for candidate in selected)
    candidate_pool_counts = Counter(candidate.ktas_label for candidate in candidates)
    return RetrievalSelection(
        selected_candidates=selected,
        quota_targets={label: quota_targets.get(label, 0) for label in KTAS_LABELS},
        final_per_class_counts={label: per_class_counts.get(label, 0) for label in KTAS_LABELS},
        candidate_pool_per_class_counts={label: candidate_pool_counts.get(label, 0) for label in KTAS_LABELS},
    )


def rerank_candidates_with_disagreement(
    candidates: list[RetrievedCandidate],
    metadata_by_id: dict[int, DisagreementMetadata],
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> list[RetrievedCandidate]:
    if alpha == 0 and beta == 0:
        return candidates

    reranked: list[RetrievedCandidate] = []
    for candidate in candidates:
        metadata = metadata_by_id.get(candidate.example_id)
        disagreement_score = metadata.disagreement_score if metadata is not None else 0.0
        feature_disagreement_score = metadata.feature_disagreement_score if metadata is not None else 0.0
        final_score = (
            candidate.similarity_score
            + alpha * disagreement_score
            + beta * feature_disagreement_score
        )
        reranked.append(
            replace(
                candidate,
                disagreement_score=disagreement_score,
                feature_disagreement_score=feature_disagreement_score,
                final_score=final_score,
            )
        )

    return sorted(
        reranked,
        key=lambda candidate: (
            candidate.final_score if candidate.final_score is not None else candidate.similarity_score,
            candidate.similarity_score,
        ),
        reverse=True,
    )


def make_retrieval_audit_record(
    *,
    query_id: Any,
    selection: RetrievalSelection,
    retrieval_strategy: str,
    candidate_pool_size: int,
    final_shot_count: int,
    gamma: float,
    minimum_floor_per_class: int,
    ktas3_cap_fraction: float,
    alpha: float = 0.0,
    beta: float = 0.0,
    neighborhood_size: int | None = None,
    precomputed_disagreement_path: str | None = None,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "retrieval_strategy": retrieval_strategy,
        "candidate_pool_size": candidate_pool_size,
        "final_shot_count": final_shot_count,
        "gamma": gamma,
        "minimum_floor_per_class": minimum_floor_per_class,
        "ktas3_cap_fraction": ktas3_cap_fraction,
        "alpha": alpha,
        "beta": beta,
        "neighborhood_size": neighborhood_size,
        "precomputed_disagreement_path": precomputed_disagreement_path,
        "selected_example_ids": [candidate.example_id for candidate in selection.selected_candidates],
        "selected_ktas_labels": [candidate.ktas_label for candidate in selection.selected_candidates],
        "selected_similarity_scores": [candidate.similarity_score for candidate in selection.selected_candidates],
        "selected_text_similarity_scores": [candidate.text_similarity for candidate in selection.selected_candidates],
        "selected_vital_similarity_scores": [candidate.vital_similarity for candidate in selection.selected_candidates],
        "selected_disagreement_scores": [candidate.disagreement_score for candidate in selection.selected_candidates],
        "selected_feature_disagreement_scores": [
            candidate.feature_disagreement_score for candidate in selection.selected_candidates
        ],
        "selected_final_scores": [
            candidate.final_score if candidate.final_score is not None else candidate.similarity_score
            for candidate in selection.selected_candidates
        ],
        "final_per_class_counts": selection.final_per_class_counts,
        "quota_targets": selection.quota_targets,
        "candidate_pool_per_class_counts": selection.candidate_pool_per_class_counts,
    }
