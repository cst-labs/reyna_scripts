import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import evaluate_ktas_bqr
from utils.ktas_bqr import (
    RetrievedCandidate,
    build_ktas_balanced_quota_candidates,
    build_ktas_candidate_pool,
    compute_feature_difference_score,
    compute_feature_statistics,
    compute_label_entropy,
    compute_quota_targets,
    load_disagreement_metadata,
    make_retrieval_audit_record,
    rerank_candidates_with_disagreement,
    select_balanced_quota_examples,
    select_similarity_examples,
)
from utils.predictors import KTASSingleCaseRetriever


class FakeEncoder:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text):
        return np.asarray(self.mapping[text], dtype=float)


def make_training_df():
    return pd.DataFrame(
        [
            {
                "Age": 50 + idx,
                "Sex": "Female" if idx % 2 == 0 else "Male",
                "Arrival mode": "Private Vehicle" if idx < 3 else "119 Ambulance",
                "Injury": "No" if idx % 3 else "Yes",
                "Chief_complain": f"complaint-{idx}",
                "Mental": "Alert" if idx < 4 else "Verbal Response",
                "Pain": 1,
                "NRS_pain": float([2, 4, 6, 9, 10, 1][idx]),
                "SBP": float([120, 130, 140, 190, 210, 80][idx]),
                "DBP": float([70, 76, 84, 120, 130, 40][idx]),
                "HR": float([80, 86, 94, 140, 160, 55][idx]),
                "RR": float([18, 19, 21, 32, 36, 30][idx]),
                "BT": float([36.5, 36.7, 37.0, 38.2, 38.8, 35.0][idx]),
                "KTAS_RN": 3,
                "Diagnosis in ED": f"diagnosis-{idx}",
                "KTAS_expert": label,
                "Length of stay_min": 60 + idx,
            }
            for idx, label in enumerate([1, 2, 3, 4, 5, 3])
        ]
    )


class QuotaTests(unittest.TestCase):
    def test_quota_targets_sum_to_final_shot_count(self):
        training_df = pd.DataFrame({"KTAS_expert": [1] * 14 + [2] * 72 + [3] * 270 + [4] * 276 + [5] * 56})
        quotas = compute_quota_targets(
            training_df,
            final_shot_count=40,
            gamma=0.95,
            minimum_floor_per_class=1,
            ktas3_cap_fraction=1.0,
        )
        self.assertEqual(sum(quotas.values()), 40)
        self.assertGreaterEqual(min(quotas.values()), 1)
        self.assertGreater(quotas[3], quotas[2])

    def test_gamma_one_preserves_frequency_order(self):
        training_df = pd.DataFrame({"KTAS_expert": [1] * 10 + [2] * 20 + [3] * 40 + [4] * 30 + [5] * 15})
        quotas = compute_quota_targets(
            training_df,
            final_shot_count=20,
            gamma=1.0,
            minimum_floor_per_class=1,
            ktas3_cap_fraction=1.0,
        )
        self.assertGreaterEqual(quotas[3], quotas[4])
        self.assertGreaterEqual(quotas[4], quotas[2])
        self.assertGreaterEqual(quotas[2], quotas[5])
        self.assertGreaterEqual(quotas[5], quotas[1])


class DisagreementScoreTests(unittest.TestCase):
    def test_entropy_from_neighbor_histogram(self):
        entropy = compute_label_entropy({2: 2, 3: 2})
        self.assertGreater(entropy, 0.4)
        self.assertLessEqual(entropy, 1.0)
        self.assertEqual(compute_label_entropy({3: 4}), 0.0)

    def test_feature_difference_uses_continuous_and_categorical_fields(self):
        training_df = make_training_df()
        stats = compute_feature_statistics(training_df)
        anchor = training_df.iloc[0]
        neighbor = training_df.iloc[4]
        score = compute_feature_difference_score(anchor, neighbor, stats)
        self.assertGreater(score, 1.0)

    def test_load_disagreement_metadata_parses_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "example_id": 1,
                            "ktas_label": 3,
                            "neighbor_ids": [2, 3],
                            "neighbor_label_histogram": {"2": 1, "3": 1},
                            "disagreement_score": 0.5,
                            "feature_disagreement_score": 0.25,
                            "adjacent_boundary_counts": {"2_3": 1, "3_4": 0},
                            "neighborhood_size": 2,
                        }
                    )
                    + "\n"
                )
            metadata = load_disagreement_metadata(path)
            self.assertEqual(metadata[1].neighbor_label_histogram[2], 1)
            self.assertEqual(metadata[1].feature_disagreement_score, 0.25)


class RetrievalSelectionTests(unittest.TestCase):
    def test_balanced_selection_redistributes_when_class_runs_out(self):
        candidates = [
            type("C", (), {"example_id": 1, "ktas_label": 1, "similarity_score": 0.99, "final_score": None}),
            type("C", (), {"example_id": 2, "ktas_label": 2, "similarity_score": 0.98, "final_score": None}),
            type("C", (), {"example_id": 3, "ktas_label": 2, "similarity_score": 0.97, "final_score": None}),
            type("C", (), {"example_id": 4, "ktas_label": 3, "similarity_score": 0.96, "final_score": None}),
            type("C", (), {"example_id": 5, "ktas_label": 4, "similarity_score": 0.95, "final_score": None}),
            type("C", (), {"example_id": 6, "ktas_label": 5, "similarity_score": 0.94, "final_score": None}),
        ]
        selection = select_balanced_quota_examples(
            candidates,
            {1: 2, 2: 1, 3: 1, 4: 1, 5: 1},
            final_shot_count=6,
        )
        self.assertEqual(len(selection.selected_candidates), 6)
        self.assertEqual(selection.final_per_class_counts[1], 1)
        self.assertEqual(selection.selected_candidates[1].example_id, 2)

    def test_reranking_changes_order_when_disagreement_scores_present(self):
        candidates = [
            RetrievedCandidate(example_id=1, ktas_label=3, similarity_score=0.90, row={}),
            RetrievedCandidate(example_id=2, ktas_label=3, similarity_score=0.89, row={}),
        ]
        metadata = {
            1: type("M", (), {"disagreement_score": 0.0, "feature_disagreement_score": 0.0}),
            2: type("M", (), {"disagreement_score": 1.0, "feature_disagreement_score": 0.5}),
        }
        reranked = rerank_candidates_with_disagreement(candidates, metadata, alpha=0.3, beta=0.2)
        self.assertEqual(reranked[0].example_id, 2)
        self.assertGreater(reranked[0].final_score, reranked[1].final_score)

    def test_zero_weights_preserve_similarity_order(self):
        candidates = [
            RetrievedCandidate(example_id=1, ktas_label=3, similarity_score=0.90, row={}),
            RetrievedCandidate(example_id=2, ktas_label=3, similarity_score=0.89, row={}),
        ]
        metadata = {
            1: type("M", (), {"disagreement_score": 0.1, "feature_disagreement_score": 0.1}),
            2: type("M", (), {"disagreement_score": 1.0, "feature_disagreement_score": 1.0}),
        }
        reranked = rerank_candidates_with_disagreement(candidates, metadata, alpha=0.0, beta=0.0)
        self.assertEqual([candidate.example_id for candidate in reranked], [1, 2])
        selection = select_similarity_examples(reranked, final_shot_count=2)
        self.assertEqual([candidate.example_id for candidate in selection.selected_candidates], [1, 2])

    def test_per_class_vital_ranking_prefers_valid_vitals_then_text_similarity(self):
        candidates = [
            RetrievedCandidate(example_id=1, ktas_label=1, similarity_score=0.81, row={}, text_similarity=0.81, vital_similarity=None),
            RetrievedCandidate(example_id=2, ktas_label=1, similarity_score=0.75, row={}, text_similarity=0.75, vital_similarity=0.91),
            RetrievedCandidate(example_id=3, ktas_label=1, similarity_score=0.74, row={}, text_similarity=0.84, vital_similarity=0.91),
            RetrievedCandidate(example_id=4, ktas_label=2, similarity_score=0.90, row={}, text_similarity=0.90, vital_similarity=0.80),
        ]
        selection = select_balanced_quota_examples(
            candidates,
            {1: 2, 2: 1, 3: 0, 4: 0, 5: 0},
            final_shot_count=3,
            use_per_class_vital_ranking=True,
        )
        self.assertEqual([candidate.example_id for candidate in selection.selected_candidates[:2]], [3, 2])

    def test_per_class_vital_ranking_redistributes_shortfall_by_best_remaining_candidate(self):
        candidates = [
            RetrievedCandidate(example_id=1, ktas_label=1, similarity_score=0.70, row={}, text_similarity=0.70, vital_similarity=0.60),
            RetrievedCandidate(example_id=2, ktas_label=2, similarity_score=0.90, row={}, text_similarity=0.90, vital_similarity=0.93),
            RetrievedCandidate(example_id=3, ktas_label=3, similarity_score=0.85, row={}, text_similarity=0.85, vital_similarity=0.89),
            RetrievedCandidate(example_id=4, ktas_label=4, similarity_score=0.88, row={}, text_similarity=0.88, vital_similarity=0.91),
            RetrievedCandidate(example_id=5, ktas_label=5, similarity_score=0.83, row={}, text_similarity=0.83, vital_similarity=0.87),
        ]
        selection = select_balanced_quota_examples(
            candidates,
            {1: 2, 2: 1, 3: 1, 4: 1, 5: 1},
            final_shot_count=6,
            use_per_class_vital_ranking=True,
        )
        self.assertEqual(len(selection.selected_candidates), 5)
        self.assertEqual(selection.final_per_class_counts[1], 1)

    def test_audit_record_includes_disagreement_fields(self):
        candidates = [
            RetrievedCandidate(
                example_id=1,
                ktas_label=2,
                similarity_score=0.9,
                row={},
                disagreement_score=0.2,
                feature_disagreement_score=0.4,
                final_score=1.04,
            )
        ]
        selection = select_similarity_examples(candidates, final_shot_count=1)
        record = make_retrieval_audit_record(
            query_id=7,
            selection=selection,
            retrieval_strategy="disagreement_feature",
            candidate_pool_size=10,
            final_shot_count=1,
            gamma=0.8,
            minimum_floor_per_class=2,
            ktas3_cap_fraction=0.3,
            alpha=0.3,
            beta=0.2,
            neighborhood_size=40,
            precomputed_disagreement_path="artifact.jsonl",
        )
        self.assertIn("selected_disagreement_scores", record)
        self.assertIn("selected_feature_disagreement_scores", record)
        self.assertIn("selected_final_scores", record)
        self.assertEqual(record["selected_final_scores"][0], 1.04)


class BaselineCompatibilityTests(unittest.TestCase):
    def test_similarity_pool_matches_existing_retriever_top_k_when_no_missing_vitals(self):
        training_df = make_training_df()
        row = pd.Series(
            {
                "Chief_complain": "query-complaint",
                "Diagnosis in ED": "query-diagnosis",
                "NRS_pain": 2.0,
                "SBP": 120.0,
                "DBP": 70.0,
                "HR": 80.0,
                "RR": 18.0,
                "BT": 36.5,
            }
        )
        mapping = {
            **{f"complaint-{idx}": [float(idx + 1), 0.0] for idx in range(len(training_df))},
            **{f"diagnosis-{idx}": [0.0, float(idx + 1)] for idx in range(len(training_df))},
            "query-complaint": [3.0, 0.0],
            "query-diagnosis": [0.0, 3.0],
        }
        encoder = FakeEncoder(mapping)
        complaint_embeddings = np.asarray([mapping[f"complaint-{idx}"] for idx in range(len(training_df))], dtype=float)
        diagnosis_embeddings = np.asarray([mapping[f"diagnosis-{idx}"] for idx in range(len(training_df))], dtype=float)

        existing = KTASSingleCaseRetriever(
            training_set=training_df,
            complaint_embeddings=complaint_embeddings,
            diagnosis_embeddings=diagnosis_embeddings,
            symptom_encoder=encoder,
        )
        existing_ids = existing.retrieve_top_k_examples(row, 3).index.tolist()

        new_candidates = build_ktas_candidate_pool(
            row=row,
            training_set=training_df,
            complaint_embeddings=complaint_embeddings,
            diagnosis_embeddings=diagnosis_embeddings,
            symptom_encoder=encoder,
            candidate_pool_size=3,
            text_shortlist_factor=3,
        )
        new_ids = [candidate.example_id for candidate in new_candidates[:3]]
        self.assertEqual(existing_ids, new_ids)

    def test_balanced_quota_shortlist_uses_text_ranking_and_keeps_all_shortlisted_cases(self):
        training_df = pd.DataFrame(
            [
                {
                    "Chief_complain": f"complaint-{idx}",
                    "Diagnosis in ED": f"diagnosis-{idx}",
                    "KTAS_expert": (idx % 5) + 1,
                    "NRS_pain": float(idx + 1),
                    "SBP": float(100 + idx),
                    "DBP": float(60 + idx),
                    "HR": float(70 + idx),
                    "RR": float(18 + (idx % 4)),
                    "BT": 36.5,
                }
                for idx in range(130)
            ]
        )
        row = pd.Series(
            {
                "Chief_complain": "query-complaint",
                "Diagnosis in ED": "query-diagnosis",
                "NRS_pain": 5.0,
                "SBP": 120.0,
                "DBP": 70.0,
                "HR": 80.0,
                "RR": 18.0,
                "BT": 36.5,
            }
        )
        mapping = {
            **{f"complaint-{idx}": [float(idx + 1), 0.0] for idx in range(len(training_df))},
            **{f"diagnosis-{idx}": [0.0, float(idx + 1)] for idx in range(len(training_df))},
            "query-complaint": [120.0, 0.0],
            "query-diagnosis": [0.0, 120.0],
        }
        encoder = FakeEncoder(mapping)
        complaint_embeddings = np.asarray([mapping[f"complaint-{idx}"] for idx in range(len(training_df))], dtype=float)
        diagnosis_embeddings = np.asarray([mapping[f"diagnosis-{idx}"] for idx in range(len(training_df))], dtype=float)

        candidates = build_ktas_balanced_quota_candidates(
            row=row,
            training_set=training_df,
            complaint_embeddings=complaint_embeddings,
            diagnosis_embeddings=diagnosis_embeddings,
            symptom_encoder=encoder,
            shortlist_size=120,
        )
        self.assertEqual(len(candidates), 120)
        self.assertEqual(candidates[0].example_id, 129)
        self.assertEqual(candidates[-1].example_id, 10)
        self.assertTrue(all(candidate.text_similarity is not None for candidate in candidates))


class EvaluationScriptTests(unittest.TestCase):
    def test_evaluation_payload_includes_boundary_and_retrieval_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            predictions_path = tmp_path / "predictions.csv"
            retrieval_path = tmp_path / "predictions_retrieval_audit.jsonl"
            pd.DataFrame(
                [
                    {"KTAS_expert": 2, "Estimated_Acuity": 3},
                    {"KTAS_expert": 3, "Estimated_Acuity": 4},
                    {"KTAS_expert": 4, "Estimated_Acuity": 3},
                ]
            ).to_csv(predictions_path, index=False)
            with retrieval_path.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "query_id": 0,
                            "retrieval_strategy": "disagreement_feature",
                            "final_per_class_counts": {"1": 2, "2": 8, "3": 12, "4": 10, "5": 8},
                            "quota_targets": {"1": 2, "2": 8, "3": 12, "4": 10, "5": 8},
                            "selected_disagreement_scores": [0.2, 0.4],
                            "selected_feature_disagreement_scores": [0.1, 0.3],
                            "candidate_pool_size": 200,
                            "final_shot_count": 40,
                            "alpha": 0.3,
                            "beta": 0.2,
                            "neighborhood_size": 40,
                        }
                    )
                    + "\n"
                )

            payload = evaluate_ktas_bqr.build_metrics_payload(predictions_path, retrieval_path)
            self.assertIn("confusion_matrix", payload)
            self.assertIn("retrieval_distribution_summary", payload)
            self.assertIn("boundary_error_summary", payload)
            self.assertEqual(payload["boundary_error_summary"]["2_to_3"]["count"], 1)
            self.assertAlmostEqual(
                payload["retrieval_distribution_summary"]["score_summary"]["mean_selected_disagreement_score"],
                0.3,
            )


if __name__ == "__main__":
    unittest.main()
