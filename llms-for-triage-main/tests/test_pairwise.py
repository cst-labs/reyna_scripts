import unittest

import numpy as np
import pandas as pd

from utils.pairwise import (
    PairPromptBuilder,
    PairRetriever,
    build_cross_pair_examples,
    derive_pair_label,
    normalize_pairwise_decision,
)
from utils.predictors import KTASSingleCaseRetriever


class DummyEncoder:
    def encode(self, text):
        text = "" if text is None else str(text)
        return np.array([len(text) % 7, sum(ord(char) for char in text) % 11], dtype=float)


def make_training_df():
    return pd.DataFrame(
        [
            {
                "Sex": "Male",
                "Age": 81,
                "Arrival mode": "119 Ambulance",
                "Injury": "No",
                "Chief_complain": "chest pain",
                "Mental": "Alert",
                "Pain": 1,
                "NRS_pain": 8.0,
                "SBP": 90.0,
                "DBP": 60.0,
                "HR": 115.0,
                "RR": 24.0,
                "BT": 37.2,
                "Diagnosis in ED": "acute coronary syndrome",
                "KTAS_expert": 2,
            },
            {
                "Sex": "Female",
                "Age": 45,
                "Arrival mode": "Private Vehicle",
                "Injury": "Yes",
                "Chief_complain": "ankle injury",
                "Mental": "Alert",
                "Pain": 1,
                "NRS_pain": 4.0,
                "SBP": 132.0,
                "DBP": 78.0,
                "HR": 88.0,
                "RR": 18.0,
                "BT": 36.7,
                "Diagnosis in ED": "ankle sprain",
                "KTAS_expert": 4,
            },
            {
                "Sex": "Male",
                "Age": 27,
                "Arrival mode": "Walking",
                "Injury": "No",
                "Chief_complain": "sore throat",
                "Mental": "Alert",
                "Pain": 1,
                "NRS_pain": 2.0,
                "SBP": 126.0,
                "DBP": 80.0,
                "HR": 76.0,
                "RR": 16.0,
                "BT": 37.0,
                "Diagnosis in ED": "pharyngitis",
                "KTAS_expert": 5,
            },
            {
                "Sex": "Female",
                "Age": 68,
                "Arrival mode": "Private Ambulance",
                "Injury": "No",
                "Chief_complain": "shortness of breath",
                "Mental": "Verbal Response",
                "Pain": 0,
                "NRS_pain": 0.0,
                "SBP": 100.0,
                "DBP": 58.0,
                "HR": 110.0,
                "RR": 28.0,
                "BT": 37.8,
                "Diagnosis in ED": "pneumonia",
                "KTAS_expert": 3,
            },
            {
                "Sex": "Male",
                "Age": 59,
                "Arrival mode": "Wheelchair",
                "Injury": "Yes",
                "Chief_complain": "head laceration",
                "Mental": "Alert",
                "Pain": 1,
                "NRS_pain": 5.0,
                "SBP": 140.0,
                "DBP": 86.0,
                "HR": 96.0,
                "RR": 19.0,
                "BT": 36.9,
                "Diagnosis in ED": "scalp laceration",
                "KTAS_expert": 4,
            },
            {
                "Sex": "Female",
                "Age": 33,
                "Arrival mode": "Private Vehicle",
                "Injury": "No",
                "Chief_complain": "migraine",
                "Mental": "Alert",
                "Pain": 1,
                "NRS_pain": 6.0,
                "SBP": 118.0,
                "DBP": 72.0,
                "HR": 82.0,
                "RR": 18.0,
                "BT": 36.8,
                "Diagnosis in ED": "headache",
                "KTAS_expert": 3,
            },
        ]
    )


def make_embeddings(row_count):
    complaint = np.array([[i + 1.0, i + 2.0] for i in range(row_count)], dtype=float)
    diagnosis = np.array([[i + 0.5, i + 1.5] for i in range(row_count)], dtype=float)
    return complaint, diagnosis


class PairwiseTests(unittest.TestCase):
    def setUp(self):
        self.training_df = make_training_df()
        complaint, diagnosis = make_embeddings(len(self.training_df))
        self.encoder = DummyEncoder()
        self.single_case_retriever = KTASSingleCaseRetriever(
            training_set=self.training_df,
            complaint_embeddings=complaint,
            diagnosis_embeddings=diagnosis,
            symptom_encoder=self.encoder,
        )
        self.pair_retriever = PairRetriever(
            training_set=self.training_df,
            complaint_embeddings=complaint,
            diagnosis_embeddings=diagnosis,
            encoder=self.encoder,
            examples_per_side=6,
        )

    def test_pair_label_derivation(self):
        self.assertEqual(derive_pair_label(2, 4), "A_BEFORE_B")
        self.assertEqual(derive_pair_label(5, 3), "B_BEFORE_A")
        self.assertIsNone(derive_pair_label(4, 4))

    def test_pairwise_retriever_uses_single_case_top_k_logic(self):
        patient = pd.Series(
            {
                "Sex": "Male",
                "Age": 80,
                "Arrival mode": "119 Ambulance",
                "Injury": "No",
                "Chief_complain": "chest pain",
                "Mental": "Alert",
                "Pain": 1,
                "NRS_pain": 8.0,
                "SBP": 92.0,
                "DBP": 60.0,
                "HR": 112.0,
                "RR": 22.0,
                "BT": 37.1,
                "Diagnosis in ED": "acute coronary syndrome",
                "KTAS_expert": 2,
            }
        )
        expected = self.single_case_retriever.retrieve_top_k_examples(patient, 3)
        actual = self.pair_retriever.retrieve_single_case_neighbors(patient, 3)
        self.assertListEqual(list(expected.index), list(actual.index))

    def test_cross_pair_examples_build_cartesian_product(self):
        a_neighbors = self.training_df.iloc[[0, 3]].copy()
        b_neighbors = self.training_df.iloc[[1, 2]].copy()
        examples = build_cross_pair_examples(a_neighbors, b_neighbors)
        self.assertEqual(len(examples), 4)
        self.assertEqual(examples[0].patient_a["Chief_complain"], "chest pain")
        self.assertEqual(examples[0].patient_b["Chief_complain"], "ankle injury")
        self.assertEqual(examples[0].decision, "A_BEFORE_B")
        self.assertEqual(examples[1].patient_b["Chief_complain"], "sore throat")

    def test_cross_pair_examples_exclude_equal_ktas(self):
        a_neighbors = self.training_df.iloc[[3]].copy()
        b_neighbors = self.training_df.iloc[[5]].copy()
        examples = build_cross_pair_examples(a_neighbors, b_neighbors)
        self.assertEqual(len(examples), 0)

    def test_pair_retrieve_returns_valid_cross_pairs(self):
        query_row = pd.Series(
            {
                "A_Sex": "Male",
                "A_Age": 80,
                "A_Arrival mode": "119 Ambulance",
                "A_Injury": "No",
                "A_Chief_complain": "chest pain",
                "A_Mental": "Alert",
                "A_Pain": 1,
                "A_NRS_pain": 8.0,
                "A_SBP": 92.0,
                "A_DBP": 60.0,
                "A_HR": 112.0,
                "A_RR": 22.0,
                "A_BT": 37.1,
                "A_Diagnosis in ED": "acute coronary syndrome",
                "A_KTAS_expert": 2,
                "B_Sex": "Female",
                "B_Age": 45,
                "B_Arrival mode": "Private Vehicle",
                "B_Injury": "Yes",
                "B_Chief_complain": "ankle injury",
                "B_Mental": "Alert",
                "B_Pain": 1,
                "B_NRS_pain": 4.0,
                "B_SBP": 132.0,
                "B_DBP": 78.0,
                "B_HR": 88.0,
                "B_RR": 18.0,
                "B_BT": 36.7,
                "B_Diagnosis in ED": "ankle sprain",
                "B_KTAS_expert": 4,
            }
        )
        patient_a, patient_b, examples = self.pair_retriever.retrieve(query_row, 6)
        self.assertEqual(patient_a["Chief_complain"], "chest pain")
        self.assertEqual(patient_b["Chief_complain"], "ankle injury")
        self.assertTrue(examples)
        self.assertLessEqual(len(examples), 36)
        self.assertTrue(all(example.decision in {"A_BEFORE_B", "B_BEFORE_A"} for example in examples))
        self.assertTrue(
            all(example.patient_a["KTAS_expert"] != example.patient_b["KTAS_expert"] for example in examples)
        )

    def test_prompt_builder_uses_binary_decisions(self):
        builder = PairPromptBuilder()
        query_row = pd.Series(
            {
                "A_Sex": "Male",
                "A_Age": 81,
                "A_Arrival mode": "119 Ambulance",
                "A_Injury": "No",
                "A_Chief_complain": "chest pain",
                "A_Mental": "Alert",
                "A_Pain": 1,
                "A_NRS_pain": 8.0,
                "A_SBP": 90.0,
                "A_DBP": 60.0,
                "A_HR": 115.0,
                "A_RR": 24.0,
                "A_BT": 37.2,
                "A_Diagnosis in ED": "acute coronary syndrome",
                "B_Sex": "Female",
                "B_Age": 45,
                "B_Arrival mode": "Private Vehicle",
                "B_Injury": "Yes",
                "B_Chief_complain": "ankle injury",
                "B_Mental": "Alert",
                "B_Pain": 1,
                "B_NRS_pain": 4.0,
                "B_SBP": 132.0,
                "B_DBP": 78.0,
                "B_HR": 88.0,
                "B_RR": 18.0,
                "B_BT": 36.7,
                "B_Diagnosis in ED": "ankle sprain",
            }
        )
        patient_a, patient_b, examples = self.pair_retriever.retrieve(query_row, 2)
        prompt = builder.build_prompt(
            patient_a=patient_a,
            patient_b=patient_b,
            examples=examples,
            strategy="KATE-PAIRWISE",
            serialization="natural",
        )
        self.assertIn("Example 1", prompt["user"])
        self.assertIn("Current comparison", prompt["user"])
        self.assertIn("A_BEFORE_B", prompt["user"])
        self.assertIn("B_BEFORE_A", prompt["user"])
        self.assertNotIn("TIE", prompt["user"])

    def test_output_parser_normalizes_and_rejects(self):
        self.assertEqual(normalize_pairwise_decision("A_BEFORE_B"), "A_BEFORE_B")
        self.assertEqual(normalize_pairwise_decision("Patient B before Patient A"), "B_BEFORE_A")
        with self.assertRaises(ValueError):
            normalize_pairwise_decision("TIE")


if __name__ == "__main__":
    unittest.main()
