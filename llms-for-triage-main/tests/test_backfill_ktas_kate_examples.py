import unittest
from pathlib import Path

import pandas as pd

from scripts import backfill_ktas_kate_examples


class FakeRetriever:
    def retrieve_top_k_examples(self, row, k):
        return pd.DataFrame(
            [
                {
                    "Age": 44,
                    "Sex": "Male",
                    "Arrival mode": "Private Vehicle",
                    "Injury": "No",
                    "Chief_complain": "headache",
                    "Mental": "Alert",
                    "Pain": 1,
                    "NRS_pain": 4.0,
                    "SBP": 128.0,
                    "DBP": 78.0,
                    "HR": 82.0,
                    "RR": 18.0,
                    "BT": 36.8,
                    "Diagnosis in ED": "migraine",
                    "KTAS_expert": 4,
                }
                for _ in range(k)
            ]
        )


class BackfillKtAsKateExamplesTests(unittest.TestCase):
    def test_infer_k_shots_from_filename(self):
        path = Path(
            "results/Triage-KTAS/"
            "Triage-KTAS_KATE_azure-gpt-5.4_json_40_shots_natural_0_578_2026-07-08_11:21:01.csv"
        )
        self.assertEqual(backfill_ktas_kate_examples.infer_k_shots(path), 40)

    def test_infer_serialization_from_filename(self):
        path = Path(
            "results/Triage-KTAS/"
            "Triage-KTAS_KATE_model_json_40_shots_natural_full_0_578_2026-07-08_11:21:01.csv"
        )
        self.assertEqual(backfill_ktas_kate_examples.infer_serialization(path), "natural_full")

    def test_backfill_dataframe_adds_retrieved_examples_column(self):
        df = pd.DataFrame(
            [
                {
                    "Age": 71,
                    "Sex": "Male",
                    "Arrival mode": "Private Vehicle",
                    "Injury": "Yes",
                    "Chief_complain": "right ocular pain",
                    "Mental": "Alert",
                    "Pain": 1,
                    "NRS_pain": 2.0,
                    "SBP": 160.0,
                    "DBP": 100.0,
                    "HR": 84.0,
                    "RR": 18.0,
                    "BT": 36.6,
                    "Diagnosis in ED": "Corneal abrasion",
                    "KTAS_expert": 4,
                }
            ]
        )

        updated = backfill_ktas_kate_examples.backfill_dataframe(
            df,
            retriever=FakeRetriever(),
            k_shots=2,
            serialization="natural",
        )

        self.assertIn("Retrieved_Examples", updated.columns)
        self.assertIn("Acuity Level: 4", updated.loc[0, "Retrieved_Examples"])


if __name__ == "__main__":
    unittest.main()
