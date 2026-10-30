import unittest
from unittest.mock import patch

import pandas as pd

from utils.predictors import Predictor


class PredictorsSerializationTests(unittest.TestCase):
    @patch("utils.predictors.utils.query_llm_full", return_value='{"Acuity": 4}')
    def test_kate_predict_uses_requested_serialization(self, mock_query):
        predictor = Predictor.__new__(Predictor)
        predictor.dataset = "Triage-KTAS"
        predictor.strategy = "KATE"
        predictor.target = "KTAS_expert"
        predictor.hippa = False
        predictor.training_set = pd.DataFrame(
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
            ]
        )
        predictor._retrieve_top_k_examples = lambda row, k: predictor.training_set

        row = pd.Series(
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
            }
        )

        prompt, _, metadata = predictor.predict(
            row=row,
            model="test-model",
            k_shots=40,
            return_json=True,
            serialization_strategy="commas",
            vitals_off=False,
            bias=False,
            debug=False,
        )

        self.assertIn("age, sex, temperature, heartrate", prompt["user"])
        self.assertIn("age, sex, temperature, heartrate", metadata["Retrieved_Examples"])
        self.assertNotIn("A 44-year-old man arrives at the emergency department", metadata["Retrieved_Examples"])


if __name__ == "__main__":
    unittest.main()
