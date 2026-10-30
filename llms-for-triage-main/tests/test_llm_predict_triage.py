import importlib.util
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm-predict-triage.py"
SPEC = importlib.util.spec_from_file_location("llm_predict_triage", MODULE_PATH)
llm_predict_triage = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = llm_predict_triage
SPEC.loader.exec_module(llm_predict_triage)


class StubPredictor:
    def __init__(self, prompt, response, metadata):
        self.prompt = prompt
        self.response = response
        self.metadata = metadata

    def predict(self, **kwargs):
        return self.prompt, self.response, self.metadata


class PredictRecordTests(unittest.TestCase):
    def test_resolve_serialization_defaults_ktas_kate_to_commas(self):
        args = SimpleNamespace(dataset="Triage-KTAS", strategy="KATE", serialization=None)
        self.assertEqual(llm_predict_triage.resolve_serialization(args), "commas")

    def test_resolve_serialization_keeps_natural_default_for_other_runs(self):
        args = SimpleNamespace(dataset="Triage-MIMIC", strategy="Vanilla", serialization=None)
        self.assertEqual(llm_predict_triage.resolve_serialization(args), "natural")

    def test_resolve_serialization_preserves_explicit_choice(self):
        args = SimpleNamespace(dataset="Triage-KTAS", strategy="KATE", serialization="natural")
        self.assertEqual(llm_predict_triage.resolve_serialization(args), "natural")

    def test_json_prediction_record_includes_retrieved_examples_metadata(self):
        row = pd.Series({"Chief_complain": "right ocular pain", "KTAS_expert": 4})
        predictor = StubPredictor(
            prompt="prompt text",
            response='{"Acuity": 4}',
            metadata={"Retrieved_Examples": "example 1\n\nexample 2"},
        )

        result = llm_predict_triage.predict(
            index=0,
            row=row,
            predictor=predictor,
            model="test-model",
            strategy="KATE",
            return_json=True,
            k_shots=40,
            serialization="natural",
            vitals_off=False,
            bias=False,
            debug=False,
        )

        self.assertEqual(result["Estimated_Acuity"], 4)
        self.assertEqual(result["Retrieved_Examples"], "example 1\n\nexample 2")
        self.assertEqual(result["prompt"], "prompt text")


if __name__ == "__main__":
    unittest.main()
