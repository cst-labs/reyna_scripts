from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import utils.predictors as predictors
import utils.prompts as prompts
import utils.utils as utils


PAIRWISE_KTAS_PROMPTS = {
    "KATE_PAIRWISE": {
        "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
        "user": (
            "You are comparing two emergency patients for queue priority under KTAS. "
            "Use the historical comparison examples to decide which patient should be prioritized first."
        ),
    },
    "KATE_PAIRWISE_COT": {
        "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
        "user": (
            "You are comparing two emergency patients for queue priority under KTAS. "
            "Use the historical comparison examples to reason carefully about which patient should be prioritized first."
        ),
    },
}

KTAS_PATIENT_COLUMNS = [
    "Sex",
    "Age",
    "Arrival mode",
    "Injury",
    "Chief_complain",
    "Mental",
    "Pain",
    "NRS_pain",
    "SBP",
    "DBP",
    "HR",
    "RR",
    "BT",
    "Diagnosis in ED",
    "KTAS_expert",
]

DEFAULT_PAIRWISE_STRATEGIES = {"KATE-PAIRWISE", "KATE-PAIRWISE-COT"}


@dataclass(frozen=True)
class PairExample:
    patient_a: pd.Series
    patient_b: pd.Series
    decision: str
    reason: Optional[str] = None
    patient_a_id: Optional[int] = None
    patient_b_id: Optional[int] = None


def is_pairwise_strategy(strategy: str) -> bool:
    return strategy in DEFAULT_PAIRWISE_STRATEGIES


def pairwise_prompt_strategy(strategy: str) -> str:
    if strategy == "KATE-PAIRWISE-COT":
        return "KATE_PAIRWISE_COT"
    return "KATE_PAIRWISE"


def derive_pair_label(ktas_a: int, ktas_b: int) -> Optional[str]:
    if ktas_a < ktas_b:
        return "A_BEFORE_B"
    if ktas_b < ktas_a:
        return "B_BEFORE_A"
    return None


def normalize_pairwise_decision(text: str) -> str:
    if text is None:
        raise ValueError("Pairwise response is empty.")
    cleaned = text.strip().upper()
    if cleaned in {"A_BEFORE_B", "B_BEFORE_A"}:
        return cleaned

    first_line = cleaned.splitlines()[0].strip() if cleaned else ""
    if first_line in {"A_BEFORE_B", "B_BEFORE_A"}:
        return first_line

    token_match = re.search(r"\b(A_BEFORE_B|B_BEFORE_A)\b", cleaned)
    if token_match:
        return token_match.group(1)

    compact = re.sub(r"[^A-Z_]", "", cleaned.replace(" ", "_"))
    if compact in {"A_BEFORE_B", "B_BEFORE_A"}:
        return compact

    if re.search(r"\bA\b.*\bBEFORE\b.*\bB\b", cleaned):
        return "A_BEFORE_B"
    if re.search(r"\bB\b.*\bBEFORE\b.*\bA\b", cleaned):
        return "B_BEFORE_A"

    raise ValueError(f"Invalid pairwise decision: {text}")


def extract_patient_row(row: pd.Series, prefix: str) -> pd.Series:
    data = {}
    for column in KTAS_PATIENT_COLUMNS:
        key = f"{prefix}_{column}"
        if key in row:
            data[column] = row[key]
    patient = pd.Series(data)
    if patient.empty:
        raise ValueError("Pairwise input rows must contain both A_... and B_... KTAS columns.")
    return patient


def build_cross_pair_examples(
    a_neighbors: pd.DataFrame,
    b_neighbors: pd.DataFrame,
) -> List[PairExample]:
    examples: List[PairExample] = []
    a_neighbors = a_neighbors.reset_index()
    b_neighbors = b_neighbors.reset_index()

    for _, a_row in a_neighbors.iterrows():
        patient_a_id = int(a_row["index"])
        patient_a = a_row.drop(labels=["index"])
        for _, b_row in b_neighbors.iterrows():
            patient_b_id = int(b_row["index"])
            patient_b = b_row.drop(labels=["index"])
            decision = derive_pair_label(
                int(patient_a["KTAS_expert"]),
                int(patient_b["KTAS_expert"]),
            )
            if decision is None:
                continue
            examples.append(
                PairExample(
                    patient_a=patient_a,
                    patient_b=patient_b,
                    decision=decision,
                    patient_a_id=patient_a_id,
                    patient_b_id=patient_b_id,
                )
            )
    return examples


class PairPromptBuilder:
    def __init__(self, dataset: str = "triage-ktas") -> None:
        self.dataset = dataset

    def _format_patient(self, row: pd.Series, serialization: str) -> str:
        return prompts.format_row(row, dataset=self.dataset, serialization=serialization)

    def format_example(self, example: PairExample, index: int, serialization: str) -> str:
        decision_text = (
            "Patient A should be prioritized before Patient B."
            if example.decision == "A_BEFORE_B"
            else "Patient B should be prioritized before Patient A."
        )
        parts = [
            f"Example {index}",
            "",
            "Patient A:",
            self._format_patient(example.patient_a, serialization),
            "",
            "Patient B:",
            self._format_patient(example.patient_b, serialization),
            "",
            f"Decision: {decision_text}",
        ]
        if example.reason:
            parts.append(f"Reason: {example.reason}")
        return "\n".join(parts)

    def build_prompt(
        self,
        *,
        patient_a: pd.Series,
        patient_b: pd.Series,
        examples: Sequence[PairExample],
        strategy: str,
        serialization: str,
    ) -> Dict[str, str]:
        prompt_template = PAIRWISE_KTAS_PROMPTS[pairwise_prompt_strategy(strategy)].copy()
        example_blocks = "\n\n--------------------------------\n\n".join(
            self.format_example(example, idx + 1, serialization)
            for idx, example in enumerate(examples)
        )
        user = (
            f"{prompt_template['user']}\n\n"
            f"{example_blocks}\n\n"
            "Current comparison\n\n"
            "Patient A:\n"
            f"{self._format_patient(patient_a, serialization)}\n\n"
            "Patient B:\n"
            f"{self._format_patient(patient_b, serialization)}\n\n"
            "Decide which patient should be prioritized first.\n\n"
            "Respond with exactly one:\n"
            "A_BEFORE_B\n"
            "B_BEFORE_A"
        )
        return {"system": prompt_template["system"], "user": user}


class PairRetriever:
    def __init__(
        self,
        *,
        training_set: pd.DataFrame,
        complaint_embeddings: np.ndarray,
        diagnosis_embeddings: np.ndarray,
        encoder,
        dataset: str = "Triage-KTAS",
        examples_per_side: int = 6,
    ) -> None:
        self.dataset = dataset
        self.training_set = training_set.reset_index(drop=True).copy()
        self.examples_per_side = examples_per_side
        self.single_case_retriever = predictors.KTASSingleCaseRetriever(
            training_set=self.training_set,
            complaint_embeddings=complaint_embeddings,
            diagnosis_embeddings=diagnosis_embeddings,
            symptom_encoder=encoder,
        )

    def retrieve_single_case_neighbors(self, patient_row: pd.Series, k: Optional[int] = None) -> pd.DataFrame:
        final_k = k if k is not None else self.examples_per_side
        return self.single_case_retriever.retrieve_top_k_examples(patient_row, final_k)

    def retrieve(self, row: pd.Series, k: Optional[int] = None) -> Tuple[pd.Series, pd.Series, List[PairExample]]:
        patient_a = extract_patient_row(row, "A")
        patient_b = extract_patient_row(row, "B")
        final_k = k if k is not None else self.examples_per_side
        a_neighbors = self.retrieve_single_case_neighbors(patient_a, final_k)
        b_neighbors = self.retrieve_single_case_neighbors(patient_b, final_k)
        return patient_a, patient_b, build_cross_pair_examples(a_neighbors, b_neighbors)


class PairPredictor:
    def __init__(
        self,
        *,
        retriever: PairRetriever,
        strategy: str = "KATE-PAIRWISE",
        hippa: bool = False,
        prompt_builder: Optional[PairPromptBuilder] = None,
    ) -> None:
        self.retriever = retriever
        self.strategy = strategy
        self.hippa = hippa
        self.prompt_builder = prompt_builder or PairPromptBuilder()

    def predict(
        self,
        *,
        row: pd.Series,
        model: str,
        k_shots: int,
        serialization_strategy: str,
        debug: bool = False,
    ) -> Tuple[Dict[str, str], str]:
        patient_a, patient_b, examples = self.retriever.retrieve(row, k_shots)
        prompt = self.prompt_builder.build_prompt(
            patient_a=patient_a,
            patient_b=patient_b,
            examples=examples,
            strategy=self.strategy,
            serialization=serialization_strategy,
        )
        if self.hippa:
            response = utils.query_gpt_safe(
                prompt,
                model=model,
                return_json=False,
                debug=debug,
                is_prompt_full=True,
        )
        else:
            response = utils.query_llm_full(prompt, model=model, return_json=False, debug=debug)
        return prompt, response
