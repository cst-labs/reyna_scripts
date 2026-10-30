from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import pandas as pd
from datasets import Dataset
import datasets.arrow_dataset as arrow_dataset
import datasets.fingerprint as datasets_fingerprint
from tqdm import tqdm

import utils.prompts as prompts
import utils.utils as utils


REPO_ROOT = Path(__file__).resolve().parents[1]


def disable_problematic_dataset_fingerprinting():
    """Work around Python 3.14 dill/pickle incompatibility in datasets fingerprinting."""
    def _stable_fingerprint(*_args, **_kwargs):
        return uuid4().hex

    datasets_fingerprint.generate_fingerprint = _stable_fingerprint
    if hasattr(arrow_dataset, "generate_fingerprint"):
        arrow_dataset.generate_fingerprint = _stable_fingerprint


disable_problematic_dataset_fingerprinting()


def build_run_name(*, model_name: str, dataset: str, num_train_epochs: int, finetuning_learning_rate: float, cpt_num_train_epochs: int, para: int, peft: bool) -> str:
    filename_parts = [
        model_name.split("/")[-1],
        dataset,
        f"epochs{num_train_epochs}",
        f"lr{finetuning_learning_rate}",
    ]
    if cpt_num_train_epochs:
        filename_parts.append(f"cpt{cpt_num_train_epochs}")
        if para > 1:
            filename_parts.append(f"para{para}")
    if peft:
        filename_parts.append("peft")
    return "_".join(filename_parts)


def get_dataset_frames(dataset: str):
    if dataset == "handbook":
        train_df = pd.read_csv(REPO_ROOT / "data/ESI-Handbook/train.csv")
        test_df = pd.read_csv(REPO_ROOT / "data/ESI-Handbook/test.csv")
        prompt_dataset = "triage-handbook"
        target_col = "acuity"
    elif dataset == "ktas":
        train_df = pd.read_csv(REPO_ROOT / "data/KTAS/train.csv")
        test_df = pd.read_csv(REPO_ROOT / "data/KTAS/test.csv")
        prompt_dataset = "triage-ktas"
        target_col = "KTAS_expert"
    elif dataset == "mimic":
        train_df = pd.read_csv(REPO_ROOT / "data/mimic-iv-private/anchor_year_group_datasets/2014_-_2016/small_train_dataset.csv")
        test_df = pd.read_csv(REPO_ROOT / "data/mimic-iv-private/anchor_year_group_datasets/2017_-_2019/test_dataset.csv")
        prompt_dataset = "triage-mimic"
        target_col = "acuity"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train_df, test_df, prompt_dataset, target_col


def build_finetuning_datasets(*, dataset: str, eos_token: str):
    train_df, test_df, prompt_dataset, target_col = get_dataset_frames(dataset)
    train_records = [
        {
            "text": prompts.format_instruction_prompt_for_finetuning(row, eos_token, dataset=prompt_dataset)["text"],
            target_col: int(row[target_col]),
        }
        for _, row in train_df.iterrows()
    ]
    test_records = [
        {
            "text": prompts.format_instruction_prompt_for_finetuning(row, eos_token, dataset=prompt_dataset, split="test")["text"],
            target_col: int(row[target_col]),
        }
        for _, row in test_df.iterrows()
    ]
    train_dataset = Dataset.from_list(train_records)
    test_dataset = Dataset.from_list(test_records)
    return train_df, test_df, train_dataset, test_dataset, target_col


def extract_response(text: str) -> int:
    match = re.search(r"Response:.*?(\d+)", text, re.DOTALL)
    if match:
        response = int(match.group(1))
        if 1 <= response <= 5:
            return response
    return -1


def generate_response(*, model, tokenizer, input_text: str, max_new_tokens: int = 75, device: str = "cuda") -> tuple[int, str]:
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return extract_response(decoded_output), decoded_output


def run_finetuned_evaluation(*, model, tokenizer, test_dataset, true_acuity_col: str, output_prefix: str | None = None, max_new_tokens: int = 75, device: str = "cuda"):
    y_true = []
    y_pred = []
    prediction_rows = []
    overtriage = 0
    undertriage = 0
    outside_by_2 = 0

    for i, sample in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        input_text = sample["text"]
        true_acuity = int(sample[true_acuity_col])
        predicted_acuity, raw_output = generate_response(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        y_true.append(true_acuity)
        y_pred.append(int(predicted_acuity))

        if predicted_acuity != true_acuity:
            if predicted_acuity < true_acuity:
                undertriage += 1
            elif predicted_acuity > true_acuity:
                overtriage += 1
            if abs(predicted_acuity - true_acuity) >= 2:
                outside_by_2 += 1

        prediction_rows.append({
            "index": i,
            "Estimated_Acuity": int(predicted_acuity),
            "True_Acuity": true_acuity,
            "Prompt": input_text,
            "Raw_Output": raw_output,
        })

    metrics = utils.evaluate_predictions(y_pred, y_true, ordinal=True, by_class=True)

    total_predictions = len(y_pred)
    overtriage_rate = overtriage / total_predictions if total_predictions else 0.0
    undertriage_rate = undertriage / total_predictions if total_predictions else 0.0
    outside_by_2_rate = outside_by_2 / total_predictions if total_predictions else 0.0

    metrics["overall"]["overtriage_count"] = overtriage
    metrics["overall"]["undertriage_count"] = undertriage
    metrics["overall"]["overtriage_rate"] = overtriage_rate
    metrics["overall"]["undertriage_rate"] = undertriage_rate
    metrics["overall"]["outside_by_2_count"] = outside_by_2
    metrics["overall"]["outside_by_2_rate"] = outside_by_2_rate

    predictions_df = pd.DataFrame(prediction_rows)
    if output_prefix:
        output_path = Path(output_prefix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(f"{output_prefix}.csv", index=False)
        utils.save_metrics(metrics, output_prefix)

    return metrics, predictions_df
