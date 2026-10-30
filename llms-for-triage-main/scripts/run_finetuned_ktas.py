#!/usr/bin/env python3
"""Run inference on saved finetuned KTAS artifacts using the existing finetuning eval logic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(REPO_ROOT))

import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from peft import PeftModel

import utils.prompts as prompts
from finetuning_utils import REPO_ROOT, run_finetuned_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KTAS inference from a saved finetuned artifact.")
    parser.add_argument("--artifact_path", required=True, help="Path to saved finetuned artifact directory.")
    parser.add_argument("--base_model_name", help="Base model name, only needed when metadata is unavailable.")
    parser.add_argument("--start", type=int, default=0, help="Start index for KTAS test set.")
    parser.add_argument("--end", type=int, default=578, help="End index for KTAS test set.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results" / "Triage-KTAS"), help="Directory for predictions and metrics.")
    parser.add_argument("--max_new_tokens", type=int, default=75, help="Maximum generated tokens.")
    return parser.parse_args()


def load_metadata(artifact_dir: Path) -> dict:
    metadata_path = artifact_dir / "run_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def build_test_dataset(*, eos_token: str, start: int, end: int):
    test_df = Dataset.from_pandas(
        pd.read_csv(REPO_ROOT / "data/KTAS/test.csv").iloc[start : end + 1]
    )
    test_dataset = test_df.map(lambda x: prompts.format_instruction_prompt_for_finetuning(x, eos_token, dataset="triage-ktas", split="test"))
    return test_dataset


def main() -> int:
    args = parse_args()
    artifact_dir = Path(args.artifact_path).resolve()
    metadata = load_metadata(artifact_dir)
    base_model_name = metadata.get("base_model_name") or args.base_model_name
    if not base_model_name:
        raise ValueError("A base model name is required. Provide --base_model_name or a saved run_metadata.json.")

    if metadata.get("peft", False):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=1024,
            load_in_8bit=False,
            full_finetuning=False,
        )
        model = PeftModel.from_pretrained(model, str(artifact_dir))
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(artifact_dir),
            max_seq_length=1024,
            load_in_8bit=False,
            full_finetuning=True,
        )

    FastLanguageModel.for_inference(model)
    test_dataset = build_test_dataset(eos_token=tokenizer.eos_token, start=args.start, end=args.end)

    artifact_name = metadata.get("filename_with_timestamp", artifact_dir.name)
    output_prefix = Path(args.output_dir) / f"{artifact_name}_finetuned_eval_{args.start}_{args.end}"
    metrics, _ = run_finetuned_evaluation(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        true_acuity_col="KTAS_expert",
        output_prefix=str(output_prefix),
        max_new_tokens=args.max_new_tokens,
    )

    print("Overall Metrics:", metrics)
    print(f"Saved predictions and metrics to {output_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
