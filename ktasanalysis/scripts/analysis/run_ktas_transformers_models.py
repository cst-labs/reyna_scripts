#!/usr/bin/env python3
"""Run KTAS prompt cases against local Transformers models.

Usage:
  python3 scripts/analysis/run_ktas_transformers_models.py
  python3 scripts/analysis/run_ktas_transformers_models.py --limit 10
  python3 scripts/analysis/run_ktas_transformers_models.py --models mistralai/Ministral-8B-Instruct-2410 WeiboAI/VibeThinker-3B
  python3 scripts/analysis/run_ktas_transformers_models.py --no-resume
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_INPUT = Path("scripts/analysis/output/ktas_prompt_cases.jsonl")
DEFAULT_OUTPUT_DIR = Path("scripts/analysis/output/transformers")
DEFAULT_MODELS = [
    "WeiboAI/VibeThinker-3B",
]

CSV_FIELDNAMES = [
    "model",
    "case_id",
    "predicted_ktas",
    "KTAS_RN",
    "KTAS_expert",
    "matches_ktas_rn",
    "matches_ktas_expert",
    "matches_either_reference",
    "latency_seconds",
    "reasoning_summary",
    "raw_model_output",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KTAS prompts against local Transformers models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Prompt case JSONL input (default: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Directory for per-model CSV outputs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Transformers model identifiers to run.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based case offset to start from.")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="Maximum tokens to generate (default: 120)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy decoding (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling when temperature > 0 (default: 0.95)")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse existing per-model output caches; start writing rows without skipping completed case_ids.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model/tokenizer code from the model repository if required.")
    parser.add_argument("--attn-implementation", default=None, help="Optional attention implementation override, e.g. flash_attention_2.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto", help="Device to run on (default: auto)")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        row.get("case_id")
        for row in read_csv_rows(path)
        if row.get("case_id")
    }


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def output_path_for_model(output_dir: Path, model_name: str) -> Path:
    return output_dir / f"{slugify_model_name(model_name)}_predictions.csv"


def ensure_output_schema(path: Path) -> None:
    if not path.exists():
        return
    rows = read_csv_rows(path)
    existing_fieldnames = list(rows[0].keys()) if rows else []
    if existing_fieldnames == CSV_FIELDNAMES:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            normalized_row = {key: row.get(key, "") for key in CSV_FIELDNAMES}
            writer.writerow(normalized_row)


def append_prediction(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: record.get(key) for key in CSV_FIELDNAMES})


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = sanitize_raw_output(text)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Unable to locate JSON object in model output: {text[:300]}")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Parsed model output is not a JSON object.")
    return parsed


def sanitize_raw_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"<\|assistant\|>", "", cleaned)
    cleaned = re.sub(r"<\|end\|>", "", cleaned)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?thinking>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def heuristic_prediction(text: str) -> dict[str, Any] | None:
    cleaned = sanitize_raw_output(text)
    ktas_match = re.search(r'"predicted_ktas"\s*:\s*"?([1-5])"?', cleaned)
    if not ktas_match:
        ktas_match = re.search(r"\bKTAS\b[^0-9]{0,10}([1-5])\b", cleaned, flags=re.IGNORECASE)
    if not ktas_match:
        ktas_match = re.search(r"\b([1-5])\b", cleaned)
    if not ktas_match:
        return None

    reasoning_match = re.search(r'"reasoning_summary"\s*:\s*"([^"]+)"', cleaned)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else cleaned.splitlines()[0].strip()
    return {
        "predicted_ktas": int(ktas_match.group(1)),
        "reasoning_summary": reasoning[:400] if reasoning else "",
    }


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return detect_device()


def resolve_torch_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(model_name: str, trust_remote_code: bool, attn_implementation: str | None, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "dtype": resolve_torch_dtype(device),
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(torch.device(device))
    model.eval()
    return tokenizer, model


def build_generation_inputs(tokenizer, system_prompt: str, user_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
    return tokenizer(prompt, return_tensors="pt")


def generate_response_text(tokenizer, model, system_prompt: str, user_prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    generation_inputs = build_generation_inputs(tokenizer, system_prompt, user_prompt)
    if isinstance(generation_inputs, torch.Tensor):
        input_ids = generation_inputs.to(model.device)
        attention_mask = torch.ones_like(input_ids)
    else:
        generation_inputs = generation_inputs.to(model.device)
        input_ids = generation_inputs["input_ids"]
        attention_mask = generation_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": False,
    }
    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)

    generated_ids = output_ids[0][input_ids.shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not text:
        raise ValueError("Model produced an empty response.")
    return text


def normalize_prediction(model_name: str, case: dict[str, Any], raw_prediction: dict[str, Any], raw_text: str, latency_seconds: float) -> dict[str, Any]:
    case_id = case["case_id"]
    if "predicted_ktas" not in raw_prediction:
        raise ValueError(f"Model output for {case_id} from {model_name} is missing `predicted_ktas`.")
    predicted_ktas = int(raw_prediction["predicted_ktas"])
    if predicted_ktas < 1 or predicted_ktas > 5:
        raise ValueError(f"`predicted_ktas` out of range for {case_id} from {model_name}: {predicted_ktas}")

    ktas_rn = int(case["KTAS_RN"])
    ktas_expert = int(case["KTAS_expert"])
    return {
        "model": model_name,
        "case_id": case_id,
        "predicted_ktas": predicted_ktas,
        "KTAS_RN": ktas_rn,
        "KTAS_expert": ktas_expert,
        "matches_ktas_rn": predicted_ktas == ktas_rn,
        "matches_ktas_expert": predicted_ktas == ktas_expert,
        "matches_either_reference": predicted_ktas in {ktas_rn, ktas_expert},
        "latency_seconds": round(latency_seconds, 3),
        "reasoning_summary": raw_prediction.get("reasoning_summary"),
        "raw_model_output": raw_text,
    }


def run_model(args: argparse.Namespace, cases: list[dict[str, Any]], model_name: str) -> tuple[int, int, Path]:
    output_path = output_path_for_model(args.output_dir, model_name)
    ensure_output_schema(output_path)
    existing_case_ids = set()
    if output_path.exists() and not args.no_resume:
        existing_case_ids = read_existing_case_ids(output_path)

    device = resolve_device(args.device)
    print(f"Loading model: {model_name} on {device}")
    tokenizer, model = load_model_and_tokenizer(
        model_name,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        device=device,
    )

    processed = 0
    skipped = 0
    for case in cases:
        case_id = case["case_id"]
        if case_id in existing_case_ids:
            skipped += 1
            continue

        try:
            started_at = time.perf_counter()
            raw_text = generate_response_text(
                tokenizer,
                model,
                system_prompt=case["system_prompt"],
                user_prompt=case["user_prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            latency_seconds = time.perf_counter() - started_at
            try:
                parsed_output = extract_json_object(raw_text)
            except (ValueError, json.JSONDecodeError):
                parsed_output = heuristic_prediction(raw_text)
                if parsed_output is None:
                    raise ValueError(f"Unable to parse model output: {sanitize_raw_output(raw_text)[:500]}")
            normalized = normalize_prediction(model_name, case, parsed_output, raw_text, latency_seconds)
            append_prediction(output_path, normalized)
            processed += 1
            print(
                f"[{model_name}] Processed {case_id}: KTAS {normalized['predicted_ktas']} "
                f"(RN match={normalized['matches_ktas_rn']}, "
                f"expert match={normalized['matches_ktas_expert']}, "
                f"either match={normalized['matches_either_reference']}, "
                f"latency={normalized['latency_seconds']}s)"
            )
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            print(f"[{model_name}] Failed on {case_id}: {exc}", file=sys.stderr)
            return processed, skipped, output_path

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return processed, skipped, output_path


def main() -> int:
    args = parse_args()
    cases = read_jsonl(args.input)
    selected_cases = cases[args.start_index :]
    if args.limit is not None:
        selected_cases = selected_cases[: args.limit]

    if not selected_cases:
        print("No cases selected for processing.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_status = 0
    for model_name in args.models:
        processed, skipped, output_path = run_model(args, selected_cases, model_name)
        print(f"[{model_name}] Completed model runs: {processed}")
        print(f"[{model_name}] Skipped existing cases: {skipped}")
        print(f"[{model_name}] Resume cache active: {not args.no_resume}")
        print(f"[{model_name}] Output path: {output_path}")
        if processed + skipped < len(selected_cases):
            overall_status = 1

    return overall_status


if __name__ == "__main__":
    raise SystemExit(main())
