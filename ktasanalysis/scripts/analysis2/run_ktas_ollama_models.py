#!/usr/bin/env python3
"""Run analysis2 KTAS prompt cases against local Ollama models."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("scripts/analysis2/output/ktas_prompt_cases.jsonl")
DEFAULT_OUTPUT_DIR = Path("scripts/analysis2/output/ollama")
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MAX_RETRIES = 3
DEFAULT_MODELS = [
    "mistral",
    "gemma2:9b",
    "llama3.1:8b",
    "medgemma:4b",
    "qwen2.5:7b",
    "qwen3:1.7b",
]

JSON_FIELDS = [
    "first_look",
    "chief_complaint",
    "chief_complaint_risk",
    "primary_modifiers",
    "secondary_modifiers",
    "decisive_factor",
    "predicted_ktas",
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
    "first_look",
    "chief_complaint",
    "chief_complaint_risk",
    "primary_modifiers",
    "secondary_modifiers",
    "decisive_factor",
    "raw_model_output",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis2 KTAS prompts against local Ollama models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Prompt case JSONL input (default: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Directory for per-model CSV outputs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Ollama model names to run.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based case offset to start from.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (default: 0.1)")
    parser.add_argument("--num-predict", type=int, default=500, help="Maximum tokens to generate (default: 500)")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout in seconds (default: 180)")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help=f"Ollama chat endpoint (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"Maximum attempts per case (default: {DEFAULT_MAX_RETRIES})")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip completed case_ids in existing output files.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional sleep between requests.")
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
    return {row.get("case_id") for row in read_csv_rows(path) if row.get("case_id")}


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

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})


def append_prediction(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: record.get(key) for key in CSV_FIELDNAMES})


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                for key in ("text", "content", "thinking"):
                    nested = item.get(key)
                    if isinstance(nested, str) and nested.strip():
                        parts.append(nested.strip())
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "response", "thinking"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def extract_response_text(response_json: dict[str, Any]) -> str:
    candidates = [
        response_json.get("message", {}).get("content"),
        response_json.get("message"),
        response_json.get("response"),
        response_json.get("content"),
    ]
    for candidate in candidates:
        text = _coerce_text(candidate)
        if text:
            return text
    raise ValueError(f"Ollama response missing readable content. Top-level keys: {sorted(response_json.keys())}")


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
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


def heuristic_prediction(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    ktas_match = re.search(r'"predicted_ktas"\s*:\s*"?([1-5])"?', cleaned)
    if not ktas_match:
        ktas_match = re.search(r"\bKTAS\b[^0-9]{0,10}([1-5])\b", cleaned, flags=re.IGNORECASE)
    if not ktas_match:
        ktas_match = re.search(r"\blevel\b[^0-9]{0,10}([1-5])\b", cleaned, flags=re.IGNORECASE)
    if not ktas_match:
        ktas_match = re.search(r"\b([1-5])\b", cleaned)
    if not ktas_match:
        return None

    return {
        "first_look": "",
        "chief_complaint": "",
        "chief_complaint_risk": "",
        "primary_modifiers": "",
        "secondary_modifiers": "",
        "decisive_factor": cleaned.splitlines()[0][:400] if cleaned else "",
        "predicted_ktas": int(ktas_match.group(1)),
    }


def call_ollama(
    *,
    url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_predict: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def format_ollama_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        if body:
            return f"HTTP {exc.code}: {body}"
        return f"HTTP {exc.code}: {exc.reason}"
    return str(exc)


def validate_prediction(case_id: str, raw_prediction: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in JSON_FIELDS if field not in raw_prediction]
    if missing:
        raise ValueError(f"Model output for {case_id} is missing required keys: {', '.join(missing)}")

    predicted_ktas = int(raw_prediction["predicted_ktas"])
    if predicted_ktas < 1 or predicted_ktas > 5:
        raise ValueError(f"`predicted_ktas` out of range for {case_id}: {predicted_ktas}")

    normalized: dict[str, Any] = {"predicted_ktas": predicted_ktas}
    for field in JSON_FIELDS:
        if field == "predicted_ktas":
            continue
        value = raw_prediction.get(field)
        normalized[field] = "" if value is None else str(value).strip()
    return normalized


def normalize_prediction(model: str, case: dict[str, Any], raw_prediction: dict[str, Any], raw_text: str, latency_seconds: float) -> dict[str, Any]:
    case_id = case["case_id"]
    structured = validate_prediction(case_id, raw_prediction)
    ktas_rn = int(case["KTAS_RN"])
    ktas_expert = int(case["KTAS_expert"])
    return {
        "model": model,
        "case_id": case_id,
        "predicted_ktas": structured["predicted_ktas"],
        "KTAS_RN": ktas_rn,
        "KTAS_expert": ktas_expert,
        "matches_ktas_rn": structured["predicted_ktas"] == ktas_rn,
        "matches_ktas_expert": structured["predicted_ktas"] == ktas_expert,
        "matches_either_reference": structured["predicted_ktas"] in {ktas_rn, ktas_expert},
        "latency_seconds": round(latency_seconds, 3),
        "first_look": structured["first_look"],
        "chief_complaint": structured["chief_complaint"],
        "chief_complaint_risk": structured["chief_complaint_risk"],
        "primary_modifiers": structured["primary_modifiers"],
        "secondary_modifiers": structured["secondary_modifiers"],
        "decisive_factor": structured["decisive_factor"],
        "raw_model_output": raw_text,
    }


def build_null_prediction(model: str, case: dict[str, Any], failure_message: str, latency_seconds: float) -> dict[str, Any]:
    return {
        "model": model,
        "case_id": case["case_id"],
        "predicted_ktas": "",
        "KTAS_RN": int(case["KTAS_RN"]),
        "KTAS_expert": int(case["KTAS_expert"]),
        "matches_ktas_rn": "",
        "matches_ktas_expert": "",
        "matches_either_reference": "",
        "latency_seconds": round(latency_seconds, 3),
        "first_look": "",
        "chief_complaint": "",
        "chief_complaint_risk": "",
        "primary_modifiers": "",
        "secondary_modifiers": "",
        "decisive_factor": "",
        "raw_model_output": failure_message,
    }


def run_model(args: argparse.Namespace, cases: list[dict[str, Any]], model_name: str) -> tuple[int, int, Path]:
    output_path = output_path_for_model(args.output_dir, model_name)
    ensure_output_schema(output_path)
    existing_case_ids = set()
    if output_path.exists() and not args.no_resume:
        existing_case_ids = read_existing_case_ids(output_path)

    processed = 0
    skipped = 0
    for case in cases:
        case_id = case["case_id"]
        if case_id in existing_case_ids:
            skipped += 1
            continue

        last_error = ""
        started_at = time.perf_counter()
        for attempt in range(1, args.max_retries + 1):
            try:
                response_json = call_ollama(
                    url=args.ollama_url,
                    model=model_name,
                    system_prompt=case["system_prompt"],
                    user_prompt=case["user_prompt"],
                    temperature=args.temperature,
                    num_predict=args.num_predict,
                    timeout=args.timeout,
                )
                latency_seconds = time.perf_counter() - started_at
                raw_text = extract_response_text(response_json)
                try:
                    parsed_output = extract_json_object(raw_text)
                except ValueError:
                    parsed_output = heuristic_prediction(raw_text)
                    if parsed_output is None:
                        raise
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
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = format_ollama_error(exc)
                print(f"[{model_name}] Attempt {attempt}/{args.max_retries} failed on {case_id}: {last_error}", file=sys.stderr)
                if attempt == args.max_retries:
                    latency_seconds = time.perf_counter() - started_at
                    append_prediction(
                        output_path,
                        build_null_prediction(model_name, case, f"null_after_{args.max_retries}_attempts: {last_error}", latency_seconds),
                    )
                    processed += 1
                    print(f"[{model_name}] Marked {case_id} as null after retries.", file=sys.stderr)
                else:
                    time.sleep(min(1.0, max(args.sleep_seconds, 0.25)))

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

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
        print(f"Running model: {model_name}")
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
