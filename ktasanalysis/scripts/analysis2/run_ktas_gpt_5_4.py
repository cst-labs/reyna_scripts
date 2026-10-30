#!/usr/bin/env python3
"""Run analysis2 KTAS prompt cases against a GPT-5.4-compatible endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("scripts/analysis2/output/ktas_prompt_cases.jsonl")
DEFAULT_OUTPUT = Path("scripts/analysis2/output/azure_gpt_5_4_predictions.csv")
DEFAULT_MODEL = "azure-gpt-5.4"
DEFAULT_ENV_PATH = Path(".env")

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
    parser = argparse.ArgumentParser(description="Run analysis2 KTAS prompts against a GPT-5.4-compatible endpoint.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Prompt case JSONL input (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Predictions CSV output (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name to send to the endpoint (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based case offset to start from.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (default: 0.1)")
    parser.add_argument("--max-output-tokens", type=int, default=500, help="Max completion tokens (default: 500)")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds (default: 120)")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip completed case_ids from existing output.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional sleep between requests.")
    parser.add_argument("--ca-bundle", type=Path, default=None, help="Path to a custom CA bundle PEM file.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification.")
    return parser.parse_args()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_base_url(raw_base_url: str) -> str:
    base = raw_base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/chat/completions"


def env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


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


def extract_response_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        raise ValueError("Model response does not contain choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        text = "".join(parts).strip()
        if text:
            return text
    raise ValueError("Model response does not contain readable message content.")


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


def call_model(
    *,
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    ssl_context: ssl.SSLContext | None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def build_ssl_context(*, ca_bundle: Path | None, insecure: bool) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()
    if ca_bundle is not None:
        return ssl.create_default_context(cafile=str(ca_bundle))
    return ssl.create_default_context()


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


def main() -> int:
    args = parse_args()
    load_env(DEFAULT_ENV_PATH)

    base_url = os.environ.get("LITELLM_BASE_URL")
    api_key = os.environ.get("LITELLM")
    if not base_url or not api_key:
        print("Missing LITELLM_BASE_URL or LITELLM in environment or .env.", file=sys.stderr)
        return 1

    ca_bundle = args.ca_bundle
    if ca_bundle is None:
        raw_bundle = os.environ.get("LITELLM_CA_BUNDLE", "").strip()
        if raw_bundle:
            ca_bundle = Path(raw_bundle)

    insecure = args.insecure or env_flag("LITELLM_INSECURE")
    if ca_bundle is not None and not ca_bundle.exists():
        print(f"CA bundle file not found: {ca_bundle}", file=sys.stderr)
        return 1
    if insecure:
        print("Warning: TLS certificate verification is disabled for this run.", file=sys.stderr)

    ssl_context = build_ssl_context(ca_bundle=ca_bundle, insecure=insecure)
    endpoint_url = normalize_base_url(base_url)

    ensure_output_schema(args.output)
    cases = read_jsonl(args.input)
    selected_cases = cases[args.start_index :]
    if args.limit is not None:
        selected_cases = selected_cases[: args.limit]

    existing_case_ids = set()
    if args.output.exists() and not args.no_resume:
        existing_case_ids = read_existing_case_ids(args.output)

    processed = 0
    skipped = 0
    for case in selected_cases:
        case_id = case["case_id"]
        if case_id in existing_case_ids:
            skipped += 1
            continue

        try:
            started_at = time.perf_counter()
            response_json = call_model(
                url=endpoint_url,
                api_key=api_key,
                model=args.model,
                system_prompt=case["system_prompt"],
                user_prompt=case["user_prompt"],
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                ssl_context=ssl_context,
            )
            latency_seconds = round(time.perf_counter() - started_at, 3)
            raw_text = extract_response_text(response_json)
            parsed_output = extract_json_object(raw_text)
            structured = validate_prediction(case_id, parsed_output)

            ktas_rn = int(case["KTAS_RN"])
            ktas_expert = int(case["KTAS_expert"])
            append_prediction(
                args.output,
                {
                    "case_id": case_id,
                    "predicted_ktas": structured["predicted_ktas"],
                    "KTAS_RN": ktas_rn,
                    "KTAS_expert": ktas_expert,
                    "matches_ktas_rn": structured["predicted_ktas"] == ktas_rn,
                    "matches_ktas_expert": structured["predicted_ktas"] == ktas_expert,
                    "matches_either_reference": structured["predicted_ktas"] in {ktas_rn, ktas_expert},
                    "latency_seconds": latency_seconds,
                    "first_look": structured["first_look"],
                    "chief_complaint": structured["chief_complaint"],
                    "chief_complaint_risk": structured["chief_complaint_risk"],
                    "primary_modifiers": structured["primary_modifiers"],
                    "secondary_modifiers": structured["secondary_modifiers"],
                    "decisive_factor": structured["decisive_factor"],
                    "raw_model_output": raw_text,
                },
            )
            processed += 1
            print(
                f"Processed {case_id}: KTAS {structured['predicted_ktas']} "
                f"(RN match={structured['predicted_ktas'] == ktas_rn}, "
                f"expert match={structured['predicted_ktas'] == ktas_expert}, "
                f"either match={structured['predicted_ktas'] in {ktas_rn, ktas_expert}}, "
                f"latency={latency_seconds}s)"
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            print(f"Failed on {case_id}: {exc}", file=sys.stderr)
            return 1

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print(f"Completed model runs: {processed}")
    print(f"Skipped existing cases: {skipped}")
    print(f"Resume cache active: {not args.no_resume}")
    print(f"Output path: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
