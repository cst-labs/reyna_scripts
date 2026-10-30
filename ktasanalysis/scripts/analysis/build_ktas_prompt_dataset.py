#!/usr/bin/env python3
"""Build prompt-ready KTAS cases from the cleaned CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ktas_prompt_templates import build_prompt_record


DEFAULT_INPUT = Path("data/originaldata_cleaned.csv")
DEFAULT_OUTPUT = Path("scripts/analysis/output/ktas_prompt_cases.jsonl")
DEFAULT_GUIDELINES = Path("data/guidelines.xlsx")

EXCLUDED_INPUT_FIELDS = {
    "KTAS_RN",
    "KTAS_expert",
    "Error_group",
    "KTAS duration_min",
    "mistriage",
    "Group",
    "Disposition",
}

SEX_MAP = {
    "1": "Female",
    "2": "Male",
}

ARRIVAL_MODE_MAP = {
    "1": "Walking",
    "2": "119 use",
    "3": "Private car",
    "4": "Private ambulance",
    "5": "Public transportation/police",
    "6": "Wheelchair",
    "7": "Others",
}

INJURY_MAP = {
    "1": "Non-injury",
    "2": "Injury",
}

MENTAL_MAP = {
    "1": "Alert",
    "2": "Verbal response",
    "3": "Pain response",
    "4": "Unconsciousness",
}

PAIN_MAP = {
    "0": "No pain",
    "1": "Pain present",
    "2": "Pain present",
}

FIELD_VALUE_MAPS = {
    "Sex": SEX_MAP,
    "Arrival mode": ARRIVAL_MODE_MAP,
    "Injury": INJURY_MAP,
    "Mental": MENTAL_MAP,
    "Pain": PAIN_MAP,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prompt-ready KTAS cases from the cleaned CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Cleaned CSV input (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Prompt JSONL output (default: {DEFAULT_OUTPUT})")
    parser.add_argument(
        "--guidelines",
        type=Path,
        default=DEFAULT_GUIDELINES,
        help=f"Guidelines workbook path, kept for documentation compatibility (default: {DEFAULT_GUIDELINES})",
    )
    return parser.parse_args()


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "#null!"}:
        return None
    return text


def maybe_numeric(value: str) -> Any:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def decode_field(field_name: str, raw_value: Any) -> Any:
    cleaned = clean_scalar(raw_value)
    if cleaned is None:
        return None

    value_map = FIELD_VALUE_MAPS.get(field_name)
    if value_map is not None:
        return value_map.get(cleaned, cleaned)

    if field_name == "NRS_pain":
        numeric = maybe_numeric(cleaned)
        return numeric

    if field_name in {"Age", "SBP", "DBP", "HR", "RR", "BT", "Saturation"}:
        return maybe_numeric(cleaned)

    return cleaned


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_model_inputs(row: dict[str, Any]) -> dict[str, Any]:
    model_inputs: dict[str, Any] = {}
    for field_name, raw_value in row.items():
        if field_name in EXCLUDED_INPUT_FIELDS:
            continue
        model_inputs[field_name] = decode_field(field_name, raw_value)
    return model_inputs


def main() -> int:
    args = parse_args()
    rows = read_csv_rows(args.input)
    prompt_records: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        case_id = f"case_{index:06d}"
        prompt_records.append(
            build_prompt_record(
                case_id=case_id,
                model_inputs=build_model_inputs(row),
                ktas_rn=decode_field("KTAS_RN", row.get("KTAS_RN")),
                ktas_expert=decode_field("KTAS_expert", row.get("KTAS_expert")),
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in prompt_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Prompt dataset written: {args.output}")
    print(f"Cases written: {len(prompt_records)}")
    print(f"Guidelines reference path: {args.guidelines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
