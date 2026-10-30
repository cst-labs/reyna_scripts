#!/usr/bin/env python3
"""Build a wide comparison CSV from analysis3 per-model KTAS prediction files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("scripts/analysis3/output/model_comparison_wide.csv")
DEFAULT_GPT_PATH = Path("scripts/analysis3/output/azure_gpt_5_4_predictions.csv")
DEFAULT_OLLAMA_DIR = Path("scripts/analysis3/output/ollama")
DEFAULT_OLLAMA_FILES = [
    "gemma2_9b_predictions.csv",
    "llama3.1_8b_predictions.csv",
    "medgemma_4b_predictions.csv",
    "mistral_predictions.csv",
    "qwen2.5_7b_predictions.csv",
]
OPTIONAL_QWEN3_FILE = "qwen3_1.7b_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a wide KTAS comparison CSV from analysis3 prediction files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Wide CSV output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--gpt-path", type=Path, default=DEFAULT_GPT_PATH, help=f"GPT prediction CSV path (default: {DEFAULT_GPT_PATH})")
    parser.add_argument("--ollama-dir", type=Path, default=DEFAULT_OLLAMA_DIR, help=f"Ollama prediction directory (default: {DEFAULT_OLLAMA_DIR})")
    parser.add_argument("--include-gpt", action="store_true", help="Require the GPT prediction CSV and include it in the wide table.")
    parser.add_argument("--include-qwen3", action="store_true", help="Include qwen3:1.7b even if it has only partial coverage.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def slugify_column(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def derive_model_name(path: Path, rows: list[dict[str, Any]]) -> str:
    if rows and rows[0].get("model"):
        return str(rows[0]["model"])
    stem = path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    return stem


def prediction_column_name(model_name: str) -> str:
    return f"pred_{slugify_column(model_name)}"


def normalize_prediction(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def load_prediction_map(path: Path) -> tuple[str, dict[str, dict[str, str]]]:
    rows = read_csv_rows(path)
    model_name = derive_model_name(path, rows)
    prediction_map: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        prediction_map[case_id] = {
            "prediction": normalize_prediction(row.get("predicted_ktas")),
            "KTAS_RN": str(row.get("KTAS_RN", "")).strip(),
            "KTAS_expert": str(row.get("KTAS_expert", "")).strip(),
        }
    return model_name, prediction_map


def main() -> int:
    args = parse_args()

    input_paths: list[Path] = []
    if args.gpt_path.exists():
        input_paths.append(args.gpt_path)
    elif args.include_gpt:
        raise SystemExit(f"Missing GPT prediction file: {args.gpt_path}")

    input_paths.extend(args.ollama_dir / filename for filename in DEFAULT_OLLAMA_FILES)
    if args.include_qwen3:
        input_paths.append(args.ollama_dir / OPTIONAL_QWEN3_FILE)

    missing_paths = [path for path in input_paths if not path.exists()]
    if missing_paths:
        raise SystemExit("Missing prediction files:\n" + "\n".join(f"- {path}" for path in missing_paths))
    if not input_paths:
        raise SystemExit("No prediction files were selected. Provide Ollama outputs or enable GPT input.")

    model_maps: list[tuple[str, dict[str, dict[str, str]]]] = [load_prediction_map(path) for path in input_paths]

    base_model_name, base_map = model_maps[0]
    case_ids = sorted(base_map.keys())
    rows_to_write: list[dict[str, str]] = []

    model_columns = [prediction_column_name(model_name) for model_name, _ in model_maps]
    fieldnames = ["case_id", "KTAS_expert", "KTAS_RN", *model_columns]

    for case_id in case_ids:
        base_row = base_map[case_id]
        merged_row = {
            "case_id": case_id,
            "KTAS_expert": base_row["KTAS_expert"],
            "KTAS_RN": base_row["KTAS_RN"],
        }
        for model_name, prediction_map in model_maps:
            merged_row[prediction_column_name(model_name)] = prediction_map.get(case_id, {}).get("prediction", "")
        rows_to_write.append(merged_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)

    print(f"Wide comparison CSV written: {args.output}")
    print(f"Reference base model: {base_model_name}")
    print(f"Cases written: {len(rows_to_write)}")
    print("Included prediction columns:")
    for column in model_columns:
        print(f"- {column}")
    if not args.gpt_path.exists() and not args.include_gpt:
        print("GPT input was not present, so the wide table was built from non-GPT outputs only.")
    if not args.include_qwen3:
        print("Excluded partial model: qwen3:1.7b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
