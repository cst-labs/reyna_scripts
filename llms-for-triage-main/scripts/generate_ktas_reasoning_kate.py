#!/usr/bin/env python3
"""
Generate KATE-based reasoning for each case in the KTAS training set.

For each patient in data/KTAS/train.csv, this script:
1. Formats the patient description using the natural language serializer
2. Sends the patient + KATE prompt + expert label to azure-gpt-5.4
3. Collects the reasoning
4. Saves the result to data/KTAS/train_with_reasoning_kate.csv

Supports checkpoint resumption and periodic saves.
"""

import sys
import os
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.prompts import format_row, INSTRUCTIONS
from utils.utils import query_llm_full


def get_system_prompt():
    """Return the system prompt with KATE instructions."""
    kate = INSTRUCTIONS["triage-ktas"]["KATE"]
    return f"{kate['system']}\n\n{kate['user']}"


def generate_reasoning_for_case(row):
    """Generate reasoning for a single case."""
    patient_desc = format_row(row, dataset='triage-ktas', serialization='natural')
    ktas_expert = int(row['KTAS_expert'])

    user_prompt = (
        f"Patient:\n{patient_desc}\n\n"
        f"The expert has assigned this patient KTAS level {ktas_expert}. "
        f"Explain why this KTAS level is appropriate for this patient."
    )

    prompt = {
        "system": get_system_prompt(),
        "user": user_prompt,
    }

    reasoning = query_llm_full(
        prompt,
        model="azure-gpt-5.4",
        temperature=0,
        top_p=0,
        max_tokens=1000,
        max_try_num=10,
        return_json=False,
    )

    return reasoning


def main():
    parser = argparse.ArgumentParser(
        description="Generate KTAS reasoning for training cases using KATE prompt."
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N rows (default: 10)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based, default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive, default: all rows)",
    )
    args = parser.parse_args()

    train_path = Path(__file__).parent.parent / "data" / "KTAS" / "train.csv"
    output_path = Path(__file__).parent.parent / "data" / "KTAS" / "train_with_reasoning_kate.csv"

    if not train_path.exists():
        print(f"Error: {train_path} not found.")
        sys.exit(1)

    df = pd.read_csv(train_path)
    total_rows = len(df)

    if args.end is None:
        args.end = total_rows
    else:
        args.end = min(args.end, total_rows)

    print(f"Train set: {total_rows} rows")
    print(f"Processing rows {args.start} to {args.end - 1}")

    if output_path.exists():
        df_out = pd.read_csv(output_path, dtype={'Reasoning': 'object'})
        print(f"Found existing output: {output_path}")
        print(f"Output has {len(df_out)} rows with {df_out['Reasoning'].notna().sum()} non-empty reasoning values")

        if len(df_out) != total_rows:
            print(f"Warning: output has {len(df_out)} rows but train has {total_rows}. Starting fresh.")
            df_out = df.copy()
            df_out['Reasoning'] = ""
            df_out['Reasoning'] = df_out['Reasoning'].astype('object')
    else:
        df_out = df.copy()
        df_out['Reasoning'] = ""
        df_out['Reasoning'] = df_out['Reasoning'].astype('object')
        print(f"Creating new output: {output_path}")

    for i in range(args.start, args.end):
        if pd.notna(df_out.loc[i, 'Reasoning']) and df_out.loc[i, 'Reasoning'].strip():
            print(f"Row {i}: Already has reasoning, skipping.")
            continue

        print(f"Row {i}: Generating reasoning...")
        try:
            reasoning = generate_reasoning_for_case(df.loc[i])
            df_out.loc[i, 'Reasoning'] = reasoning if reasoning else ""
            print(f"  ✓ Done")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            df_out.loc[i, 'Reasoning'] = ""

        if (i - args.start + 1) % args.save_every == 0:
            print(f"Saving checkpoint after row {i}...")
            df_out.to_csv(output_path, index=False)
            print(f"  Saved to {output_path}")

    print(f"Final save...")
    df_out.to_csv(output_path, index=False)
    print(f"✓ Complete. Output saved to {output_path}")
    print(f"  Total rows: {len(df_out)}")
    print(f"  Rows with reasoning: {df_out['Reasoning'].notna().sum()}")


if __name__ == "__main__":
    main()
