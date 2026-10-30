#!/usr/bin/env python3
"""
Fill reasoning for the KTAS edtriage train split.

This script reuses reasoning from data/KTAS/train_with_reasoning.csv whenever a
row in data/KTAS/edtriage_split/train.csv matches a row from
data/KTAS/train_numeric.csv. For rows without an existing match, it generates
reasoning with the same KTAS prompt flow used in scripts/generate_ktas_reasoning.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.prompts import INSTRUCTIONS, format_row
from utils.utils import query_llm_full


NUMERIC_MATCH_COLUMNS = [
    "Group",
    "Sex",
    "Age",
    "Patients number per hour",
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
    "KTAS_RN",
    "Diagnosis in ED",
    "KTAS_expert",
    "Length of stay_min",
]

NUMERIC_COLUMNS = [
    "Group",
    "Sex",
    "Age",
    "Patients number per hour",
    "Arrival mode",
    "Injury",
    "Mental",
    "Pain",
    "NRS_pain",
    "SBP",
    "DBP",
    "HR",
    "RR",
    "BT",
    "Saturation",
    "KTAS_RN",
    "Disposition",
    "KTAS_expert",
    "Error_group",
    "Length of stay_min",
    "mistriage",
]


def normalize_value(column: str, value) -> object:
    if pd.isna(value):
        return None
    if column in NUMERIC_COLUMNS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return str(value).strip()


def row_key(row: pd.Series, columns: list[str]) -> tuple[object, ...]:
    return tuple(normalize_value(column, row.get(column)) for column in columns)


def build_code_map(numeric_df: pd.DataFrame, text_df: pd.DataFrame, column: str) -> dict[object, str]:
    mapping: dict[object, str] = {}
    for _, pair in pd.DataFrame(
        {"code": numeric_df[column], "label": text_df[column]}
    ).dropna().drop_duplicates().iterrows():
        mapping[normalize_value(column, pair["code"])] = str(pair["label"]).strip()
    return mapping


def denormalize_row(
    row: pd.Series,
    arrival_map: dict[object, str],
    injury_map: dict[object, str],
    mental_map: dict[object, str],
    sex_map: dict[object, str],
) -> pd.Series:
    out = row.copy()

    arrival_key = normalize_value("Arrival mode", row.get("Arrival mode"))
    injury_key = normalize_value("Injury", row.get("Injury"))
    mental_key = normalize_value("Mental", row.get("Mental"))
    sex_key = normalize_value("Sex", row.get("Sex"))

    if arrival_key in arrival_map:
        out["Arrival mode"] = arrival_map[arrival_key]
    if injury_key in injury_map:
        out["Injury"] = injury_map[injury_key]
    if mental_key in mental_map:
        out["Mental"] = mental_map[mental_key]
    if sex_key in sex_map:
        out["Sex"] = sex_map[sex_key]

    return out


def format_number(value, suffix: str = "") -> str:
    if pd.isna(value):
        return "unknown"
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else "unknown"
    if number.is_integer():
        return f"{int(number)}{suffix}"
    return f"{number:.1f}{suffix}"


def format_pain(row: pd.Series) -> str:
    pain_flag = pd.to_numeric(row.get("Pain"), errors="coerce")
    pain_score = pd.to_numeric(row.get("NRS_pain"), errors="coerce")
    if pain_flag == 1 and pd.notna(pain_score):
        return f"{int(pain_score)}/10"
    if pain_flag == 1:
        return "present, but the score is not documented"
    return "no reported pain score"


def abnormal_flags(row: pd.Series, mental_map: dict[object, str]) -> list[str]:
    flags: list[str] = []
    sbp = pd.to_numeric(row.get("SBP"), errors="coerce")
    hr = pd.to_numeric(row.get("HR"), errors="coerce")
    rr = pd.to_numeric(row.get("RR"), errors="coerce")
    bt = pd.to_numeric(row.get("BT"), errors="coerce")
    sat = pd.to_numeric(row.get("Saturation"), errors="coerce")
    mental = mental_map.get(normalize_value("Mental", row.get("Mental")), str(row.get("Mental", "")))

    if mental and mental.lower() not in {"alert", "1"}:
        flags.append(f"mental status is {mental.lower()}")
    if pd.notna(sbp) and sbp < 90:
        flags.append(f"systolic blood pressure is low at {int(sbp)} mmHg")
    if pd.notna(hr) and (hr >= 120 or hr <= 50):
        flags.append(f"heart rate is concerning at {int(hr)}/min")
    if pd.notna(rr) and (rr >= 24 or rr <= 10):
        flags.append(f"respiratory rate is concerning at {int(rr)}/min")
    if pd.notna(sat) and sat < 92:
        flags.append(f"oxygen saturation is reduced at {int(sat)}%")
    if pd.notna(bt) and (bt >= 38.5 or bt < 35.5):
        flags.append(f"temperature is abnormal at {bt:.1f}°C")
    return flags


def ktas_level_blurb(level: int) -> str:
    blurbs = {
        1: "KTAS 1 fits patients who need immediate aggressive assessment and treatment.",
        2: "KTAS 2 fits high-risk patients who are not in full arrest but could deteriorate quickly.",
        3: "KTAS 3 fits patients who are currently stable but have meaningful acute risk and need prompt evaluation.",
        4: "KTAS 4 fits lower-acuity patients who are stable but still need ED evaluation and treatment.",
        5: "KTAS 5 fits minor or low-risk problems in a stable patient without urgent warning signs.",
    }
    return blurbs.get(level, f"KTAS {level} is the documented expert acuity.")


def get_system_prompt() -> str:
    cot_system = INSTRUCTIONS["triage-ktas"]["CoT"]["system"]
    cot_user = INSTRUCTIONS["triage-ktas"]["CoT"]["user"]
    return f"{cot_system}\n\n{cot_user}"


def generate_reasoning_with_llm(row: pd.Series) -> str:
    patient_desc = format_row(row, dataset="triage-ktas", serialization="natural")
    ktas_expert = int(pd.to_numeric(row.get("KTAS_expert"), errors="coerce"))

    user_prompt = (
        f"Patient:\n{patient_desc}\n\n"
        f"The expert has assigned this patient KTAS level {ktas_expert}. "
        f"Explain step by step why this KTAS level is correct for this patient."
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

    return reasoning if reasoning else ""


def generate_reasoning(
    row: pd.Series,
    arrival_map: dict[object, str],
    injury_map: dict[object, str],
    mental_map: dict[object, str],
) -> str:
    level = int(pd.to_numeric(row.get("KTAS_expert"), errors="coerce"))
    complaint = str(row.get("Chief_complain", "the presenting complaint")).strip()
    diagnosis = str(row.get("Diagnosis in ED", "the suspected ED diagnosis")).strip()
    arrival = arrival_map.get(normalize_value("Arrival mode", row.get("Arrival mode")), "the documented mode of arrival")
    injury = injury_map.get(normalize_value("Injury", row.get("Injury")), str(row.get("Injury", "")))
    mental = mental_map.get(normalize_value("Mental", row.get("Mental")), str(row.get("Mental", "")))
    pain = format_pain(row)
    flags = abnormal_flags(row, mental_map)

    lines = [
        f"KTAS level {level} is appropriate for this patient.",
        "",
        "### Step 1: Check for immediate life-threatening problems",
        f"- Mental status: {mental if mental else 'not clearly documented'}",
        f"- Blood pressure: {format_number(row.get('SBP'))}/{format_number(row.get('DBP'))} mmHg",
        f"- Heart rate: {format_number(row.get('HR'))}/min",
        f"- Respiratory rate: {format_number(row.get('RR'))}/min",
        f"- Temperature: {format_number(row.get('BT'), '°C')}",
    ]

    sat = pd.to_numeric(row.get("Saturation"), errors="coerce")
    if pd.notna(sat):
        lines.append(f"- Oxygen saturation: {int(sat)}%")
    lines.append(f"- Pain assessment: {pain}")

    if flags:
        lines.append("")
        lines.append("These findings show potential instability or at least enough physiologic concern that the patient cannot be treated as a minor low-acuity case:")
        for flag in flags:
            lines.append(f"- {flag}")
    else:
        lines.append("")
        lines.append("There is no clear evidence here of immediate airway failure, profound shock, or another resuscitation-level presentation.")

    lines.extend(
        [
            "",
            "### Step 2: Assess the clinical problem",
            f"- Chief complaint: {complaint}",
            f"- ED diagnosis: {diagnosis}",
            f"- Arrival mode: {arrival}",
        ]
    )
    if injury:
        lines.append(f"- Injury status: {injury}")

    level_specific = {
        1: "The combination of presentation and documented acuity indicates a condition requiring immediate clinician attention and no meaningful waiting time.",
        2: "This presentation is concerning for a potentially serious threat to life, limb, or organ function, even if the patient is not yet in complete physiologic collapse.",
        3: "This problem needs prompt workup and treatment because it carries real acute risk, but the documented vitals do not show the kind of extreme instability that would force KTAS 1 or 2.",
        4: "This appears to be an acute but comparatively stable problem. The patient still needs medical assessment, yet there are not enough red flags to justify a higher-acuity category.",
        5: "This looks like a minor complaint with a stable presentation and without warning signs suggesting urgent deterioration.",
    }
    lines.append(level_specific.get(level, ktas_level_blurb(level)))

    lines.extend(
        [
            "",
            "### Step 3: Determine final KTAS level",
            ktas_level_blurb(level),
        ]
    )

    if level == 1:
        lines.append("Because the expert label is KTAS 1, the safest interpretation is that this case requires immediate intervention rather than queued evaluation.")
    elif level == 2:
        lines.append("That is why KTAS 2 is more appropriate than KTAS 3 to 5 for this case.")
    elif level == 3:
        lines.append("That balance of meaningful acute concern with relative physiologic stability is most consistent with KTAS 3.")
    elif level == 4:
        lines.append("That makes KTAS 4 a reasonable fit over both higher-acuity and very minor KTAS 5 presentations.")
    elif level == 5:
        lines.append("That is why KTAS 5 is appropriate for this stable, low-risk presentation.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill reasoning for KTAS edtriage train split.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/KTAS/edtriage_split/train_with_reasoning.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N generated rows.",
    )
    parser.add_argument(
        "--local-fallback",
        action="store_true",
        help="If the LLM generation fails for a row, fall back to deterministic local reasoning.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file by keeping rows that already have reasoning.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    ktas_dir = repo_root / "data" / "KTAS"

    split_train_path = ktas_dir / "edtriage_split" / "train.csv"
    train_numeric_path = ktas_dir / "train_numeric.csv"
    train_text_path = ktas_dir / "train.csv"
    train_reasoning_path = ktas_dir / "train_with_reasoning.csv"

    split_df = pd.read_csv(split_train_path)
    train_numeric_df = pd.read_csv(train_numeric_path)
    train_text_df = pd.read_csv(train_text_path)
    train_reasoning_df = pd.read_csv(train_reasoning_path)

    if not (len(train_numeric_df) == len(train_text_df) == len(train_reasoning_df)):
        raise ValueError("Expected train_numeric, train, and train_with_reasoning to have aligned row counts.")

    arrival_map = build_code_map(train_numeric_df, train_text_df, "Arrival mode")
    injury_map = build_code_map(train_numeric_df, train_text_df, "Injury")
    mental_map = build_code_map(train_numeric_df, train_text_df, "Mental")
    sex_map = build_code_map(train_numeric_df, train_text_df, "Sex")

    reasoning_lookup: dict[tuple[object, ...], str] = {}
    for idx, row in train_numeric_df.iterrows():
        key = row_key(row, NUMERIC_MATCH_COLUMNS)
        reasoning = train_reasoning_df.loc[idx, "Reasoning"]
        if pd.notna(reasoning) and str(reasoning).strip() and key not in reasoning_lookup:
            reasoning_lookup[key] = str(reasoning).strip()

    output_path = args.output
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    output_df = split_df.copy()
    output_df["Reasoning"] = ""
    output_df["Reasoning_source"] = ""

    if args.resume and output_path.exists():
        existing_df = pd.read_csv(output_path)
        if len(existing_df) <= len(split_df) and "Reasoning" in existing_df.columns:
            for column in ("Reasoning", "Reasoning_source"):
                if column in existing_df.columns:
                    output_df.loc[existing_df.index, column] = existing_df[column]
            print(f"Resuming from existing output: {output_path}")
            already_done = (
                output_df["Reasoning"].fillna("").astype(str).str.strip().ne("").sum()
            )
            print(f"Existing rows with reasoning: {already_done}")
        else:
            print(f"Ignoring existing output at {output_path} because its shape/schema does not match the split train file.")

    generated_since_save = 0

    for idx, row in split_df.iterrows():
        existing_reasoning = str(output_df.at[idx, "Reasoning"]).strip() if pd.notna(output_df.at[idx, "Reasoning"]) else ""
        if existing_reasoning:
            continue

        key = row_key(row, NUMERIC_MATCH_COLUMNS)
        reasoning = reasoning_lookup.get(key)
        if reasoning:
            output_df.at[idx, "Reasoning"] = reasoning
            output_df.at[idx, "Reasoning_source"] = "reused_from_train_with_reasoning"
            continue

        try:
            llm_row = denormalize_row(row, arrival_map, injury_map, mental_map, sex_map)
            generated = generate_reasoning_with_llm(llm_row)
            if not generated.strip():
                raise ValueError("Empty reasoning returned from LLM.")
            output_df.at[idx, "Reasoning"] = generated
            output_df.at[idx, "Reasoning_source"] = "generated_with_azure_gpt_5_4"
        except Exception:
            if not args.local_fallback:
                raise
            generated = generate_reasoning(row, arrival_map, injury_map, mental_map)
            output_df.at[idx, "Reasoning"] = generated
            output_df.at[idx, "Reasoning_source"] = "generated_locally_fallback"

        generated_since_save += 1
        if generated_since_save % args.save_every == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    source_counts = output_df["Reasoning_source"].fillna("").value_counts().to_dict()
    reused = source_counts.get("reused_from_train_with_reasoning", 0)
    generated = source_counts.get("generated_with_azure_gpt_5_4", 0)
    fallback = source_counts.get("generated_locally_fallback", 0)
    print(f"Wrote {len(output_df)} rows to {output_path}")
    print(f"Reused reasoning: {reused}")
    print(f"Generated with azure-gpt-5.4: {generated}")
    print(f"Generated with local fallback: {fallback}")


if __name__ == "__main__":
    main()
