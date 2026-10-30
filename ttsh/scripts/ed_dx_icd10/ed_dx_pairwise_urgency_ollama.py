"""
Pairwise ED-diagnosis urgency comparisons with local Ollama/Gemma 4.

Builds the top 100 ED Dx mentions from data/nni_cleaned_orders.csv, asks the
model which diagnosis in each unordered pair has greater clinical urgency for
emergency prioritisation, and repeats each pair 10 times at temperature 0.

Calls: local Ollama REST API (http://localhost:11434/api/chat), default model
gemma4:e4b, with think=true. Override with OLLAMA_MODEL=<name>.

Checkpointing: every completed repeat is appended immediately to raw.jsonl.
Rerunning skips any already-completed (model, prompt_version, pair_id, repeat)
record, making the run resumable.

Outputs:
  analysis/ed_dx_pairwise_urgency_gemma4-e4b/raw.jsonl
  analysis/ed_dx_pairwise_urgency_gemma4-e4b/results.xlsx

Use --rebuild-only to rebuild the xlsx from raw.jsonl without calling Ollama.
Use --limit-pairs N for a small smoke/progress run.
"""
import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from itertools import combinations

import pandas as pd
from openpyxl.utils import get_column_letter


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE, "data", "nni_cleaned_orders.csv")
ANALYSIS = os.path.join(BASE, "analysis")

OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
MODEL_SLUG = re.sub(r"[^A-Za-z0-9_.-]+", "-", OLLAMA_MODEL).strip("-")
OUT_DIR = os.path.join(ANALYSIS, f"ed_dx_pairwise_urgency_{MODEL_SLUG}")
JSONL_PATH = os.path.join(OUT_DIR, "raw.jsonl")
XLSX_PATH = os.path.join(OUT_DIR, "results.xlsx")

TOP_N = 100
N_REPEATS = int(os.environ.get("N_REPEATS", "5"))
PROMPT_VERSION = "ed_dx_pairwise_urgency_v1"

CHOICE_A = "A"
CHOICE_B = "B"
CHOICE_TIE = "Tie/Insufficient information"
VALID_CHOICES = {CHOICE_A, CHOICE_B, CHOICE_TIE}

SYSTEM_PROMPT = """You are an emergency medicine triage reviewer supporting retrospective CT prioritisation analysis.

Task: compare two emergency department diagnoses under the same otherwise unspecified clinical context and decide which diagnosis usually represents greater clinical urgency for emergency prioritisation.

Use only the diagnosis labels provided. Assume the patient is in an ED/acute-care setting and that emergency prioritisation means the diagnosis with greater immediate risk of death, neurological deterioration, time-critical morbidity, or need for urgent intervention if delayed. Do not invent age, vital signs, mechanism severity, imaging findings, anticoagulation status, pregnancy status, or other missing details.

Choose:
- A if Diagnosis A is clearly more clinically urgent than Diagnosis B.
- B if Diagnosis B is clearly more clinically urgent than Diagnosis A.
- Tie/Insufficient information if the urgency is similar, highly context-dependent, ambiguous, nonspecific, or cannot be distinguished from the labels alone.

Return JSON only, with no markdown:
{"choice":"A|B|Tie/Insufficient information","confidence":"low|medium|high","rationale":"one concise sentence"}
"""


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text


def top_ed_dx_mentions():
    df = pd.read_csv(DATA_PATH)
    mentions = []
    for value in df["ED Dx"].fillna(""):
        seen = set()
        for part in str(value).split(";"):
            dx = normalize_text(part)
            if dx and dx not in seen:
                mentions.append(dx)
                seen.add(dx)
    counts = pd.Series(mentions, name="diagnosis").value_counts().head(TOP_N)
    return [
        {"rank": i, "diagnosis": dx, "n_orders": int(n)}
        for i, (dx, n) in enumerate(counts.items(), start=1)
    ]


def pair_id(dx_a, dx_b):
    digest = hashlib.sha1(f"{dx_a}\n{dx_b}".encode("utf-8")).hexdigest()[:12]
    return digest


def build_pairs(diagnoses):
    rows = []
    for a, b in combinations(diagnoses, 2):
        rows.append({
            "pair_id": pair_id(a["diagnosis"], b["diagnosis"]),
            "rank_a": a["rank"],
            "diagnosis_a": a["diagnosis"],
            "n_orders_a": a["n_orders"],
            "rank_b": b["rank"],
            "diagnosis_b": b["diagnosis"],
            "n_orders_b": b["n_orders"],
        })
    return rows


def build_user_prompt(dx_a, dx_b):
    return (
        "Given the same clinical context, which diagnosis represents greater "
        "clinical urgency for emergency prioritisation: Diagnosis A or Diagnosis B?\n\n"
        f"Diagnosis A: {dx_a}\n"
        f"Diagnosis B: {dx_b}\n\n"
        "Choose A, B, or Tie/Insufficient information."
    )


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 80)
    ws.freeze_panes = "A2"


def excel_cell(value, limit=32760):
    if value is None:
        return value
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + " ... [truncated in xlsx; full value in raw.jsonl]"


def ollama_chat(system_prompt, user_prompt, retries=2):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": True,
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(1, retries + 2):
        request = urllib.request.Request(
            OLLAMA_CHAT_URL, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if attempt > retries:
                raise SystemExit(
                    f"Ollama returned HTTP {e.code} at {OLLAMA_CHAT_URL} after {retries + 1} attempts.\n"
                    f"Response body: {body}"
                )
            print(f"  (attempt {attempt} failed: HTTP {e.code}: {body}; retrying...)")
            time.sleep(2)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt > retries:
                raise SystemExit(
                    f"Could not reach Ollama at {OLLAMA_CHAT_URL} after {retries + 1} attempts ({e}). "
                    "Is the Ollama daemon running? Try `ollama serve`."
                )
            print(f"  (attempt {attempt} failed: {e}; retrying...)")
            time.sleep(2)


def parse_choice(content):
    text = (content or "").strip()
    parsed = None
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = None
    if isinstance(parsed, dict):
        choice = str(parsed.get("choice", "")).strip()
        if choice in VALID_CHOICES:
            return choice, parsed
        if choice.lower() in {"tie", "insufficient", "insufficient information"}:
            parsed["choice"] = CHOICE_TIE
            return CHOICE_TIE, parsed

    cleaned = re.sub(r"[*_`#]", "", text).strip()
    match = re.search(r"\b(choice|answer)\s*[:=-]\s*(A|B)\b", cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(2).upper(), parsed
    if re.search(r"\bTie\b|\bInsufficient information\b", cleaned, flags=re.IGNORECASE):
        return CHOICE_TIE, parsed
    match = re.search(r"^\s*(A|B)\s*$", cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper(), parsed
    return "UNPARSEABLE", parsed


def load_records(jsonl_path):
    records = []
    if not os.path.exists(jsonl_path):
        return records
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def scoped_records(records):
    return [
        r for r in records
        if r.get("model") == OLLAMA_MODEL and r.get("prompt_version") == PROMPT_VERSION
    ]


def sort_records(records):
    return sorted(
        records,
        key=lambda r: (
            int(r.get("repeat_index") or 0),
            float(r.get("created_at_unix") or 0),
        ),
    )


def records_by_pair(records):
    grouped = {}
    for rec in scoped_records(records):
        grouped.setdefault(rec.get("pair_id"), []).append(rec)
    return {pair_id: sort_records(group) for pair_id, group in grouped.items()}


def next_repeat_index(existing_records):
    used = [
        int(r.get("repeat_index"))
        for r in existing_records
        if str(r.get("repeat_index", "")).isdigit()
    ]
    return max(used, default=0) + 1


def append_record(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def run_one(pair, repeat_index):
    user_prompt = build_user_prompt(pair["diagnosis_a"], pair["diagnosis_b"])
    result = ollama_chat(SYSTEM_PROMPT, user_prompt)
    msg = result.get("message", {})
    content = (msg.get("content") or "").strip()
    thinking = (msg.get("thinking") or "").strip()
    choice, parsed_json = parse_choice(content)
    return {
        "model": OLLAMA_MODEL,
        "prompt_version": PROMPT_VERSION,
        "pair_id": pair["pair_id"],
        "repeat_index": repeat_index,
        "rank_a": pair["rank_a"],
        "diagnosis_a": pair["diagnosis_a"],
        "n_orders_a": pair["n_orders_a"],
        "rank_b": pair["rank_b"],
        "diagnosis_b": pair["diagnosis_b"],
        "n_orders_b": pair["n_orders_b"],
        "choice": choice,
        "parsed_json": parsed_json,
        "content": content,
        "thinking": thinking,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "created_at_unix": time.time(),
    }


def build_xlsx(diagnoses, pairs, jsonl_path, xlsx_path):
    records = load_records(jsonl_path)
    grouped = records_by_pair(records)
    selected_records = []
    summary_rows = []
    for pair in pairs:
        cached_for_pair = grouped.get(pair["pair_id"], [])
        selected_for_pair = cached_for_pair[:N_REPEATS]
        selected_records.extend(selected_for_pair)
        choices = [r.get("choice") for r in selected_for_pair]
        counts = Counter(choices)
        parseable = [c for c in choices if c in VALID_CHOICES]
        parseable_counts = Counter(parseable)
        majority_choice, majority_n = parseable_counts.most_common(1)[0] if parseable_counts else (None, 0)
        if majority_choice == CHOICE_A:
            majority_dx = pair["diagnosis_a"]
        elif majority_choice == CHOICE_B:
            majority_dx = pair["diagnosis_b"]
        elif majority_choice == CHOICE_TIE:
            majority_dx = CHOICE_TIE
        else:
            majority_dx = None
        summary_rows.append({
            **pair,
            "n_completed": len(choices),
            "n_cached_total": len(cached_for_pair),
            "n_cached_extra_ignored": max(0, len(cached_for_pair) - N_REPEATS),
            "n_a": counts.get(CHOICE_A, 0),
            "n_b": counts.get(CHOICE_B, 0),
            "n_tie_insufficient": counts.get(CHOICE_TIE, 0),
            "n_unparseable": counts.get("UNPARSEABLE", 0),
            "majority_choice": majority_choice,
            "majority_diagnosis": majority_dx,
            "majority_n": majority_n,
            "self_consistency_rate": majority_n / len(parseable) if parseable else None,
        })
    pair_summary = pd.DataFrame(summary_rows)

    diagnosis_rows = []
    for dx in diagnoses:
        dx_name = dx["diagnosis"]
        involved = pair_summary[
            (pair_summary["diagnosis_a"] == dx_name) | (pair_summary["diagnosis_b"] == dx_name)
        ]
        wins = int((involved["majority_diagnosis"] == dx_name).sum()) if not involved.empty else 0
        ties = int((involved["majority_choice"] == CHOICE_TIE).sum()) if not involved.empty else 0
        losses = int(involved["majority_choice"].isin([CHOICE_A, CHOICE_B]).sum() - wins) if not involved.empty else 0
        diagnosis_rows.append({**dx, "majority_wins": wins, "majority_losses": losses, "majority_ties": ties})
    diagnosis_summary = pd.DataFrame(diagnosis_rows)

    notes = pd.DataFrame([
        {"Note": f"Top {TOP_N} diagnoses are ED Dx mentions from {DATA_PATH}, split on semicolons and de-duplicated within each order."},
        {"Note": f"All unordered pairs are compared: {len(pairs):,} pairs x {N_REPEATS} repeats = {len(pairs) * N_REPEATS:,} planned model calls."},
        {"Note": f"Ollama model={OLLAMA_MODEL}; URL={OLLAMA_CHAT_URL}; think=true; options.temperature=0."},
        {"Note": f"JSONL is append-only and resumable. A pair is considered complete once it has any {N_REPEATS} cached runs for this model/prompt, even if those runs were originally numbered above {N_REPEATS}."},
        {"Note": f"If a pair has more than {N_REPEATS} cached runs from an earlier larger repeat target, the extra rows remain in raw.jsonl but are ignored in Pair_Summary and Runs."},
        {"Note": f"System prompt version: {PROMPT_VERSION}."},
    ])

    runs = pd.DataFrame(selected_records)
    if runs.empty:
        runs_xlsx = pd.DataFrame(columns=[
            "model", "prompt_version", "pair_id", "repeat_index", "rank_a",
            "diagnosis_a", "rank_b", "diagnosis_b", "choice", "confidence",
            "rationale", "content", "thinking",
        ])
    else:
        runs_xlsx = runs.copy()
        runs_xlsx["confidence"] = runs_xlsx["parsed_json"].apply(
            lambda x: x.get("confidence") if isinstance(x, dict) else None
        )
        runs_xlsx["rationale"] = runs_xlsx["parsed_json"].apply(
            lambda x: x.get("rationale") if isinstance(x, dict) else None
        )
        keep_cols = [
            "model", "prompt_version", "pair_id", "repeat_index", "rank_a",
            "diagnosis_a", "n_orders_a", "rank_b", "diagnosis_b", "n_orders_b",
            "choice", "confidence", "rationale", "content", "thinking",
            "created_at_unix",
        ]
        runs_xlsx = runs_xlsx[[c for c in keep_cols if c in runs_xlsx.columns]]
        for col in ["content", "thinking", "rationale"]:
            if col in runs_xlsx.columns:
                runs_xlsx[col] = runs_xlsx[col].map(excel_cell)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="Notes", index=False)
        diagnosis_summary.to_excel(writer, sheet_name="Diagnoses", index=False)
        pair_summary.to_excel(writer, sheet_name="Pair_Summary", index=False)
        runs_xlsx.to_excel(writer, sheet_name="Runs", index=False)
        pd.DataFrame([{"system_prompt": SYSTEM_PROMPT}]).to_excel(writer, sheet_name="Prompt", index=False)
        for ws in writer.book.worksheets:
            auto_col_width(ws)
    active_done = sum(min(len(grouped.get(pair["pair_id"], [])), N_REPEATS) for pair in pairs)
    print(f"Wrote {xlsx_path}")
    print(f"Completed runs used for {N_REPEATS}-repeat target: {active_done:,} / {len(pairs) * N_REPEATS:,}")
    print(f"Raw cached rows retained: {len(records):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-only", action="store_true", help="Only rebuild results.xlsx from raw.jsonl.")
    parser.add_argument("--limit-pairs", type=int, default=None, help="Only process the first N pairs.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    diagnoses = top_ed_dx_mentions()
    pairs = build_pairs(diagnoses)
    if args.limit_pairs is not None:
        pairs = pairs[: args.limit_pairs]

    if not args.rebuild_only:
        records = load_records(JSONL_PATH)
        grouped = records_by_pair(records)
        total = len(pairs) * N_REPEATS
        planned_done = sum(min(len(grouped.get(pair["pair_id"], [])), N_REPEATS) for pair in pairs)
        print(f"Writing checkpoint to {JSONL_PATH}")
        print(f"Model: {OLLAMA_MODEL}; pairs: {len(pairs):,}; repeats per pair: {N_REPEATS}; planned calls: {total:,}")
        print(f"Already complete for {N_REPEATS}-repeat target: {planned_done:,}/{total:,}")
        call_i = planned_done
        for pair in pairs:
            cached_for_pair = grouped.get(pair["pair_id"], [])
            while len(cached_for_pair) < N_REPEATS:
                repeat = next_repeat_index(cached_for_pair)
                call_i += 1
                print(
                    f"[{call_i:,}/{total:,}] repeat {repeat}/{N_REPEATS}: "
                    f"{pair['diagnosis_a']}  vs  {pair['diagnosis_b']}"
                )
                record = run_one(pair, repeat)
                append_record(JSONL_PATH, record)
                cached_for_pair.append(record)
                cached_for_pair = sort_records(cached_for_pair)
                grouped[pair["pair_id"]] = cached_for_pair

    build_xlsx(diagnoses, build_pairs(diagnoses), JSONL_PATH, XLSX_PATH)


if __name__ == "__main__":
    main()
