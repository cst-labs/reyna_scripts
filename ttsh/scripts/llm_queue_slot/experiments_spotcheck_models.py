"""
One-shot spot check of two new local models/APIs against the existing
gemma4:e4b baseline, on a handful of notable cases pulled from already-run
experiments (NOT a full rerun):

  Part A - ED Dx urgency pairwise choice (A/B/Tie), vs the cached
           Bradley-Terry results in
           analysis/ed_dx_pairwise_urgency_gemma4-e4b/bradley_terry_ranking.xlsx
           Models: gemma4:e4b (Ollama), TwiL-LM3 (Ollama, pulled from HF
           GGUF), Laya (Unsloth Studio's local Decision API -- a
           classification endpoint, not chat; mapped onto this task as one
           `choice` question over A/B/Tie).

  Part B - CT queue insertion-position agent, vs the pinned case list /
           cached run in analysis/nni_llm_queue_slot_consistency/. Reuses
           build_snapshot/build_prompt_no_tools/parse logic from
           nni_llm_queue_slot_consistency_v2.py directly (imported, not copied). Laya is
           skipped here -- it's single-text classification, this task is
           multi-item sequential reasoning, wrong shape for the API.

No new checkpoint/xlsx files are written; this just prints a comparison
table to stdout.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

import pandas as pd

import nni_llm_queue_slot_consistency_v2 as pa

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS = os.path.join(BASE, "analysis")

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
LAYA_URL = "http://127.0.0.1:8888/v1/systemone"
LAYA_KEY = os.environ.get("LAYA_API_KEY", "sk-unsloth-00ac45fcd24dbbd9d5aa466aec6ecea2")
TWIL_MODEL = "hf.co/webAI-Official/TwIL-LM3:Q4_K_M"
CHAT_MODELS = ["gemma4:e4b", TWIL_MODEL]

BT_XLSX = os.path.join(ANALYSIS, "ed_dx_pairwise_urgency_gemma4-e4b", "bradley_terry_ranking.xlsx")
CASE_LIST_PATH = os.path.join(ANALYSIS, "nni_llm_queue_slot_consistency", "case_list.json")

CHOICE_A, CHOICE_B, CHOICE_TIE = "A", "B", "Tie/Insufficient information"
VALID_CHOICES = {CHOICE_A, CHOICE_B, CHOICE_TIE}

ED_DX_SYSTEM_PROMPT = """You are an emergency medicine triage reviewer supporting retrospective CT prioritisation analysis.

Task: compare two emergency department diagnoses under the same otherwise unspecified clinical context and decide which diagnosis usually represents greater clinical urgency for emergency prioritisation.

Use only the diagnosis labels provided. Assume the patient is in an ED/acute-care setting and that emergency prioritisation means the diagnosis with greater immediate risk of death, neurological deterioration, time-critical morbidity, or need for urgent intervention if delayed. Do not invent age, vital signs, mechanism severity, imaging findings, anticoagulation status, pregnancy status, or other missing details.

Choose:
- A if Diagnosis A is clearly more clinically urgent than Diagnosis B.
- B if Diagnosis B is clearly more clinically urgent than Diagnosis A.
- Tie/Insufficient information if the urgency is similar, highly context-dependent, ambiguous, nonspecific, or cannot be distinguished from the labels alone.

Return JSON only, with no markdown:
{"choice":"A|B|Tie/Insufficient information","confidence":"low|medium|high","rationale":"one concise sentence"}
"""

# -----------------------------------------------------------------------
# Notable ED Dx pairs, pulled from the already-cached gemma4 Bradley-Terry
# run rather than hand-picked, so the comparison is against real results:
#   - clear-cut: gemma4's #1 vs #100
#   - mid-table judgment call: two adjacent, similarly-scored diagnoses
#   - two of the biggest BT-rank vs observed-priority disagreements
# -----------------------------------------------------------------------
NOTABLE_PAIRS = [
    ("Cardiac arrest", "Abrasion", "clear-cut (gemma4 rank 1 vs 100)"),
    ("Declining functional status", "Fracture of ribs, multiple", "mid-table judgment call (gemma4 rank 49 vs 50)"),
    ("Head injury", "LOC (loss of consciousness)",
     "LOC is gemma4's biggest BT-vs-observed-priority outlier (-66 ranks)"),
    ("Head injury", "Cough", "Cough is gemma4's biggest BT-vs-observed-priority outlier (+64 ranks)"),
]


def ollama_chat(model, system_prompt, user_prompt, retries=2):
    payload = {
        "model": model,
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
            with urllib.request.urlopen(request, timeout=600) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt > retries:
                raise SystemExit(f"Could not reach Ollama ({model}) after {retries + 1} attempts: {e}")
            print(f"    (attempt {attempt} failed: {e}; retrying...)")
            time.sleep(2)


def parse_choice(content):
    text = (content or "").strip()
    parsed = None
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
            return choice
        if choice.lower().startswith("tie") or "insufficient" in choice.lower():
            return CHOICE_TIE
    cleaned = re.sub(r"[*_`#]", "", text)
    if re.search(r"\bTie\b|\bInsufficient information\b", cleaned, flags=re.IGNORECASE):
        return CHOICE_TIE
    match = re.search(r"\b(A|B)\b", cleaned)
    if match:
        return match.group(1).upper()
    return "UNPARSEABLE"


def laya_choice(dx_a, dx_b):
    payload = {
        "model": "laya",
        "state": f"Diagnosis A: {dx_a}\nDiagnosis B: {dx_b}",
        "questions": {
            "urgency": {
                "type": "choice",
                "instructions": (
                    "In an ED/acute-care setting, which diagnosis usually represents greater "
                    "clinical urgency for emergency prioritisation (immediate risk of death, "
                    "neurological deterioration, time-critical morbidity, or need for urgent "
                    "intervention if delayed)?"
                ),
                "criteria": {
                    "A": "Diagnosis A is clearly more clinically urgent than Diagnosis B",
                    "B": "Diagnosis B is clearly more clinically urgent than Diagnosis A",
                    "Tie": "similar urgency, ambiguous, or cannot be distinguished from the labels alone",
                },
            }
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        LAYA_URL, data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LAYA_KEY}"},
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    answer = body["answers"]["urgency"]
    choice = {"A": CHOICE_A, "B": CHOICE_B, "Tie": CHOICE_TIE}.get(answer["choice"], "UNPARSEABLE")
    return choice, answer["probabilities"]


def run_ed_dx_spotcheck():
    print("=" * 90)
    print("PART A -- ED Dx urgency: notable pairs, gemma4 vs TwiL-LM3 vs Laya")
    print("=" * 90)

    fc = pd.read_excel(BT_XLSX, sheet_name="Final_Comparison").set_index("diagnosis")

    rows = []
    for dx_a, dx_b, note in NOTABLE_PAIRS:
        print(f"\n--- {dx_a}  vs  {dx_b}   ({note}) ---")
        user_prompt = f"Diagnosis A: {dx_a}\nDiagnosis B: {dx_b}"

        row = {"diagnosis_a": dx_a, "diagnosis_b": dx_b, "note": note}
        for model in CHAT_MODELS:
            result = ollama_chat(model, ED_DX_SYSTEM_PROMPT, user_prompt)
            content = (result.get("message", {}) or {}).get("content", "")
            choice = parse_choice(content)
            row[model] = choice
            print(f"  {model:30s} -> {choice}")

        try:
            laya_pick, laya_probs = laya_choice(dx_a, dx_b)
            row["laya"] = f"{laya_pick} {laya_probs}"
            print(f"  {'laya':30s} -> {laya_pick}  probs={laya_probs}")
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError) as e:
            row["laya"] = f"ERROR: {e}"
            print(f"  {'laya':30s} -> ERROR: {e}")

        for dx in (dx_a, dx_b):
            if dx in fc.index:
                print(f"  [gemma4 cached BT rank] {dx}: bt_rank={fc.loc[dx, 'bt_rank']}, "
                      f"observed_rank={fc.loc[dx, 'observed_rank']}")
        rows.append(row)

    print("\nSummary:")
    print(pd.DataFrame(rows).to_string(index=False))


# -----------------------------------------------------------------------
# Notable queue-shift cases, pulled from the pinned 250-case list and the
# cached gemma4 no-tools run:
#   - a "tiny" bucket case (simplest decision)
#   - a "huge" bucket case (hardest, most room for disagreement)
#   - a case gemma4 actually got wrong (FAIL verdict) in the cached run
# -----------------------------------------------------------------------
NOTABLE_QUEUE_LABELS = [
    "tiny (0-1 waiting) / random-1",
    "huge (8+ waiting) / random-1",
    "small (2-3 waiting) / random-12",  # gemma4 FAIL in cached run: correct=4, model said 3
]


def run_queue_shift_spotcheck():
    print("\n" + "=" * 90)
    print("PART B -- CT queue insertion position: notable cases, gemma4 vs TwiL-LM3")
    print("(Laya skipped: this task is multi-item sequential reasoning, not single-text classification)")
    print("=" * 90)

    with open(CASE_LIST_PATH) as f:
        pinned = {entry["label"]: entry for entry in json.load(f)}

    print("Loading analysis/ct_queue_reconstruction.xlsx ...")
    q = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")
    nni = q[(q["site"] == "NNI") & q["data_quality_flag"].isna()].copy()

    print("Loading merged/ct_orders_cleaned.xlsx ...")
    img = pd.read_excel(os.path.join(BASE, "merged", "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
    nni = nni.merge(img[["Order ID", "Ordering Question Responses"]], on="Order ID", how="left")
    nni["priority_tier"] = nni["Ordering Question Responses"].apply(pa.parse_priority)
    nni["priority_score"] = nni["priority_tier"].map(pa.TIER_RANK)

    print("Loading analysis/ct_scan_time_by_type.xlsx ...")
    scan_time_by_type = pd.read_excel(
        os.path.join(ANALYSIS, "ct_scan_time_by_type.xlsx"), sheet_name="NNI_ScanTime_By_Type_Merged"
    )
    median_scan_time_min = round(
        (scan_time_by_type["N"] * scan_time_by_type["Median"]).sum() / scan_time_by_type["N"].sum(), 1
    )

    candidates = nni[
        nni["parent_order_dt"].notna()
        & nni["exam_start_dt"].notna()
        & (nni["wait_to_start_min"] >= 0)
        & nni["queue_position_at_arrival"].notna()
    ].copy()
    candidates_by_id = candidates.set_index(candidates["Order ID"].astype(str), drop=False)

    rows = []
    for label in NOTABLE_QUEUE_LABELS:
        entry = pinned[label]
        order_id = entry["order_id"]
        pick = candidates_by_id.loc[order_id]
        if isinstance(pick, pd.DataFrame):
            pick = pick.iloc[0]

        snapshot = pa.build_snapshot(nni, pick)
        correct_position = 1 + int((snapshot["priority_score"] <= pick["priority_score"]).sum())
        system_prompt, user_prompt = pa.build_prompt_no_tools(snapshot, pick, median_scan_time_min)

        print(f"\n--- {label}  (bucket={entry['queue_length_bucket']}, order_id={order_id}, "
              f"n_waiting={len(snapshot)}, correct_position={correct_position}) ---")

        row = {"label": label, "n_waiting": len(snapshot), "correct_position": correct_position}
        for model in CHAT_MODELS:
            result = ollama_chat(model, system_prompt, user_prompt)
            content = (result.get("message", {}) or {}).get("content", "")
            model_position = pa.extract_stated_position(content)
            verdict = pa.grade(model_position, correct_position)
            row[model] = f"{model_position} ({verdict})"
            print(f"  {model:30s} -> position={model_position}  [{verdict}]")
        rows.append(row)

    print("\nSummary:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run_ed_dx_spotcheck()
    run_queue_shift_spotcheck()
