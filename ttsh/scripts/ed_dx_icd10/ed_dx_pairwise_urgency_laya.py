"""
Full pairwise ED-diagnosis urgency comparisons via Laya (Unsloth Studio's
local Decision API), one repeat per pair -- Laya is a calibrated classifier,
not a sampled chat model, so repeated calls at the same input are not
expected to add signal the way they do for gemma4/TwiL-LM3.

Reuses the diagnosis universe / pair-building / checkpoint schema from
ed_dx_pairwise_urgency_ollama.py so ed_dx_bradley_terry_ranking.py can fit
this output the same way it fits gemma4's.

Checkpointing: same append-only raw.jsonl pattern, resumable.

Outputs:
  analysis/ed_dx_pairwise_urgency_laya/raw.jsonl

Use --rebuild-only to just report progress without calling Laya.
Use --limit-pairs N for a smoke test.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

from ed_dx_pairwise_urgency_ollama import (
    ANALYSIS,
    build_pairs,
    load_records,
    top_ed_dx_mentions,
)

LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8888/v1/systemone")
LAYA_KEY = os.environ.get("LAYA_API_KEY", "sk-unsloth-00ac45fcd24dbbd9d5aa466aec6ecea2")

MODEL = "laya"
PROMPT_VERSION = "ed_dx_pairwise_urgency_v1"
OUT_DIR = os.path.join(ANALYSIS, "ed_dx_pairwise_urgency_laya")
JSONL_PATH = os.path.join(OUT_DIR, "raw.jsonl")
N_REPEATS = int(os.environ.get("N_REPEATS", "1"))

CHOICE_A = "A"
CHOICE_B = "B"
CHOICE_TIE = "Tie/Insufficient information"

URGENCY_QUESTION = {
    "type": "choice",
    "instructions": (
        "In an ED/acute-care setting, which diagnosis usually represents greater "
        "clinical urgency for emergency prioritisation (immediate risk of death, "
        "neurological deterioration, time-critical morbidity, or need for urgent "
        "intervention if delayed)? Do not invent age, vital signs, mechanism "
        "severity, imaging findings, or other missing details -- use only the labels."
    ),
    "criteria": {
        "A": "Diagnosis A is clearly more clinically urgent than Diagnosis B",
        "B": "Diagnosis B is clearly more clinically urgent than Diagnosis A",
        "Tie": "similar urgency, ambiguous, or cannot be distinguished from the labels alone",
    },
}


def laya_call(dx_a, dx_b, retries=2):
    payload = {
        "model": MODEL,
        "state": f"Diagnosis A: {dx_a}\nDiagnosis B: {dx_b}",
        "questions": {"urgency": URGENCY_QUESTION},
    }
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(1, retries + 2):
        request = urllib.request.Request(
            LAYA_URL, data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {LAYA_KEY}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt > retries:
                raise SystemExit(f"Could not reach Laya at {LAYA_URL} after {retries + 1} attempts: {e}")
            print(f"  (attempt {attempt} failed: {e}; retrying...)")
            time.sleep(2)


def run_one(pair, repeat_index):
    body = laya_call(pair["diagnosis_a"], pair["diagnosis_b"])
    answer = body["answers"]["urgency"]
    choice = {"A": CHOICE_A, "B": CHOICE_B, "Tie": CHOICE_TIE}.get(answer.get("choice"), "UNPARSEABLE")
    return {
        "model": MODEL,
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
        "parsed_json": answer,
        "content": json.dumps(answer, ensure_ascii=False),
        "thinking": "",
        "system_prompt": json.dumps(URGENCY_QUESTION, ensure_ascii=False),
        "user_prompt": f"Diagnosis A: {pair['diagnosis_a']}\nDiagnosis B: {pair['diagnosis_b']}",
        "created_at_unix": time.time(),
    }


def append_record(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-only", action="store_true", help="Only report progress, do not call Laya.")
    parser.add_argument("--limit-pairs", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    diagnoses = top_ed_dx_mentions()
    pairs = build_pairs(diagnoses)
    if args.limit_pairs is not None:
        pairs = pairs[: args.limit_pairs]

    records = load_records(JSONL_PATH)
    cached_counts = {}
    for r in records:
        if r.get("model") == MODEL and r.get("prompt_version") == PROMPT_VERSION:
            cached_counts[r["pair_id"]] = cached_counts.get(r["pair_id"], 0) + 1

    total = len(pairs) * N_REPEATS
    done = sum(min(cached_counts.get(p["pair_id"], 0), N_REPEATS) for p in pairs)
    print(f"Model: {MODEL}; pairs: {len(pairs):,}; repeats per pair: {N_REPEATS}; planned calls: {total:,}")
    print(f"Already complete: {done:,}/{total:,}")

    if args.rebuild_only:
        return

    call_i = done
    t0 = time.time()
    for pair in pairs:
        have = cached_counts.get(pair["pair_id"], 0)
        for repeat in range(have + 1, N_REPEATS + 1):
            call_i += 1
            if call_i % 50 == 0 or call_i == 1:
                elapsed = time.time() - t0
                rate = (call_i - done) / elapsed if elapsed > 0 and call_i > done else 0
                eta_min = (total - call_i) / rate / 60 if rate > 0 else float("nan")
                print(f"[{call_i:,}/{total:,}] {pair['diagnosis_a']} vs {pair['diagnosis_b']}  (eta {eta_min:.0f} min)")
            record = run_one(pair, repeat)
            append_record(JSONL_PATH, record)
            cached_counts[pair["pair_id"]] = repeat

    print(f"Done. Wrote up to {total:,} records to {JSONL_PATH}")


if __name__ == "__main__":
    main()
