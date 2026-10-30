"""
nni_llm_queue_slot_tests.py
Runs several real historical NNI queue-slotting scenarios (varying queue
size and priority mix) through qwen3:1.7b via Ollama, same idea as
nni_llm_queue_slot.py's single worked example, and auto-grades each answer
against the deterministic policy-correct slot: priority first (P1 > P2 >
P3 > Unknown), then first-come-first-served within the same tier -- the
same comparator ct_queue_priority_policy_check.py uses to pick who's served
next (min by (tier_rank, parent_order_dt)), applied here to rank where a
new arrival would insert.

Test-case selection: stratified across 5 real queue-length buckets (tiny
0-1, small 2-3, medium 4-5, large 6-7, huge 8+ orders already waiting,
using queue_position_at_arrival already computed by build_ct_queue.py's
waiting-line sweep) x the 4 priority tiers (P1, P2, P3, Unknown), so the
~50 cases deliberately span short and long real queues rather than only
the largest ones.

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders -- site, parent_order_dt,
                                           exam_start_dt, queue_position_at_arrival)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)
  analysis/ct_scan_time_by_type.xlsx     (NNI_ScanTime_By_Type_Merged --
                                           already-computed per-type medians
                                           of time_to_next_start_min)

Calls: local Ollama REST API (http://localhost:11434/api/generate), model
qwen3:1.7b. No new pip dependencies -- uses stdlib urllib.

Outputs: analysis/nni_llm_queue_slot_tests.xlsx (Summary + per-case detail,
including the exact prompt and full raw response) -- in addition to
printing each case's prompt/answer/verdict and the final pass-rate summary
table to the console.
"""
import json
import os
import re
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
ANALYSIS = os.path.join(BASE, "analysis")
os.makedirs(ANALYSIS, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:e4b"
MODEL_SLUG = OLLAMA_MODEL.replace(":", "-")
N_TARGET_CASES = 50

TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
PRIORITY_PATTERN = re.compile(r"(?:ED )?Scan Priority:\s*(P[1-3])")


def is_blank(v):
    if v is None:
        return True
    try:
        if isinstance(v, float) and np.isnan(v):
            return True
    except Exception:
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def parse_priority(val):
    if is_blank(val):
        return "Unknown"
    m = PRIORITY_PATTERN.search(str(val))
    return m.group(1) if m else "Unknown"


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 60)


def extract_stated_position(text):
    """Best-effort extraction of the position number the model settled on --
    prefers a number attached to the word 'position', falling back to a
    number attached to 'answer'. Returns None if nothing matches."""
    cleaned = re.sub(r"[*_`#]", "", text)
    matches = re.findall(r"position\s*(?:number)?\s*(?:is|:|-|–)?\s*(\d+)", cleaned, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"answer\s*(?:is|:|-|–)?\s*(?:position\s*)?(\d+)", cleaned, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    return None


def build_snapshot(nni, new_order):
    snapshot_time = new_order["parent_order_dt"]
    snapshot = nni[
        (nni["parent_order_dt"] < snapshot_time)
        & (nni["exam_start_dt"] > snapshot_time)
        & (nni["Order ID"] != new_order["Order ID"])
    ].copy()
    snapshot["waiting_so_far_min"] = round(
        (snapshot_time - snapshot["parent_order_dt"]).dt.total_seconds() / 60, 1
    )
    return snapshot.sort_values("parent_order_dt").reset_index(drop=True)


def build_prompt(snapshot, new_order, median_scan_time_min):
    lines = [
        "You are scheduling scans for a hospital CT scanner queue.",
        "Policy: scans are served by priority first -- P1 is most urgent, "
        "then P2, then P3, then Unknown priority last. Within the same "
        "priority, first-come-first-served: whoever has been waiting longer "
        "goes first.",
        "Priority score: P1=0 (most urgent), P2=1, P3=2, Unknown=3 (least "
        "urgent). Lower score means more urgent.",
        "",
        "Current queue, in the order these patients arrived:",
    ]
    for i, r in enumerate(snapshot.itertuples(), start=1):
        lines.append(
            f"Item {i}: priority {r.priority_tier} (score {r.priority_score}), "
            f"waited {r.waiting_so_far_min} min so far, median scan time {median_scan_time_min} min"
        )
    lines += [
        "",
        "A new order has just arrived:",
        f"New order: priority {new_order['priority_tier']} (score {new_order['priority_score']}), "
        f"waited 0 min so far, median scan time {median_scan_time_min} min",
        "",
        "At what position should this new order be slotted into the queue "
        "(position 1 = scanned next, before everyone else)? Answer with the "
        "position number and a short reason.",
    ]
    return "\n".join(lines)


def call_ollama(prompt, retries=2):
    payload = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
    for attempt in range(1, retries + 2):
        request = urllib.request.Request(
            OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt > retries:
                raise SystemExit(
                    f"Could not reach Ollama at {OLLAMA_URL} after {retries + 1} attempts ({e}). "
                    f"Is the Ollama daemon running? Try `ollama serve`."
                )
            print(f"  (attempt {attempt} timed out/failed: {e}; retrying...)")
    raw_response = result.get("response", "")
    final_answer = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
    return raw_response, final_answer


# ---------------------------------------------------------------------------
# Load & join: NNI orders + priority tier
# ---------------------------------------------------------------------------
print("Loading analysis/ct_queue_reconstruction.xlsx ...")
q = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")
nni = q[(q["site"] == "NNI") & q["data_quality_flag"].isna()].copy()

print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
nni = nni.merge(img[["Order ID", "Ordering Question Responses"]], on="Order ID", how="left")

nni["priority_tier"] = nni["Ordering Question Responses"].apply(parse_priority)
nni["priority_score"] = nni["priority_tier"].map(TIER_RANK)

# ---------------------------------------------------------------------------
# Median scan time, combined across CT types -- reuse the per-type list
# already computed by ct_scan_time_by_type.py.
# ---------------------------------------------------------------------------
print("Loading analysis/ct_scan_time_by_type.xlsx ...")
scan_time_by_type = pd.read_excel(
    os.path.join(ANALYSIS, "ct_scan_time_by_type.xlsx"), sheet_name="NNI_ScanTime_By_Type_Merged"
)
median_scan_time_min = round(
    (scan_time_by_type["N"] * scan_time_by_type["Median"]).sum() / scan_time_by_type["N"].sum(), 1
)
print(f"  N-weighted average of per-type medians (time_to_next_start_min): {median_scan_time_min} min\n")

# ---------------------------------------------------------------------------
# Select test cases: stratify across queue-length buckets (how many real
# orders were already waiting when the new order arrived), and within each
# bucket spread evenly across priority tiers -- so unlike a "largest queue
# only" sample, this deliberately covers short queues too.
# ---------------------------------------------------------------------------
candidates = nni[
    nni["parent_order_dt"].notna()
    & nni["exam_start_dt"].notna()
    & (nni["wait_to_start_min"] >= 0)
    & nni["queue_position_at_arrival"].notna()
].copy()

QUEUE_LENGTH_BUCKETS = [
    ("tiny (0-1 waiting)", 0, 1),
    ("small (2-3 waiting)", 2, 3),
    ("medium (4-5 waiting)", 4, 5),
    ("large (6-7 waiting)", 6, 7),
    ("huge (8+ waiting)", 8, None),
]

n_per_bucket = -(-N_TARGET_CASES // len(QUEUE_LENGTH_BUCKETS))  # ceil division
n_per_tier_per_bucket = -(-n_per_bucket // len(TIER_ORDER))  # ceil division

selected_order_ids = []
cases = []
for bucket_name, lo, hi in QUEUE_LENGTH_BUCKETS:
    if hi is None:
        bucket_pool = candidates[candidates["queue_position_at_arrival"] >= lo]
    else:
        bucket_pool = candidates[
            (candidates["queue_position_at_arrival"] >= lo) & (candidates["queue_position_at_arrival"] <= hi)
        ]

    bucket_cases = []
    bucket_selected_ids = []
    for tier in TIER_ORDER:
        tier_pool = bucket_pool[
            (bucket_pool["priority_tier"] == tier) & (~bucket_pool["Order ID"].isin(selected_order_ids))
        ].sort_values(["queue_position_at_arrival", "parent_order_dt"], ascending=[False, True])
        for rank, (_, pick) in enumerate(tier_pool.head(n_per_tier_per_bucket).iterrows(), start=1):
            bucket_selected_ids.append(pick["Order ID"])
            bucket_cases.append((f"{bucket_name} / top-{rank} queue for new {tier}", bucket_name, pick))

    remaining_in_bucket = n_per_bucket - len(bucket_cases)
    if remaining_in_bucket > 0:
        extra_bucket_pool = bucket_pool[
            ~bucket_pool["Order ID"].isin(selected_order_ids + bucket_selected_ids)
        ].sort_values(["queue_position_at_arrival", "parent_order_dt"], ascending=[False, True])
        for _, pick in extra_bucket_pool.head(remaining_in_bucket).iterrows():
            bucket_selected_ids.append(pick["Order ID"])
            bucket_cases.append((f"{bucket_name} / generic case", bucket_name, pick))

    selected_order_ids.extend(bucket_selected_ids)
    cases.extend(bucket_cases)

remaining_needed = N_TARGET_CASES - len(cases)
if remaining_needed > 0:
    extra_pool = candidates[~candidates["Order ID"].isin(selected_order_ids)].sort_values(
        ["queue_position_at_arrival", "parent_order_dt"], ascending=[False, True]
    )
    for _, pick in extra_pool.head(remaining_needed).iterrows():
        selected_order_ids.append(pick["Order ID"])
        cases.append(("generic case (any bucket/tier)", "unbucketed", pick))

cases = cases[:N_TARGET_CASES]

print(f"Selected {len(cases)} test case(s)\n")
print(pd.Series([b for _, b, _ in cases]).value_counts().to_string())
print()

# ---------------------------------------------------------------------------
# Run each case
# ---------------------------------------------------------------------------
results = []
for i, (label, queue_length_bucket, new_order) in enumerate(cases, start=1):
    print("=" * 78)
    print(f"CASE {i} ({label}): new order {new_order['Order ID']} arriving at "
          f"{new_order['parent_order_dt']}, priority {new_order['priority_tier']}")

    snapshot = build_snapshot(nni, new_order)
    correct_position = 1 + int((snapshot["priority_score"] <= new_order["priority_score"]).sum())

    print(f"Queue snapshot: {len(snapshot)} order(s) already waiting")
    print(
        snapshot[["Order ID", "priority_tier", "priority_score", "waiting_so_far_min"]]
        .to_string(index=False)
    )
    print(f"Deterministic policy-correct position: {correct_position}")

    prompt = build_prompt(snapshot, new_order, median_scan_time_min)
    print("\n--- Prompt sent to qwen3:1.7b ---")
    print(prompt)

    print(f"\nCalling Ollama ({OLLAMA_MODEL}) ...")
    raw_response, final_answer = call_ollama(prompt)
    print("\n--- Model's final answer (reasoning stripped) ---")
    print(final_answer)

    model_position = extract_stated_position(final_answer)
    if model_position is None:
        verdict = "UNPARSEABLE"
    elif model_position == correct_position:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    print(f"\nModel's stated position: {model_position} -> {verdict}")

    results.append({
        "case": i,
        "label": label,
        "queue_length_bucket": queue_length_bucket,
        "new_order_priority": new_order["priority_tier"],
        "n_waiting": len(snapshot),
        "correct_position": correct_position,
        "model_position": model_position,
        "verdict": verdict,
        "order_id": new_order["Order ID"],
        "prompt": prompt,
        "raw_response": raw_response,
        "final_answer": final_answer,
    })

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SUMMARY")
all_results = pd.DataFrame(results)
summary = all_results.drop(columns=["order_id", "prompt", "raw_response", "final_answer"])
print(summary.to_string(index=False))
n_pass = int((summary["verdict"] == "PASS").sum())
print(f"\nPass rate: {n_pass}/{len(summary)} ({n_pass / len(summary):.0%})")

by_length = summary.groupby("queue_length_bucket").agg(
    n_cases=("verdict", "size"),
    n_pass=("verdict", lambda s: (s == "PASS").sum()),
)
by_length["pass_rate"] = by_length["n_pass"] / by_length["n_cases"]
bucket_order = [b for b, _, _ in QUEUE_LENGTH_BUCKETS if b in by_length.index] + [
    b for b in by_length.index if b not in [x[0] for x in QUEUE_LENGTH_BUCKETS]
]
by_length = by_length.loc[bucket_order].reset_index()
print("\nPass rate by queue-length bucket:")
print(by_length.to_string(index=False))

# ---------------------------------------------------------------------------
# Record output
# ---------------------------------------------------------------------------
out_path = os.path.join(ANALYSIS, f"nni_llm_queue_slot_tests_{MODEL_SLUG}.xlsx")
print(f"\nWriting {out_path} ...")
notes = pd.DataFrame([
    {"Note": "Approach: no tools -- the model reasons about priority and wait-time comparisons "
             "itself in free text, from the queue listing given in the prompt."},
    {"Note": f"Model: {OLLAMA_MODEL} via Ollama /api/generate. correct_position is the deterministic "
             "policy-optimal slot (priority tier, then earliest arrival) computed in code, not by the model."},
    {"Note": "model_position is parsed from the model's final answer text via extract_stated_position(); "
             "UNPARSEABLE means no position number could be found in that text."},
    {"Note": "Test cases are stratified across 5 real queue-length buckets (queue_length_bucket, based on "
             "queue_position_at_arrival: how many orders were already waiting) x the 4 priority tiers, so "
             "results can be compared by how long the real queue was, not just by priority mix."},
])
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    notes.to_excel(writer, sheet_name="Notes", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)
    by_length.to_excel(writer, sheet_name="Summary_by_length", index=False)
    all_results.to_excel(writer, sheet_name="Cases", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)
print("Done.")
