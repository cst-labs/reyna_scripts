"""
nni_llm_queue_slot_tests_tools.py
Same 6 real historical NNI queue-slotting test cases as
nni_llm_queue_slot_tests.py, but instead of letting qwen3:1.7b reason about
priority/wait-time comparisons itself in free text, the model is given two
Ollama tool-calling functions -- compare_priority(item_a, item_b) and
compare_wait_time(item_a, item_b) -- and must call them to determine
ordering. Each tool call is executed locally against the real snapshot data
(not the model's own transcribed numbers), so every individual comparison
the model relies on is guaranteed correct; this isolates whether giving the
model deterministic comparison primitives fixes the FCFS-direction mistake
found in the no-tools run (nni_llm_queue_slot_tests.py: 1/6 pass), rather
than testing the model's arithmetic.

Test-case selection, snapshot construction, and grading are identical to
nni_llm_queue_slot_tests.py (same comparator: priority tier, then earliest
arrival) so the two runs are directly comparable.

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders -- site, parent_order_dt,
                                           exam_start_dt, queue_position_at_arrival)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)
  analysis/ct_scan_time_by_type.xlsx     (NNI_ScanTime_By_Type_Merged --
                                           already-computed per-type medians
                                           of time_to_next_start_min)

Calls: local Ollama REST API (http://localhost:11434/api/chat), model
qwen3:1.7b, with a `tools` schema for compare_priority/compare_wait_time.
No new pip dependencies -- uses stdlib urllib.

Outputs: analysis/nni_llm_queue_slot_tests_tools.xlsx (Summary + per-case
detail, including the full tool-calling transcript) -- in addition to
printing each case's transcript/answer/verdict and a final pass-rate
summary table to the console.
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

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e4b"
MODEL_SLUG = OLLAMA_MODEL.replace(":", "-")
N_TARGET_CASES = 50
MAX_TOOL_ROUNDS = 10

TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
PRIORITY_PATTERN = re.compile(r"(?:ED )?Scan Priority:\s*(P[1-3])")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compare_priority",
            "description": (
                "Compares two queue items by priority tier ONLY (ignoring wait time). "
                "Returns which one should be scanned first based on priority alone -- "
                "P1 beats P2 beats P3 beats Unknown. If both items have the same "
                "priority tier, returns a tie -- then use compare_wait_time to break it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_a": {"type": "string", "description": "label of the first item, e.g. 'item_3' or 'new_order'"},
                    "item_b": {"type": "string", "description": "label of the second item, e.g. 'item_7' or 'new_order'"},
                },
                "required": ["item_a", "item_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_wait_time",
            "description": (
                "Compares two queue items by how long they have waited so far. Returns "
                "which one should be scanned first under first-come-first-served -- the "
                "item that has waited LONGER goes first. Only meaningful when both items "
                "have the same priority tier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_a": {"type": "string", "description": "label of the first item, e.g. 'item_3' or 'new_order'"},
                    "item_b": {"type": "string", "description": "label of the second item, e.g. 'item_7' or 'new_order'"},
                },
                "required": ["item_a", "item_b"],
            },
        },
    },
]


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
    """Same best-effort extraction as nni_llm_queue_slot_tests.py."""
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


def build_item_lookup(snapshot, new_order):
    lookup = {}
    for i, r in enumerate(snapshot.itertuples(), start=1):
        lookup[f"item_{i}"] = {"priority_score": r.priority_score, "priority_tier": r.priority_tier,
                                "waiting_so_far_min": r.waiting_so_far_min}
    lookup["new_order"] = {"priority_score": new_order["priority_score"], "priority_tier": new_order["priority_tier"],
                            "waiting_so_far_min": 0.0}
    return lookup


def normalize_label(raw):
    s = str(raw).strip().lower().replace(" ", "_")
    if "new" in s:
        return "new_order"
    m = re.search(r"(\d+)", s)
    if m:
        return f"item_{m.group(1)}"
    return s


def execute_tool(name, args, item_lookup):
    a = normalize_label(args.get("item_a", ""))
    b = normalize_label(args.get("item_b", ""))
    if a not in item_lookup or b not in item_lookup:
        return {"error": f"unknown item label(s): {args.get('item_a')!r}, {args.get('item_b')!r}. "
                          f"Valid labels: {sorted(item_lookup)}"}
    pa, pb = item_lookup[a], item_lookup[b]
    if name == "compare_priority":
        if pa["priority_score"] < pb["priority_score"]:
            result = a
        elif pb["priority_score"] < pa["priority_score"]:
            result = b
        else:
            result = "tie"
        return {"item_a": a, "priority_tier_a": pa["priority_tier"],
                "item_b": b, "priority_tier_b": pb["priority_tier"], "result": result}
    elif name == "compare_wait_time":
        if pa["waiting_so_far_min"] > pb["waiting_so_far_min"]:
            result = a
        elif pb["waiting_so_far_min"] > pa["waiting_so_far_min"]:
            result = b
        else:
            result = "tie"
        return {"item_a": a, "waited_min_a": pa["waiting_so_far_min"],
                "item_b": b, "waited_min_b": pb["waiting_so_far_min"], "result": result}
    return {"error": f"unknown tool {name}"}


def build_tool_prompt(snapshot, new_order, median_scan_time_min):
    lines = [
        "You are scheduling scans for a hospital CT scanner queue.",
        "Policy: scans are served by priority first -- P1 is most urgent, then P2, then "
        "P3, then Unknown priority last. Within the same priority, first-come-first-"
        "served: whoever has been waiting longer goes first.",
        "You have two tools available: compare_priority(item_a, item_b) and "
        "compare_wait_time(item_a, item_b). Each takes two item labels and tells you "
        "which one should be scanned first by that one criterion. Use compare_priority "
        "first; if it returns a tie, use compare_wait_time to break it. Do NOT try to "
        "compare the numbers yourself -- always call a tool to make the comparison.",
        "",
        "Current queue, labelled item_1 (arrived first) through "
        f"item_{len(snapshot)} (arrived most recently):",
    ]
    for i, r in enumerate(snapshot.itertuples(), start=1):
        lines.append(
            f"item_{i}: priority {r.priority_tier}, waited {r.waiting_so_far_min} min so far, "
            f"median scan time {median_scan_time_min} min"
        )
    lines += [
        "",
        f"new_order: priority {new_order['priority_tier']}, waited 0 min so far, "
        f"median scan time {median_scan_time_min} min",
        "",
        "Use the tools to work out where new_order belongs in the queue (position 1 = "
        "scanned next, before everyone else). When you are done, reply with plain text "
        "in the form 'Position: N' and a short reason -- do not call any more tools "
        "once you know the answer.",
    ]
    return "\n".join(lines)


def run_tool_calling_case(prompt, item_lookup, max_rounds=MAX_TOOL_ROUNDS):
    messages = [{"role": "user", "content": prompt}]
    transcript = [f"[USER]\n{prompt}"]
    hit_cap = False
    content = ""

    for round_i in range(1, max_rounds + 1):
        payload = json.dumps({
            "model": OLLAMA_MODEL, "messages": messages, "tools": TOOLS, "stream": False,
        }).encode("utf-8")
        result = None
        for attempt in range(1, 3):
            request = urllib.request.Request(
                OLLAMA_CHAT_URL, data=payload, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt > 1:
                    raise SystemExit(
                        f"Could not reach Ollama at {OLLAMA_CHAT_URL} after 2 attempts ({e}). "
                        f"Is the Ollama daemon running? Try `ollama serve`."
                    )
                print(f"  (round {round_i} attempt {attempt} timed out/failed: {e}; retrying...)")

        msg = result.get("message", {})
        content = msg.get("content", "") or ""
        thinking = msg.get("thinking", "") or ""
        tool_calls = msg.get("tool_calls") or []
        transcript.append(
            f"[ASSISTANT round {round_i}] thinking: {thinking}\ncontent: {content}\n"
            f"tool_calls: {json.dumps(tool_calls)}"
        )
        messages.append(msg)

        if not tool_calls:
            return content, "\n\n".join(transcript), round_i, False

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments", {}) or {}
            tool_result = execute_tool(name, args, item_lookup)
            transcript.append(f"[TOOL RESULT: {name}({args})] {json.dumps(tool_result)}")
            messages.append({"role": "tool", "content": json.dumps(tool_result)})

    hit_cap = True
    return content, "\n\n".join(transcript), max_rounds, hit_cap


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
# Select test cases -- identical selection logic to nni_llm_queue_slot_tests.py
# for direct comparability: stratify across queue-length buckets (how many
# real orders were already waiting when the new order arrived), and within
# each bucket spread evenly across priority tiers -- so the sample covers
# short queues too, not just the largest ones.
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
# Run each case through the tool-calling loop
# ---------------------------------------------------------------------------
results = []
for i, (label, queue_length_bucket, new_order) in enumerate(cases, start=1):
    print("=" * 78)
    print(f"CASE {i} ({label}): new order {new_order['Order ID']} arriving at "
          f"{new_order['parent_order_dt']}, priority {new_order['priority_tier']}")

    snapshot = build_snapshot(nni, new_order)
    correct_position = 1 + int((snapshot["priority_score"] <= new_order["priority_score"]).sum())
    item_lookup = build_item_lookup(snapshot, new_order)

    print(f"Queue snapshot: {len(snapshot)} order(s) already waiting")
    print(f"Deterministic policy-correct position: {correct_position}")

    prompt = build_tool_prompt(snapshot, new_order, median_scan_time_min)
    print("\n--- Prompt sent to qwen3:1.7b (with tools) ---")
    print(prompt)

    print(f"\nCalling Ollama ({OLLAMA_MODEL}) with tool-calling loop (max {MAX_TOOL_ROUNDS} rounds) ...")
    final_answer, transcript, n_rounds, hit_cap = run_tool_calling_case(prompt, item_lookup)
    print(f"\n--- Transcript ({n_rounds} round(s), hit_cap={hit_cap}) ---")
    print(transcript)
    print("\n--- Model's final answer ---")
    print(final_answer)

    if hit_cap:
        verdict = "INCOMPLETE"
        model_position = None
    else:
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
        "n_rounds": n_rounds,
        "hit_round_cap": hit_cap,
        "verdict": verdict,
        "order_id": new_order["Order ID"],
        "prompt": prompt,
        "transcript": transcript,
        "final_answer": final_answer,
    })

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SUMMARY")
all_results = pd.DataFrame(results)
summary = all_results.drop(columns=["order_id", "prompt", "transcript", "final_answer"])
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
out_path = os.path.join(ANALYSIS, f"nni_llm_queue_slot_tests_tools_{MODEL_SLUG}.xlsx")
print(f"\nWriting {out_path} ...")
notes = pd.DataFrame([
    {"Note": "Approach: the model is given two Ollama tool-calling functions, "
             "compare_priority(item_a, item_b) and compare_wait_time(item_a, item_b), and "
             "is instructed to call them instead of comparing priority/wait-time numbers itself."},
    {"Note": "Tool calls are executed locally against the real snapshot data (looked up by item "
             "label, e.g. 'item_3' or 'new_order') -- every individual comparison the model relies "
             "on is guaranteed correct; this isolates whether tool use fixes the model's own "
             "reasoning about ordering, not its ability to transcribe/compute numbers."},
    {"Note": f"Model: {OLLAMA_MODEL} via Ollama /api/chat. Max {MAX_TOOL_ROUNDS} tool-calling rounds "
             "per case; hit_round_cap=True means the model never settled on a final content-only "
             "answer within that budget (verdict INCOMPLETE)."},
    {"Note": "Same test cases and same deterministic correct_position (priority tier, then earliest "
             "arrival) as nni_llm_queue_slot_tests.py, for direct comparison against the no-tools run."},
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
