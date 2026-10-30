"""
nni_llm_queue_slot_tests_huge_supplement.py
Supplements the "huge (8+ waiting)" queue-length bucket -- which only had 2
real cases in the original 50-case length-stratified runs -- with 10
additional real historical NNI cases, and re-runs all four existing
experiment variants (qwen3:1.7b / gemma4:e4b, each no-tools and tool-
calling) on just these new cases. Results are merged into the four existing
.xlsx outputs as additional case rows (Summary and Summary_by_length
recomputed over the combined set); the original case rows are left as-is.

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders -- site, parent_order_dt,
                                           exam_start_dt, queue_position_at_arrival)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)
  analysis/ct_scan_time_by_type.xlsx     (NNI_ScanTime_By_Type_Merged)
  analysis/nni_llm_queue_slot_tests.xlsx                  (existing qwen3:1.7b no-tools
                                                            results -- used to find which
                                                            huge-bucket order IDs are
                                                            already tested, and merged into)
  analysis/nni_llm_queue_slot_tests_tools.xlsx             (existing qwen3:1.7b tools results)
  analysis/nni_llm_queue_slot_tests_gemma4-e4b.xlsx        (existing gemma4:e4b no-tools results)
  analysis/nni_llm_queue_slot_tests_tools_gemma4-e4b.xlsx  (existing gemma4:e4b tools results)

Calls: local Ollama REST API, models qwen3:1.7b and gemma4:e4b, one model at
a time (no-tools then tools for that model) to avoid request contention.

Outputs: rewrites the four .xlsx files above with 10 additional huge-bucket
case rows appended and Summary/Summary_by_length recomputed over the full
(original + supplement) case set.
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

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MAX_TOOL_ROUNDS = 10
N_SUPPLEMENT = 10
HUGE_BUCKET_NAME = "huge (8+ waiting)"

TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
PRIORITY_PATTERN = re.compile(r"(?:ED )?Scan Priority:\s*(P[1-3])")

QUEUE_LENGTH_BUCKETS = [
    ("tiny (0-1 waiting)", 0, 1),
    ("small (2-3 waiting)", 2, 3),
    ("medium (4-5 waiting)", 4, 5),
    ("large (6-7 waiting)", 6, 7),
    (HUGE_BUCKET_NAME, 8, None),
]

TARGETS = [
    ("qwen3:1.7b", False, os.path.join(ANALYSIS, "nni_llm_queue_slot_tests.xlsx")),
    ("qwen3:1.7b", True, os.path.join(ANALYSIS, "nni_llm_queue_slot_tests_tools.xlsx")),
    ("gemma4:e4b", False, os.path.join(ANALYSIS, "nni_llm_queue_slot_tests_gemma4-e4b.xlsx")),
    ("gemma4:e4b", True, os.path.join(ANALYSIS, "nni_llm_queue_slot_tests_tools_gemma4-e4b.xlsx")),
]

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


def call_ollama(prompt, model, retries=2):
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
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


def run_tool_calling_case(prompt, item_lookup, model, max_rounds=MAX_TOOL_ROUNDS):
    messages = [{"role": "user", "content": prompt}]
    transcript = [f"[USER]\n{prompt}"]
    content = ""

    for round_i in range(1, max_rounds + 1):
        payload = json.dumps({
            "model": model, "messages": messages, "tools": TOOLS, "stream": False,
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

    return content, "\n\n".join(transcript), max_rounds, True


# ---------------------------------------------------------------------------
# Load & join: NNI orders + priority tier (same as the original scripts)
# ---------------------------------------------------------------------------
print("Loading analysis/ct_queue_reconstruction.xlsx ...")
q = pd.read_excel(os.path.join(ANALYSIS, "ct_queue_reconstruction.xlsx"), sheet_name="CT_Orders")
nni = q[(q["site"] == "NNI") & q["data_quality_flag"].isna()].copy()

print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
nni = nni.merge(img[["Order ID", "Ordering Question Responses"]], on="Order ID", how="left")

nni["priority_tier"] = nni["Ordering Question Responses"].apply(parse_priority)
nni["priority_score"] = nni["priority_tier"].map(TIER_RANK)

print("Loading analysis/ct_scan_time_by_type.xlsx ...")
scan_time_by_type = pd.read_excel(
    os.path.join(ANALYSIS, "ct_scan_time_by_type.xlsx"), sheet_name="NNI_ScanTime_By_Type_Merged"
)
median_scan_time_min = round(
    (scan_time_by_type["N"] * scan_time_by_type["Median"]).sum() / scan_time_by_type["N"].sum(), 1
)
print(f"  N-weighted average of per-type medians (time_to_next_start_min): {median_scan_time_min} min\n")

candidates = nni[
    nni["parent_order_dt"].notna()
    & nni["exam_start_dt"].notna()
    & (nni["wait_to_start_min"] >= 0)
    & nni["queue_position_at_arrival"].notna()
].copy()

huge_pool = candidates[candidates["queue_position_at_arrival"] >= 8]

# ---------------------------------------------------------------------------
# Find which huge-bucket order IDs are already tested (read from the
# existing qwen3:1.7b no-tools results -- case selection is deterministic
# and model-independent, so all four existing outputs share the same
# huge-bucket order IDs).
# ---------------------------------------------------------------------------
already_used_order_ids = set()
for _, _, path in TARGETS:
    if os.path.exists(path):
        try:
            existing_cases = pd.read_excel(path, sheet_name="Cases")
            huge_existing = existing_cases[existing_cases["queue_length_bucket"] == HUGE_BUCKET_NAME]
            already_used_order_ids.update(huge_existing["order_id"].tolist())
        except Exception as e:
            print(f"  (could not read existing cases from {path}: {e})")
print(f"Already-tested huge-bucket order IDs ({len(already_used_order_ids)}): {sorted(already_used_order_ids)}\n")

new_pool = huge_pool[~huge_pool["Order ID"].isin(already_used_order_ids)]

n_per_tier = -(-N_SUPPLEMENT // len(TIER_ORDER))  # ceil division

selected_ids = []
new_cases = []
for tier in TIER_ORDER:
    tier_pool = new_pool[
        (new_pool["priority_tier"] == tier) & (~new_pool["Order ID"].isin(selected_ids))
    ].sort_values(["queue_position_at_arrival", "parent_order_dt"], ascending=[False, True])
    for rank, (_, pick) in enumerate(tier_pool.head(n_per_tier).iterrows(), start=1):
        selected_ids.append(pick["Order ID"])
        new_cases.append((f"{HUGE_BUCKET_NAME} / supplement-{rank} queue for new {tier}", pick))

remaining_needed = N_SUPPLEMENT - len(new_cases)
if remaining_needed > 0:
    extra_pool = new_pool[~new_pool["Order ID"].isin(selected_ids)].sort_values(
        ["queue_position_at_arrival", "parent_order_dt"], ascending=[False, True]
    )
    for _, pick in extra_pool.head(remaining_needed).iterrows():
        selected_ids.append(pick["Order ID"])
        new_cases.append((f"{HUGE_BUCKET_NAME} / supplement generic case", pick))

new_cases = new_cases[:N_SUPPLEMENT]
print(f"Selected {len(new_cases)} supplemental huge-bucket case(s)\n")
for label, pick in new_cases:
    print(f"  {label}: order {pick['Order ID']}, queue_position_at_arrival={pick['queue_position_at_arrival']}")
print()

# Precompute snapshot / correct_position / prompts / item_lookups once --
# reused identically across all four model/mode combos.
prepared = []
for label, new_order in new_cases:
    snapshot = build_snapshot(nni, new_order)
    correct_position = 1 + int((snapshot["priority_score"] <= new_order["priority_score"]).sum())
    prepared.append({
        "label": label,
        "new_order": new_order,
        "snapshot": snapshot,
        "correct_position": correct_position,
        "n_waiting": len(snapshot),
        "no_tools_prompt": build_prompt(snapshot, new_order, median_scan_time_min),
        "tool_prompt": build_tool_prompt(snapshot, new_order, median_scan_time_min),
        "item_lookup": build_item_lookup(snapshot, new_order),
    })


def run_no_tools(model, cases):
    results = []
    for i, c in enumerate(cases, start=1):
        print(f"  [no-tools/{model}] case {i}/{len(cases)}: {c['label']}")
        raw_response, final_answer = call_ollama(c["no_tools_prompt"], model)
        model_position = extract_stated_position(final_answer)
        if model_position is None:
            verdict = "UNPARSEABLE"
        elif model_position == c["correct_position"]:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        results.append({
            "label": c["label"],
            "queue_length_bucket": HUGE_BUCKET_NAME,
            "new_order_priority": c["new_order"]["priority_tier"],
            "n_waiting": c["n_waiting"],
            "correct_position": c["correct_position"],
            "model_position": model_position,
            "verdict": verdict,
            "order_id": c["new_order"]["Order ID"],
            "prompt": c["no_tools_prompt"],
            "raw_response": raw_response,
            "final_answer": final_answer,
        })
        print(f"    -> model_position={model_position}, verdict={verdict}")
    return results


def run_tools(model, cases):
    results = []
    for i, c in enumerate(cases, start=1):
        print(f"  [tools/{model}] case {i}/{len(cases)}: {c['label']}")
        final_answer, transcript, n_rounds, hit_cap = run_tool_calling_case(
            c["tool_prompt"], c["item_lookup"], model
        )
        if hit_cap:
            verdict = "INCOMPLETE"
            model_position = None
        else:
            model_position = extract_stated_position(final_answer)
            if model_position is None:
                verdict = "UNPARSEABLE"
            elif model_position == c["correct_position"]:
                verdict = "PASS"
            else:
                verdict = "FAIL"
        results.append({
            "label": c["label"],
            "queue_length_bucket": HUGE_BUCKET_NAME,
            "new_order_priority": c["new_order"]["priority_tier"],
            "n_waiting": c["n_waiting"],
            "correct_position": c["correct_position"],
            "model_position": model_position,
            "n_rounds": n_rounds,
            "hit_round_cap": hit_cap,
            "verdict": verdict,
            "order_id": c["new_order"]["Order ID"],
            "prompt": c["tool_prompt"],
            "transcript": transcript,
            "final_answer": final_answer,
        })
        print(f"    -> model_position={model_position}, verdict={verdict}, n_rounds={n_rounds}, hit_cap={hit_cap}")
    return results


def merge_into_xlsx(path, new_rows_df):
    existing_cases = pd.read_excel(path, sheet_name="Cases")
    existing_notes = pd.read_excel(path, sheet_name="Notes")

    next_case_num = int(existing_cases["case"].max()) + 1
    new_rows_df = new_rows_df.copy()
    new_rows_df.insert(0, "case", range(next_case_num, next_case_num + len(new_rows_df)))

    combined_cases = pd.concat([existing_cases, new_rows_df], ignore_index=True)

    text_cols = [c for c in ["order_id", "prompt", "raw_response", "transcript", "final_answer"] if c in combined_cases.columns]
    summary = combined_cases.drop(columns=text_cols)

    by_length = summary.groupby("queue_length_bucket").agg(
        n_cases=("verdict", "size"),
        n_pass=("verdict", lambda s: (s == "PASS").sum()),
    )
    by_length["pass_rate"] = by_length["n_pass"] / by_length["n_cases"]
    bucket_order = [b for b, _, _ in QUEUE_LENGTH_BUCKETS if b in by_length.index] + [
        b for b in by_length.index if b not in [x[0] for x in QUEUE_LENGTH_BUCKETS]
    ]
    by_length = by_length.loc[bucket_order].reset_index()

    supplement_note = pd.DataFrame([{
        "Note": f"Supplemented with {len(new_rows_df)} additional real huge-bucket "
                "(queue_position_at_arrival >= 8) cases (nni_llm_queue_slot_tests_huge_supplement.py), "
                f"appended as case {next_case_num}-{next_case_num + len(new_rows_df) - 1}. "
                "Original cases above are unchanged."
    }])
    notes = pd.concat([existing_notes, supplement_note], ignore_index=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="Notes", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        by_length.to_excel(writer, sheet_name="Summary_by_length", index=False)
        combined_cases.to_excel(writer, sheet_name="Cases", index=False)
        for sheet in writer.sheets.values():
            auto_col_width(sheet)

    n_pass = int((summary["verdict"] == "PASS").sum())
    huge_summary = by_length[by_length["queue_length_bucket"] == HUGE_BUCKET_NAME].iloc[0]
    print(f"  Merged into {path}")
    print(f"    Overall: {n_pass}/{len(summary)} ({n_pass/len(summary):.0%})")
    print(f"    Huge bucket now: {int(huge_summary['n_pass'])}/{int(huge_summary['n_cases'])} "
          f"({huge_summary['pass_rate']:.0%})")


# ---------------------------------------------------------------------------
# Run each model (no-tools then tools) and merge into its two output files
# ---------------------------------------------------------------------------
for model in ["qwen3:1.7b", "gemma4:e4b"]:
    print("=" * 78)
    print(f"MODEL: {model}")

    print("\n--- no-tools ---")
    no_tools_results = run_no_tools(model, prepared)
    no_tools_path = next(p for m, tools, p in TARGETS if m == model and not tools)
    merge_into_xlsx(no_tools_path, pd.DataFrame(no_tools_results))

    print("\n--- tools ---")
    tools_results = run_tools(model, prepared)
    tools_path = next(p for m, tools, p in TARGETS if m == model and tools)
    merge_into_xlsx(tools_path, pd.DataFrame(tools_results))

print("\n" + "=" * 78)
print("Done.")
