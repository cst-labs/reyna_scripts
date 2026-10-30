"""
nni_llm_queue_slot_consistency.py
Random-sample consistency + reasoning-logging experiment for the NNI CT
queue-slotting task, across both models (qwen3:1.7b, gemma4:e4b) and both
approaches (no-tools, tool-calling).

Differs from nni_llm_queue_slot_tests.py / _tools.py in three ways:
  1. Case selection is a plain uniform-random sample of 50 real snapshots
     per queue-length bucket (250 total), not stratified by priority tier
     and not biased toward the longest real queue in each combo.
  2. Each snapshot's exact prompt is sent 3 separate times per model/mode,
     to measure self-consistency (does the same prompt get the same
     answer?) in addition to correctness.
  3. Reasoning is captured natively via Ollama's `think` option
     (POST /api/chat, "think": true -> message.thinking) instead of being
     requested inline in the answer text; the prompt now asks for ONLY the
     position number. Every individual tool call (tool-calling mode) is
     logged as its own structured record, not folded into a transcript
     string.

Reads:
  analysis/ct_queue_reconstruction.xlsx  (CT_Orders -- site, parent_order_dt,
                                           exam_start_dt, queue_position_at_arrival)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)
  analysis/ct_scan_time_by_type.xlsx     (NNI_ScanTime_By_Type_Merged --
                                           already-computed per-type medians
                                           of time_to_next_start_min)

Calls: local Ollama REST API (http://localhost:11434/api/chat), models
qwen3:1.7b and gemma4:e4b, with "think": true. No new pip dependencies --
uses stdlib urllib.

Checkpointing: every completed snapshot (3 runs, for the current model/mode)
is appended as one JSON line to the current run's raw.jsonl immediately, so
a killed/interrupted run can be resumed (re-running this script skips any
(model, tool_mode, order_id) already present in that file).

Results are split into two folders under analysis/nni_llm_queue_slot_consistency/
because the prompt design changed (system/user split, sorted queue, explicit
pairwise-slotting instructions) partway through this experiment, so the two
are not directly comparable:
  legacy/   -- runs from the old (arrival-order, single-turn-prompt) design
  current/  -- runs from the current design; this is what new runs write to
The pinned 250-case list (case_list.json, at the top level) is shared by
both -- case selection doesn't depend on prompt wording.

Outputs (under current/):
  raw.jsonl      (checkpoint, source of truth)
  results.xlsx   (assembled from the JSONL: Notes, Summary, Cases, Runs, Tool_Calls)

Set env var SMOKE=1 to run a fast smoke test instead of the full experiment
(both models, both modes, 2 random cases per bucket, 1 repeat each, writing
to current/smoketest_* files instead of the real checkpoint/output).

Set env var OUTPUT_TAG=<tag> to write to current/raw_<tag>.jsonl and
current/results_<tag>.xlsx instead (e.g. for a one-off scoped run), while
still reading the same shared case_list.json.

Set env var OUTPUT_DIR=<name> to write to a brand new sibling folder
analysis/nni_llm_queue_slot_consistency/<name>/raw.jsonl and results.xlsx
(e.g. after hand-editing the prompts again to the point results aren't
comparable to current/ either), while still reading the same shared
case_list.json. Takes priority over OUTPUT_TAG if both are set.

Any single model call whose verdict comes back INCOMPLETE (tools mode: hit
the tool-call round cap) or UNPARSEABLE (no position number found in the
reply) is retried in place, up to MAX_RUN_ATTEMPTS (3) attempts total, before
being checkpointed -- so a case only ends up INCOMPLETE/UNPARSEABLE in the
output if it failed 3 times in a row.

Pass --rebuild-only to just rebuild the xlsx from the existing JSONL without
calling Ollama again (useful to inspect progress mid-run).

Pass --retry-bad to re-run (with the same up-to-3-attempts retry) only the
checkpointed runs whose verdict is still INCOMPLETE or UNPARSEABLE, in
place, leaving PASS/FAIL runs untouched, then rewrite the JSONL and rebuild
the xlsx.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MERGED = os.path.join(BASE, "merged")
ANALYSIS = os.path.join(BASE, "analysis")
EXP_DIR = os.path.join(ANALYSIS, "nni_llm_queue_slot_consistency")
LEGACY_DIR = os.path.join(EXP_DIR, "legacy")
CURRENT_DIR = os.path.join(EXP_DIR, "current")
os.makedirs(LEGACY_DIR, exist_ok=True)
os.makedirs(CURRENT_DIR, exist_ok=True)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MAX_TOOL_ROUNDS = 10
MAX_RUN_ATTEMPTS = 3
RANDOM_SEED = 42


def tool_round_cap(n_waiting):
    """Some models issue one tool call per round while linearly scanning the
    queue, needing up to two calls (compare_priority, then compare_wait_time
    on a tie) per existing item plus a final content-only round -- so the
    cap must scale with queue size rather than stay fixed, or larger queues
    deterministically hit MAX_TOOL_ROUNDS before finishing the scan."""
    return max(MAX_TOOL_ROUNDS, 2 * n_waiting + 3)


SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    MODELS = ["qwen3:1.7b", "gemma4:e4b"]
    TOOL_MODES = ["no-tools", "tools"]
    N_PER_BUCKET = 2
    N_REPEATS_BY_MODEL = {"qwen3:1.7b": 1, "gemma4:e4b": 1}
    JSONL_PATH = os.path.join(CURRENT_DIR, "smoketest_raw.jsonl")
    XLSX_PATH = os.path.join(CURRENT_DIR, "smoketest_results.xlsx")
    CASE_LIST_PATH = os.path.join(CURRENT_DIR, "smoketest_case_list.json")
else:
    MODELS = ["qwen3:1.7b", "gemma4:e4b"]
    TOOL_MODES = ["no-tools", "tools"]
    N_PER_BUCKET = 50
    N_REPEATS_BY_MODEL = {"qwen3:1.7b": 3, "gemma4:e4b": 3}
    JSONL_PATH = os.path.join(CURRENT_DIR, "raw.jsonl")
    XLSX_PATH = os.path.join(CURRENT_DIR, "results.xlsx")
    CASE_LIST_PATH = os.path.join(EXP_DIR, "case_list.json")

ONLY_MODELS = os.environ.get("ONLY_MODELS")
if ONLY_MODELS:
    wanted = [m.strip() for m in ONLY_MODELS.split(",") if m.strip()]
    MODELS = [m for m in MODELS if m in wanted]

N_REPEATS_OVERRIDE = os.environ.get("N_REPEATS")
if N_REPEATS_OVERRIDE:
    N_REPEATS_BY_MODEL = {m: int(N_REPEATS_OVERRIDE) for m in N_REPEATS_BY_MODEL}

OUTPUT_TAG = os.environ.get("OUTPUT_TAG")
if OUTPUT_TAG:
    # CASE_LIST_PATH is deliberately left pointing at the shared pinned case
    # list -- only where results get written changes, not which cases run.
    JSONL_PATH = os.path.join(CURRENT_DIR, f"raw_{OUTPUT_TAG}.jsonl")
    XLSX_PATH = os.path.join(CURRENT_DIR, f"results_{OUTPUT_TAG}.xlsx")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if OUTPUT_DIR:
    # A whole new sibling folder under EXP_DIR (e.g. for a new prompt design
    # that isn't directly comparable to current/ either) -- CASE_LIST_PATH is
    # still left pointing at the shared pinned case list.
    RUN_DIR = os.path.join(EXP_DIR, OUTPUT_DIR)
    os.makedirs(RUN_DIR, exist_ok=True)
    JSONL_PATH = os.path.join(RUN_DIR, "raw.jsonl")
    XLSX_PATH = os.path.join(RUN_DIR, "results.xlsx")

TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
PRIORITY_PATTERN = re.compile(r"(?:ED )?Scan Priority:\s*(P[1-3])")

QUEUE_LENGTH_BUCKETS = [
    ("tiny (0-1 waiting)", 0, 1),
    ("small (2-3 waiting)", 2, 3),
    ("medium (4-5 waiting)", 4, 5),
    ("large (6-7 waiting)", 6, 7),
    ("huge (8+ waiting)", 8, None),
]
BUCKET_ORDER = [b for b, _, _ in QUEUE_LENGTH_BUCKETS]
OVERALL_LABEL = "Overall (all buckets)"

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
                    "item_a": {"type": "string", "description": "label of the first item, e.g. 'item_3' or 'new_item'"},
                    "item_b": {"type": "string", "description": "label of the second item, e.g. 'item_7' or 'new_item'"},
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
                    "item_a": {"type": "string", "description": "label of the first item, e.g. 'item_3' or 'new_item'"},
                    "item_b": {"type": "string", "description": "label of the second item, e.g. 'item_7' or 'new_item'"},
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
    """Best-effort extraction of the position number the model settled on."""
    cleaned = re.sub(r"[*_`#]", "", text or "")
    matches = re.findall(r"position\s*(?:number)?\s*(?:is|:|-|–)?\s*(\d+)", cleaned, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"answer\s*(?:is|:|-|–)?\s*(?:position\s*)?(\d+)", cleaned, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"^\s*(\d+)\s*$", cleaned)
    if matches:
        return int(matches[-1])
    return None


def grade(model_position, correct_position):
    if model_position is None:
        return "UNPARSEABLE"
    return "PASS" if model_position == correct_position else "FAIL"


def build_snapshot(nni, new_item):
    snapshot_time = new_item["parent_order_dt"]
    snapshot = nni[
        (nni["parent_order_dt"] < snapshot_time)
        & (nni["exam_start_dt"] > snapshot_time)
        & (nni["Order ID"] != new_item["Order ID"])
    ].copy()
    snapshot["waiting_so_far_min"] = round(
        (snapshot_time - snapshot["parent_order_dt"]).dt.total_seconds() / 60, 1
    )
    return snapshot.sort_values(
        ["priority_score", "waiting_so_far_min"], ascending=[True, False]
    ).reset_index(drop=True)


def build_item_lookup(snapshot, new_item):
    lookup = {}
    for i, r in enumerate(snapshot.itertuples(), start=1):
        lookup[f"item_{i}"] = {"priority_score": r.priority_score, "priority_tier": r.priority_tier,
                                "waiting_so_far_min": r.waiting_so_far_min}
    lookup["new_item"] = {"priority_score": new_item["priority_score"], "priority_tier": new_item["priority_tier"],
                            "waiting_so_far_min": 0.0}
    return lookup


def normalize_label(raw):
    s = str(raw).strip().lower().replace(" ", "_")
    if "new" in s:
        return "new_item"
    m = re.search(r"(\d+)", s)
    if m:
        return f"item_{m.group(1)}"
    return s


def execute_tool(name, args, item_lookup):
    if not isinstance(args, dict) or "item_a" not in args or "item_b" not in args:
        return {"error": f"malformed arguments for {name}: expected an object with "
                          f"'item_a' and 'item_b', got {args!r}"}
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
        winner = "tie" if result == "tie" else ("item_a" if result == a else "item_b")
        return {"item_a": a, "priority_tier_a": pa["priority_tier"],
                 "item_b": b, "priority_tier_b": pb["priority_tier"], "result": result, "winner": winner}
    elif name == "compare_wait_time":
        if pa["waiting_so_far_min"] > pb["waiting_so_far_min"]:
            result = a
        elif pb["waiting_so_far_min"] > pa["waiting_so_far_min"]:
            result = b
        else:
            result = "tie"
        winner = "tie" if result == "tie" else ("item_a" if result == a else "item_b")
        return {"item_a": a, "waited_min_a": pa["waiting_so_far_min"],
                 "item_b": b, "waited_min_b": pb["waiting_so_far_min"], "result": result, "winner": winner}
    return {"error": f"unknown tool {name}"}


def build_prompt_no_tools(snapshot, new_item, median_scan_time_min):
    system_lines = [
        "Your role is to determine the insertion position of one newly arrived scan item "
        "in an existing hospital CT scanner queue.",

        "Your only task is to find the position of the new item in the existing queue. "
        "Do not create a schedule, estimate completion times, assign scan times, "
        "or reorder the existing queue.",

        "The existing queue is already sorted in scan order. If the queue contains "
        "k items, item_1 is scanned next, item_2 is scanned second, and item_k is "
        "scanned last. Do not change the order of existing items.",

        "The ordering policy is strictly priority first, then waiting time: "
        "The priority order, from most urgent to least urgent, is exactly:",
        "P1 goes before P2; P2 goes before P3; P3 goes before Unknown",
        "This ordering is fixed. "
        "Never reverse it, sort it numerically, or infer a different ordering.",

        "Within the same priority tier, use first-come-first-served ordering: "
        "the item that has waited longer, "
        "must be scanned first.",
        "Examples:",
        "- If new_item is P3 and item_i is Unknown, new_item goes BEFORE item_i.",
        "- If new_item is Unknown and item_i is P3, continue past item_i.",
        "- If new_item has waited 5 minutes and item_i has waited 20 minutes, "
        "continue past item_i.",
        "- If new_item has waited 20 minutes and item_i has waited 5 minutes, "
        "insert before item_i."

        "To find the insertion position, go down the existing queue from item_1 "
        "to item_k and identify the first item that should be scanned after the "
        "new item according to the priority and first-come-first-served rules.",

        "If an existing item has a more urgent priority than the new item, it stays "
        "before the new item. If it has the same priority, it also stays before the "
        "new item because it has waited longer.",

        "When the first existing item that should go after the new item is found, insert the new item "
        "immediately before it.",

        "If every existing item should get scanned before the new item, append the new "
        "item after the current queue.",

        "The valid positions are 1 through k+1, where k is the number of existing "
        "items. Position 1 is before item_1, and position k+1 is after item_k. "
        "If k=0, the queue is empty and the answer is Position: 1."

    ]
    
    system_prompt = "\n".join(system_lines)

    user_lines = []
    if len(snapshot) == 0:
        user_lines.append("*** QUEUE IS EMPTY -- no patients currently waiting ***")
    elif len(snapshot) == 1:
        user_lines.append("Current queue, already sorted into scan order (item_1 = next to be scanned):")
        for i, r in enumerate(snapshot.itertuples(), start=1):
            user_lines.append(
                f"item_{i}: priority {r.priority_tier}, waited {r.waiting_so_far_min} min so far, "
                f"median scan time {median_scan_time_min} min"
            )
    else:
        user_lines.append(
            f"Current queue, already sorted into scan order (item_1 = next to be "
            f"scanned ... item_{len(snapshot)} = last item to be scanned):"
        )
        for i, r in enumerate(snapshot.itertuples(), start=1):
            user_lines.append(
                f"item_{i}: priority {r.priority_tier}, waited {r.waiting_so_far_min} min so far, "
                f"median scan time {median_scan_time_min} min"
            )
    user_lines += [
        "",
        "A new item has just arrived:",
        f"new_item: priority {new_item['priority_tier']}, waited 0 min so far, "
        f"median scan time {median_scan_time_min} min",
        "",
        "At what position should this new item be slotted into the queue? ",
        "Reply with exactly one line in this format: Position: <integer>. "
        "The integer must be the insertion position of the new item in the queue. "
        "Do not include reasoning or any other text."
    ]
    user_prompt = "\n".join(user_lines)
    return system_prompt, user_prompt


def build_prompt_tools(snapshot, new_item, median_scan_time_min):
    system_lines = [
        "Your role is to determine the insertion position of one newly arrived "
        "scan item in an existing hospital CT scanner queue.",

        "Your only task is to find where the new item should be inserted. Do not "
        "create a schedule, estimate completion times, assign scan times, or "
        "reorder existing queue items.",

        "The existing queue is already sorted in scan order. If it contains k "
        "items, item_1 is scanned next, item_2 is scanned second, and item_k is "
        "scanned last. Preserve the order of all existing items.",

        "You have two tools: compare_priority(item_a, item_b) and "
        "compare_wait_time(item_a, item_b). Each tool compares the two provided "
        "item labels and reports which item should be scanned first according to "
        "that criterion, or reports a tie.",

        "Each tool's result includes a 'winner' field, whose value is exactly "
        "'item_a', 'item_b', or 'tie' -- an unambiguous way to read which side won "
        "without matching it back to the item labels yourself.",

        "For every non-empty queue, you must use the tools to compare the new item "
        "with existing queue items. Do not compare priority values or waiting-time "
        "numbers yourself. Always call compare_priority first. Only call "
        "compare_wait_time when compare_priority reports a tie.",

        "Use the native structured tool interface for every tool call. Do not "
        "write or simulate tool calls as ordinary text. Lines such as "
        "'Tool Call: compare_priority(...)' or inline text like "
        "'compare_priority(item_a: new_item, item_b: item_1)' are not real tool "
        "calls and will be rejected -- you must invoke the actual tool.",

        "Scan the existing queue from item_1 through item_k. For each item_i in the existing queue, "
        "compare new_item with item_i using compare_priority(new_item, item_i).",

        "If compare_priority reports that new_item should be scanned first, insert "
        "new_item immediately before item_i and stop.",

        "If compare_priority reports that item_i should be scanned first, leave "
        "item_i before new_item and continue to the next queue item.",

        "If compare_priority reports a tie, call "
        "compare_wait_time(new_item, item_i). If the wait-time comparison reports "
        "that new_item should be scanned first, insert it immediately before "
        "item_i. If it reports that item_i should be scanned first, continue to "
        "the next queue item.",

        "If both tools report a tie, keep the existing item before new_item because "
        "the existing item is already in the queue and new_item has just arrived.",

        "If new_item should come after every existing item, append it to the end "
        "of the queue. If the queue contains k items, this is position k+1. "
        "If the queue is empty, return position 1 without making any tool calls.",

        "Position 1 means immediately before item_1. Position k+1 means "
        "immediately after item_k.",
    ]
    system_prompt = "\n".join(system_lines)

    user_lines = []
    if len(snapshot) == 0:
        user_lines.append("*** QUEUE IS EMPTY -- no patients currently waiting ***")
    elif len(snapshot) == 1:
        user_lines.append(
            "Current queue, already sorted into scan order, labelled item_1 (the only item, next to be scanned):"
        )
        for i, r in enumerate(snapshot.itertuples(), start=1):
            user_lines.append(
                f"item_{i}: priority {r.priority_tier}, waited {r.waiting_so_far_min} min so far, "
                f"median scan time {median_scan_time_min} min"
            )
    else:
        user_lines.append(
            f"Current queue, already sorted into scan order, labelled item_1 (next to "
            f"be scanned) through item_{len(snapshot)} (last to be scanned):"
        )
        for i, r in enumerate(snapshot.itertuples(), start=1):
            user_lines.append(
                f"item_{i}: priority {r.priority_tier}, waited {r.waiting_so_far_min} min so far, "
                f"median scan time {median_scan_time_min} min"
            )
    user_lines += [
        "",
        f"new_item: priority {new_item['priority_tier']}, waited 0 min so far, "
        f"median scan time {median_scan_time_min} min",
        "",
        "Use the tools to work out where new_item belongs in the queue. ",
        "Reply with exactly one line in this format: Position: <integer>. "
        "The integer must be the insertion position of the new item in the queue. "
        "Do not include reasoning or any other text."
    ]
    user_prompt = "\n".join(user_lines)
    return system_prompt, user_prompt


def ollama_chat(model, messages, tools=None, retries=2):
    payload = {"model": model, "messages": messages, "stream": False, "think": True}
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(1, retries + 2):
        request = urllib.request.Request(
            OLLAMA_CHAT_URL, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt > retries:
                raise SystemExit(
                    f"Could not reach Ollama at {OLLAMA_CHAT_URL} after {retries + 1} attempts ({e}). "
                    f"Is the Ollama daemon running? Try `ollama serve`."
                )
            print(f"  (attempt {attempt} timed out/failed: {e}; retrying...)")


def run_no_tools_once(model, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = ollama_chat(model, messages)
    msg = result.get("message", {})
    content = (msg.get("content") or "").strip()
    thinking = (msg.get("thinking") or "").strip()
    model_position = extract_stated_position(content)
    return {
        "run_index": None, "model_position": model_position, "verdict": grade(model_position, None),
        "content": content, "thinking": thinking, "n_rounds": None, "hit_round_cap": False, "tool_calls": [],
        "ungrounded": False, "n_protocol_retries": 0,
    }


NUDGE_TO_USE_TOOLS = (
    "You answered without calling any tools. You must use the compare_priority "
    "and compare_wait_time tools for every comparison -- do not write out tool "
    "calls or their results as text yourself. Call compare_priority(new_item, "
    "item_1) now."
)

MAX_PROTOCOL_RETRIES = 3

NUDGE_FAKE_TOOL_CALL = (
    "Your previous response was ordinary text, not a structured tool call. "
    "Use the native tool interface now. Do not describe the call in text."
)

FAKE_TOOL_CALL_PATTERN = re.compile(
    r"tool\s*call\s*[:\-]|compare_priority\s*[\(\{]|compare_wait_time\s*[\(\{]",
    re.IGNORECASE,
)


def looks_like_fake_tool_call(content):
    """Heuristic for a model narrating a tool call as plain text (e.g. 'Tool
    Call: compare_priority{item_a: new_item, item_b: item_1}') instead of
    using the API's structured tool_calls field. Only ever consulted when
    tool_calls is empty -- never applied to a real structured call."""
    if not content:
        return False
    cleaned = re.sub(r"[*_`#]", "", content)
    return bool(FAKE_TOOL_CALL_PATTERN.search(cleaned))


def run_tools_once(model, system_prompt, user_prompt, item_lookup, max_rounds=MAX_TOOL_ROUNDS,
                    max_protocol_retries=MAX_PROTOCOL_RETRIES):
    """Runs the tool-calling round loop against two independent budgets: a
    real comparison-round budget (round_i, capped at max_rounds, unchanged
    from before) and a separate fake-textual-tool-call retry budget
    (protocol_retries, capped at max_protocol_retries) that does not consume
    round_i. A final answer is only accepted ungrounded (no real tool call
    ever made) when the queue was empty (n_waiting == 0)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    n_waiting = len(item_lookup) - 1
    tool_call_log = []
    round_log = []
    content, thinking = "", ""
    made_real_tool_call = False
    round_i = 0
    protocol_retries = 0
    for _ in range(max_rounds + max_protocol_retries):
        result = ollama_chat(model, messages, tools=TOOLS)
        msg = result.get("message", {})
        content = (msg.get("content") or "").strip()
        thinking = (msg.get("thinking") or "").strip()
        tool_calls = msg.get("tool_calls") or []
        round_log.append({"round": round_i + 1, "thinking": thinking, "content": content, "n_tool_calls": len(tool_calls)})
        messages.append(msg)

        if tool_calls:
            round_i += 1
            made_real_tool_call = True
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name")
                args = fn.get("arguments", {}) or {}
                tool_result = execute_tool(name, args, item_lookup)
                tool_call_log.append({"round": round_i, "tool_name": name, "arguments": args, "result": tool_result})
                messages.append({"role": "tool", "content": json.dumps(tool_result)})
            if round_i >= max_rounds:
                return {"content": content, "thinking": thinking, "n_rounds": round_i, "hit_round_cap": True,
                        "tool_call_log": tool_call_log, "round_log": round_log, "ungrounded": False,
                        "no_tool_support": False, "protocol_retries": protocol_retries}
            continue

        if looks_like_fake_tool_call(content):
            protocol_retries += 1
            if protocol_retries < max_protocol_retries:
                messages.append({"role": "user", "content": NUDGE_FAKE_TOOL_CALL})
                continue
            if made_real_tool_call:
                return {"content": content, "thinking": thinking, "n_rounds": round_i, "hit_round_cap": True,
                        "tool_call_log": tool_call_log, "round_log": round_log, "ungrounded": False,
                        "no_tool_support": False, "protocol_retries": protocol_retries}
            return {"content": content, "thinking": thinking, "n_rounds": round_i, "hit_round_cap": False,
                    "tool_call_log": tool_call_log, "round_log": round_log, "ungrounded": True,
                    "no_tool_support": True, "protocol_retries": protocol_retries}

        if n_waiting == 0 or made_real_tool_call:
            return {"content": content, "thinking": thinking, "n_rounds": round_i, "hit_round_cap": False,
                    "tool_call_log": tool_call_log, "round_log": round_log, "ungrounded": False,
                    "no_tool_support": False, "protocol_retries": protocol_retries}

        round_i += 1
        if round_i >= max_rounds:
            return {"content": content, "thinking": thinking, "n_rounds": round_i, "hit_round_cap": False,
                    "tool_call_log": tool_call_log, "round_log": round_log, "ungrounded": True,
                    "no_tool_support": True, "protocol_retries": protocol_retries}
        messages.append({"role": "user", "content": NUDGE_TO_USE_TOOLS})

    return {"content": content, "thinking": thinking, "n_rounds": round_i, "hit_round_cap": True,
            "tool_call_log": tool_call_log, "round_log": round_log, "ungrounded": not made_real_tool_call,
            "no_tool_support": not made_real_tool_call, "protocol_retries": protocol_retries}


def run_no_tools_attempt(model, system_prompt, user_prompt, correct_position):
    r = run_no_tools_once(model, system_prompt, user_prompt)
    r["verdict"] = grade(r["model_position"], correct_position)
    return r


def run_tools_attempt(model, system_prompt, user_prompt, item_lookup, correct_position, max_rounds):
    outcome = run_tools_once(model, system_prompt, user_prompt, item_lookup, max_rounds=max_rounds)
    if outcome["no_tool_support"]:
        verdict, model_position = "NO_TOOL_SUPPORT", None
    elif outcome["hit_round_cap"]:
        verdict, model_position = "INCOMPLETE", None
    else:
        model_position = extract_stated_position(outcome["content"])
        verdict = grade(model_position, correct_position)
    return {
        "model_position": model_position, "verdict": verdict,
        "content": outcome["content"], "thinking": outcome["thinking"],
        "n_rounds": outcome["n_rounds"], "hit_round_cap": outcome["hit_round_cap"],
        "tool_calls": outcome["tool_call_log"], "rounds": outcome["round_log"],
        "ungrounded": outcome["ungrounded"], "n_protocol_retries": outcome["protocol_retries"],
    }


def run_with_retries(attempt_fn, max_attempts=MAX_RUN_ATTEMPTS):
    """Calls attempt_fn() (a zero-arg thunk that issues one live model call and
    returns a run dict with a "verdict") up to max_attempts times, stopping as
    soon as the verdict is not INCOMPLETE/UNPARSEABLE/NO_TOOL_SUPPORT. Returns
    the last attempt made, with an "attempts" count attached."""
    r = None
    for attempt in range(1, max_attempts + 1):
        r = attempt_fn()
        if r["verdict"] not in ("INCOMPLETE", "UNPARSEABLE", "NO_TOOL_SUPPORT") or attempt == max_attempts:
            break
    r["attempts"] = attempt
    return r


def run_case(model, tool_mode, snapshot, new_item, correct_position, median_scan_time_min,
             start_run_index=1, end_run_index=None):
    if end_run_index is None:
        end_run_index = N_REPEATS_BY_MODEL[model]
    runs = []
    if tool_mode == "no-tools":
        system_prompt, user_prompt = build_prompt_no_tools(snapshot, new_item, median_scan_time_min)
        for run_idx in range(start_run_index, end_run_index + 1):
            r = run_with_retries(lambda: run_no_tools_attempt(model, system_prompt, user_prompt, correct_position))
            r["run_index"] = run_idx
            runs.append(r)
    else:
        system_prompt, user_prompt = build_prompt_tools(snapshot, new_item, median_scan_time_min)
        item_lookup = build_item_lookup(snapshot, new_item)
        max_rounds = tool_round_cap(len(snapshot))
        for run_idx in range(start_run_index, end_run_index + 1):
            r = run_with_retries(lambda: run_tools_attempt(
                model, system_prompt, user_prompt, item_lookup, correct_position, max_rounds
            ))
            r["run_index"] = run_idx
            runs.append(r)
    return system_prompt, user_prompt, runs


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_run_counts(jsonl_path):
    """(model, tool_mode, order_id) -> number of runs already checkpointed.

    A case can be spread across more than one JSONL line (e.g. a top-up run
    adding extra repeats to a case that was already checkpointed with fewer
    runs), so counts are summed across every line sharing the same key.
    """
    counts = {}
    if not os.path.exists(jsonl_path):
        return counts
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["model"], rec["tool_mode"], rec["order_id"])
                counts[key] = counts.get(key, 0) + len(rec["runs"])
            except (json.JSONDecodeError, KeyError):
                continue
    return counts


def append_record(jsonl_path, record):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_all_records(jsonl_path):
    """Loads the checkpoint, merging runs for any case split across multiple
    lines (e.g. an original run plus a later top-up of extra repeats) into
    a single record with a combined "runs" list, keyed on (model, tool_mode,
    order_id)."""
    merged = {}
    order = []
    if not os.path.exists(jsonl_path):
        return []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec["model"], rec["tool_mode"], rec["order_id"])
            if key not in merged:
                merged[key] = dict(rec)
                merged[key]["runs"] = list(rec["runs"])
                order.append(key)
            else:
                merged[key]["runs"].extend(rec["runs"])
    return [merged[k] for k in order]


# ---------------------------------------------------------------------------
# Final xlsx assembly from the JSONL checkpoint
# ---------------------------------------------------------------------------
def build_xlsx(jsonl_path, xlsx_path):
    records = load_all_records(jsonl_path)
    print(f"Assembling {xlsx_path} from {len(records)} checkpointed snapshot(s) in {jsonl_path} ...")

    cases_rows, runs_rows, tool_calls_rows = [], [], []
    for rec in records:
        runs = rec["runs"]
        positions = [r["model_position"] for r in runs]
        n_runs = len(runs)
        n_pass = sum(1 for r in runs if r["verdict"] == "PASS")
        counts = Counter(positions)
        most_common_position, most_common_count = counts.most_common(1)[0] if counts else (None, 0)
        cases_rows.append({
            "model": rec["model"], "tool_mode": rec["tool_mode"], "case_index": rec["case_index"],
            "label": rec["label"], "queue_length_bucket": rec["queue_length_bucket"],
            "new_item_priority": rec["new_item_priority"], "n_waiting": rec["n_waiting"],
            "correct_position": rec["correct_position"], "order_id": rec["order_id"],
            "n_runs": n_runs, "n_distinct_positions": len(counts),
            "most_common_position": most_common_position,
            "self_consistency_rate": most_common_count / n_runs if n_runs else None,
            "n_pass": n_pass, "pass_rate": n_pass / n_runs if n_runs else None,
        })
        for r in runs:
            runs_rows.append({
                "model": rec["model"], "tool_mode": rec["tool_mode"], "case_index": rec["case_index"],
                "label": rec["label"], "queue_length_bucket": rec["queue_length_bucket"],
                "order_id": rec["order_id"], "run_index": r["run_index"],
                "correct_position": rec["correct_position"], "model_position": r["model_position"],
                "verdict": r["verdict"], "n_rounds": r.get("n_rounds"), "hit_round_cap": r.get("hit_round_cap"),
                "ungrounded": r.get("ungrounded", False), "n_protocol_retries": r.get("n_protocol_retries", 0),
                "content": r["content"], "thinking": r["thinking"],
            })
            for tc in r.get("tool_calls", []):
                tool_calls_rows.append({
                    "model": rec["model"], "tool_mode": rec["tool_mode"], "case_index": rec["case_index"],
                    "order_id": rec["order_id"], "run_index": r["run_index"], "round": tc["round"],
                    "tool_name": tc["tool_name"], "arguments": json.dumps(tc["arguments"]),
                    "result": json.dumps(tc["result"]),
                })

    cases = pd.DataFrame(cases_rows)
    runs_df = pd.DataFrame(runs_rows)
    tool_calls = pd.DataFrame(tool_calls_rows)

    summary_rows = []
    if not cases.empty:
        for model in cases["model"].unique():
            for tool_mode in cases[cases["model"] == model]["tool_mode"].unique():
                sub = cases[(cases["model"] == model) & (cases["tool_mode"] == tool_mode)]
                for bucket in BUCKET_ORDER:
                    bsub = sub[sub["queue_length_bucket"] == bucket]
                    if len(bsub) == 0:
                        continue
                    summary_rows.append({
                        "model": model, "tool_mode": tool_mode, "queue_length_bucket": bucket,
                        "n_cases": len(bsub), "n_runs_total": int(bsub["n_runs"].sum()),
                        "n_pass_total": int(bsub["n_pass"].sum()),
                        "pass_rate": bsub["n_pass"].sum() / bsub["n_runs"].sum(),
                        "mean_self_consistency_rate": bsub["self_consistency_rate"].mean(),
                    })
                summary_rows.append({
                    "model": model, "tool_mode": tool_mode, "queue_length_bucket": OVERALL_LABEL,
                    "n_cases": len(sub), "n_runs_total": int(sub["n_runs"].sum()),
                    "n_pass_total": int(sub["n_pass"].sum()),
                    "pass_rate": sub["n_pass"].sum() / sub["n_runs"].sum(),
                    "mean_self_consistency_rate": sub["self_consistency_rate"].mean(),
                })
    summary = pd.DataFrame(summary_rows)

    notes = pd.DataFrame([
        {"Note": f"250 snapshots (50 randomly sampled per queue-length bucket, random_state={RANDOM_SEED}), "
                 "reused identically across all model/tool_mode combinations. Unlike the earlier stratified "
                 "experiments, this sample is NOT further split by priority tier -- plain uniform random draw "
                 "within each bucket."},
        {"Note": f"Each snapshot's exact prompt is sent N_REPEATS separate times per model/tool_mode (no "
                 "temperature override -- Ollama's default sampling), to measure self-consistency in addition "
                 f"to correctness. N_REPEATS is per-model: {N_REPEATS_BY_MODEL}. "
                 "self_consistency_rate = (count of the most common stated position) / n_runs."},
        {"Note": "Reasoning is captured via Ollama's native 'think' option (POST /api/chat, think=true) as a "
                 "separate 'thinking' field, not requested inline in the answer text -- the prompt asks for "
                 "ONLY the position number."},
        {"Note": "Tool-calling mode logs every individual tool invocation as its own row in Tool_Calls (round, "
                 "tool_name, arguments, result), rather than a flat transcript string. hit_round_cap=True means "
                 f"the model never settled on a content-only final answer within {MAX_TOOL_ROUNDS} rounds (verdict INCOMPLETE)."},
        {"Note": "correct_position is the deterministic policy-optimal slot (priority tier, then earliest "
                 "arrival) computed in code, not by the model -- same comparator as the earlier experiments."},
        {"Note": f"Source checkpoint: {os.path.basename(jsonl_path)} (append-only JSONL, one line per completed "
                 "snapshot per model/tool_mode; resumable -- rerunning the script skips (model, tool_mode, "
                 "order_id) combinations already present)."},
    ])

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="Notes", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        cases.to_excel(writer, sheet_name="Cases", index=False)
        runs_df.to_excel(writer, sheet_name="Runs", index=False)
        tool_calls.to_excel(writer, sheet_name="Tool_Calls", index=False)
        for sheet in writer.sheets.values():
            auto_col_width(sheet)
    print(f"Wrote {xlsx_path} (Cases={len(cases)}, Runs={len(runs_df)}, Tool_Calls={len(tool_calls)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def retry_bad_runs(nni, candidates):
    records = load_all_records(JSONL_PATH)
    candidates_by_id = candidates.set_index(candidates["Order ID"].astype(str), drop=False)
    n_retried = 0
    for rec in records:
        bad_idxs = [i for i, r in enumerate(rec["runs"]) if r["verdict"] in ("INCOMPLETE", "UNPARSEABLE", "NO_TOOL_SUPPORT")]
        if not bad_idxs:
            continue
        order_id = rec["order_id"]
        if order_id not in candidates_by_id.index:
            print(f"WARNING: order_id {order_id} no longer found in candidates, skipping retry for it")
            continue
        pick = candidates_by_id.loc[order_id]
        if isinstance(pick, pd.DataFrame):
            pick = pick.iloc[0]
        snapshot = build_snapshot(nni, pick)
        correct_position = 1 + int((snapshot["priority_score"] <= pick["priority_score"]).sum())
        system_prompt, user_prompt = rec["system_prompt"], rec["prompt"]
        model, tool_mode = rec["model"], rec["tool_mode"]
        for i in bad_idxs:
            run_idx = rec["runs"][i]["run_index"]
            old_verdict = rec["runs"][i]["verdict"]
            if tool_mode == "no-tools":
                r = run_with_retries(lambda: run_no_tools_attempt(model, system_prompt, user_prompt, correct_position))
            else:
                item_lookup = build_item_lookup(snapshot, pick)
                max_rounds = tool_round_cap(len(snapshot))
                r = run_with_retries(lambda: run_tools_attempt(
                    model, system_prompt, user_prompt, item_lookup, correct_position, max_rounds
                ))
            r["run_index"] = run_idx
            rec["runs"][i] = r
            n_retried += 1
            print(f"  retried {model}/{tool_mode} order {order_id} run {run_idx}: {old_verdict} -> {r['verdict']}")

    with open(JSONL_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nRetried {n_retried} incomplete/unparseable run(s); rewrote {JSONL_PATH}")


def main():
    rebuild_only = "--rebuild-only" in sys.argv
    retry_bad = "--retry-bad" in sys.argv
    if rebuild_only:
        build_xlsx(JSONL_PATH, XLSX_PATH)
        return

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

    if retry_bad:
        retry_bad_runs(nni, candidates)
        build_xlsx(JSONL_PATH, XLSX_PATH)
        return

    # -----------------------------------------------------------------------
    # Case list is pinned to disk on first run and reused forever after --
    # the source spreadsheets get regenerated externally while this
    # experiment runs across days, which would otherwise silently reshuffle
    # a fresh random sample out from under an in-progress run.
    # -----------------------------------------------------------------------
    if os.path.exists(CASE_LIST_PATH):
        with open(CASE_LIST_PATH) as f:
            pinned = json.load(f)
        candidates_by_id = candidates.set_index(candidates["Order ID"].astype(str), drop=False)
        cases = []
        missing = []
        for entry in pinned:
            oid = entry["order_id"]
            if oid not in candidates_by_id.index:
                missing.append(oid)
                continue
            pick = candidates_by_id.loc[oid]
            if isinstance(pick, pd.DataFrame):
                pick = pick.iloc[0]
            cases.append((entry["label"], entry["queue_length_bucket"], pick))
        if missing:
            print(f"WARNING: {len(missing)} pinned order_id(s) no longer found in candidates, skipping: {missing}\n")
        print(f"Loaded {len(cases)} pinned snapshot(s) from {CASE_LIST_PATH}\n")
    else:
        cases = []
        for bucket_name, lo, hi in QUEUE_LENGTH_BUCKETS:
            if hi is None:
                bucket_pool = candidates[candidates["queue_position_at_arrival"] >= lo]
            else:
                bucket_pool = candidates[
                    (candidates["queue_position_at_arrival"] >= lo) & (candidates["queue_position_at_arrival"] <= hi)
                ]
            sample = bucket_pool.sample(n=min(N_PER_BUCKET, len(bucket_pool)), random_state=RANDOM_SEED)
            for idx, (_, pick) in enumerate(sample.iterrows(), start=1):
                cases.append((f"{bucket_name} / random-{idx}", bucket_name, pick))
        with open(CASE_LIST_PATH, "w") as f:
            json.dump(
                [{"label": label, "queue_length_bucket": bucket, "order_id": str(pick["Order ID"])}
                 for label, bucket, pick in cases],
                f, indent=2,
            )
        print(f"Sampled {len(cases)} snapshot(s) ({N_PER_BUCKET} per bucket x {len(QUEUE_LENGTH_BUCKETS)} buckets) "
              f"and pinned the list to {CASE_LIST_PATH} for all future runs\n")

    run_counts = load_run_counts(JSONL_PATH)
    if run_counts:
        print(f"Resuming: {len(run_counts)} (model, tool_mode, order_id) combo(s) already have checkpointed run(s) "
              f"in {JSONL_PATH}\n")

    total_combos = len(MODELS) * len(TOOL_MODES) * len(cases)
    done_so_far = 0
    for model in MODELS:
        n_repeats = N_REPEATS_BY_MODEL[model]
        for tool_mode in TOOL_MODES:
            print("=" * 78)
            print(f"MODEL {model} / {tool_mode}")
            for i, (label, bucket, new_item) in enumerate(cases, start=1):
                done_so_far += 1
                order_id = str(new_item["Order ID"])
                key = (model, tool_mode, order_id)
                n_done = run_counts.get(key, 0)
                if n_done >= n_repeats:
                    print(f"  [{model}/{tool_mode}] snapshot {i}/{len(cases)} (order {order_id}) "
                          f"-- already done ({n_done}/{n_repeats} runs), skipping")
                    continue

                t0 = time.time()
                snapshot = build_snapshot(nni, new_item)
                correct_position = 1 + int((snapshot["priority_score"] <= new_item["priority_score"]).sum())

                system_prompt, prompt, runs = run_case(
                    model, tool_mode, snapshot, new_item, correct_position, median_scan_time_min,
                    start_run_index=n_done + 1, end_run_index=n_repeats,
                )

                record = {
                    "model": model, "tool_mode": tool_mode, "case_index": i, "label": label,
                    "queue_length_bucket": bucket, "order_id": order_id,
                    "new_item_priority": new_item["priority_tier"], "n_waiting": len(snapshot),
                    "correct_position": correct_position, "system_prompt": system_prompt,
                    "prompt": prompt, "runs": runs,
                }
                append_record(JSONL_PATH, record)
                run_counts[key] = n_done + len(runs)

                verdicts = [r["verdict"] for r in runs]
                elapsed = time.time() - t0
                print(f"  [{model}/{tool_mode}] snapshot {i}/{len(cases)} (bucket={bucket}, order {order_id}) "
                      f"-- {verdicts} -- done in {elapsed:.0f}s ({done_so_far}/{total_combos} overall)")

    print("\nAll model/tool_mode/snapshot combinations complete.")
    build_xlsx(JSONL_PATH, XLSX_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
