"""
nni_llm_queue_slot_target_time_gemma4.py
Random-sample consistency + reasoning-logging experiment for the NNI CT
queue-slotting task, using gemma4:e4b and a target-aware scheduling rule
that considers priority, FCFS, approximate scan duration, and priority-
specific target times.

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
                                           exam_start_dt, queue_position_at_arrival,
                                           exam_duration_min_approx)
  merged/ct_orders_cleaned.xlsx          (Imaging -- Ordering Question Responses)

Calls: local Ollama REST API (http://localhost:11434/api/chat), model
gemma4:e4b, with "think": true. No new pip dependencies --
uses stdlib urllib.

Checkpointing: every completed snapshot (3 runs, for the current model/mode)
is appended as one JSON line to the current run's raw.jsonl immediately, so
a killed/interrupted run can be resumed (re-running this script skips any
(model, tool_mode, order_id) already present in that file).

Results write to a separate target_time_gemma4/ folder under
analysis/nni_llm_queue_slot_consistency/ because the correct answer now uses
target-time feasibility and scan-duration arithmetic, so it is not directly
comparable to the priority/FCFS-only v2 run.

Outputs (under target_time_gemma4/):
  raw.jsonl      (checkpoint, source of truth)
  results.xlsx   (assembled from the JSONL: Notes, Summary, Cases, Runs, Tool_Calls)

Set env var SMOKE=1 to run a fast smoke test instead of the full experiment
(gemma4:e4b, both modes, 2 random cases per bucket, 1 repeat each, writing
to target_time_gemma4/smoketest_* files instead of the real checkpoint/output).

Set env var OUTPUT_TAG=<tag> to write to target_time_gemma4/raw_<tag>.jsonl
and target_time_gemma4/results_<tag>.xlsx instead (e.g. for a one-off scoped run), while
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
TARGET_DIR = os.path.join(EXP_DIR, "target_time_gemma4_no_tools_range_prompt_compare")
os.makedirs(TARGET_DIR, exist_ok=True)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MAX_TOOL_ROUNDS = 10
MAX_RUN_ATTEMPTS = 3
RANDOM_SEED = 42


def tool_round_cap(n_waiting):
    """Some models issue one tool call per round while linearly scanning the
    queue, needing up to two calls (compare_priority, then compare_wait_time
    on a tie) per existing item, then one or more evaluate_position calls
    for target-time feasibility, plus a final content-only round -- so the
    cap must scale with queue size rather than stay fixed, or larger queues
    deterministically hit MAX_TOOL_ROUNDS before finishing the scan."""
    return max(MAX_TOOL_ROUNDS, 3 * n_waiting + 5)


SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    MODELS = ["gemma4:e4b"]
    TOOL_MODES = ["no-tools"]
    PROMPT_VARIANTS = ["less_details", "more_details"]
    N_PER_BUCKET = 2
    N_REPEATS_BY_MODEL = {"gemma4:e4b": 1}
    JSONL_PATH = os.path.join(TARGET_DIR, "smoketest_raw.jsonl")
    XLSX_PATH = os.path.join(TARGET_DIR, "smoketest_results.xlsx")
    CASE_LIST_PATH = os.path.join(TARGET_DIR, "smoketest_case_list.json")
else:
    MODELS = ["gemma4:e4b"]
    TOOL_MODES = ["no-tools"]
    PROMPT_VARIANTS = ["less_details", "more_details"]
    N_PER_BUCKET = 5
    N_REPEATS_BY_MODEL = {"gemma4:e4b": 3}
    JSONL_PATH = os.path.join(TARGET_DIR, "raw.jsonl")
    XLSX_PATH = os.path.join(TARGET_DIR, "results.xlsx")
    CASE_LIST_PATH = os.path.join(TARGET_DIR, "case_list.json")

ONLY_MODELS = os.environ.get("ONLY_MODELS")
if ONLY_MODELS:
    wanted = [m.strip() for m in ONLY_MODELS.split(",") if m.strip()]
    MODELS = [m for m in MODELS if m in wanted]

N_REPEATS_OVERRIDE = os.environ.get("N_REPEATS")
if N_REPEATS_OVERRIDE:
    N_REPEATS_BY_MODEL = {m: int(N_REPEATS_OVERRIDE) for m in N_REPEATS_BY_MODEL}

ONLY_PROMPT_VARIANTS = os.environ.get("ONLY_PROMPT_VARIANTS")
if ONLY_PROMPT_VARIANTS:
    wanted = [v.strip() for v in ONLY_PROMPT_VARIANTS.split(",") if v.strip()]
    PROMPT_VARIANTS = [v for v in PROMPT_VARIANTS if v in wanted]

OUTPUT_TAG = os.environ.get("OUTPUT_TAG")
if OUTPUT_TAG:
    # CASE_LIST_PATH is deliberately left pointing at the shared pinned case
    # list -- only where results get written changes, not which cases run.
    JSONL_PATH = os.path.join(TARGET_DIR, f"raw_{OUTPUT_TAG}.jsonl")
    XLSX_PATH = os.path.join(TARGET_DIR, f"results_{OUTPUT_TAG}.xlsx")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
if OUTPUT_DIR:
    # A whole new sibling folder under EXP_DIR (e.g. for a new prompt design
    # that should not share another prompt design's outputs or pinned cases).
    RUN_DIR = os.path.join(EXP_DIR, OUTPUT_DIR)
    os.makedirs(RUN_DIR, exist_ok=True)
    JSONL_PATH = os.path.join(RUN_DIR, "raw.jsonl")
    XLSX_PATH = os.path.join(RUN_DIR, "results.xlsx")
    CASE_LIST_PATH = os.path.join(RUN_DIR, "case_list.json")

TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
TIER_RANK = {"P1": 0, "P2": 1, "P3": 2, "Unknown": 3}
SLA_TARGETS = {"P1": 30, "P2": 120, "P3": 360}
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
    {
        "type": "function",
        "function": {
            "name": "evaluate_position",
            "description": (
                "Evaluates one candidate insertion position for new_item using "
                "the approximate scan durations. Returns new_item's projected "
                "wait, whether it meets its priority target, which higher-priority "
                "existing items new_item would cut ahead of, and whether any such "
                "higher-priority item would exceed its own target after being delayed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "integer",
                        "description": "candidate position for new_item, from 1 through k+1",
                    },
                },
                "required": ["position"],
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


def target_for_priority(priority_tier):
    target = SLA_TARGETS.get(priority_tier)
    return None if target is None else float(target)


def clean_scan_duration(val):
    try:
        if pd.isna(val):
            return 0.0
        return max(0.0, float(val))
    except (TypeError, ValueError):
        return 0.0


def scan_duration_stats(nni):
    durations = nni["exam_duration_min_approx"].apply(clean_scan_duration)
    durations = durations[durations.notna() & (durations >= 0)]
    median = float(durations.median()) if len(durations) else 0.0
    sd = float(durations.std(ddof=1)) if len(durations) > 1 else 0.0
    return {
        "median": round(median, 1),
        "sd": round(sd, 1),
        "lower_2sd": round(max(0.0, median - 2 * sd), 1),
        "upper_2sd": round(median + 2 * sd, 1),
    }


def format_duration_range(stats):
    return (
        f"median {stats['median']:.1f} min, "
        f"median +/- 2 SD range {stats['lower_2sd']:.1f}-{stats['upper_2sd']:.1f} min"
    )


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
    snapshot["scan_duration_min"] = snapshot["exam_duration_min_approx"].apply(clean_scan_duration).round(1)
    return snapshot.sort_values(
        ["priority_score", "waiting_so_far_min"], ascending=[True, False]
    ).reset_index(drop=True)


def build_item_lookup(snapshot, new_item):
    lookup = {}
    for i, r in enumerate(snapshot.itertuples(), start=1):
        lookup[f"item_{i}"] = {"priority_score": int(r.priority_score), "priority_tier": r.priority_tier,
                                "waiting_so_far_min": float(r.waiting_so_far_min),
                                "target_min": target_for_priority(r.priority_tier),
                                "scan_duration_min": float(r.scan_duration_min)}
    lookup["new_item"] = {"priority_score": int(new_item["priority_score"]), "priority_tier": new_item["priority_tier"],
                            "waiting_so_far_min": 0.0,
                            "target_min": target_for_priority(new_item["priority_tier"]),
                            "scan_duration_min": clean_scan_duration(new_item["exam_duration_min_approx"])}
    return lookup


def baseline_position(snapshot, new_item):
    """The v2 priority/FCFS position: all more urgent existing items stay
    before new_item, and same-priority existing items also stay before it
    because new_item has just arrived and has waited 0 minutes."""
    return 1 + int((snapshot["priority_score"] <= new_item["priority_score"]).sum())


def ordered_labels_for_position(item_lookup, position):
    existing = [f"item_{i}" for i in range(1, len(item_lookup))]
    try:
        pos = int(position)
    except (TypeError, ValueError):
        return None
    if pos < 1 or pos > len(existing) + 1:
        return None
    labels = list(existing)
    labels.insert(pos - 1, "new_item")
    return labels


def evaluate_candidate_position(item_lookup, position):
    labels = ordered_labels_for_position(item_lookup, position)
    if labels is None:
        return {"error": f"position must be between 1 and {len(item_lookup)}, got {position!r}"}

    elapsed_before_start = 0.0
    projected = {}
    for label in labels:
        item = item_lookup[label]
        projected_wait = item["waiting_so_far_min"] + elapsed_before_start
        target = item["target_min"]
        target_met = None if target is None else projected_wait <= target
        projected[label] = {
            "label": label,
            "priority_tier": item["priority_tier"],
            "priority_score": item["priority_score"],
            "waited_so_far_min": round(item["waiting_so_far_min"], 1),
            "scan_duration_min": round(item["scan_duration_min"], 1),
            "projected_wait_min": round(projected_wait, 1),
            "target_min": target,
            "target_met": target_met,
        }
        elapsed_before_start += item["scan_duration_min"]

    new_item = item_lookup["new_item"]
    cut_ahead = []
    new_index = labels.index("new_item")
    for label in labels[new_index + 1:]:
        existing = item_lookup[label]
        if existing["priority_score"] < new_item["priority_score"]:
            p = projected[label]
            cut_ahead.append({
                "label": label,
                "priority_tier": p["priority_tier"],
                "projected_wait_min": p["projected_wait_min"],
                "target_min": p["target_min"],
                "target_met": p["target_met"],
            })

    higher_priority_breaches = [
        item for item in cut_ahead if item["target_met"] is False
    ]
    return {
        "position": int(position),
        "ordered_labels": labels,
        "new_item_projected_wait_min": projected["new_item"]["projected_wait_min"],
        "new_item_target_min": projected["new_item"]["target_min"],
        "new_item_target_met": projected["new_item"]["target_met"],
        "higher_priority_cut_ahead": cut_ahead,
        "n_higher_priority_cut_ahead": len(cut_ahead),
        "higher_priority_breach_if_chosen": bool(higher_priority_breaches),
        "allowed_under_higher_priority_target_rule": not higher_priority_breaches,
        "projected_items": list(projected.values()),
    }


def choose_target_aware_position(snapshot, new_item):
    item_lookup = build_item_lookup(snapshot, new_item)
    base_pos = baseline_position(snapshot, new_item)
    one_down_pos = min(base_pos + 1, len(snapshot) + 1)
    baseline_eval = evaluate_candidate_position(item_lookup, base_pos)
    one_down_eval = evaluate_candidate_position(item_lookup, one_down_pos)

    lower_priority_breaches_at_baseline = []
    base_labels = baseline_eval["ordered_labels"]
    new_index = base_labels.index("new_item")
    projected_by_label = {item["label"]: item for item in baseline_eval["projected_items"]}
    new_priority_score = int(new_item["priority_score"])
    for label in base_labels[new_index + 1:]:
        existing = item_lookup[label]
        projected = projected_by_label[label]
        if existing["priority_score"] > new_priority_score and projected["target_met"] is False:
            lower_priority_breaches_at_baseline.append({
                "label": label,
                "priority_tier": projected["priority_tier"],
                "projected_wait_min": projected["projected_wait_min"],
                "target_min": projected["target_min"],
            })

    one_down_avoids_lower_priority_breach = True
    for breached in lower_priority_breaches_at_baseline:
        one_down_projected = {
            item["label"]: item for item in one_down_eval["projected_items"]
        }.get(breached["label"])
        if one_down_projected is None or one_down_projected["target_met"] is False:
            one_down_avoids_lower_priority_breach = False
            break

    one_down_protects_new_item = one_down_eval["new_item_target_met"] is not False
    use_one_down = (
        one_down_pos != base_pos
        and bool(lower_priority_breaches_at_baseline)
        and one_down_avoids_lower_priority_breach
        and one_down_protects_new_item
    )
    chosen = dict(one_down_eval if use_one_down else baseline_eval)
    chosen["baseline_position"] = base_pos
    chosen["one_down_position"] = one_down_pos
    chosen["accepted_positions"] = sorted({base_pos, one_down_pos})
    chosen["baseline_position_evaluation"] = baseline_eval
    chosen["one_down_position_evaluation"] = one_down_eval
    chosen["lower_priority_breaches_at_baseline"] = lower_priority_breaches_at_baseline
    chosen["selection_reason"] = (
        "one-down chosen because baseline would delay a lower-priority existing item past target and one-down avoids it"
        if use_one_down else
        "baseline chosen because one-down was unnecessary, invalid, or would make new_item miss its own target"
    )
    return chosen["position"], chosen


def score_position(model_position, ideal_position, accepted_positions):
    if model_position is None:
        return -1
    if model_position == ideal_position:
        return 1
    return 0 if model_position in set(accepted_positions) else -1


def grade_position(model_position, ideal_position, accepted_positions):
    if model_position is None:
        return "UNPARSEABLE"
    return "PASS" if score_position(model_position, ideal_position, accepted_positions) == 1 else "FAIL"


def format_target(priority_tier):
    target = target_for_priority(priority_tier)
    return "no target" if target is None else f"target {target:g} min"


def normalize_label(raw):
    s = str(raw).strip().lower().replace(" ", "_")
    if "new" in s:
        return "new_item"
    m = re.search(r"(\d+)", s)
    if m:
        return f"item_{m.group(1)}"
    return s


def execute_tool(name, args, item_lookup):
    if name == "evaluate_position":
        if not isinstance(args, dict) or "position" not in args:
            return {"error": f"malformed arguments for {name}: expected an object with "
                              f"'position', got {args!r}"}
        return evaluate_candidate_position(item_lookup, args.get("position"))

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


def build_prompt_no_tools(snapshot, new_item, duration_stats=None, prompt_variant="more_details"):
    if duration_stats is None:
        duration_stats = {"median": 0.0, "sd": 0.0, "lower_2sd": 0.0, "upper_2sd": 0.0}
    duration_range = format_duration_range(duration_stats)
    system_lines_less_details = [
        "Determine the final insertion position of one newly arrived scan item "
        "in an existing hospital CT scanner queue.",

        "The existing queue is already sorted in baseline scan order. Preserve the "
        "relative order of all existing items.",

        "The baseline order follows the fixed priority order P1, P2, P3, Unknown, "
        "from most urgent to least urgent, with first-come-first-served ordering "
        "within each priority.",

        "The target thresholds are P1 = 30 minutes, P2 = 120 minutes, and "
        "P3 = 360 minutes. Unknown priority has no threshold.",

        "Exact scan durations are not known at scheduling time. Use the shared "
        f"historical NNI scan-duration range for each scan: {duration_range}.",

        "Position 1 is before item_1, and position k+1 is after all k existing "
        "items. If the queue is empty, return Position: 1.",

        "Reply with exactly one line in the format Position: N. Do not include "
        "reasoning or any other text."
    ]
    system_lines_more_details = [
        "Determine the final insertion position of one newly arrived scan item "
        "in an existing hospital CT scanner queue.",

        "The existing queue is already sorted in baseline scan order. Preserve the "
        "relative order of all existing items.",

        "The baseline order follows the fixed priority order P1, P2, P3, Unknown, "
        "from most urgent to least urgent, with first-come-first-served ordering "
        "within each priority.",

        "The target thresholds are P1 = 30 minutes, P2 = 120 minutes, and "
        "P3 = 360 minutes. Unknown priority has no threshold.",

        "Choose the latest insertion position that best satisfies the target "
        "thresholds while preserving the existing queue order as much as possible.",

        "A lower-priority new_item must never be inserted before an existing item "
        "with a higher priority.",

        "A higher-priority new_item may be placed immediately after a lower-"
        "priority item if placing new_item before it would cause the lower-"
        "priority item to exceed its target threshold.",

        "This is the only exception to priority ordering. The exception may delay "
        "new_item by at most one queue position relative to its baseline position. "
        "new_item must not be moved below two or more lower-priority items.",

        "A target threshold is the maximum amount of time an order should ideally "
        "wait before its scan begins. The target thresholds are P1 = 30 minutes, "
        "P2 = 120 minutes, and P3 = 360 minutes. Unknown priority has no threshold.",

        "Exact scan durations are not known at scheduling time. Use the shared "
        f"historical NNI scan-duration range for each scan: {duration_range}.",

        "Position 1 is before item_1, and position k+1 is after all k existing "
        "items. If the queue is empty, return Position: 1.",

        "Reply with exactly one line in the format Position: N. Do not include "
        "reasoning or any other text."
    ]
    if prompt_variant == "less_details":
        system_lines = system_lines_less_details
    elif prompt_variant == "more_details":
        system_lines = system_lines_more_details
    else:
        raise ValueError(f"unknown prompt_variant {prompt_variant!r}")

    system_prompt = "\n".join(system_lines)

    user_lines = []
    if len(snapshot) == 0:
        user_lines.append("*** QUEUE IS EMPTY -- no patients currently waiting ***")
    elif len(snapshot) == 1:
        user_lines.append("Current queue, already sorted into scan order (item_1 = next to be scanned):")
        for i, r in enumerate(snapshot.itertuples(), start=1):
            user_lines.append(
                f"item_{i}: priority {r.priority_tier}, waited {r.waiting_so_far_min} min so far, "
                f"{format_target(r.priority_tier)}"
            )
    else:
        user_lines.append(
            f"Current queue, already sorted into scan order (item_1 = next to be "
            f"scanned ... item_{len(snapshot)} = last item to be scanned):"
        )
        for i, r in enumerate(snapshot.itertuples(), start=1):
            user_lines.append(
                f"item_{i}: priority {r.priority_tier}, waited {r.waiting_so_far_min} min so far, "
                f"{format_target(r.priority_tier)}"
            )
    user_lines += [
        "",
        f"Historical NNI scan-duration estimate for any scan: {duration_range}",
        "",
        "A new item has just arrived:",
        f"new_item: priority {new_item['priority_tier']}, waited 0 min so far, "
        f"{format_target(new_item['priority_tier'])}",
        "",
        "At what position should this new item be slotted into the queue? ",
        "Reply with exactly one line in this format: Position: <integer>. "
        "The integer must be the insertion position of the new item in the queue. "
        "Do not include reasoning or any other text."
    ]
    user_prompt = "\n".join(user_lines)
    return system_prompt, user_prompt


def build_prompt_tools(snapshot, new_item, median_scan_time_min=None):
    system_lines = [
        "Your role is to determine the insertion position of one newly arrived "
        "scan item in an existing hospital CT scanner queue, using a target-aware "
        "scheduling rule.",

        "Your only task is to find where the new item should be inserted. Do not "
        "create calendar timestamps or reorder existing queue items relative to "
        "each other.",

        "The existing queue is already sorted in scan order. If it contains k "
        "items, item_1 is scanned next, item_2 is scanned second, and item_k is "
        "scanned last. Preserve the order of all existing items.",

        "You have three tools: compare_priority(item_a, item_b), "
        "compare_wait_time(item_a, item_b), and evaluate_position(position). "
        "The comparison tools report which item should be scanned first by that "
        "one criterion, or report a tie. evaluate_position checks scan-duration "
        "and target-time feasibility for one candidate position.",

        "Each tool's result includes a 'winner' field, whose value is exactly "
        "'item_a', 'item_b', or 'tie' -- an unambiguous way to read which side won "
        "without matching it back to the item labels yourself.",

        "For every non-empty queue, you must use the tools. Use compare_priority "
        "and compare_wait_time to find the baseline priority/FCFS position. Do not "
        "compare priority values or waiting-time numbers yourself. Always call "
        "compare_priority first. Only call compare_wait_time when compare_priority "
        "reports a tie.",

        "Then use evaluate_position for candidate positions at or before the "
        "baseline position. You may move new_item earlier only if needed to avoid "
        "exceeding its own target. Choose the latest position that keeps new_item "
        "within target and does not make a higher-priority existing item that "
        "new_item cuts ahead of exceed its own target.",

        "If no protected position keeps new_item within target, choose the latest "
        "position that still protects higher-priority target times. If new_item is "
        "Unknown priority, it has no target and should use the baseline priority/"
        "FIFO position.",

        "Use the native structured tool interface for every tool call. Do not "
        "write or simulate tool calls as ordinary text. Lines such as "
        "'Tool Call: compare_priority(...)' or inline text like "
        "'compare_priority(item_a: new_item, item_b: item_1)' are not real tool "
        "calls and will be rejected -- you must invoke the actual tool.",

        "Scan the existing queue from item_1 through item_k. For each item_i in "
        "the existing queue, compare new_item with item_i using "
        "compare_priority(new_item, item_i).",

        "If compare_priority reports that new_item should be scanned first, the "
        "baseline position is immediately before item_i.",

        "If compare_priority reports that item_i should be scanned first, leave "
        "item_i before new_item and continue to the next queue item.",

        "If compare_priority reports a tie, call compare_wait_time(new_item, "
        "item_i). If the wait-time comparison reports that new_item should be "
        "scanned first, the baseline position is immediately before item_i. If "
        "it reports that item_i should be scanned first, continue to the next "
        "queue item.",

        "If both tools report a tie, keep the existing item before new_item because "
        "the existing item is already in the queue and new_item has just arrived.",

        "If new_item should come after every existing item under priority/FCFS, "
        "the baseline position is k+1. If the queue is empty, return position 1 "
        "without making any tool calls.",

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
                f"{format_target(r.priority_tier)}, approximate scan duration {r.scan_duration_min} min"
            )
    else:
        user_lines.append(
            f"Current queue, already sorted into scan order, labelled item_1 (next to "
            f"be scanned) through item_{len(snapshot)} (last to be scanned):"
        )
        for i, r in enumerate(snapshot.itertuples(), start=1):
            user_lines.append(
                f"item_{i}: priority {r.priority_tier}, waited {r.waiting_so_far_min} min so far, "
                f"{format_target(r.priority_tier)}, approximate scan duration {r.scan_duration_min} min"
            )
    user_lines += [
        "",
        f"new_item: priority {new_item['priority_tier']}, waited 0 min so far, "
        f"{format_target(new_item['priority_tier'])}, approximate scan duration "
        f"{clean_scan_duration(new_item['exam_duration_min_approx']):.1f} min",
        "",
        "Use the tools to work out the baseline priority/FCFS position, then use "
        "evaluate_position to choose the target-aware final position. ",
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
    "You answered without calling any tools. You must use the structured tools: "
    "compare_priority and compare_wait_time for ordering comparisons, and "
    "evaluate_position to check target-time feasibility. Do not write out tool "
    "calls or their results as text yourself. Call compare_priority(new_item, "
    "item_1) now."
)

MAX_PROTOCOL_RETRIES = 3

NUDGE_FAKE_TOOL_CALL = (
    "Your previous response was ordinary text, not a structured tool call. "
    "Use the native tool interface now. Do not describe the call in text."
)

FAKE_TOOL_CALL_PATTERN = re.compile(
    r"tool\s*call\s*[:\-]|compare_priority\s*[\(\{]|compare_wait_time\s*[\(\{]|evaluate_position\s*[\(\{]",
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


def run_no_tools_attempt(model, system_prompt, user_prompt, correct_position, accepted_positions):
    r = run_no_tools_once(model, system_prompt, user_prompt)
    r["position_score"] = score_position(r["model_position"], correct_position, accepted_positions)
    r["verdict"] = grade_position(r["model_position"], correct_position, accepted_positions)
    return r


def run_tools_attempt(model, system_prompt, user_prompt, item_lookup, correct_position, accepted_positions, max_rounds):
    outcome = run_tools_once(model, system_prompt, user_prompt, item_lookup, max_rounds=max_rounds)
    if outcome["no_tool_support"]:
        verdict, model_position = "NO_TOOL_SUPPORT", None
    elif outcome["hit_round_cap"]:
        verdict, model_position = "INCOMPLETE", None
    else:
        model_position = extract_stated_position(outcome["content"])
        verdict = grade_position(model_position, correct_position, accepted_positions)
    return {
        "model_position": model_position, "verdict": verdict,
        "position_score": score_position(model_position, correct_position, accepted_positions),
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


def run_case(model, tool_mode, prompt_variant, snapshot, new_item, correct_position, duration_stats,
             start_run_index=1, end_run_index=None):
    if end_run_index is None:
        end_run_index = N_REPEATS_BY_MODEL[model]
    case_meta = target_case_metadata(snapshot, new_item)
    accepted_positions = case_meta["accepted_positions"]
    runs = []
    if tool_mode == "no-tools":
        system_prompt, user_prompt = build_prompt_no_tools(
            snapshot, new_item, duration_stats, prompt_variant=prompt_variant
        )
        for run_idx in range(start_run_index, end_run_index + 1):
            r = run_with_retries(lambda: run_no_tools_attempt(
                model, system_prompt, user_prompt, correct_position, accepted_positions
            ))
            r["run_index"] = run_idx
            runs.append(r)
    else:
        system_prompt, user_prompt = build_prompt_tools(snapshot, new_item, duration_stats)
        item_lookup = build_item_lookup(snapshot, new_item)
        max_rounds = tool_round_cap(len(snapshot))
        for run_idx in range(start_run_index, end_run_index + 1):
            r = run_with_retries(lambda: run_tools_attempt(
                model, system_prompt, user_prompt, item_lookup, correct_position, accepted_positions, max_rounds
            ))
            r["run_index"] = run_idx
            runs.append(r)
    return system_prompt, user_prompt, runs


def target_case_metadata(snapshot, new_item):
    base_pos = baseline_position(snapshot, new_item)
    correct_position, chosen_eval = choose_target_aware_position(snapshot, new_item)
    accepted_positions = chosen_eval["accepted_positions"]
    return {
        "baseline_position": base_pos,
        "one_down_position": chosen_eval["one_down_position"],
        "ideal_position": correct_position,
        "accepted_positions": accepted_positions,
        "correct_position": correct_position,
        "new_item_target_min": target_for_priority(new_item["priority_tier"]),
        "projected_new_item_wait_min": chosen_eval["new_item_projected_wait_min"],
        "target_met_at_correct_position": chosen_eval["new_item_target_met"],
        "n_higher_priority_cut_ahead": chosen_eval["n_higher_priority_cut_ahead"],
        "higher_priority_breach_if_chosen": chosen_eval["higher_priority_breach_if_chosen"],
        "selection_reason": chosen_eval["selection_reason"],
        "ground_truth_reason": chosen_eval["selection_reason"],
        "chosen_position_evaluation": chosen_eval,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def checkpoint_key(rec):
    return (
        rec["model"],
        rec["tool_mode"],
        rec.get("prompt_variant", "more_details"),
        rec["order_id"],
    )


def load_run_counts(jsonl_path):
    """(model, tool_mode, prompt_variant, order_id) -> number of runs already checkpointed.

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
                key = checkpoint_key(rec)
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
    prompt_variant, order_id)."""
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
            key = checkpoint_key(rec)
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
        accepted_positions = rec.get("accepted_positions")
        if accepted_positions is None:
            accepted_positions = [rec.get("correct_position")]
        run_scores = [
            r.get(
                "position_score",
                score_position(r.get("model_position"), rec["correct_position"], accepted_positions),
            )
            for r in runs
        ]
        n_score_1 = sum(1 for s in run_scores if s == 1)
        n_score_0 = sum(1 for s in run_scores if s == 0)
        n_score_neg1 = sum(1 for s in run_scores if s == -1)
        counts = Counter(positions)
        most_common_position, most_common_count = counts.most_common(1)[0] if counts else (None, 0)
        cases_rows.append({
            "model": rec["model"], "tool_mode": rec["tool_mode"],
            "prompt_variant": rec.get("prompt_variant", "more_details"), "case_index": rec["case_index"],
            "label": rec["label"], "queue_length_bucket": rec["queue_length_bucket"],
            "new_item_priority": rec["new_item_priority"], "n_waiting": rec["n_waiting"],
            "baseline_position": rec.get("baseline_position"),
            "one_down_position": rec.get("one_down_position"),
            "ideal_position": rec.get("ideal_position", rec["correct_position"]),
            "accepted_positions": json.dumps(accepted_positions),
            "correct_position": rec["correct_position"], "order_id": rec["order_id"],
            "scan_duration_median_min": rec.get("scan_duration_median_min"),
            "scan_duration_sd_min": rec.get("scan_duration_sd_min"),
            "scan_duration_lower_2sd_min": rec.get("scan_duration_lower_2sd_min"),
            "scan_duration_upper_2sd_min": rec.get("scan_duration_upper_2sd_min"),
            "new_item_target_min": rec.get("new_item_target_min"),
            "projected_new_item_wait_min": rec.get("projected_new_item_wait_min"),
            "target_met_at_correct_position": rec.get("target_met_at_correct_position"),
            "n_higher_priority_cut_ahead": rec.get("n_higher_priority_cut_ahead"),
            "higher_priority_breach_if_chosen": rec.get("higher_priority_breach_if_chosen"),
            "position_shift_vs_baseline": (
                rec["correct_position"] - rec["baseline_position"]
                if rec.get("baseline_position") is not None else None
            ),
            "selection_reason": rec.get("selection_reason"),
            "ground_truth_reason": rec.get("ground_truth_reason", rec.get("selection_reason")),
            "n_runs": n_runs, "n_distinct_positions": len(counts),
            "most_common_position": most_common_position,
            "self_consistency_rate": most_common_count / n_runs if n_runs else None,
            "n_pass": n_pass, "pass_rate": n_pass / n_runs if n_runs else None,
            "n_score_1": n_score_1,
            "n_score_0": n_score_0,
            "n_score_neg1": n_score_neg1,
            "mean_position_score": sum(run_scores) / n_runs if n_runs else None,
        })
        for r, position_score in zip(runs, run_scores):
            runs_rows.append({
                "model": rec["model"], "tool_mode": rec["tool_mode"],
                "prompt_variant": rec.get("prompt_variant", "more_details"), "case_index": rec["case_index"],
                "label": rec["label"], "queue_length_bucket": rec["queue_length_bucket"],
                "order_id": rec["order_id"], "run_index": r["run_index"],
                "baseline_position": rec.get("baseline_position"),
                "one_down_position": rec.get("one_down_position"),
                "ideal_position": rec.get("ideal_position", rec["correct_position"]),
                "accepted_positions": json.dumps(accepted_positions),
                "correct_position": rec["correct_position"], "model_position": r["model_position"],
                "position_score": position_score,
                "new_item_target_min": rec.get("new_item_target_min"),
                "projected_new_item_wait_min": rec.get("projected_new_item_wait_min"),
                "target_met_at_correct_position": rec.get("target_met_at_correct_position"),
                "n_higher_priority_cut_ahead": rec.get("n_higher_priority_cut_ahead"),
                "higher_priority_breach_if_chosen": rec.get("higher_priority_breach_if_chosen"),
                "ground_truth_reason": rec.get("ground_truth_reason", rec.get("selection_reason")),
                "verdict": r["verdict"], "n_rounds": r.get("n_rounds"), "hit_round_cap": r.get("hit_round_cap"),
                "ungrounded": r.get("ungrounded", False), "n_protocol_retries": r.get("n_protocol_retries", 0),
                "content": r["content"], "thinking": r["thinking"],
            })
            for tc in r.get("tool_calls", []):
                tool_calls_rows.append({
                    "model": rec["model"], "tool_mode": rec["tool_mode"],
                    "prompt_variant": rec.get("prompt_variant", "more_details"), "case_index": rec["case_index"],
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
                mode_sub = cases[(cases["model"] == model) & (cases["tool_mode"] == tool_mode)]
                for prompt_variant in mode_sub["prompt_variant"].unique():
                    sub = mode_sub[mode_sub["prompt_variant"] == prompt_variant]
                    for bucket in BUCKET_ORDER:
                        bsub = sub[sub["queue_length_bucket"] == bucket]
                        if len(bsub) == 0:
                            continue
                        summary_rows.append({
                            "model": model, "tool_mode": tool_mode, "prompt_variant": prompt_variant,
                            "queue_length_bucket": bucket,
                            "n_cases": len(bsub), "n_runs_total": int(bsub["n_runs"].sum()),
                            "n_pass_total": int(bsub["n_pass"].sum()),
                            "n_score_1_total": int(bsub["n_score_1"].sum()),
                            "n_score_0_total": int(bsub["n_score_0"].sum()),
                            "n_score_neg1_total": int(bsub["n_score_neg1"].sum()),
                            "pass_rate": bsub["n_pass"].sum() / bsub["n_runs"].sum(),
                            "mean_position_score": bsub["mean_position_score"].mean(),
                            "mean_self_consistency_rate": bsub["self_consistency_rate"].mean(),
                        })
                    summary_rows.append({
                        "model": model, "tool_mode": tool_mode, "prompt_variant": prompt_variant,
                        "queue_length_bucket": OVERALL_LABEL,
                        "n_cases": len(sub), "n_runs_total": int(sub["n_runs"].sum()),
                        "n_pass_total": int(sub["n_pass"].sum()),
                        "n_score_1_total": int(sub["n_score_1"].sum()),
                        "n_score_0_total": int(sub["n_score_0"].sum()),
                        "n_score_neg1_total": int(sub["n_score_neg1"].sum()),
                        "pass_rate": sub["n_pass"].sum() / sub["n_runs"].sum(),
                        "mean_position_score": sub["mean_position_score"].mean(),
                        "mean_self_consistency_rate": sub["self_consistency_rate"].mean(),
                    })
    summary = pd.DataFrame(summary_rows)

    notes = pd.DataFrame([
        {"Note": f"{N_PER_BUCKET * len(QUEUE_LENGTH_BUCKETS)} shared snapshots ({N_PER_BUCKET} randomly sampled per "
                 f"queue-length bucket, random_state={RANDOM_SEED}). The sample is a plain uniform random draw "
                 "within each queue-length bucket and is reused across both prompt variants via this experiment "
                 "folder's pinned case_list.json."},
        {"Note": f"Each snapshot's exact prompt is sent N_REPEATS separate times per model/tool_mode/prompt_variant (no "
                 "temperature override -- Ollama's default sampling), to measure self-consistency in addition "
                 f"to correctness. N_REPEATS is per-model: {N_REPEATS_BY_MODEL}. "
                 "self_consistency_rate = (count of the most common stated position) / n_runs."},
        {"Note": "Reasoning is captured via Ollama's native 'think' option (POST /api/chat, think=true) as a "
                 "separate 'thinking' field, not requested inline in the answer text -- the prompt asks for "
                 "ONLY the position number."},
        {"Note": f"This experiment runs no-tools prompts only for prompt variants {PROMPT_VARIANTS}. "
                 "Tool-calling support remains in the script for older rebuilds and manual overrides, but the "
                 "default TOOL_MODES list is ['no-tools']."},
        {"Note": "The no-tools prompt hides exact per-order approximate scan durations. It shows only the "
                 "historical NNI scan-duration median and median +/- 2 SD range computed from valid "
                 "exam_duration_min_approx values."},
        {"Note": "Ground truth is computed with actual exam_duration_min_approx values from "
                 "analysis/ct_queue_reconstruction.xlsx. The ideal position is either the priority/FCFS "
                 "baseline_position or one_down_position = min(baseline_position + 1, n_waiting + 1)."},
        {"Note": "A run receives position_score=1 if the model chose ideal_position, 0 if it chose the other "
                 "accepted position among baseline_position/one_down_position, and -1 if it chose any other "
                 "position. verdict remains PASS only for score 1; unparseable outputs are retried and score "
                 "-1 if still unparseable after retries."},
        {"Note": "Cases includes n_score_1, n_score_0, and n_score_neg1 for that case's repeats. Summary includes "
                 "n_score_1_total, n_score_0_total, and n_score_neg1_total by bucket and overall."},
        {"Note": "SLA targets: P1<=30 min, P2<=120 min, P3<=360 min. Unknown-priority orders have no target; "
                 "an Unknown new_item therefore uses the baseline priority/FCFS position."},
        {"Note": f"Source checkpoint: {os.path.basename(jsonl_path)} (append-only JSONL, one line per completed "
                 "snapshot per model/tool_mode/prompt_variant; resumable -- rerunning the script skips "
                 "(model, tool_mode, prompt_variant, order_id) combinations already present)."},
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
        case_meta = target_case_metadata(snapshot, pick)
        correct_position = case_meta["correct_position"]
        rec.update(case_meta)
        system_prompt, user_prompt = rec["system_prompt"], rec["prompt"]
        model, tool_mode = rec["model"], rec["tool_mode"]
        prompt_variant = rec.get("prompt_variant", "more_details")
        for i in bad_idxs:
            run_idx = rec["runs"][i]["run_index"]
            old_verdict = rec["runs"][i]["verdict"]
            accepted_positions = rec.get("accepted_positions", [correct_position])
            if tool_mode == "no-tools":
                r = run_with_retries(lambda: run_no_tools_attempt(
                    model, system_prompt, user_prompt, correct_position, accepted_positions
                ))
            else:
                item_lookup = build_item_lookup(snapshot, pick)
                max_rounds = tool_round_cap(len(snapshot))
                r = run_with_retries(lambda: run_tools_attempt(
                    model, system_prompt, user_prompt, item_lookup, correct_position, accepted_positions, max_rounds
                ))
            r["run_index"] = run_idx
            rec["runs"][i] = r
            n_retried += 1
            print(
                f"  retried {model}/{tool_mode}/{prompt_variant} order {order_id} "
                f"run {run_idx}: {old_verdict} -> {r['verdict']}"
            )

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
    nni["exam_duration_min_approx"] = nni["exam_duration_min_approx"].apply(clean_scan_duration)
    duration_stats = scan_duration_stats(nni)
    print("Using actual exam_duration_min_approx for ground truth.")
    print(f"Prompt scan-duration range: {format_duration_range(duration_stats)}\n")

    candidates = nni[
        nni["parent_order_dt"].notna()
        & nni["exam_start_dt"].notna()
        & nni["exam_duration_min_approx"].notna()
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
        print(f"Resuming: {len(run_counts)} (model, tool_mode, prompt_variant, order_id) combo(s) already have "
              f"checkpointed run(s) in {JSONL_PATH}\n")

    total_combos = len(MODELS) * len(TOOL_MODES) * len(PROMPT_VARIANTS) * len(cases)
    done_so_far = 0
    for model in MODELS:
        n_repeats = N_REPEATS_BY_MODEL[model]
        for tool_mode in TOOL_MODES:
            for prompt_variant in PROMPT_VARIANTS:
                print("=" * 78)
                print(f"MODEL {model} / {tool_mode} / {prompt_variant}")
                for i, (label, bucket, new_item) in enumerate(cases, start=1):
                    done_so_far += 1
                    order_id = str(new_item["Order ID"])
                    key = (model, tool_mode, prompt_variant, order_id)
                    n_done = run_counts.get(key, 0)
                    if n_done >= n_repeats:
                        print(f"  [{model}/{tool_mode}/{prompt_variant}] snapshot {i}/{len(cases)} "
                              f"(order {order_id}) -- already done ({n_done}/{n_repeats} runs), skipping")
                        continue

                    t0 = time.time()
                    snapshot = build_snapshot(nni, new_item)
                    case_meta = target_case_metadata(snapshot, new_item)
                    correct_position = case_meta["correct_position"]

                    system_prompt, prompt, runs = run_case(
                        model, tool_mode, prompt_variant, snapshot, new_item, correct_position, duration_stats,
                        start_run_index=n_done + 1, end_run_index=n_repeats,
                    )

                    record = {
                        "model": model, "tool_mode": tool_mode, "prompt_variant": prompt_variant,
                        "case_index": i, "label": label,
                        "queue_length_bucket": bucket, "order_id": order_id,
                        "new_item_priority": new_item["priority_tier"], "n_waiting": len(snapshot),
                        "correct_position": correct_position, "system_prompt": system_prompt,
                        "prompt": prompt, "runs": runs,
                        "scan_duration_median_min": duration_stats["median"],
                        "scan_duration_sd_min": duration_stats["sd"],
                        "scan_duration_lower_2sd_min": duration_stats["lower_2sd"],
                        "scan_duration_upper_2sd_min": duration_stats["upper_2sd"],
                    }
                    record.update(case_meta)
                    append_record(JSONL_PATH, record)
                    run_counts[key] = n_done + len(runs)

                    verdicts = [r["verdict"] for r in runs]
                    elapsed = time.time() - t0
                    print(
                        f"  [{model}/{tool_mode}/{prompt_variant}] snapshot {i}/{len(cases)} "
                        f"(bucket={bucket}, order {order_id}) -- {verdicts} -- done in {elapsed:.0f}s "
                        f"({done_so_far}/{total_combos} overall)"
                    )

    print("\nAll model/tool_mode/snapshot combinations complete.")
    build_xlsx(JSONL_PATH, XLSX_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
