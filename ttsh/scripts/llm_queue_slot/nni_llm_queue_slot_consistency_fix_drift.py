"""
nni_llm_queue_slot_consistency_fix_drift.py
Regenerates gemma4:e4b's drifted repeat runs by replaying each case's
*original* run-1 prompt verbatim (and, for tool-calling mode, an
item_lookup reconstructed straight from that same prompt text) -- instead
of re-deriving the snapshot from the live source spreadsheet, which has
moved on since these cases were first run and no longer reproduces the
original snapshot (verified: 0/147 match live data as of the drift
investigation). The prompt text itself is a complete, exact record of the
original snapshot (every item's priority tier + wait time), so no
information is lost by reusing it instead of the live source.

Only touches drifted gemma4:e4b keys -- i.e. (model, tool_mode, order_id)
combos where the JSONL's checkpointed lines disagree on n_waiting/
correct_position. Everything else (qwen3:1.7b, and gemma4:e4b's already-
stable cases) is left untouched. Run 1's own line is never modified --
only the missing repeats are generated and appended, keyed to run 1's
snapshot and correct_position.

Reads/writes: analysis/nni_llm_queue_slot_consistency_raw.jsonl
(backs up to analysis/nni_llm_queue_slot_consistency_raw.jsonl.bak_pre_drift_fix
before making any change).
"""
import json
import os
import re
import shutil
import sys

from nni_llm_queue_slot_consistency import (
    JSONL_PATH, N_REPEATS_BY_MODEL, TIER_RANK,
    run_no_tools_once, run_tools_once, extract_stated_position, grade,
)

ITEM_LINE = re.compile(r"^item_(\d+): priority (\w+), waited ([\d.]+) min so far")
NEW_ORDER_LINE = re.compile(r"^new_order: priority (\w+), waited ([\d.]+) min so far")

LIMIT = int(os.environ.get("FIX_DRIFT_LIMIT", "0")) or None


def parse_item_lookup(prompt):
    lookup = {}
    for line in prompt.splitlines():
        line = line.strip()
        m = ITEM_LINE.match(line)
        if m:
            idx, tier, waited = m.group(1), m.group(2), float(m.group(3))
            lookup[f"item_{idx}"] = {
                "priority_score": TIER_RANK[tier], "priority_tier": tier, "waiting_so_far_min": waited,
            }
            continue
        m = NEW_ORDER_LINE.match(line)
        if m:
            tier, waited = m.group(1), float(m.group(2))
            lookup["new_order"] = {
                "priority_score": TIER_RANK[tier], "priority_tier": tier, "waiting_so_far_min": waited,
            }
    return lookup


def main():
    lines = open(JSONL_PATH).readlines()
    recs_by_key, idxs_by_key = {}, {}
    for i, line in enumerate(lines):
        line_s = line.strip()
        if not line_s:
            continue
        rec = json.loads(line_s)
        key = (rec["model"], rec["tool_mode"], rec["order_id"])
        recs_by_key.setdefault(key, []).append(rec)
        idxs_by_key.setdefault(key, []).append(i)

    drifted = []
    for key, recs in recs_by_key.items():
        if key[0] != "gemma4:e4b":
            continue
        if len({r["n_waiting"] for r in recs}) > 1 or len({r["correct_position"] for r in recs}) > 1:
            drifted.append(key)

    if LIMIT:
        drifted = drifted[:LIMIT]
    print(f"{len(drifted)} drifted gemma4:e4b key(s) to fix\n")

    backup_path = JSONL_PATH + ".bak_pre_drift_fix"
    if not os.path.exists(backup_path):
        shutil.copy(JSONL_PATH, backup_path)
        print(f"Backed up original JSONL to {backup_path}\n")

    drop_idxs = set()
    new_lines = []
    for n, key in enumerate(drifted, start=1):
        model, tool_mode, order_id = key
        recs = recs_by_key[key]
        drop_idxs.update(idxs_by_key[key])

        base = recs[0]
        prompt = base["prompt"]
        correct_position = base["correct_position"]
        done_indices = {r["run_index"] for r in base["runs"]}
        n_repeats = N_REPEATS_BY_MODEL[model]
        missing = [i for i in range(1, n_repeats + 1) if i not in done_indices]

        if not missing:
            print(f"  [{n}/{len(drifted)}] {model}/{tool_mode} order {order_id} -- run 1 already has all "
                  f"{n_repeats} repeats?? skipping")
            new_lines.append(json.dumps(base) + "\n")
            continue

        item_lookup = parse_item_lookup(prompt) if tool_mode == "tools" else None

        new_runs = list(base["runs"])
        for run_idx in missing:
            if tool_mode == "no-tools":
                r = run_no_tools_once(model, prompt)
                r["run_index"] = run_idx
                r["verdict"] = grade(r["model_position"], correct_position)
            else:
                content, thinking, n_rounds, hit_cap, tool_call_log, round_log = run_tools_once(model, prompt, item_lookup)
                if hit_cap:
                    verdict, model_position = "INCOMPLETE", None
                else:
                    model_position = extract_stated_position(content)
                    verdict = grade(model_position, correct_position)
                r = {
                    "run_index": run_idx, "model_position": model_position, "verdict": verdict,
                    "content": content, "thinking": thinking, "n_rounds": n_rounds, "hit_round_cap": hit_cap,
                    "tool_calls": tool_call_log, "rounds": round_log,
                }
            new_runs.append(r)
            print(f"  [{n}/{len(drifted)}] {model}/{tool_mode} order {order_id} run {run_idx} -> {r['verdict']}")

        new_record = dict(base)
        new_record["runs"] = new_runs
        new_lines.append(json.dumps(new_record) + "\n")

    kept = [line for i, line in enumerate(lines) if i not in drop_idxs]
    with open(JSONL_PATH, "w") as f:
        f.writelines(kept)
        f.writelines(new_lines)

    print(f"\nDone. Replaced {len(drop_idxs)} old line(s) with {len(new_lines)} consolidated, now-stable line(s).")


if __name__ == "__main__":
    main()
