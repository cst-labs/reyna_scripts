"""
nni_llm_queue_slot_consistency_align_snapshots.py
One-off fix: for the small number of order_ids where the 4 (model, tool_mode)
combos currently disagree on which queue snapshot they used (qwen vs gemma,
or no-tools vs tools, diverged because each combo was actually run at a
different real-world moment while the live source data kept changing),
pick the majority snapshot per order_id and redo the minority combos'
3 repeats from scratch against that exact same snapshot.

The canonical snapshot's item-level content (priority tier + wait time per
item) is recovered by parsing it straight out of an existing combo's own
stored prompt text -- not re-derived from the live source spreadsheet, which
has since moved on and no longer reproduces these snapshots.

Backs up the raw JSONL first, then removes the old line(s) for each fixed
combo and appends one fresh merged record in their place.
"""
import json
import re
import shutil
from collections import Counter

import pandas as pd

import nni_llm_queue_slot_consistency as base

ITEM_RE = re.compile(r"^(?:Item|item_)\s?(\d+):\s*priority\s+(\w+)(?:\s*\(score\s*\d+\))?,\s*waited\s+([\d.]+)\s*min so far")
MEDIAN_RE = re.compile(r"median scan time\s+([\d.]+)\s*min")

COMBOS = [
    ("qwen3:1.7b", "no-tools"), ("qwen3:1.7b", "tools"),
    ("gemma4:e4b", "no-tools"), ("gemma4:e4b", "tools"),
]


def parse_items(prompt):
    items = []
    for line in prompt.splitlines():
        m = ITEM_RE.match(line.strip())
        if m:
            items.append({"priority_tier": m.group(2), "waiting_so_far_min": float(m.group(3))})
    m = MEDIAN_RE.search(prompt)
    median_scan_time_min = float(m.group(1)) if m else None
    return items, median_scan_time_min


def load_lines(path):
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def main():
    records = load_lines(base.JSONL_PATH)
    recs_by_key = {}
    for rec in records:
        key = (rec["model"], rec["tool_mode"], rec["order_id"])
        recs_by_key.setdefault(key, []).append(rec)

    pinned = json.load(open(base.CASE_LIST_PATH))
    order_ids = [e["order_id"] for e in pinned]
    label_bucket = {e["order_id"]: (e["label"], e["queue_length_bucket"]) for e in pinned}

    to_fix = []  # list of (order_id, model, tool_mode, canonical_n_waiting, canonical_correct_position, source_rec)
    for oid in order_ids:
        combo_val = {}
        for model, tm in COMBOS:
            r0 = recs_by_key[(model, tm, oid)][0]
            combo_val[(model, tm)] = (r0["n_waiting"], r0["correct_position"])
        counts = Counter(combo_val[c] for c in COMBOS)
        canonical = counts.most_common(1)[0][0]
        source_combo = next(c for c in COMBOS if combo_val[c] == canonical)
        source_rec = recs_by_key[(source_combo[0], source_combo[1], oid)][0]
        for model, tm in COMBOS:
            if combo_val[(model, tm)] != canonical:
                to_fix.append((oid, model, tm, canonical[0], canonical[1], source_rec))

    print(f"{len(to_fix)} combo(s) to redo (out of {len(order_ids) * 4})")
    if not to_fix:
        print("Nothing to do.")
        return

    shutil.copy(base.JSONL_PATH, base.JSONL_PATH + ".bak_pre_align_snapshots")

    for i, (oid, model, tm, n_waiting, correct_position, source_rec) in enumerate(to_fix, start=1):
        items, median_scan_time_min = parse_items(source_rec["prompt"])
        assert len(items) == n_waiting, f"{oid}: parsed {len(items)} items, expected {n_waiting}"

        for it in items:
            it["priority_score"] = base.TIER_RANK[it["priority_tier"]]
        snapshot = pd.DataFrame(items)

        new_order_tier = source_rec["new_order_priority"]
        new_order = pd.Series({"priority_tier": new_order_tier, "priority_score": base.TIER_RANK[new_order_tier]})

        label, bucket = label_bucket[oid]
        t0 = base.time.time()
        prompt, runs = base.run_case(model, tm, snapshot, new_order, correct_position, median_scan_time_min)
        elapsed = base.time.time() - t0

        new_record = {
            "model": model, "tool_mode": tm, "case_index": None, "label": label,
            "queue_length_bucket": bucket, "order_id": oid,
            "new_order_priority": new_order_tier, "n_waiting": n_waiting,
            "correct_position": correct_position, "prompt": prompt, "runs": runs,
        }

        # Remove old line(s) for this combo, append the new merged one.
        key = (model, tm, oid)
        remaining = [r for r in records if (r["model"], r["tool_mode"], r["order_id"]) != key]
        remaining.append(new_record)
        with open(base.JSONL_PATH, "w") as f:
            for r in remaining:
                f.write(json.dumps(r) + "\n")
        records = remaining

        verdicts = [r["verdict"] for r in runs]
        print(f"[{i}/{len(to_fix)}] {model}/{tm} order {oid} -> {verdicts} ({elapsed:.0f}s)")

    print("\nDone realigning. Rebuilding results.xlsx ...")
    base.build_xlsx(base.JSONL_PATH, base.XLSX_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
