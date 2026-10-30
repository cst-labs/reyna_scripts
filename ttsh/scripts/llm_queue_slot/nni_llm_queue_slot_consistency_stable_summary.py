"""
nni_llm_queue_slot_consistency_stable_summary.py
Summary workbook for the qwen3:1.7b / gemma4:e4b queue-slotting experiment.

Accuracy (Headline / By_Bucket / By_Priority_Tier / Verdict_Breakdown /
Tool_Use_Stats) is computed from each case's first run only (run_index == 1)
for BOTH models -- one run per case, apples-to-apples. Cross-run agreement
is reported separately in the Consistency sheet, using whichever repeats
exist for each case, restricted to cases where all repeats verifiably saw
the identical queue snapshot (see below).

Two corrections vs. a naive "just group the checkpoint data" summary:

1. Only run_index == 1 is used for accuracy, for both models. gemma4:e4b's
   2nd/3rd repeats were added later in a separate top-up run, and for ~30%
   of cases the source spreadsheets had been regenerated in between, so the
   top-up saw a different queue snapshot than the original run (different
   n_waiting/correct_position -- confirmed by diffing prompt text). Using
   only run 1 sidesteps this by construction. qwen3:1.7b had zero such
   drift (all 3 repeats written in one continuous run), but is restricted to
   run 1 too, for a consistent "one run per case" accuracy methodology
   across both models.

2. Queue-length bucket is taken from the pinned case list
   (analysis/nni_llm_queue_slot_consistency_case_list.json), not from the
   "queue_length_bucket" field stored on each checkpoint record. The pinned
   list is the ground truth (exactly 50 cases per bucket, by design -- see
   the sampling step in nni_llm_queue_slot_consistency.py). 18 of qwen's 250
   already-checkpointed records (11 no-tools, 7 tools) were written *before*
   the case list was pinned, and still carry the bucket label that was true
   at that original sampling time -- which for a few cases no longer matches
   the now-pinned bucket because the underlying queue for that order_id had
   already drifted by the time the pin file was written. Re-deriving the
   bucket from the pin file (rather than trusting the stored field) is what
   makes every model/tool_mode combo show exactly 50/50/50/50/50.

Reads:
  analysis/nni_llm_queue_slot_consistency_case_list.json   (ground-truth order_id -> bucket)
  analysis/nni_llm_queue_slot_consistency_raw.jsonl        (to classify stable/drifted repeats)
  analysis/nni_llm_queue_slot_consistency_results.xlsx     (Cases, Runs, Tool_Calls)

Outputs: analysis/nni_llm_queue_slot_consistency_stable_summary.xlsx
"""
import json
import os

import pandas as pd
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS = os.path.join(BASE, "analysis")
CASE_LIST_PATH = os.path.join(ANALYSIS, "nni_llm_queue_slot_consistency_case_list.json")
RESULTS_PATH = os.path.join(ANALYSIS, "nni_llm_queue_slot_consistency_results.xlsx")
OUT_PATH = os.path.join(ANALYSIS, "nni_llm_queue_slot_consistency_stable_summary.xlsx")

BUCKET_ORDER = [
    "tiny (0-1 waiting)", "small (2-3 waiting)", "medium (4-5 waiting)",
    "large (6-7 waiting)", "huge (8+ waiting)",
]
TIER_ORDER = ["P1", "P2", "P3", "Unknown"]


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 60)


def write_table(writer, sheet_name, formula_note, df):
    """Writes a one-line 'how this is calculated' note, then the table two rows below."""
    pd.DataFrame({"how this table is calculated": [formula_note]}).to_excel(
        writer, sheet_name=sheet_name, index=False, header=False
    )
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)


# ---------------------------------------------------------------------------
# Ground-truth order_id -> bucket, from the pinned case list.
# ---------------------------------------------------------------------------
pinned = json.load(open(CASE_LIST_PATH))
pinned_bucket = {e["order_id"]: e["queue_length_bucket"] for e in pinned}
print(f"Pinned case list: {len(pinned)} cases, "
      f"{pd.Series(pinned_bucket.values()).value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# Load Cases / Runs / Tool_Calls.
# ---------------------------------------------------------------------------
cases = pd.read_excel(RESULTS_PATH, sheet_name="Cases")
runs = pd.read_excel(RESULTS_PATH, sheet_name="Runs")
tool_calls = pd.read_excel(RESULTS_PATH, sheet_name="Tool_Calls")

for df in (cases, runs, tool_calls):
    df["order_id"] = df["order_id"].astype(str)

# Overwrite the stored (possibly stale) bucket with the pinned ground truth.
for df in (cases, runs):
    df["queue_length_bucket"] = df["order_id"].map(pinned_bucket)

# ---------------------------------------------------------------------------
# Scope: accuracy tables use run_index == 1 only, for both models.
# ---------------------------------------------------------------------------
runs_kept = runs[runs["run_index"] == 1].copy()
tool_calls_kept = tool_calls[tool_calls["run_index"] == 1].copy()

case_meta = cases[["model", "tool_mode", "order_id", "new_order_priority"]].drop_duplicates()
runs_kept = runs_kept.merge(case_meta, on=["model", "tool_mode", "order_id"], how="left")

print("\nn_cases / n_runs kept per model/tool_mode:")
print(runs_kept.groupby(["model", "tool_mode"]).agg(
    n_cases=("order_id", "nunique"), n_runs=("verdict", "size")
).to_string())

bucket_counts = runs_kept.drop_duplicates(["model", "tool_mode", "order_id"]).groupby(
    ["model", "tool_mode", "queue_length_bucket"]
).size()
print("\nn_cases per bucket per model/tool_mode (should be exactly 50 everywhere):")
print(bucket_counts.to_string())

# ---------------------------------------------------------------------------
# Headline: pass rate (n_pass / n_runs)
# ---------------------------------------------------------------------------
headline = runs_kept.groupby(["model", "tool_mode"]).agg(
    n_cases=("order_id", "nunique"),
    n_runs=("verdict", "size"),
    n_pass=("verdict", lambda s: (s == "PASS").sum()),
).reset_index()
headline["pass_rate"] = headline["n_pass"] / headline["n_runs"]
headline = headline.sort_values(["model", "tool_mode"])
HEADLINE_NOTE = (
    "pass_rate = n_pass / n_runs, where n_pass counts runs whose verdict == 'PASS' "
    "(model_position exactly equal to correct_position). n_runs = n_cases (run_index == 1 only, "
    "for both models -- one run per case). See the Consistency sheet for cross-run agreement."
)

# ---------------------------------------------------------------------------
# Pass rate by queue-length bucket (pinned bucket, not the stored field)
# ---------------------------------------------------------------------------
by_bucket = runs_kept.groupby(["model", "tool_mode", "queue_length_bucket"]).agg(
    n_cases=("order_id", "nunique"),
    n_runs=("verdict", "size"),
    n_pass=("verdict", lambda s: (s == "PASS").sum()),
).reset_index()
by_bucket["pass_rate"] = by_bucket["n_pass"] / by_bucket["n_runs"]
bucket_rank = {b: i for i, b in enumerate(BUCKET_ORDER)}
by_bucket = by_bucket.sort_values(
    ["model", "tool_mode", "queue_length_bucket"],
    key=lambda s: s.map(bucket_rank) if s.name == "queue_length_bucket" else s,
)
BY_BUCKET_NOTE = (
    "Same pass_rate formula as Headline, grouped by queue_length_bucket. Bucket is taken from "
    "the pinned case list (order_id -> bucket, fixed at sampling time), not from the bucket field "
    "stored on each run -- so n_cases is exactly 50 per bucket for every model/tool_mode."
)

# ---------------------------------------------------------------------------
# Pass rate by the new order's own priority tier
# ---------------------------------------------------------------------------
by_priority = runs_kept.groupby(["model", "tool_mode", "new_order_priority"]).agg(
    n_cases=("order_id", "nunique"),
    n_runs=("verdict", "size"),
    n_pass=("verdict", lambda s: (s == "PASS").sum()),
).reset_index()
by_priority["pass_rate"] = by_priority["n_pass"] / by_priority["n_runs"]
tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
by_priority = by_priority.sort_values(
    ["model", "tool_mode", "new_order_priority"],
    key=lambda s: s.map(tier_rank) if s.name == "new_order_priority" else s,
)
BY_PRIORITY_NOTE = (
    "Same pass_rate formula as Headline, grouped by new_order_priority -- the priority tier "
    "(P1/P2/P3/Unknown) of the new order being slotted into the queue, not of the orders "
    "already waiting. Bucket sizes are the natural mix of tiers within the random sample, not "
    "controlled to be equal."
)

# ---------------------------------------------------------------------------
# Verdict breakdown
# ---------------------------------------------------------------------------
verdict_breakdown = (
    runs_kept.groupby(["model", "tool_mode", "verdict"]).size().rename("n_runs").reset_index()
).sort_values(["model", "tool_mode", "verdict"])
VERDICT_NOTE = (
    "Raw count of runs by verdict. PASS = model_position == correct_position exactly. "
    "FAIL = a numeric position was extracted but it was wrong. UNPARSEABLE = no position number "
    "could be extracted from the model's answer. INCOMPLETE = tool-calling run hit the "
    "MAX_TOOL_ROUNDS cap without producing a final answer."
)

# ---------------------------------------------------------------------------
# Tool-use stats (tool-calling mode only)
# ---------------------------------------------------------------------------
tool_runs = runs_kept[runs_kept["tool_mode"] == "tools"].copy()
tool_calls_per_run = tool_calls_kept.groupby(["model", "tool_mode", "order_id", "run_index"]).size().rename(
    "n_tool_calls"
).reset_index()

rounds_stats = tool_runs.groupby("model").agg(
    mean_rounds_per_run=("n_rounds", "mean"),
    hit_round_cap_sum=("hit_round_cap", "sum"),
).reset_index()

calls_stats = tool_calls_per_run.groupby("model")["n_tool_calls"].agg(
    mean_calls_per_run="mean", median_calls_per_run="median", max_calls_per_run="max",
).reset_index()

tool_calls_kept = tool_calls_kept.copy()
tool_calls_kept["is_error"] = tool_calls_kept["result"].astype(str).str.contains("error", case=False, na=False)
error_stats = tool_calls_kept.groupby("model")["is_error"].agg(
    n_errors="sum", n_tool_calls="count", error_rate="mean"
).reset_index()

tool_call_name_counts = (
    tool_calls_kept.groupby(["model", "tool_name"]).size().rename("n_calls").reset_index()
).sort_values(["model", "n_calls"], ascending=[True, False])

tool_use_stats = rounds_stats.merge(calls_stats, on="model").merge(error_stats, on="model")
tool_use_stats = tool_use_stats[[
    "model", "mean_rounds_per_run", "hit_round_cap_sum",
    "mean_calls_per_run", "median_calls_per_run", "max_calls_per_run",
    "n_tool_calls", "n_errors", "error_rate",
]]
TOOL_USE_NOTE = (
    "mean/hit_round_cap_sum computed over tool-mode runs only (1 row/run, from Runs.n_rounds and "
    "Runs.hit_round_cap). mean/median/max_calls_per_run = tool-call rows grouped by (order_id, "
    "run_index), then averaged across runs. error_rate = n_errors / n_tool_calls, where an error "
    "is a tool-call result string containing the word 'error'. Table below the stats is total "
    "calls per tool_name."
)

print("\nHeadline:")
print(headline.to_string(index=False))
print("\nTool-use stats:")
print(tool_use_stats.to_string(index=False))

# ---------------------------------------------------------------------------
# Consistency: agreement across a case's repeats, restricted to cases where
# every checkpointed line for that (model, tool_mode, order_id) agrees on
# n_waiting and correct_position -- i.e. every repeat genuinely saw the same
# snapshot. Cases that fail this (source spreadsheets regenerated between an
# original run and a later top-up) are excluded and counted separately,
# since cross-run agreement is only meaningful when the runs answered the
# same question.
# ---------------------------------------------------------------------------
lines_by_key = {}
with open(os.path.join(ANALYSIS, "nni_llm_queue_slot_consistency_raw.jsonl")) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        key = (rec["model"], rec["tool_mode"], rec["order_id"])
        lines_by_key.setdefault(key, []).append(rec)

stable_keys, drifted_keys = set(), set()
for key, recs in lines_by_key.items():
    n_waiting_vals = {r["n_waiting"] for r in recs}
    correct_pos_vals = {r["correct_position"] for r in recs}
    if len(n_waiting_vals) == 1 and len(correct_pos_vals) == 1:
        stable_keys.add(key)
    else:
        drifted_keys.add(key)

cases_stable = cases[cases.apply(
    lambda r: (r["model"], r["tool_mode"], r["order_id"]) in stable_keys, axis=1
)].copy()

consistency = cases_stable.groupby(["model", "tool_mode"]).agg(
    n_cases=("order_id", "size"),
    mean_n_runs=("n_runs", "mean"),
    mean_self_consistency=("self_consistency_rate", "mean"),
    n_fully_consistent=("n_distinct_positions", lambda s: (s == 1).sum()),
).reset_index()
consistency["pct_fully_consistent"] = consistency["n_fully_consistent"] / consistency["n_cases"]

drifted_counts = pd.Series([f"{m}/{t}" for m, t, _ in drifted_keys]).value_counts()
n_drifted_excluded = pd.DataFrame({
    "model_tool_mode": drifted_counts.index, "n_cases_excluded_drifted": drifted_counts.values
})

consistency_dist = (
    cases_stable.groupby(["model", "tool_mode", "n_distinct_positions"]).size()
    .rename("n_cases").reset_index()
).sort_values(["model", "tool_mode", "n_distinct_positions"])

CONSISTENCY_NOTE = (
    "Cross-run agreement across each case's repeats (qwen: 3 runs/case; gemma: up to 3 runs/case), "
    "restricted to cases where every checkpointed repeat verifiably saw the identical queue "
    "snapshot (n_waiting and correct_position agree across all its checkpoint lines) -- cases "
    "that fail this are excluded here and counted in the table below. self_consistency_rate per "
    "case = (count of the most-common model_position across its runs) / n_runs; "
    "mean_self_consistency averages that across cases. pct_fully_consistent = share of cases "
    "where n_distinct_positions == 1 (every repeat gave the exact same position)."
)

print("\nConsistency (stable cases only):")
print(consistency.to_string(index=False))
if len(n_drifted_excluded):
    print("\nCases excluded from Consistency due to snapshot drift between repeats:")
    print(n_drifted_excluded.to_string(index=False))

# ---------------------------------------------------------------------------
# Notes sheet
# ---------------------------------------------------------------------------
notes_text = pd.DataFrame({"note": [
    "Accuracy tables (Headline, By_Bucket, By_Priority_Tier, Verdict_Breakdown, Tool_Use_Stats) "
    "use run_index == 1 only, for both models -- one run per case, apples-to-apples.",
    "The Consistency sheet separately reports cross-run agreement using each case's available "
    "repeats, restricted to cases where all repeats verifiably saw the same snapshot (see its "
    "own note). gemma4:e4b's 2nd/3rd repeats were added later in a separate top-up run, and for "
    "~30% of its cases the source spreadsheets had been regenerated in between, so those cases "
    "are excluded from Consistency (counted separately) since their repeats answered different "
    "questions, not just the same question answered differently. qwen3:1.7b had zero such drift.",
    "queue_length_bucket everywhere below is looked up from the pinned case list "
    "(nni_llm_queue_slot_consistency_case_list.json), which fixes exactly 50 cases per bucket "
    "by design. 18 of qwen's 250 checkpointed records (11 no-tools, 7 tools) were written before "
    "the case list was pinned and still carry a stale bucket label from their original sampling "
    "time; using the pin file instead of the stored field is what makes n_cases sum to exactly "
    "50/50/50/50/50 for every model/tool_mode in the By_Bucket sheet.",
]})

# ---------------------------------------------------------------------------
# Write workbook
# ---------------------------------------------------------------------------
with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    notes_text.to_excel(writer, sheet_name="Notes", index=False)
    write_table(writer, "Headline", HEADLINE_NOTE, headline)
    write_table(writer, "By_Bucket", BY_BUCKET_NOTE, by_bucket)
    write_table(writer, "By_Priority_Tier", BY_PRIORITY_NOTE, by_priority)
    write_table(writer, "Verdict_Breakdown", VERDICT_NOTE, verdict_breakdown)
    write_table(writer, "Tool_Use_Stats", TOOL_USE_NOTE, tool_use_stats)
    tool_call_name_counts.to_excel(
        writer, sheet_name="Tool_Use_Stats", index=False, startrow=len(tool_use_stats) + 5
    )
    write_table(writer, "Consistency", CONSISTENCY_NOTE, consistency)
    consistency_dist.to_excel(writer, sheet_name="Consistency", index=False, startrow=len(consistency) + 5)
    n_drifted_excluded.to_excel(
        writer, sheet_name="Consistency", index=False,
        startrow=len(consistency) + 5 + len(consistency_dist) + 3,
    )
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print(f"\nWrote {OUT_PATH}")
