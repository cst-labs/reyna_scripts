"""
nni_llm_queue_slot_consistency_gemma_run1_summary.py
Summary workbook variant that drops self-consistency entirely and, for
gemma4:e4b specifically, uses only each case's original run (run_index == 1)
rather than pooling across its 3 repeats -- avoids the top-up snapshot-drift
issue by construction, since only one (the original) run per gemma case is
ever included. qwen3:1.7b is unaffected (no drift, all 3 repeats retained
and pooled as before).

Reads: analysis/nni_llm_queue_slot_consistency_results.xlsx (Cases, Runs, Tool_Calls)
Outputs: analysis/nni_llm_queue_slot_consistency_gemma_run1_summary.xlsx
"""
import os

import pandas as pd
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS = os.path.join(BASE, "analysis")
RESULTS_PATH = os.path.join(ANALYSIS, "nni_llm_queue_slot_consistency_results.xlsx")
OUT_PATH = os.path.join(ANALYSIS, "nni_llm_queue_slot_consistency_gemma_run1_summary.xlsx")

BUCKET_ORDER = [
    "tiny (0-1 waiting)", "small (2-3 waiting)", "medium (4-5 waiting)",
    "large (6-7 waiting)", "huge (8+ waiting)",
]
TIER_ORDER = ["P1", "P2", "P3", "Unknown"]


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 60)


cases = pd.read_excel(RESULTS_PATH, sheet_name="Cases")
runs = pd.read_excel(RESULTS_PATH, sheet_name="Runs")
tool_calls = pd.read_excel(RESULTS_PATH, sheet_name="Tool_Calls")

for df in (cases, runs, tool_calls):
    df["order_id"] = df["order_id"].astype(str)

# ---------------------------------------------------------------------------
# Scope: qwen3:1.7b keeps all 3 repeats (no drift); gemma4:e4b keeps only its
# original run_index == 1 (its own repeats can be dropped without touching
# the drift question -- each case still reflects exactly one, real run).
# ---------------------------------------------------------------------------
runs_kept = pd.concat([
    runs[(runs["model"] == "qwen3:1.7b")],
    runs[(runs["model"] == "gemma4:e4b") & (runs["run_index"] == 1)],
], ignore_index=True)

tool_calls_kept = pd.concat([
    tool_calls[(tool_calls["model"] == "qwen3:1.7b")],
    tool_calls[(tool_calls["model"] == "gemma4:e4b") & (tool_calls["run_index"] == 1)],
], ignore_index=True)

case_meta = cases[["model", "tool_mode", "order_id", "new_order_priority"]].drop_duplicates()
runs_kept = runs_kept.merge(case_meta, on=["model", "tool_mode", "order_id"], how="left")

print("n_runs kept per model/tool_mode:")
print(runs_kept.groupby(["model", "tool_mode"]).size().to_string())

# ---------------------------------------------------------------------------
# Headline: pass rate only (no self-consistency)
# ---------------------------------------------------------------------------
headline = runs_kept.groupby(["model", "tool_mode"]).agg(
    n_cases=("order_id", "nunique"),
    n_runs=("verdict", "size"),
    n_pass=("verdict", lambda s: (s == "PASS").sum()),
).reset_index()
headline["pass_rate"] = headline["n_pass"] / headline["n_runs"]
headline = headline.sort_values(["model", "tool_mode"])

print("\nHeadline:")
print(headline.to_string(index=False))

# ---------------------------------------------------------------------------
# Pass rate by queue-length bucket
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

# ---------------------------------------------------------------------------
# Verdict breakdown
# ---------------------------------------------------------------------------
verdict_breakdown = (
    runs_kept.groupby(["model", "tool_mode", "verdict"]).size().rename("n_runs").reset_index()
).sort_values(["model", "tool_mode", "verdict"])

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

print("\nTool-use stats:")
print(tool_use_stats.to_string(index=False))
print("\nTool call name counts:")
print(tool_call_name_counts.to_string(index=False))

# ---------------------------------------------------------------------------
# Notes + write workbook
# ---------------------------------------------------------------------------
notes_text = pd.DataFrame({"note": [
    "gemma4:e4b: only run_index == 1 (each case's original, single run) is included -- "
    "its 2nd/3rd top-up repeats are excluded entirely, so self-consistency does not apply "
    "and there is no snapshot-drift concern (only one real run per case is used).",
    "qwen3:1.7b: unchanged -- all 3 repeats per case are pooled into the pass rate below "
    "(qwen had zero snapshot drift across its repeats).",
    "Self-consistency is intentionally omitted from this workbook.",
]})

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    notes_text.to_excel(writer, sheet_name="Notes", index=False)
    headline.to_excel(writer, sheet_name="Headline", index=False)
    by_bucket.to_excel(writer, sheet_name="By_Bucket", index=False)
    by_priority.to_excel(writer, sheet_name="By_Priority_Tier", index=False)
    verdict_breakdown.to_excel(writer, sheet_name="Verdict_Breakdown", index=False)
    tool_use_stats.to_excel(writer, sheet_name="Tool_Use_Stats", index=False)
    tool_call_name_counts.to_excel(writer, sheet_name="Tool_Use_Stats", index=False, startrow=len(tool_use_stats) + 3)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print(f"\nWrote {OUT_PATH}")
