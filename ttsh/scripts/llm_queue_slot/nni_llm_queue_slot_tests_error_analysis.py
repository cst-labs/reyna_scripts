"""
nni_llm_queue_slot_tests_error_analysis.py
Simple error analysis on top of the four NNI queue-slotting LLM experiments
(qwen3:1.7b / gemma4:e4b, each no-tools and tool-calling): for every FAIL,
looks at the signed error (model_position - correct_position) and buckets
it into a few common failure modes -- anchoring on position 1 ("new order
always goes next"), anchoring on the back of the queue, an off-by-one slip,
or a larger miscalculation -- plus a breakdown of pass rate by the new
order's own priority tier, and how often tool-calling runs went
UNPARSEABLE or hit the round cap.

Reads:
  analysis/nni_llm_queue_slot_tests.xlsx                  (Cases -- qwen3:1.7b no-tools)
  analysis/nni_llm_queue_slot_tests_tools.xlsx             (Cases -- qwen3:1.7b tools)
  analysis/nni_llm_queue_slot_tests_gemma4-e4b.xlsx        (Cases -- gemma4:e4b no-tools)
  analysis/nni_llm_queue_slot_tests_tools_gemma4-e4b.xlsx  (Cases -- gemma4:e4b tools)

Outputs: adds two sheets to the existing analysis/nni_llm_queue_slot_tests_all_results.xlsx
(alongside the "Summary" sheet already there): "Error_Analysis" (failure-mode
counts per model/tool_mode) and "By_Priority" (pass rate by the new order's
own priority tier, per model/tool_mode).
"""
import os

import pandas as pd
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS = os.path.join(BASE, "analysis")

SOURCES = [
    ("qwen3:1.7b", "no-tools", os.path.join(ANALYSIS, "nni_llm_queue_slot_tests.xlsx")),
    ("qwen3:1.7b", "tools", os.path.join(ANALYSIS, "nni_llm_queue_slot_tests_tools.xlsx")),
    ("gemma4:e4b", "no-tools", os.path.join(ANALYSIS, "nni_llm_queue_slot_tests_gemma4-e4b.xlsx")),
    ("gemma4:e4b", "tools", os.path.join(ANALYSIS, "nni_llm_queue_slot_tests_tools_gemma4-e4b.xlsx")),
]
TIER_ORDER = ["P1", "P2", "P3", "Unknown"]
ALL_RESULTS_PATH = os.path.join(ANALYSIS, "nni_llm_queue_slot_tests_all_results.xlsx")


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 60)


def classify_error(row):
    if row["verdict"] == "PASS":
        return "pass"
    if row["verdict"] == "UNPARSEABLE":
        return "unparseable (no position found in answer)"
    if row["verdict"] == "INCOMPLETE":
        return "incomplete (hit tool-call round cap)"
    mp, cp, n = row["model_position"], row["correct_position"], row["n_waiting"]
    err = mp - cp
    if mp == 1 and cp > 1:
        return "anchored on position 1 (ignored queue ahead)"
    if mp == n + 1 and cp <= n:
        return "anchored on back of queue (ignored priority)"
    if abs(err) == 1:
        return "off-by-one"
    return "larger miscalculation (|error| >= 2)"


all_cases = []
for model, tool_mode, path in SOURCES:
    cases = pd.read_excel(path, sheet_name="Cases")
    cases["model"] = model
    cases["tool_mode"] = tool_mode
    all_cases.append(cases)
all_cases = pd.concat(all_cases, ignore_index=True)

all_cases["error_type"] = all_cases.apply(classify_error, axis=1)
all_cases["signed_error"] = all_cases["model_position"] - all_cases["correct_position"]

# ---------------------------------------------------------------------------
# Error-mode breakdown per model/tool_mode
# ---------------------------------------------------------------------------
error_analysis = (
    all_cases.groupby(["model", "tool_mode", "error_type"]).size().rename("n_cases").reset_index()
)
totals = all_cases.groupby(["model", "tool_mode"]).size().rename("n_total").reset_index()
error_analysis = error_analysis.merge(totals, on=["model", "tool_mode"])
error_analysis["share_of_all_cases"] = error_analysis["n_cases"] / error_analysis["n_total"]
error_analysis = error_analysis.drop(columns=["n_total"]).sort_values(
    ["model", "tool_mode", "n_cases"], ascending=[True, True, False]
)

print("Error-mode breakdown (share of ALL cases per model/tool_mode, including passes):")
print(error_analysis.to_string(index=False))

# Signed-error stats, FAIL cases only (numeric model_position, wrong answer)
fail_numeric = all_cases[(all_cases["verdict"] == "FAIL")]
signed_stats = fail_numeric.groupby(["model", "tool_mode"])["signed_error"].agg(
    n_fail="size", mean_signed_error="mean", median_signed_error="median"
).reset_index()
print("\nSigned error on FAILs (positive = model placed it later than correct, i.e. too low priority):")
print(signed_stats.to_string(index=False))

# ---------------------------------------------------------------------------
# Pass rate by the new order's own priority tier
# ---------------------------------------------------------------------------
by_priority = all_cases.groupby(["model", "tool_mode", "new_order_priority"]).agg(
    n_cases=("verdict", "size"),
    n_pass=("verdict", lambda s: (s == "PASS").sum()),
).reset_index()
by_priority["pass_rate"] = by_priority["n_pass"] / by_priority["n_cases"]
tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
by_priority = by_priority.sort_values(
    ["model", "tool_mode", "new_order_priority"],
    key=lambda s: s.map(tier_rank) if s.name == "new_order_priority" else s,
)

print("\nPass rate by new order's own priority tier:")
print(by_priority.to_string(index=False))

# ---------------------------------------------------------------------------
# Write into the existing consolidated workbook alongside "Summary"
# ---------------------------------------------------------------------------
existing_summary = pd.read_excel(ALL_RESULTS_PATH, sheet_name="Summary")

print(f"\nUpdating {ALL_RESULTS_PATH} with Error_Analysis and By_Priority sheets ...")
with pd.ExcelWriter(ALL_RESULTS_PATH, engine="openpyxl") as writer:
    existing_summary.to_excel(writer, sheet_name="Summary", index=False)
    error_analysis.to_excel(writer, sheet_name="Error_Analysis", index=False)
    by_priority.to_excel(writer, sheet_name="By_Priority", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)
print("Done.")
