"""
nni_llm_queue_slot_tests_combine_summary.py
Builds one long summary table across all four NNI queue-slotting LLM
experiments (qwen3:1.7b / gemma4:e4b, each no-tools and tool-calling) --
pass rate by model x tool_mode x queue-length bucket, plus an overall row
per model/mode, so all four experiments can be compared side by side
without digging through each experiment's own Summary_by_length sheet.

Reads:
  analysis/nni_llm_queue_slot_tests.xlsx                  (Cases -- qwen3:1.7b no-tools)
  analysis/nni_llm_queue_slot_tests_tools.xlsx             (Cases -- qwen3:1.7b tools)
  analysis/nni_llm_queue_slot_tests_gemma4-e4b.xlsx        (Cases -- gemma4:e4b no-tools)
  analysis/nni_llm_queue_slot_tests_tools_gemma4-e4b.xlsx  (Cases -- gemma4:e4b tools)

Outputs: analysis/nni_llm_queue_slot_tests_all_results.xlsx, sheet
"Summary" -- one long table: model, tool_mode, queue_length_bucket, n_cases,
n_pass, pass_rate.
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

BUCKET_ORDER = [
    "tiny (0-1 waiting)", "small (2-3 waiting)", "medium (4-5 waiting)",
    "large (6-7 waiting)", "huge (8+ waiting)",
]
OVERALL_LABEL = "Overall (all buckets)"


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 60)


rows = []
for model, tool_mode, path in SOURCES:
    print(f"Reading {path} ...")
    cases = pd.read_excel(path, sheet_name="Cases")

    by_bucket = cases.groupby("queue_length_bucket").agg(
        n_cases=("verdict", "size"),
        n_pass=("verdict", lambda s: (s == "PASS").sum()),
    )
    for bucket in BUCKET_ORDER:
        if bucket in by_bucket.index:
            r = by_bucket.loc[bucket]
            rows.append({"model": model, "tool_mode": tool_mode, "queue_length_bucket": bucket,
                         "n_cases": int(r["n_cases"]), "n_pass": int(r["n_pass"])})

    n_cases, n_pass = len(cases), int((cases["verdict"] == "PASS").sum())
    rows.append({"model": model, "tool_mode": tool_mode, "queue_length_bucket": OVERALL_LABEL,
                 "n_cases": n_cases, "n_pass": n_pass})

summary = pd.DataFrame(rows)
summary["pass_rate"] = summary["n_pass"] / summary["n_cases"]

print(f"\nCombined summary across {len(SOURCES)} experiments:")
print(summary.to_string(index=False))

out_path = os.path.join(ANALYSIS, "nni_llm_queue_slot_tests_all_results.xlsx")
print(f"\nWriting {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)
print("Done.")
