"""
nni_llm_queue_slot_consistency_pass_rate_chart.py
Line chart of pass rate (y) vs queue-length bucket (x), one line per
model x tool_mode combination, using the By_Bucket sheet from the
consistency-experiment stable summary (run_index == 1 only, both models,
exactly 50 cases per bucket).

Reads: analysis/nni_llm_queue_slot_consistency_stable_summary.xlsx, sheet "By_Bucket"
Outputs: analysis/nni_llm_queue_slot_consistency_pass_rate_by_bucket.png
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYSIS = os.path.join(BASE, "analysis")

BUCKET_ORDER = [
    "tiny (0-1 waiting)", "small (2-3 waiting)", "medium (4-5 waiting)",
    "large (6-7 waiting)", "huge (8+ waiting)",
]

by_bucket = pd.read_excel(
    os.path.join(ANALYSIS, "nni_llm_queue_slot_consistency_stable_summary.xlsx"),
    sheet_name="By_Bucket", skiprows=2,
)

STYLES = {
    ("qwen3:1.7b", "no-tools"): dict(color="tab:blue", linestyle="--", marker="o"),
    ("qwen3:1.7b", "tools"): dict(color="tab:blue", linestyle="-", marker="s"),
    ("gemma4:e4b", "no-tools"): dict(color="tab:orange", linestyle="--", marker="o"),
    ("gemma4:e4b", "tools"): dict(color="tab:orange", linestyle="-", marker="s"),
}

fig, ax = plt.subplots(figsize=(9, 5.5))
for (model, tool_mode), style in STYLES.items():
    sub = by_bucket[(by_bucket["model"] == model) & (by_bucket["tool_mode"] == tool_mode)]
    sub = sub.set_index("queue_length_bucket").loc[BUCKET_ORDER].reset_index()
    ax.plot(sub["queue_length_bucket"], sub["pass_rate"] * 100, label=f"{model} ({tool_mode})", **style)

ax.set_xlabel("Queue-length bucket (orders already waiting)")
ax.set_ylabel("Pass rate (%)")
ax.set_title("NNI queue-slotting pass rate by queue length (run 1, 50 cases/bucket)")
ax.set_ylim(-5, 105)
ax.set_xticklabels([b.replace(" waiting)", ")") for b in BUCKET_ORDER], rotation=15, ha="right")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

out_path = os.path.join(ANALYSIS, "nni_llm_queue_slot_consistency_pass_rate_by_bucket.png")
fig.savefig(out_path, dpi=150)
print(f"Wrote {out_path}")
