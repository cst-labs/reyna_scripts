"""
Ordering Question Responses checklist analysis (CT orderables only)
====================================================================
Restricted to orders where Procedure Category == "IMG CT ORDERABLES"
(the true CT-scan subset, n~10,112). For each checklist question embedded in
the free-text "Ordering Question Responses" field, compute % null (question
not asked/answered) against that fixed CT-orderable total, and list the
possible answer values it takes.

Ordering Question Responses format: one or more ";"-separated blocks (one per
order resubmission/amendment), each block a "\n"-separated list of
"Question: Answer" lines. Only the first block per row is used, since later
blocks are edits of the same order, not new questions.
"""
import os
import re
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

OUT = "analysis"
os.makedirs(OUT, exist_ok=True)

# Palette (dataviz skill reference palette, light mode) — matches
# consult_acuity_field_heatmap.py for visual consistency across analysis/.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRIDLINE,
})


def truncate_label(q, width=58):
    return q if len(q) <= width else q[: width - 1].rstrip() + "…"


# ── Load & restrict to CT orderables only ──────────────────────────────────
ct = pd.read_excel("merged/ct_orders_cleaned.xlsx", sheet_name="Imaging")
ct = ct[ct["Ordering Question Responses"].notna()].copy()
ct_only = ct[ct["Procedure Category"] == "IMG CT ORDERABLES"].copy()
TOTAL_N = len(ct_only)
print(f"CT orderables only: n={TOTAL_N:,} (out of {len(ct):,} orders with any Ordering Question Responses)")

ct_only["first_block"] = ct_only["Ordering Question Responses"].astype(str).str.split("; ").str[0]


def parse_block(text):
    answers = {}
    for line in text.split("\n"):
        parts = re.split(r"::|:", line, maxsplit=1)
        if len(parts) == 2:
            answers[parts[0].strip()] = parts[1].strip()
    return answers


ct_only["parsed"] = ct_only["first_block"].apply(parse_block)

# ── Checklist: questions appearing often enough to be a standard part of the
#    form, not a one-off/rare field ─────────────────────────────────────────
freq = Counter()
for d in ct_only["parsed"]:
    freq.update(d.keys())

MIN_N = 50
CHECKLIST = [q for q, n in freq.most_common() if n >= MIN_N]
print(f"{len(CHECKLIST)} checklist questions with >= {MIN_N} occurrences (out of {len(freq)} distinct questions seen)")

# ── Per-question: % null (vs fixed CT-orderable total) + possible answers ──
rows = []
answer_rows = []
for q in CHECKLIST:
    present_mask = ct_only["parsed"].apply(lambda d, q=q: q in d)
    n_present = int(present_mask.sum())
    pct_null = round(100 * (1 - n_present / TOTAL_N), 1)

    answers = ct_only.loc[present_mask, "parsed"].apply(lambda d, q=q: d[q])
    value_counts = answers.value_counts()
    top_values = "; ".join(f"{v} (n={c:,})" for v, c in value_counts.head(8).items())

    rows.append({
        "Question": q,
        "% null": pct_null,
        "n present": n_present,
        "n total (CT orderables)": TOTAL_N,
        "n distinct answers": value_counts.shape[0],
        "Possible answers (top 8)": top_values,
    })
    for v, c in value_counts.items():
        answer_rows.append({"Question": q, "Answer": v, "n": int(c)})

null_df = pd.DataFrame(rows).sort_values("% null")
answers_df = pd.DataFrame(answer_rows)

print()
print(null_df[["Question", "% null", "n present", "n distinct answers"]].to_string(index=False))

# ── Heatmap: % null per question (CT orderables only) ──────────────────────
pct_null_col = null_df.set_index("Question")[["% null"]]
pct_null_col.index = [truncate_label(q) for q in pct_null_col.index]
pct_null_col.columns = ["% null"]

fig_h = max(3, 0.42 * len(pct_null_col) + 1.4)
fig, ax = plt.subplots(figsize=(10.5, fig_h))
seq_cmap = sns.light_palette(SEQ_BLUE[4], as_cmap=True)
sns.heatmap(
    pct_null_col, annot=True, fmt=".0f", cmap=seq_cmap, vmin=0, vmax=100,
    linewidths=2, linecolor=SURFACE, cbar_kws={"label": "% null (question not present)"},
    annot_kws={"color": INK_PRIMARY, "fontsize": 9}, ax=ax,
)
ax.set_title(f"Checklist question missingness — CT orderables only (n={TOTAL_N:,})",
             fontsize=11, color=INK_PRIMARY, loc="left")
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="y", labelsize=9)
ax.tick_params(axis="x", labelsize=9)
fig.subplots_adjust(left=0.42, right=0.95, top=0.9, bottom=0.06)
fig.savefig(os.path.join(OUT, "heatmap_ct_orderables_question_null_pct.png"), dpi=150, facecolor=SURFACE)
plt.close(fig)

# ── Excel summary ───────────────────────────────────────────────────────────
with pd.ExcelWriter(os.path.join(OUT, "checklist_question_analysis.xlsx"), engine="openpyxl") as writer:
    null_df.to_excel(writer, sheet_name="null_pct_ct_orderables", index=False)
    answers_df.to_excel(writer, sheet_name="answer_value_counts", index=False)

print(f"\nWrote analysis/heatmap_ct_orderables_question_null_pct.png")
print(f"Wrote {OUT}/checklist_question_analysis.xlsx (sheets: null_pct_ct_orderables, answer_value_counts)")
