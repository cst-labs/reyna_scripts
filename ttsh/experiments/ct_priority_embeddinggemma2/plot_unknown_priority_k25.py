"""Stacked-bar figure of predicted priority for the unlabelled new-data orders."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import run_experiment as exp

OUT = exp.RESULTS_DIR / "unknown_priority_k25"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dc"
COLORS = {"P1": "#2a78d6", "P2": "#eb6834", "P3": "#1baf7a"}
LABEL_INK = {"P1": "#ffffff", "P2": INK, "P3": INK}
NAMES = {
    "base_plus_care_location_acuity|inverse_sqrt": "1  care + acuity, inverse_sqrt",
    "base_plus_care_location|inverse_sqrt": "2  care location, inverse_sqrt",
    "base_plus_care_location_acuity|inverse": "3  care + acuity, inverse",
    "base_plus_care_location|inverse": "4  care location, inverse",
    "base_plus_acuity|inverse_sqrt": "5  acuity, inverse_sqrt",
}

pred = pd.read_excel(OUT / "unknown_priority_distribution.xlsx", sheet_name="Predictions")
shares = {n: pred[c].value_counts(normalize=True).reindex(exp.LABELS, fill_value=0) * 100 for c, n in NAMES.items()}
overall = pd.DataFrame(shares).T
overall.loc["Labelled new data (6,417)"] = [43.9, 23.8, 32.3]
overall.loc["Historical (8,233)"] = [34.6, 17.8, 47.6]

fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.8}, facecolor=SURFACE)


def stack(ax, df, y):
    for yi, (_, row) in zip(y, df.iterrows()):
        left = 0
        for lab in exp.LABELS:
            v = row[lab]
            ax.barh(yi, v, left=left, height=0.62, color=COLORS[lab], edgecolor=SURFACE, linewidth=2)
            if v >= 5:
                ax.text(left + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=9.5, color=LABEL_INK[lab])
            left += v
    ax.set_xlim(0, 100)
    ax.set_facecolor(SURFACE)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])


ya = [6, 5, 4, 3, 2, 0.6, -0.4]
stack(a, overall, ya)
a.set_yticks(ya)
a.set_yticklabels(overall.index, color=INK)
a.set_ylim(-1.1, 6.7)
a.axhline(1.3, color=GRID, linewidth=1, linestyle=(0, (4, 3)))
a.set_title("Predicted priority, all 682 unknowns", loc="left", fontsize=11.5, color=INK, pad=12, fontweight="bold")

pred["month"] = pd.Categorical(pred["month"], ["Jul 2026", "Aug 2026", "Sep 2026"], ordered=True)
rows, labels, ys, y = [], [], [], 7.6
for cfg, name in [("base_plus_care_location_acuity|inverse_sqrt", "1  inverse_sqrt"), ("base_plus_care_location_acuity|inverse", "3  inverse")]:
    y -= 0.5
    for m, g in pred.groupby("month", observed=True):
        y -= 1
        rows.append(g[cfg].value_counts(normalize=True).reindex(exp.LABELS, fill_value=0) * 100)
        labels.append(f"{name} · {m[:3]} (n={len(g)})")
        ys.append(y)
stack(b, pd.DataFrame(rows), ys)
b.set_yticks(ys)
b.set_yticklabels(labels, color=INK)
b.set_ylim(min(ys) - 0.8, max(ys) + 1.1)
b.set_title("By month, care + acuity template", loc="left", fontsize=11.5, color=INK, pad=12, fontweight="bold")

handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[l]) for l in exp.LABELS]
fig.legend(handles, exp.LABELS, loc="lower center", ncol=3, frameon=False, fontsize=10, labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("k=25 EmbeddingGemma-2 predictions for CT orders with no recorded priority", x=0.015, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.subplots_adjust(left=0.2, right=0.985, top=0.84, bottom=0.14)
fig.savefig(OUT / "unknown_priority_distribution.png", dpi=170, facecolor=SURFACE)
print("saved")
