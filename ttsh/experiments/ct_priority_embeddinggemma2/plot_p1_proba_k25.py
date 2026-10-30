"""Histograms of P(P1) among unknown-priority cases predicted P1, per configuration."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_experiment as exp

OUT = exp.RESULTS_DIR / "unknown_priority_k25"
SURFACE, INK, INK2, GRID, P1 = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dc", "#2a78d6"
NAMES = {
    "base_plus_care_location_acuity|inverse_sqrt": "1  care + acuity, inverse_sqrt",
    "base_plus_care_location|inverse_sqrt": "2  care location, inverse_sqrt",
    "base_plus_care_location_acuity|inverse": "3  care + acuity, inverse",
    "base_plus_care_location|inverse": "4  care location, inverse",
    "base_plus_acuity|inverse_sqrt": "5  acuity, inverse_sqrt",
}
pred = pd.read_excel(OUT / "unknown_priority_distribution.xlsx", sheet_name="Predictions")
edges = np.arange(0.3, 1.0001, 0.05)

fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharex=True, sharey=True, facecolor=SURFACE)
for ax, (c, name) in zip(axes, NAMES.items()):
    x = pred.loc[pred[c] == "P1", f"{c}|prob_P1"]
    ax.hist(x, bins=edges, color=P1, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(x.median(), color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(x.median() - 0.015, 0.97, f"median {x.median():.2f}", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=8.5, color=INK)
    ax.set_title(f"{name}\nn={len(x)}", loc="left", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(0.3, 1.0)
axes[0].set_ylabel("cases predicted P1", color=INK2, fontsize=9.5)
fig.supxlabel("k-NN probability of P1", color=INK2, fontsize=10)
fig.suptitle("How confident is each P1 prediction? P(P1) among the unknown cases predicted P1", x=0.01, ha="left",
             fontsize=12.5, color=INK, fontweight="bold")
fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.2, wspace=0.12)
fig.savefig(OUT / "p1_proba_distribution.png", dpi=170, facecolor=SURFACE)
print("saved")
