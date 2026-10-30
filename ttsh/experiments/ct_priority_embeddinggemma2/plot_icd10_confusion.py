"""Render confusion-matrix heatmaps (PNG) for the 5 configs compared in
P2_P3_IMPROVEMENT_STUDY.md: the baseline (reused numbers) + the 4 new ICD10
templates (from test_confusion_matrices.xlsx)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import run_experiment as exp

OUT = exp.RESULTS_DIR / "p2_p3_icd10_5fold"
IMG_DIR = OUT / "confusion_heatmaps"
IMG_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CM = {
    ("P1", "P1"): 2242, ("P1", "P2"): 371, ("P1", "P3"): 204,
    ("P2", "P1"): 84, ("P2", "P2"): 1013, ("P2", "P3"): 429,
    ("P3", "P1"): 68, ("P3", "P2"): 799, ("P3", "P3"): 1207,
}


def plot_cm(ax, cm: np.ndarray, title: str, vmax: int):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(3)); ax.set_xticklabels(exp.LABELS)
    ax.set_yticks(range(3)); ax.set_yticklabels(exp.LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=9)
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9, color=color)
    return im


def main() -> None:
    base_cm = np.array([[BASELINE_CM[(a, p)] for p in exp.LABELS] for a in exp.LABELS])

    df = pd.read_excel(OUT / "test_confusion_matrices.xlsx")
    configs = [("base_plus_care_location_acuity_diagnosis (baseline)", base_cm)]
    for template in df["template"].unique():
        sub = df[df["template"] == template]
        cm = np.zeros((3, 3), dtype=int)
        for _, r in sub.iterrows():
            cm[exp.LABELS.index(r["actual"]), exp.LABELS.index(r["predicted"])] = r["count"]
        configs.append((str(template), cm))

    vmax = max(cm.max() for _, cm in configs)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    im = None
    for ax, (title, cm) in zip(axes, configs):
        im = plot_cm(ax, cm, title, vmax)
    for ax in axes[len(configs):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, label="count")
    fig.suptitle("Test-set confusion matrices (raw counts), best 5", fontsize=12)

    out_path = IMG_DIR / "all_five_confusion_matrices.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
