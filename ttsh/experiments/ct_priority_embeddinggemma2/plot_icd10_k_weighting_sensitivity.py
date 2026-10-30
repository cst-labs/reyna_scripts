"""k/weighting sensitivity for the 5 configs in P2_P3_IMPROVEMENT_STUDY.md:
for each, try k in {15, 25} x weighting in {uniform, similarity_weighted}
(class_adjustment held fixed at "inverse", what all 5 selected configs use),
score on the test set, report macro/micro/weighted F1 and P1/P2/P3 recall.
Reuses cached embeddings -- no re-encoding."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import run_experiment as exp
from evaluate_care_acuity_diagnosis_5fold import label_metrics
from plot_icd10_calibration import CONFIGS

OUT = exp.RESULTS_DIR / "p2_p3_icd10_5fold"
IMG_DIR = OUT / "confusion_heatmaps"
IMG_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [15, 25]
WEIGHTINGS = ["uniform", "similarity_weighted"]
CLASS_ADJUSTMENT = "inverse"  # fixed: what all 5 selected configs already use


def main() -> None:
    hist = pd.read_excel(exp.DATA_DIR / "historical_cases.xlsx")
    hist = hist[hist["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    y = hist["Priority Label"].to_numpy()

    test_raw = pd.read_excel(exp.DATA_DIR / "newdata_test_cases.xlsx")
    test = test_raw[test_raw["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    y_test = test["Priority Label"].to_numpy()

    rows: list[dict] = []
    for name, sel_k, sel_weighting, _adj, hist_npz, test_npz in CONFIGS:
        hist_vec = np.load(hist_npz)["embeddings"]
        test_vec = np.load(test_npz)["embeddings"]
        similarity = test_vec @ hist_vec.T
        for k in K_VALUES:
            for weighting in WEIGHTINGS:
                pred, *_ = exp.knn_scores(similarity, y, k, weighting, CLASS_ADJUSTMENT)
                scores = label_metrics(y_test, pred)
                rows.append({
                    "template": name, "k": k, "weighting": weighting,
                    "class_adjustment": CLASS_ADJUSTMENT,
                    "is_selected": (k == sel_k and weighting == sel_weighting),
                    "macro_f1": scores["macro_f1"],
                    "micro_f1": f1_score(y_test, pred, labels=exp.LABELS, average="micro", zero_division=0),
                    "weighted_f1": f1_score(y_test, pred, labels=exp.LABELS, average="weighted", zero_division=0),
                    "P1_recall": scores["P1_recall"], "P2_recall": scores["P2_recall"], "P3_recall": scores["P3_recall"],
                })
        print(f"Done: {name}", flush=True)

    df = pd.DataFrame(rows)
    out_xlsx = OUT / "k_weighting_sensitivity.xlsx"
    df.to_excel(out_xlsx, index=False)
    out_xlsx.chmod(0o600)
    print(f"Saved: {out_xlsx}", flush=True)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    metrics = ["macro_f1", "micro_f1", "weighted_f1", "P1_recall", "P2_recall", "P3_recall"]
    templates = df["template"].unique()
    combos = [(k, w) for k in K_VALUES for w in WEIGHTINGS]
    combo_labels = [f"k={k}\n{w}" for k, w in combos]
    colors = ["#9ecae1", "#3182bd", "#fdae6b", "#e6550d"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    x = np.arange(len(metrics))
    width = 0.2
    for ax, template in zip(axes, templates):
        sub = df[df["template"] == template]
        for i, (k, w) in enumerate(combos):
            row = sub[(sub["k"] == k) & (sub["weighting"] == w)].iloc[0]
            vals = [row[m] for m in metrics]
            bars = ax.bar(x + (i - 1.5) * width, vals, width=width, color=colors[i], label=combo_labels[i])
            if row["is_selected"]:
                for b in bars:
                    b.set_edgecolor("black")
                    b.set_linewidth(1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(template, fontsize=9)
        ax.legend(fontsize=6, loc="lower right")
    for ax in axes[len(templates):]:
        ax.axis("off")

    fig.suptitle("k x weighting sensitivity (class_adjustment=inverse), test set, best 5\n"
                  "black outline = this config's already-selected k/weighting", fontsize=11)
    fig.tight_layout()
    out_png = IMG_DIR / "k_weighting_sensitivity.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}", flush=True)


if __name__ == "__main__":
    main()
