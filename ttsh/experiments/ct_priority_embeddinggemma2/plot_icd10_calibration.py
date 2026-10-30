"""Calibration check for the 5 configs in P2_P3_IMPROVEMENT_STUDY.md: bucket
each test-set prediction's predicted-class probability (max over P1/P2/P3)
into bins, split each bucket's accuracy by which class was predicted
(P1/P2/P3), plot those three accuracy bars per bucket. Reuses cached
embeddings -- no re-encoding.

Produces two images: one at each template's already-selected k/weighting
(CONFIGS), and one forcing k=25/similarity_weighted on all 5 templates
(CONFIGS_K25_SIM) to see whether the per-class calibration pattern is
sensitive to that choice."""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_experiment as exp

OUT = exp.RESULTS_DIR / "p2_p3_icd10_5fold"
IMG_DIR = OUT / "confusion_heatmaps"
IMG_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CACHE_HIST = exp.ARTIFACTS_DIR / "embeddings" / "base_plus_care_location_acuity_diagnosis.npz"
BASELINE_CACHE_TEST = exp.RESULTS_DIR / "care_acuity_diagnosis_5fold" / "embeddings" / "base_plus_care_location_acuity_diagnosis.npz"

# (display name, k, weighting, class_adjustment, hist cache npz, test cache npz)
CONFIGS = [
    ("base_plus_care_location_acuity_diagnosis", 15, "similarity_weighted", "inverse",
     BASELINE_CACHE_HIST, BASELINE_CACHE_TEST),
    ("diagnosis_base_plus_icd3", 25, "similarity_weighted", "inverse",
     exp.ARTIFACTS_DIR / "embeddings" / "diagnosis_base_plus_icd3__hist_fix.npz",
     exp.ARTIFACTS_DIR / "embeddings" / "diagnosis_base_plus_icd3__test_fix.npz"),
    ("diagnosis_base_plus_icd3_classification_type", 25, "similarity_weighted", "inverse",
     exp.ARTIFACTS_DIR / "embeddings" / "diagnosis_base_plus_icd3_classification_type__hist_fix.npz",
     exp.ARTIFACTS_DIR / "embeddings" / "diagnosis_base_plus_icd3_classification_type__test_fix.npz"),
    ("care_location_acuity_plus_icd3", 15, "uniform", "inverse",
     exp.ARTIFACTS_DIR / "embeddings" / "care_location_acuity_plus_icd3__hist_fix.npz",
     exp.ARTIFACTS_DIR / "embeddings" / "care_location_acuity_plus_icd3__test_fix.npz"),
    ("diagnosis_base_plus_icd3_cfs", 25, "uniform", "inverse",
     exp.ARTIFACTS_DIR / "embeddings" / "diagnosis_base_plus_icd3_cfs__hist_fix.npz",
     exp.ARTIFACTS_DIR / "embeddings" / "diagnosis_base_plus_icd3_cfs__test_fix.npz"),
]

# Same 5 templates/embeddings, but k and weighting forced to k=25/similarity_weighted.
CONFIGS_K25_SIM = [
    (name, 25, "similarity_weighted", adjustment, hist_npz, test_npz)
    for name, _k, _w, adjustment, hist_npz, test_npz in CONFIGS
]

# Same 5 templates/embeddings, but k and weighting forced to k=25/uniform.
CONFIGS_K25_UNIFORM = [
    (name, 25, "uniform", adjustment, hist_npz, test_npz)
    for name, _k, _w, adjustment, hist_npz, test_npz in CONFIGS
]

BIN_WIDTH = 0.10  # 10% buckets; pass --bin-width 0.05 for 5% buckets
LABEL_COLORS = {"P1": "tab:blue", "P2": "tab:orange", "P3": "tab:green"}


def bucket_by_predicted_class(prob: np.ndarray, pred: np.ndarray, correct: np.ndarray,
                              bin_width: float) -> pd.DataFrame:
    n_bins = int(round(1.0 / bin_width))
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        bucket_mask = bin_idx == b
        if bucket_mask.sum() == 0:
            continue
        for label in exp.LABELS:
            mask = bucket_mask & (pred == label)
            if mask.sum() == 0:
                continue
            rows.append({
                "bin_low": edges[b], "bin_high": edges[b + 1],
                "predicted_label": label,
                "n": int(mask.sum()),
                "mean_prob": float(prob[mask].mean()),
                "mean_accuracy": float(correct[mask].mean()),
            })
    return pd.DataFrame(rows)


def plot_panel(ax, bdf: pd.DataFrame, bin_width: float, title: str) -> None:
    width = bin_width * 0.25
    offsets = {"P1": -width, "P2": 0.0, "P3": width}
    for label in exp.LABELS:
        sub = bdf[bdf["predicted_label"] == label]
        if sub.empty:
            continue
        centers = (sub["bin_low"] + sub["bin_high"]) / 2
        ax.bar(centers + offsets[label], sub["mean_accuracy"], width=width,
               color=LABEL_COLORS[label], label=f"{label} accuracy")
        for _, r in sub.iterrows():
            c = (r["bin_low"] + r["bin_high"]) / 2 + offsets[label]
            ax.annotate(f"{r['n']}", (c, r["mean_accuracy"]), fontsize=5, ha="center",
                       textcoords="offset points", xytext=(0, 2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.08)
    ax.set_xlabel("Probability bucket")
    ax.set_ylabel("Accuracy (fraction correct)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="upper left")


def run_configs(configs: list, y: np.ndarray, y_test: np.ndarray, bin_width: float,
                suptitle: str, out_stem: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    all_bucket_rows = []

    for ax, (name, k, weighting, adjustment, hist_npz, test_npz) in zip(axes, configs):
        hist_vec = np.load(hist_npz)["embeddings"]
        test_vec = np.load(test_npz)["embeddings"]
        similarity = test_vec @ hist_vec.T
        pred, probs, *_ = exp.knn_scores(similarity, y, k, weighting, adjustment)[:2]
        max_prob = probs.max(axis=1)
        correct = (pred == y_test).astype(float)

        bdf = bucket_by_predicted_class(max_prob, pred, correct, bin_width)
        bdf.insert(0, "template", name)
        bdf.insert(1, "k", k)
        bdf.insert(2, "weighting", weighting)
        all_bucket_rows.append(bdf)

        plot_panel(ax, bdf, bin_width, f"{name}\n(k={k}, {weighting})")

    for ax in axes[len(configs):]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    out_png = IMG_DIR / f"{out_stem}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}", flush=True)

    bucket_df = pd.concat(all_bucket_rows, ignore_index=True)
    out_xlsx = OUT / f"{out_stem}.xlsx"
    bucket_df.to_excel(out_xlsx, index=False)
    print(f"Saved: {out_xlsx}", flush=True)
    print(bucket_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-width", type=float, default=BIN_WIDTH,
                        help="0.10 for 10%% buckets, 0.05 for 5%% buckets")
    args = parser.parse_args()
    pct = int(round(args.bin_width * 100))

    hist = pd.read_excel(exp.DATA_DIR / "historical_cases.xlsx")
    hist = hist[hist["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    y = hist["Priority Label"].to_numpy()

    test_raw = pd.read_excel(exp.DATA_DIR / "newdata_test_cases.xlsx")
    test = test_raw[test_raw["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    y_test = test["Priority Label"].to_numpy()

    run_configs(CONFIGS, y, y_test, args.bin_width,
               f"Accuracy by predicted class, {pct}% buckets, test set, selected k/weighting per template",
               f"calibration_by_class_{pct}pct")
    run_configs(CONFIGS_K25_SIM, y, y_test, args.bin_width,
               f"Accuracy by predicted class, {pct}% buckets, test set, k=25/similarity_weighted for all 5",
               f"calibration_by_class_k25_simweighted_{pct}pct")
    run_configs(CONFIGS_K25_UNIFORM, y, y_test, args.bin_width,
               f"Accuracy by predicted class, {pct}% buckets, test set, k=25/uniform for all 5",
               f"calibration_by_class_k25_uniform_{pct}pct")


if __name__ == "__main__":
    main()
