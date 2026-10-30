"""Five-fold patient-grouped evaluation of care/acuity/diagnosis EmbeddingGemma templates."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold

import evaluate_care_acuity_diagnosis as focused
import run_experiment as exp


OUT = exp.RESULTS_DIR / "care_acuity_diagnosis_5fold"
OUT.mkdir(parents=True, exist_ok=True)
OUT.chmod(0o700)
SEED = 42
WEIGHTINGS = ["uniform", "similarity_weighted"]
ADJUSTMENTS = list(exp.CLASS_ADJUSTMENTS)


def label_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    p, r, f, support = precision_recall_fscore_support(
        y, pred, labels=exp.LABELS, zero_division=0
    )
    row = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=exp.LABELS, average="macro", zero_division=0),
    }
    for i, label in enumerate(exp.LABELS):
        row[f"{label}_precision"] = p[i]
        row[f"{label}_recall"] = r[i]
        row[f"{label}_f1"] = f[i]
        row[f"{label}_support"] = int(support[i])
    return row


def ensure_cached_historical_vectors(template: str, fields: list[str], hist: pd.DataFrame) -> np.ndarray:
    cache_dir = exp.ARTIFACTS_DIR / "embeddings"
    npz_path, meta_path = cache_dir / f"{template}.npz", cache_dir / f"{template}.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing historical embedding cache for {template}: {npz_path}")
    meta = json.loads(meta_path.read_text())
    texts = exp.render_cases(hist, fields)
    ids = hist["Case ID"].astype(str).tolist()
    ids_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    text_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()
    if meta.get("model_id") != exp.MODEL_ID or meta.get("ids_hash") != ids_hash or meta.get("text_hash") != text_hash:
        raise ValueError(f"Cached embeddings do not match current rows/template for {template}")
    matrix = np.load(npz_path)["embeddings"]
    if matrix.shape != (len(hist), 768) or not np.isfinite(matrix).all():
        raise ValueError(f"Unexpected embedding matrix for {template}: {matrix.shape}")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
        raise ValueError(f"Embeddings are not L2-normalized for {template}")
    return matrix


def main() -> None:
    hist_path = exp.DATA_DIR / "historical_cases.xlsx"
    test_path = exp.DATA_DIR / "newdata_test_cases.xlsx"
    hist = pd.read_excel(hist_path)
    test = pd.read_excel(test_path)
    hist = hist[hist["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    labeled_test = test[test["Priority Label"].isin(exp.LABELS)].reset_index(drop=True)
    y = hist["Priority Label"].to_numpy()
    y_test = labeled_test["Priority Label"].to_numpy()
    templates = focused.candidate_templates()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(hist, y, groups=hist["Patient Group ID"]))
    for train_idx, val_idx in splits:
        if set(hist.iloc[train_idx]["Patient Group ID"]) & set(hist.iloc[val_idx]["Patient Group ID"]):
            raise RuntimeError("Patient group leakage between fold train and validation")
    if set(hist["Patient Group ID"]) & set(labeled_test["Patient Group ID"]):
        raise RuntimeError("Patient group overlap between historical and test cohort")

    config_rows: list[dict] = []
    fold_rows: list[dict] = []
    hist_vectors: dict[str, np.ndarray] = {}
    for template, fields in templates.items():
        vectors = ensure_cached_historical_vectors(template, fields, hist)
        hist_vectors[template] = vectors
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            similarity = vectors[val_idx] @ vectors[train_idx].T
            neighbor_order = np.argsort(-similarity, axis=1)[:, :max(exp.K_VALUES)]
            neighbor_sims = np.take_along_axis(similarity, neighbor_order, axis=1)
            y_train, y_val = y[train_idx], y[val_idx]
            for k in exp.K_VALUES:
                neighbor_labels = y_train[neighbor_order[:, :k]]
                top_sims = neighbor_sims[:, :k]
                for weighting in WEIGHTINGS:
                    for adjustment in ADJUSTMENTS:
                        pred, *_ = exp.knn_from_sorted_neighbors(
                            neighbor_labels, top_sims, y_train, k, weighting, adjustment
                        )
                        scores = label_metrics(y_val, pred)
                        config_rows.append({
                            "template": template, "fold": fold, "k": k,
                            "weighting": weighting, "class_adjustment": adjustment,
                            **{f"validation_{key}": value for key, value in scores.items()},
                        })
            # Retain every fold's split counts for transparent reporting.
            counts = pd.Series(y_val).value_counts()
            fold_rows.append({
                "fold": fold, "train_cases": len(train_idx), "validation_cases": len(val_idx),
                "train_patients": hist.iloc[train_idx]["Patient Group ID"].nunique(),
                "validation_patients": hist.iloc[val_idx]["Patient Group ID"].nunique(),
                **{f"validation_{lab}_n": int(counts.get(lab, 0)) for lab in exp.LABELS},
            })
        print(f"Completed five folds: {template}", flush=True)

    fold_df = pd.DataFrame(config_rows)
    metric_cols = [c for c in fold_df.columns if c.startswith("validation_")]
    group_cols = ["template", "k", "weighting", "class_adjustment"]
    mean_df = fold_df.groupby(group_cols, as_index=False)[metric_cols].mean()
    sd_df = fold_df.groupby(group_cols, as_index=False)[metric_cols].std().rename(
        columns={c: c.replace("validation_", "sd_validation_") for c in metric_cols}
    )
    mean_df = mean_df.merge(sd_df, on=group_cols)
    field_counts = {name: len(fields) for name, fields in templates.items()}
    # One selected vote/adjustment setting per template and k; macro-F1 leads,
    # then P1 recall, balanced accuracy, uniform vote, then less adjustment.
    adj_rank = {"none": 0, "inverse_sqrt": 1, "inverse": 2}
    mean_df["_weight_rank"] = (mean_df["weighting"] != "uniform").astype(int)
    mean_df["_adjust_rank"] = mean_df["class_adjustment"].map(adj_rank)
    best_per_k = (mean_df.sort_values(
        ["validation_macro_f1", "validation_P1_recall", "validation_balanced_accuracy",
         "_weight_rank", "_adjust_rank"],
        ascending=[False, False, False, True, True], kind="stable")
        .drop_duplicates(["template", "k"]).drop(columns=["_weight_rank", "_adjust_rank"]))
    best_per_k["input_field_count"] = best_per_k["template"].map(field_counts)
    # Rank five distinct input templates by their strongest mean CV configuration.
    template_best = (mean_df.sort_values(
        ["validation_macro_f1", "validation_P1_recall", "validation_balanced_accuracy",
         "k", "_weight_rank", "_adjust_rank"],
        ascending=[False, False, False, True, True, True], kind="stable")
        .drop_duplicates("template").head(5).reset_index(drop=True))

    # Embed only the selected five templates on the existing new-data cohort.
    class Args:
        device = "cpu"
        batch_size = 32
        local_files_only = True
    model = exp.embedder(Args())
    revision = exp.model_revision(model)
    test_metrics: list[dict] = []
    cms: list[dict] = []
    selected_configs = []
    for rank, row in enumerate(template_best.to_dict("records"), start=1):
        template, fields = row["template"], templates[row["template"]]
        texts = exp.render_cases(labeled_test, fields)
        test_cache = OUT / "embeddings"
        test_cache.mkdir(mode=0o700, exist_ok=True)
        test_cache.chmod(0o700)
        cache_stem = test_cache / template
        text_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()
        id_hash = hashlib.sha256("\n".join(labeled_test["Case ID"].astype(str)).encode()).hexdigest()
        test_vec = None
        if cache_stem.with_suffix(".npz").exists() and cache_stem.with_suffix(".json").exists():
            cache_meta = json.loads(cache_stem.with_suffix(".json").read_text())
            if cache_meta.get("text_hash") == text_hash and cache_meta.get("ids_hash") == id_hash and cache_meta.get("model_revision") == revision:
                test_vec = np.load(cache_stem.with_suffix(".npz"))["embeddings"]
        if test_vec is None:
            test_vec = np.asarray(model.encode(
                texts, prompt=exp.TASK_PREFIX, batch_size=32,
                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
            ), dtype=np.float32)
            np.savez_compressed(cache_stem.with_suffix(".npz"), embeddings=test_vec)
            cache_stem.with_suffix(".npz").chmod(0o600)
            cache_stem.with_suffix(".json").write_text(json.dumps({
                "model_id": exp.MODEL_ID, "model_revision": revision,
                "task_prefix": exp.TASK_PREFIX, "dimension": 768,
                "normalized": True, "text_hash": text_hash, "ids_hash": id_hash,
            }, indent=2) + "\n")
            cache_stem.with_suffix(".json").chmod(0o600)
        if test_vec.shape != (len(labeled_test), 768):
            raise ValueError(f"Unexpected test embedding shape for {template}: {test_vec.shape}")
        matrix = test_vec @ hist_vectors[template].T
        k, weighting, adjustment = int(row["k"]), row["weighting"], row["class_adjustment"]
        neighbor_order = np.argsort(-matrix, axis=1)[:, :k]
        neighbor_sims = np.take_along_axis(matrix, neighbor_order, axis=1)
        pred, *_ = exp.knn_from_sorted_neighbors(
            y[neighbor_order], neighbor_sims, y, k, weighting, adjustment
        )
        name = f"rank{rank}:{template}:k{k}:{weighting}:{adjustment}"
        m = label_metrics(y_test, pred)
        test_metrics.append({"rank": rank, "model": name, "template": template,
                             "k": k, "weighting": weighting,
                             "class_adjustment": adjustment,
                             "cv_macro_f1_mean": row["validation_macro_f1"],
                             "cv_macro_f1_sd": row["sd_validation_macro_f1"], **m})
        cm = confusion_matrix(y_test, pred, labels=exp.LABELS)
        for i, actual in enumerate(exp.LABELS):
            for j, predicted in enumerate(exp.LABELS):
                cms.append({"rank": rank, "model": name, "actual": actual,
                            "predicted": predicted, "count": int(cm[i, j])})
        selected_configs.append({"rank": rank, "template": template, "fields": fields,
                                 "k": k, "weighting": weighting,
                                 "class_adjustment": adjustment,
                                 "cv_macro_f1_mean": row["validation_macro_f1"],
                                 "cv_macro_f1_sd": row["sd_validation_macro_f1"]})
        print(f"Test scored: {name}", flush=True)

    print("All selected test predictions complete; assembling output tables.", flush=True)
    test_df = pd.DataFrame(test_metrics)
    cm_df = pd.DataFrame(cms)
    fold_summary = pd.DataFrame(fold_rows).drop_duplicates("fold").sort_values("fold")
    # Plot mean macro-F1 at each k, with vote/adjustment selected by the same
    # CV mean-macro-F1 rule for each template × k.
    print("Building the macro-F1 plot.", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
    base_names = [n for n, f in templates.items() if f and f[0] == "Order Description"]
    no_base_names = [n for n, f in templates.items() if n not in base_names]
    palette = plt.get_cmap("tab10")
    for ax, names, title in [
        (axes[0], base_names, "Fixed order text included"),
        (axes[1], no_base_names, "Candidate fields only"),
    ]:
        for i, name in enumerate(names):
            sub = best_per_k[best_per_k["template"] == name].sort_values("k")
            ax.plot(sub["k"], sub["validation_macro_f1"], marker="o",
                    label=name.replace("base_", "").replace("_only", ""),
                    color=palette(i % 10))
        ax.set_title(title)
        ax.set_xlabel("Number of nearest neighbors (k)")
        ax.set_xticks(exp.K_VALUES)
        ax.grid(True, alpha=.25)
    axes[0].set_ylabel("Mean five-fold validation macro-F1")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle("EmbeddingGemma care, acuity, and diagnosis combinations\n"
                 "Per-template/k best vote and class adjustment chosen by mean CV macro-F1")
    fig.tight_layout()
    plot_path = OUT / "macro_f1_by_k.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}", flush=True)

    test_cm_path = OUT / "top_five_test_confusion_heatmaps.png"
    labels = exp.LABELS
    test_matrices = []
    for rank in range(1, 6):
        rows = cm_df[cm_df["rank"] == rank]
        matrix = rows.pivot(index="actual", columns="predicted", values="count").reindex(
            index=labels, columns=labels
        )
        test_matrices.append(matrix.to_numpy(dtype=int))
    test_vmax = max(int(matrix.max()) for matrix in test_matrices)
    fig, grid = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    heat_axes = grid.ravel()
    display_names = {
        "base_order_info": "Base text only",
        "base_plus_care_location": "Base text + care location",
        "base_plus_acuity": "Base text + acuity",
        "base_plus_diagnosis": "Base text + diagnosis",
        "base_plus_care_location_acuity": "Base text + care location + acuity",
        "base_plus_care_location_diagnosis": "Base text + care location + diagnosis",
        "base_plus_acuity_diagnosis": "Base text + acuity + diagnosis",
        "base_plus_care_location_acuity_diagnosis": "Base text + care location + acuity + diagnosis",
    }
    heat_image = None
    for i, ax in enumerate(heat_axes):
        if i >= len(test_df):
            ax.axis("off")
            continue
        row = test_df.iloc[i]
        heat_image = ax.imshow(test_matrices[i], cmap="Blues", vmin=0, vmax=test_vmax,
                               interpolation="nearest")
        title = display_names.get(row["template"], row["template"])
        ax.set_title(f"Rank {int(row['rank'])}: {title}\\n"
                     f"k={int(row['k'])}, {row['weighting']}, {row['class_adjustment']}",
                     fontsize=10, pad=9)
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("Predicted priority")
        ax.set_ylabel("Actual priority")
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for r in range(3):
            for c in range(3):
                value = test_matrices[i][r, c]
                color = "white" if value >= test_vmax * .52 else "#17212b"
                ax.text(c, r, f"{value:,}", ha="center", va="center", color=color,
                        fontsize=12, fontweight="bold")
    fig.suptitle("Top five test-set confusion matrices — common count scale", fontsize=15)
    fig.colorbar(heat_image, ax=heat_axes.tolist(), shrink=.85, pad=.025,
                 label="Number of cases")
    fig.savefig(test_cm_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    test_cm_path.chmod(0o600)

    workbook = OUT / "five_fold_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        fold_summary.to_excel(writer, sheet_name="Fold sizes", index=False)
        fold_df.to_excel(writer, sheet_name="All fold configurations", index=False)
        mean_df.drop(columns=["_weight_rank", "_adjust_rank"], errors="ignore").to_excel(
            writer, sheet_name="All config means SD", index=False)
        best_per_k.to_excel(writer, sheet_name="Best setting by template k", index=False)
        template_best.to_excel(writer, sheet_name="Top five templates", index=False)
        test_df.to_excel(writer, sheet_name="Top five test metrics", index=False)
        cm_df.to_excel(writer, sheet_name="Top five test confusion", index=False)
    workbook.chmod(0o600)
    print(f"Workbook written: {workbook}", flush=True)
    plot_path.chmod(0o600)

    split_json = {
        "seed": SEED,
        "splitter": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
        "group_field": "Patient Group ID",
        "folds": [
            {"fold": i, "train_case_ids": hist.iloc[tr]["Case ID"].astype(str).tolist(),
             "validation_case_ids": hist.iloc[va]["Case ID"].astype(str).tolist()}
            for i, (tr, va) in enumerate(splits, start=1)
        ],
        "historical_case_ids": hist["Case ID"].astype(str).tolist(),
        "test_case_ids": labeled_test["Case ID"].astype(str).tolist(),
    }
    split_path = OUT / "split_manifest.json"
    split_path.write_text(json.dumps(split_json, indent=2) + "\n")
    split_path.chmod(0o600)
    meta = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": exp.MODEL_ID, "model_revision": revision,
        "model_snapshot_commit": exp.model_snapshot_commit(),
        "task_prefix": exp.TASK_PREFIX, "dimension": 768,
        "normalization": "L2-normalized; cosine similarity is dot product",
        "seed": SEED, "folds": 5, "templates": templates,
        "k_values": exp.K_VALUES, "voting": WEIGHTINGS,
        "class_adjustments": exp.CLASS_ADJUSTMENTS,
        "n_configurations_per_fold": len(templates) * len(exp.K_VALUES) * len(WEIGHTINGS) * len(ADJUSTMENTS),
        "selection": "Mean five-fold macro-F1; tie-break P1 recall, balanced accuracy, smaller k, uniform vote, then less class adjustment. Top five are distinct templates.",
        "test_reference": "All labeled historical cases; test data was examined in previous work and is retrospective, not blind.",
        "historical_labeled_n": len(hist), "newdata_labeled_n": len(labeled_test),
        "top_five": selected_configs,
        "test_results": test_metrics,
        "test_confusion_heatmap": test_cm_path.name,
        "artifact_workbook": workbook.name, "plot": plot_path.name,
    }
    meta_path = OUT / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    meta_path.chmod(0o600)

    md = [
        "# Five-fold EmbeddingGemma care, acuity, and diagnosis experiment",
        "",
        f"Run completed: {meta['run_utc']}",
        "",
        "## Question and design",
        "",
        "This run extends the earlier first-fold comparison to all five patient-grouped folds. It compares the seven non-empty combinations of Care location, Acuity, and Diagnosis both with and without the fixed order-information text, plus the fixed-text-only template (15 templates total). Each template uses all listed fields as labeled plain-text lines; priority remains the separate target and is never embedded.",
        "",
        "- Fixed text: Order Description; Working Diagnosis / Reason; Relevant Clinical Information.",
        "- Care location: First ED Care Area; Primary Care Area.",
        "- Acuity: Triage Acuity; Consult Acuity; KIV Uptriage.",
        "- Diagnosis: Primary ED Dx; ED Dx.",
        "- For each of 15 templates: k = 1, 3, 5, 9, 15, 25; uniform or similarity-weighted voting; no, inverse-square-root, or inverse-frequency class adjustment. This is 540 settings per fold, 2,700 fold-level evaluations total.",
        "- Same `google/embeddinggemma-2` local model, SentenceSimilarity prefix, normalized 768-dimensional embeddings, cosine dot product, and seed-42 patient-grouped folds.",
        "- Missing/blank text fields are represented as `Unknown`.",
        "",
        "## Fold sizes",
        "",
        "| Fold | Training cases | Validation cases | Training patients | Validation patients | P1/P2/P3 validation counts |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in fold_summary.to_dict("records"):
        md.append(f"| {r['fold']} | {r['train_cases']} | {r['validation_cases']} | {r['train_patients']} | {r['validation_patients']} | {r['validation_P1_n']}/{r['validation_P2_n']}/{r['validation_P3_n']} |")
    md.extend([
        "",
        "## Results across all k values",
        "",
        "For each template and k, this table reports the voting rule/class adjustment with the highest mean five-fold macro-F1 (ties resolved by P1 recall, balanced accuracy, uniform vote, then less adjustment). Macro-F1, balanced accuracy, and class recalls are the means across folds; standard deviations are in the workbook. These are internal model-selection estimates because the same folds compare the configurations.",
        "",
        "Input templates are ordered by their highest mean five-fold macro-F1 across any k (descending). Within each template, k values are listed in ascending order. The per-k macro-F1 shown is the best mean across voting and class-adjustment settings for that template and k. Input-field count includes the fixed three-field text base where present.",
        "",
        "| Input fields (n) | Input template | k | Vote | Class adjustment | Macro-F1 mean | Balanced accuracy mean | P1/P2/P3 recall mean |",
        "|---:|---|---:|---|---|---:|---:|---:|",
    ])
    template_order = (best_per_k.groupby("template")["validation_macro_f1"].max()
                      .sort_values(ascending=False, kind="stable").index.tolist())
    template_rank = {name: rank for rank, name in enumerate(template_order)}
    ordered_k_rows = best_per_k.assign(_template_rank=best_per_k["template"].map(template_rank)).sort_values(
        ["_template_rank", "k"], ascending=[True, True], kind="stable"
    )
    for _, r in ordered_k_rows.iterrows():
        label = r["template"].replace("base_", "").replace("_only", " (candidate fields only)")
        md.append(f"| {int(r['input_field_count'])} | {label} | {int(r['k'])} | {r['weighting']} | {r['class_adjustment']} | {r['validation_macro_f1']:.3f} | {r['validation_balanced_accuracy']:.3f} | {r['validation_P1_recall']:.3f}/{r['validation_P2_recall']:.3f}/{r['validation_P3_recall']:.3f} |")
    md.extend(["", "### Macro-F1 line chart", "", "![Mean five-fold macro-F1 by k and input template](macro_f1_by_k.png)",
               "", "Each line plots the best mean macro-F1 for a template at each k after choosing voting and class adjustment using the same five-fold mean. The left panel includes fixed order text; the right uses candidate fields only.",
               "", "## Top five distinct templates on the new-data set", "",
               "The top five distinct input templates were ranked by their best mean five-fold macro-F1, with settings selected using the same tie-break rule. Each was then embedded for the labeled new-data cases and compared with all labeled historical cases. The existing new-data set has already been examined in previous analyses, so these results are retrospective and not blind.",
               "", "| Rank | Template | k | Vote | Class adjustment | CV macro-F1 mean (SD) | Test accuracy | Test macro-F1 | Test balanced accuracy | P1/P2/P3 recall |",
               "|---:|---|---:|---|---|---:|---:|---:|---:|---:|"])
    for _, r in test_df.iterrows():
        md.append(f"| {r['rank']} | {r['template']} | {int(r['k'])} | {r['weighting']} | {r['class_adjustment']} | {r['cv_macro_f1_mean']:.3f} ({r['cv_macro_f1_sd']:.3f}) | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['balanced_accuracy']:.3f} | {r['P1_recall']:.3f}/{r['P2_recall']:.3f}/{r['P3_recall']:.3f} |")
    md.extend(["", "### Test confusion matrices on one shared scale", "",
               "![Five test-set confusion matrices on a common count scale](top_five_test_confusion_heatmaps.png)",
               "", "All five panels use the same color range; rows are actual priority and columns are predicted priority."])
    for _, r in test_df.iterrows():
        sub = cm_df[cm_df["rank"] == r["rank"]]
        mat = sub.pivot(index="actual", columns="predicted", values="count").reindex(index=exp.LABELS, columns=exp.LABELS)
        md.extend(["", f"### Rank {int(r['rank'])}: {r['template']} — confusion matrix (rows=actual, columns=predicted)", "",
                   "| Actual \\ Predicted | P1 | P2 | P3 |", "|---|---:|---:|---:|"])
        for lab in exp.LABELS:
            md.append(f"| {lab} | {int(mat.loc[lab, 'P1'])} | {int(mat.loc[lab, 'P2'])} | {int(mat.loc[lab, 'P3'])} |")
    md.extend([
        "", "## Interpretation and limitations", "",
        "Compare the per-class recalls and confusion matrices as well as aggregate scores; class prevalence differs between historical and new-data cohorts. The top-five test comparison is exploratory because this new-data cohort has been used in earlier analyses and the templates/settings are now selected using five-fold historical results. It does not provide a final unbiased estimate. No geometry diagnostic establishes clinical reliability; the measured target is agreement with recorded priorities.",
        "", "## Reproducibility and files", "",
        f"- Detailed workbook: [`{workbook.name}`]({workbook.name}) (all per-fold configurations, means/SD, best-by-k summaries, selected top-five metrics and confusion matrices).",
        f"- Plot: [`{plot_path.name}`]({plot_path.name}).",
        f"- Split manifest: [`{split_path.name}`]({split_path.name}) (pseudonymous case/group IDs; keep access restricted).",
        f"- Run metadata: [`{meta_path.name}`]({meta_path.name}).",
        "- Method script: [`evaluate_care_acuity_diagnosis_5fold.py`](../evaluate_care_acuity_diagnosis_5fold.py).",
        "",
    ])
    report_path = OUT / "FIVE_FOLD_RESULTS.md"
    report_path.write_text("\n".join(md))
    report_path.chmod(0o600)
    print(f"Report written: {report_path}", flush=True)
    print(json.dumps({"top_five": selected_configs, "test_results": test_metrics,
                      "report": str(report_path), "workbook": str(workbook),
                      "plot": str(plot_path)}, indent=2, default=str))


if __name__ == "__main__":
    main()
