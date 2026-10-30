#!/usr/bin/env python3
"""Full KTAS error analysis for wide-format prediction CSV files.

Expected input:
  - one row per case
  - at least one expert-reference column and one RN-reference column
  - one or more prediction columns, automatically detected unless overridden

Usage:
  python3 scripts/analysis/full_ktas_error_analysis.py --input data/ktas_predictions.csv
  python3 scripts/analysis/full_ktas_error_analysis.py --input data/ktas_predictions.csv --prediction-columns pred_model_a pred_model_b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency for full KTAS error analysis: "
        f"{exc.name}. Install pandas, numpy, scikit-learn, matplotlib, and seaborn first."
    ) from exc


DEFAULT_OUTPUT_DIR = Path("scripts/analysis/output/full_error_analysis")
LABEL_ORDER = [1, 2, 3, 4, 5]
EXCLUDED_PREDICTION_HINTS = {
    "caseid",
    "id",
    "ktasexpert",
    "ktasrn",
    "latencyseconds",
    "matchesktasrn",
    "matchesktasexpert",
    "matcheseitherreference",
    "reasoningsummary",
    "rawmodeloutput",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full KTAS error analysis on a wide-format CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Wide CSV with one row per case.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output folder (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--prediction-columns", nargs="*", default=None, help="Optional explicit prediction columns.")
    parser.add_argument("--case-id-column", default=None, help="Optional case-id column override.")
    parser.add_argument("--expert-column", default=None, help="Optional KTAS_expert column override.")
    parser.add_argument("--rn-column", default=None, help="Optional KTAS_RN column override.")
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    resolved = path.expanduser()
    if resolved.exists():
        return resolved

    cwd = Path.cwd()
    candidates = sorted(cwd.rglob("*.csv"))
    candidate_lines = [f"  - {candidate}" for candidate in candidates[:10]]
    candidate_text = "\n".join(candidate_lines) if candidate_lines else "  - No CSV files found under the current working directory."
    raise SystemExit(
        "Input CSV not found.\n"
        f"Requested path: {path}\n"
        f"Resolved path: {resolved.resolve(strict=False)}\n"
        f"Current working directory: {cwd}\n"
        "Nearby CSV files:\n"
        f"{candidate_text}"
    )


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def find_column(columns: list[str], explicit: str | None, aliases: list[str], label: str) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"{label} column not found: {explicit}")
        return explicit

    normalized_map = {normalize_name(column): column for column in columns}
    for alias in aliases:
        if alias in normalized_map:
            return normalized_map[alias]

    raise ValueError(f"Could not identify {label} column. Available columns: {columns}")


def clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"nan", "none", "null", "#null!"}:
            return np.nan
        return stripped
    return value


def convert_label_series(series: pd.Series) -> tuple[pd.Series, dict[str, int] | None]:
    cleaned = series.map(clean_scalar)
    non_null = cleaned.dropna()
    if non_null.empty:
        return pd.Series([np.nan] * len(series), index=series.index, dtype="float64"), None

    numeric = pd.to_numeric(non_null.astype(str), errors="coerce")
    if numeric.notna().all():
        converted = pd.to_numeric(cleaned.astype(str), errors="coerce")
        return converted.astype("float64"), None

    extracted_numbers = non_null.astype(str).str.extract(r"([1-5])", expand=False)
    if extracted_numbers.notna().all():
        converted = cleaned.astype(str).str.extract(r"([1-5])", expand=False)
        return pd.to_numeric(converted, errors="coerce").astype("float64"), None

    unique_labels = list(dict.fromkeys(non_null.astype(str).tolist()))
    if len(unique_labels) > len(LABEL_ORDER):
        raise ValueError(f"Label column has more than 5 distinct non-null categories: {unique_labels}")

    mapping = {label: LABEL_ORDER[index] for index, label in enumerate(sorted(unique_labels))}
    converted = cleaned.map(lambda value: mapping.get(value) if value in mapping else np.nan)
    return converted.astype("float64"), mapping


def detect_prediction_columns(
    df: pd.DataFrame,
    expert_column: str,
    rn_column: str,
    explicit_prediction_columns: list[str] | None,
) -> list[str]:
    if explicit_prediction_columns is not None and explicit_prediction_columns:
        missing = [column for column in explicit_prediction_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Prediction columns not found: {missing}")
        return explicit_prediction_columns

    candidate_columns: list[str] = []
    for column in df.columns:
        normalized = normalize_name(column)
        if column in {expert_column, rn_column}:
            continue
        if normalized in EXCLUDED_PREDICTION_HINTS:
            continue

        converted, _ = convert_label_series(df[column])
        observed = sorted({int(value) for value in converted.dropna().tolist()})
        if observed and set(observed).issubset(set(LABEL_ORDER)):
            candidate_columns.append(column)

    if not candidate_columns:
        raise ValueError("No prediction columns were detected automatically.")
    return candidate_columns


def maybe_get_case_ids(df: pd.DataFrame, explicit: str | None) -> pd.Series:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Case-id column not found: {explicit}")
        return df[explicit].astype(str)

    for candidate in ("case_id", "caseid", "id"):
        normalized = normalize_name(candidate)
        for column in df.columns:
            if normalize_name(column) == normalized:
                return df[column].astype(str)
    return pd.Series([f"row_{index + 1:06d}" for index in range(len(df))], index=df.index, dtype="object")


def prepare_long_records(
    df: pd.DataFrame,
    case_ids: pd.Series,
    expert_column: str,
    rn_column: str,
    prediction_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, int] | None, dict[str, int] | None, dict[str, dict[str, int] | None]]:
    expert_numeric, expert_mapping = convert_label_series(df[expert_column])
    rn_numeric, rn_mapping = convert_label_series(df[rn_column])

    records: list[dict[str, Any]] = []
    prediction_mappings: dict[str, dict[str, int] | None] = {}

    for prediction_column in prediction_columns:
        prediction_numeric, prediction_mapping = convert_label_series(df[prediction_column])
        prediction_mappings[prediction_column] = prediction_mapping

        subset = pd.DataFrame(
            {
                "case_id": case_ids,
                "model": prediction_column,
                "KTAS_expert": expert_numeric,
                "KTAS_RN": rn_numeric,
                "prediction": prediction_numeric,
            }
        )
        subset = subset.dropna(subset=["prediction", "KTAS_expert", "KTAS_RN"]).copy()
        subset["KTAS_expert"] = subset["KTAS_expert"].astype(int)
        subset["KTAS_RN"] = subset["KTAS_RN"].astype(int)
        subset["prediction"] = subset["prediction"].astype(int)
        records.extend(subset.to_dict(orient="records"))

    long_df = pd.DataFrame.from_records(records)
    if long_df.empty:
        raise ValueError("No valid model/reference rows remain after dropping missing labels.")
    return long_df, expert_mapping, rn_mapping, prediction_mappings


def safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return balanced_accuracy_score(y_true, y_pred)


def compute_reference_metrics(model_df: pd.DataFrame, reference_column: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = model_df[reference_column].to_numpy(dtype=int)
    y_pred = model_df["prediction"].to_numpy(dtype=int)
    signed_deviation = y_pred - y_true
    absolute_deviation = np.abs(signed_deviation)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    per_class_df = pd.DataFrame(
        {
            "class_label": LABEL_ORDER,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    deviation_counts = pd.Series(signed_deviation).value_counts().sort_index()
    deviation_df = pd.DataFrame(
        {
            "signed_deviation": deviation_counts.index.astype(int),
            "count": deviation_counts.values.astype(int),
        }
    )
    deviation_df["absolute_deviation"] = deviation_df["signed_deviation"].abs()

    absolute_counts = pd.Series(absolute_deviation).value_counts().sort_index()
    absolute_df = pd.DataFrame(
        {
            "absolute_deviation": absolute_counts.index.astype(int),
            "count": absolute_counts.values.astype(int),
        }
    )

    catastrophic_mask = absolute_deviation >= 2
    under_mask = signed_deviation > 0  # KTAS 1 = most urgent, so a higher number is less urgent
    over_mask = signed_deviation < 0

    metrics = {
        "n": int(len(model_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(safe_balanced_accuracy(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "weighted_kappa_linear": float(cohen_kappa_score(y_true, y_pred, weights="linear")),
        "weighted_kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "macro_f1": float(np.mean(f1)),
        "within_1_accuracy": float(np.mean(absolute_deviation <= 1)),
        "mean_signed_deviation": float(np.mean(signed_deviation)),
        "mean_absolute_deviation": float(np.mean(absolute_deviation)),
        "under_triage_rate": float(np.mean(under_mask)),
        "over_triage_rate": float(np.mean(over_mask)),
        "exact_match_count": int(np.sum(absolute_deviation == 0)),
        "adjacent_error_count": int(np.sum(absolute_deviation == 1)),
        "catastrophic_error_count": int(np.sum(catastrophic_mask)),
        "catastrophic_error_rate": float(np.mean(catastrophic_mask)),
    }
    return metrics, per_class_df, deviation_df, absolute_df


def build_metrics_tables(
    long_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    per_class_rows: list[pd.DataFrame] = []
    signed_rows: list[pd.DataFrame] = []
    absolute_rows: list[pd.DataFrame] = []

    for model_name in sorted(long_df["model"].unique()):
        model_df = long_df[long_df["model"] == model_name].copy()
        for reference_column, reference_label in (("KTAS_expert", "expert"), ("KTAS_RN", "rn")):
            metrics, per_class_df, signed_df, absolute_df = compute_reference_metrics(model_df, reference_column)
            metrics_rows.append(
                {
                    "model": model_name,
                    "reference": reference_label,
                    **metrics,
                }
            )
            per_class_df.insert(0, "reference", reference_label)
            per_class_df.insert(0, "model", model_name)
            signed_df.insert(0, "reference", reference_label)
            signed_df.insert(0, "model", model_name)
            absolute_df.insert(0, "reference", reference_label)
            absolute_df.insert(0, "model", model_name)
            per_class_rows.append(per_class_df)
            signed_rows.append(signed_df)
            absolute_rows.append(absolute_df)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(
        by=["reference", "weighted_kappa_quadratic", "accuracy", "within_1_accuracy"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    per_class_metrics_df = pd.concat(per_class_rows, ignore_index=True)
    signed_deviation_df = pd.concat(signed_rows, ignore_index=True)
    absolute_deviation_df = pd.concat(absolute_rows, ignore_index=True)

    expert_rank_df = (
        metrics_df[metrics_df["reference"] == "expert"]
        .sort_values(by=["weighted_kappa_quadratic", "accuracy", "within_1_accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    expert_rank_df.insert(0, "rank", np.arange(1, len(expert_rank_df) + 1))

    return metrics_df, per_class_metrics_df, signed_deviation_df, absolute_deviation_df, expert_rank_df


def build_comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    expert = metrics_df[metrics_df["reference"] == "expert"].copy()
    rn = metrics_df[metrics_df["reference"] == "rn"].copy()
    expert = expert.add_prefix("expert_")
    rn = rn.add_prefix("rn_")
    merged = expert.merge(rn, left_on="expert_model", right_on="rn_model", how="outer")
    merged["model"] = merged["expert_model"].fillna(merged["rn_model"])
    preferred_columns = [
        "model",
        "expert_n",
        "expert_accuracy",
        "expert_balanced_accuracy",
        "expert_cohen_kappa",
        "expert_weighted_kappa_quadratic",
        "expert_macro_f1",
        "expert_within_1_accuracy",
        "expert_under_triage_rate",
        "expert_over_triage_rate",
        "expert_catastrophic_error_rate",
        "rn_n",
        "rn_accuracy",
        "rn_balanced_accuracy",
        "rn_cohen_kappa",
        "rn_weighted_kappa_quadratic",
        "rn_macro_f1",
        "rn_within_1_accuracy",
        "rn_under_triage_rate",
        "rn_over_triage_rate",
        "rn_catastrophic_error_rate",
    ]
    return merged[preferred_columns].sort_values(
        by=["expert_weighted_kappa_quadratic", "expert_accuracy", "expert_within_1_accuracy"],
        ascending=False,
    ).reset_index(drop=True)


def plot_confusion_heatmap(model: str, reference_label: str, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, normalize: bool) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    data = matrix.astype(int)
    annot_data: np.ndarray[Any, Any] = data
    fmt = "d"
    cbar_label = "Count"
    if normalize:
        normalized_data = data.astype(float)
        row_sums = normalized_data.sum(axis=1, keepdims=True)
        data = np.divide(normalized_data, row_sums, out=np.zeros_like(normalized_data), where=row_sums != 0)
        annot_data = data
        fmt = ".2f"
        cbar_label = "Row-normalized proportion"

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        data,
        annot=annot_data,
        fmt=fmt,
        cmap="Blues",
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title(f"{model}: {reference_label} confusion{' (normalized)' if normalize else ''}")
    ax.set_xlabel("Predicted KTAS")
    ax.set_ylabel("Reference KTAS")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_deviation_histogram(model: str, reference_label: str, values: np.ndarray, output_path: Path, absolute: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if absolute:
        bins = np.arange(values.max() + 2) - 0.5 if len(values) else np.array([-0.5, 0.5])
        ax.hist(values, bins=bins, color="#2b8cbe", edgecolor="black")
        ax.set_xlabel("Absolute deviation |prediction - reference|")
    else:
        min_value = int(values.min()) if len(values) else -1
        max_value = int(values.max()) if len(values) else 1
        bins = np.arange(min_value, max_value + 2) - 0.5
        ax.hist(values, bins=bins, color="#de2d26", edgecolor="black")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Signed deviation prediction - reference")
    ax.set_ylabel("Case count")
    ax.set_title(f"{model}: {reference_label} {'absolute' if absolute else 'signed'} deviation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_bar(model: str, per_class_df: pd.DataFrame, output_path: Path) -> None:
    melted = per_class_df.melt(
        id_vars=["class_label"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=melted, x="class_label", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("KTAS class")
    ax.set_ylabel("Score")
    ax.set_title(f"{model}: per-class precision/recall/F1 vs expert")
    ax.legend(title="Metric")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_under_over(model: str, metrics: dict[str, Any], output_path: Path) -> None:
    plot_df = pd.DataFrame(
        {
            "type": ["Under-triage", "Over-triage", "Catastrophic"],
            "rate": [
                metrics["under_triage_rate"],
                metrics["over_triage_rate"],
                metrics["catastrophic_error_rate"],
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.barplot(data=plot_df, x="type", y="rate", palette=["#1b9e77", "#d95f02", "#7570b3"], ax=ax)
    ax.set_ylim(0, max(0.05, plot_df["rate"].max() * 1.15))
    ax.set_ylabel("Rate")
    ax.set_xlabel("")
    ax.set_title(f"{model}: error profile vs expert")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_model_metric(comparison_df: pd.DataFrame, metric_columns: list[str], metric_label: str, output_path: Path) -> None:
    plot_df = comparison_df[["model", *metric_columns]].copy()
    melted = plot_df.melt(id_vars=["model"], value_vars=metric_columns, var_name="metric", value_name="score")
    melted["metric"] = melted["metric"].map(
        {
            metric_columns[0]: "Expert",
            metric_columns[1]: "RN",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Model")
    ax.set_title(f"Model comparison: {metric_label}")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_figures(long_df: pd.DataFrame, metrics_df: pd.DataFrame, per_class_df: pd.DataFrame, output_dir: Path) -> list[str]:
    figure_paths: list[str] = []
    sns.set_theme(style="whitegrid", context="talk")

    expert_metrics_by_model = {
        row["model"]: row
        for _, row in metrics_df[metrics_df["reference"] == "expert"].iterrows()
    }

    for model_name in sorted(long_df["model"].unique()):
        model_df = long_df[long_df["model"] == model_name].copy()
        y_true = model_df["KTAS_expert"].to_numpy(dtype=int)
        y_pred = model_df["prediction"].to_numpy(dtype=int)
        signed_deviation = y_pred - y_true
        absolute_deviation = np.abs(signed_deviation)
        model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)

        heatmap_path = output_dir / f"{model_slug}_expert_confusion_matrix.png"
        normalized_heatmap_path = output_dir / f"{model_slug}_expert_confusion_matrix_normalized.png"
        signed_hist_path = output_dir / f"{model_slug}_expert_signed_deviation_histogram.png"
        absolute_hist_path = output_dir / f"{model_slug}_expert_absolute_deviation_histogram.png"
        per_class_path = output_dir / f"{model_slug}_expert_per_class_scores.png"
        under_over_path = output_dir / f"{model_slug}_expert_under_over_catastrophic.png"

        plot_confusion_heatmap(model_name, "KTAS_expert", y_true, y_pred, heatmap_path, normalize=False)
        plot_confusion_heatmap(model_name, "KTAS_expert", y_true, y_pred, normalized_heatmap_path, normalize=True)
        plot_deviation_histogram(model_name, "KTAS_expert", signed_deviation, signed_hist_path, absolute=False)
        plot_deviation_histogram(model_name, "KTAS_expert", absolute_deviation, absolute_hist_path, absolute=True)
        plot_per_class_bar(
            model_name,
            per_class_df[(per_class_df["model"] == model_name) & (per_class_df["reference"] == "expert")].copy(),
            per_class_path,
        )
        plot_under_over(model_name, expert_metrics_by_model[model_name], under_over_path)

        figure_paths.extend(
            [
                str(heatmap_path),
                str(normalized_heatmap_path),
                str(signed_hist_path),
                str(absolute_hist_path),
                str(per_class_path),
                str(under_over_path),
            ]
        )

    comparison_accuracy_path = output_dir / "models_accuracy_comparison.png"
    comparison_kappa_path = output_dir / "models_quadratic_weighted_kappa_comparison.png"
    comparison_within1_path = output_dir / "models_within_1_accuracy_comparison.png"

    comparison_df = build_comparison_table(metrics_df)
    plot_grouped_model_metric(comparison_df, ["expert_accuracy", "rn_accuracy"], "Accuracy", comparison_accuracy_path)
    plot_grouped_model_metric(
        comparison_df,
        ["expert_weighted_kappa_quadratic", "rn_weighted_kappa_quadratic"],
        "Quadratic weighted kappa",
        comparison_kappa_path,
    )
    plot_grouped_model_metric(
        comparison_df,
        ["expert_within_1_accuracy", "rn_within_1_accuracy"],
        "Within ±1 accuracy",
        comparison_within1_path,
    )
    figure_paths.extend([str(comparison_accuracy_path), str(comparison_kappa_path), str(comparison_within1_path)])
    return figure_paths


def format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_narrative(metrics_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    expert_df = metrics_df[metrics_df["reference"] == "expert"].sort_values(
        by=["weighted_kappa_quadratic", "accuracy", "within_1_accuracy"],
        ascending=False,
    )
    if expert_df.empty:
        return "No expert-referenced metrics were available."

    best = expert_df.iloc[0]
    worst = expert_df.iloc[-1]
    adjacent_share = (best["adjacent_error_count"] / max(best["n"] - best["exact_match_count"], 1)) if best["n"] else 0.0
    direction = "under-triage" if best["under_triage_rate"] > best["over_triage_rate"] else "over-triage"

    lines = [
        f"Best agreement with KTAS_expert came from `{best['model']}` with quadratic weighted kappa {best['weighted_kappa_quadratic']:.3f}, accuracy {format_rate(best['accuracy'])}, and within ±1 accuracy {format_rate(best['within_1_accuracy'])}.",
        f"Its residual errors were mostly {'adjacent-class mistakes' if adjacent_share >= 0.5 else 'spread beyond adjacent classes'}, with catastrophic errors (absolute deviation >= 2) occurring in {format_rate(best['catastrophic_error_rate'])} of cases.",
        f"`{best['model']}` leaned toward {direction} compared with KTAS_expert: under-triage {format_rate(best['under_triage_rate'])} versus over-triage {format_rate(best['over_triage_rate'])}.",
    ]

    if len(expert_df) > 1:
        lines.append(
            f"The weakest expert agreement was `{worst['model']}`, which reached weighted kappa {worst['weighted_kappa_quadratic']:.3f} and accuracy {format_rate(worst['accuracy'])}."
        )

    if {"expert_accuracy", "rn_accuracy"}.issubset(comparison_df.columns):
        rn_gap_df = comparison_df.copy()
        rn_gap_df["accuracy_gap_expert_minus_rn"] = rn_gap_df["expert_accuracy"] - rn_gap_df["rn_accuracy"]
        largest_gap = rn_gap_df.iloc[rn_gap_df["accuracy_gap_expert_minus_rn"].abs().argmax()]
        lines.append(
            f"The largest difference between expert and RN agreement was for `{largest_gap['model']}`, with expert accuracy {format_rate(largest_gap['expert_accuracy'])} and RN accuracy {format_rate(largest_gap['rn_accuracy'])}."
        )

    lines.append(
        "Under-triage means the predicted KTAS number is higher (less urgent) than the reference; over-triage means it is lower (more urgent)."
    )
    return "\n".join(lines)


def print_summary(comparison_df: pd.DataFrame, narrative: str, table_paths: list[str], figure_paths: list[str]) -> None:
    display_columns = [
        "model",
        "expert_accuracy",
        "expert_balanced_accuracy",
        "expert_weighted_kappa_quadratic",
        "expert_macro_f1",
        "expert_within_1_accuracy",
        "expert_under_triage_rate",
        "expert_over_triage_rate",
        "expert_catastrophic_error_rate",
        "rn_accuracy",
        "rn_weighted_kappa_quadratic",
    ]
    existing_columns = [column for column in display_columns if column in comparison_df.columns]
    print("\nMetrics table")
    print(comparison_df[existing_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMain interpretation")
    print(narrative)
    print("\nSaved tables")
    for path in table_paths:
        print(f"- {path}")
    print("\nSaved figures")
    for path in figure_paths:
        print(f"- {path}")


def main() -> int:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    df = pd.read_csv(input_path)
    columns = list(df.columns)

    expert_column = find_column(
        columns,
        args.expert_column,
        aliases=["ktasexpert", "expert", "expertktas"],
        label="KTAS_expert",
    )
    rn_column = find_column(
        columns,
        args.rn_column,
        aliases=["ktasrn", "rn", "triagern", "nursektas"],
        label="KTAS_RN",
    )
    case_ids = maybe_get_case_ids(df, args.case_id_column)
    prediction_columns = detect_prediction_columns(df, expert_column, rn_column, args.prediction_columns)

    long_df, expert_mapping, rn_mapping, prediction_mappings = prepare_long_records(
        df,
        case_ids,
        expert_column,
        rn_column,
        prediction_columns,
    )

    metrics_df, per_class_df, signed_deviation_df, absolute_deviation_df, expert_rank_df = build_metrics_tables(long_df)
    comparison_df = build_comparison_table(metrics_df)

    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = tables_dir / "model_reference_metrics.csv"
    comparison_path = tables_dir / "model_comparison_table.csv"
    per_class_path = tables_dir / "per_class_metrics.csv"
    signed_path = tables_dir / "signed_deviation_counts.csv"
    absolute_path = tables_dir / "absolute_deviation_counts.csv"
    rank_path = tables_dir / "expert_model_ranking.csv"
    long_path = tables_dir / "row_level_long_analysis.csv"
    metadata_path = tables_dir / "label_mappings.json"

    metrics_df.to_csv(metrics_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    signed_deviation_df.to_csv(signed_path, index=False)
    absolute_deviation_df.to_csv(absolute_path, index=False)
    expert_rank_df.to_csv(rank_path, index=False)
    long_df.to_csv(long_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "input_csv": str(args.input),
                "resolved_input_csv": str(input_path.resolve()),
                "expert_column": expert_column,
                "rn_column": rn_column,
                "prediction_columns": prediction_columns,
                "expert_label_mapping": expert_mapping,
                "rn_label_mapping": rn_mapping,
                "prediction_label_mappings": prediction_mappings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure_paths = save_figures(long_df, metrics_df, per_class_df, figures_dir)
    narrative = build_narrative(metrics_df, comparison_df)
    narrative_path = output_dir / "clinical_interpretation.txt"
    narrative_path.write_text(narrative + "\n", encoding="utf-8")

    print_summary(
        comparison_df,
        narrative,
        [
            str(metrics_path),
            str(comparison_path),
            str(per_class_path),
            str(signed_path),
            str(absolute_path),
            str(rank_path),
            str(long_path),
            str(metadata_path),
            str(narrative_path),
        ],
        figure_paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
