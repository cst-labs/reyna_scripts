#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.prompts as prompts
import utils.utils as utils
from utils.ktas_bqr import (
    build_ktas_balanced_quota_candidates,
    build_ktas_candidate_pool,
    compute_quota_targets,
    load_disagreement_metadata,
    make_retrieval_audit_record,
    rerank_candidates_with_disagreement,
    select_balanced_quota_examples,
    select_similarity_examples,
)


PREDICTORS_SPEC = importlib.util.spec_from_file_location(
    "llm_predict_triage_module",
    ROOT / "llm-predict-triage.py",
)
llm_predict_triage = importlib.util.module_from_spec(PREDICTORS_SPEC)
assert PREDICTORS_SPEC.loader is not None
PREDICTORS_SPEC.loader.exec_module(llm_predict_triage)


DEFAULT_ENCODER = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
DISAGREEMENT_RETRIEVAL_STRATEGIES = {"disagreement", "feature_disagreement", "disagreement_feature"}
BQR_SHORTLIST_SIZE = 120
BQR_DEFAULT_GAMMA = 0.95
BQR_DEFAULT_FLOOR = 1
BQR_DEFAULT_CAP = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KTAS KATE experiments with baseline, balanced-quota, or disagreement-aware retrieval.")
    parser.add_argument("--start", type=int, default=0, help="Start index for the KTAS test split.")
    parser.add_argument("--end", type=int, default=578, help="End index for the KTAS test split.")
    parser.add_argument("--model", required=True, help="LLM model to use.")
    parser.add_argument("--strategy", default="KATE", choices=["KATE", "KATECoT", "KATEAutoCoT"], help="KTAS prompting strategy.")
    parser.add_argument("--json", action="store_true", help="Request JSON responses from the model.")
    parser.add_argument("--serialization", default="natural", choices=["spaces", "commas", "newline", "json", "natural", "natural_sex_race", "natural_full"], help="Prompt serialization.")
    parser.add_argument(
        "--retrieval-strategy",
        default="kate",
        choices=["kate", "balanced_quota", "disagreement", "feature_disagreement", "disagreement_feature"],
        help="Retrieval selector.",
    )
    parser.add_argument("--candidate-pool-size", type=int, default=None, help="Candidate pool size before final selection.")
    parser.add_argument("--final-shot-count", type=int, default=40, help="Final number of in-context examples.")
    parser.add_argument("--gamma", type=float, default=None, help="Quota flattening exponent for balanced quota retrieval.")
    parser.add_argument("--minimum-floor-per-class", type=int, default=None, help="Minimum quota per KTAS class for balanced quota retrieval.")
    parser.add_argument("--ktas3-cap-fraction", type=float, default=None, help="Maximum quota fraction for KTAS 3 in balanced quota retrieval.")
    parser.add_argument("--precomputed-disagreement-path", help="JSONL artifact produced by precompute_ktas_disagreement.py.")
    parser.add_argument("--neighborhood-size", type=int, help="Optional expected neighborhood size for disagreement artifact validation.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Weight for label disagreement score.")
    parser.add_argument("--beta", type=float, default=0.2, help="Weight for feature disagreement score.")
    parser.add_argument("--encoder-model", default=DEFAULT_ENCODER, help="SentenceTransformer model used for retrieval.")
    parser.add_argument("--encoder-retries", type=int, default=3, help="Number of retries when loading the sentence encoder.")
    parser.add_argument("--encoder-retry-delay", type=int, default=2, help="Seconds to wait between encoder retries.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for local models.")
    parser.add_argument("--num_predict", type=int, default=400, help="Maximum generated tokens for local models.")
    parser.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434/api/chat", help="Ollama chat endpoint.")
    parser.add_argument("--ollama_timeout", type=int, default=180, help="Ollama request timeout in seconds.")
    parser.add_argument("--ollama_max_retries", type=int, default=3, help="Maximum Ollama retries per request.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "Triage-KTAS-Retrieval"), help="Directory for predictions and metrics.")
    parser.add_argument("--run-id", type=str, help="Optional fixed run identifier.")
    parser.add_argument("--save-prompt-in-first-row-only", action="store_true", help="Match the original runner and only keep the full prompt on the first record.")
    return parser.parse_args()


def load_sentence_encoder_with_retry(model_name: str, max_retries: int, retry_delay_seconds: int):
    return llm_predict_triage.load_sentence_encoder_with_retry(
        model_name,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )


def _query_model(prompt: dict[str, str], *, model: str, return_json: bool, debug: bool) -> str:
    if utils._DATASETS["Triage-KTAS"]["hippa"]:
        return utils.query_gpt_safe(prompt, model=model, return_json=return_json, debug=debug, is_prompt_full=True)
    return utils.query_llm_full(prompt, model=model, return_json=return_json, debug=debug)


def _build_run_stem(args: argparse.Namespace, timestamp: str) -> str:
    return (
        f"Triage-KTAS_{args.strategy}_{args.retrieval_strategy}_{args.model}"
        f"_{args.final_shot_count}_shots_{args.serialization}"
        f"_pool{args.candidate_pool_size}_alpha{args.alpha}_beta{args.beta}"
        f"_gamma{args.gamma}_floor{args.minimum_floor_per_class}_cap{args.ktas3_cap_fraction}"
        f"_{args.start}_{args.end}_{timestamp}"
    )


def _prediction_record(
    *,
    index: int,
    row: pd.Series,
    prompt: dict[str, str],
    response: str,
    formatted_examples: str,
    audit_record: dict[str, object],
    model: str,
    strategy: str,
    return_json: bool,
    debug: bool,
    keep_prompt: bool,
) -> dict[str, object]:
    response_data = llm_predict_triage.parse_response_with_fallback(
        response,
        model=model,
        return_json=return_json,
        debug=debug,
    )
    record: dict[str, object] = {
        "Estimated_Acuity": response_data["Acuity"],
        "Reasoning": response_data.get("Reasoning") if "CoT" in strategy else None,
        "Retrieved_Examples": formatted_examples,
        "Retrieved_Example_IDs": json.dumps(audit_record["selected_example_ids"]),
        "Retrieved_Example_Labels": json.dumps(audit_record["selected_ktas_labels"]),
        "Retrieved_Example_Similarities": json.dumps(audit_record["selected_similarity_scores"]),
        "Retrieved_Example_Disagreement_Scores": json.dumps(audit_record["selected_disagreement_scores"]),
        "Retrieved_Example_Feature_Disagreement_Scores": json.dumps(
            audit_record["selected_feature_disagreement_scores"]
        ),
        "Retrieved_Example_Final_Scores": json.dumps(audit_record["selected_final_scores"]),
        "Retrieved_Quota_Targets": json.dumps(audit_record["quota_targets"]),
        "Retrieved_Class_Counts": json.dumps(audit_record["final_per_class_counts"]),
        **row.to_dict(),
    }
    if keep_prompt or index == 0:
        record["prompt"] = prompt
    return record


def _validate_metadata(metadata_by_id: dict[int, object], expected_neighborhood_size: int | None) -> None:
    if expected_neighborhood_size is None or not metadata_by_id:
        return
    first_record = next(iter(metadata_by_id.values()))
    actual = getattr(first_record, "neighborhood_size", None)
    if actual is not None and actual != expected_neighborhood_size:
        raise ValueError(
            f"Precomputed disagreement artifact has neighborhood size {actual}, "
            f"but --neighborhood-size requested {expected_neighborhood_size}."
        )


def _strategy_weights(retrieval_strategy: str, alpha: float, beta: float) -> tuple[float, float]:
    if retrieval_strategy == "disagreement":
        return alpha, 0.0
    if retrieval_strategy == "feature_disagreement":
        return 0.0, beta
    if retrieval_strategy == "disagreement_feature":
        return alpha, beta
    return 0.0, 0.0


def _resolve_strategy_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.retrieval_strategy == "balanced_quota":
        if args.candidate_pool_size is None:
            args.candidate_pool_size = BQR_SHORTLIST_SIZE
        if args.gamma is None:
            args.gamma = BQR_DEFAULT_GAMMA
        if args.minimum_floor_per_class is None:
            args.minimum_floor_per_class = BQR_DEFAULT_FLOOR
        if args.ktas3_cap_fraction is None:
            args.ktas3_cap_fraction = BQR_DEFAULT_CAP
    else:
        if args.candidate_pool_size is None:
            args.candidate_pool_size = 200
        if args.gamma is None:
            args.gamma = 0.8
        if args.minimum_floor_per_class is None:
            args.minimum_floor_per_class = 2
        if args.ktas3_cap_fraction is None:
            args.ktas3_cap_fraction = 0.30
    return args


def main() -> int:
    args = parse_args()
    args = _resolve_strategy_defaults(args)
    utils.set_ollama_config(
        url=args.ollama_url,
        timeout=args.ollama_timeout,
        max_retries=args.ollama_max_retries,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )

    if args.retrieval_strategy in DISAGREEMENT_RETRIEVAL_STRATEGIES and not args.precomputed_disagreement_path:
        raise ValueError("Disagreement-aware retrieval requires --precomputed-disagreement-path.")

    dataset = utils._DATASETS["Triage-KTAS"]
    training_df = utils.load_dataset(dataset["training_set_filepath"], dataset["format"], 0, 1_000_000).reset_index(drop=True)
    test_df = utils.load_dataset(dataset["test_set_filepath"], dataset["format"], args.start, args.end).reset_index(drop=True)
    complaint_embeddings = np.load(dataset["training_complaint_embeddings_filepath"], allow_pickle=True)
    diagnosis_embeddings = np.load(dataset["training_diagnosis_embeddings_filepath"], allow_pickle=True)
    encoder = load_sentence_encoder_with_retry(
        args.encoder_model,
        max_retries=args.encoder_retries,
        retry_delay_seconds=args.encoder_retry_delay,
    )

    metadata_by_id = {}
    if args.precomputed_disagreement_path:
        metadata_by_id = load_disagreement_metadata(args.precomputed_disagreement_path)
        _validate_metadata(metadata_by_id, args.neighborhood_size)

    quota_targets = None
    if args.retrieval_strategy == "balanced_quota":
        quota_targets = compute_quota_targets(
            training_df,
            final_shot_count=args.final_shot_count,
            gamma=args.gamma,
            minimum_floor_per_class=args.minimum_floor_per_class,
            ktas3_cap_fraction=args.ktas3_cap_fraction,
        )

    timestamp = args.run_id or datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stem = _build_run_stem(args, timestamp)
    predictions_path = output_dir / f"{run_stem}.csv"
    audit_path = output_dir / f"{run_stem}_retrieval_audit.jsonl"
    metrics_path = output_dir / f"{run_stem}_metrics.json"

    strategy_alpha, strategy_beta = _strategy_weights(args.retrieval_strategy, args.alpha, args.beta)

    predictions: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        if args.retrieval_strategy == "balanced_quota":
            candidates = build_ktas_balanced_quota_candidates(
                row=row,
                training_set=training_df,
                complaint_embeddings=complaint_embeddings,
                diagnosis_embeddings=diagnosis_embeddings,
                symptom_encoder=encoder,
                shortlist_size=args.candidate_pool_size,
            )
        else:
            candidates = build_ktas_candidate_pool(
                row=row,
                training_set=training_df,
                complaint_embeddings=complaint_embeddings,
                diagnosis_embeddings=diagnosis_embeddings,
                symptom_encoder=encoder,
                candidate_pool_size=args.candidate_pool_size,
            )

        if args.retrieval_strategy in DISAGREEMENT_RETRIEVAL_STRATEGIES:
            candidates = rerank_candidates_with_disagreement(
                candidates,
                metadata_by_id,
                alpha=strategy_alpha,
                beta=strategy_beta,
            )

        if args.retrieval_strategy == "balanced_quota":
            assert quota_targets is not None
            selection = select_balanced_quota_examples(
                candidates,
                quota_targets,
                final_shot_count=args.final_shot_count,
                use_per_class_vital_ranking=True,
            )
        else:
            selection = select_similarity_examples(
                candidates,
                final_shot_count=args.final_shot_count,
            )

        formatted_examples = "\n\n".join(
            prompts.format_row(pd.Series(candidate.row), dataset="triage-ktas", serialization=args.serialization)
            + f"\nAcuity Level: {candidate.ktas_label}"
            for candidate in selection.selected_candidates
        )
        prompt = prompts.format_instruction_prompt_for_blackbox(
            row=row,
            strategy=args.strategy,
            dataset="triage-ktas",
            serialization=args.serialization,
            return_json=args.json,
            examples=formatted_examples,
        )
        response = _query_model(
            prompt,
            model=args.model,
            return_json=args.json,
            debug=args.debug,
        )
        audit_record = make_retrieval_audit_record(
            query_id=int(index + args.start),
            selection=selection,
            retrieval_strategy=args.retrieval_strategy,
            candidate_pool_size=args.candidate_pool_size,
            final_shot_count=args.final_shot_count,
            gamma=args.gamma,
            minimum_floor_per_class=args.minimum_floor_per_class,
            ktas3_cap_fraction=args.ktas3_cap_fraction,
            alpha=strategy_alpha,
            beta=strategy_beta,
            neighborhood_size=args.neighborhood_size,
            precomputed_disagreement_path=args.precomputed_disagreement_path,
        )
        predictions.append(
            _prediction_record(
                index=index,
                row=row,
                prompt=prompt,
                response=response,
                formatted_examples=formatted_examples,
                audit_record=audit_record,
                model=args.model,
                strategy=args.strategy,
                return_json=args.json,
                debug=args.debug,
                keep_prompt=not args.save_prompt_in_first_row_only,
            )
        )
        audit_records.append(audit_record)

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(predictions_path, index=False)
    with audit_path.open("w", encoding="utf-8") as handle:
        for record in audit_records:
            handle.write(json.dumps(record) + "\n")

    truth = test_df["KTAS_expert"]
    pred = predictions_df["Estimated_Acuity"]
    adjusted_predictions = [
        true if abs(true - predicted) <= 1 else predicted
        for true, predicted in zip(truth, pred)
    ]
    aggregate_retrieval_counts = {label: [] for label in [1, 2, 3, 4, 5]}
    for record in audit_records:
        counts = {int(label): int(count) for label, count in record["final_per_class_counts"].items()}
        for label in aggregate_retrieval_counts:
            aggregate_retrieval_counts[label].append(counts.get(label, 0))

    metrics = {
        "overall": {
            "accuracy": utils.accuracy_score(truth, pred),
            "precision": utils.precision_score(truth, pred, average="weighted", zero_division=0),
            "recall": utils.recall_score(truth, pred, average="weighted", zero_division=0),
            "f1_score": utils.f1_score(truth, pred, average="weighted", zero_division=0),
            "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
            "adjusted_accuracy": utils.accuracy_score(truth, adjusted_predictions),
            "quadratic_kappa": utils.cohen_kappa_score(truth, pred, weights="quadratic"),
            "mae": utils.mean_absolute_error(truth, pred),
            "mse": utils.mean_squared_error(truth, pred),
        },
        "by_class": classification_report(
            truth,
            pred,
            labels=[1, 2, 3, 4, 5],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": [1, 2, 3, 4, 5],
            "matrix": confusion_matrix(truth, pred, labels=[1, 2, 3, 4, 5]).tolist(),
        },
        "retrieval_distribution_summary": {
            "per_query": [
                {
                    "query_id": record["query_id"],
                    "final_per_class_counts": record["final_per_class_counts"],
                    "quota_targets": record["quota_targets"],
                    "retrieval_strategy": record["retrieval_strategy"],
                }
                for record in audit_records
            ],
            "aggregate": {
                str(label): {
                    "mean": sum(values) / len(values) if values else 0.0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                }
                for label, values in aggregate_retrieval_counts.items()
            },
        },
    }
    metrics["run_config"] = {
        "model": args.model,
        "strategy": args.strategy,
        "retrieval_strategy": args.retrieval_strategy,
        "candidate_pool_size": args.candidate_pool_size,
        "final_shot_count": args.final_shot_count,
        "gamma": args.gamma,
        "minimum_floor_per_class": args.minimum_floor_per_class,
        "ktas3_cap_fraction": args.ktas3_cap_fraction,
        "alpha": strategy_alpha,
        "beta": strategy_beta,
        "neighborhood_size": args.neighborhood_size,
        "precomputed_disagreement_path": args.precomputed_disagreement_path,
        "serialization": args.serialization,
        "predictions_path": str(predictions_path),
        "retrieval_audit_path": str(audit_path),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Predictions saved to {predictions_path}")
    print(f"Retrieval audit saved to {audit_path}")
    print(f"Metrics saved to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
