# KTAS Analysis Scripts

## Build prompt-ready cases

```bash
python3 scripts/analysis/build_ktas_prompt_dataset.py
```

Default output:

- `scripts/analysis/output/ktas_prompt_cases.jsonl`

Each JSONL row contains:

- `case_id`
- `model_inputs`
- `KTAS_RN`
- `KTAS_expert`
- `system_prompt`
- `user_prompt`

The prompt builder uses the cleaned CSV, excludes `KTAS_RN`, `KTAS_expert`, `Error_group`, `KTAS duration_min`, `mistriage`, `Group`, and `Disposition` from the model input fields, and keeps `Diagnosis in ED`.

## Run the model through LiteLLM

Set these values in `.env`:

- `LITELLM_BASE_URL`
- `LITELLM`
- `LITELLM_CA_BUNDLE` (optional, path to PEM bundle)
- `LITELLM_INSECURE` (optional, `true` to disable TLS verification)

Run the model:

```bash
python3 scripts/analysis/run_ktas_model.py
```

By default, the runner uses the existing output CSV as a cache. Each completed case is appended immediately, so if you stop the run and start it again, previously completed `case_id`s are skipped automatically and the job continues from the remaining cases.

Useful options:

```bash
python3 scripts/analysis/run_ktas_model.py --limit 10
python3 scripts/analysis/run_ktas_model.py --no-resume
python3 scripts/analysis/run_ktas_model.py --model azure-gpt-5.4
python3 scripts/analysis/run_ktas_model.py --ca-bundle /path/to/corporate-ca.pem
python3 scripts/analysis/run_ktas_model.py --insecure
```

If you see `CERTIFICATE_VERIFY_FAILED`, prefer `--ca-bundle` or `LITELLM_CA_BUNDLE` with your organization CA certificate. Use `--insecure` only as a temporary workaround when you trust the endpoint and need to bypass certificate validation.

Default output:

- `scripts/analysis/output/azure_gpt_5_4_predictions.csv`

## Run local Ollama models

Run the KTAS prompts against these default local models:

- `mistral`
- `gemma2:9b`
- `llama3.1:8b`
- `medgemma:4b`
- `qwen2.5:7b`
- `qwen3:1.7b`

```bash
python3 scripts/analysis/run_ktas_ollama_models.py
```

Useful options:

```bash
python3 scripts/analysis/run_ktas_ollama_models.py --limit 10
python3 scripts/analysis/run_ktas_ollama_models.py --models gemma2:9b llama3.1:8b
python3 scripts/analysis/run_ktas_ollama_models.py --no-resume
```

The runner writes one cached CSV per model under:

- `scripts/analysis/output/ollama/`

Each per-model CSV includes:

- `model`
- `case_id`
- `predicted_ktas`
- `KTAS_RN`
- `KTAS_expert`
- `matches_ktas_rn`
- `matches_ktas_expert`
- `matches_either_reference`
- `latency_seconds`
- `reasoning_summary`
- `raw_model_output`

## Run local Transformers models

Run the KTAS prompts against these default local model identifiers:

- `WeiboAI/VibeThinker-3B`

```bash
python3 scripts/analysis/run_ktas_transformers_models.py
```

Useful options:

```bash
python3 scripts/analysis/run_ktas_transformers_models.py --limit 10
python3 scripts/analysis/run_ktas_transformers_models.py --models WeiboAI/VibeThinker-3B
python3 scripts/analysis/run_ktas_transformers_models.py --no-resume
python3 scripts/analysis/run_ktas_transformers_models.py --trust-remote-code
```

The runner writes one cached CSV per model under:

- `scripts/analysis/output/transformers/`

Each per-model CSV includes:

- `model`
- `case_id`
- `predicted_ktas`
- `KTAS_RN`
- `KTAS_expert`
- `matches_ktas_rn`
- `matches_ktas_expert`
- `matches_either_reference`
- `latency_seconds`
- `reasoning_summary`
- `raw_model_output`

## Run full error analysis on a wide CSV

Build a wide CSV from the existing GPT and Ollama per-model outputs:

```bash
python3 scripts/analysis/build_wide_model_comparison_csv.py
```

This includes:

- `azure_gpt_5_4`
- `gemma2:9b`
- `llama3.1:8b`
- `medgemma:4b`
- `mistral`
- `qwen2.5:7b`

`qwen3:1.7b` is excluded by default because its current output file is only partial. Include it explicitly if needed:

```bash
python3 scripts/analysis/build_wide_model_comparison_csv.py --include-qwen3
```

Use this when you have one CSV with `KTAS_expert`, `KTAS_RN`, and multiple model prediction columns in the same table.

```bash
python3 scripts/analysis/full_ktas_error_analysis.py --input path/to/wide_predictions.csv
```

Optional explicit prediction-column selection:

```bash
python3 scripts/analysis/full_ktas_error_analysis.py \
  --input path/to/wide_predictions.csv \
  --prediction-columns pred_model_a pred_model_b
```

Outputs are written under:

- `scripts/analysis/output/full_error_analysis/`

Saved artifacts include:

- publication-ready figures for confusion matrices, deviations, and cross-model comparisons
- a short clinical interpretation text summary

## Generate the final markdown summary table

```bash
python3 scripts/analysis/generate_two_table_markdown.py
```

Current final output:

- `scripts/analysis/output/full_error_analysis_models/ktas_two_table_summary.md`
