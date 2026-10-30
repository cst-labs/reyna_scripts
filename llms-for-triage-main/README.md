# llms-for-triage

Predicting emergency-department triage acuity with LLMs, using zero-shot, few-shot, and retrieval-based (KATE) prompting. The main dataset is the Korean Triage and Acuity Scale (KTAS) data. MIMIC-IV and ESI-Handbook datasets are also supported. The repo also has classical-ML baselines and a LoRA fine-tuning path.

## Layout

```
llm-predict-triage.py     Run an LLM over a test set, save predictions + metrics to results/
llm-evaluate-triage.py    Re-score a saved predictions CSV
utils/
  utils.py                Dataset registry (_DATASETS), LLM clients, metrics
  predictors.py           Predictor classes (zero-shot, few-shot, KATE retrieval)
  prompts.py              Prompt templates and per-dataset strategies (INSTRUCTIONS)
  pairwise.py             Pairwise "which patient first" retrieval + prediction
  ktas_bqr.py             BQR example-selection for KTAS retrieval
  config_demo.py          Template for API keys (see Setup)
scripts/
  generate_ktas_reasoning*.py      Generate LLM reasoning for each training case
  fill_edtriage_train_reasoning.py Reuse/generate reasoning for the edtriage split
  run_ktas_bqr_experiments.py, sweep_ktas_bqr.py, evaluate_ktas_bqr.py   BQR experiments
  run_ktas_pairwise_ollama_batch.sh   Pairwise run across every local Ollama model
  generate_balanced_ktas_pairwise_subset.py   Build a balanced pairwise test set
  analyze_ktas_*.py, summarize_ktas_metrics.py, precompute_ktas_disagreement.py   Analysis
  ml_ktas_cv.py, ml_triage_cv.py, edtriage_rf_ada_no_nurse.py   Classical-ML baselines
  finetuning.py, finetuning_utils.py, run_finetuned_ktas.py     LoRA fine-tune + inference (unsloth)
  convert_edtriage_split_to_text.py   Convert numeric-coded split to text schema
data/
  KTAS/                   KTAS train/test CSVs, embeddings, reasoning-augmented train sets
  mimic-iv-public/        Public MIMIC-IV triage subsets
  mimic-iv-private/       Credentialed MIMIC-IV data (not distributed; gitignored)
  ESI-Handbook/           ESI handbook text, paraphrases, embeddings
models/edtriage_no_nurse/ Saved scaler / TF-IDF for the Random Forest baseline
tests/                    pytest suite
edtriage.ipynb            Original notebook the RF/AdaBoost baseline replicates
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn tqdm python-dotenv pydantic \
            sentence-transformers openai anthropic google-genai pytest
# optional: xgboost, torch, transformers (ML baselines); unsloth, datasets (fine-tuning)
```

There is no requirements file yet. The list above is inferred from imports.

API keys: `utils/utils.py` imports keys from `utils/config.py` first, then falls back to `utils/config_demo.py` (which is blank). To use hosted models, create `utils/config.py` with the same variable names as `config_demo.py`:

```python
OPENAI_API_KEY = "..."
ANTHROPIC_KEY = "..."
GEMINI_KEY = "..."
TOGETHER_API_KEY = "..."
DATABRICKS_TOKEN = "..."
```

LiteLLM is configured through the `LITELLM` and `LITELLM_BASE_URL` environment variables (a `.env` file works). Local models go through Ollama at `http://127.0.0.1:11434` (override with `--ollama_url`).

## Usage

Run from the repo root.

```bash
# Zero/few-shot or KATE retrieval on KTAS
python3 llm-predict-triage.py --dataset Triage-KTAS --model gemma2:9b \
    --strategy KATE --k_shots 5 --serialization natural --run_id my-run --auto_resume

# KATE with reasoning-augmented retrieved examples
python3 llm-predict-triage.py --dataset Triage-KTAS --model gemma2:9b \
    --strategy KATE --k_shots 1 --reasoning_file train_with_reasoning_kate.csv

# Pairwise ordering ("who should be seen first")
python3 llm-predict-triage.py --dataset Triage-KTAS-Pairwise --model llama3.1:8b \
    --strategy KATE-PAIRWISE --k_shots 6
```

- Datasets: `Triage-KTAS`, `Triage-KTAS-Pairwise`, `Triage-Handbook`, `Triage-Public`, `Triage-Counterfactual`, plus the private MIMIC variants if you have the data. See `_DATASETS` in `utils/utils.py`.
- Strategies available for each dataset are listed in `utils/prompts.py` (`INSTRUCTIONS`). The script errors out with the supported list if you pick an invalid one.
- Outputs go to `results/<dataset>/`, with periodic checkpoints in `results/checkpoints/<dataset>/`. Use `--run_id` with `--auto_resume` to continue an interrupted run.
- Metrics (accuracy, F1, quadratic kappa, per-class) are written next to the predictions as `*_metrics.json`.

Generate training-set reasoning (needs a configured hosted model):

```bash
python3 scripts/generate_ktas_reasoning.py --save_every 50
```

Summarise many runs: `python3 scripts/summarize_ktas_metrics.py`.

## Tests

```bash
pytest tests/
```

## Data notes

- MIMIC-IV data is credentialed. `data/mimic-iv-private/` is intentionally not in the repo. Only `mimic-iv-public/` is included.
- `data/ESI-Handbook/README.md` documents how the handbook text was cleaned.
