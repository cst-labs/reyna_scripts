# KTAS Analysis

Benchmarks LLMs on Korean Triage and Acuity Scale (KTAS) assignment for emergency department cases, and compares their predictions against the nurse (`KTAS_RN`) and expert (`KTAS_expert`) labels.

## Layout

```
data/
  originaldata.xlsx          raw workbook (sheet "N=1267 data"), contains identifying columns
  originaldata_cleaned.csv   de-identified CSV used by everything below
  guidelines.xlsx            KTAS guideline reference
scripts/
  clean_original_data.py     xlsx -> cleaned csv
  analysis/                  v1: level-definition prompt (output: predicted_ktas, reasoning_summary)
  analysis2/                 v2: step-by-step KTAS decision prompt (7 JSON keys)
  analysis3/                 v3: concise prompt (clinical_summary, ktas_justification, predicted_ktas)
```

Each `analysis*` folder has its own prompt template, dataset builder, runners and `output/`. The three versions differ mainly in `ktas_prompt_templates.py`; the other scripts point at their own `output/` directory. One exception: `analysis2/` and `analysis3/` `full_ktas_error_analysis.py` are thin wrappers that import the metrics and plotting code from `analysis/full_ktas_error_analysis.py`, so keep `analysis/` in place.

## What each script does

### Data preparation

| Script | What it does |
| --- | --- |
| `scripts/clean_original_data.py` | Reads the raw `.xlsx` (no Excel library, parses the XML directly), drops `Patients number per hour` and `Length of stay_min`, drops every row containing Korean (Hangul) text, writes `data/originaldata_cleaned.csv`. |
| `*/build_ktas_prompt_dataset.py` | Turns each cleaned row into a prompt record in `ktas_prompt_cases.jsonl` (`case_000001`, ...). Decodes numeric codes to text (sex, arrival mode, injury, mental state, pain). Removes `KTAS_RN`, `KTAS_expert`, `Error_group`, `KTAS duration_min`, `mistriage`, `Group` and `Disposition` from the model input so the answer can't leak. The two KTAS labels are kept beside the prompt for scoring only. |
| `*/ktas_prompt_templates.py` | Holds the system prompt and user-prompt format. This is the one file that defines each version (see below). |

### Prompt versions

| Folder | Prompt style | Model returns |
| --- | --- | --- |
| `analysis/` | Defines KTAS levels 1-5 with response-time targets, a 6-step decision flow and field definitions. | `predicted_ktas`, `reasoning_summary` |
| `analysis2/` | Forces a 5-step sequence: first look, chief complaint, primary modifiers, secondary modifiers, final level. Tells the model not to average conflicting findings and not to escalate on unsupported risk. | `first_look`, `chief_complaint`, `chief_complaint_risk`, `primary_modifiers`, `secondary_modifiers`, `decisive_factor`, `predicted_ktas` |
| `analysis3/` | Short prompt that tells the model to integrate the whole presentation and not over-weight a single finding, vital sign or the ED diagnosis. | `clinical_summary`, `ktas_justification`, `predicted_ktas` |

### Model runners

All runners read `ktas_prompt_cases.jsonl`, record `predicted_ktas`, latency, the raw output and whether the prediction matches `KTAS_RN` and `KTAS_expert`, and resume from the existing CSV.

| Script | Runs |
| --- | --- |
| `analysis/run_ktas_model.py`, `analysis2/` and `analysis3/` `run_ktas_gpt_5_4.py` | `azure-gpt-5.4` (or `--model`) through the LiteLLM endpoint in `.env`. Supports `--ca-bundle` and `--insecure` for TLS problems. |
| `*/run_ktas_ollama_models.py` | Local Ollama models (`mistral`, `gemma2:9b`, `llama3.1:8b`, `medgemma:4b`, `qwen2.5:7b`, `qwen3:1.7b`), one CSV per model, up to 3 attempts per case. |
| `analysis/run_ktas_transformers_models.py` | Hugging Face models loaded locally (default `WeiboAI/VibeThinker-3B`), one CSV per model. `analysis/` only. |

### Evaluation

| Script | What it does |
| --- | --- |
| `*/build_wide_model_comparison_csv.py` | Merges the per-model prediction files into one table: `case_id`, `KTAS_expert`, `KTAS_RN` and a `pred_<model>` column per model. Excludes `qwen3:1.7b` by default; add `--include-qwen3` to keep it. |
| `analysis/full_ktas_error_analysis.py` | The main evaluation. Scores every prediction column against `KTAS_expert` and against `KTAS_RN`. Metrics: accuracy, balanced accuracy, Cohen's kappa (plain, linear and quadratic weighted), macro F1, per-class precision/recall/F1, within-1-level accuracy, mean signed and absolute deviation, under-triage and over-triage rates, and "catastrophic" errors (off by 2 or more levels). Writes tables (CSV/JSON) and figures (confusion matrices, deviation histograms, per-class scores, under/over/catastrophic bars, cross-model comparisons) plus `clinical_interpretation.txt`. |
| `analysis2/` and `analysis3/` `full_ktas_error_analysis.py` | Wrappers that reuse the `analysis/` code, change the output folder and ignore that version's JSON text columns when detecting prediction columns. |
| `analysis/generate_two_table_markdown.py` | Ranks models by accuracy against `KTAS_expert` and adds a `KTAS_RN` row (nurse vs expert) as a human baseline. Writes `ktas_two_table_summary.md`. |
| `analysis2/` and `analysis3/` `generate_expert_summary_table.py` | Same idea with columns Accuracy, Over-triage, Under-triage, Within 1, More than 1. Writes `ktas_expert_summary.md`. |

Sign convention: deviation = `prediction - reference`. In KTAS, level 1 is the most urgent. A positive deviation (predicted number higher, i.e. less urgent than the reference) is under-triage; a negative deviation (more urgent than the reference) is over-triage.

The output files already in `scripts/analysis*/output/` (summary markdown, `model_reference_metrics.csv`, `clinical_interpretation.txt`, the `*_under_over_catastrophic.png` figures) were generated with the old, swapped labels. Re-run steps 5 and 6 of the pipeline to regenerate them. Until then, treat their under/over columns as swapped.

## Setup

Python 3. Data cleaning and the GPT and Ollama runners use only the standard library. Error analysis and summary tables need `pandas`, `numpy`, `scikit-learn`, `matplotlib` and `seaborn`. The Transformers runner needs `torch` and `transformers`. Ollama models need a local [Ollama](https://ollama.com) server.

Create a `.env` in the repo root for the GPT runner:

```
LITELLM=<api key>
LITELLM_BASE_URL=<endpoint>
LITELLM_CA_BUNDLE=<optional path to PEM>
```

`.env` is gitignored. Never commit it.

## Pipeline

Run from the repo root. Replace `analysis3` with `analysis` or `analysis2` for the other prompt versions.

```bash
# 1. Clean the raw workbook (drops 2 columns and rows containing Korean text)
python3 scripts/clean_original_data.py

# 2. Build prompt-ready cases (excludes KTAS_RN/KTAS_expert and other leakage columns from model input)
python3 scripts/analysis3/build_ktas_prompt_dataset.py

# 3. Run models (all resumable: finished case_ids are skipped; use --limit N to smoke-test, --no-resume to restart)
python3 scripts/analysis3/run_ktas_gpt_5_4.py            # analysis2/3: GPT via LiteLLM
python3 scripts/analysis/run_ktas_model.py               # analysis (v1) equivalent
python3 scripts/analysis3/run_ktas_ollama_models.py      # mistral, gemma2:9b, llama3.1:8b, medgemma:4b, qwen2.5:7b, qwen3:1.7b
python3 scripts/analysis/run_ktas_transformers_models.py # v1 only: HF models, e.g. WeiboAI/VibeThinker-3B

# 4. Merge per-model predictions into one wide CSV
python3 scripts/analysis3/build_wide_model_comparison_csv.py

# 5. Error analysis (confusion matrices, deviation histograms, kappa, over/under-triage)
python3 scripts/analysis3/full_ktas_error_analysis.py --input scripts/analysis3/output/model_comparison_wide.csv

# 6. Summary table
python3 scripts/analysis3/generate_expert_summary_table.py     # analysis2/3
python3 scripts/analysis/generate_two_table_markdown.py        # analysis (v1)
```

TLS errors (`CERTIFICATE_VERIFY_FAILED`) on the GPT runner: set `LITELLM_CA_BUNDLE` or pass `--ca-bundle`. `--insecure` disables verification and is a last resort.

More flag detail for v1 is in `scripts/analysis/README.md`.

## Outputs

Under each `scripts/analysis*/output/`:

- `ktas_prompt_cases.jsonl` - prompts sent to the models
- `*_predictions.csv`, `ollama/`, `transformers/` - per-model predictions with latency and raw output
- `model_comparison_wide.csv` - one row per case, one prediction column per model
- `full_error_analysis_models/` - `tables/`, `figures/`, `clinical_interpretation.txt`, and the markdown summary

## Results snapshot (analysis3, vs. expert, n=1209)

Accuracy and within-1 are unaffected by the under/over label fix.

| Model | Accuracy | Within 1 |
| --- | ---: | ---: |
| gemma2:9b | 43.7% | 95.8% |
| qwen2.5:7b | 35.0% | 75.3% |
| llama3.1:8b | 33.3% | 84.4% |
| medgemma:4b | 32.1% | 75.8% |
| mistral | 31.9% | 78.3% |
| qwen3:1.7b | 14.7% | 51.5% |
| KTAS_RN (nurse, reference) | 85.8% | 98.6% |

Full table (regenerate to get corrected under/over columns): `scripts/analysis3/output/full_error_analysis_models/ktas_expert_summary.md`.

## Notes

- The LLM prompts are an operational approximation of KTAS, not the official manual.
- `data/originaldata.xlsx` holds identifying columns; keep it out of any shared or public repo.
