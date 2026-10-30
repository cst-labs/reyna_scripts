# ED triage prediction: classical ML (admission) and LLMs (ESI level)

This repo holds two related lines of work on the same de-identified emergency department (ED) dataset:

1. **Admission prediction (R)**: the original 2018 study, which predicts hospital admission from triage and history data with XGBoost, a deep neural network, and logistic regression.
2. **ESI triage-level prediction with LLMs (Python)**: a newer experiment that asks LLMs to predict the Emergency Severity Index (ESI, 1-5) from triage information, using KATE (kNN-retrieved few-shot examples).

> **Citation.** All research using this dataset should cite: Hong WS, Haimovich AD, Taylor RA (2018) *Predicting hospital admission at emergency department triage using machine learning.* PLoS ONE 13(7): e0201016. https://doi.org/10.1371/journal.pone.0201016
>
> [![DOI](https://zenodo.org/badge/128982821.svg)](https://zenodo.org/badge/latestdoi/128982821)

## Layout

```
master.R, build_*.R, youdens.R   R analysis scripts (run from the repo root)
Scripts/R/                       R data-processing scripts 1-26 (need raw EDW extracts, not included)
Scripts/*.py                     data prep for the LLM experiment
llm-predict-triage1.py           run LLM predictions
llm-evaluate-triage1.py          score predictions
triage_utils/                    shared code for the LLM scripts (dataset registry, prompts, predictors, metrics)
Results/                         dataset, derived CSVs/embeddings, and LLM results
.env                             LITELLM / LITELLM_BASE_URL for hosted models (git-ignored; never commit)
```

## Data (`Results/`)

| File | What it is |
|---|---|
| `5v_cleandf.RData` | 560,486 x 972 R dataframe, one row per ED visit. The full de-identified dataset for the R experiments. Export with `readr::write_csv` (~1.7 GB) to use outside R. |
| `5v_relevant_features.csv` | Reduced table (223,972 rows) with `esi`, `age`, `gender`, `arrivalmode`, `previousdispo`, six triage vitals, `past_medical_history`, `chief_complaint`. Built by `Scripts/filter_relevant_features.py`. |
| `5v_relevant_features_train.csv` / `_test_500.csv` | Stratified split on `esi` (seed 42): 500-row test set, rest is train. |
| `5v_relevant_features_kate_train_50000.csv` | 50,000-row stratified retrieval pool for KATE. |
| `..._kate_train_50000_embeddings.npy` | BioBERT sentence embeddings of the pool (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`). |
| `Triage-ESI-Cleaned/` | Per-model prediction CSVs, `_metrics.json`, and collated summaries (see below). |

## Experiment 1: admission prediction (R)

Binary outcome: admitted vs. not. Run scripts from the repo root. Processing scripts in `Scripts/R/` take raw SQL-extract CSVs (not included, PHI); the analysis scripts use `Results/5v_cleandf.RData`.

**Models**
- **XGBoost**: `build_boost.R` (cross-validated hyperparameter search; also writes the design matrix used by later scripts), `build_final.R` (final model, test AUC with 95% CI), `build_importance_ci.R` (100 retrainings for average information gain per variable).
- **Deep neural network (keras)**: `build_keras.R` (tuning), `build_final_keras.R` (final).
- **Logistic regression**: `build_final_lr.R` (keras network with no hidden layer).

**Feature-set ablations.** Each model is repeated on subsets of the variables:

| Variant | Variables | Scripts (suffix) |
|---|---|---|
| full | everything | none |
| `onlytriage` | triage-time variables only | `build_boost_onlytriage`, `build_final_onlytriage`, `build_keras_onlytriage`, `build_final_keras_onlytriage`, `build_final_lr_onlytriage` |
| `onlyhx` | history variables only | same pattern with `onlyhx` |
| `topvars` | ESI, demographics, outpatient medication counts, hospital-usage stats (top XGBoost information gain) | `build_boost_topvars`, `build_final_topvars` |

Run the XGBoost script of a variant first; it writes the design matrix the keras/LR scripts reuse. Variable filtering lives in `Scripts/R/23_useonlytriage.R` and `24_usetopvars.R`.

**Other analyses**
- `Scripts/R/20c_splitdataindex_testplateau.R`: training-set size plateau (1%, 10%, 30%, 50%, 80%, 100% of train, fixed test set).
- `youdens.R`: Youden-index cutoff, sensitivity, specificity, PPV, NPV for each model (also fixed-specificity comparisons).
- Splits (`20b_splitdataindex.R`): 56,000 test, 56,000 validation, remainder train; validation/train re-split for 5-fold CV.

**Processing pipeline** (`master.R` runs `Scripts/R/1`-`15`, then `20`-`26` build model inputs). Per visit it derives: ESI and demographics (1), arrival time (2), previous disposition (3), past medical history as AHRQ CCS categories (4), labs and historical lab summaries (6, 8, 10), prior visit/admission counts (7, 7b), ED and triage vitals plus history (9, 9b, 9c), imaging/EKG counts (11, 12), outpatient medication classes (13), surgical history (14), top-200 chief complaints (15). Scripts 20-26 clean, split, build the numeric matrix, run the XGBoost worker, and impute/scale for keras.

## Experiment 2: LLM ESI prediction (Python)

**Task.** Given age, sex, arrival mode, previous disposition, triage vitals, past medical history, and chief complaint, predict ESI level 1-5. Evaluated on the 500-row held-out test set.

**Pipeline**
```bash
python Scripts/filter_relevant_features.py          # 5v_cleandf.RData -> relevant-features CSV (needs pyreadr)
python Scripts/create_cleaned_triage_split.py       # train + 500-row test
python Scripts/create_cleaned_kate_training_pool.py # 50k retrieval pool
python Scripts/generate_cleaned_kate_embeddings.py  # BioBERT embeddings (needs sentence-transformers)

python llm-predict-triage1.py --dataset Triage-ESI-Cleaned --model gemma2:9b \
    --strategy KATE --k_shots 5 --json --start 0 --end 499
python llm-evaluate-triage1.py --dataset Triage-ESI-Cleaned --start 0 --end 499 \
    --parameters <prediction-file-stem> --ordinal --by_class
```
The derived files are already in `Results/`, so you can start at the predict step.

**Models.** Local models via Ollama (default `http://127.0.0.1:11434`): `gemma2:9b`, `qwen2.5:7b`, `medgemma:4b`, `llama3.1:8b`, `qwen3:1.7b`. Hosted models via LiteLLM (`azure-*`; configure `.env`), plus OpenAI/Claude/Gemini providers. The provider is chosen from the model-name prefix (`triage_utils/core.py`).

**Strategies** (`--strategy`, in `triage_utils/predictors.py`): `Vanilla`, `Vanillav0`, `CoT`, `AutoCoT`, `DemonstrationCoT`, `FewShot`, `FewShotCoT`, `KATE`, `KATECoT`, `KATEAutoCoT`, `SelfConsistency`, `MultiAgent`. Only **KATE with 5 shots** has been run on this dataset so far.

**Knobs and ablations**: `--k_shots`, `--serialization` (`spaces|commas|newline|json|natural|natural_sex_race|natural_full`), `--json` (structured output), `--detailed_instructions`, `--vitals_off` (drop vitals), `--bias`, `--k_shots_ablation`, `--temperature`, `--num_predict`, `--save_every` (checkpointing, resumable with `--load_predictions`). These exist in the code but have not been run here.

**Results so far** (KATE, 5 shots, json, natural serialization, n=500; `Results/Triage-ESI-Cleaned/collated_metrics_5shot.csv`):

| Model | Accuracy | Weighted F1 | Quadratic kappa | MSE |
|---|---|---|---|---|
| azure-gpt-5.4 | 0.616 | 0.610 | 0.610 | 0.456 |
| gemma2:9b | 0.582 | 0.577 | 0.560 | 0.514 |
| qwen2.5:7b | 0.578 | 0.579 | 0.535 | 0.580 |
| medgemma:4b | 0.578 | 0.576 | 0.515 | 0.582 |
| llama3.1:8b | 0.562 | 0.561 | 0.537 | 0.588 |
| qwen3:1.7b | 0.560 | 0.544 | 0.474 | 0.560 |

Other files in `Results/Triage-ESI-Cleaned/`: `collated_error_analysis_*`, `top_confusions_5shot.csv`, `per_true_esi_predicted_level_stats_5shot.csv`, `one_row_per_true_esi_level_5shot.csv`.

## Setup notes

- **R**: `readr plyr dplyr reshape2 parallel caret xgboost doMC pROC keras`. Set the working directory to the repo root. XGBoost results used 5 threads.
- **Python** (no pinned requirements file): `pandas numpy scikit-learn tqdm matplotlib sentence-transformers pyreadr`, plus a client for whichever model provider you use.
- `triage_utils/core.py` also lists legacy datasets (MedQA, MIMIC-IV, KTAS) that need a `./data/` directory that is not in this repo. Only `Triage-ESI-Cleaned` works out of the box.
- `llm-evaluate-triage1.py` reads predictions from `./results/<dataset>/`, while this repo stores them in `Results/`. That works on macOS (case-insensitive) but needs a rename or symlink on Linux.
