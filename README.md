# experiments

Workspace for emergency department (ED) triage and prioritisation research. Most folders ask whether machine learning or LLMs can reproduce or support clinical urgency decisions. Each folder is self-contained and has its own README.

Datasets, model weights, and generated outputs are kept locally and excluded from Git. To run the analyses, obtain the required data through its authorized source.

| Folder | One-line summary | Data | Language |
|---|---|---|---|
| [`admissionprediction-master/`](#admissionprediction-master) | Predict hospital admission (R) and ESI triage level with LLMs (Python) | US ED, 560k visits | R, Python |
| [`iranhospital/`](#iranhospital) | Pairwise "which patient is more urgent?" LLM benchmark | Iranian ED triage data | Python |
| [`ktasanalysis/`](#ktasanalysis) | LLMs assigning KTAS levels, compared with nurse and expert labels | Korean ED, 1,267 cases | Python |
| [`llms-for-triage-main/`](#llms-for-triage-main) | Zero-shot, few-shot and KATE-retrieval LLM triage, plus ML baselines and LoRA fine-tuning | KTAS, MIMIC-IV, ESI Handbook | Python |
| [`ttsh/`](#ttsh) | IMDA-TTSH ED CT scan prioritisation proof of concept | Tan Tock Seng Hospital CT orders, Nov 2025 to Sep 2026 | Python |

---

## admissionprediction-master

Two lines of work on the same de-identified ED dataset, built on the 2018 paper by Hong, Haimovich and Taylor (PLoS ONE 13(7): e0201016). Anyone using the dataset should cite that paper.

1. **Admission prediction (R).** Predicts admitted vs. not admitted from triage and history data with XGBoost, a Keras deep neural network and logistic regression. It includes feature-set ablations (`onlytriage`, `onlyhx`, `topvars`), a training-size plateau analysis and Youden-index cutoffs.
2. **ESI triage-level prediction with LLMs (Python).** Asks LLMs to predict the Emergency Severity Index (1-5) from triage information, using KATE (kNN-retrieved few-shot examples built on BioBERT embeddings).

Key pieces:
- `master.R` and `Scripts/R/1-26` run the processing pipeline. This needs raw EDW extracts that are not included.
- `build_*.R` train the models. `youdens.R` computes the operating-point metrics.
- `llm-predict-triage1.py`, `llm-evaluate-triage1.py` and `triage_utils/` run and score the LLM experiment.
- `Results/` holds `5v_cleandf.RData` (560,486 x 972), the reduced feature CSVs, the train/test split, the KATE pool with embeddings, and per-model predictions in `Triage-ESI-Cleaned/`.

## iranhospital

**ED Triage Pairwise Urgency Benchmark.** Builds pairs of real triage cases (ground truth grade 1-5) and asks an LLM "which patient is more urgent, A or B?". The model's pick is scored against the grades.

- **Variants:** chief-complaint-only, full fields with missing-aware vitals, same-grade pairs, and a status-field ablation (does `CriticalStatus`, `StuporStatus` or `NeedFastExecute` leak the answer).
- **Experiments, roughly in order run:**
  1. Single-day prototype (2017-05-09).
  2. Status-field ablation on a fixed 50-pair sample, across Claude Opus and GPT models.
  3. Latency benchmark per model.
  4. Main experiment: a balanced cross-grade benchmark of 85 cases (17 per grade), giving 2,890 pairs.
  5. Comparison of hosted frontier models with small local models.
- **Layout:** `data/` (not public, do not commit), many `generate*`, `pairwisetest*` and summary scripts at the top level, and results under `output/` (`benchmarks/`, `results/`).

## ktasanalysis

Benchmarks LLMs on **Korean Triage and Acuity Scale (KTAS)** assignment and compares their predictions with the nurse (`KTAS_RN`) and expert (`KTAS_expert`) labels.

- `data/`: raw workbook, the de-identified `originaldata_cleaned.csv` used by everything else, and the KTAS guideline reference.
- `scripts/clean_original_data.py`: xlsx to cleaned CSV. It drops identifying columns and rows containing Korean text.
- Three prompt versions, each with its own prompt template, dataset builder, runners and `output/`:
  - `analysis/` (v1): level definitions and a 6-step decision flow.
  - `analysis2/` (v2): a forced 5-step reasoning sequence with 7 JSON keys.
  - `analysis3/` (v3): a concise prompt (clinical summary, justification, predicted KTAS).
- The `analysis2/` and `analysis3/` error-analysis scripts import from `analysis/`, so keep `analysis/` in place.
- Model input has the labels removed so the answer cannot leak.

## llms-for-triage-main

The broadest LLM-triage codebase here. It predicts ED triage acuity using **zero-shot, few-shot and retrieval-based (KATE) prompting**. KTAS is the main dataset. MIMIC-IV and the ESI Handbook are also supported.

- `llm-predict-triage.py` runs an LLM over a test set and saves predictions and metrics to `results/`. `llm-evaluate-triage.py` re-scores a saved predictions CSV.
- `utils/`: dataset registry and LLM clients, predictor classes, prompt templates, pairwise "who first" prediction, and BQR example selection.
- `scripts/`: reasoning generation for training cases, BQR experiments, pairwise runs across local Ollama models, error and neighbour analysis, classical-ML baselines (RF, AdaBoost, CV) and LoRA fine-tuning with unsloth.
- `data/`: `KTAS/`, `mimic-iv-public/`, `mimic-iv-private/` (credentialed, not distributed) and `ESI-Handbook/`.
- Also contains saved baseline models, a pytest suite and the original `edtriage.ipynb` notebook.

## ttsh

**IMDA-TTSH CT Scan Prioritizer**: an analysis workspace for a proof of concept on prioritising ED CT scans at Tan Tock Seng Hospital. The data is real, decrypted hospital extracts, so treat it as sensitive.

- `data/`: monthly "All Pt" and "CT Orders" extracts (Nov 2025 to Apr 2026), `newdata/` (Jul to Sep 2026), `encrypted/`, and `nni_cleaned_orders.csv`.
- `merged/`: combined and cleaned workbooks produced by the data-prep scripts.
- `scripts/`, by topic:
  - `data_prep`: merging, cleaning and data-quality checks.
  - `ct_queue_eda`: CT queue reconstruction and descriptive analysis.
  - `priority_policy`: policy-compliance checks and one-scanner simulations (NNI and DDR-Emergency).
  - `scan_priority_ml`: models that predict scan priority, saved in `models/`.
  - `ed_dx_icd10`: diagnosis urgency, using pairwise LLM choices and Bradley-Terry ranking.
  - `llm_queue_slot`: LLM queue-slotting experiments with small Ollama models (qwen3, gemma4).
  - `embeddings`: embedding utilities.
- `experiments/ct_priority_embeddinggemma2/`: a self-contained EmbeddingGemma k-NN study with its own README.
- `analysis/`: a flat folder of all script outputs (xlsx, png, md).
- `docs/DATA_CLEANING_NOTES.md`, `archive/` (superseded scripts), and `brainbodysep.tsv`, which must stay at the `ttsh/` root. It also holds a status deck (`20 Oct IMDA-TTSH CT Scan Prioritizer Update .pptx`).
- Run scripts with `.venv/bin/python scripts/<topic>/<script>.py`.

---

## Notes

- **Environments:** Folders keep their own `.venv`, so activate the one inside the folder you are working in.
