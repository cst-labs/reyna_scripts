# IMDA-TTSH CT Scan Prioritizer: analysis workspace

Analysis scripts, outputs and experiments for the TTSH ED CT scan prioritisation proof of concept.
For the project write-up see [`report/report.md`](report/report.md); for data-cleaning rules see
[`docs/DATA_CLEANING_NOTES.md`](docs/DATA_CLEANING_NOTES.md).

## Layout

| Path | What it holds |
|---|---|
| `scripts/` | All analysis code, grouped by topic (see below) |
| `data/` | Raw monthly extracts (`2025 11` to `2026 4` decrypted, `newdata/` Jul-Sep 2026, `encrypted/`) and `nni_cleaned_orders.csv` |
| `merged/` | Combined and cleaned workbooks produced by `scripts/data_prep` |
| `analysis/` | All script outputs (xlsx, png, md). Still one flat folder; file names start with the producing script's topic |
| `models/` | Saved scan-priority models (`.joblib`) |
| `experiments/ct_priority_embeddinggemma2/` | Self-contained EmbeddingGemma k-NN study with its own [README](experiments/ct_priority_embeddinggemma2/README.md), scripts, data and results |
| `docs/` | Data-cleaning notes |
| `archive/` | Superseded scripts and old backups. Nothing deleted |
| `brainbodysep.tsv` | Order Type to Site lookup. **Must stay at the repo root**: read by `clean_ct_orders.py` and `ct_scan_time_by_type.py` |

## Running scripts

Every script resolves the repo root from its own location, so it works from any working directory:

```sh
.venv/bin/python scripts/<topic>/<script>.py
```

Scripts import siblings by flat module name (e.g. `tune_*` imports `train_nni_scan_priority`), so keep each
topic folder's files together. Most scripts read `merged/` or `data/` and write to `analysis/`, overwriting the
previous output of the same name.

## Pipeline order

```
data/ (raw, decrypted)
  -> data_prep/merge_data.py                 -> merged/all_pt_combined.xlsx, ct_orders_combined.xlsx
  -> data_prep/clean_ct_orders.py            -> merged/ct_orders_cleaned.xlsx  (Jul-Sep: clean_ct_orders_newdata.py)
  -> data_prep/export_nni_cleaned_orders.py  -> data/nni_cleaned_orders.csv
  -> ct_queue_eda/build_ct_queue*.py         -> analysis/ct_queue_reconstruction*.xlsx
  -> ct_queue_eda, priority_policy, scan_priority_ml, ed_dx_icd10, llm_queue_slot analyses -> analysis/
```

## scripts/

### data_prep: merging, cleaning, data-quality
`merge_data` (decrypt + combine months), `trim_columns`, `clean_ct_orders` / `clean_ct_orders_newdata` (canonical cleaned CT orders),
`export_nni_cleaned_orders` (final NNI CSV), `consolidate_orders`, `analyse_merged`, `build_combined_field_reference`,
`trace_case_counts`, `export_imaging_not_allpt`, `patient_timeline`, `imaging_vs_lab_timing`, `checklist_question_analysis`,
`compile_report` (builds `analysis/master_report.xlsx`).

### ct_queue_eda: queue reconstruction and descriptive analysis
`build_ct_queue` (NNI / DR queues) and `build_ct_queue_newdata` (one unified queue, Jul-Sep 2026);
`ct_queue_analysis(_newdata)`, `ct_queue_priority_policy_check`, `ct_scan_time_by_type`, `ct_site_conflict_review`,
`ct_triage_analysis`, `ct_wait_time_by_consult_acuity`, `acuity_ct_deep_analysis`, `acuity_distribution_analysis`,
`consult_acuity_field_heatmap`.

### priority_policy: policy compliance and one-scanner simulation
NNI: `nni_queue_priority_policy_check`, `nni_single_scanner_policy_check`, `nni_one_scanner_estimation`,
`nni_priority_policy_simulation`, `nni_priority_sequence_rearrangement`, `nni_overtake_breakdown`,
`nni_empty_scanner_buffer_analysis`, `nni_wait_time_by_scan_priority`, `nni_scan_priority_location_check`.
DDR-Emergency (new scanner): `ddr_emergency_queue_priority_policy_check`, `ddr_emergency_one_scanner_estimation`,
`ddr_emergency_priority_policy_simulation`.

### scan_priority_ml: predicting Scan Priority (models in `models/`)
`train_nni_scan_priority` (base), `tune_nni_scan_priority_models`, `tune_nni_rf_et_scan_priority`,
`tune_nni_scan_priority_rf_v2`, `tune_nni_p2_scan_priority_rf_v2`, `nni_p2_scan_priority_rf` (P2-consult cohort),
`nni_p2_scan_priority_feature_distributions`, `nni_scan_priority_feature_distributions_allcases`.
Write-up: `analysis/nni_scan_priority_model_experiments.md`.

### ed_dx_icd10: diagnosis urgency
`ed_dx_pairwise_urgency_ollama` / `_laya` (pairwise LLM urgency choices), `ed_dx_bradley_terry_ranking`,
`ed_dx_scan_priority_distribution`, `icd10_scan_priority_distribution`, `icd10_description_vs_ed_dx_check`.
Outputs under `analysis/ed_dx_pairwise_urgency_*`.

### llm_queue_slot: LLM queue-slotting experiments (qwen3:1.7b, gemma4:e4b via Ollama)
- `nni_llm_queue_slot`, `_tests`, `_tests_tools`: first worked examples, no-tools vs tool-calling.
- `_tests_*` (`huge_supplement`, `combine_summary`, `error_analysis`, `pass_rate_chart`): supplements and summaries.
- `_consistency`, `_consistency_v2`: random-sample consistency runs; `_align_snapshots`, `_fix_drift`, `_stable_summary`,
  `_gemma_run1_summary`, `_gemma_wrong_entity_cases`, `_pass_rate_chart`: follow-ups.
- `_target_time_gemma4_less_fix_iqr`, `_less_fix_mean_sd_split5`: latest target-time variants.
- `experiments_spotcheck_models`: spot check of other models against the gemma4:e4b baseline (imports `_consistency_v2`).

Outputs under `analysis/nni_llm_queue_slot_consistency/` and `analysis/nni_llm_queue_slot_tests*.xlsx`.

### embeddings
`download_model.py` caches `google/embeddinggemma-2` locally. The embedding study itself lives in `experiments/`.

## archive/
- `scripts/llm_queue_slot/`: `positionagent.py` (byte-identical to `nni_llm_queue_slot_consistency_v2.py`) and the earlier
  `target_time_gemma4` / `_less_fix` / `_v2` / `_v3` variants. Their root path is already patched, so they still run.
- `backups/data/`: 12 `nni_cleaned_orders.csv.bak_*` files. `backups/merged/`: `ct_orders_cleaned_backup{,2,3}.xlsx`.

