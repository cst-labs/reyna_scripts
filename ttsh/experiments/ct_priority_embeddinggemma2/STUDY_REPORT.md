# Retrospective study report: CT scan-priority classification prototype

**Study status:** retrospective model development and input-field assessment. The available Jul–Sep 2026 evaluation file has already been examined, so there is no blind test result.  
**Run date:** 14–15 October 2026.  
**Intended use:** research on retrieving similar historical CT orders and suggesting a recorded priority, with clinician deferral for uncertain cases. This is not an autonomous clinical decision system.

## Study question and scope

For a new CT order, do the fields available when priority is assigned help retrieve similar historical orders and classify the priority recorded for those orders as P1, P2, or P3? The measured target is agreement with the priority recorded in `Ordering Question Responses`. It is not clinical appropriateness, patient outcome, radiology benefit, or safety. No independent clinician adjudication was available.

The report follows the study workflow in order: assemble cases and labels; establish input provenance and timing; construct model inputs; define validation; compare methods; examine results; assess deferral; then record limitations and the next evaluation needed.

## Step 1 — Assemble the historical and new-data cases

### Sources and matching

- Historical orders came from `merged/ct_orders_combined.xlsx`, sheet `Imaging`, retaining rows whose `Procedure Category` equals `IMG CT ORDERABLES`. Patient attributes were joined from `merged/all_pt_combined.xlsx`, sheet `All Patients`.
- New-data orders came from the original July, August, and September 2026 order exports; All Patients records came from the corresponding monthly extracts.
- The study retained unmerged order rows. Overlapping orders were not consolidated. A row was included only when its CSN matched an All Patients record. The join was many orders to one patient; duplicate CSNs in All Patients were reduced to the first row and the join enforced `m:1` uniqueness.
- Multiple orders from one patient remain separate cases but were kept together during patient-level validation splits.

### Priority target and exclusions

The target was parsed only from explicit `ED Scan Priority:` or `Scan Priority:` entries in `Ordering Question Responses`; accepted labels were P1, P2, and P3. Text such as “p2 giddiness” was not treated as a label. Repeated occurrences of the same priority were collapsed to one label. Rows containing distinct priority labels were marked composite and excluded from single-label evaluation for review. No composite-priority rows remained in the prepared sets below. Missing and unparsed priority responses were excluded from model fitting and labeled metrics.

| Prepared set | Matched source rows | Unmatched rows dropped | Usable single-priority labels | Missing response | Unparsed response | Composite labels |
|---|---:|---:|---:|---:|---:|---:|
| Historical | 14,345 | 257 | 8,233 | 44 | 6,068 | 0 |
| New data | 7,099 | 3 | 6,417 | 16 | 666 | 0 |

Six new-data rows marked `repeated_same_priority` were retained after collapsing repeated identical labels; 6,411 were marked `single_priority`. All 8,233 historical labeled rows were `single_priority`.

| Set | P1 | P2 | P3 |
|---|---:|---:|---:|
| Historical labeled, n=8,233 | 2,852 (34.6%) | 1,465 (17.8%) | 3,916 (47.6%) |
| New-data labeled, n=6,417 | 2,817 (43.9%) | 1,526 (23.8%) | 2,074 (32.3%) |

The majority label shifts from P3 in historical cases to P1 in new data. Accuracy alone is therefore inadequate, and differences between historical validation and new-data metrics may reflect prevalence and case-mix shift. Historical and new-data patient-group IDs do not overlap. The sets contain 7,935 and 5,099 distinct patient groups, respectively.

## Step 2 — Establish which fields can be inputs

Priority is the separate prediction label. It is never included in case text, structured model inputs, similarity vectors, or neighbor features. Only the two specifically parsed clinical answers below are taken from the response blob; the whole `Ordering Question Responses` field is not embedded.

### Provenance of “Working Diagnosis / Reason” and “Relevant Clinical Information”

Both are derived columns created by `prepare_cases.py` from the CT-order source column **`Ordering Question Responses`**. The preparation code parses values following these exact response keys:

| Prepared case-workbook field | Key parsed from `Ordering Question Responses` |
|---|---|
| `Working Diagnosis / Reason` | `Working diagnosis/Reason for exam` |
| `Relevant Clinical Information` | `Relevant clinical information` |

The parser extracts the answer after each named key; it does not copy the entire response blob. The scan-priority key is parsed separately to form the target. This distinction is important: the derived diagnosis/reason and clinical-information answers are model inputs; the priority answer remains label-only.

### Candidate inputs and timing

| Field group | Fields | Historical populated | New-data populated | Timing basis or caveat |
|---|---|---:|---:|---|
| Order and indication | Order Description; Working Diagnosis / Reason; Relevant Clinical Information | Core inputs | Core inputs | Derived from order description and the two named response keys above. |
| Diagnosis | Primary ED Dx; ED Dx | 100% | 100% | User confirmed ED diagnoses precede priority assignment; fields may be redundant. |
| Acuity | Triage Acuity; Consult Acuity; KIV Uptriage | Consult 100%; KIV 9.0% | Consult 100%; KIV 12.1% | User confirmed Consult Acuity and KIV Uptriage timing. Missing KIV is unknown, not “no.” |
| Arrival/presentation | Arrival Method; Chief Complaint | 100%; 99.8% | 100%; 91.7% | Requested candidate fields; chief complaint is more often missing in new data. |
| Care location/frailty | First ED Care Area; Primary Care Area; CFS (ED) | 100%; 100%; 65.5% | 100%; 100%; 52.5% | CFS has substantial missingness; care areas may encode workflow as well as clinical context. |
| Early observations | BP first; BP before clinician | 100%; 80.5% | 100%; 73.0% | Historical `BP before Consult` and new `BP before First EPP` were normalized to `BP before clinician`. BP last and flow timestamps were excluded. |
| Reattendance | Reatt Same ED; Reatt Reason | 8.7%; 1.5% | 8.5%; 1.6% | Sparse; absence is not assumed to mean “no.” Verify availability at the exact index decision. |
| Prior utilization/acuity flags | 72h Reatt NGEMR ED; 72h Reatt NGEMR ED then Admitted; HD/ICU from GW within 24H | 100% each | 100% each | User confirmed these are available before the CT order is placed. Their source-system timestamps/provenance still need verification before prospective use. |

`Classification Type` and `ED Primary Dx ICD10` are present in prepared data where available but were not evaluated as model inputs. Avoided inputs include direct priority fields, post-order BP/flow fields, identifiers, and the full response blob. Text inputs render missing or blank values as `Unknown`. Structured inputs impute missing numeric values with the training-fold median and a missingness indicator; missing categorical values are `Unknown` before one-hot encoding.

## Step 3 — Construct the case representations

### EmbeddingGemma retrieval

The model was `google/embeddinggemma-2`, local snapshot commit `914f7f89142e33e77833254d9c9b90c3cef7303b`. Historical cases and new queries used the same SentenceSimilarity prefix: `task: sentence similarity | query: `. Embeddings were 768-dimensional float32 vectors, L2-normalized. Cosine similarity was calculated as the dot product of normalized vectors. The historical reference set has about 8.2k cases, so direct vector comparison was sufficient; a vector database was not needed.

Case text was rendered as labeled lines (`Field: value`), with missing/blank values set to `Unknown`. The initial experiment assessed 16 predefined templates: description only; progressive indication, diagnosis, age, triage acuity and sex; each requested input block added to that base; the three pre-order flags individually and together; and combined context. A template called `plus_all_confirmed_context` is a code label, not proof that every field’s source timestamp has been independently verified.

For k-NN, the search used k ∈ {1, 3, 5, 9, 15, 25}, uniform or similarity-weighted voting, and no, inverse-square-root, or inverse class-frequency adjustment: 576 configurations in total. Similarity-weighted voting sums nonnegative cosine scores by class. Class adjustment multiplies class scores by inverse prevalence raised to α ∈ {0, 0.5, 1}; normalized adjusted scores are used for the vote. Ties resolve P1, P2, P3.

### Structured baselines

The majority baseline predicts the most common training label. The structured logistic-regression model uses numeric age and parsed systolic/diastolic components from BP first and BP before clinician. Categorical features are Sex, Arrival Method, Chief Complaint, KIV Uptriage, Reatt Same ED, Reatt Reason, Triage Acuity, Consult Acuity, Order Description, Working Diagnosis / Reason, Relevant Clinical Information, Primary ED Dx, ED Dx, First ED Care Area, Primary Care Area, and CFS (ED). Rare categorical levels (<5 training cases) are grouped; numeric features are median-imputed, scaled, and accompanied by missingness indicators. Its original validation grid used C ∈ {0.03, 0.1, 0.3, 1, 3, 10} and class weights balanced, none, P1×1.5, or P1×2. The selected setting was C=0.03 with balanced weights.

The later linear SVM used field-name-prefixed text with TF-IDF word unigrams/bigrams (`min_df=2`, at most 50,000 features, sublinear TF), one-hot categorical inputs with infrequent levels grouped at five cases, and numeric age/BP with median imputation, missingness indicators, and scaling. All transformations were fitted inside training folds. The classifier was multiclass `LinearSVC` with squared hinge loss.

## Step 4 — Define patient-level evaluation

Two historical validation designs were used at different stages:

1. **Initial prototype selection:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, using the first fold as a holdout (6,586 train and 1,647 validation cases). Patient group was the split unit. The held-out new-data workbook contains separate patients, but its metrics and case review have already been examined, so it is retrospective rather than blind.
2. **Later input-field assessment:** five patient-grouped folds over all historical labeled cases, `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)`. This script did not open `newdata_test_cases.xlsx`. Structured field ablations held logistic-regression settings fixed at C=0.03/balanced. That C had been selected in the earlier prototype, so this later assessment still has prior model-selection exposure.

Embedding template/k/vote/adjustment variants in the later field study used the same five folds, but selected the best mean macro F1 across those same folds. These are internal model-selection comparisons and may be optimistic, not independent performance estimates. Linear SVM hyperparameters were tuned within each outer training partition using three patient-grouped inner folds, then scored on the five outer folds. Its grid was C ∈ {0.01, 0.1, 1, 10} and class weights None/balanced; mean inner macro F1 selected the setting, with balanced accuracy and smaller C as tie-breaks. No new-data labels were used in the later field or SVM studies.

The original EmbeddingGemma and logistic regression models were refit on all historical labeled rows before scoring the separate Jul–Sep new-data file. Although separate from training, this file has been viewed in earlier analysis and cannot serve as an untouched test set.

## Step 5 — Compare the methods

### Initial grouped validation fold: input and k-NN selection

The initial-fold experiment was rerun on 15 October after adding the three user-confirmed pre-order flags to candidate inputs. It compares 16 predefined templates on the first patient-grouped seed-42 fold. For each template it tests six k values, both voting rules, and all three class adjustments: 16 × 6 × 2 × 3 = 576 configurations. The setting with the highest validation macro F1 is selected (ties: higher P1 recall, then balanced accuracy, smaller k, uniform vote, and less class adjustment). The selected embedding configuration is then used to score the separate Jul–Sep data file.

This compares the 16 specified candidates, not every possible subset of every field. It includes the base and progressive text/diagnosis inputs, each requested input block added to the base, the three pre-order flags individually and together, and the full-context template. The first-fold results select the candidate and also provide its validation score, so that score is selection-biased. The Jul–Sep file was examined in earlier work and is retrospective, not blind.

| Model | Accuracy | Balanced accuracy | Macro F1 | P1 precision / recall | P2 precision / recall | P3 precision / recall |
|---|---:|---:|---:|---:|---:|---:|
| Majority (P3) | 0.476 | 0.333 | 0.215 | 0.000 / 0.000 | 0.000 / 0.000 | 0.476 / 1.000 |
| EmbeddingGemma k-NN | 0.687 | 0.594 | 0.598 | 0.865 / 0.753 | 0.321 / 0.212 | 0.668 / 0.816 |
| Structured logistic regression | 0.649 | 0.640 | 0.625 | 0.866 / 0.818 | 0.311 / 0.532 | 0.736 / 0.570 |

Best k-NN result within each of the 16 initial-fold templates:

| Template | k | Vote / class adjustment | Macro F1 | Balanced accuracy | P1 / P2 / P3 recall |
|---|---:|---|---:|---:|---:|
| Care areas + CFS | 15 | Uniform / inverse sqrt | 0.598 | 0.594 | 0.753 / 0.212 / 0.816 |
| All context, including three flags | 25 | Uniform / inverse | 0.594 | 0.596 | 0.753 / 0.406 / 0.630 |
| Triage acuity | 25 | Uniform / inverse | 0.594 | 0.593 | 0.739 / 0.403 / 0.636 |
| Arrival + chief complaint | 9 | Uniform / inverse | 0.587 | 0.591 | 0.770 / 0.389 / 0.612 |
| Early BP | 15 | Uniform / inverse sqrt | 0.587 | 0.584 | 0.747 / 0.208 / 0.796 |
| Sex | 25 | Uniform / inverse | 0.580 | 0.579 | 0.749 / 0.375 / 0.614 |
| Diagnosis fields | 15 | Uniform / inverse sqrt | 0.578 | 0.577 | 0.740 / 0.270 / 0.722 |
| KIV + Consult Acuity | 5 | Uniform / none | 0.576 | 0.576 | 0.749 / 0.195 / 0.784 |
| Indication fields | 15 | Uniform / inverse sqrt | 0.575 | 0.577 | 0.763 / 0.273 / 0.694 |
| Reattendance | 9 | Uniform / inverse sqrt | 0.571 | 0.568 | 0.719 / 0.195 / 0.790 |
| Three timing-confirmed flags | 15 | Similarity-weighted / inverse | 0.584 | 0.584 | 0.742 / 0.386 / 0.624 |
| 72h reattendance | 9 | Uniform / inverse sqrt | 0.583 | 0.578 | 0.728 / 0.222 / 0.784 |
| HD/ICU within 24h | 9 | Uniform / inverse sqrt | 0.582 | 0.578 | 0.728 / 0.215 / 0.790 |
| 72h reattendance then admitted | 9 | Uniform / inverse sqrt | 0.578 | 0.573 | 0.721 / 0.208 / 0.790 |
| Age | 3 | Uniform / inverse sqrt | 0.558 | 0.554 | 0.674 / 0.256 / 0.731 |
| Description only | 5 | Uniform / inverse sqrt | 0.348 | 0.360 | 0.540 / 0.092 / 0.448 |

The selected setting was care areas/CFS with k=15, uniform voting, and inverse-square-root class adjustment (macro F1 0.598, balanced accuracy 0.594). The closest alternative by macro F1 was full context (0.594); it had higher P2 recall but lower P3 recall. Description alone performed poorly relative to templates including indication and diagnosis. This ranks only the predefined candidates; it does not establish universal feature optimality. All 576 results are in `results/baseline_knn/evaluation_results.xlsx`, sheet `kNN validation configs`.

### Focused combinations of care location, acuity, and diagnosis

To answer the specific combination question, a separate search compared every non-empty combination of three groups both with and without the same fixed order-information text: `Order Description`, `Working Diagnosis / Reason`, and `Relevant Clinical Information`. It tested all seven non-empty group combinations with the fixed text, those same seven without it, and a fixed-text-only baseline: 15 templates. Care location was `First ED Care Area` plus `Primary Care Area`; acuity was `Triage Acuity`, `Consult Acuity`, and `KIV Uptriage`; diagnosis was `Primary ED Dx` plus `ED Dx`. CFS, age, sex, BP, arrival, and reattendance were not part of this focused comparison. `Working Diagnosis / Reason` remained in the fixed base, so the diagnosis group specifically tests the two ED diagnosis columns. No empty-input template was tested because it would contain no case information.

For every template, the concatenated labeled fields were embedded with EmbeddingGemma using the same SentenceSimilarity setup. Cosine similarity ranked historical neighbors, and k-NN voted over their priorities. On the first seed-42 patient-grouped fold, the best of 36 settings per template (six k values × two voting rules × three class adjustments) was selected, for 540 settings total. The best group combination was then applied to the already-examined new-data file. The validation estimates are selection results and the new-data results are retrospective, not blind.

| Group combination | With fixed text: macro F1 / balanced accuracy / P2 recall | Without fixed text: macro F1 / balanced accuracy / P2 recall |
|---|---:|---:|
| Fixed text only | 0.580 / 0.581 / 0.280 | Not applicable (no group fields) |
| Care location | **0.627 / 0.623 / 0.304** | 0.575 / 0.577 / 0.392 |
| Acuity | 0.620 / 0.616 / 0.287 | 0.576 / 0.596 / 0.075 |
| Diagnosis | 0.579 / 0.578 / 0.276 | 0.539 / 0.545 / 0.396 |
| Care location + acuity | 0.626 / **0.631** / 0.440 | 0.599 / 0.592 / 0.355 |
| Care location + diagnosis | 0.625 / 0.619 / 0.300 | **0.613 / 0.605 / 0.266** |
| Acuity + diagnosis | 0.604 / 0.600 / 0.263 | 0.589 / 0.589 / **0.451** |
| Care location + acuity + diagnosis | 0.618 / 0.625 / **0.485** | 0.613 / 0.607 / 0.225 |

For all seven matched group combinations, including the fixed order-information text gave higher validation macro F1 than omitting it. Care location with the fixed text was the overall winner by macro F1. Without the fixed text, care location plus diagnosis was highest by macro F1 (0.613); acuity plus diagnosis had the highest P2 recall (0.451). With the fixed text, care location plus acuity had the highest balanced accuracy, and all three groups had the highest P2 recall (0.485). These are differences in agreement with the recorded labels; clinical stakeholders must decide whether the class trade-offs are useful.

The overall winner (fixed text + care location, k=3, uniform vote, inverse-square-root adjustment) scored on the retrospective new-data file at accuracy 0.683, balanced accuracy 0.634, macro F1 0.630, and P1/P2/P3 recall 0.825/0.326/0.751. The matched no-fixed-text care-location-only version had lower validation macro F1 (0.575), and was not the selected configuration. Results are not blind. The detailed 540-setting table, confusion matrices, suggestions, and de-identified supporting neighbors are in `results/care_acuity_diagnosis_single_split/care_acuity_diagnosis_combinations.xlsx`; template definitions, selection settings, and patient-group split IDs are in the adjacent metadata and split-manifest JSON files.

The care-location-only winner (k=3, uniform, inverse-square-root adjustment) scored on the retrospective new-data file at accuracy 0.683, balanced accuracy 0.634, macro F1 0.630, and P1/P2/P3 recall 0.825/0.326/0.751. Do not conflate this with the earlier broader-template search, which selected a different template and k. The detailed 288-setting table, confusion matrices, new-data suggestions, and de-identified supporting neighbors are in `results/care_acuity_diagnosis_single_split/care_acuity_diagnosis_combinations.xlsx`; template definitions, selection settings, and patient-group split IDs are in the adjacent metadata and split-manifest JSON files.

### Later historical input-field assessment

The fixed structured logistic-regression ablation changes one requested field/block at a time from the core set or removes a single field from the all-fields set. Scores are five-fold historical means.

| Structured input set | Macro F1 (SD) | Balanced accuracy | P1 recall | P2 recall | P3 recall |
|---|---:|---:|---:|---:|---:|
| Core only | 0.542 (0.006) | 0.561 | 0.611 | 0.580 | 0.493 |
| Core + Consult Acuity | 0.621 (0.016) | 0.643 | 0.833 | 0.571 | 0.525 |
| Core + First ED Care Area | 0.619 (0.013) | 0.636 | 0.761 | 0.612 | 0.535 |
| Core + Triage Acuity | 0.617 (0.012) | 0.634 | 0.761 | 0.608 | 0.535 |
| Core + Primary Care Area | 0.596 (0.015) | 0.611 | 0.712 | 0.574 | 0.546 |
| Core + Chief Complaint | 0.568 (0.012) | 0.587 | 0.680 | 0.575 | 0.507 |
| Core + Arrival Method | 0.563 (0.004) | 0.581 | 0.706 | 0.533 | 0.505 |
| Core + CFS (ED) | 0.563 (0.008) | 0.582 | 0.665 | 0.569 | 0.511 |
| Core + BP before clinician | 0.550 (0.010) | 0.568 | 0.661 | 0.549 | 0.494 |
| Core + BP first | 0.547 (0.008) | 0.564 | 0.623 | 0.567 | 0.502 |
| Core + Age at Visit | 0.546 (0.008) | 0.566 | 0.623 | 0.576 | 0.499 |
| Core + KIV Uptriage | 0.546 (0.005) | 0.566 | 0.621 | 0.580 | 0.496 |
| Core + Sex | 0.545 (0.005) | 0.564 | 0.614 | 0.583 | 0.495 |
| Core + Reatt Same ED | 0.543 (0.007) | 0.563 | 0.611 | 0.586 | 0.493 |
| Core + Reatt Reason | 0.542 (0.008) | 0.562 | 0.611 | 0.580 | 0.494 |
| All assessed fields | 0.625 (0.016) | 0.639 | 0.818 | 0.533 | 0.567 |

The strongest standalone additions to core were Consult Acuity, First ED Care Area, and Triage Acuity. Removing Arrival Method from the all-fields set gave 0.628 macro F1; removing Consult Acuity gave 0.621 and removing First ED Care Area gave 0.624. Most other single-field removals changed macro F1 by less than 0.005. These are associations within previously reused historical data, not causal field effects.

Six verified EmbeddingGemma template caches were available for the later five-fold comparison; the other initial template caches failed current case/text identity checks or included query cases and were not reused. Results:

| Template family | Selected k / vote / adjustment | Macro F1 | Balanced accuracy | P1 / P2 / P3 recall |
|---|---|---:|---:|---:|
| Care areas + CFS | 15 / similarity-weighted / inverse sqrt | 0.612 | 0.606 | 0.767 / 0.254 / 0.798 |
| All context in then-current cache | 15 / similarity-weighted / inverse | 0.609 | 0.615 | 0.777 / 0.461 / 0.606 |
| Arrival + chief complaint | 15 / similarity-weighted / inverse sqrt | 0.601 | 0.598 | 0.782 / 0.239 / 0.775 |
| Early BP | 15 / uniform / inverse sqrt | 0.598 | 0.594 | 0.758 / 0.232 / 0.792 |
| Reattendance | 25 / uniform / inverse | 0.590 | 0.594 | 0.734 / 0.447 / 0.599 |
| KIV + Consult Acuity | 15 / uniform / inverse sqrt | 0.589 | 0.586 | 0.755 / 0.235 / 0.768 |

The older “all context” cache was not the same as the subsequently revised template including the three timing-confirmed flags. Do not compare these two all-context rows as if they were the same experiment.

### Follow-up study of the three pre-order flags

On 15 October, the historical-only five-fold field assessment was refreshed to test `72h Reatt NGEMR ED`, `72h Reatt NGEMR ED then Admitted`, and `HD/ICU from GW within 24H` separately and together. User confirmation established their intended pre-order timing; exact source-system provenance remains to verify before prospective use. These runs use the same 8,233 historical cases, normalized 768-dimensional vectors, and seed-2026 patient folds. Within each template, k, voting rule, and class adjustment were selected on these same folds, so treat the scores as internal feature-selection results.

| EmbeddingGemma template | Selected k / vote / adjustment | Macro F1 | Balanced accuracy | P1 / P2 / P3 recall |
|---|---|---:|---:|---:|
| Base: core text, age, triage acuity, sex | 15 / uniform / inverse sqrt | 0.600 | 0.594 | 0.756 / 0.235 / 0.792 |
| Base + 72h Reatt NGEMR ED | 9 / similarity-weighted / inverse sqrt | 0.600 | 0.594 | 0.739 / 0.258 / 0.785 |
| Base + 72h Reatt then Admitted | 9 / similarity-weighted / inverse sqrt | 0.603 | 0.596 | 0.743 / 0.256 / 0.791 |
| Base + HD/ICU from GW within 24H | 9 / similarity-weighted / inverse sqrt | 0.599 | 0.593 | 0.738 / 0.254 / 0.787 |
| Base + all three flags | 15 / uniform / inverse | 0.593 | 0.597 | 0.740 / 0.442 / 0.608 |
| Full context including the three flags | Selected in field study | 0.606 | 0.602 | — |

Each single-flag addition made little difference to macro F1. Adding all three to the matched base raised P2 recall by about 0.21 absolute, reduced P3 recall by about 0.18, and lowered macro F1 by 0.007. The refreshed all-fields structured logistic model scored macro F1 0.6254 with the flags and 0.6254 without them (difference +0.000073); core plus the three flags scored 0.546 vs 0.542 for core alone. Whether higher P2 recall justifies lower P3 recall requires a clinical error-cost decision; these retrospective labels cannot settle it.

### Tuned linear SVM

The linear SVM was evaluated with nested patient grouping on the same seed-2026 outer folds. It did not use new-data rows or labels.

| Input set | Macro F1 (SD) | Balanced accuracy | Accuracy | P1 / P2 / P3 recall |
|---|---:|---:|---:|---:|
| Core presenting text | 0.609 (0.014) | 0.607 | 0.678 | 0.772 / 0.297 / 0.752 |
| Matched base: core + age, triage acuity, sex | 0.656 (0.011) | 0.651 | 0.728 | 0.839 / 0.309 / 0.805 |
| Matched base + all three flags | 0.655 (0.011) | 0.650 | 0.728 | 0.844 / 0.306 / 0.801 |
| All candidate fields, without the three flags | 0.660 (0.002) | 0.657 | 0.733 | 0.872 / 0.312 / 0.788 |
| All candidate fields, including the three flags | 0.659 (0.003) | 0.657 | 0.732 | 0.872 / 0.311 / 0.787 |

Balanced class weights were selected in every outer fold. C=1.0 was selected for all-fields variants in all folds; matched-base variants selected C=1.0 in four folds and C=0.1 in one. Adding the three flags changed the all-fields macro F1 by -0.001 and did not improve P2 recall. The initial full-field SVM run produced convergence warnings and was rerun at `tol=0.001`, `max_iter=10000`; selected C=1 fits converged in all folds. C=10 inner fits still warned and had lower inner macro F1 than C=1.

### Retrospective results on the existing Jul–Sep file

After selecting care areas/CFS, k=15, uniform voting, and inverse-square-root adjustment on the initial fold, the script refit the retrieval reference using all historical labeled rows and scored 6,417 labeled Jul–Sep orders. This file has already been examined in earlier work, so these are retrospective results, not blind test estimates. The later five-fold field study did not select this configuration.

| Model | Accuracy | Balanced accuracy | Macro F1 | P1 P/R | P2 P/R | P3 P/R |
|---|---:|---:|---:|---:|---:|---:|
| Majority (P3 historical class) | 0.323 | 0.333 | 0.163 | 0.000 / 0.000 | 0.000 / 0.000 | 0.323 / 1.000 |
| EmbeddingGemma k-NN, care areas + CFS, k=15 | 0.700 | 0.665 | 0.664 | 0.942 / 0.791 | 0.523 / 0.431 | 0.575 / 0.774 |
| Structured logistic regression | 0.722 | 0.703 | 0.695 | 0.956 / 0.865 | 0.489 / 0.702 | 0.671 / 0.542 |

Actual rows by predicted columns:

| EmbeddingGemma | P1 | P2 | P3 |
|---|---:|---:|---:|
| P1 | 2,228 | 194 | 395 |
| P2 | 76 | 658 | 792 |
| P3 | 62 | 407 | 1,605 |

| Structured logistic regression | P1 | P2 | P3 |
|---|---:|---:|---:|
| P1 | 2,437 | 232 | 148 |
| P2 | 51 | 1,072 | 403 |
| P3 | 60 | 889 | 1,125 |

The embedding model labeled 589/2,817 actual P1 cases as lower priority (194 P2, 395 P3); the structured model did so for 380/2,817 (232 P2, 148 P3). These are important cases for clinician error review, but the counts do not establish harm or safety. The structured model’s higher new-data scores than its initial validation scores cannot establish generalization, particularly with the change in prevalence and prior inspection of the file.

## Step 6 — Evaluate a clinician-deferral rule

For the original selected EmbeddingGemma configuration, thresholds were searched on validation only over neighbor agreement {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, validation-derived top-cosine quantiles {0, 0.10, 0.25, 0.50, 0.75, 0.90}, and vote margin {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}. Maximizing coverage × accepted-case balanced accuracy selected minimum agreement 0.50, minimum cosine 0.9392, and margin 0.0. Validation coverage was 89.9%; accepted-case balanced accuracy 0.598, macro F1 0.596, P1 recall 0.779.

Applied unchanged to the previously inspected Jul–Sep file, 5,654/6,417 labeled cases were accepted (88.1%) and 763 deferred. Accepted-case balanced accuracy was 0.666, macro F1 0.663, P1 recall 0.828; coverage among all rows, including unlabeled cases, was 86.8%. A margin threshold of zero provides no additional selectivity. This is only a retrospective threshold candidate. Any operating threshold requires clinical agreement and uncertainty must defer to a clinician; it is not clinically validated.

## Step 7 — Review errors and exploratory embedding diagnostics

Primary summaries were accuracy, balanced accuracy, macro F1, per-priority precision/recall, confusion matrix, P1 false negatives, and abstention coverage. The report does not encode the relative clinical cost of an adjacent error and a larger downgrade; this requires clinical input.

Exploratory embedding geometry gave mean within-class cosine distances of 0.0319 (P1), 0.0349 (P2), and 0.0326 (P3). Pairwise centroid cosine distances were 0.0072 (P1–P2), 0.0087 (P1–P3), and 0.0004 (P2–P3); sampled cosine silhouette was 0.0462 and Fisher between/within scatter ratio was 0.0553. These measures describe overlap in this embedding space only. Centroid distance, intra-class distance, silhouette, and Fisher ratio are not evidence of clinical reliability.

## Step 8 — Interpret results and state what remains unresolved

Within reused historical validation data, the structured model and tuned SVM were competitive with EmbeddingGemma. Consult Acuity and ED care-area fields were among the strongest standalone structured additions to core inputs. In the SVM, broad text plus structured fields performed better than core text alone; adding the three prior-utilization flags did not improve results. EmbeddingGemma field choices changed the P2/P3 balance substantially, with similar overall macro F1. No one candidate is established as a clinically preferred model.

Key limitations and unresolved items:

- Existing new-data results are retrospective and previously examined. A truly blind, later cohort is absent.
- The study predicts recorded priority, not an adjudicated correct priority. Label consistency and clinical meaning require review.
- Historical and new-data label prevalence differs, and new data contain substantial missing/unparsed priority responses.
- Repeated cases from the same patient are grouped within each split, but past rounds of model selection reused the historical cohort; later five-fold scores are not independent confirmation.
- Source timestamps for all proposed predictors have not been independently audited. User confirmed the three 72-hour/24-hour flags are pre-order; that operational provenance should be documented before prospective evaluation.
- No confidence intervals, paired inferential comparisons, calibration assessment, or subgroup performance analysis were produced.
- Text and vectors remain sensitive health information after direct identifiers are removed.

## Step 9 — Next evaluation and reproducibility

### Next blind evaluation

Use a consecutive post-September cohort with matched All Patients data and labels held by a data steward independent of model development. Freeze the input whitelist, template, embedding snapshot, preprocessing, hyperparameters, abstention threshold, code commit, and analysis plan before test labels are accessed. Give the modeling team an input-only file, keep labels separate, group all patient orders in one partition, and reveal labels once after predictions and defer status are locked. Report prespecified metrics, confusion matrix, urgent-class errors, coverage, missingness, and subgroup results. Do not revise on the blind cohort and continue to call it blind.

### Reproducibility and privacy

- Embedding model: `google/embeddinggemma-2`, commit `914f7f89142e33e77833254d9c9b90c3cef7303b`; prefix `task: sentence similarity | query: `; 768 dimensions; float32; L2 normalization; cosine dot product.
- Runtime: CPU, batch size 32. Recorded versions: NumPy 2.4.4, pandas 3.0.3, scikit-learn 1.9.0, sentence-transformers 6.1.0, Transformers 5.19.0, PyTorch 2.10.0.
- Seeds: 42 for original first-fold analysis; 2026 for five-fold field and SVM assessment. Exact pseudonymous case/patient group assignments, hashes, model snapshot and settings are in metadata and split manifests.
- HMAC pseudonymous IDs were created from CSN and order identity using `privacy_key.bin`; raw CSNs/order IDs are omitted from prepared workbooks. Keep the key, workbooks, cached vectors, review outputs, and neighbors access-restricted. Case text and embeddings remain sensitive clinical data.
- Embeddings were generated locally from cached model weights; case text was not sent to Hugging Face. Predictions and neighbor examples are research outputs and must not be used for live prioritization.

Reproduction commands from the repository root:

```sh
.venv/bin/python experiments/ct_priority_embeddinggemma2/prepare_cases.py
.venv/bin/python experiments/ct_priority_embeddinggemma2/run_experiment.py --device cpu --batch-size 32 --local-files-only
.venv/bin/python experiments/ct_priority_embeddinggemma2/evaluate_input_fields.py
.venv/bin/python experiments/ct_priority_embeddinggemma2/evaluate_linear_svm.py
.venv/bin/python experiments/ct_priority_embeddinggemma2/evaluate_care_acuity_diagnosis.py --device cpu --batch-size 32 --local-files-only
```

The `run_experiment.py` command regenerates the already-inspected Jul–Sep retrospective evaluation; it does not create a blind test.

### Main artifacts

- `data/historical_cases.xlsx` — historical labeled unmerged orders and candidate fields.
- `data/newdata_test_cases.xlsx` — Jul–Sep input rows, labels, audit fields; previously inspected, not blind.
- `data/composite_priority_review.xlsx` — composite-label audit output.
- `results/baseline_knn/evaluation_results.xlsx` — original one-fold selection, retrospective new-data evaluation, confusion matrices, abstention, suggestions and supporting neighbors.
- `results/field_selection/field_selection_validation.xlsx` — historical five-fold field assessment, including the refreshed three-flag analysis.
- `results/linear_svm/linear_svm_validation.xlsx`, `results/linear_svm/linear_svm_full_fields_validation.xlsx` — nested grouped SVM results.
- `results/care_acuity_diagnosis_single_split/care_acuity_diagnosis_combinations.xlsx` — focused initial-fold search over every combination of care location, acuity, and ED diagnosis, with retrospective new-data scoring and supporting neighbors.
- `results/baseline_knn/run_metadata.json`, `results/baseline_knn/split_manifest.json`, field/SVM metadata and split manifests — hashes, settings, and pseudonymous split assignments.
- `results/care_acuity_diagnosis_single_split/care_acuity_diagnosis_metadata.json`, `results/care_acuity_diagnosis_single_split/care_acuity_diagnosis_splits.json` — focused-search groups, selected settings, and pseudonymous patient-group assignments.
- `prepare_cases.py`, `run_experiment.py`, `evaluate_input_fields.py`, `evaluate_linear_svm.py`, `evaluate_care_acuity_diagnosis.py` — data preparation and evaluation scripts.
