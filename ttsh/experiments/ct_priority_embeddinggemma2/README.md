# CT priority EmbeddingGemma 2 experiment data

Run `python3 experiments/ct_priority_embeddinggemma2/prepare_cases.py` to
regenerate the prepared workbooks. Both sets use original, unmerged CT order
rows filtered to `Procedure Category == IMG CT ORDERABLES`. An order is kept
only when its CSN matches an All Patients row. Historical orders come from
`merged/ct_orders_combined.xlsx`; new-data orders come from the original July,
August, and September 2026 CT exports. Patient fields are joined from
`merged/all_pt_combined.xlsx` for historical rows and the matching period's
All Patients extracts for new-data rows.

Run the full evaluation with:

```sh
python3 experiments/ct_priority_embeddinggemma2/run_experiment.py
```

The model is loaded locally as `google/embeddinggemma-2`. The script uses the
same `task: sentence similarity | query: ` prefix for reference and query
orders, 768 dimensions, float32 inference, L2 normalization, and cosine via
the dot product of normalized vectors. It compares description-only text with
indications, ED diagnoses, age, triage acuity, sex, KIV Uptriage, Consult
Acuity, Arrival Method, Chief Complaint, First ED Care Area, Primary Care
Area, CFS (ED), early BP, and current-encounter reattendance fields. It tests
k = 1, 3, 5, 9, 15, and 25, uniform and similarity-weighted votes, and three
class adjustments; majority and structured logistic-regression baselines are
also evaluated.

Historical cases are split by patient with a seeded five-fold stratified group
split (first fold for validation). The best k-NN setting is chosen on
validation by macro-F1, with ties resolved by P1 recall, balanced accuracy,
smaller k, then uniform voting. The abstention candidate is selected on
validation by maximizing coverage × selective balanced accuracy over the
neighbor-agreement, top cosine similarity, and vote-margin grid. These are
research heuristics only; thresholds require clinical agreement. Output
suggestions are explicitly marked research candidates, and uncertain cases
defer to clinician review.

Results, validation split IDs, run metadata, and de-identified supporting
neighbors are written under `results/`. Cached vectors are written under
`artifacts/embeddings/`; vectors and outputs can contain sensitive clinical
information even though direct IDs are omitted, so keep these directories
access-restricted. No case text is sent to the model host: only public model
weights are downloaded, then inference runs locally. If the model is not
cached and network access is unavailable, run `--structured-only` for the
non-embedding baselines or cache the model before a full run.

The completed run, selected configurations, test results, urgent-class error
counts, and validation-only abstention candidate are summarized in
`results/embedding_run_status.md`. The full metrics, confusion matrices,
model comparisons, de-identified review output, and supporting neighbors are
in `results/baseline_knn/evaluation_results.xlsx`. The detailed methods, input ablations,
results, limitations, and blind-test requirements are in `STUDY_REPORT.md`.
Do not tune further on held-out new-data
results. This remains retrospective research against recorded priority labels,
not an autonomous or clinically validated system.

Run `python3 experiments/ct_priority_embeddinggemma2/evaluate_input_fields.py`
to compare input fields using five patient-grouped folds on historical cases.
This analysis does not open the new-data file; see
`results/field_selection/field_selection_validation.xlsx` for fold scores and settings. Neither
the historical source nor the existing Jul–Sep 2026 new-data file can now be
called a blind test: both have been used in previous experiments. A new
consecutive cohort with labels held by an independent reviewer is required for
a genuinely blind evaluation.

Run `python3 experiments/ct_priority_embeddinggemma2/evaluate_linear_svm.py`
to fit and tune the patient-grouped linear SVM baseline using nested validation.
The full-field tolerance check is run with
`python3 experiments/ct_priority_embeddinggemma2/evaluate_linear_svm.py --full-fields-only`.
Both commands use historical cases only.

The user confirmed that `72h Reatt NGEMR ED`,
`72h Reatt NGEMR ED then Admitted`, and `HD/ICU from GW within 24H` are
available before the CT order is placed. They are eligible candidate inputs
and are compared individually and jointly. This timing is a user-confirmed
data-provenance assumption; verify it against source-system timestamps before
any prospective use. Priority remains a separate target and is never included
in embedding text. BP before-consult and BP before-first-EPP source variants
are normalized as `BP before clinician`; later BP fields are not model inputs.

Priority is extracted only from the explicit `ED Scan Priority:` field in
`Ordering Question Responses`, so strings such as “p2 giddiness” in clinical
text are not treated as labels. Repeated occurrences of the same priority
collapse to one order label. Distinct priority labels on one original order
are sent for review.

`data/historical_cases.xlsx` contains only historical orders with one
normalized priority class. `data/newdata_test_cases.xlsx` retains all new CT
orders for inference; unlabeled targets remain marked in `Label Status`.
Composite cases from both sources are listed in
`data/composite_priority_review.xlsx` and are not scored as single-label
cases.

The prepared workbooks omit CSN and raw order IDs. `privacy_key.bin` is a
random HMAC key used only to create stable pseudonymous case and patient group
IDs across regenerations and both datasets. Keep it access-restricted and do
not distribute it with the workbooks. Regeneration requires retaining this
key.

Preparation audit from the current source files (matched orders only):

- Historical CT: 14,345 matched original order rows; 8,233 labeled orders in
  the database file (P1 2,852; P2 1,465; P3 3,916), plus 6,068 rows without a
  parseable priority and 44 blank priority responses.
- New-data CT: 7,099 matched original order rows; 6,417 orders with one
  priority after collapsing repeats (P1 2,817; P2 1,526; P3 2,074), 666 rows
  without a parseable priority, 16 blank priority responses, and no composite
  labels on individual orders.
- The source contained 257 historical and 3 new-data CT orders without a
  matching All Patients CSN; these were dropped. No patient-group overlap was
  found between the current historical and new-data sets.
