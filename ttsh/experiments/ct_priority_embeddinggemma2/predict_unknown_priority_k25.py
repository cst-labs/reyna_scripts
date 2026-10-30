"""Fit top-5 k=25 configs on historical cases; predict priority for unlabeled new-data orders."""
from __future__ import annotations

import numpy as np
import pandas as pd

import evaluate_care_acuity_diagnosis as focused
import evaluate_care_acuity_diagnosis_5fold as fold5
import run_experiment as exp

OUT = exp.RESULTS_DIR / "unknown_priority_k25"
# Top 5 by CV macro-F1 rank + P1-recall rank among k=25 settings (uniform == similarity-weighted at k=25).
SHORTLIST = [
    ("base_plus_care_location_acuity", "inverse_sqrt"),
    ("base_plus_care_location", "inverse_sqrt"),
    ("base_plus_care_location_acuity", "inverse"),
    ("base_plus_care_location", "inverse"),
    ("base_plus_acuity", "inverse_sqrt"),
]

hist = pd.read_excel(exp.DATA_DIR / "historical_cases.xlsx")
new = pd.read_excel(exp.DATA_DIR / "newdata_test_cases.xlsx")
unk = new[new["Priority Label"].isna()].reset_index(drop=True)
print("unknown-priority rows:", len(unk), unk["Label Status"].value_counts().to_dict())

templates = focused.candidate_templates()
model = exp.embedder(type("A", (), {"device": "cpu", "local_files_only": True})())
y = hist["Priority Label"].to_numpy()
enc = lambda df, f: model.encode(exp.render_cases(df, f), prompt=exp.TASK_PREFIX, batch_size=32,
                                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

vecs = {}
for t in {t for t, _ in SHORTLIST}:
    try:
        h = fold5.ensure_cached_historical_vectors(t, templates[t], hist)
    except Exception as e:
        print(f"hist cache unusable for {t} ({e}); re-encoding")
        h = enc(hist, templates[t])
    vecs[t] = (h, enc(unk, templates[t]))

rows, preds, prob_rows = [], unk[["Case ID", "month", "Label Status"]].copy(), []
for t, adj in SHORTLIST:
    h, u = vecs[t]
    pred, probs, *_ = exp.knn_scores(u @ h.T, y, 25, "uniform", adj)
    name = f"{t}|{adj}"
    preds[name] = pred
    for i, lab in enumerate(exp.LABELS):
        preds[f"{name}|prob_{lab}"] = probs[:, i]
    top = probs.max(axis=1)
    prob_rows.append({"config": name, **{f"mean_prob_{lab}": probs[:, i].mean() for i, lab in enumerate(exp.LABELS)},
                      "mean_top_prob": top.mean(), "median_top_prob": np.median(top),
                      "pct_top_prob_ge_0.6": 100 * (top >= 0.6).mean()})
    for lab in exp.LABELS:
        rows.append({"config": name, "priority": lab, "n": int((pred == lab).sum()),
                     "pct": round(100 * (pred == lab).mean(), 1)})
dist = pd.DataFrame(rows).pivot(index="config", columns="priority", values=["n", "pct"])
print(dist.to_string())
probs_df = pd.DataFrame(prob_rows).set_index("config")
print(probs_df.round(3).to_string())
by_month = preds[[c for c in preds if "|prob_" not in c]].melt(id_vars=["Case ID", "month", "Label Status"], var_name="config", value_name="pred") \
    .groupby(["config", "month"])["pred"].value_counts(normalize=True).unstack().round(3)
print(by_month.to_string())
OUT.mkdir(exist_ok=True)
with pd.ExcelWriter(OUT / "unknown_priority_distribution.xlsx") as w:
    dist.to_excel(w, sheet_name="Distribution")
    probs_df.to_excel(w, sheet_name="Mean probabilities")
    by_month.to_excel(w, sheet_name="By month")
    preds.to_excel(w, sheet_name="Predictions", index=False)
