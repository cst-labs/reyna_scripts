"""
Bradley-Terry urgency ranking for ED Dx pairwise model choices.

Reads the cached Gemma pairwise diagnosis comparisons, fits one latent urgency
score per top-100 ED Dx mention, computes observed scan-priority scores from
the real priority distribution, and writes a comparison workbook.

Outputs:
  analysis/ed_dx_pairwise_urgency_gemma4-e4b/bradley_terry_ranking.xlsx

Environment overrides:
  N_BOOTSTRAPS=1000
  N_REPEATS=5
  BOOTSTRAP_SEED=42
"""
import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter
from scipy.optimize import minimize
from scipy.special import expit


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE, "data", "nni_cleaned_orders.csv")
PRIORITY_XLSX = os.path.join(BASE, "analysis", "ed_dx_scan_priority_distribution.xlsx")

MODEL = os.environ.get("MODEL", "gemma4:e4b")
_MODEL_SLUG = re.sub(r"[^A-Za-z0-9_.-]+", "-", MODEL).strip("-")
PAIRWISE_DIR = os.environ.get(
    "PAIRWISE_DIR", os.path.join(BASE, "analysis", f"ed_dx_pairwise_urgency_{_MODEL_SLUG}")
)
PAIRWISE_JSONL = os.path.join(PAIRWISE_DIR, "raw.jsonl")
OUT_XLSX = os.path.join(PAIRWISE_DIR, "bradley_terry_ranking.xlsx")

PROMPT_VERSION = "ed_dx_pairwise_urgency_v1"
TOP_N = 100
N_REPEATS = int(os.environ.get("N_REPEATS", "5"))
N_BOOTSTRAPS = int(os.environ.get("N_BOOTSTRAPS", "1000"))
BOOTSTRAP_SEED = int(os.environ.get("BOOTSTRAP_SEED", "42"))

CHOICE_A = "A"
CHOICE_B = "B"
CHOICE_TIE = "Tie/Insufficient information"
VALID_CHOICES = {CHOICE_A, CHOICE_B, CHOICE_TIE}


def normalize_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def top_ed_dx_mentions():
    df = pd.read_csv(DATA_PATH)
    mentions = []
    for value in df["ED Dx"].fillna(""):
        seen = set()
        for part in str(value).split(";"):
            dx = normalize_text(part)
            if dx and dx not in seen:
                mentions.append(dx)
                seen.add(dx)
    mention_counts = pd.Series(mentions, name="diagnosis").value_counts().head(TOP_N)
    primary_counts = df["Primary ED Dx"].map(normalize_text).value_counts()
    return pd.DataFrame(
        [
            {
                "source_rank": i,
                "diagnosis": dx,
                "n_orders": int(primary_counts.get(dx, 0)),
                "n_orders_ed_dx_mentions": int(n),
            }
            for i, (dx, n) in enumerate(mention_counts.items(), start=1)
        ]
    )


def pair_id(dx_a, dx_b):
    import hashlib

    return hashlib.sha1(f"{dx_a}\n{dx_b}".encode("utf-8")).hexdigest()[:12]


def build_pairs(diagnoses):
    rows = []
    records = diagnoses.to_dict("records")
    for a, b in combinations(records, 2):
        rows.append(
            {
                "pair_id": pair_id(a["diagnosis"], b["diagnosis"]),
                "diagnosis_a": a["diagnosis"],
                "diagnosis_b": b["diagnosis"],
            }
        )
    return rows


def sort_records(records):
    return sorted(
        records,
        key=lambda r: (
            int(r.get("repeat_index") or 0),
            float(r.get("created_at_unix") or 0),
        ),
    )


def load_pairwise_records(pairs):
    pair_lookup = {p["pair_id"]: p for p in pairs}
    grouped = defaultdict(list)
    n_raw = 0
    n_bad_json = 0
    with open(PAIRWISE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad_json += 1
                continue
            n_raw += 1
            if rec.get("model") != MODEL or rec.get("prompt_version") != PROMPT_VERSION:
                continue
            if rec.get("pair_id") not in pair_lookup:
                continue
            grouped[rec["pair_id"]].append(rec)

    selected = []
    pair_rows = []
    for pair in pairs:
        cached = sort_records(grouped.get(pair["pair_id"], []))
        use = cached[:N_REPEATS]
        selected.extend(use)
        counts = Counter(r.get("choice") for r in use)
        pair_rows.append(
            {
                **pair,
                "n_selected": len(use),
                "n_cached_total": len(cached),
                "n_cached_extra_ignored": max(0, len(cached) - N_REPEATS),
                "n_a": counts.get(CHOICE_A, 0),
                "n_b": counts.get(CHOICE_B, 0),
                "n_tie": counts.get(CHOICE_TIE, 0),
                "n_unparseable": counts.get("UNPARSEABLE", 0),
            }
        )
    return selected, pd.DataFrame(pair_rows), n_raw, n_bad_json


def choices_to_bt_rows(records, diagnosis_to_idx):
    rows = []
    for rec in records:
        choice = rec.get("choice")
        if choice not in VALID_CHOICES:
            continue
        i = diagnosis_to_idx[rec["diagnosis_a"]]
        j = diagnosis_to_idx[rec["diagnosis_b"]]
        if choice == CHOICE_A:
            win_i, win_j = 1.0, 0.0
        elif choice == CHOICE_B:
            win_i, win_j = 0.0, 1.0
        else:
            win_i, win_j = 0.5, 0.5
        rows.append((i, j, win_i, win_j))
    return aggregate_bt_rows(rows)


def aggregate_bt_rows(rows):
    totals = defaultdict(lambda: [0.0, 0.0])
    for i, j, win_i, win_j in rows:
        if i == j:
            continue
        if i > j:
            i, j, win_i, win_j = j, i, win_j, win_i
        totals[(i, j)][0] += win_i
        totals[(i, j)][1] += win_j
    return np.array([(i, j, w[0], w[1]) for (i, j), w in totals.items()], dtype=float)


def fit_bradley_terry(bt_rows, n_items):
    if len(bt_rows) == 0:
        raise ValueError("No valid pairwise rows available for Bradley-Terry fitting.")

    i_idx = bt_rows[:, 0].astype(int)
    j_idx = bt_rows[:, 1].astype(int)
    win_i = bt_rows[:, 2]
    win_j = bt_rows[:, 3]
    total = win_i + win_j

    def unpack(theta):
        scores = np.zeros(n_items)
        scores[:-1] = theta
        return scores

    def objective(theta):
        scores = unpack(theta)
        si = scores[i_idx]
        sj = scores[j_idx]
        value = np.sum(total * np.logaddexp(si, sj) - win_i * si - win_j * sj)
        grad_full = np.zeros(n_items)
        p_i = expit(si - sj)
        grad_i = total * p_i - win_i
        grad_j = total * (1.0 - p_i) - win_j
        np.add.at(grad_full, i_idx, grad_i)
        np.add.at(grad_full, j_idx, grad_j)
        return float(value), grad_full[:-1]

    result = minimize(
        objective,
        np.zeros(n_items - 1),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"Bradley-Terry fit did not converge: {result.message}")
    scores = unpack(result.x)
    scores = scores - scores.mean()
    return scores, result


def dense_ranks_desc(values):
    order = np.argsort(-np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), dtype=int)
    last_value = None
    current_rank = 0
    for pos, idx in enumerate(order, start=1):
        value = values[idx]
        if last_value is None or value != last_value:
            current_rank = pos
            last_value = value
        ranks[idx] = current_rank
    return ranks


def per_diagnosis_bt_counts(records, diagnoses):
    rows = []
    for dx in diagnoses:
        involved = [r for r in records if r.get("diagnosis_a") == dx or r.get("diagnosis_b") == dx]
        wins = losses = ties = unparseable = 0
        for rec in involved:
            choice = rec.get("choice")
            is_a = rec.get("diagnosis_a") == dx
            if choice == CHOICE_TIE:
                ties += 1
            elif choice == "UNPARSEABLE":
                unparseable += 1
            elif (choice == CHOICE_A and is_a) or (choice == CHOICE_B and not is_a):
                wins += 1
            elif choice in {CHOICE_A, CHOICE_B}:
                losses += 1
        valid_n = wins + losses + ties
        rows.append(
            {
                "diagnosis": dx,
                "model_comparison_count": valid_n,
                "completion_count": len(involved),
                "model_win_total": wins + 0.5 * ties,
                "model_loss_total": losses + 0.5 * ties,
                "tie_count": ties,
                "tie_pct": ties / valid_n if valid_n else np.nan,
                "unparseable_count": unparseable,
                "pairs_with_data": len({r.get("pair_id") for r in involved}),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_scores(selected_records, diagnoses, diagnosis_to_idx, rng):
    by_pair = defaultdict(list)
    for rec in selected_records:
        if rec.get("choice") in VALID_CHOICES:
            by_pair[rec["pair_id"]].append(rec)
    pair_groups = list(by_pair.values())
    n_items = len(diagnoses)
    scores = np.full((N_BOOTSTRAPS, n_items), np.nan)
    ranks = np.full((N_BOOTSTRAPS, n_items), np.nan)
    failures = 0

    for b in range(N_BOOTSTRAPS):
        sampled_rows = []
        for group in pair_groups:
            draw_idx = rng.integers(0, len(group), size=len(group))
            for k in draw_idx:
                rec = group[int(k)]
                i = diagnosis_to_idx[rec["diagnosis_a"]]
                j = diagnosis_to_idx[rec["diagnosis_b"]]
                choice = rec["choice"]
                if choice == CHOICE_A:
                    sampled_rows.append((i, j, 1.0, 0.0))
                elif choice == CHOICE_B:
                    sampled_rows.append((i, j, 0.0, 1.0))
                else:
                    sampled_rows.append((i, j, 0.5, 0.5))
        try:
            fit, _ = fit_bradley_terry(aggregate_bt_rows(sampled_rows), n_items)
        except Exception:
            failures += 1
            continue
        scores[b, :] = fit
        ranks[b, :] = dense_ranks_desc(fit)
        if (b + 1) % 100 == 0 or b + 1 == N_BOOTSTRAPS:
            print(f"  bootstrap {b + 1:,}/{N_BOOTSTRAPS:,}")
    return scores, ranks, failures


def observed_priority(diagnoses):
    obs = pd.read_excel(PRIORITY_XLSX, sheet_name="primary_dx_wide")
    obs = obs.rename(columns={"Primary ED Dx": "diagnosis"})
    obs = diagnoses[["diagnosis"]].merge(obs, on="diagnosis", how="left")
    for col in ["P1", "P2", "P3", "Unknown", "Total"]:
        obs[col] = obs[col].fillna(0).astype(int)
    denom_known = obs["P1"] + obs["P2"] + obs["P3"]
    denom_total = denom_known + obs["Unknown"]
    obs["observed_priority_score"] = np.where(
        denom_known > 0,
        (3 * obs["P1"] + 2 * obs["P2"] + obs["P3"]) / denom_known,
        np.nan,
    )
    obs["unknown_rate"] = np.where(denom_total > 0, obs["Unknown"] / denom_total, np.nan)
    for col in ["P1", "P2", "P3", "Unknown"]:
        obs[f"{col}_pct"] = np.where(denom_total > 0, obs[col] / denom_total, np.nan)
    obs["observed_rank"] = dense_ranks_desc(obs["observed_priority_score"].fillna(-np.inf).to_numpy())
    return obs.sort_values(["observed_rank", "diagnosis"]).reset_index(drop=True)


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 70)
    ws.freeze_panes = "A2"


def main():
    os.makedirs(PAIRWISE_DIR, exist_ok=True)
    diagnoses_df = top_ed_dx_mentions()
    diagnoses = diagnoses_df["diagnosis"].tolist()
    diagnosis_to_idx = {dx: i for i, dx in enumerate(diagnoses)}
    pairs = build_pairs(diagnoses_df)

    print(f"Loading pairwise cache: {PAIRWISE_JSONL}")
    selected_records, pair_summary, n_raw, n_bad_json = load_pairwise_records(pairs)
    print(f"Raw rows={n_raw:,}; selected rows={len(selected_records):,}; bad JSON={n_bad_json:,}")

    bt_rows = choices_to_bt_rows(selected_records, diagnosis_to_idx)
    print(f"Fitting Bradley-Terry model on {len(bt_rows):,} aggregate pair rows ...")
    bt_scores, result = fit_bradley_terry(bt_rows, len(diagnoses))
    bt_ranks = dense_ranks_desc(bt_scores)

    print(f"Running {N_BOOTSTRAPS:,} bootstrap samples ...")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_scores, boot_ranks, boot_failures = bootstrap_scores(selected_records, diagnoses, diagnosis_to_idx, rng)
    valid_boot = ~np.isnan(boot_scores).all(axis=1)
    if valid_boot.sum() == 0:
        raise RuntimeError("All bootstrap fits failed.")

    ci_low = np.nanpercentile(boot_scores, 2.5, axis=0)
    ci_high = np.nanpercentile(boot_scores, 97.5, axis=0)
    top3_pct = np.nanmean(boot_ranks <= 3, axis=0)
    rank_median = np.nanmedian(boot_ranks, axis=0)
    rank_low = np.nanpercentile(boot_ranks, 2.5, axis=0)
    rank_high = np.nanpercentile(boot_ranks, 97.5, axis=0)

    counts = per_diagnosis_bt_counts(selected_records, diagnoses)
    bt = diagnoses_df.merge(counts, on="diagnosis", how="left")
    bt["bt_score"] = bt_scores
    bt["bt_rank"] = bt_ranks
    bt["bt_score_ci_low"] = ci_low
    bt["bt_score_ci_high"] = ci_high
    bt["bootstrap_top3_pct"] = top3_pct
    bt["complete_win_separation"] = (bt["model_loss_total"] == 0) & (bt["model_comparison_count"] > 0)
    bt["complete_loss_separation"] = (bt["model_win_total"] == 0) & (bt["model_comparison_count"] > 0)
    bt = bt.sort_values(["bt_rank", "diagnosis"]).reset_index(drop=True)

    obs = observed_priority(diagnoses_df)
    boot_rank_summary = pd.DataFrame(
        {
            "diagnosis": diagnoses,
            "bootstrap_rank_median": rank_median,
            "bootstrap_rank_ci_low": rank_low,
            "bootstrap_rank_ci_high": rank_high,
            "bootstrap_top3_pct": top3_pct,
        }
    ).sort_values(["bootstrap_rank_median", "diagnosis"]).reset_index(drop=True)

    final = bt[
        [
            "diagnosis",
            "bt_rank",
            "bt_score",
            "bt_score_ci_low",
            "bt_score_ci_high",
            "bootstrap_top3_pct",
            "model_comparison_count",
            "tie_pct",
            "complete_win_separation",
            "complete_loss_separation",
        ]
    ].merge(
        obs[["diagnosis", "observed_rank", "observed_priority_score", "unknown_rate"]],
        on="diagnosis",
        how="left",
    )
    final["rank_difference_bt_minus_observed"] = final["bt_rank"] - final["observed_rank"]
    final = final.sort_values(["bt_rank", "diagnosis"]).reset_index(drop=True)

    notes = pd.DataFrame(
        [
            {"Note": f"Pairwise input: {PAIRWISE_JSONL}"},
            {"Note": f"Observed-priority input: {PRIORITY_XLSX}, sheet primary_dx_wide"},
            {"Note": f"Model={MODEL}; prompt_version={PROMPT_VERSION}; repeat target={N_REPEATS}; selected model rows={len(selected_records):,}."},
            {"Note": "Diagnosis universe is still the top-100 ED Dx mention list used by the pairwise experiment; n_orders is now the Primary ED Dx count, and n_orders_ed_dx_mentions keeps the original mention count for audit."},
            {"Note": "Bradley-Terry ties are handled as half-win for each diagnosis."},
            {"Note": f"Bootstrap resamples cached model choices within each pair; N_BOOTSTRAPS={N_BOOTSTRAPS:,}; seed={BOOTSTRAP_SEED}; failed bootstrap fits={boot_failures:,}."},
            {"Note": "Observed priority score = (3*P1 + 2*P2 + P3) / (P1 + P2 + P3). Unknown is excluded from that score and reported separately as unknown_rate."},
            {"Note": "BT diagnoses remain the top-100 ED Dx mention universe; observed-priority values are matched from primary diagnoses and may be zero/missing when a mention diagnosis is not present as a primary diagnosis."},
            {"Note": "Results are provisional if the pairwise cache is incomplete; model_comparison_count and pair summary counts show actual coverage."},
            {"Note": "complete_win_separation=True means a diagnosis had no model losses after tie half-splitting; an unregularized BT score can be effectively unbounded and should be interpreted by rank/top-3 stability rather than raw magnitude."},
            {"Note": f"BT optimizer converged={result.success}; objective={result.fun:.6f}; iterations={result.nit}."},
        ]
    )

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        bt.to_excel(writer, sheet_name="BT_Ranking", index=False)
        obs.to_excel(writer, sheet_name="Observed_Priority", index=False)
        final.to_excel(writer, sheet_name="Final_Comparison", index=False)
        boot_rank_summary.to_excel(writer, sheet_name="Bootstrap_Ranks", index=False)
        pair_summary.to_excel(writer, sheet_name="Pair_Summary", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
        for ws in writer.book.worksheets:
            auto_col_width(ws)

    if len(final) != TOP_N:
        raise RuntimeError(f"Expected {TOP_N} diagnoses in Final_Comparison, found {len(final)}")
    print(f"Wrote {OUT_XLSX}")
    print("\nTop 10 BT urgency ranks:")
    print(final.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
