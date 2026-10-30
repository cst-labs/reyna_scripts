import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

OUT = "analysis"
os.makedirs(OUT, exist_ok=True)

# ── Palette (dataviz skill reference palette, light mode) ──────────────────────
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
CAT_COLORS = {"P1": "#2a78d6", "P2": "#eb6834", "P3": "#1baf7a"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

ACUITY_ORDER = ["P1", "P2", "P3"]

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRIDLINE,
})

# ── Load & clean ────────────────────────────────────────────────────────────────
df = pd.read_excel("merged/all_pt_cleaned.xlsx")

df["Consult Acuity"] = df["Consult Acuity"].replace({"(P) P2": "P2"})
df = df[df["Consult Acuity"].isin(ACUITY_ORDER)].copy()

df["Triage Acuity"] = df["Triage Acuity"].str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
df.loc[~df["Triage Acuity"].isin(ACUITY_ORDER), "Triage Acuity"] = pd.NA

AGE_GROUP_ORDER = ["<19", "19 to <40", "40 to <65", "65 to <75", "75 to <85", ">/= 85"]
df["Age Group"] = pd.Categorical(df["Age Group"], categories=AGE_GROUP_ORDER, ordered=True)

CFS_ORDER = [
    "1 - Very Fit", "2 - Well", "3 - Managing Well", "4 - Very Mild Frailty",
    "5 - Mild Frailty", "6 - Moderate Frailty", "7 - Severe Frailty",
    "8 - Very Severe Frailty", "9 - Terminally Ill",
]
df["CFS (ED)"] = pd.Categorical(df["CFS (ED)"], categories=CFS_ORDER, ordered=True)

REATT_ORDER = ["24H ReAtt", "72H ReAtt", "30D ReAtt"]
df["Reatt Same ED"] = pd.Categorical(df["Reatt Same ED"], categories=REATT_ORDER, ordered=True)

# Singapore MOH-style blood-pressure categories, derived from raw "SYS/DIA"
# strings (an optional leading "(!) " abnormal-result flag is stripped first).
BP_CATEGORY_ORDER = [
    "Normal BP", "High-Normal BP",
    "Grade 1 Hypertension", "Grade 2 Hypertension", "Isolated Systolic Hypertension",
]


def classify_bp(series):
    parsed = series.str.replace(r"^\(!\)\s*", "", regex=True).str.extract(r"(\d+)\s*/\s*(\d+)")
    sysbp = parsed[0].astype(float)
    diabp = parsed[1].astype(float)
    mask = sysbp.notna() & diabp.notna()

    # Priority is checked worst-to-best, first match wins (np.select).
    # Isolated Systolic Hypertension is checked first so it always wins over
    # Grade 1/2 whenever diastolic stays under 90 — that's the defining trait
    # of "isolated", regardless of how high the systolic reading climbs.
    conditions = [
        (sysbp >= 140) & (diabp < 90),
        (sysbp >= 160) | (diabp >= 100),
        (sysbp >= 140) | (diabp >= 90),
        (sysbp >= 130) | (diabp >= 80),
    ]
    choices = [
        "Isolated Systolic Hypertension", "Grade 2 Hypertension", "Grade 1 Hypertension",
        "High-Normal BP",
    ]
    category = pd.Series(np.select(conditions, choices, default="Normal BP"), index=series.index)
    category[~mask] = pd.NA
    return pd.Categorical(category, categories=BP_CATEGORY_ORDER, ordered=True)


df["BP first Category"] = classify_bp(df["BP first"])
df["BP before Consult Category"] = classify_bp(df["BP before Consult"])

# ── Crosstab heatmaps ────────────────────────────────────────────────────────────
HEATMAP_FIELDS = [
    "Triage Acuity",
    "Arrival Method",
    "ED Disposition",
    "Referral Source Type",
    "First ED Care Area",
    "Primary Care Area",
    "Classification Type",
    "Sex",
    "CFS (ED)",
    "Age Group",
    "BP first Category",
    "BP before Consult Category",
    "Reatt Same ED",
]

crosstabs = {}


def slugify(field):
    return re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")


def plot_heatmap(field):
    sub = df[[field, "Consult Acuity"]].dropna()
    counts = pd.crosstab(sub[field], sub["Consult Acuity"]).reindex(columns=ACUITY_ORDER, fill_value=0)
    if isinstance(sub[field].dtype, pd.CategoricalDtype) and sub[field].dtype.ordered:
        counts = counts.reindex([c for c in sub[field].dtype.categories if c in counts.index])
    else:
        counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    crosstabs[field] = (counts, pct)

    fig_h = max(2.5, 0.45 * len(counts) + 1.2)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    seq_cmap = sns.light_palette(SEQ_BLUE[4], as_cmap=True)
    sns.heatmap(
        pct, annot=counts, fmt="d", cmap=seq_cmap, vmin=0, vmax=100,
        linewidths=2, linecolor=SURFACE, cbar_kws={"label": "% within row"},
        annot_kws={"color": INK_PRIMARY, "fontsize": 9}, ax=ax,
    )
    ax.set_title(f"{field} vs. Consult Acuity\n(n={len(sub):,}; cell = count, color = row %)",
                 fontsize=11, color=INK_PRIMARY, loc="left")
    ax.set_xlabel("Consult Acuity", color=INK_SECONDARY)
    ax.set_ylabel(field, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"heatmap_{slugify(field)}_vs_consult_acuity.png"), dpi=150,
                facecolor=SURFACE)
    plt.close(fig)


for field in HEATMAP_FIELDS:
    plot_heatmap(field)

# ── Rate charts (secondary binary/near-binary fields) ───────────────────────────
RATE_FIELDS = [
    ("72h Reatt NGEMR ED", lambda s: s == 1),
    ("72h Reatt NGEMR ED then Admitted", lambda s: s == 1),
    ("HD/ICU from GW within 24H", lambda s: s == 1),
]

rate_tables = {}


def plot_rate_chart(field, flag_fn):
    flagged = flag_fn(df[field])
    grp = pd.DataFrame({"flag": flagged, "acuity": df["Consult Acuity"]})
    tbl = grp.groupby("acuity")["flag"].agg(n="size", rate=lambda s: s.mean() * 100).reindex(ACUITY_ORDER)
    rate_tables[field] = tbl

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(ACUITY_ORDER, tbl["rate"], color=[CAT_COLORS[a] for a in ACUITY_ORDER], width=0.6)
    for bar, (acuity, row) in zip(bars, tbl.iterrows()):
        ax.annotate(f"{row['rate']:.1f}%\n(n={int(row['n']):,})",
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)
    ax.set_ylim(0, max(tbl["rate"].max() * 1.35, 5))
    ax.set_title(f"{field} rate by Consult Acuity", fontsize=11, color=INK_PRIMARY, loc="left")
    ax.set_ylabel("Rate (%)", color=INK_SECONDARY)
    ax.set_xlabel("Consult Acuity", color=INK_SECONDARY)
    sns.despine(ax=ax, left=True)
    ax.grid(axis="y", color=GRIDLINE)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"rate_{slugify(field)}_by_consult_acuity.png"), dpi=150,
                facecolor=SURFACE)
    plt.close(fig)


for field, flag_fn in RATE_FIELDS:
    plot_rate_chart(field, flag_fn)

# ── Excel summary ────────────────────────────────────────────────────────────────
with pd.ExcelWriter(os.path.join(OUT, "consult_acuity_field_crosstabs.xlsx"), engine="openpyxl") as writer:
    for field, (counts, pct) in crosstabs.items():
        sheet = slugify(field)[:31]
        counts_out = counts.copy()
        counts_out.columns = pd.MultiIndex.from_product([["Count"], counts_out.columns])
        pct_out = pct.round(1)
        pct_out.columns = pd.MultiIndex.from_product([["Row %"], pct_out.columns])
        combined = pd.concat([counts_out, pct_out], axis=1)
        combined.to_excel(writer, sheet_name=sheet)

    rate_rows = []
    for field, tbl in rate_tables.items():
        for acuity, row in tbl.iterrows():
            rate_rows.append({"Field": field, "Consult Acuity": acuity, "n": row["n"], "Rate %": round(row["rate"], 1)})
    pd.DataFrame(rate_rows).to_excel(writer, sheet_name="secondary_rates", index=False)

print(f"Wrote {len(HEATMAP_FIELDS)} heatmaps and {len(RATE_FIELDS)} rate charts to {OUT}/")
print(f"Wrote crosstab summary to {OUT}/consult_acuity_field_crosstabs.xlsx")
