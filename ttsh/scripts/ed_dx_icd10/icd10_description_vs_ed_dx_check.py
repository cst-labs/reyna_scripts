import os
import re
from difflib import SequenceMatcher

import pandas as pd
from openpyxl.utils import get_column_letter

from icd10_scan_priority_distribution import lookup_icd_title


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDERS_PATH = os.path.join(BASE, "data", "nni_cleaned_orders.csv")
ALLPT_PATH = os.path.join(BASE, "merged", "all_pt_cleaned.xlsx")
OUT_PATH = os.path.join(BASE, "analysis", "icd10_description_vs_ed_dx_check.xlsx")

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "in",
    "of",
    "or",
    "other",
    "the",
    "to",
    "unspecified",
    "with",
    "without",
}

SYNONYMS = {
    "ams": "altered mental status",
    "cva": "stroke",
    "giddiness": "dizziness",
    "haemorrhage": "hemorrhage",
    "haemorrhoids": "hemorrhoids",
    "hi": "head injury",
    "ich": "intracerebral hemorrhage",
    "loc": "loss consciousness",
    "rta": "road traffic accident",
    "sah": "subarachnoid hemorrhage",
}


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize(value):
    text = clean_text(value).lower()
    text = text.replace("&", " and ")
    for src, dst in SYNONYMS.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(value):
    return {tok for tok in normalize(value).split() if tok not in STOPWORDS and len(tok) > 1}


def score_pair(official, free_text):
    official_norm = normalize(official)
    free_norm = normalize(free_text)
    seq = SequenceMatcher(None, official_norm, free_norm).ratio()
    official_tokens = tokens(official)
    free_tokens = tokens(free_text)
    if not official_tokens and not free_tokens:
        overlap = 1.0
    elif not official_tokens or not free_tokens:
        overlap = 0.0
    else:
        overlap = len(official_tokens & free_tokens) / len(official_tokens | free_tokens)
    return round(seq, 3), round(overlap, 3)


def classify(row):
    seq = row["Text Similarity"]
    overlap = row["Token Overlap"]
    top_share = row["Top Primary ED Dx % of Code"]
    if seq >= 0.78 or overlap >= 0.5:
        return "Likely match"
    if top_share >= 80 and (seq >= 0.45 or overlap >= 0.25):
        return "Likely match - wording differs"
    if row["ICD Title Source"].endswith("parent"):
        return "Review - parent-code title"
    if seq >= 0.45 or overlap >= 0.2:
        return "Review - broad/partial wording"
    return "Review - weak text match"


def auto_width(ws):
    for col in ws.columns:
        max_len = max((len(str(cell.value)) for cell in col if cell.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 2, 70)
    ws.freeze_panes = "A2"


def top_values(series, n=5):
    counts = series.map(clean_text).value_counts().head(n)
    return "; ".join(f"{idx} ({val})" for idx, val in counts.items())


def main():
    orders = pd.read_csv(ORDERS_PATH)
    allpt = pd.read_excel(
        ALLPT_PATH,
        sheet_name="All Patients",
        usecols=["CSN", "ED Primary Dx ICD10", "Primary ED Dx", "ED Dx"],
    )
    merged = orders.merge(
        allpt[["CSN", "ED Primary Dx ICD10", "Primary ED Dx", "ED Dx"]],
        on="CSN",
        how="left",
        suffixes=("_order", "_allpt"),
        validate="many_to_one",
    )
    merged["ED Primary Dx ICD10"] = merged["ED Primary Dx ICD10"].map(clean_text)
    merged["Primary ED Dx_allpt"] = merged["Primary ED Dx_allpt"].map(clean_text)
    merged["ED Dx_allpt"] = merged["ED Dx_allpt"].map(clean_text)

    rows = []
    for code, sub in merged.groupby("ED Primary Dx ICD10"):
        title = lookup_icd_title(code)
        primary_counts = sub["Primary ED Dx_allpt"].value_counts()
        top_primary = primary_counts.index[0]
        top_n = int(primary_counts.iloc[0])
        seq, overlap = score_pair(title["Official Dx"], top_primary)
        row = {
            "ED Primary Dx ICD10": code,
            **title,
            "N Orders": len(sub),
            "Unique Primary ED Dx": sub["Primary ED Dx_allpt"].nunique(),
            "Top Primary ED Dx": top_primary,
            "Top Primary ED Dx N": top_n,
            "Top Primary ED Dx % of Code": round(100 * top_n / len(sub), 1),
            "Text Similarity": seq,
            "Token Overlap": overlap,
            "Top 5 Primary ED Dx": top_values(sub["Primary ED Dx_allpt"]),
            "Top 5 ED Dx": top_values(sub["ED Dx_allpt"]),
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("N Orders", ascending=False)
    summary["Match Review"] = summary.apply(classify, axis=1)

    status_summary = (
        summary.groupby("Match Review", dropna=False)
        .agg(Codes=("ED Primary Dx ICD10", "count"), Orders=("N Orders", "sum"))
        .reset_index()
        .sort_values("Orders", ascending=False)
    )
    status_summary["% Orders"] = (100 * status_summary["Orders"] / summary["N Orders"].sum()).round(1)

    review = summary[summary["Match Review"].str.startswith("Review")].copy()
    review = review.sort_values(["N Orders", "ED Primary Dx ICD10"], ascending=[False, True])

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        status_summary.to_excel(writer, sheet_name="summary", index=False)
        summary.to_excel(writer, sheet_name="code_level_check", index=False)
        review.to_excel(writer, sheet_name="review_flags", index=False)
        for ws in writer.book.worksheets:
            auto_width(ws)

    print(f"Wrote {OUT_PATH}")
    print(status_summary.to_string(index=False))
    print("\nTop review flags by order volume:")
    cols = [
        "ED Primary Dx ICD10",
        "Official Dx",
        "N Orders",
        "Top Primary ED Dx",
        "Top Primary ED Dx % of Code",
        "Match Review",
        "ICD Title Source",
    ]
    print(review[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
