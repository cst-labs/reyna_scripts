import os
import re

import pandas as pd
import simple_icd_10 as who_icd10
import simple_icd_10_cm as icd10cm
from openpyxl.utils import get_column_letter


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDERS_PATH = os.path.join(BASE, "data", "nni_cleaned_orders.csv")
ALLPT_PATH = os.path.join(BASE, "merged", "all_pt_cleaned.xlsx")
OUT_PATH = os.path.join(BASE, "analysis", "icd10_scan_priority_distribution.xlsx")

PRIORITY_ORDER = ["P1", "P2", "P3", "Unknown"]
TOP_N = 50
MANUAL_ICD10_TITLES = {
    "I84.3": {
        "Official Dx": "External thrombosed haemorrhoids",
        "ICD Title Source": "Manual ICD-10 legacy title",
        "Matched ICD Code": "I84.3",
    }
}


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def auto_width(ws):
    for col in ws.columns:
        max_len = max((len(str(cell.value)) for cell in col if cell.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 2, 60)
    ws.freeze_panes = "A2"


def parent_candidates(code):
    candidates = []
    if "." in code:
        stem, tail = code.split(".", 1)
        for end in range(len(tail) - 1, -1, -1):
            candidate = stem + (f".{tail[:end]}" if end else "")
            candidates.append(candidate)
    raw = code.replace(".", "")
    if len(raw) > 3:
        for end in range(len(raw) - 1, 2, -1):
            candidate = raw[:3] + (f".{raw[3:end]}" if end > 3 else "")
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def lookup_icd_title(code):
    if code in MANUAL_ICD10_TITLES:
        return MANUAL_ICD10_TITLES[code]
    if who_icd10.is_valid_item(code):
        return {
            "Official Dx": who_icd10.get_description(code),
            "ICD Title Source": "WHO ICD-10 exact",
            "Matched ICD Code": code,
        }
    if icd10cm.is_valid_item(code):
        return {
            "Official Dx": icd10cm.get_description(code),
            "ICD Title Source": "ICD-10-CM exact",
            "Matched ICD Code": code,
        }
    for parent in parent_candidates(code):
        if who_icd10.is_valid_item(parent):
            return {
                "Official Dx": who_icd10.get_description(parent),
                "ICD Title Source": "WHO ICD-10 parent",
                "Matched ICD Code": parent,
            }
    for parent in parent_candidates(code):
        if icd10cm.is_valid_item(parent):
            return {
                "Official Dx": icd10cm.get_description(parent),
                "ICD Title Source": "ICD-10-CM parent",
                "Matched ICD Code": parent,
            }
    return {
        "Official Dx": "",
        "ICD Title Source": "Unmapped",
        "Matched ICD Code": "",
    }


def build_distribution(df):
    code_map = (
        pd.DataFrame(
            [
                {"ED Primary Dx ICD10": code, **lookup_icd_title(code)}
                for code in sorted(df["ED Primary Dx ICD10"].unique())
            ]
        )
        .sort_values("ED Primary Dx ICD10")
        .reset_index(drop=True)
    )
    counts = (
        df.groupby(["ED Primary Dx ICD10", "Scan Priority"], observed=False)
        .size()
        .rename("N")
        .reset_index()
    )
    code_totals = counts.groupby("ED Primary Dx ICD10")["N"].transform("sum")
    priority_totals = counts.groupby("Scan Priority", observed=False)["N"].transform("sum")
    counts["% Within ICD10 Code"] = (100 * counts["N"] / code_totals).round(1)
    counts["% Within Scan Priority"] = (100 * counts["N"] / priority_totals).round(1)
    counts = counts.merge(code_map, on="ED Primary Dx ICD10", how="left")
    counts = counts.sort_values(["ED Primary Dx ICD10", "Scan Priority"]).reset_index(drop=True)

    n_wide = (
        counts.pivot_table(
            index="ED Primary Dx ICD10",
            columns="Scan Priority",
            values="N",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(columns=PRIORITY_ORDER, fill_value=0)
        .astype(int)
    )
    pct_code_wide = n_wide.div(n_wide.sum(axis=1), axis=0).mul(100).round(1)
    pct_code_wide.columns = [f"{col} % Within ICD10 Code" for col in pct_code_wide.columns]

    priority_denoms = n_wide.sum(axis=0)
    pct_priority_wide = n_wide.div(priority_denoms, axis=1).mul(100).round(1)
    pct_priority_wide.columns = [f"{col} % Within Scan Priority" for col in pct_priority_wide.columns]

    n_wide.columns = [f"{col} N" for col in n_wide.columns]
    wide = pd.concat([n_wide, pct_code_wide, pct_priority_wide], axis=1)
    wide["Total N"] = n_wide.sum(axis=1)
    wide = wide.reset_index().merge(code_map, on="ED Primary Dx ICD10", how="left")
    front_cols = ["ED Primary Dx ICD10", "Official Dx", "ICD Title Source", "Matched ICD Code", "Total N"]
    wide = wide[front_cols + [col for col in wide.columns if col not in front_cols]]
    return counts, wide.sort_values("Total N", ascending=False).reset_index(drop=True), code_map


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    orders = pd.read_csv(ORDERS_PATH)
    allpt = pd.read_excel(
        ALLPT_PATH,
        sheet_name="All Patients",
        usecols=["CSN", "ED Primary Dx ICD10", "Primary ED Dx", "ED Dx"],
    )

    allpt_dupes = int(allpt.duplicated("CSN").sum())
    merged = orders.merge(
        allpt[["CSN", "ED Primary Dx ICD10"]],
        on="CSN",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    merged["ED Primary Dx ICD10"] = merged["ED Primary Dx ICD10"].map(clean_text)
    merged["Scan Priority"] = merged["Scan Priority"].map(clean_text)
    merged.loc[~merged["Scan Priority"].isin(PRIORITY_ORDER), "Scan Priority"] = "Unknown"
    merged["Scan Priority"] = pd.Categorical(
        merged["Scan Priority"], categories=PRIORITY_ORDER, ordered=True
    )

    missing_icd10 = merged["ED Primary Dx ICD10"].eq("")
    audit = pd.DataFrame(
        [
            {"Metric": "NNI order rows", "Value": len(orders)},
            {"Metric": "Unique NNI CSNs", "Value": orders["CSN"].nunique()},
            {"Metric": "All Pt rows", "Value": len(allpt)},
            {"Metric": "Unique All Pt CSNs", "Value": allpt["CSN"].nunique()},
            {"Metric": "Duplicate All Pt CSNs", "Value": allpt_dupes},
            {"Metric": "NNI orders matched to All Pt", "Value": int((merged["_merge"] == "both").sum())},
            {"Metric": "NNI orders not matched to All Pt", "Value": int((merged["_merge"] == "left_only").sum())},
            {"Metric": "NNI orders missing ICD10 after join", "Value": int(missing_icd10.sum())},
            {"Metric": "Distinct ICD10 codes in NNI orders", "Value": merged.loc[~missing_icd10, "ED Primary Dx ICD10"].nunique()},
        ]
    )

    priority_counts = (
        merged["Scan Priority"]
        .value_counts(dropna=False)
        .reindex(PRIORITY_ORDER, fill_value=0)
        .rename_axis("Scan Priority")
        .reset_index(name="N")
    )
    priority_counts["% of Orders"] = (100 * priority_counts["N"] / len(merged)).round(1)

    usable = merged.loc[~missing_icd10].copy()
    long, wide, code_map = build_distribution(usable)
    top_by_priority = (
        long.sort_values(["Scan Priority", "N"], ascending=[True, False])
        .assign(Rank=lambda x: x.groupby("Scan Priority", observed=False).cumcount() + 1)
    )
    top_by_priority = top_by_priority[top_by_priority["Rank"] <= TOP_N][
        [
            "Scan Priority",
            "Rank",
            "ED Primary Dx ICD10",
            "Official Dx",
            "ICD Title Source",
            "Matched ICD Code",
            "N",
            "% Within Scan Priority",
            "% Within ICD10 Code",
        ]
    ]

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        audit.to_excel(writer, sheet_name="audit", index=False)
        priority_counts.to_excel(writer, sheet_name="priority_counts", index=False)
        code_map.to_excel(writer, sheet_name="icd10_code_map", index=False)
        wide.to_excel(writer, sheet_name="icd10_wide", index=False)
        top_by_priority.to_excel(writer, sheet_name="top_icd10_by_priority", index=False)
        long.to_excel(writer, sheet_name="icd10_long", index=False)
        for ws in writer.book.worksheets:
            auto_width(ws)

    print(f"Read {len(orders):,} NNI order rows from {ORDERS_PATH}")
    print(f"All Pt CSN duplicates: {allpt_dupes:,}")
    print(f"Orders matched to All Pt: {(merged['_merge'] == 'both').sum():,} / {len(merged):,}")
    print(f"Orders missing ICD10 after join: {missing_icd10.sum():,} / {len(merged):,}")
    print(f"Distinct ICD10 codes in NNI orders: {usable['ED Primary Dx ICD10'].nunique():,}")
    print("ICD title mapping coverage:")
    print(code_map["ICD Title Source"].value_counts().to_string())
    print(f"Wrote {OUT_PATH}")

    print("\nScan priority distribution:")
    print(priority_counts.to_string(index=False))

    print("\nTop ICD10 codes by scan priority:")
    for priority in PRIORITY_ORDER:
        sub = top_by_priority[top_by_priority["Scan Priority"] == priority].head(10)
        print(f"\n{priority}")
        print(sub[["ED Primary Dx ICD10", "Official Dx", "N", "% Within Scan Priority", "% Within ICD10 Code"]].to_string(index=False))


if __name__ == "__main__":
    main()
