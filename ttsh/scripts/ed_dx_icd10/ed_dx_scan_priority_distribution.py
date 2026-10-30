import os
import re

import pandas as pd
from openpyxl.utils import get_column_letter


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE, "data", "nni_cleaned_orders.csv")
OUT_PATH = os.path.join(BASE, "analysis", "ed_dx_scan_priority_distribution.xlsx")

PRIORITY_ORDER = ["P1", "P2", "P3", "Unknown"]
TOP_N = 50


def normalize_text(value):
    if pd.isna(value):
        return "Unknown"
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else "Unknown"


def add_priority_share(table):
    priority_totals = table.groupby("Scan Priority")["N"].transform("sum")
    table["% Within Scan Priority"] = (100 * table["N"] / priority_totals).round(1)
    dx_totals = table.groupby(table.columns[0])["N"].transform("sum")
    table["% of Diagnosis in Priority"] = (100 * table["N"] / dx_totals).round(1)
    return table


def crosstab_distribution(df, dx_col):
    work = df[["Scan Priority", dx_col]].copy()
    work[dx_col] = work[dx_col].map(normalize_text)
    counts = (
        work.groupby([dx_col, "Scan Priority"], dropna=False)
        .size()
        .rename("N")
        .reset_index()
    )
    counts["Scan Priority"] = pd.Categorical(
        counts["Scan Priority"], categories=PRIORITY_ORDER, ordered=True
    )
    counts = counts.sort_values([dx_col, "Scan Priority"]).reset_index(drop=True)
    return add_priority_share(counts)


def wide_distribution(long_table, dx_col):
    wide_n = (
        long_table.pivot_table(
            index=dx_col,
            columns="Scan Priority",
            values="N",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(columns=PRIORITY_ORDER, fill_value=0)
        .astype(int)
    )
    wide_pct = wide_n.div(wide_n.sum(axis=1), axis=0).mul(100).round(1)
    wide_pct.columns = [f"{col} %" for col in wide_pct.columns]
    wide = pd.concat([wide_n, wide_pct], axis=1)
    wide["Total"] = wide_n.sum(axis=1)
    return wide.sort_values("Total", ascending=False).reset_index()


def top_by_priority(long_table, dx_col, n=TOP_N):
    ranked = long_table.copy()
    ranked["Rank Within Priority"] = (
        ranked.sort_values(["Scan Priority", "N"], ascending=[True, False])
        .groupby("Scan Priority", observed=False)
        .cumcount()
        + 1
    )
    return ranked[ranked["Rank Within Priority"] <= n].sort_values(
        ["Scan Priority", "Rank Within Priority"]
    )


def explode_ed_dx(df):
    rows = []
    for _, row in df[["Scan Priority", "ED Dx"]].iterrows():
        parts = [normalize_text(part) for part in str(row["ED Dx"]).split(";")]
        for dx in dict.fromkeys(parts):
            rows.append({"Scan Priority": row["Scan Priority"], "ED Dx Mention": dx})
    return pd.DataFrame(rows)


def auto_width(ws):
    for col in ws.columns:
        max_len = max((len(str(cell.value)) for cell in col if cell.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 2, 55)
    ws.freeze_panes = "A2"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["Scan Priority"] = df["Scan Priority"].map(normalize_text)
    df.loc[~df["Scan Priority"].isin(PRIORITY_ORDER), "Scan Priority"] = "Unknown"
    df["Scan Priority"] = pd.Categorical(
        df["Scan Priority"], categories=PRIORITY_ORDER, ordered=True
    )

    priority_counts = (
        df["Scan Priority"]
        .value_counts(dropna=False)
        .reindex(PRIORITY_ORDER, fill_value=0)
        .rename_axis("Scan Priority")
        .reset_index(name="N")
    )
    priority_counts["% of Orders"] = (100 * priority_counts["N"] / len(df)).round(1)

    primary_long = crosstab_distribution(df, "Primary ED Dx")
    primary_wide = wide_distribution(primary_long, "Primary ED Dx")
    primary_top = top_by_priority(primary_long, "Primary ED Dx")

    ed_mentions = explode_ed_dx(df)
    ed_long = crosstab_distribution(ed_mentions, "ED Dx Mention")
    ed_wide = wide_distribution(ed_long, "ED Dx Mention")
    ed_top = top_by_priority(ed_long, "ED Dx Mention")

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        priority_counts.to_excel(writer, sheet_name="priority_counts", index=False)
        primary_wide.to_excel(writer, sheet_name="primary_dx_wide", index=False)
        primary_top.to_excel(writer, sheet_name="primary_dx_top_by_prio", index=False)
        primary_long.to_excel(writer, sheet_name="primary_dx_long", index=False)
        ed_wide.to_excel(writer, sheet_name="ed_dx_mentions_wide", index=False)
        ed_top.to_excel(writer, sheet_name="ed_dx_mentions_top_by_prio", index=False)
        ed_long.to_excel(writer, sheet_name="ed_dx_mentions_long", index=False)
        for ws in writer.book.worksheets:
            auto_width(ws)

    print(f"Read {len(df):,} NNI orders from {DATA_PATH}")
    print(f"Wrote {OUT_PATH}")
    print("\nScan priority distribution:")
    print(priority_counts.to_string(index=False))
    print("\nTop primary ED diagnoses by scan priority:")
    for priority in PRIORITY_ORDER:
        sub = primary_top[primary_top["Scan Priority"] == priority].head(10)
        print(f"\n{priority}")
        print(sub[["Primary ED Dx", "N", "% Within Scan Priority", "% of Diagnosis in Priority"]].to_string(index=False))
    print("\nTop ED Dx mentions by scan priority:")
    for priority in PRIORITY_ORDER:
        sub = ed_top[ed_top["Scan Priority"] == priority].head(10)
        print(f"\n{priority}")
        print(sub[["ED Dx Mention", "N", "% Within Scan Priority", "% of Diagnosis in Priority"]].to_string(index=False))


if __name__ == "__main__":
    main()
