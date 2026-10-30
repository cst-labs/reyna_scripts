import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import (
    PatternFill, Font, Alignment, Border, Side, GradientFill
)
from openpyxl.utils import get_column_letter

# ── Load data ──────────────────────────────────────────────────────────────────
df_allpt = pd.read_excel("merged/all_pt_combined.xlsx",
                          usecols=["Consult Acuity", "Triage Acuity"])
df_ct_raw = pd.read_excel("merged/ct_orders_combined.xlsx",
                           usecols=["CSN", "Acuity Abbr", "Order Description",
                                    "Procedure Category",
                                    "Exam Start Date", "Exam Start Time",
                                    "Exam End Date", "Exam End Time"])

total_allpt = len(df_allpt)

# Deduplicate CT orders to one row per unique patient for acuity distribution.
# All patients have consistent acuity across orders, so first() is safe.
df_ct = (df_ct_raw.groupby("CSN")["Acuity Abbr"]
                  .first()
                  .reset_index())
total_ct = len(df_ct)

# All imaging scan types — classify by Procedure Category (authoritative system field)
PROC_CAT_MAP = {
    "IMG CT ORDERABLES":                  "CT",
    "IMG DIAGNOSTIC IMAGING ORDERABLES":  "XR",
    "IMG MRI ORDERABLES":                 "MRI",
    "IMG US ORDERABLES":                  "US",
    "IMG IR ORDERABLES":                  "IR",
    "IMG DXA ORDERABLES":                 "DXA",
    "IMG NM ORDERABLES":                  "NM",
    "IMG MAMMOGRAPHY ORDERABLES":         "Mammography",
    "IMG FLUOROSCOPY ORDERABLES":         "Fluoroscopy",
    "CV VASCULAR ORDERABLES":             "Vascular",
}

df_ct_raw["Modality"] = df_ct_raw["Procedure Category"].map(PROC_CAT_MAP).fillna("Other")

# Parse exam times for all orders
df_ct_raw["start"] = pd.to_datetime(
    df_ct_raw["Exam Start Date"].astype(str) + " " + df_ct_raw["Exam Start Time"].astype(str),
    errors="coerce")
df_ct_raw["end"] = pd.to_datetime(
    df_ct_raw["Exam End Date"].astype(str) + " " + df_ct_raw["Exam End Time"].astype(str),
    errors="coerce")
df_ct_raw["exam_duration_mins"] = (df_ct_raw["end"] - df_ct_raw["start"]).dt.total_seconds() / 60

# CT-only subset (used by Scan Timing sheet)
ct_orders = df_ct_raw[df_ct_raw["Modality"] == "CT"].copy()

total_all_orders = len(df_ct_raw)
total_ct_orders  = len(ct_orders)   # kept for subtitle compatibility

# All imaging scan types table (replaces CT-only)
MODALITY_ORDER = ["CT", "XR", "MRI", "US", "IR", "DXA", "NM", "Mammography", "Fluoroscopy", "Vascular", "Other"]
all_scan_types = (df_ct_raw.groupby(["Modality", "Order Description"])
                  .agg(
                      n_orders   =("CSN", "count"),
                      n_patients =("CSN", "nunique"),
                      med_dur    =("exam_duration_mins", lambda x: round(x.dropna().median(), 1) if x.dropna().shape[0] > 0 else None),
                      mean_dur   =("exam_duration_mins", lambda x: round(x.dropna().mean(),   1) if x.dropna().shape[0] > 0 else None),
                  )
                  .reset_index()
                  .sort_values(
                      ["Modality", "n_orders"],
                      key=lambda col: col.map(MODALITY_ORDER.index) if col.name == "Modality" else col,
                      ascending=[True, False])
                  .reset_index(drop=True))
all_scan_types["pct_of_all"] = all_scan_types["n_orders"] / total_all_orders * 100

# For flagging: CT-only results that changed
ct_scan_types = all_scan_types[all_scan_types["Modality"] == "CT"].copy()
ct_scan_types["pct"] = ct_scan_types["n_orders"] / total_ct_orders * 100


def build_distribution(series, total, label_order=None):
    """
    Returns a list of dicts:
      {"value", "count", "pct", "display"}
    Missing (NaN) is included as a row.
    Rows are ordered by label_order if provided, then alphabetically,
    with Missing always last.
    """
    vc = series.value_counts(dropna=True)
    missing_count = series.isna().sum()

    rows = []
    for val, cnt in vc.items():
        rows.append({"value": val, "count": cnt, "pct": cnt / total * 100})

    # sort
    if label_order:
        def sort_key(r):
            try:
                return label_order.index(r["value"])
            except ValueError:
                return len(label_order) + 1
        rows.sort(key=sort_key)
    else:
        rows.sort(key=lambda r: str(r["value"]))

    # append missing row
    rows.append({"value": "Missing", "count": missing_count,
                 "pct": missing_count / total * 100})

    for r in rows:
        r["display"] = f"{r['count']:,} ({r['pct']:.1f}%)"

    return rows


# Normalise Consult Acuity: strip "(P) " prefix so "(P) P2" → "P2"
def normalise_consult(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    if s.startswith("(P) "):
        s = s[4:]
    return s

df_allpt["Consult Acuity Norm"] = df_allpt["Consult Acuity"].apply(normalise_consult)

# Normalise Triage Acuity: keep only first token (e.g. "P2 (req trolley bed)" → "P2")
def normalise_triage(val):
    if pd.isna(val):
        return val
    return str(val).strip().split()[0]

df_allpt["Triage Acuity Norm"] = df_allpt["Triage Acuity"].apply(normalise_triage)

ACUITY_ORDER = ["P1", "P2", "P3", "P4"]

consult_rows  = build_distribution(df_allpt["Consult Acuity Norm"], total_allpt, ACUITY_ORDER)
triage_rows   = build_distribution(df_allpt["Triage Acuity Norm"],  total_allpt, ACUITY_ORDER)
ct_acuity_rows = build_distribution(df_ct["Acuity Abbr"],            total_ct,    ACUITY_ORDER)

# ── Styles ─────────────────────────────────────────────────────────────────────
CLR_HEADER_DARK  = "1F3864"   # dark navy
CLR_HEADER_MID   = "2E75B6"   # mid blue
CLR_SECTION_BG   = "D6E4F0"   # light blue section title
CLR_ROW_ALT      = "EBF3FB"   # alternating row
CLR_MISSING      = "FFF2CC"   # yellow for missing row
CLR_TOTAL_BG     = "D9E1F2"   # total row
CLR_P1           = "FCE4D6"   # soft red
CLR_P2           = "FDEBD0"   # soft orange
CLR_P3           = "E8F5E9"   # soft green
CLR_P4           = "EDE7F6"   # soft purple

ACUITY_CLR = {"P1": CLR_P1, "P2": CLR_P2, "P3": CLR_P3, "P4": CLR_P4}

def fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def thin_border():
    s = Side(style="thin", color="BFBFBF")
    return Border(left=s, right=s, top=s, bottom=s)

def header_font(white=True):
    return Font(bold=True, color="FFFFFF" if white else "1F3864", size=11)

def normal_font(bold=False):
    return Font(bold=bold, size=11)

def center():
    return Alignment(horizontal="center", vertical="center")

def left():
    return Alignment(horizontal="left", vertical="center")


# ── Build workbook ─────────────────────────────────────────────────────────────
wb = Workbook()
ws = wb.active
ws.title = "Acuity Distribution"
ws.sheet_view.showGridLines = False

# column widths
ws.column_dimensions["A"].width = 32
ws.column_dimensions["B"].width = 22
ws.column_dimensions["C"].width = 22
ws.column_dimensions["D"].width = 4   # spacer
ws.column_dimensions["E"].width = 32
ws.column_dimensions["F"].width = 22
ws.column_dimensions["G"].width = 22
ws.column_dimensions["H"].width = 4   # spacer
ws.column_dimensions["I"].width = 32
ws.column_dimensions["J"].width = 22
ws.column_dimensions["K"].width = 22


def write_section(ws, start_row, start_col, title, subtitle, rows,
                  col_headers=("Acuity", "Count", "Percentage")):
    """Write one acuity table block. Returns the next free row."""
    r = start_row
    sc = start_col
    sc_lbl = get_column_letter(sc)
    sc_cnt = get_column_letter(sc + 1)
    sc_pct = get_column_letter(sc + 2)

    # ── Section title (merged across 3 cols) ──
    ws.merge_cells(f"{sc_lbl}{r}:{sc_pct}{r}")
    cell = ws[f"{sc_lbl}{r}"]
    cell.value = title
    cell.fill  = fill(CLR_HEADER_DARK)
    cell.font  = Font(bold=True, color="FFFFFF", size=13)
    cell.alignment = center()
    cell.border = thin_border()
    ws[f"{sc_cnt}{r}"].border = thin_border()
    ws[f"{sc_pct}{r}"].border = thin_border()
    r += 1

    # ── Subtitle row ──
    ws.merge_cells(f"{sc_lbl}{r}:{sc_pct}{r}")
    cell = ws[f"{sc_lbl}{r}"]
    cell.value = subtitle
    cell.fill  = fill(CLR_HEADER_MID)
    cell.font  = Font(italic=True, color="FFFFFF", size=10)
    cell.alignment = center()
    cell.border = thin_border()
    ws[f"{sc_cnt}{r}"].border = thin_border()
    ws[f"{sc_pct}{r}"].border = thin_border()
    r += 1

    # ── Column headers ──
    for col_offset, hdr in enumerate(col_headers):
        c = get_column_letter(sc + col_offset)
        ws[f"{c}{r}"].value = hdr
        ws[f"{c}{r}"].fill  = fill(CLR_SECTION_BG)
        ws[f"{c}{r}"].font  = Font(bold=True, color="1F3864", size=11)
        ws[f"{c}{r}"].alignment = center()
        ws[f"{c}{r}"].border = thin_border()
    r += 1

    # ── Data rows ──
    for i, row_data in enumerate(rows):
        val   = row_data["value"]
        count = row_data["count"]
        pct   = row_data["pct"]

        if val == "Missing":
            bg = CLR_MISSING
        elif val in ACUITY_CLR:
            bg = ACUITY_CLR[val]
        else:
            bg = CLR_ROW_ALT if i % 2 == 0 else "FFFFFF"

        ws[f"{sc_lbl}{r}"].value = val
        ws[f"{sc_lbl}{r}"].fill  = fill(bg)
        ws[f"{sc_lbl}{r}"].font  = normal_font(bold=(val == "Missing"))
        ws[f"{sc_lbl}{r}"].alignment = left()
        ws[f"{sc_lbl}{r}"].border = thin_border()

        ws[f"{sc_cnt}{r}"].value = count
        ws[f"{sc_cnt}{r}"].fill  = fill(bg)
        ws[f"{sc_cnt}{r}"].font  = normal_font()
        ws[f"{sc_cnt}{r}"].alignment = center()
        ws[f"{sc_cnt}{r}"].border = thin_border()

        ws[f"{sc_pct}{r}"].value = f"{pct:.1f}%"
        ws[f"{sc_pct}{r}"].fill  = fill(bg)
        ws[f"{sc_pct}{r}"].font  = normal_font()
        ws[f"{sc_pct}{r}"].alignment = center()
        ws[f"{sc_pct}{r}"].border = thin_border()

        r += 1

    # ── Total row ──
    non_missing = [x for x in rows if x["value"] != "Missing"]
    tot = sum(x["count"] for x in rows)  # includes missing
    valid_tot = sum(x["count"] for x in non_missing)

    ws.merge_cells(f"{sc_lbl}{r}:{sc_lbl}{r}")
    ws[f"{sc_lbl}{r}"].value = "Total"
    ws[f"{sc_lbl}{r}"].fill  = fill(CLR_TOTAL_BG)
    ws[f"{sc_lbl}{r}"].font  = Font(bold=True, size=11)
    ws[f"{sc_lbl}{r}"].alignment = left()
    ws[f"{sc_lbl}{r}"].border = thin_border()

    ws[f"{sc_cnt}{r}"].value = tot
    ws[f"{sc_cnt}{r}"].fill  = fill(CLR_TOTAL_BG)
    ws[f"{sc_cnt}{r}"].font  = Font(bold=True, size=11)
    ws[f"{sc_cnt}{r}"].alignment = center()
    ws[f"{sc_cnt}{r}"].border = thin_border()

    ws[f"{sc_pct}{r}"].value = "100.0%"
    ws[f"{sc_pct}{r}"].fill  = fill(CLR_TOTAL_BG)
    ws[f"{sc_pct}{r}"].font  = Font(bold=True, size=11)
    ws[f"{sc_pct}{r}"].alignment = center()
    ws[f"{sc_pct}{r}"].border = thin_border()

    return r + 2   # leave a gap


# ── Master page title ──────────────────────────────────────────────────────────
ws.row_dimensions[1].height = 30
ws.merge_cells("A1:K1")
title_cell = ws["A1"]
title_cell.value = "Acuity Distribution Analysis"
title_cell.fill  = fill(CLR_HEADER_DARK)
title_cell.font  = Font(bold=True, color="FFFFFF", size=16)
title_cell.alignment = center()
title_cell.border = thin_border()

ws.row_dimensions[2].height = 18
ws.merge_cells("A2:K2")
sub_cell = ws["A2"]
sub_cell.value = (f"All Patients (n={total_allpt:,})  •  CT: unique patients (n={total_ct:,})  "
                  f"•  Note: normalised values shown (e.g. '(P) P2' → 'P2', "
                  f"'P2 (req trolley bed)' → 'P2')  •  CT acuity deduplicated to one row per patient")
sub_cell.fill  = fill(CLR_HEADER_MID)
sub_cell.font  = Font(italic=True, color="FFFFFF", size=10)
sub_cell.alignment = center()
sub_cell.border = thin_border()

START_ROW = 4

# ── Write three tables side by side ──────────────────────────────────────────
write_section(
    ws, START_ROW, 1,
    "Consult Acuity",
    f"All Patients  |  n = {total_allpt:,}",
    consult_rows,
    col_headers=("Consult Acuity", "Count", "Percentage")
)

write_section(
    ws, START_ROW, 5,
    "Triage Acuity",
    f"All Patients  |  n = {total_allpt:,}",
    triage_rows,
    col_headers=("Triage Acuity", "Count", "Percentage")
)

write_section(
    ws, START_ROW, 9,
    "CT Acuity (Unique Patients)",
    f"Unique CT Patients  |  n = {total_ct:,}",
    ct_acuity_rows,
    col_headers=("Acuity Abbr", "Count", "Percentage")
)

# ── CT Scan Types table (below acuity tables) ─────────────────────────────────
SCAN_START_ROW = 14  # leaves a blank row gap after the acuity tables

# Section title
ws.merge_cells(f"A{SCAN_START_ROW}:G{SCAN_START_ROW}")
t = ws[f"A{SCAN_START_ROW}"]
t.value = "All Imaging Scan Types Distribution  ⚠ Updated: now includes XR, MRI, US and Other (previously CT only)"
t.fill  = fill(CLR_HEADER_DARK)
t.font  = Font(bold=True, color="FFFFFF", size=13)
t.alignment = center()
t.border = thin_border()
for col in ["B","C","D","E","F","G","H"]:
    ws[f"{col}{SCAN_START_ROW}"].border = thin_border()

# Subtitle — now covers all imaging
ws.merge_cells(f"A{SCAN_START_ROW+1}:H{SCAN_START_ROW+1}")
s = ws[f"A{SCAN_START_ROW+1}"]
s.value = (f"All imaging orders (CT + XR + MRI + US + Other)  |  "
           f"n = {total_all_orders:,} orders  •  {len(all_scan_types)} scan types  •  "
           f"⚠ Previously CT only ({total_ct_orders:,} orders, {len(ct_scan_types)} types) — "
           f"now expanded to all modalities  •  Duration = Exam End − Exam Start")
s.fill  = fill(CLR_HEADER_MID)
s.font  = Font(italic=True, color="FFFFFF", size=10)
s.alignment = center()
s.border = thin_border()
for col in ["B","C","D","E","F","G","H"]:
    ws[f"{col}{SCAN_START_ROW+1}"].border = thin_border()

# Modality colour map
MOD_CLR = {
    "CT":          "D6E4F0",   # light blue
    "XR":          "E2EFDA",   # light green
    "MRI":         "EDE7F6",   # light purple
    "US":          "FFF9C4",   # light yellow
    "IR":          "FCE4D6",   # light orange
    "DXA":         "F0F0F0",   # light grey
    "NM":          "E8F4F8",   # light teal
    "Mammography": "FDE8F0",   # light pink
    "Fluoroscopy": "F5F0E8",   # light tan
    "Vascular":    "E8F0E8",   # light sage
    "Other":       "F5F5F5",   # grey fallback
}

# Column headers — now 7 cols (added Modality + updated % label)
scan_hdrs = ["Modality", "Scan Type", "Orders (n)", "% of All Imaging",
             "Unique Patients", "Median Duration (mins)", "Mean Duration (mins)"]
for i, hdr in enumerate(scan_hdrs):
    col = get_column_letter(i + 1)
    c = ws[f"{col}{SCAN_START_ROW+2}"]
    c.value = hdr
    c.fill  = fill(CLR_SECTION_BG)
    c.font  = Font(bold=True, color="1F3864", size=11)
    c.alignment = center()
    c.border = thin_border()

# Data rows — grouped by modality with subtotal rows
ridx_global = 0
prev_mod = None
for _, row_data in all_scan_types.iterrows():
    mod = row_data["Modality"]

    # Modality subtotal header when group changes
    if mod != prev_mod:
        if prev_mod is not None:
            # subtotal for previous modality
            sub = all_scan_types[all_scan_types["Modality"] == prev_mod]
            st_r = SCAN_START_ROW + 3 + ridx_global
            ws.merge_cells(f"A{st_r}:B{st_r}")
            ws[f"A{st_r}"].value = f"{prev_mod} Subtotal"
            ws[f"A{st_r}"].fill  = fill(MOD_CLR.get(prev_mod, "F5F5F5"))
            ws[f"A{st_r}"].font  = Font(bold=True, size=11)
            ws[f"A{st_r}"].alignment = left()
            ws[f"A{st_r}"].border = thin_border()
            ws[f"B{st_r}"].border = thin_border()
            for col, val in [("C", int(sub["n_orders"].sum())),
                              ("D", f"{sub['n_orders'].sum()/total_all_orders*100:.1f}%"),
                              ("E", int(sub["n_patients"].max())),
                              ("F", ""), ("G", "")]:
                ws[f"{col}{st_r}"].value = val
                ws[f"{col}{st_r}"].fill  = fill(MOD_CLR.get(prev_mod, "F5F5F5"))
                ws[f"{col}{st_r}"].font  = Font(bold=True, size=11)
                ws[f"{col}{st_r}"].alignment = center()
                ws[f"{col}{st_r}"].border = thin_border()
            ridx_global += 1
        prev_mod = mod

    r = SCAN_START_ROW + 3 + ridx_global
    bg = MOD_CLR.get(mod, "F5F5F5") if ridx_global % 2 == 0 else "FFFFFF"

    cells = [
        ("A", mod,                                                               "center"),
        ("B", row_data["Order Description"],                                     "left"),
        ("C", int(row_data["n_orders"]),                                         "center"),
        ("D", f"{row_data['pct_of_all']:.1f}%",                                  "center"),
        ("E", int(row_data["n_patients"]),                                        "center"),
        ("F", row_data["med_dur"] if pd.notna(row_data["med_dur"]) else "—",     "center"),
        ("G", row_data["mean_dur"] if pd.notna(row_data["mean_dur"]) else "—",   "center"),
    ]
    for col, val, align in cells:
        c = ws[f"{col}{r}"]
        c.value = val
        c.fill  = fill(bg)
        c.font  = normal_font()
        c.alignment = left() if align == "left" else center()
        c.border = thin_border()
    ridx_global += 1

# Last modality subtotal
if prev_mod:
    sub = all_scan_types[all_scan_types["Modality"] == prev_mod]
    st_r = SCAN_START_ROW + 3 + ridx_global
    ws.merge_cells(f"A{st_r}:B{st_r}")
    ws[f"A{st_r}"].value = f"{prev_mod} Subtotal"
    ws[f"A{st_r}"].fill  = fill(MOD_CLR.get(prev_mod, "F5F5F5"))
    ws[f"A{st_r}"].font  = Font(bold=True, size=11)
    ws[f"A{st_r}"].alignment = left()
    ws[f"A{st_r}"].border = thin_border()
    ws[f"B{st_r}"].border = thin_border()
    for col, val in [("C", int(sub["n_orders"].sum())),
                     ("D", f"{sub['n_orders'].sum()/total_all_orders*100:.1f}%"),
                     ("E", int(sub["n_patients"].max())),
                     ("F", ""), ("G", "")]:
        ws[f"{col}{st_r}"].value = val
        ws[f"{col}{st_r}"].fill  = fill(MOD_CLR.get(prev_mod, "F5F5F5"))
        ws[f"{col}{st_r}"].font  = Font(bold=True, size=11)
        ws[f"{col}{st_r}"].alignment = center()
        ws[f"{col}{st_r}"].border = thin_border()
    ridx_global += 1

# Grand total row
tot_r = SCAN_START_ROW + 3 + ridx_global
ws.merge_cells(f"A{tot_r}:B{tot_r}")
ws[f"A{tot_r}"].value = "Grand Total (All Imaging)"
ws[f"A{tot_r}"].fill  = fill(CLR_TOTAL_BG)
ws[f"A{tot_r}"].font  = Font(bold=True, size=11)
ws[f"A{tot_r}"].alignment = left()
ws[f"A{tot_r}"].border = thin_border()
ws[f"B{tot_r}"].border = thin_border()
for col, val in [
    ("C", total_all_orders),
    ("D", "100.0%"),
    ("E", df_ct_raw["CSN"].nunique()),
    ("F", round(df_ct_raw["exam_duration_mins"].dropna().median(), 1)),
    ("G", round(df_ct_raw["exam_duration_mins"].dropna().mean(),   1)),
]:
    c = ws[f"{col}{tot_r}"]
    c.value = val
    c.fill  = fill(CLR_TOTAL_BG)
    c.font  = Font(bold=True, size=11)
    c.alignment = center()
    c.border = thin_border()

# Column widths
ws.column_dimensions["A"].width = 10   # Modality
ws.column_dimensions["B"].width = 46   # Scan Type
ws.column_dimensions["C"].width = 14   # Orders
ws.column_dimensions["D"].width = 20   # %
ws.column_dimensions["E"].width = 18   # Patients
ws.column_dimensions["F"].width = 24   # Median dur
ws.column_dimensions["G"].width = 22   # Mean dur

# ── Row heights for data area ──────────────────────────────────────────────────
for r in range(START_ROW, ws.max_row + 1):
    ws.row_dimensions[r].height = 20

# ── Freeze panes ──────────────────────────────────────────────────────────────
ws.freeze_panes = "A3"


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 2 – Scan Timing (concurrent vs sequential)
# ══════════════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("Scan Timing")
ws2.sheet_view.showGridLines = False

# ── Shared classification function ───────────────────────────────────────────
def classify_patient(grp):
    grp = grp.sort_values("start")
    valid = grp.dropna(subset=["start", "end"])
    if len(valid) < 2:
        return "missing"
    starts = valid["start"].tolist()
    ends   = valid["end"].tolist()
    if len(set(zip([s.isoformat() for s in starts], [e.isoformat() for e in ends]))) == 1:
        return "identical_window"
    has_overlap = any(starts[i+1] < ends[i] for i in range(len(valid)-1))
    has_seq     = any(starts[i+1] >= ends[i] for i in range(len(valid)-1))
    if has_overlap and has_seq: return "mixed"
    if has_overlap: return "overlapping"
    return "sequential"

def gap_stats_for(df_subset, pt_class_series):
    seq_csns = pt_class_series[pt_class_series.isin(["sequential", "mixed"])].index
    seq_df = df_subset[df_subset["CSN"].isin(seq_csns)].copy()
    all_gaps = []
    for _, grp in seq_df.groupby("CSN"):
        grp = grp.sort_values("start").dropna(subset=["start","end"])
        for i in range(len(grp)-1):
            gap = (grp.iloc[i+1]["start"] - grp.iloc[i]["end"]).total_seconds() / 60
            if grp.iloc[i+1]["start"] >= grp.iloc[i]["end"]:  # only real gaps
                all_gaps.append(gap)
    return pd.Series(all_gaps).dropna()

def span_by_n_orders(df_all, col="n_orders"):
    pt = df_all.groupby("CSN").agg(
        n_orders    =("CSN",   "count"),
        first_start =("start", "min"),
        last_end    =("end",   "max"),
    ).reset_index()
    pt["span_mins"] = (pt["last_end"] - pt["first_start"]).dt.total_seconds() / 60
    return (pt.groupby("n_orders")
              .agg(n_pts    =("CSN", "count"),
                   med_span =("span_mins", lambda x: round(x.dropna().median(),1)),
                   mean_span=("span_mins", lambda x: round(x.dropna().mean(),1)))
              .reset_index()), pt

# ── CT-only classification (existing) ────────────────────────────────────────
multi_ct  = ct_orders.groupby("CSN").filter(lambda x: len(x) > 1).copy()
multi_pts_ct = multi_ct["CSN"].nunique()
pt_class_ct  = multi_ct.groupby("CSN").apply(classify_patient, include_groups=False)
cc = pt_class_ct.value_counts()
ct_identical_n   = cc.get("identical_window", 0)
ct_overlapping_n = cc.get("overlapping",      0)
ct_mixed_n       = cc.get("mixed",            0)
ct_sequential_n  = cc.get("sequential",       0)
ct_missing_n     = cc.get("missing",          0)
ct_same_session  = ct_identical_n + ct_overlapping_n
ct_gaps          = gap_stats_for(multi_ct, pt_class_ct)
span_ct, pt_span_ct = span_by_n_orders(ct_orders)

# ── All-imaging classification (new) ─────────────────────────────────────────
multi_all    = df_ct_raw.groupby("CSN").filter(lambda x: len(x) > 1).copy()
multi_pts_all = multi_all["CSN"].nunique()
pt_class_all  = multi_all.groupby("CSN").apply(classify_patient, include_groups=False)
ca = pt_class_all.value_counts()
all_identical_n   = ca.get("identical_window", 0)
all_overlapping_n = ca.get("overlapping",      0)
all_mixed_n       = ca.get("mixed",            0)
all_sequential_n  = ca.get("sequential",       0)
all_missing_n     = ca.get("missing",          0)
all_same_session  = all_identical_n + all_overlapping_n
all_gaps          = gap_stats_for(multi_all, pt_class_all)
span_all, pt_span_all = span_by_n_orders(df_ct_raw)

# ── Sheet helpers ─────────────────────────────────────────────────────────────
def sh2_colhdr(ws, row, cols_vals):
    for col, val in cols_vals:
        c = ws[f"{col}{row}"]
        c.value = val; c.fill = fill(CLR_SECTION_BG)
        c.font  = Font(bold=True, color="1F3864", size=11)
        c.alignment = center(); c.border = thin_border()

def sh2_cell(ws, addr, value, bg, bold=False, align="center"):
    c = ws[addr]
    c.value = value; c.fill = fill(bg)
    c.font  = normal_font(bold=bold)
    c.alignment = left() if align == "left" else center()
    c.border = thin_border()

def sh2_section_title(ws, r, start_col, n_cols, value, bg=None):
    bg = bg or CLR_HEADER_DARK
    sc = get_column_letter(start_col)
    ec = get_column_letter(start_col + n_cols - 1)
    ws.merge_cells(f"{sc}{r}:{ec}{r}")
    c = ws[f"{sc}{r}"]
    c.value = value; c.fill = fill(bg)
    c.font  = Font(bold=True, color="FFFFFF", size=12)
    c.alignment = center(); c.border = thin_border()
    for ci in range(1, n_cols):
        ws[f"{get_column_letter(start_col+ci)}{r}"].border = thin_border()

def sh2_subtitle(ws, r, start_col, n_cols, value):
    sc = get_column_letter(start_col)
    ec = get_column_letter(start_col + n_cols - 1)
    ws.merge_cells(f"{sc}{r}:{ec}{r}")
    c = ws[f"{sc}{r}"]
    c.value = value; c.fill = fill(CLR_HEADER_MID)
    c.font  = Font(italic=True, color="FFFFFF", size=10)
    c.alignment = center(); c.border = thin_border()
    for ci in range(1, n_cols):
        ws[f"{get_column_letter(start_col+ci)}{r}"].border = thin_border()

# ── Page title ────────────────────────────────────────────────────────────────
ws2.merge_cells("A1:J1")
c = ws2["A1"]
c.value = "Imaging Session Timing — Concurrent vs Sequential  |  CT Only vs All Imaging"
c.fill  = fill(CLR_HEADER_DARK); c.font = Font(bold=True, color="FFFFFF", size=14)
c.alignment = center(); c.border = thin_border()
for col in "BCDEFGHIJ": ws2[f"{col}1"].border = thin_border()

ws2.merge_cells("A2:J2")
c = ws2["A2"]
c.value = ("Left: CT orders only (existing analysis). Right: all imaging (CT + XR + MRI + US + Other). "
           "⚠ Pattern flips when all imaging included — most patients become mixed/sequential "
           "because CT and XR typically happen at different times during the visit.")
c.fill  = fill(CLR_HEADER_MID); c.font = Font(italic=True, color="FFFFFF", size=10)
c.alignment = center(); c.border = thin_border()
for col in "BCDEFGHIJ": ws2[f"{col}2"].border = thin_border()

# Column layout: A-D = CT only, E = spacer, F-I = All imaging
# ── Table 1: Classification — side by side ────────────────────────────────────
r = 4

# Left header
sh2_section_title(ws2, r, 1, 4, f"CT Only — Patients with 2+ CT scans (n={multi_pts_ct:,})", bg="1F5C99")
# Right header
sh2_section_title(ws2, r, 6, 4, f"All Imaging — Patients with 2+ imaging orders (n={multi_pts_all:,})", bg="1F3864")
r += 1

sh2_colhdr(ws2, r, [("A","Category"), ("B","n"), ("C","%"), ("D","Interpretation"),
                     ("F","Category"), ("G","n"), ("H","%"), ("I","Interpretation")])
r += 1

CLR_IDENTICAL  = "E8F5E9"
CLR_OVERLAP    = CLR_P3
CLR_MIXED      = CLR_P2
CLR_SEQ        = CLR_P1
CLR_MISS       = CLR_MISSING

ct_rows = [
    ("Identical window",  ct_identical_n,   ct_identical_n/multi_pts_ct*100,   "All scans share exact same start & end — one session", CLR_IDENTICAL),
    ("Overlapping",       ct_overlapping_n, ct_overlapping_n/multi_pts_ct*100, "Near-simultaneous starts, windows overlap",            CLR_OVERLAP),
    ("Mixed",             ct_mixed_n,       ct_mixed_n/multi_pts_ct*100,       "Some concurrent, some follow-up later in visit",       CLR_MIXED),
    ("Sequential",        ct_sequential_n,  ct_sequential_n/multi_pts_ct*100,  "All scans separated by a gap",                         CLR_SEQ),
    ("Missing times",     ct_missing_n,     ct_missing_n/multi_pts_ct*100,     "Cannot classify",                                      CLR_MISS),
]
all_rows = [
    ("Identical window",  all_identical_n,   all_identical_n/multi_pts_all*100,   "All orders share exact same start & end", CLR_IDENTICAL),
    ("Overlapping",       all_overlapping_n, all_overlapping_n/multi_pts_all*100, "Near-simultaneous starts, windows overlap", CLR_OVERLAP),
    ("Mixed",             all_mixed_n,       all_mixed_n/multi_pts_all*100,       "Some concurrent, some at different times — e.g. CT at 10am, XR at 2pm", CLR_MIXED),
    ("Sequential",        all_sequential_n,  all_sequential_n/multi_pts_all*100,  "All orders separated by a gap",            CLR_SEQ),
    ("Missing times",     all_missing_n,     all_missing_n/multi_pts_all*100,     "Cannot classify",                          CLR_MISS),
]

for (lbl_ct, n_ct, pct_ct, interp_ct, bg_ct), (lbl_all, n_all, pct_all, interp_all, bg_all) in zip(ct_rows, all_rows):
    sh2_cell(ws2, f"A{r}", lbl_ct,            bg_ct,  align="left")
    sh2_cell(ws2, f"B{r}", n_ct,              bg_ct)
    sh2_cell(ws2, f"C{r}", f"{pct_ct:.1f}%",  bg_ct)
    sh2_cell(ws2, f"D{r}", interp_ct,         bg_ct,  align="left")
    sh2_cell(ws2, f"F{r}", lbl_all,           bg_all, align="left")
    sh2_cell(ws2, f"G{r}", n_all,             bg_all)
    sh2_cell(ws2, f"H{r}", f"{pct_all:.1f}%", bg_all)
    sh2_cell(ws2, f"I{r}", interp_all,        bg_all, align="left")
    r += 1

# Totals
for sc, n, multi in [("A", multi_pts_ct, True), ("F", multi_pts_all, True)]:
    sh2_cell(ws2, f"{sc}{r}", "Total",  CLR_TOTAL_BG, bold=True, align="left")
    sh2_cell(ws2, f"{get_column_letter(ord(sc)-64+1)}{r}", n, CLR_TOTAL_BG, bold=True)
    sh2_cell(ws2, f"{get_column_letter(ord(sc)-64+2)}{r}", "100.0%", CLR_TOTAL_BG, bold=True)
    sh2_cell(ws2, f"{get_column_letter(ord(sc)-64+3)}{r}", "", CLR_TOTAL_BG)
r += 2

# Key insight note spanning full width
ws2.merge_cells(f"A{r}:I{r}")
c = ws2[f"A{r}"]
c.value = (
    f"⚠ Key difference: CT-only shows {ct_same_session:,} ({ct_same_session/multi_pts_ct*100:.0f}%) patients as same-session "
    f"(identical/overlapping) because multiple CT scan types are run as one protocol. "
    f"When all imaging is included, the dominant pattern flips: {all_mixed_n:,} ({all_mixed_n/multi_pts_all*100:.0f}%) "
    f"are Mixed and {all_sequential_n:,} ({all_sequential_n/multi_pts_all*100:.0f}%) are Sequential — "
    f"because CT and XR are typically ordered for the same patient but performed at different times during the visit."
)
c.fill  = fill("FFF9C4"); c.font = Font(italic=True, color="1F3864", size=10)
c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
c.border = thin_border()
for col in "BCDEFGHI": ws2[f"{col}{r}"].border = thin_border()
ws2.row_dimensions[r].height = 50
r += 2

# ── Table 2: Gap stats — side by side ─────────────────────────────────────────
sh2_section_title(ws2, r, 1, 4,
    f"Table 2 — Inter-Imaging Wait Gap  |  CT Only (Sequential+Mixed, n={ct_sequential_n+ct_mixed_n:,})",
    bg="1F5C99")
sh2_section_title(ws2, r, 6, 4,
    f"Table 2 — Inter-Imaging Wait Gap  |  All Imaging (Sequential+Mixed, n={all_sequential_n+all_mixed_n:,})",
    bg="1F3864")
r += 1
sh2_colhdr(ws2, r, [("A","Statistic"), ("B","Minutes"), ("C","Hours"), ("D",""),
                     ("F","Statistic"), ("G","Minutes"), ("H","Hours"), ("I","")])
r += 1

def gap_row(gs, stat):
    if stat == "n":   return len(gs), "—"
    if stat == "med": return round(gs.median(),1), round(gs.median()/60,2)
    if stat == "mean":return round(gs.mean(),1),   round(gs.mean()/60,2)
    if stat == "p25": return round(gs.quantile(0.25),1), round(gs.quantile(0.25)/60,2)
    if stat == "p75": return round(gs.quantile(0.75),1), round(gs.quantile(0.75)/60,2)
    if stat == "max": return round(gs.max(),1),    round(gs.max()/60,2)

for i, (label, stat) in enumerate([("n gaps","n"),("Median","med"),("Mean","mean"),
                                     ("P25","p25"),("P75","p75"),("Max","max")]):
    bg = CLR_ROW_ALT if i % 2 == 0 else "FFFFFF"
    ct_m, ct_h  = gap_row(ct_gaps,  stat)
    all_m, all_h = gap_row(all_gaps, stat)
    sh2_cell(ws2, f"A{r}", label, bg, align="left");  sh2_cell(ws2, f"B{r}", ct_m,  bg); sh2_cell(ws2, f"C{r}", ct_h,  bg); sh2_cell(ws2, f"D{r}", "", bg)
    sh2_cell(ws2, f"F{r}", label, bg, align="left");  sh2_cell(ws2, f"G{r}", all_m, bg); sh2_cell(ws2, f"H{r}", all_h, bg); sh2_cell(ws2, f"I{r}", "", bg)
    r += 1
r += 1

# ── Table 3: Span by n orders — side by side ──────────────────────────────────
sh2_section_title(ws2, r, 1, 4, "Table 3 — Span by Number of Orders  |  CT Only", bg="1F5C99")
sh2_section_title(ws2, r, 6, 4, "Table 3 — Span by Number of Orders  |  All Imaging", bg="1F3864")
r += 1
sub_note = "Span = first exam start → last exam end. For concurrent patients, span ≈ protocol duration, NOT sum of scans."
sh2_subtitle(ws2, r, 1, 4, sub_note)
sh2_subtitle(ws2, r, 6, 4, sub_note)
r += 1
sh2_colhdr(ws2, r, [("A","Orders"), ("B","Patients"), ("C","Median Span (mins)"), ("D","Mean Span (mins)"),
                     ("F","Orders"), ("G","Patients"), ("H","Median Span (mins)"), ("I","Mean Span (mins)")])
r += 1

SCAN_N_COLORS = {1: CLR_P3, 2: CLR_P2, 3: CLR_P1}
max_n = max(span_ct["n_orders"].max(), span_all["n_orders"].max())
ct_span_dict  = {int(row["n_orders"]): row for _, row in span_ct.iterrows()}
all_span_dict = {int(row["n_orders"]): row for _, row in span_all.iterrows()}

for n in range(1, int(max_n) + 1):
    if n not in ct_span_dict and n not in all_span_dict:
        continue
    bg = SCAN_N_COLORS.get(n, CLR_ROW_ALT)
    label = f"{n} order{'s' if n>1 else ''}"
    # CT side
    if n in ct_span_dict:
        rd = ct_span_dict[n]
        sh2_cell(ws2, f"A{r}", label,                  bg, align="left")
        sh2_cell(ws2, f"B{r}", int(rd["n_pts"]),       bg)
        sh2_cell(ws2, f"C{r}", rd["med_span"],         bg)
        sh2_cell(ws2, f"D{r}", rd["mean_span"],        bg)
    else:
        for col in "ABCD": sh2_cell(ws2, f"{col}{r}", "—", "F5F5F5")
    # All imaging side
    if n in all_span_dict:
        rd = all_span_dict[n]
        sh2_cell(ws2, f"F{r}", label,                  bg, align="left")
        sh2_cell(ws2, f"G{r}", int(rd["n_pts"]),       bg)
        sh2_cell(ws2, f"H{r}", rd["med_span"],         bg)
        sh2_cell(ws2, f"I{r}", rd["mean_span"],        bg)
    else:
        for col in "FGHI": sh2_cell(ws2, f"{col}{r}", "—", "F5F5F5")
    r += 1

# Totals
sh2_cell(ws2, f"A{r}", "All patients", CLR_TOTAL_BG, bold=True, align="left")
sh2_cell(ws2, f"B{r}", ct_orders["CSN"].nunique(),    CLR_TOTAL_BG, bold=True)
sh2_cell(ws2, f"C{r}", round(pt_span_ct["span_mins"].dropna().median(),1), CLR_TOTAL_BG, bold=True)
sh2_cell(ws2, f"D{r}", round(pt_span_ct["span_mins"].dropna().mean(),1),   CLR_TOTAL_BG, bold=True)
sh2_cell(ws2, f"F{r}", "All patients", CLR_TOTAL_BG, bold=True, align="left")
sh2_cell(ws2, f"G{r}", df_ct_raw["CSN"].nunique(),    CLR_TOTAL_BG, bold=True)
sh2_cell(ws2, f"H{r}", round(pt_span_all["span_mins"].dropna().median(),1), CLR_TOTAL_BG, bold=True)
sh2_cell(ws2, f"I{r}", round(pt_span_all["span_mins"].dropna().mean(),1),   CLR_TOTAL_BG, bold=True)

ws2.column_dimensions["A"].width = 20
ws2.column_dimensions["B"].width = 14
ws2.column_dimensions["C"].width = 22
ws2.column_dimensions["D"].width = 42
ws2.column_dimensions["E"].width = 3    # spacer
ws2.column_dimensions["F"].width = 20
ws2.column_dimensions["G"].width = 14
ws2.column_dimensions["H"].width = 22
ws2.column_dimensions["I"].width = 44
for row_num in range(1, ws2.max_row + 1):
    ws2.row_dimensions[row_num].height = max(ws2.row_dimensions[row_num].height or 0, 20)
ws2.freeze_panes = "A3"


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 3 – Acuity Cross-Comparison
# ══════════════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("Acuity Cross-Comparison")
ws3.sheet_view.showGridLines = False

# ── Build patient-level joined dataset ───────────────────────────────────────
df_allpt_full = pd.read_excel("merged/all_pt_combined.xlsx",
                               usecols=["CSN", "Consult Acuity", "Triage Acuity"])

def norm_consult(v):
    if pd.isna(v): return v
    s = str(v).strip()
    return s[4:] if s.startswith("(P) ") else s

def norm_triage(v):
    if pd.isna(v): return v
    return str(v).strip().split()[0]

df_allpt_full["Consult_N"] = df_allpt_full["Consult Acuity"].apply(norm_consult)
df_allpt_full["Triage_N"]  = df_allpt_full["Triage Acuity"].apply(norm_triage)

ct_acuity_pt = (df_ct_raw.groupby("CSN")["Acuity Abbr"].first().reset_index()
                .rename(columns={"Acuity Abbr": "CT_Acuity"}))

joined = df_allpt_full.merge(ct_acuity_pt, on="CSN", how="inner")
n_joined = len(joined)

# Agreement counts (patient level)
agree_triage_consult = (joined["Triage_N"] == joined["Consult_N"]).sum()
agree_consult_ct     = (joined["Consult_N"] == joined["CT_Acuity"]).sum()
agree_all            = ((joined["Triage_N"] == joined["Consult_N"]) &
                        (joined["Consult_N"] == joined["CT_Acuity"])).sum()

disagree_triage_consult = n_joined - agree_triage_consult
disagree_consult_ct     = n_joined - agree_consult_ct
disagree_all            = n_joined - agree_all

# Triage vs Consult disagreements breakdown
diff_tc = joined[joined["Triage_N"] != joined["Consult_N"]].copy()
diff_tc["combo"] = diff_tc["Triage_N"].fillna("Missing") + " → " + diff_tc["Consult_N"].fillna("Missing")
tc_breakdown = diff_tc["combo"].value_counts().reset_index()
tc_breakdown.columns = ["Triage → Consult", "Count"]
tc_breakdown["% of all patients"] = tc_breakdown["Count"] / n_joined * 100

def uptriage_label(combo):
    order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    parts = combo.split(" → ")
    if len(parts) != 2: return "—"
    a, b = parts[0].strip(), parts[1].strip()
    oa, ob = order.get(a, 99), order.get(b, 99)
    if ob < oa: return "Uptriaged"
    if ob > oa: return "Downtriaged"
    return "Same"
tc_breakdown["Direction"] = tc_breakdown["Triage → Consult"].apply(uptriage_label)

# Full cross-tab
crosstab = (joined.groupby(
                [joined["Triage_N"].fillna("Missing"),
                 joined["Consult_N"].fillna("Missing"),
                 joined["CT_Acuity"].fillna("Missing")],
                dropna=False)
            .size()
            .reset_index(name="Count")
            .rename(columns={"Triage_N": "Triage Acuity", "Consult_N": "Consult Acuity", "CT_Acuity": "CT Acuity"})
            .sort_values("Count", ascending=False)
            .reset_index(drop=True))
crosstab["% of patients"] = crosstab["Count"] / n_joined * 100
crosstab["All agree"] = (crosstab["Triage Acuity"] == crosstab["Consult Acuity"]) & \
                        (crosstab["Consult Acuity"] == crosstab["CT Acuity"])

# ── Sheet layout ──────────────────────────────────────────────────────────────
def sh3_cell(ws, addr, value, bg, bold=False, align="center"):
    c = ws[addr]
    c.value = value; c.fill = fill(bg)
    c.font  = normal_font(bold=bold)
    c.alignment = left() if align == "left" else center()
    c.border = thin_border()

def sh3_hdr(ws, r, cols_vals):
    for col, val in cols_vals:
        c = ws[f"{col}{r}"]
        c.value = val; c.fill = fill(CLR_SECTION_BG)
        c.font  = Font(bold=True, color="1F3864", size=11)
        c.alignment = center(); c.border = thin_border()

# Page title
ws3.merge_cells("A1:G1")
c = ws3["A1"]
c.value = "Acuity Cross-Comparison — Same Patient, Three Acuity Sources"
c.fill  = fill(CLR_HEADER_DARK); c.font = Font(bold=True, color="FFFFFF", size=14)
c.alignment = center(); c.border = thin_border()
for col in "BCDEFG": ws3[f"{col}1"].border = thin_border()

ws3.merge_cells("A2:G2")
c = ws3["A2"]
c.value = (f"Patients matched across All Pt + CT Orders on CSN  |  n = {n_joined:,} patients  "
           f"•  Triage normalised to first token  •  Consult normalised (strip '(P) ')")
c.fill  = fill(CLR_HEADER_MID); c.font = Font(italic=True, color="FFFFFF", size=10)
c.alignment = center(); c.border = thin_border()
for col in "BCDEFG": ws3[f"{col}2"].border = thin_border()

# ── Table 1: Agreement summary ────────────────────────────────────────────────
r = 4
ws3.merge_cells(f"A{r}:E{r}")
ws3[f"A{r}"].value = "Table 1 — Acuity Agreement Summary (patient level)"
ws3[f"A{r}"].fill  = fill(CLR_HEADER_DARK); ws3[f"A{r}"].font = Font(bold=True, color="FFFFFF", size=12)
ws3[f"A{r}"].alignment = center(); ws3[f"A{r}"].border = thin_border()
for col in ["B","C","D","E"]: ws3[f"{col}{r}"].border = thin_border()
r += 1
sh3_hdr(ws3, r, [("A","Comparison"), ("B","Agree (n)"), ("C","Agree (%)"),
                 ("D","Disagree (n)"), ("E","Disagree (%)")])
r += 1

agree_rows = [
    ("Triage == Consult",        agree_triage_consult, disagree_triage_consult, CLR_ROW_ALT),
    ("Consult == CT Acuity",     agree_consult_ct,     disagree_consult_ct,     "FFFFFF"),
    ("All three agree",          agree_all,            disagree_all,            CLR_ROW_ALT),
]
for label, ag, dis, bg in agree_rows:
    sh3_cell(ws3, f"A{r}", label, bg, align="left")
    sh3_cell(ws3, f"B{r}", ag,  bg)
    sh3_cell(ws3, f"C{r}", f"{ag/n_joined*100:.1f}%",  bg)
    sh3_cell(ws3, f"D{r}", dis, bg)
    sh3_cell(ws3, f"E{r}", f"{dis/n_joined*100:.1f}%", bg)
    r += 1
r += 1

# ── Table 2: Triage vs Consult disagreement breakdown ────────────────────────
ws3.merge_cells(f"A{r}:E{r}")
ws3[f"A{r}"].value = "Table 2 — Triage vs Consult Disagreements (uptriage / downtriage)"
ws3[f"A{r}"].fill  = fill(CLR_HEADER_DARK); ws3[f"A{r}"].font = Font(bold=True, color="FFFFFF", size=12)
ws3[f"A{r}"].alignment = center(); ws3[f"A{r}"].border = thin_border()
for col in ["B","C","D","E"]: ws3[f"{col}{r}"].border = thin_border()
r += 1
sh3_hdr(ws3, r, [("A","Triage → Consult"), ("B","Direction"),
                 ("C","Count"), ("D","% of all patients")])
r += 1

DIR_COLORS = {"Uptriaged": CLR_P1, "Downtriaged": CLR_P3, "—": CLR_MISSING}
for i, (_, row_data) in enumerate(tc_breakdown.iterrows()):
    bg = DIR_COLORS.get(row_data["Direction"], CLR_ROW_ALT if i % 2 == 0 else "FFFFFF")
    sh3_cell(ws3, f"A{r}", row_data["Triage → Consult"], bg, align="left")
    sh3_cell(ws3, f"B{r}", row_data["Direction"],        bg)
    sh3_cell(ws3, f"C{r}", int(row_data["Count"]),       bg)
    sh3_cell(ws3, f"D{r}", f"{row_data['% of all patients']:.1f}%", bg)
    r += 1

sh3_cell(ws3, f"A{r}", f"Total disagreements (Triage ≠ Consult)", CLR_TOTAL_BG, bold=True, align="left")
sh3_cell(ws3, f"B{r}", "",                                         CLR_TOTAL_BG)
sh3_cell(ws3, f"C{r}", int(tc_breakdown["Count"].sum()),           CLR_TOTAL_BG, bold=True)
sh3_cell(ws3, f"D{r}", f"{tc_breakdown['Count'].sum()/n_joined*100:.1f}%", CLR_TOTAL_BG, bold=True)
r += 2

# ── Table 3: Full cross-tab ───────────────────────────────────────────────────
ws3.merge_cells(f"A{r}:F{r}")
ws3[f"A{r}"].value = "Table 3 — Full Combination Cross-Tab (Triage × Consult × CT Acuity)"
ws3[f"A{r}"].fill  = fill(CLR_HEADER_DARK); ws3[f"A{r}"].font = Font(bold=True, color="FFFFFF", size=12)
ws3[f"A{r}"].alignment = center(); ws3[f"A{r}"].border = thin_border()
for col in ["B","C","D","E","F"]: ws3[f"{col}{r}"].border = thin_border()
r += 1
sh3_hdr(ws3, r, [("A","Triage Acuity"), ("B","Consult Acuity"), ("C","CT Acuity"),
                 ("D","Count"), ("E","% of patients"), ("F","All Agree?")])
r += 1

for i, (_, row_data) in enumerate(crosstab.iterrows()):
    agreed = row_data["All agree"]
    bg = CLR_P3 if agreed else (CLR_P1 if i % 2 == 0 else CLR_P2)
    sh3_cell(ws3, f"A{r}", row_data["Triage Acuity"],  bg, align="left")
    sh3_cell(ws3, f"B{r}", row_data["Consult Acuity"], bg, align="left")
    sh3_cell(ws3, f"C{r}", row_data["CT Acuity"],      bg, align="left")
    sh3_cell(ws3, f"D{r}", int(row_data["Count"]),     bg)
    sh3_cell(ws3, f"E{r}", f"{row_data['% of patients']:.1f}%", bg)
    sh3_cell(ws3, f"F{r}", "Yes" if agreed else "No",  bg)
    r += 1

sh3_cell(ws3, f"A{r}", "Total", CLR_TOTAL_BG, bold=True, align="left")
sh3_cell(ws3, f"B{r}", "",      CLR_TOTAL_BG)
sh3_cell(ws3, f"C{r}", "",      CLR_TOTAL_BG)
sh3_cell(ws3, f"D{r}", n_joined, CLR_TOTAL_BG, bold=True)
sh3_cell(ws3, f"E{r}", "100.0%", CLR_TOTAL_BG, bold=True)
sh3_cell(ws3, f"F{r}", "",       CLR_TOTAL_BG)

ws3.column_dimensions["A"].width = 18
ws3.column_dimensions["B"].width = 18
ws3.column_dimensions["C"].width = 16
ws3.column_dimensions["D"].width = 14
ws3.column_dimensions["E"].width = 18
ws3.column_dimensions["F"].width = 12
for row_num in range(1, ws3.max_row + 1):
    ws3.row_dimensions[row_num].height = max(ws3.row_dimensions[row_num].height or 0, 20)
ws3.freeze_panes = "A3"


# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "analysis/acuity_distribution.xlsx"
wb.save(out_path)
print(f"Saved → {out_path}")
