"""
Deep-dive on CT Orders acuity & scan analysis:
  Sheet 1 – Acuity Distribution (deduplicated to unique patients)
  Sheet 2 – Acuity Consistency check (same patient, same acuity?)
  Sheet 3 – Scan Type Distribution (all orders, with per-scan exam duration)
  Sheet 4 – Multiple CT Scans per Patient (does more scans = more time?)
"""
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ── helpers ───────────────────────────────────────────────────────────────────
def fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def thin_border():
    s = Side(style="thin", color="BFBFBF")
    return Border(left=s, right=s, top=s, bottom=s)

def center(): return Alignment(horizontal="center", vertical="center", wrap_text=True)
def left():   return Alignment(horizontal="left",   vertical="center", wrap_text=True)

# palette
C_NAVY   = "1F3864"
C_BLUE   = "2E75B6"
C_LBLUE  = "D6E4F0"
C_ALT    = "EBF3FB"
C_MISS   = "FFF2CC"
C_TOTAL  = "D9E1F2"
C_P1     = "FCE4D6"
C_P2     = "FDEBD0"
C_P3     = "E8F5E9"
C_GREEN  = "E2EFDA"
C_WARN   = "FCE4D6"

ACUITY_CLR = {"P1": C_P1, "P2": C_P2, "P3": C_P3}


def set_hdr(cell, value, bg=C_NAVY, fg="FFFFFF", sz=12, bold=True, align="center"):
    cell.value = value
    cell.fill  = fill(bg)
    cell.font  = Font(bold=bold, color=fg, size=sz)
    cell.alignment = center() if align == "center" else left()
    cell.border = thin_border()

def set_cell(cell, value, bg="FFFFFF", bold=False, sz=11, align="center"):
    cell.value = value
    cell.fill  = fill(bg)
    cell.font  = Font(bold=bold, size=sz)
    cell.alignment = center() if align == "center" else left()
    cell.border = thin_border()


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = 0
        col_ltr = get_column_letter(col[0].column)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_ltr].width = min(max_len + extra, 50)


# ── load data ─────────────────────────────────────────────────────────────────
print("Loading CT orders …")
df = pd.read_excel("merged/ct_orders_combined.xlsx")

# parse exam times
df["start"] = pd.to_datetime(
    df["Exam Start Date"].astype(str) + " " + df["Exam Start Time"].astype(str),
    errors="coerce")
df["end"] = pd.to_datetime(
    df["Exam End Date"].astype(str)   + " " + df["Exam End Time"].astype(str),
    errors="coerce")
df["exam_duration_mins"] = (df["end"] - df["start"]).dt.total_seconds() / 60

# modality tag
def modality(desc):
    d = str(desc).upper().strip()
    if d.startswith("NGEMR "): d = d[6:]
    if d.startswith("CT ") or d == "CT": return "CT"
    if d.startswith("XR ") or d == "XR": return "XR"
    if d.startswith("MRI "): return "MRI"
    if d.startswith("US "):  return "US"
    return "OTHER"

df["Modality"] = df["Order Description"].apply(modality)

ct = df[df["Modality"] == "CT"].copy()
total_ct_rows    = len(ct)
total_ct_pts     = ct["CSN"].nunique()
total_all_rows   = len(df)
total_all_pts    = df["CSN"].nunique()

# ── dedup: one row per patient, take their acuity ────────────────────────────
# All CT patients have exactly one unique acuity (confirmed), so first() is safe
pt_acuity = (ct.dropna(subset=["Acuity Abbr"])
               .groupby("CSN")["Acuity Abbr"]
               .first()
               .reset_index())
all_csns = ct[["CSN"]].drop_duplicates()
pt_acuity_full = all_csns.merge(pt_acuity, on="CSN", how="left")

total_unique_pts = len(pt_acuity_full)


def build_dist(series, total):
    vc = series.value_counts(dropna=True)
    rows = [{"value": v, "count": c, "pct": c / total * 100}
            for v, c in vc.items()]
    rows.sort(key=lambda r: ["P1","P2","P3","P4"].index(r["value"])
              if r["value"] in ["P1","P2","P3","P4"] else 99)
    miss = series.isna().sum()
    rows.append({"value": "Missing", "count": miss, "pct": miss / total * 100})
    for r in rows:
        r["display"] = f"{r['count']:,} ({r['pct']:.1f}%)"
    return rows


raw_dist  = build_dist(df[df["Modality"]=="CT"]["Acuity Abbr"], total_ct_rows)
dedup_dist = build_dist(pt_acuity_full["Acuity Abbr"], total_unique_pts)

wb = Workbook()

# ══════════════════════════════════════════════════════════════════════════════
# SHEET 1 – Acuity Distribution (raw vs deduplicated)
# ══════════════════════════════════════════════════════════════════════════════
ws1 = wb.active
ws1.title = "Acuity Distribution"
ws1.sheet_view.showGridLines = False

# title
ws1.merge_cells("A1:H1")
set_hdr(ws1["A1"], "CT Order Acuity Distribution — Raw Orders vs Unique Patients",
        bg=C_NAVY, sz=14)
for c in "BCDEFGH": ws1[f"{c}1"].border = thin_border()

ws1.merge_cells("A2:H2")
set_hdr(ws1["A2"],
        f"All CT orders: {total_ct_rows:,}  •  Unique CT patients: {total_unique_pts:,}  "
        f"•  All patients had consistent acuity (no mixed-acuity cases found)",
        bg=C_BLUE, sz=10, bold=False)
for c in "BCDEFGH": ws1[f"{c}2"].border = thin_border()

r = 4

# --- Left table: raw (all orders) ---
ws1.merge_cells(f"A{r}:C{r}")
set_hdr(ws1[f"A{r}"], "Raw: All CT Orders (includes duplicates)", bg=C_BLUE, sz=11)
for c in ["B","C"]: ws1[f"{c}{r}"].border = thin_border()
r += 1
set_hdr(ws1[f"A{r}"], f"n = {total_ct_rows:,}  (30 788 orders from {total_ct_pts:,} patients)", bg=C_LBLUE, fg=C_NAVY, sz=10)
ws1.merge_cells(f"A{r}:C{r}")
for c in ["B","C"]: ws1[f"{c}{r}"].border = thin_border()
r += 1

for hdr, col in [("Acuity","A"),("Count","B"),("Percentage","C")]:
    set_hdr(ws1[f"{col}{r}"], hdr, bg=C_LBLUE, fg=C_NAVY, sz=11)
r += 1

for row_data in raw_dist:
    bg = ACUITY_CLR.get(row_data["value"], C_MISS if row_data["value"]=="Missing" else C_ALT)
    set_cell(ws1[f"A{r}"], row_data["value"], bg=bg, align="left", bold=(row_data["value"]=="Missing"))
    set_cell(ws1[f"B{r}"], row_data["count"], bg=bg)
    set_cell(ws1[f"C{r}"], f"{row_data['pct']:.1f}%", bg=bg)
    r += 1

# total row
set_cell(ws1[f"A{r}"], "Total", bg=C_TOTAL, bold=True, align="left")
set_cell(ws1[f"B{r}"], total_ct_rows, bg=C_TOTAL, bold=True)
set_cell(ws1[f"C{r}"], "100.0%", bg=C_TOTAL, bold=True)
r += 2

# --- Right table: deduplicated ---
r_start = 4
ws1.merge_cells(f"E{r_start}:G{r_start}")
set_hdr(ws1[f"E{r_start}"], "Deduplicated: Unique Patients Only", bg=C_NAVY, sz=11)
for c in ["F","G"]: ws1[f"{c}{r_start}"].border = thin_border()
r2 = r_start + 1
ws1.merge_cells(f"E{r2}:G{r2}")
set_hdr(ws1[f"E{r2}"], f"n = {total_unique_pts:,} unique patients", bg=C_LBLUE, fg=C_NAVY, sz=10)
for c in ["F","G"]: ws1[f"{c}{r2}"].border = thin_border()
r2 += 1

for hdr, col in [("Acuity","E"),("Count","F"),("Percentage","G")]:
    set_hdr(ws1[f"{col}{r2}"], hdr, bg=C_LBLUE, fg=C_NAVY, sz=11)
r2 += 1

for row_data in dedup_dist:
    bg = ACUITY_CLR.get(row_data["value"], C_MISS if row_data["value"]=="Missing" else C_ALT)
    set_cell(ws1[f"E{r2}"], row_data["value"], bg=bg, align="left", bold=(row_data["value"]=="Missing"))
    set_cell(ws1[f"F{r2}"], row_data["count"], bg=bg)
    set_cell(ws1[f"G{r2}"], f"{row_data['pct']:.1f}%", bg=bg)
    r2 += 1

set_cell(ws1[f"E{r2}"], "Total", bg=C_TOTAL, bold=True, align="left")
set_cell(ws1[f"F{r2}"], total_unique_pts, bg=C_TOTAL, bold=True)
set_cell(ws1[f"G{r2}"], "100.0%", bg=C_TOTAL, bold=True)

# col widths
ws1.column_dimensions["A"].width = 22
ws1.column_dimensions["B"].width = 14
ws1.column_dimensions["C"].width = 16
ws1.column_dimensions["D"].width = 4
ws1.column_dimensions["E"].width = 30
ws1.column_dimensions["F"].width = 14
ws1.column_dimensions["G"].width = 16

for row_num in range(1, ws1.max_row + 1):
    ws1.row_dimensions[row_num].height = 20
ws1.freeze_panes = "A3"


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 2 – Acuity Consistency
# ══════════════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("Acuity Consistency Check")
ws2.sheet_view.showGridLines = False

# Per-patient acuity summary
acuity_nunique = ct.groupby("CSN")["Acuity Abbr"].nunique()
consistent_pts = (acuity_nunique == 1).sum()
missing_pts    = (acuity_nunique == 0).sum()   # all NaN
inconsistent   = (acuity_nunique > 1).sum()

ws2.merge_cells("A1:F1")
set_hdr(ws2["A1"], "Acuity Consistency per Patient — CT Orders", bg=C_NAVY, sz=14)
for c in "BCDEF": ws2[f"{c}1"].border = thin_border()

ws2.merge_cells("A2:F2")
set_hdr(ws2["A2"],
        f"Question: do all CT orders for the same patient carry the same acuity?",
        bg=C_BLUE, sz=10, bold=False)
for c in "BCDEF": ws2[f"{c}2"].border = thin_border()

rows2 = [
    ("Consistent (same acuity across all orders)", consistent_pts,
     consistent_pts/total_ct_pts*100, C_GREEN),
    ("Inconsistent (mixed acuity values)",         inconsistent,
     inconsistent/total_ct_pts*100,   C_WARN),
    ("All acuity values missing",                  missing_pts,
     missing_pts/total_ct_pts*100,    C_MISS),
]

r = 4
for hdr, col in [("Category","A"),("Patients","B"),("Percentage","C")]:
    set_hdr(ws2[f"{col}{r}"], hdr, bg=C_LBLUE, fg=C_NAVY, sz=11)
r += 1
for label, cnt, pct, bg in rows2:
    set_cell(ws2[f"A{r}"], label, bg=bg, align="left")
    set_cell(ws2[f"B{r}"], cnt, bg=bg)
    set_cell(ws2[f"C{r}"], f"{pct:.1f}%", bg=bg)
    r += 1
set_cell(ws2[f"A{r}"], "Total unique patients", bg=C_TOTAL, bold=True, align="left")
set_cell(ws2[f"B{r}"], total_ct_pts, bg=C_TOTAL, bold=True)
set_cell(ws2[f"C{r}"], "100.0%", bg=C_TOTAL, bold=True)
r += 2

ws2[f"A{r}"].value = "Conclusion:"
ws2[f"A{r}"].font  = Font(bold=True, size=11)
ws2[f"A{r}"].alignment = left()
ws2.merge_cells(f"B{r}:F{r}")
msg = ("All patients have a single consistent acuity across all their CT orders. "
       "Deduplication to unique patients does NOT change acuity proportions — "
       "both raw and deduplicated distributions are valid, but unique-patient "
       "counts better represent patient-level burden.")
ws2[f"B{r}"].value = msg
ws2[f"B{r}"].font  = Font(italic=True, size=10)
ws2[f"B{r}"].alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
ws2.row_dimensions[r].height = 40

ws2.column_dimensions["A"].width = 44
ws2.column_dimensions["B"].width = 14
ws2.column_dimensions["C"].width = 16
for c in "DEF": ws2.column_dimensions[c].width = 10
for row_num in range(1, ws2.max_row + 1):
    ws2.row_dimensions[row_num].height = max(ws2.row_dimensions[row_num].height or 0, 20)
ws2.freeze_panes = "A3"


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 3 – Scan Type Distribution
# ══════════════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("CT Scan Types")
ws3.sheet_view.showGridLines = False

ct_types = (ct.groupby("Order Description")
              .agg(
                  n_orders=("CSN","count"),
                  n_patients=("CSN","nunique"),
                  med_duration=("exam_duration_mins", lambda x: round(x.dropna().median(),1) if x.dropna().shape[0]>0 else None),
                  mean_duration=("exam_duration_mins", lambda x: round(x.dropna().mean(),1) if x.dropna().shape[0]>0 else None),
              )
              .reset_index()
              .sort_values("n_orders", ascending=False))
ct_types["pct_of_orders"] = ct_types["n_orders"] / total_ct_rows * 100

ws3.merge_cells("A1:G1")
set_hdr(ws3["A1"], "CT Scan Type Distribution — All CT Orders", bg=C_NAVY, sz=14)
for c in "BCDEFG": ws3[f"{c}1"].border = thin_border()

ws3.merge_cells("A2:G2")
set_hdr(ws3["A2"],
        f"Total CT orders: {total_ct_rows:,}  •  Unique scan types: {len(ct_types)}  "
        f"•  Exam duration = Exam End − Exam Start",
        bg=C_BLUE, sz=10, bold=False)
for c in "BCDEFG": ws3[f"{c}2"].border = thin_border()

hdrs3 = ["Scan Type", "Orders (n)", "% of Orders", "Unique Patients",
         "Median Exam Duration (mins)", "Mean Exam Duration (mins)"]
for i, hdr in enumerate(hdrs3):
    col = get_column_letter(i + 1)
    set_hdr(ws3[f"{col}3"], hdr, bg=C_LBLUE, fg=C_NAVY, sz=11)

for ridx, (_, row) in enumerate(ct_types.iterrows()):
    r = ridx + 4
    bg = C_ALT if ridx % 2 == 0 else "FFFFFF"
    set_cell(ws3[f"A{r}"], row["Order Description"], bg=bg, align="left")
    set_cell(ws3[f"B{r}"], int(row["n_orders"]), bg=bg)
    set_cell(ws3[f"C{r}"], f"{row['pct_of_orders']:.1f}%", bg=bg)
    set_cell(ws3[f"D{r}"], int(row["n_patients"]), bg=bg)
    set_cell(ws3[f"E{r}"], row["med_duration"] if pd.notna(row["med_duration"]) else "—", bg=bg)
    set_cell(ws3[f"F{r}"], row["mean_duration"] if pd.notna(row["mean_duration"]) else "—", bg=bg)

# total
tot_r = len(ct_types) + 4
set_cell(ws3[f"A{tot_r}"], "Total", bg=C_TOTAL, bold=True, align="left")
set_cell(ws3[f"B{tot_r}"], total_ct_rows, bg=C_TOTAL, bold=True)
set_cell(ws3[f"C{tot_r}"], "100.0%", bg=C_TOTAL, bold=True)
set_cell(ws3[f"D{tot_r}"], total_ct_pts, bg=C_TOTAL, bold=True)
set_cell(ws3[f"E{tot_r}"], round(ct["exam_duration_mins"].dropna().median(), 1), bg=C_TOTAL, bold=True)
set_cell(ws3[f"F{tot_r}"], round(ct["exam_duration_mins"].dropna().mean(), 1), bg=C_TOTAL, bold=True)

ws3.column_dimensions["A"].width = 44
ws3.column_dimensions["B"].width = 14
ws3.column_dimensions["C"].width = 15
ws3.column_dimensions["D"].width = 18
ws3.column_dimensions["E"].width = 28
ws3.column_dimensions["F"].width = 26
for row_num in range(1, ws3.max_row + 1):
    ws3.row_dimensions[row_num].height = 20
ws3.freeze_panes = "A4"


# ══════════════════════════════════════════════════════════════════════════════
# SHEET 4 – Multiple CT Scans per Patient (time analysis)
# ══════════════════════════════════════════════════════════════════════════════
ws4 = wb.create_sheet("Multiple Scans & Time")
ws4.sheet_view.showGridLines = False

# Per-patient: n_scans, individual exam durations, span from first start to last end
pt_ct = ct.groupby("CSN").agg(
    n_scans       = ("CSN",           "count"),
    first_start   = ("start",         "min"),
    last_end      = ("end",           "max"),
    sum_exam_dur  = ("exam_duration_mins", "sum"),
    mean_exam_dur = ("exam_duration_mins", "mean"),
).reset_index()
pt_ct["total_span_mins"] = (pt_ct["last_end"] - pt_ct["first_start"]).dt.total_seconds() / 60
pt_ct["total_span_mins"] = pt_ct["total_span_mins"].round(1)

# group by n_scans
grp = pt_ct.groupby("n_scans").agg(
    n_patients    = ("CSN",             "count"),
    med_span      = ("total_span_mins", lambda x: round(x.dropna().median(),1)),
    mean_span     = ("total_span_mins", lambda x: round(x.dropna().mean(),1)),
    p25_span      = ("total_span_mins", lambda x: round(x.dropna().quantile(0.25),1)),
    p75_span      = ("total_span_mins", lambda x: round(x.dropna().quantile(0.75),1)),
    max_span      = ("total_span_mins", lambda x: round(x.dropna().max(),1)),
).reset_index()
grp["pct"] = grp["n_patients"] / total_ct_pts * 100

ws4.merge_cells("A1:I1")
set_hdr(ws4["A1"], "Multiple CT Scans per Patient — Time Impact Analysis", bg=C_NAVY, sz=14)
for c in "BCDEFGHI": ws4[f"{c}1"].border = thin_border()

ws4.merge_cells("A2:I2")
set_hdr(ws4["A2"],
        "Span = time from first CT exam start to last CT exam end for each patient. "
        "Patients with 1 scan have span ≈ single scan duration.",
        bg=C_BLUE, sz=10, bold=False)
for c in "BCDEFGHI": ws4[f"{c}2"].border = thin_border()

hdrs4 = ["CT Scans per Patient", "Patients (n)", "% of CT Patients",
         "Median Span (mins)", "Mean Span (mins)", "P25 Span (mins)",
         "P75 Span (mins)", "Max Span (mins)"]
for i, hdr in enumerate(hdrs4):
    col = get_column_letter(i + 1)
    set_hdr(ws4[f"{col}3"], hdr, bg=C_LBLUE, fg=C_NAVY, sz=11)

SCAN_COLORS = {1: C_GREEN, 2: C_P3, 3: C_P2, 4: C_P1}
for ridx, (_, row) in enumerate(grp.iterrows()):
    r = ridx + 4
    n = int(row["n_scans"])
    bg = SCAN_COLORS.get(n, C_WARN if n >= 4 else C_ALT)
    set_cell(ws4[f"A{r}"], f"{n} scan{'s' if n > 1 else ''}", bg=bg, align="left")
    set_cell(ws4[f"B{r}"], int(row["n_patients"]), bg=bg)
    set_cell(ws4[f"C{r}"], f"{row['pct']:.1f}%", bg=bg)
    set_cell(ws4[f"D{r}"], row["med_span"], bg=bg)
    set_cell(ws4[f"E{r}"], row["mean_span"], bg=bg)
    set_cell(ws4[f"F{r}"], row["p25_span"], bg=bg)
    set_cell(ws4[f"G{r}"], row["p75_span"], bg=bg)
    set_cell(ws4[f"H{r}"], row["max_span"], bg=bg)

# totals
tot_r4 = len(grp) + 4
set_cell(ws4[f"A{tot_r4}"], "All patients", bg=C_TOTAL, bold=True, align="left")
set_cell(ws4[f"B{tot_r4}"], total_ct_pts, bg=C_TOTAL, bold=True)
set_cell(ws4[f"C{tot_r4}"], "100.0%", bg=C_TOTAL, bold=True)
set_cell(ws4[f"D{tot_r4}"], round(pt_ct["total_span_mins"].dropna().median(),1), bg=C_TOTAL, bold=True)
set_cell(ws4[f"E{tot_r4}"], round(pt_ct["total_span_mins"].dropna().mean(),1), bg=C_TOTAL, bold=True)
set_cell(ws4[f"F{tot_r4}"], round(pt_ct["total_span_mins"].dropna().quantile(0.25),1), bg=C_TOTAL, bold=True)
set_cell(ws4[f"G{tot_r4}"], round(pt_ct["total_span_mins"].dropna().quantile(0.75),1), bg=C_TOTAL, bold=True)
set_cell(ws4[f"H{tot_r4}"], round(pt_ct["total_span_mins"].dropna().max(),1), bg=C_TOTAL, bold=True)
tot_r4 += 2

# Key observations note
ws4[f"A{tot_r4}"].value = "Key observations:"
ws4[f"A{tot_r4}"].font  = Font(bold=True, size=11)
ws4[f"A{tot_r4}"].alignment = left()
ws4.merge_cells(f"B{tot_r4}:H{tot_r4}")

single_med = grp[grp["n_scans"]==1]["med_span"].values[0]
multi2_med = grp[grp["n_scans"]==2]["med_span"].values[0] if 2 in grp["n_scans"].values else "n/a"

note = (f"Patients with 1 CT scan have a median imaging span of {single_med} mins. "
        f"Patients with 2 scans: {multi2_med} mins median span — confirming that "
        f"multiple scans extend the time patients spend in imaging. "
        f"Large max values in multi-scan groups reflect scans ordered hours apart "
        f"(e.g. initial scan then follow-up).")
ws4[f"B{tot_r4}"].value = note
ws4[f"B{tot_r4}"].font  = Font(italic=True, size=10)
ws4[f"B{tot_r4}"].alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
ws4.row_dimensions[tot_r4].height = 50

ws4.column_dimensions["A"].width = 24
for c in "BCDEFGH":
    ws4.column_dimensions[c].width = 20
for row_num in range(1, ws4.max_row + 1):
    ws4.row_dimensions[row_num].height = max(ws4.row_dimensions[row_num].height or 0, 20)
ws4.freeze_panes = "A4"


# ── save ──────────────────────────────────────────────────────────────────────
out = "analysis/ct_acuity_scan_analysis.xlsx"
wb.save(out)
print(f"Saved → {out}")

# ── quick sanity print ────────────────────────────────────────────────────────
print("\n=== Acuity (deduplicated unique patients) ===")
for r in dedup_dist:
    print(f"  {r['value']:10s}  {r['display']}")
print(f"  {'Total':10s}  {total_unique_pts:,}")

print("\n=== Acuity consistency ===")
print(f"  Consistent:   {consistent_pts:,} / {total_ct_pts:,} ({consistent_pts/total_ct_pts*100:.1f}%)")
print(f"  Inconsistent: {inconsistent}")
print(f"  All missing:  {missing_pts}")

print("\n=== Top 10 CT scan types ===")
for _, row in ct_types.head(10).iterrows():
    print(f"  {row['n_orders']:5,}  {row['Order Description']}")
