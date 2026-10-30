"""
clean_ct_orders.py
Cleans merged/ct_orders_combined.xlsx and writes merged/ct_orders_cleaned.xlsx:

  Both sheets:
    - Rows whose CSN has no matching entry in merged/all_pt_combined.xlsx
      are dropped.

  Imaging:
    - Rows whose Exam Start/End windows overlap or are identical (same CSN
      AND same Procedure Category) are merged into one row (transitive
      interval-merge, never across Procedure Category -- an overlapping CT
      and XR order on the same CSN are kept as separate rows). This applies
      even when a group's members resolve to different physical CT sites
      (NNI vs DR, per Ordering Question Responses' Performing Location) --
      they're still merged as long as their exam windows overlap and it's
      the same patient; see 'Site' below for how the merged row's site is
      picked.
    - Exam Start/End still missing after merge fall back to Order Completion
      Date/Time; if that's also missing, flagged via 'Exam Time Source'.
    - Order Date/Order Time columns dropped (Parent Order Date/Time remain).
    - Rows with no Parent Order time, no Exam Start, or a zero/negative-
      duration exam window (Exam End <= Exam Start) are dropped. CT rows
      (Procedure Category == "IMG CT ORDERABLES") are additionally dropped
      for a negative wait time (Exam Start < Parent Order); zero wait time
      is retained.
    - 'Procedure Location' is added from the Ordering Question Responses'
      Performing Location, falling back to the tsv-derived CT site only when
      Performing Location is empty -- decided per original row before
      merging, so a merged group takes the value belonging to whichever
      original row had the earliest Exam Start.
    - 'Site': the physical CT site (NNI/DR) derived from that parsed
      Performing Location, falling back to brainbodysep.tsv only when
      Performing Location is empty. For overlapping merged groups with
      conflicting locations, the merged row takes the site of whichever
      original row had the earliest Exam Start.

  Lab:
    - No row-merging, no Order Completion fallback.
    - Order Date/Order Time columns dropped (Parent Order Date/Time remain).
"""

import os
import re
import math
from datetime import datetime, timedelta, date, time as time_type

import openpyxl
import pandas as pd

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
OUT_PATH = os.path.join(MERGED, "ct_orders_cleaned.xlsx")


# ---------------------------------------------------------------------------
# Helpers (same pattern as imaging_vs_lab_timing.py / compile_report.py)
# ---------------------------------------------------------------------------
def load_sheet(path, sheet):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return pd.DataFrame()
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    seen = {}; uh = []
    for h in headers:
        if h in seen: seen[h] += 1; uh.append(f"{h}_{seen[h]}")
        else: seen[h] = 0; uh.append(h)
    return pd.DataFrame(rows[1:], columns=uh)


def to_dt(val):
    if val is None:
        return None
    try:
        if isinstance(val, float) and math.isnan(val):
            return None
    except Exception:
        pass
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        return datetime(val.year, val.month, val.day)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
                    "%d/%m/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return datetime(1899, 12, 30) + timedelta(days=f)
    except Exception:
        return None


def to_time(val):
    """Parse a time-only value: time/datetime object, OLE fraction, or string
    ('13:27:00' / '1:59 PM' / ' 1:59 PM')."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.time()
    if isinstance(val, time_type):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"):
            try:
                return datetime.strptime(s, fmt).time()
            except ValueError:
                continue
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        if 0 <= f < 2:
            ts = round(f * 86400); h, r = divmod(ts, 3600); m, s = divmod(r, 60)
            return time_type(h % 24, m, s)
    except Exception:
        pass
    return None


def combine_dt(d_val, t_val):
    d = to_dt(d_val)
    if d is None:
        return None
    if t_val is None:
        return d
    t = to_time(t_val)
    if t is None:
        return d
    return datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)


def is_blank(v):
    if v is None:
        return True
    try:
        if isinstance(v, float) and math.isnan(v):
            return True
    except Exception:
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def split_joined(val):
    """clean_ct_orders.py joins values from merged rows with '; '. Only split
    when val is actually a joined string -- stringifying a single raw
    date/datetime value here would break to_dt's isinstance checks and its
    string-format parsing (which doesn't fall back to float/date-serial
    parsing), so a non-joined value is passed through untouched."""
    if is_blank(val):
        return [None]
    if isinstance(val, str) and "; " in val:
        return val.split("; ")
    return [val]


def earliest_parent_order_dt(date_val, time_val):
    """Parent Order Date/Time can be a '; '-joined multi-value string on rows
    merged from originally-separate orders. Returns the earliest combined
    timestamp across all joined candidates (same pattern as build_ct_queue.py)."""
    d_parts = split_joined(date_val)
    t_parts = split_joined(time_val)
    n = max(len(d_parts), len(t_parts))
    if len(d_parts) < n:
        d_parts = d_parts + [d_parts[-1]] * (n - len(d_parts))
    if len(t_parts) < n:
        t_parts = t_parts + [t_parts[-1]] * (n - len(t_parts))
    dts = [combine_dt(d, t) for d, t in zip(d_parts, t_parts)]
    dts = [dt for dt in dts if dt is not None]
    return min(dts) if dts else None


PRIORITY_PATTERN = re.compile(r"(?:ED )?Scan Priority:\s*(P[1-3])")
LOC_PATTERN = re.compile(r"Performing Location:\s*([^\n;]+)")


def parse_priority(val):
    """Scan Priority (P1/P2/P3) parsed from Ordering Question Responses, same
    pattern as build_ct_queue.py -- 'Unknown' if blank or not found."""
    if is_blank(val):
        return "Unknown"
    m = PRIORITY_PATTERN.search(str(val))
    return m.group(1) if m else "Unknown"


def parse_performing_location(val):
    """Performing Location answer(s) from Ordering Question Responses, same
    pattern as nni_scan_priority_location_check.py -- 'Ordering Question
    Responses' can hold multiple '; '-joined submission blocks, so this
    collects every distinct location recorded across all of them."""
    if is_blank(val):
        return None
    locs = sorted(set(m.strip() for m in LOC_PATTERN.findall(str(val))))
    return " + ".join(locs) if locs else None


# Modality mapping, same as acuity_distribution_analysis.py
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


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
ct_path = os.path.join(MERGED, "ct_orders_combined.xlsx")
print("Loading merged/ct_orders_combined.xlsx ...")
df_img = load_sheet(ct_path, "Imaging")
df_lab = load_sheet(ct_path, "Lab")
print(f"  Imaging: {len(df_img):,} rows  |  Lab: {len(df_lab):,} rows")

print("Loading merged/all_pt_combined.xlsx for CSN cross-check...")
allpt_path = os.path.join(MERGED, "all_pt_combined.xlsx")
df_allpt = load_sheet(allpt_path, "All Patients")
allpt_csns = set(df_allpt["CSN"].dropna().apply(lambda v: str(v).strip()))

_img_csn = df_img["CSN"].apply(lambda v: str(v).strip() if not is_blank(v) else "")
_lab_csn = df_lab["CSN"].apply(lambda v: str(v).strip() if not is_blank(v) else "")
n_img_not_allpt = int((~_img_csn.isin(allpt_csns)).sum())
n_lab_not_allpt = int((~_lab_csn.isin(allpt_csns)).sum())
df_img = df_img[_img_csn.isin(allpt_csns)].reset_index(drop=True)
df_lab = df_lab[_lab_csn.isin(allpt_csns)].reset_index(drop=True)
print(f"  Dropped rows with no matching All Pt CSN: Imaging={n_img_not_allpt:,}  Lab={n_lab_not_allpt:,}")

df_img["Modality"] = df_img["Procedure Category"].map(PROC_CAT_MAP).fillna("Other")
df_img["_start"] = df_img.apply(lambda r: combine_dt(r["Exam Start Date"], r["Exam Start Time"]), axis=1)
df_img["_end"]   = df_img.apply(lambda r: combine_dt(r["Exam End Date"],   r["Exam End Time"]),   axis=1)

print("Loading brainbodysep.tsv for CT site mapping...")
site_map_df = pd.read_csv(os.path.join(BASE, "brainbodysep.tsv"), sep="\t")
SITE_MAP = dict(zip(site_map_df["Order Type"].str.strip(), site_map_df["Site of Scan"].str.strip()))


def row_sites(order_desc):
    """Physical CT site(s) (NNI/DR) a single original row's Order Description
    maps to, via brainbodysep.tsv token lookup -- computed per original row,
    before any cross-row merging."""
    if is_blank(order_desc):
        return set()
    tokens = [t.strip() for t in str(order_desc).split(";")]
    return {SITE_MAP[t] for t in tokens if t in SITE_MAP}


# Site / Procedure Location (per original row, before any merging): the
# Ordering Question Responses' Performing Location is the canonical source of
# truth. Site is derived from that parsed location (NNI/DR), falling back to
# brainbodysep.tsv only when the Performing Location is empty. Computed pre-
# merge so a merged group can later pick the value belonging to whichever
# original row had the earliest Exam Start (see merge_group below).
LOC_TO_SITE = {
    "National Neuroscience Institute": "NNI",
    "Tan Tock Seng Hospital": "DR",
}


def site_from_performing_location(loc):
    if is_blank(loc):
        return None
    sites = {LOC_TO_SITE[p.strip()] for p in str(loc).split(" + ") if p.strip() in LOC_TO_SITE}
    return next(iter(sites)) if len(sites) == 1 else None


def row_site_single(order_desc):
    sites = row_sites(order_desc)
    return next(iter(sites)) if len(sites) == 1 else None


df_img["_perf_loc_row"] = df_img["Ordering Question Responses"].apply(parse_performing_location)
df_img["_site_from_perf_row"] = df_img["_perf_loc_row"].apply(site_from_performing_location)
df_img["_site_from_tsv_row"] = df_img["Order Description"].apply(row_site_single)
_has_perf_loc = df_img["_perf_loc_row"].apply(lambda v: not is_blank(v))
df_img["_site_row"] = df_img["_site_from_perf_row"].where(_has_perf_loc, df_img["_site_from_tsv_row"])
df_img["_site_row"] = df_img["_site_row"].where(df_img["Procedure Category"] == "IMG CT ORDERABLES")
df_img["_proc_loc_row"] = df_img["_perf_loc_row"].where(_has_perf_loc, df_img["_site_row"])


# ---------------------------------------------------------------------------
# 1a. Build overlap groups per CSN + Procedure Category (transitive
# interval-merge) -- kept separate per category so an overlapping CT and XR
# order on the same CSN are never merged into one row.
# ---------------------------------------------------------------------------
print("Grouping overlapping/identical-window imaging orders per CSN + Procedure Category...")

df_img = df_img.reset_index(drop=True)
df_img["_group_id"] = -1

group_counter = 0
for (csn, proc_cat), idxs in df_img.groupby(["CSN", "Procedure Category"]).groups.items():
    idxs = list(idxs)
    # Rows with no exam window: each is its own group
    have_window = [i for i in idxs if df_img.at[i, "_start"] is not None and df_img.at[i, "_end"] is not None]
    no_window   = [i for i in idxs if i not in have_window]

    for i in no_window:
        df_img.at[i, "_group_id"] = group_counter
        group_counter += 1

    if have_window:
        have_window.sort(key=lambda i: df_img.at[i, "_start"])
        cur_group = [have_window[0]]
        cur_max_end = df_img.at[have_window[0], "_end"]
        for i in have_window[1:]:
            s = df_img.at[i, "_start"]
            if s <= cur_max_end:
                # Overlaps (or identical) with running group -> same session
                cur_group.append(i)
                e = df_img.at[i, "_end"]
                if e > cur_max_end:
                    cur_max_end = e
            else:
                # Sequential -> close out current group, start new one
                for j in cur_group:
                    df_img.at[j, "_group_id"] = group_counter
                group_counter += 1
                cur_group = [i]
                cur_max_end = df_img.at[i, "_end"]
        for j in cur_group:
            df_img.at[j, "_group_id"] = group_counter
        group_counter += 1

group_sizes = df_img.groupby("_group_id").size()
n_merged_groups = (group_sizes > 1).sum()
print(f"  {len(df_img):,} rows -> {df_img['_group_id'].nunique():,} groups "
      f"({n_merged_groups:,} groups merged from 2+ rows)")
print("  Merged-group size distribution:")
for size, cnt in group_sizes[group_sizes > 1].value_counts().sort_index().items():
    print(f"    {size} rows -> {cnt} groups")


# ---------------------------------------------------------------------------
# Merge each group into a single row
# ---------------------------------------------------------------------------
def join_unique(values, sep="; "):
    seen = []
    for v in values:
        if is_blank(v):
            continue
        s = str(v).strip()
        if s not in seen:
            seen.append(s)
    return sep.join(seen)


# Date/Time column pairs that must be merged together (never independently),
# otherwise deduping each side separately can desync date[i] from time[i].
PAIRED_DATETIME_COLS = [
    ("Order Date", "Order Time"),
    ("Order Completion Date", "Order Completion Time"),
    ("Collection Date", "Collection Time"),
    ("Received Date", "Received Time"),
    ("Result Date", "Result Time"),
    ("Parent Order Date", "Parent Order Time"),
    ("Arrival Date", "Arrival Time"),
    ("Bed Ready Date", "Bed Ready Time"),
    ("Bed Assigned Date", "Bed Assigned Time"),
    ("ED Case End Date", "ED Case End Time"),
    ("Admit Dispo Date", "Admit Dispo Time"),
    ("Admit EDTC/U Date", "Admit EDTC/U Time"),
    ("EDTC/U IP Date", "EDTC/U IP Time"),
]

NON_MERGE_COLS = {"_start", "_end", "_group_id",
                   "_perf_loc_row", "_site_row", "_proc_loc_row"}


def merge_paired_datetime(sub, date_col, time_col):
    """Dedupe (date, time) as a single unit so the two output columns stay aligned."""
    pairs = list(zip(sub[date_col].tolist(), sub[time_col].tolist()))
    seen_keys = []
    kept = []
    for d, t in pairs:
        if is_blank(d) and is_blank(t):
            continue
        key = f"{d}|{t}"
        if key not in seen_keys:
            seen_keys.append(key)
            kept.append((d, t))
    if not kept:
        return pairs[0][0], pairs[0][1]
    if len(kept) == 1:
        return kept[0][0], kept[0][1]
    # Join in lockstep (no independent re-dedup) so date[i]/time[i] stay paired.
    date_str = "; ".join(str(d).strip() if not is_blank(d) else "" for d, _ in kept)
    time_str = "; ".join(str(t).strip() if not is_blank(t) else "" for _, t in kept)
    return date_str, time_str


def merge_group(sub):
    if len(sub) == 1:
        row = sub.iloc[0].to_dict()
        row["N Merged Rows"] = 1
        row["Merged Order IDs"] = str(row.get("Order ID", ""))
        row["Procedure Location"] = row.get("_proc_loc_row")
        row["Site"] = row.get("_site_row")
        return row

    paired_cols = {c for pair in PAIRED_DATETIME_COLS for c in pair}

    merged = {}
    for col in sub.columns:
        if col in NON_MERGE_COLS or col in paired_cols:
            continue
        vals = sub[col].tolist()
        non_null = [v for v in vals if not is_blank(v)]
        distinct = set(str(v).strip() for v in non_null)
        if len(distinct) <= 1:
            merged[col] = non_null[0] if non_null else vals[0]
        else:
            merged[col] = join_unique(vals)

    for date_col, time_col in PAIRED_DATETIME_COLS:
        merged[date_col], merged[time_col] = merge_paired_datetime(sub, date_col, time_col)

    starts = [s for s in sub["_start"].tolist() if s is not None]
    ends   = [e for e in sub["_end"].tolist()   if e is not None]
    if starts:
        earliest_start = min(starts)
        merged["Exam Start Date"] = earliest_start.date()
        merged["Exam Start Time"] = earliest_start.time()
    if ends:
        latest_end = max(ends)
        merged["Exam End Date"] = latest_end.date()
        merged["Exam End Time"] = latest_end.time()

    merged["N Merged Rows"] = len(sub)
    merged["Merged Order IDs"] = ", ".join(str(v) for v in sub["Order ID"].tolist())

    # Procedure Location / Site: from whichever original row has the
    # earliest Exam Start -- rows with no Exam Start sort last so they never
    # win over a row that actually has one. This is also how a group whose
    # members span two physical CT sites (NNI/DR) resolves to a single Site.
    sort_key = sub["_start"].apply(lambda s: s if s is not None else datetime.max)
    earliest_idx = sort_key.idxmin()
    merged["Procedure Location"] = sub.loc[earliest_idx, "_proc_loc_row"]
    merged["Site"] = sub.loc[earliest_idx, "_site_row"]
    return merged


print("Merging groups into single rows...")
merged_rows = [merge_group(grp) for _, grp in df_img.groupby("_group_id", sort=False)]
df_img_merged = pd.DataFrame(merged_rows)
print(f"  Imaging: {len(df_img):,} rows -> {len(df_img_merged):,} rows after merge")


# ---------------------------------------------------------------------------
# 1b. Exam Start/End fallback to Order Completion Date/Time
# ---------------------------------------------------------------------------
print("Applying Order Completion fallback for missing Exam Start/End...")

def fallback_exam_time(row):
    start_missing = is_blank(row.get("Exam Start Date")) or is_blank(row.get("Exam Start Time"))
    end_missing   = is_blank(row.get("Exam End Date"))   or is_blank(row.get("Exam End Time"))
    oc_date, oc_time = row.get("Order Completion Date"), row.get("Order Completion Time")
    oc_available = not is_blank(oc_date) and not is_blank(oc_time)

    source = "Exam"
    if start_missing:
        if oc_available:
            row["Exam Start Date"] = oc_date
            row["Exam Start Time"] = oc_time
            source = "Order Completion"
        else:
            source = "Missing" if end_missing else source
    if end_missing:
        if oc_available:
            row["Exam End Date"] = oc_date
            row["Exam End Time"] = oc_time
            if source == "Exam":
                source = "Order Completion"
        else:
            if start_missing:
                source = "Missing"
    row["Exam Time Source"] = source
    return row

df_img_merged = df_img_merged.apply(fallback_exam_time, axis=1)

n_missing = (df_img_merged["Exam Time Source"] == "Missing").sum()
n_oc      = (df_img_merged["Exam Time Source"] == "Order Completion").sum()
print(f"  Exam Time Source: Exam={len(df_img_merged) - n_missing - n_oc:,}  "
      f"Order Completion={n_oc:,}  Missing={n_missing:,}")


# ---------------------------------------------------------------------------
# 1b2. Imaging only: drop Order Completion Date/Time now that its values
# have been folded into Exam Start/End by the fallback above. Lab keeps
# its Order Completion Date/Time untouched (it has no Exam Start/End).
# ---------------------------------------------------------------------------
for col in ("Order Completion Date", "Order Completion Time"):
    if col in df_img_merged.columns:
        df_img_merged.drop(columns=[col], inplace=True)


# ---------------------------------------------------------------------------
# 1c. Drop Order Date/Order Time (both sheets) — Parent Order Date/Time remain
# ---------------------------------------------------------------------------
for df in (df_img_merged, df_lab):
    for col in ("Order Date", "Order Time"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)


# ---------------------------------------------------------------------------
# Column tidy-up: drop helper cols, reorder month first
# ---------------------------------------------------------------------------
for col in ("_start", "_end", "_group_id", "Modality",
            "_perf_loc_row", "_site_from_perf_row", "_site_from_tsv_row",
            "_site_row", "_proc_loc_row"):
    if col in df_img_merged.columns:
        df_img_merged.drop(columns=[col], inplace=True)


# ---------------------------------------------------------------------------
# Drop Imaging rows with no Parent Order time, no Exam Start, or a
# zero/negative-duration exam window (Exam Start vs. Exam End) -- these
# can't be placed on a timeline for queue reconstruction or timing
# analysis, so every downstream script gets a consistently-clean
# population instead of each one filtering it out independently.
#
# Negative wait time (Parent Order vs. Exam Start) is checked CT-only
# (Procedure Category == "IMG CT ORDERABLES") -- wait-to-scan-start is a
# CT queue concept (see build_ct_queue.py); non-CT modalities (XR, MRI,
# etc.) aren't held to it.
# ---------------------------------------------------------------------------
print("Dropping Imaging rows with no Parent Order time / no Exam Start / zero-or-negative exam duration / (CT only) negative wait time...")

_parent_order_dt = df_img_merged.apply(
    lambda r: earliest_parent_order_dt(r["Parent Order Date"], r["Parent Order Time"]), axis=1
)
_exam_start_dt = df_img_merged.apply(lambda r: combine_dt(r["Exam Start Date"], r["Exam Start Time"]), axis=1)
_exam_end_dt = df_img_merged.apply(lambda r: combine_dt(r["Exam End Date"], r["Exam End Time"]), axis=1)
_is_ct = df_img_merged["Procedure Category"] == "IMG CT ORDERABLES"

no_parent_order = _parent_order_dt.isna()
no_exam_start = _exam_start_dt.isna()
zero_duration = _exam_start_dt.notna() & _exam_end_dt.notna() & (_exam_start_dt == _exam_end_dt)
negative_duration = _exam_start_dt.notna() & _exam_end_dt.notna() & (_exam_end_dt < _exam_start_dt)
zero_wait = _is_ct & _parent_order_dt.notna() & _exam_start_dt.notna() & (_parent_order_dt == _exam_start_dt)
negative_wait = _is_ct & _parent_order_dt.notna() & _exam_start_dt.notna() & (_exam_start_dt < _parent_order_dt)
drop_mask = (no_parent_order | no_exam_start | zero_duration | negative_duration | negative_wait).to_numpy()

print(f"  no Parent Order time: {int(no_parent_order.sum()):,}")
print(f"  no Exam Start: {int(no_exam_start.sum()):,}")
print(f"  zero-duration exam window (Exam Start == Exam End): {int(zero_duration.sum()):,}")
print(f"  negative-duration exam window (Exam End < Exam Start): {int(negative_duration.sum()):,}")
print(f"  zero wait time retained, CT only (Parent Order == Exam Start): {int(zero_wait.sum()):,}")
print(f"  negative wait time, CT only (Exam Start < Parent Order): {int(negative_wait.sum()):,}")
print(f"  Total dropped (a row can meet more than one condition): {int(drop_mask.sum()):,} of {len(df_img_merged):,}")

df_img_merged = df_img_merged[~drop_mask].reset_index(drop=True)
_parent_order_dt = _parent_order_dt[~drop_mask].reset_index(drop=True)
_exam_start_dt = _exam_start_dt[~drop_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Wait Time (min): Exam Start minus Parent Order, on the same timestamps used
# to decide which rows survive the drop above -- so it's always well-defined
# for every remaining row.
# ---------------------------------------------------------------------------
df_img_merged["Wait Time (min)"] = (_exam_start_dt - _parent_order_dt).dt.total_seconds() / 60


# ---------------------------------------------------------------------------
# Site: already assigned in merge_group above (tsv-derived physical CT site
# of whichever original row had the earliest Exam Start).
# ---------------------------------------------------------------------------
n_site = df_img_merged["Site"].notna().sum()
print(f"  Site assigned (CT rows): {n_site:,} of {len(df_img_merged):,}  "
      f"({df_img_merged['Site'].value_counts().to_dict()})")


# ---------------------------------------------------------------------------
# Scan Priority / Performing Location: both parsed from "Ordering Question
# Responses" (same patterns as build_ct_queue.py / nni_scan_priority_
# location_check.py), so downstream scripts no longer need to re-derive them.
# ---------------------------------------------------------------------------
df_img_merged["Scan Priority"] = df_img_merged["Ordering Question Responses"].apply(parse_priority)
df_img_merged["Performing Location"] = df_img_merged["Ordering Question Responses"].apply(parse_performing_location)
print(f"  Scan Priority assigned: {(df_img_merged['Scan Priority'] != 'Unknown').sum():,} of {len(df_img_merged):,} "
      f"({df_img_merged['Scan Priority'].value_counts().to_dict()})")
print(f"  Performing Location captured: {df_img_merged['Performing Location'].notna().sum():,} of {len(df_img_merged):,}")


def reorder_month_first(df):
    cols = ["month"] + [c for c in df.columns if c != "month"]
    return df[cols]

df_img_merged = reorder_month_first(df_img_merged)
df_lab        = reorder_month_first(df_lab)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
print("Writing merged/ct_orders_cleaned.xlsx...")
with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    df_img_merged.to_excel(writer, sheet_name="Imaging", index=False)
    df_lab.to_excel(writer, sheet_name="Lab", index=False)

size_kb = os.path.getsize(OUT_PATH) // 1024
print(f"  -> ct_orders_cleaned.xlsx  (Imaging: {len(df_img_merged):,} rows  |  "
      f"Lab: {len(df_lab):,} rows)  [{size_kb} KB]")
print("Done.")
