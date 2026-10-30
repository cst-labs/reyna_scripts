"""
clean_ct_orders_newdata.py
Builds a merged CT-orders file for the NEW data (Jul/Aug/Sep 2026), analogous
to clean_ct_orders.py but scoped to just what was requested so far -- a
deliberately smaller sequence than the old-data pipeline:

  Both sheets (Imaging, Lab):
    - The 3 months of data/newdata CT & Lab Orders are combined into one
      file (month-tagged).
    - Rows whose CSN has no matching entry in that same month's All Pt are
      dropped (same cross-check as clean_ct_orders.py / Sheet 6 of
      analysis/newdata_report.xlsx).

  Imaging only:
    - 'Performing Location' == 'TTSH EMERGENCY [106003000]' is replaced with
      'TTSH DDR-EMERGENCY [106003002]' before merging.
    - Rows whose Exam Start/End windows overlap or are identical (same CSN
      AND same Procedure Category) are merged into one row -- same
      transitive interval-merge as clean_ct_orders.py (see Sheet 7). The
      merge key is (CSN, Procedure Category), NOT Performing Location, so
      rows from different Performing Locations can be merged together.
    - For merged groups with conflicting 'Performing Location' values, the
      location from whichever member row has the earliest Exam Start wins --
      same tie-break rule as clean_ct_orders.py. The losing location value
      is discarded (not joined with '; '). No 'Site' (NNI/DR) is assigned
      at this stage -- that's deferred.

    - Rows with zero exam duration (Exam Start == Exam End) or zero/negative
      CT wait time (Exam Start <= Parent Order) are NOT dropped -- flagged
      instead and logged to a 'Flagged Rows' sheet in the output file.

  NOT done yet (deferred, not part of this pass):
    - Exam Start/End -> Order Completion fallback
    - Order Date/Order Time column drops
    - the no-Parent-Order-time / no-Exam-Start / negative-duration
      drop-conditions
    - Site assignment

Writes merged/ct_orders_newdata_merged.xlsx (new file; does not touch any
existing file under merged/ or data/newdata/).
"""
import os
import math
from datetime import datetime, timedelta, date, time as time_type

import openpyxl
import pandas as pd

BASE    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEWDATA = os.path.join(BASE, "data", "newdata")
MERGED  = os.path.join(BASE, "merged")
OUT_PATH = os.path.join(MERGED, "ct_orders_newdata_merged.xlsx")

NEW_MONTHS = [
    ("Jul 2026", "2026 7 All Pt (decrypted).xlsx", "Sheet2",
                 "2026 7 CT & Lab Orders (decrypted).xlsx", "Jul Imaging", "Jul Lab"),
    ("Aug 2026", "2026 8 All Pt (decrypted).xlsx", "Sheet1",
                 "2026 8 CT & Lab Orders (decrypted).xlsx", "Aug Imaging", "Aug Lab"),
    ("Sep 2026", "2026 9 All Pt (decrypted).xlsx", "Sheet1",
                 "2026 9 CT & Lab Orders (decrypted).xlsx", "Sep Imaging", "Sep Lab"),
]

TTSH_EMERGENCY_OLD = "TTSH EMERGENCY [106003000]"
TTSH_EMERGENCY_NEW = "TTSH DDR-EMERGENCY [106003002]"


# ---------------------------------------------------------------------------
# Helpers (same pattern used throughout this session's new-data scripts)
# ---------------------------------------------------------------------------
def wb_to_df(path, sheet):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    seen, uh = {}, []
    for h in headers:
        if h in seen:
            seen[h] += 1
            uh.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            uh.append(h)
    return pd.DataFrame(rows[1:], columns=uh)


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


def csn_norm(v):
    if is_blank(v):
        return ""
    if isinstance(v, float):
        return str(int(v))
    return str(v).strip()


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
            ts = round(f * 86400)
            h, r = divmod(ts, 3600)
            m, s = divmod(r, 60)
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


def split_joined(val):
    """A merged row's Parent Order Date/Time can be a '; '-joined multi-value
    string. Only split when val is actually a joined string."""
    if is_blank(val):
        return [None]
    if isinstance(val, str) and "; " in val:
        return val.split("; ")
    return [val]


def earliest_parent_order_dt(date_val, time_val):
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


def join_unique(values, sep="; "):
    seen = []
    for v in values:
        if is_blank(v):
            continue
        s = str(v).strip()
        if s not in seen:
            seen.append(s)
    return sep.join(seen)


# ---------------------------------------------------------------------------
# Load + combine the 3 months
# ---------------------------------------------------------------------------
print("Loading new data (Jul/Aug/Sep 2026)...")
all_ap, all_img, all_lab = [], [], []
for label, ap_fname, ap_sheet, ct_fname, img_sheet, lab_sheet in NEW_MONTHS:
    df_ap = wb_to_df(os.path.join(NEWDATA, ap_fname), ap_sheet)
    df_ap["month"] = label
    all_ap.append(df_ap)

    df_img = wb_to_df(os.path.join(NEWDATA, ct_fname), img_sheet)
    df_img["month"] = label
    all_img.append(df_img)

    df_lab = wb_to_df(os.path.join(NEWDATA, ct_fname), lab_sheet)
    df_lab["month"] = label
    all_lab.append(df_lab)

df_ap = pd.concat(all_ap, ignore_index=True)
df_img = pd.concat(all_img, ignore_index=True)
df_lab = pd.concat(all_lab, ignore_index=True)
print(f"  All Pt: {len(df_ap):,} rows  |  Imaging: {len(df_img):,} rows  |  Lab: {len(df_lab):,} rows")


# ---------------------------------------------------------------------------
# CSN cross-check drop (both sheets) -- matches Sheet 6 / clean_ct_orders.py:
# CSN just needs to appear in All Pt across the combined 3 months, not
# necessarily the same month's All Pt.
# ---------------------------------------------------------------------------
print("Dropping rows with no matching All Pt CSN...")
allpt_csns = set(df_ap["CSN"].dropna().apply(csn_norm)) - {""}

_img_csn = df_img["CSN"].apply(csn_norm)
_lab_csn = df_lab["CSN"].apply(csn_norm)
print(f"  Imaging: dropping {int((~_img_csn.isin(allpt_csns)).sum()):,} of {len(df_img):,} rows")
print(f"  Lab: dropping {int((~_lab_csn.isin(allpt_csns)).sum()):,} of {len(df_lab):,} rows")

df_img = df_img[_img_csn.isin(allpt_csns)].reset_index(drop=True)
df_lab = df_lab[_lab_csn.isin(allpt_csns)].reset_index(drop=True)
print(f"  -> Imaging: {len(df_img):,} rows  |  Lab: {len(df_lab):,} rows")


# ---------------------------------------------------------------------------
# TTSH EMERGENCY -> TTSH DDR-EMERGENCY, on Performing Location (pre-merge)
# ---------------------------------------------------------------------------
n_replaced = int((df_img["Performing Location"] == TTSH_EMERGENCY_OLD).sum())
df_img["Performing Location"] = df_img["Performing Location"].replace(
    TTSH_EMERGENCY_OLD, TTSH_EMERGENCY_NEW
)
print(f"Replaced 'TTSH EMERGENCY' -> 'TTSH DDR-EMERGENCY' in Performing Location: {n_replaced:,} rows")


# ---------------------------------------------------------------------------
# Overlap-merge Imaging rows per (CSN, Procedure Category) -- same transitive
# interval-merge as clean_ct_orders.py. Matches Sheet 7.
# ---------------------------------------------------------------------------
print("Grouping overlapping/identical-window imaging orders per CSN + Procedure Category...")

df_img["_start"] = df_img.apply(lambda r: combine_dt(r["Exam Start Date"], r["Exam Start Time"]), axis=1)
df_img["_end"]   = df_img.apply(lambda r: combine_dt(r["Exam End Date"],   r["Exam End Time"]),   axis=1)

df_img = df_img.reset_index(drop=True)
df_img["_group_id"] = -1
group_counter = 0
for (csn, proc_cat), idxs in df_img.groupby(["CSN", "Procedure Category"]).groups.items():
    idxs = list(idxs)
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
                cur_group.append(i)
                e = df_img.at[i, "_end"]
                if e > cur_max_end:
                    cur_max_end = e
            else:
                for j in cur_group:
                    df_img.at[j, "_group_id"] = group_counter
                group_counter += 1
                cur_group = [i]
                cur_max_end = df_img.at[i, "_end"]
        for j in cur_group:
            df_img.at[j, "_group_id"] = group_counter
        group_counter += 1

group_sizes = df_img.groupby("_group_id").size()
n_merged_groups = int((group_sizes > 1).sum())
print(f"  {len(df_img):,} rows -> {df_img['_group_id'].nunique():,} groups "
      f"({n_merged_groups:,} groups merged from 2+ rows)")
for size, cnt in group_sizes[group_sizes > 1].value_counts().sort_index().items():
    print(f"    {size} rows -> {cnt} groups")


# ---------------------------------------------------------------------------
# Merge each group into a single row. Grouping key is (CSN, Procedure
# Category) + overlapping/identical exam windows -- NOT Performing Location,
# so two rows from different locations can land in the same group.
# 'Performing Location' is excluded from the generic distinct-value/join
# handling below and resolved on its own: the value from whichever member
# row has the earliest Exam Start wins (same tie-break clean_ct_orders.py
# uses for Procedure Location/Site) -- conflicting locations are NOT joined
# with '; '. No Site (NNI/DR) is assigned at this stage.
# ---------------------------------------------------------------------------
PAIRED_DATETIME_COLS = [
    pair for pair in [
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
    ] if pair[0] in df_img.columns and pair[1] in df_img.columns
]
NON_MERGE_COLS = {"_start", "_end", "_group_id"}


def merge_paired_datetime(sub, date_col, time_col):
    pairs = list(zip(sub[date_col].tolist(), sub[time_col].tolist()))
    seen_keys, kept = [], []
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
    date_str = "; ".join(str(d).strip() if not is_blank(d) else "" for d, _ in kept)
    time_str = "; ".join(str(t).strip() if not is_blank(t) else "" for _, t in kept)
    return date_str, time_str


def merge_group(sub):
    if len(sub) == 1:
        row = sub.iloc[0].to_dict()
        row["N Merged Rows"] = 1
        row["Merged Order IDs"] = str(row.get("Order ID", ""))
        return row

    paired_cols = {c for pair in PAIRED_DATETIME_COLS for c in pair}

    merged = {}
    for col in sub.columns:
        if col in NON_MERGE_COLS or col in paired_cols or col == "Performing Location":
            continue
        vals = sub[col].tolist()
        non_null = [v for v in vals if not is_blank(v)]
        distinct = set(str(v).strip() for v in non_null)
        if len(distinct) <= 1:
            merged[col] = non_null[0] if non_null else vals[0]
        else:
            merged[col] = join_unique(vals)

    # Performing Location: from whichever member row has the earliest Exam
    # Start -- same tie-break clean_ct_orders.py uses for Procedure
    # Location/Site. Rows with no Exam Start sort last so they never win.
    sort_key = sub["_start"].apply(lambda s: s if s is not None else datetime.max)
    earliest_idx = sort_key.idxmin()
    merged["Performing Location"] = sub.loc[earliest_idx, "Performing Location"]

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
    return merged


print("Merging groups into single rows...")
merged_rows = [merge_group(grp) for _, grp in df_img.groupby("_group_id", sort=False)]
df_img_merged = pd.DataFrame(merged_rows)
print(f"  Imaging: {len(df_img):,} rows -> {len(df_img_merged):,} rows after merge")

n_loc_conflicts = sum(
    1 for _, grp in df_img.groupby("_group_id", sort=False)
    if len(grp) > 1 and grp["Performing Location"].dropna().apply(lambda v: str(v).strip()).nunique() > 1
)
print(f"  Merged groups with conflicting Performing Location (resolved to earliest Exam Start): {n_loc_conflicts:,}")

for col in ("_start", "_end", "_group_id"):
    if col in df_img_merged.columns:
        df_img_merged.drop(columns=[col], inplace=True)


# ---------------------------------------------------------------------------
# Flag (don't drop) rows with zero exam duration or zero/negative CT wait
# time -- same conditions clean_ct_orders.py drops, but logged here instead.
# ---------------------------------------------------------------------------
print("Flagging (not dropping) rows with zero exam duration / zero-or-negative CT wait time...")

_exam_start_dt = df_img_merged.apply(lambda r: combine_dt(r["Exam Start Date"], r["Exam Start Time"]), axis=1)
_exam_end_dt = df_img_merged.apply(lambda r: combine_dt(r["Exam End Date"], r["Exam End Time"]), axis=1)
_parent_order_dt = df_img_merged.apply(
    lambda r: earliest_parent_order_dt(r["Parent Order Date"], r["Parent Order Time"]), axis=1
)
_is_ct = df_img_merged["Procedure Category"] == "IMG CT ORDERABLES"

zero_duration = _exam_start_dt.notna() & _exam_end_dt.notna() & (_exam_start_dt == _exam_end_dt)
zero_or_neg_wait = _is_ct & _parent_order_dt.notna() & _exam_start_dt.notna() & (_exam_start_dt <= _parent_order_dt)

df_img_merged["Flag: Zero Exam Duration"] = zero_duration
df_img_merged["Flag: Zero/Negative CT Wait Time"] = zero_or_neg_wait

print(f"  Zero-duration exam window (Exam Start == Exam End): {int(zero_duration.sum()):,}")
print(f"  Zero/negative CT wait time (Exam Start <= Parent Order): {int(zero_or_neg_wait.sum()):,}")

flagged_mask = zero_duration | zero_or_neg_wait
df_flagged = df_img_merged[flagged_mask].copy()
print(f"  Total flagged rows (not dropped): {int(flagged_mask.sum()):,} of {len(df_img_merged):,}")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
def reorder_month_first(df):
    cols = ["month"] + [c for c in df.columns if c != "month"]
    return df[cols]


df_img_merged = reorder_month_first(df_img_merged)
df_lab        = reorder_month_first(df_lab)
df_flagged    = reorder_month_first(df_flagged)

print(f"Writing {os.path.basename(OUT_PATH)}...")
with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    df_img_merged.to_excel(writer, sheet_name="Imaging", index=False)
    df_lab.to_excel(writer, sheet_name="Lab", index=False)
    df_flagged.to_excel(writer, sheet_name="Flagged Rows", index=False)

size_kb = os.path.getsize(OUT_PATH) // 1024
print(f"  -> {os.path.basename(OUT_PATH)}  (Imaging: {len(df_img_merged):,} rows  |  "
      f"Lab: {len(df_lab):,} rows)  [{size_kb} KB]")
print("Done.")
