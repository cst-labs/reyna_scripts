"""
patient_timeline.py
Builds a per-patient chronological event trace, from ED arrival through
the START of the (last-ordered) CT exam, for every CSN that has at least
one CT order. Lab tests and non-CT imaging orders in that window are
included alongside the CT events.

Reads:
  merged/all_pt_combined.xlsx    (All Patients sheet)
  merged/ct_orders_cleaned.xlsx  (Imaging + Lab sheets, from clean_ct_orders.py)

Writes:
  analysis/patient_timeline.xlsx
    Timeline      — one row per timestamped event, sorted by CSN then time
    Patient Index — one row per CSN: summary + CT Exam Start Time

Events with a missing timestamp are kept (flagged, sorted last within their
CSN). Events after a patient's CT Exam Start Time are kept (flagged), not
dropped. When a CSN has multiple CT orders, CT Exam Start Time is the LATEST
of their Exam Start times (i.e. the start of the last-ordered CT).
"""

import os
import math
from datetime import datetime, timedelta, date, time as time_type

import openpyxl
import pandas as pd

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
OUT    = os.path.join(BASE, "analysis")
OUT_PATH = os.path.join(OUT, "patient_timeline.xlsx")


# ---------------------------------------------------------------------------
# Helpers (same parsing as clean_ct_orders.py)
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
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
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
            ts = round(f * 86400); h, r = divmod(ts, 3600); m, s = divmod(r, 60)
            return time_type(h % 24, m, s)
    except Exception:
        pass
    return None


def combine_dt(d_val, t_val):
    """Combine a date value with a time value. If either is a ';'-joined
    string (from a merged imaging row), take the FIRST segment only —
    downstream event rows are per-source-row already, so multi-value
    strings here mean 'row represents multiple orders'; earliest is used
    for chronological placement."""
    if isinstance(d_val, str) and ";" in d_val:
        d_val = d_val.split(";")[0].strip()
    if isinstance(t_val, str) and ";" in t_val:
        t_val = t_val.split(";")[0].strip()
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


def csn_key(v):
    return str(v).strip()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading merged/all_pt_combined.xlsx ...")
df_ap = load_sheet(os.path.join(MERGED, "all_pt_combined.xlsx"), "All Patients")
print(f"  All Pt: {len(df_ap):,} rows")

print("Loading merged/ct_orders_cleaned.xlsx ...")
df_img = load_sheet(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), "Imaging")
df_lab = load_sheet(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), "Lab")
print(f"  Imaging: {len(df_img):,} rows  |  Lab: {len(df_lab):,} rows")

df_ap["CSN_key"]  = df_ap["CSN"].apply(csn_key)
df_img["CSN_key"] = df_img["CSN"].apply(csn_key)
df_lab["CSN_key"] = df_lab["CSN"].apply(csn_key)


# ---------------------------------------------------------------------------
# 2a. Scope: CSNs with at least one CT order; compute CT Exam Start Time
# ---------------------------------------------------------------------------
print("Identifying CT orders and computing CT Exam Start Time per patient...")

def has_ct(proc_cat):
    return "IMG CT ORDERABLES" in str(proc_cat) if not is_blank(proc_cat) else False

df_img["_is_ct"] = df_img["Procedure Category"].apply(has_ct)
df_ct = df_img[df_img["_is_ct"]].copy()

df_ct["_start_dt"] = df_ct.apply(
    lambda r: combine_dt(r["Exam Start Date"], r["Exam Start Time"]), axis=1)

ct_scope_csns = set(df_ct["CSN_key"].unique())
print(f"  CSNs with >=1 CT order: {len(ct_scope_csns):,}")

ct_start = (df_ct.groupby("CSN_key")["_start_dt"]
            .max()  # latest of the CT orders' Exam Start times, per CSN
            .rename("ct_start_dt")
            .reset_index())
n_incomplete = ct_start["ct_start_dt"].isna().sum()
print(f"  CT Exam Start Time computed for {len(ct_start) - n_incomplete:,} CSNs; "
      f"{n_incomplete:,} flagged Incomplete (no CT Exam Start Date/Time)")

ct_start_map = dict(zip(ct_start["CSN_key"], ct_start["ct_start_dt"]))


# ---------------------------------------------------------------------------
# 2b. Build the long event table
# ---------------------------------------------------------------------------
print("Building event rows...")

events = []

def add_event(csn, month, event_time, category, event_type, description="",
              modality="", order_id=""):
    events.append({
        "CSN": csn,
        "Month": month,
        "Event Time": event_time,
        "Event Category": category,
        "Event Type": event_type,
        "Description": description,
        "Modality": modality,
        "Order ID": order_id,
    })

# --- All Pt events (per CSN, in scope only) ---
ap_scope = df_ap[df_ap["CSN_key"].isin(ct_scope_csns)]
AP_EVENTS = [
    ("Arrival",           "Arrival",             "Arrival Date",           "Arrival Time"),
    ("ED",                "ED Disposition",       "ED Dispo Last Date",     "ED Dispo Last Time"),
    ("ED",                "Bed Assigned",          "Bed Assigned Date",      "Bed Assigned Time"),
    ("ED",                "Depart ED",             "Depart Date",            "Depart Time"),
    ("ED",                "Hospital Discharge",    "Hospital Discharge Date","Hospital Discharge Time"),
]
for _, row in ap_scope.iterrows():
    csn = row["CSN_key"]
    month = row.get("month", "")
    for category, etype, dcol, tcol in AP_EVENTS:
        dt = combine_dt(row.get(dcol), row.get(tcol))
        add_event(csn, month, dt, category, etype)

# Imaging and Lab sheets no longer carry their own 'month' column (trimmed
# per keep-list) — Month is looked up from All Pt via CSN instead.
month_by_csn = dict(zip(df_ap["CSN_key"], df_ap.get("month", pd.Series(dtype=object))))

# --- Imaging events (any modality, in-scope CSNs only) ---
# 'Read/Ackn on' is no longer in the Imaging keep-list, so no Read/Ack event
# is generated. Order ID is used directly ('Merged Order IDs' was dropped).
img_scope = df_img[df_img["CSN_key"].isin(ct_scope_csns)]
for _, row in img_scope.iterrows():
    csn = row["CSN_key"]
    month = month_by_csn.get(csn, "")
    desc = row.get("Order Description", "")
    modality = "CT" if row["_is_ct"] else str(row.get("Procedure Category", ""))
    order_id = row.get("Order ID", "")

    order_dt = combine_dt(row.get("Parent Order Date"), row.get("Parent Order Time"))
    add_event(csn, month, order_dt, "Imaging", "Order Placed", desc, modality, order_id)

    start_dt = combine_dt(row.get("Exam Start Date"), row.get("Exam Start Time"))
    add_event(csn, month, start_dt, "Imaging", "Exam Start", desc, modality, order_id)

    end_dt = combine_dt(row.get("Exam End Date"), row.get("Exam End Time"))
    add_event(csn, month, end_dt, "Imaging", "Exam End", desc, modality, order_id)

    result_dt = combine_dt(row.get("Result Date"), row.get("Result Time"))
    add_event(csn, month, result_dt, "Imaging", "Result", desc, modality, order_id)

# --- Lab events (in-scope CSNs only) ---
lab_scope = df_lab[df_lab["CSN_key"].isin(ct_scope_csns)]
for _, row in lab_scope.iterrows():
    csn = row["CSN_key"]
    month = month_by_csn.get(csn, "")
    desc = row.get("Order Description", "")
    order_id = row.get("Order ID", "")

    order_dt = combine_dt(row.get("Parent Order Date"), row.get("Parent Order Time"))
    add_event(csn, month, order_dt, "Lab", "Order Placed", desc, "", order_id)

    coll_dt = combine_dt(row.get("Collection Date"), row.get("Collection Time"))
    add_event(csn, month, coll_dt, "Lab", "Collection", desc, "", order_id)

    recv_dt = combine_dt(row.get("Received Date"), row.get("Received Time"))
    add_event(csn, month, recv_dt, "Lab", "Received", desc, "", order_id)

    result_dt = combine_dt(row.get("Result Date"), row.get("Result Time"))
    add_event(csn, month, result_dt, "Lab", "Result", desc, "", order_id)

df_events = pd.DataFrame(events)
print(f"  Total event rows: {len(df_events):,}  across {df_events['CSN'].nunique():,} CSNs")


# ---------------------------------------------------------------------------
# 2c. Flag missing time / after-CT-done; sort
# ---------------------------------------------------------------------------
print("Flagging and sorting...")

df_events["Time Missing"] = df_events["Event Time"].isna()

df_events["_ct_start"] = df_events["CSN"].map(ct_start_map)
df_events["After CT Exam Start"] = df_events.apply(
    lambda r: (pd.notna(r["Event Time"]) and pd.notna(r["_ct_start"])
               and r["Event Time"] > r["_ct_start"]),
    axis=1)
df_events.drop(columns=["_ct_start"], inplace=True)

# Sort key: missing-time rows sort last within their CSN block
SORT_MAX = datetime(9999, 12, 31)
df_events["_sort_time"] = df_events["Event Time"].apply(lambda v: v if pd.notna(v) else SORT_MAX)
df_events = df_events.sort_values(["CSN", "_sort_time"]).drop(columns=["_sort_time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2d. Patient Index sheet
# ---------------------------------------------------------------------------
print("Building Patient Index...")

ap_idx = ap_scope.set_index("CSN_key")

index_rows = []
n_img_by_csn = img_scope.groupby("CSN_key").size()
n_lab_by_csn = lab_scope.groupby("CSN_key").size()
n_events_by_csn = df_events.groupby("CSN").size()

for csn in sorted(ct_scope_csns):
    ap_row = ap_idx.loc[csn] if csn in ap_idx.index else None
    if isinstance(ap_row, pd.DataFrame):
        ap_row = ap_row.iloc[0]
    ct_start_dt = ct_start_map.get(csn)
    index_rows.append({
        "CSN": csn,
        "Month": ap_row["month"] if ap_row is not None else "",
        "Acuity": ap_row.get("Triage Acuity", "") if ap_row is not None else "",
        "ED Disposition": ap_row.get("ED Disposition", "") if ap_row is not None else "",
        "Arrival Time": combine_dt(ap_row.get("Arrival Date"), ap_row.get("Arrival Time")) if ap_row is not None else None,
        "CT Exam Start Time": ct_start_dt if pd.notna(ct_start_dt) else "Incomplete",
        "N Events": int(n_events_by_csn.get(csn, 0)),
        "N Imaging Orders": int(n_img_by_csn.get(csn, 0)),
        "N Lab Orders": int(n_lab_by_csn.get(csn, 0)),
    })

df_index = pd.DataFrame(index_rows)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
print("Writing analysis/patient_timeline.xlsx...")
os.makedirs(OUT, exist_ok=True)
with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    df_events.to_excel(writer, sheet_name="Timeline", index=False)
    df_index.to_excel(writer, sheet_name="Patient Index", index=False)

size_kb = os.path.getsize(OUT_PATH) // 1024
print(f"  -> patient_timeline.xlsx  (Timeline: {len(df_events):,} rows  |  "
      f"Patient Index: {len(df_index):,} CSNs)  [{size_kb} KB]")

print("\n=== Summary ===")
print(f"  CSNs in scope (>=1 CT order):            {len(ct_scope_csns):,}")
print(f"  CSNs flagged Incomplete (no CT exam start): {n_incomplete:,}")
print(f"  Event rows with missing time:             {df_events['Time Missing'].sum():,}")
print(f"  Event rows flagged After CT Exam Start:   {df_events['After CT Exam Start'].sum():,}")
print("Done.")
