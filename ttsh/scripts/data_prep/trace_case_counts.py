from __future__ import annotations

import math
import re
from datetime import date, datetime, time as time_type, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter


BASE = Path(__file__).resolve().parents[2]
MERGED = BASE / "merged"
ANALYSIS = BASE / "analysis"
DATA = BASE / "data"
ANALYSIS.mkdir(exist_ok=True)

NNI = "National Neuroscience Institute"
TTSH = "Tan Tock Seng Hospital"
MIXED_NNI_TTSH = f"{NNI} + {TTSH}"
LOC_PATTERN = re.compile(r"Performing Location:\s*([^\n;]+)")
LOC_TO_SITE = {
    NNI: "NNI",
    TTSH: "DR",
}


def is_blank(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.strip() == ""


def to_dt(value):
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in (
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return datetime(1899, 12, 30) + timedelta(days=f)
    except Exception:
        return None


def to_time(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, time_type):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"):
            try:
                return datetime.strptime(text, fmt).time()
            except ValueError:
                continue
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        if 0 <= f < 2:
            seconds = round(f * 86400)
            hour, rest = divmod(seconds, 3600)
            minute, second = divmod(rest, 60)
            return time_type(hour % 24, minute, second)
    except Exception:
        pass
    return None


def combine_dt(date_value, time_value):
    d = to_dt(date_value)
    if d is None:
        return pd.NaT
    t = to_time(time_value)
    if t is None:
        return pd.Timestamp(d)
    return pd.Timestamp(datetime(d.year, d.month, d.day, t.hour, t.minute, t.second))


def fallback_exam_time(row):
    start_missing = is_blank(row.get("Exam Start Date")) or is_blank(row.get("Exam Start Time"))
    end_missing = is_blank(row.get("Exam End Date")) or is_blank(row.get("Exam End Time"))
    oc_available = not is_blank(row.get("Order Completion Date")) and not is_blank(
        row.get("Order Completion Time")
    )

    source = "Exam"
    if start_missing:
        if oc_available:
            row["Exam Start Date"] = row.get("Order Completion Date")
            row["Exam Start Time"] = row.get("Order Completion Time")
            source = "Order Completion"
        else:
            source = "Missing" if end_missing else source
    if end_missing:
        if oc_available:
            row["Exam End Date"] = row.get("Order Completion Date")
            row["Exam End Time"] = row.get("Order Completion Time")
            if source == "Exam":
                source = "Order Completion"
        elif start_missing:
            source = "Missing"
    row["Exam Time Source"] = source
    return row


def parse_performing_location(value):
    if is_blank(value):
        return None
    locs = sorted({match.strip() for match in LOC_PATTERN.findall(str(value))})
    return " + ".join(locs) if locs else None


def sites_from_performing_location(loc):
    if is_blank(loc):
        return set()
    return {LOC_TO_SITE[p.strip()] for p in str(loc).split(" + ") if p.strip() in LOC_TO_SITE}


def pct(numerator, denominator):
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return 100 * numerator / denominator


def summarize(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "N": len(values),
        "Mean": round(values.mean(), 2) if len(values) else np.nan,
        "Median": round(values.median(), 1) if len(values) else np.nan,
        "P25": round(values.quantile(0.25), 1) if len(values) else np.nan,
        "P75": round(values.quantile(0.75), 1) if len(values) else np.nan,
        "P90": round(values.quantile(0.90), 1) if len(values) else np.nan,
        "P95": round(values.quantile(0.95), 1) if len(values) else np.nan,
        "Max": round(values.max(), 1) if len(values) else np.nan,
    }


def exam_time_source_breakdown(df, scope):
    start_present = ~(df["Exam Start Date"].apply(is_blank) | df["Exam Start Time"].apply(is_blank))
    end_present = ~(df["Exam End Date"].apply(is_blank) | df["Exam End Time"].apply(is_blank))
    exam_both = start_present & end_present
    oc_present = ~(df["Order Completion Date"].apply(is_blank) | df["Order Completion Time"].apply(is_blank))
    filled_from_completion = ~exam_both & oc_present
    neither = ~exam_both & ~oc_present
    return pd.DataFrame(
        [
            {
                "scope": scope,
                "category": "Already has non-null Exam Start and Exam End",
                "rows": int(exam_both.sum()),
                "patients": df.loc[exam_both, "CSN_key"].nunique(),
            },
            {
                "scope": scope,
                "category": "Exam Start/End filled from Order Completion",
                "rows": int(filled_from_completion.sum()),
                "patients": df.loc[filled_from_completion, "CSN_key"].nunique(),
            },
            {
                "scope": scope,
                "category": "Neither complete Exam Start/End nor Order Completion",
                "rows": int(neither.sum()),
                "patients": df.loc[neither, "CSN_key"].nunique(),
            },
            {"scope": scope, "category": "Total", "rows": len(df), "patients": df["CSN_key"].nunique()},
        ]
    )


def auto_col_width(writer):
    for sheet in writer.sheets.values():
        for col in sheet.columns:
            max_len = max((len(str(cell.value)) for cell in col if cell.value is not None), default=0)
            sheet.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 2, 70)


def order_ids(value):
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = [item.strip() for item in text.replace(";", ",").split(",")]
    return [item.removesuffix(".0") for item in parts if item]


def order_key(row):
    ids = order_ids(row.get("Merged Order IDs", "")) or order_ids(row.get("Order ID", ""))
    return "|".join(sorted(set(ids), key=lambda item: (len(item), item)))


def compute_final_scan_duration(df):
    start = pd.to_datetime(
        pd.to_datetime(df["Exam Start Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        + " "
        + df["Exam Start Time"].astype(str).str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)", expand=False),
        errors="coerce",
    )
    end = pd.to_datetime(
        pd.to_datetime(df["Exam End Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        + " "
        + df["Exam End Time"].astype(str).str.extract(r"(\d{1,2}:\d{2}(?::\d{2})?)", expand=False),
        errors="coerce",
    )
    timeline = pd.DataFrame({"start": start, "end": end, "key": df["_order_key"]}, index=df.index)
    timeline = timeline.sort_values(["start", "key"], kind="stable")
    timeline["prev_end"] = timeline["end"].shift(1)
    timeline["next_start"] = timeline["start"].shift(-1)

    exam_duration = (timeline["end"] - timeline["start"]).dt.total_seconds() / 60
    next_start_duration = (timeline["next_start"] - timeline["start"]).dt.total_seconds() / 60
    prev_gap = (timeline["start"] - timeline["prev_end"]).dt.total_seconds() / 60

    valid_exam = exam_duration.notna() & (exam_duration >= 0)
    valid_next = next_start_duration.notna() & (next_start_duration >= 0)
    old_picks_exam = valid_exam & (~valid_next | (exam_duration <= next_start_duration))
    add_gap_buffer = old_picks_exam & prev_gap.notna() & (prev_gap > 0)

    duration = next_start_duration.copy()
    duration.loc[add_gap_buffer] = exam_duration.loc[add_gap_buffer] + 17.9
    no_valid_next = ~valid_next & valid_exam
    duration.loc[no_valid_next & ~add_gap_buffer] = exam_duration.loc[no_valid_next & ~add_gap_buffer]
    branch = pd.Series("next exam start - exam start", index=timeline.index)
    branch.loc[add_gap_buffer] = "exam end - exam start + 17.9"
    branch.loc[no_valid_next & ~add_gap_buffer] = "boundary: exam end - exam start"
    return duration.reindex(df.index), branch.reindex(df.index)


def build_site_conflict_stats(ct_after_allpt):
    df = ct_after_allpt.apply(fallback_exam_time, axis=1)
    df["_start"] = df.apply(lambda r: combine_dt(r["Exam Start Date"], r["Exam Start Time"]), axis=1)
    df["_end"] = df.apply(lambda r: combine_dt(r["Exam End Date"], r["Exam End Time"]), axis=1)
    df["_performing_location_row"] = df["Ordering Question Responses"].apply(parse_performing_location)
    df["_sites"] = df["_performing_location_row"].apply(sites_from_performing_location)
    valid = df[df["_start"].notna() & df["_end"].notna()].copy()

    groups = []
    gid = 0
    for (csn, proc_cat), grp in valid.groupby(["CSN_key", "Procedure Category"], dropna=False):
        grp = grp.sort_values("_start", kind="stable")
        current = []
        current_end = None
        for _, row in grp.iterrows():
            if current and row["_start"] <= current_end:
                current.append(row)
                current_end = max(current_end, row["_end"])
            else:
                if len(current) > 1:
                    gid += 1
                    groups.append((gid, current))
                current = [row]
                current_end = row["_end"]
        if len(current) > 1:
            gid += 1
            groups.append((gid, current))

    rows = []
    for gid, members in groups:
        descriptions = {str(row["Order Description"]).strip() for row in members if not is_blank(row["Order Description"])}
        union_sites = set()
        starts = []
        for row in members:
            union_sites |= row["_sites"]
            starts.append(row["_start"])
        if len(descriptions) > 1 and len(union_sites) > 1:
            earliest = min(starts)
            earliest_members = [row for row in members if row["_start"] == earliest]
            earliest_sites = set()
            for row in earliest_members:
                earliest_sites |= row["_sites"]
            rows.append(
                {
                    "group_id": gid,
                    "CSN_key": str(members[0]["CSN_key"]),
                    "n_orders_in_group": len(members),
                    "n_order_descriptions": len(descriptions),
                    "site_union": " + ".join(sorted(union_sites)),
                    "earliest_exam_start": earliest,
                    "earliest_row_site_taken": " + ".join(sorted(earliest_sites)),
                    "order_descriptions": "; ".join(sorted(descriptions)),
                }
            )
    conflict_groups = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {"metric": "post-zero-drop overlapping CT groups with multiple procedures and disagreeing parsed Performing Location sites", "value": len(conflict_groups)},
            {"metric": "patients in those groups", "value": conflict_groups["CSN_key"].nunique() if len(conflict_groups) else 0},
            {"metric": "original orders inside those groups", "value": int(conflict_groups["n_orders_in_group"].sum()) if len(conflict_groups) else 0},
        ]
    )
    return summary, conflict_groups


def main():
    allpt = pd.read_excel(MERGED / "all_pt_combined.xlsx", sheet_name="All Patients")
    imaging = pd.read_excel(MERGED / "ct_orders_combined.xlsx", sheet_name="Imaging")
    cleaned = pd.read_excel(MERGED / "ct_orders_cleaned.xlsx", sheet_name="Imaging")
    final_nni = pd.read_csv(DATA / "nni_cleaned_orders.csv")

    allpt["CSN_key"] = allpt["CSN"].astype(str).str.strip()
    allpt_csns = set(allpt["CSN_key"])

    raw_ct = imaging[imaging["Procedure Category"].astype(str).eq("IMG CT ORDERABLES")].copy()
    raw_ct["CSN_key"] = raw_ct["CSN"].astype(str).str.strip()
    matched_ct = raw_ct[raw_ct["CSN_key"].isin(allpt_csns)].copy()

    start_missing = matched_ct["Exam Start Date"].apply(is_blank) | matched_ct["Exam Start Time"].apply(is_blank)
    end_missing = matched_ct["Exam End Date"].apply(is_blank) | matched_ct["Exam End Time"].apply(is_blank)
    oc_missing = matched_ct["Order Completion Date"].apply(is_blank) | matched_ct["Order Completion Time"].apply(is_blank)
    no_exam_no_oc = start_missing & end_missing & oc_missing

    ct_fallback = matched_ct.apply(fallback_exam_time, axis=1)
    ct_fallback["exam_start_dt"] = ct_fallback.apply(lambda r: combine_dt(r["Exam Start Date"], r["Exam Start Time"]), axis=1)
    ct_fallback["exam_end_dt"] = ct_fallback.apply(lambda r: combine_dt(r["Exam End Date"], r["Exam End Time"]), axis=1)
    ct_fallback["parent_order_dt"] = ct_fallback.apply(lambda r: combine_dt(r["Parent Order Date"], r["Parent Order Time"]), axis=1)
    ct_fallback["wait_min"] = (ct_fallback["exam_start_dt"] - ct_fallback["parent_order_dt"]).dt.total_seconds() / 60
    negative_wait = ct_fallback["wait_min"] < 0
    zero_duration_raw = ct_fallback["exam_start_dt"].notna() & ct_fallback["exam_end_dt"].notna() & (
        ct_fallback["exam_start_dt"] == ct_fallback["exam_end_dt"]
    )
    requested_early_keep = ~negative_wait & ~no_exam_no_oc

    rows = []

    def add_step(step, orders_before, dropped, orders_after, patients_before, patients_after, note=""):
        rows.append(
            {
                "step": step,
                "orders_before": int(orders_before),
                "orders_dropped_at_step": int(dropped),
                "orders_remaining": int(orders_after),
                "patients_before": int(patients_before),
                "patients_dropped_at_step": int(patients_before - patients_after),
                "patients_remaining": int(patients_after),
                "note": note,
            }
        )

    add_step(
        "Start: all raw CT orders from combined CT workbook",
        len(raw_ct),
        0,
        len(raw_ct),
        raw_ct["CSN_key"].nunique(),
        raw_ct["CSN_key"].nunique(),
        f"All Pt source has {len(allpt):,} rows / {allpt['CSN_key'].nunique():,} patients.",
    )
    add_step(
        "Drop CT orders whose CSN has no matching All Pt entry",
        len(raw_ct),
        len(raw_ct) - len(matched_ct),
        len(matched_ct),
        raw_ct["CSN_key"].nunique(),
        matched_ct["CSN_key"].nunique(),
    )
    after_negative = ct_fallback[~negative_wait]
    add_step(
        "Drop CT orders with negative wait time after exam/order-completion fallback",
        len(matched_ct),
        int(negative_wait.sum()),
        len(after_negative),
        matched_ct["CSN_key"].nunique(),
        after_negative["CSN_key"].nunique(),
        f"Zero wait retained at this point: {int((ct_fallback['wait_min'] == 0).sum()):,}.",
    )
    after_no_exam_no_oc = ct_fallback[~negative_wait & ~no_exam_no_oc]
    add_step(
        "Drop CT orders with neither exam start/end nor order-completion data",
        len(after_negative),
        int((~negative_wait & no_exam_no_oc).sum()),
        len(after_no_exam_no_oc),
        after_negative["CSN_key"].nunique(),
        after_no_exam_no_oc["CSN_key"].nunique(),
        "This follows the ordering requested here; the legacy multiple-CT workbook applied this timing drop before All Pt matching.",
    )

    initial_steps = pd.DataFrame(rows)

    imaging_scope = imaging.copy()
    imaging_scope["CSN_key"] = imaging_scope["CSN"].astype(str).str.strip()
    matched_imaging = imaging_scope[imaging_scope["CSN_key"].isin(allpt_csns)].copy()
    img_start_missing = matched_imaging["Exam Start Date"].apply(is_blank) | matched_imaging["Exam Start Time"].apply(is_blank)
    img_end_missing = matched_imaging["Exam End Date"].apply(is_blank) | matched_imaging["Exam End Time"].apply(is_blank)
    img_oc_missing = matched_imaging["Order Completion Date"].apply(is_blank) | matched_imaging["Order Completion Time"].apply(is_blank)
    img_no_exam_no_oc = img_start_missing & img_end_missing & img_oc_missing
    img_fallback = matched_imaging.apply(fallback_exam_time, axis=1)
    img_fallback["exam_start_dt"] = img_fallback.apply(lambda r: combine_dt(r["Exam Start Date"], r["Exam Start Time"]), axis=1)
    img_fallback["exam_end_dt"] = img_fallback.apply(lambda r: combine_dt(r["Exam End Date"], r["Exam End Time"]), axis=1)
    img_fallback["parent_order_dt"] = img_fallback.apply(lambda r: combine_dt(r["Parent Order Date"], r["Parent Order Time"]), axis=1)
    img_fallback["wait_min"] = (img_fallback["exam_start_dt"] - img_fallback["parent_order_dt"]).dt.total_seconds() / 60
    img_negative_wait = img_fallback["wait_min"] < 0
    img_zero_duration = img_fallback["exam_start_dt"].notna() & img_fallback["exam_end_dt"].notna() & (
        img_fallback["exam_start_dt"] == img_fallback["exam_end_dt"]
    )
    img_after_negative = img_fallback[~img_negative_wait]
    img_after_no_exam_no_oc = img_fallback[~img_negative_wait & ~img_no_exam_no_oc]
    img_after_zero_duration = img_fallback[~img_negative_wait & ~img_no_exam_no_oc & ~img_zero_duration]
    all_imaging_steps = pd.DataFrame(
        [
            {
                "step": "Start: all raw imaging orders from combined CT workbook",
                "orders_before": len(imaging_scope),
                "orders_dropped_at_step": 0,
                "orders_remaining": len(imaging_scope),
                "patients_before": imaging_scope["CSN_key"].nunique(),
                "patients_dropped_at_step": 0,
                "patients_remaining": imaging_scope["CSN_key"].nunique(),
            },
            {
                "step": "Drop imaging orders whose CSN has no matching All Pt entry",
                "orders_before": len(imaging_scope),
                "orders_dropped_at_step": len(imaging_scope) - len(matched_imaging),
                "orders_remaining": len(matched_imaging),
                "patients_before": imaging_scope["CSN_key"].nunique(),
                "patients_dropped_at_step": imaging_scope["CSN_key"].nunique() - matched_imaging["CSN_key"].nunique(),
                "patients_remaining": matched_imaging["CSN_key"].nunique(),
            },
            {
                "step": "Drop imaging orders with negative wait time after exam/order-completion fallback",
                "orders_before": len(matched_imaging),
                "orders_dropped_at_step": int(img_negative_wait.sum()),
                "orders_remaining": len(img_after_negative),
                "patients_before": matched_imaging["CSN_key"].nunique(),
                "patients_dropped_at_step": matched_imaging["CSN_key"].nunique() - img_after_negative["CSN_key"].nunique(),
                "patients_remaining": img_after_negative["CSN_key"].nunique(),
            },
            {
                "step": "Drop imaging orders with neither exam start/end nor order-completion data",
                "orders_before": len(img_after_negative),
                "orders_dropped_at_step": int((~img_negative_wait & img_no_exam_no_oc).sum()),
                "orders_remaining": len(img_after_no_exam_no_oc),
                "patients_before": img_after_negative["CSN_key"].nunique(),
                "patients_dropped_at_step": img_after_negative["CSN_key"].nunique() - img_after_no_exam_no_oc["CSN_key"].nunique(),
                "patients_remaining": img_after_no_exam_no_oc["CSN_key"].nunique(),
            },
            {
                "step": "Drop zero-minute exam-window imaging rows",
                "orders_before": len(img_after_no_exam_no_oc),
                "orders_dropped_at_step": int((~img_negative_wait & ~img_no_exam_no_oc & img_zero_duration).sum()),
                "orders_remaining": len(img_after_zero_duration),
                "patients_before": img_after_no_exam_no_oc["CSN_key"].nunique(),
                "patients_dropped_at_step": img_after_no_exam_no_oc["CSN_key"].nunique() - img_after_zero_duration["CSN_key"].nunique(),
                "patients_remaining": img_after_zero_duration["CSN_key"].nunique(),
            },
        ]
    )

    all_imaging_unmatched_categories = (
        imaging_scope[~imaging_scope["CSN_key"].isin(allpt_csns)]["Procedure Category"]
        .value_counts(dropna=False)
        .rename_axis("Procedure Category")
        .reset_index(name="unmatched_order_rows")
    )
    exam_time_breakdown = pd.concat(
        [
            exam_time_source_breakdown(imaging_scope, "All imaging raw"),
            exam_time_source_breakdown(matched_imaging, "All imaging after All Pt match"),
            exam_time_source_breakdown(raw_ct, "CT raw"),
            exam_time_source_breakdown(matched_ct, "CT after All Pt match"),
        ],
        ignore_index=True,
    )

    multiple_review = pd.ExcelFile(ANALYSIS / "ct_multiple_orders_filtered_review_include_zero_wait.xlsx")
    ct_per_patient = pd.read_excel(multiple_review, sheet_name="ct_orders_per_patient")
    timing_classes = pd.read_excel(multiple_review, sheet_name="multi_ct_timing_classes")
    legacy_steps = pd.read_excel(multiple_review, sheet_name="filter_steps")

    zero_exam_drop_summary = pd.DataFrame(
        [
            {"metric": "CT rows after requested early filters", "value": int(requested_early_keep.sum())},
            {
                "metric": "zero-minute exam-window rows (Exam Start == Exam End) dropped next",
                "value": int((requested_early_keep & zero_duration_raw).sum()),
            },
            {
                "metric": "% zero-minute exam-window rows at that stage",
                "value": round(pct((requested_early_keep & zero_duration_raw).sum(), requested_early_keep.sum()), 3),
            },
            {
                "metric": "CT rows remaining after zero-minute exam-window drop",
                "value": int((requested_early_keep & ~zero_duration_raw).sum()),
            },
            {
                "metric": "patients remaining after zero-minute exam-window drop",
                "value": ct_fallback.loc[requested_early_keep & ~zero_duration_raw, "CSN_key"].nunique(),
            },
        ]
    )
    post_zero_ct = ct_fallback[requested_early_keep & ~zero_duration_raw].copy()
    site_conflict_summary, site_conflict_groups = build_site_conflict_stats(post_zero_ct)

    queue_summary = pd.read_excel(ANALYSIS / "ct_queue_reconstruction.xlsx", sheet_name="Summary")
    wait_by_priority = pd.read_excel(ANALYSIS / "nni_wait_time_by_scan_priority.xlsx", sheet_name="Distribution_by_Priority")
    target_compliance = pd.read_excel(ANALYSIS / "nni_wait_time_by_scan_priority.xlsx", sheet_name="SLA_Compliance")
    old_scan_orders = pd.read_excel(ANALYSIS / "ct_scan_time_by_type.xlsx", sheet_name="Merged_CT_Orders")
    old_scan_summary = []
    for site, grp in old_scan_orders.groupby("Site"):
        old_scan_summary.append({"Site": site, **summarize(grp["scan_duration_min"])})
    old_scan_summary = pd.DataFrame(old_scan_summary)
    old_scan_time_by_type = pd.concat(
        [
            pd.read_excel(ANALYSIS / "ct_scan_time_by_type.xlsx", sheet_name="NNI_ScanTime_By_Type_Merged").assign(Site="NNI"),
            pd.read_excel(ANALYSIS / "ct_scan_time_by_type.xlsx", sheet_name="DR_ScanTime_By_Type_Merged").assign(Site="DR"),
        ],
        ignore_index=True,
    )

    historical_seq_overall = pd.read_excel(ANALYSIS / "nni_queue_priority_policy_check.xlsx", sheet_name="Overall_Compliance")
    historical_seq_by_tier = pd.read_excel(ANALYSIS / "nni_queue_priority_policy_check.xlsx", sheet_name="Compliance_by_Tier")
    simulation_summary = pd.read_excel(ANALYSIS / "nni_priority_policy_simulation.xlsx", sheet_name="Tier_Summary")
    queue_orders = pd.read_excel(ANALYSIS / "ct_queue_reconstruction.xlsx", sheet_name="CT_Orders")
    queue_nni_served = queue_orders[
        (queue_orders["site"] == "NNI")
        & (queue_orders["wait_to_start_min"] >= 0)
    ].copy()
    queue_nni_served["_order_key"] = queue_nni_served.apply(order_key, axis=1)

    nni_cleaned = cleaned[cleaned["Site"].astype(str).str.strip().str.upper().eq("NNI")].copy()
    nni_cleaned["_order_key"] = nni_cleaned.apply(order_key, axis=1)
    performing_location = nni_cleaned["Performing Location"].fillna("").str.strip()
    procedure_location = nni_cleaned["Procedure Location"].fillna("").str.strip()
    keep_location = pd.Series(True, index=nni_cleaned.index)
    duration, branch = compute_final_scan_duration(nni_cleaned.loc[keep_location].copy())
    final_candidate = nni_cleaned.loc[keep_location].copy()
    final_candidate["scan_duration_recomputed"] = duration
    final_candidate["duration_branch"] = branch
    keep_nonnegative_duration = final_candidate["scan_duration_recomputed"].notna() & (final_candidate["scan_duration_recomputed"] >= 0)
    final_after_nonnegative_duration = final_candidate[keep_nonnegative_duration].copy()
    zero_scan_duration = final_after_nonnegative_duration["scan_duration_recomputed"] == 0
    final_after_zero_drop = final_after_nonnegative_duration[~zero_scan_duration].copy()
    final_keys = set(final_after_zero_drop["_order_key"])
    queue_keys = set(queue_nni_served["_order_key"])
    queue_to_final_removed = nni_cleaned[nni_cleaned["_order_key"].isin(queue_keys - final_keys)].copy()
    queue_to_final_added = nni_cleaned[nni_cleaned["_order_key"].isin(final_keys - queue_keys)].copy()
    queue_to_final_bridge = pd.DataFrame(
        [
            {"bridge_item": "NNI service-start analysis population (last site scan included)", "rows": len(queue_nni_served)},
            {
                "bridge_item": "served rows not in final export",
                "rows": -len(queue_to_final_removed),
            },
            {
                "bridge_item": "final-export rows not in service-start analysis population",
                "rows": len(queue_to_final_added),
            },
            {"bridge_item": "Final exported NNI rows after zero scan-duration drop", "rows": len(final_after_zero_drop)},
            {
                "bridge_item": "Simulated current sequence-rule cohort (final exported rows)",
                "rows": int(simulation_summary["N"].sum()),
            },
        ]
    )
    cleaned_ct = cleaned[cleaned["Procedure Category"].astype(str).eq("IMG CT ORDERABLES")].copy()
    fallback_ct = cleaned_ct[
        cleaned_ct["Performing Location"].isna()
        & cleaned_ct["Site"].notna()
    ].copy()
    tsv_fallback_summary = pd.DataFrame(
        [
            {"metric": "cleaned CT rows with empty parsed Performing Location using TSV fallback", "value": len(fallback_ct)},
            {"metric": "patients in TSV fallback rows", "value": fallback_ct["CSN"].astype(str).nunique()},
            {"metric": "fallback rows assigned NNI", "value": int((fallback_ct["Site"] == "NNI").sum())},
            {"metric": "fallback rows assigned DR", "value": int((fallback_ct["Site"] == "DR").sum())},
        ]
    )

    final_export_summary = pd.DataFrame(
        [
            {"metric": "canonical cleaned NNI rows before final CSV filters", "value": len(nni_cleaned)},
            {"metric": "rows passing final Site==NNI filter", "value": len(final_candidate)},
            {"metric": "rows dropped for missing/negative scan_duration", "value": int((~keep_nonnegative_duration).sum())},
            {"metric": "rows before zero scan-duration drop", "value": len(final_after_nonnegative_duration)},
            {"metric": "rows dropped for zero scan_duration", "value": int(zero_scan_duration.sum())},
            {"metric": "final rows currently written to data/nni_cleaned_orders.csv", "value": len(final_nni)},
            {"metric": "recomputed final rows after zero scan-duration drop", "value": len(final_after_zero_drop)},
        ]
    )
    duration_branch = final_candidate["duration_branch"].value_counts(dropna=False).rename_axis("duration_rule_branch").reset_index(name="rows")

    zero_scan_detail = final_after_nonnegative_duration.loc[
        zero_scan_duration,
        ["CSN", "Order ID", "Order Description", "Exam Start Date", "Exam Start Time", "Exam End Date", "Exam End Time", "scan_duration_recomputed"],
    ]

    with pd.ExcelWriter(ANALYSIS / "case_count_trace.xlsx", engine="openpyxl") as writer:
        initial_steps.to_excel(writer, sheet_name="01_requested_filter_steps", index=False)
        all_imaging_steps.to_excel(writer, sheet_name="02_all_imaging_steps", index=False)
        all_imaging_unmatched_categories.to_excel(writer, sheet_name="03_all_img_unmatched_type", index=False)
        exam_time_breakdown.to_excel(writer, sheet_name="04_exam_time_breakdown", index=False)
        legacy_steps.to_excel(writer, sheet_name="05_legacy_multi_ct_steps", index=False)
        ct_per_patient.to_excel(writer, sheet_name="06_ct_orders_per_patient", index=False)
        timing_classes.to_excel(writer, sheet_name="07_multi_ct_timing", index=False)
        site_conflict_summary.to_excel(writer, sheet_name="08_site_conflict_summary", index=False)
        site_conflict_groups.to_excel(writer, sheet_name="09_site_conflict_groups", index=False)
        zero_exam_drop_summary.to_excel(writer, sheet_name="10_zero_exam_time_drop", index=False)
        wait_by_priority.to_excel(writer, sheet_name="11_wait_by_priority_site", index=False)
        target_compliance.to_excel(writer, sheet_name="12_target_compliance", index=False)
        queue_summary.to_excel(writer, sheet_name="13_queue_summary", index=False)
        old_scan_summary.to_excel(writer, sheet_name="14_old_scan_duration_site", index=False)
        old_scan_time_by_type.to_excel(writer, sheet_name="15_old_scan_duration_type", index=False)
        historical_seq_overall.to_excel(writer, sheet_name="16_nni_seq_overall", index=False)
        historical_seq_by_tier.to_excel(writer, sheet_name="17_nni_seq_by_tier", index=False)
        queue_to_final_bridge.to_excel(writer, sheet_name="18_queue_to_final_bridge", index=False)
        tsv_fallback_summary.to_excel(writer, sheet_name="19_tsv_fallback_empty_loc", index=False)
        final_export_summary.to_excel(writer, sheet_name="20_final_export_trace", index=False)
        duration_branch.to_excel(writer, sheet_name="21_duration_rule_branch", index=False)
        zero_scan_detail.to_excel(writer, sheet_name="22_zero_scan_duration_rows", index=False)
        simulation_summary.to_excel(writer, sheet_name="23_simulated_seq_rules", index=False)
        auto_col_width(writer)

    write_markdown(
        initial_steps,
        all_imaging_steps,
        all_imaging_unmatched_categories,
        exam_time_breakdown,
        ct_per_patient,
        timing_classes,
        site_conflict_summary,
        zero_exam_drop_summary,
        wait_by_priority,
        target_compliance,
        old_scan_summary,
        historical_seq_overall,
        historical_seq_by_tier,
        queue_to_final_bridge,
        tsv_fallback_summary,
        final_export_summary,
        simulation_summary,
    )


def md_table(df):
    out = df.copy()
    out = out.replace({np.nan: ""})
    out = out.astype(str)
    headers = list(out.columns)
    rows = out.values.tolist()
    widths = [
        max([len(str(header))] + [len(str(row[i])) for row in rows])
        for i, header in enumerate(headers)
    ]

    def fmt_row(values):
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(values)) + " |"

    lines = [fmt_row(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def write_markdown(
    initial_steps,
    all_imaging_steps,
    all_imaging_unmatched_categories,
    exam_time_breakdown,
    ct_per_patient,
    timing_classes,
    site_conflict_summary,
    zero_exam_drop_summary,
    wait_by_priority,
    target_compliance,
    old_scan_summary,
    historical_seq_overall,
    historical_seq_by_tier,
    queue_to_final_bridge,
    tsv_fallback_summary,
    final_export_summary,
    simulation_summary,
):
    lines = [
        "# Case Count Trace To Final NNI Cohort",
        "",
        "Generated by `trace_case_counts.py` from the existing source and analysis workbooks.",
        "",
        "## Requested Early Filter Order",
        "",
        md_table(initial_steps),
        "",
        "## All Imaging Orders Using Same Early Filter Order",
        "",
        md_table(all_imaging_steps),
        "",
        "Unmatched all-imaging order rows by procedure category:",
        "",
        md_table(all_imaging_unmatched_categories),
        "",
        "## Exam-Time Availability Before Fallback",
        "",
        md_table(exam_time_breakdown),
        "",
        "## Multiple CT Scans After Early Filters",
        "",
        "Patient distribution by number of CT orders/scans:",
        "",
        md_table(ct_per_patient),
        "",
        "Timing classes among patients with multiple CT scans:",
        "",
        md_table(timing_classes),
        "",
        "## Merged Overlapping Orders With Disagreeing Site Mappings",
        "",
        md_table(site_conflict_summary),
        "",
        "For these groups, the production cleaner resolves `Site` and `Procedure Location` from the original row with the earliest Exam Start.",
        "",
        "## Zero Exam-Time Rows Dropped",
        "",
        md_table(zero_exam_drop_summary),
        "",
        "## Reconstructed Queue Wait Time By Site And Scan Priority",
        "",
        md_table(wait_by_priority),
        "",
        "## Historical Orders Within Target Time",
        "",
        md_table(target_compliance),
        "",
        "## Old Scan Duration Distribution",
        "",
        "Old method: shorter of `Exam End - Exam Start` and `next Exam Start - Exam Start`, by physical site.",
        "",
        md_table(old_scan_summary),
        "",
        "## NNI Compliance With Existing Sequence Rule",
        "",
        md_table(historical_seq_overall),
        "",
        md_table(historical_seq_by_tier),
        "",
        "## Bridge From Queue Population To Final Export",
        "",
        md_table(queue_to_final_bridge),
        "",
        "## TSV Fallback For Empty Performing Location",
        "",
        md_table(tsv_fallback_summary),
        "",
        "## Final NNI Export And Zero Scan Duration",
        "",
        md_table(final_export_summary),
        "",
        "## Simulated Queue Using Current Sequence Rules",
        "",
        md_table(simulation_summary),
        "",
    ]
    (ANALYSIS / "case_count_trace.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
