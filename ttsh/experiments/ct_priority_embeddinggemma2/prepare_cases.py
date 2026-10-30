"""Prepare de-identified, unmerged CT priority prototype datasets.

Both sources use original CT order rows and retain orders with a matching All
Patients entry. Repeated mentions of one priority collapse to one label;
orders containing multiple distinct priorities go to a review workbook.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = OUT / "data"
KEY_PATH = OUT / "privacy_key.bin"
PRIORITIES = {"P1", "P2", "P3"}
PRIORITY_RE = re.compile(r"(?:ED\s+)?Scan\s+Priority\s*:\s*(P[123])\b", re.IGNORECASE)

PATIENT_FIELDS = [
    "Age at Visit", "Sex", "Arrival Method", "Chief Complaint",
    "KIV Uptriage", "Triage Acuity", "Consult Acuity",
    "First ED Care Area", "Primary Care Area", "Classification Type",
    "CFS (ED)", "ED Primary Dx ICD10",
    # User-confirmed as available before the CT order and therefore eligible
    # for prediction inputs; their provenance/timing assumption is recorded in
    # the study report and run metadata.
    "Reatt Same ED", "Reatt Reason", "72h Reatt NGEMR ED",
    "72h Reatt NGEMR ED then Admitted", "HD/ICU from GW within 24H",
    # Keep only early BP measurements; omit BP last and flow timestamps.
    "BP first", "BP before First EPP", "BP before Consult",
]
ORDER_FIELDS = [
    "month", "Order Description", "Primary ED Dx", "ED Dx",
    "Ordering Question Responses", "Scan Priority", "N Merged Rows",
    "Merged Order IDs", "CSN", "Order ID", "Procedure Category",
]


def get_key() -> bytes:
    if not KEY_PATH.exists():
        KEY_PATH.write_bytes(__import__("secrets").token_bytes(32))
        KEY_PATH.chmod(0o600)
    return KEY_PATH.read_bytes()


def opaque_id(key: bytes, kind: str, value: object) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    return hmac.new(key, f"{kind}:{raw}".encode(), hashlib.sha256).hexdigest()[:24]


def priority_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    seen: list[str] = []
    for found in PRIORITY_RE.findall(str(value)):
        label = found.upper()
        if label in PRIORITIES and label not in seen:
            seen.append(label)
    return seen


def label_status(value: object) -> tuple[str, str, str]:
    if pd.isna(value) or not str(value).strip():
        return "", "missing", ""
    occurrences = [p.upper() for p in PRIORITY_RE.findall(str(value))]
    distinct = list(dict.fromkeys(occurrences))
    if not distinct:
        return "", "unparsed", ""
    if len(distinct) > 1:
        return "", "composite_priority", "; ".join(distinct)
    status = "repeated_same_priority" if len(occurrences) > 1 else "single_priority"
    return distinct[0], status, "; ".join(distinct)


def parse_response(text: object, question: str) -> str:
    """Read only the named, approved response value(s), never the full blob."""
    if pd.isna(text):
        return ""
    # Values in merged responses are separated by newlines and/or semicolons.
    pattern = re.compile(rf"(?:^|[\n;])\s*{re.escape(question)}\s*::?\s*([^\n;]+)", re.I)
    values = []
    for match in pattern.finditer(str(text)):
        value = match.group(1).strip()
        if value and value not in values:
            values.append(value)
    return " | ".join(values)


def read_excel(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, dtype={"CSN": str})


def prepare(source: pd.DataFrame, allpt: pd.DataFrame, source_name: str, key: bytes,
            target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    source = source.copy()
    allpt = allpt.copy()
    source["CSN"] = source["CSN"].astype("string").str.strip()
    allpt["CSN"] = allpt["CSN"].astype("string").str.strip()
    allpt = allpt.drop_duplicates("CSN", keep="first")

    valid_patient_fields = [c for c in PATIENT_FIELDS if c in allpt.columns]
    patient = allpt[["CSN", *valid_patient_fields]]
    allpt_csns = set(patient["CSN"].dropna())
    unmatched_order_rows = int((~source["CSN"].isin(allpt_csns)).sum())
    joined = source.merge(patient, on="CSN", how="inner", validate="m:1", suffixes=("", "_AllPt"))
    statuses = joined[target_col].map(label_status)
    joined[["Priority Label", "Label Status", "Observed Priorities"]] = pd.DataFrame(
        statuses.tolist(), index=joined.index
    )

    # Whitelist approved free-text response values for indication fields.
    joined["Working Diagnosis / Reason"] = joined["Ordering Question Responses"].map(
        lambda v: parse_response(v, "Working diagnosis/Reason for exam")
    )
    joined["Relevant Clinical Information"] = joined["Ordering Question Responses"].map(
        lambda v: parse_response(v, "Relevant clinical information")
    )
    # Historical and new All Pt exports use different names for the early
    # pre-clinician BP; normalize them so each partition uses the same field.
    bp_epp = joined.get("BP before First EPP", pd.Series(pd.NA, index=joined.index))
    bp_consult = joined.get("BP before Consult", pd.Series(pd.NA, index=joined.index))
    joined["BP before clinician"] = bp_epp.combine_first(bp_consult)

    patient_ids = joined["CSN"].map(lambda v: opaque_id(key, "patient", v))
    raw_case = joined.get("Order ID", pd.Series(joined.index, index=joined.index))
    case_ids = [opaque_id(key, f"case:{source_name}", f"{csn}|{order}") for csn, order in zip(joined["CSN"], raw_case)]
    joined["Case ID"] = case_ids
    joined["Patient Group ID"] = patient_ids
    joined["Source"] = source_name

    # Candidate input fields and audit columns are stored separately from the
    # target label. Only explicitly listed template fields enter embeddings.
    output_cols = [
        "Case ID", "Patient Group ID", "Source", "month", "Priority Label",
        "Label Status", "Order Description", "Working Diagnosis / Reason",
        "Relevant Clinical Information", "Primary ED Dx", "ED Dx",
        *valid_patient_fields,
        "BP before clinician",
    ]
    output_cols = list(dict.fromkeys(c for c in output_cols if c in joined.columns))
    prepared = joined[output_cols].copy()
    # The database set excludes missing/unparsed/composite priorities. The
    # Test file retains unlabeled new orders for inference; composite labels
    # remain explicitly marked and are not scored as single-label cases.
    review = joined[joined["Label Status"] == "composite_priority"].copy()
    if source_name == "historical_database":
        prepared = prepared[prepared["Label Status"].isin(["single_priority", "repeated_same_priority"])].copy()
    stats = {
        "matched_order_rows": len(joined),
        "unmatched_order_rows_dropped": unmatched_order_rows,
        **{str(k): int(v) for k, v in joined["Label Status"].value_counts(dropna=False).items()},
        "rows_prepared": len(prepared),
        "composites_review": len(review),
    }
    return prepared, review, stats


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    DATA.chmod(0o700)
    key = get_key()

    historical = read_excel(ROOT / "merged/ct_orders_combined.xlsx", "Imaging")
    historical = historical[historical["Procedure Category"] == "IMG CT ORDERABLES"].copy()
    historical_allpt = read_excel(ROOT / "merged/all_pt_combined.xlsx", "All Patients")
    hist, hist_review, hist_stats = prepare(
        historical, historical_allpt, "historical_database", key, "Ordering Question Responses"
    )

    new_parts = []
    for month, filename, sheet in [
        ("Jul 2026", "2026 7 CT & Lab Orders (decrypted).xlsx", "Jul Imaging"),
        ("Aug 2026", "2026 8 CT & Lab Orders (decrypted).xlsx", "Aug Imaging"),
        ("Sep 2026", "2026 9 CT & Lab Orders (decrypted).xlsx", "Sep Imaging"),
    ]:
        part = read_excel(ROOT / "data/newdata" / filename, sheet)
        part["month"] = month
        part = part[part["Procedure Category"] == "IMG CT ORDERABLES"].copy()
        new_parts.append(part)
    new = pd.concat(new_parts, ignore_index=True)
    new_allpt_parts = []
    for month, filename, sheet in [
        ("Jul 2026", "2026 7 All Pt (decrypted).xlsx", "Sheet2"),
        ("Aug 2026", "2026 8 All Pt (decrypted).xlsx", "Sheet1"),
        ("Sep 2026", "2026 9 All Pt (decrypted).xlsx", "Sheet1"),
    ]:
        part = read_excel(ROOT / "data/newdata" / filename, sheet)
        part["month"] = month
        new_allpt_parts.append(part)
    new_allpt = pd.concat(new_allpt_parts, ignore_index=True)
    test, test_review, test_stats = prepare(new, new_allpt, "newdata_test", key, "Ordering Question Responses")

    hist.to_excel(DATA / "historical_cases.xlsx", index=False)
    test.to_excel(DATA / "newdata_test_cases.xlsx", index=False)
    with pd.ExcelWriter(DATA / "composite_priority_review.xlsx", engine="openpyxl") as writer:
        review_cols = ["Case ID", "month", "Observed Priorities", "Order Description", "Primary ED Dx", "ED Dx"]
        hist_review[review_cols].to_excel(
            writer, sheet_name="Historical composites", index=False
        )
        test_review[review_cols].to_excel(
            writer, sheet_name="New-data composites", index=False
        )
    for path in DATA.glob("*.xlsx"):
        path.chmod(0o600)

    print("Historical CT prepared:", hist_stats)
    print("New-data CT prepared:", test_stats)
    print("Prepared files:", DATA / "historical_cases.xlsx", DATA / "newdata_test_cases.xlsx")
    print("Composite review:", DATA / "composite_priority_review.xlsx")
    print("Historical class counts:", hist["Priority Label"].value_counts().to_dict())
    print("New-data class counts (single or repeated labels):", test.loc[test["Label Status"].isin(["single_priority", "repeated_same_priority"]), "Priority Label"].value_counts().to_dict())
    historical_patients = set(hist["Patient Group ID"])
    new_patients = set(test["Patient Group ID"])
    print("Patient overlap historical vs new-data:", len(historical_patients & new_patients))


if __name__ == "__main__":
    main()
