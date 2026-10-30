"""
nni_scan_priority_location_check.py
Two data-quality checks on CT orders in merged/ct_orders_cleaned.xlsx, run
against two populations:

  - "AllCT": every row with a non-blank "Site" (NNI/DR), assigned upstream by
    clean_ct_orders.py from Order Description via brainbodysep.tsv -- includes
    rows bundled with non-CT procedures.
  - "PureCT": AllCT rows further restricted to Procedure Category CONTAINING
    "IMG CT ORDERABLES" (so combo orders like "IMG CT ORDERABLES; IMG
    DIAGNOSTIC IMAGING ORDERABLES" are included, not just an exact match),
    with a non-null "Ordering Question Responses".

Site assignment (and resolving cases where Order Description tokens would
otherwise span both NNI and DR) now happens once, upstream, in
clean_ct_orders.py -- see its "Site" column and the cross-row merge-conflict
handling in its grouping step. This script just reads that field.

Checks:
1. Scan priority completeness: does every NNI-site CT order have an
   "ED Scan Priority" answer anywhere in its "Ordering Question Responses"
   text?
2. Location alignment: does the scan location implied by scan type
   (the "Site" column: NNI vs DR) agree with the "Performing Location"
   answer(s) actually recorded in "Ordering Question Responses"?

"Ordering Question Responses" can hold more than one submission block
(joined by "; " when clean_ct_orders.py merges rows from resubmitted/
amended orders), each block a "\n"-separated list of "Question: Answer"
lines. Both checks look across ALL blocks on the row (was the answer *ever*
recorded / did the location *ever* match), not just the first block.

Outputs: analysis/nni_scan_priority_location_check.xlsx
"""
import os
import re

import pandas as pd
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED = os.path.join(BASE, "merged")
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

LOC_PATTERN = re.compile(r"Performing Location:\s*([^\n;]+)")
PRIORITY_PATTERN = re.compile(r"(?:ED )?Scan Priority:\s*([^\n;]+)")


def is_blank(v):
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def extract_locations(val):
    if is_blank(val):
        return []
    return [m.strip() for m in LOC_PATTERN.findall(str(val))]


def has_scan_priority(val):
    if is_blank(val):
        return False
    return bool(PRIORITY_PATTERN.search(str(val)))


def check_alignment(site, locs):
    locset = set(locs)
    if not locset:
        return "no_location_captured"
    if site == "NNI":
        return "aligned" if "National Neuroscience Institute" in locset else "misaligned"
    # site == DR: aligned unless the record shows ONLY the NNI location
    if "National Neuroscience Institute" in locset and "Tan Tock Seng Hospital" not in locset:
        return "misaligned"
    return "aligned"


def auto_col_width(ws, extra=2):
    for col in ws.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 60)


def run_checks(ct, label):
    """ct must already have a 'site' column (NNI/DR). Returns detail frames
    + a summary-rows list for this population."""
    ct = ct.copy()
    ct["has_scan_priority"] = ct["Ordering Question Responses"].apply(has_scan_priority)
    ct["responses_blank"] = ct["Ordering Question Responses"].apply(is_blank)
    ct["locations_found"] = ct["Ordering Question Responses"].apply(extract_locations)
    ct["loc_str"] = ct["locations_found"].apply(lambda l: " + ".join(sorted(set(l)))).astype(object)
    ct["alignment"] = [check_alignment(s, l) for s, l in zip(ct["site"], ct["locations_found"])]

    nni = ct[ct["site"] == "NNI"]
    n_nni = len(nni)
    n_nni_missing = int((~nni["has_scan_priority"]).sum())
    n_nni_missing_blank_resp = int((nni["responses_blank"]).sum())
    n_nni_missing_with_resp = n_nni_missing - n_nni_missing_blank_resp
    alignment_counts = ct["alignment"].value_counts()

    print(f"\n[{label}] CT rows: {len(ct):,}  (site split: {ct['site'].value_counts().to_dict()})")
    print(f"[{label}] Check 1 — NNI orders missing a Scan Priority answer: "
          f"{n_nni_missing:,} / {n_nni:,} ({100 * n_nni_missing / n_nni:.1f}%)")
    print(f"[{label}] Check 2 — site (scan type) vs Performing Location alignment:")
    print(alignment_counts.to_string())

    detail_cols = ["CSN", "Order ID", "Order Description", "Order Status", "site",
                   "has_scan_priority", "loc_str", "alignment"]

    nni_missing_priority = ct[(ct["site"] == "NNI") & (~ct["has_scan_priority"])][detail_cols] \
        .sort_values(["Order Description", "CSN"]).reset_index(drop=True)
    misaligned = ct[ct["alignment"] == "misaligned"][detail_cols] \
        .sort_values(["site", "Order Description", "CSN"]).reset_index(drop=True)
    no_location = ct[ct["alignment"] == "no_location_captured"][detail_cols] \
        .sort_values(["site", "CSN"]).reset_index(drop=True)

    summary_rows = [
        {"Metric": f"Total CT rows ({label})", "Value": len(ct)},
        {"Metric": "NNI-site CT rows", "Value": n_nni},
        {"Metric": "DR-site CT rows", "Value": int((ct["site"] == "DR").sum())},
        {"Metric": "", "Value": ""},
        {"Metric": "-- Check 1: Scan priority completeness (NNI only) --", "Value": ""},
        {"Metric": "NNI rows with a Scan Priority answer", "Value": n_nni - n_nni_missing},
        {"Metric": "NNI rows missing a Scan Priority answer", "Value": n_nni_missing},
        {"Metric": "  of which: Ordering Question Responses entirely blank", "Value": n_nni_missing_blank_resp},
        {"Metric": "  of which: responses present but no Scan Priority sub-answer", "Value": n_nni_missing_with_resp},
        {"Metric": "% NNI missing Scan Priority", "Value": round(100 * n_nni_missing / n_nni, 1)},
        {"Metric": "", "Value": ""},
        {"Metric": "-- Check 2: Site (scan type) vs Performing Location alignment --", "Value": ""},
    ]
    for al, cnt in alignment_counts.items():
        summary_rows.append({"Metric": f"  {al}", "Value": int(cnt)})

    return nni_missing_priority, misaligned, no_location, summary_rows


# ---------------------------------------------------------------------------
# Load: Site is now assigned upstream by clean_ct_orders.py
# ---------------------------------------------------------------------------
print("Loading merged/ct_orders_cleaned.xlsx ...")
img = pd.read_excel(os.path.join(MERGED, "ct_orders_cleaned.xlsx"), sheet_name="Imaging")
img["site"] = img["Site"]

all_ct = img[img["site"].notna()].copy()
print(f"AllCT (Site assigned): {len(all_ct):,}")

pure_ct = all_ct[
    all_ct["Ordering Question Responses"].notna()
    & all_ct["Procedure Category"].fillna("").str.contains("IMG CT ORDERABLES")
].copy()
print(f"Pure CT orderables (Procedure Category contains 'IMG CT ORDERABLES', non-null responses): {len(pure_ct):,}")

# ---------------------------------------------------------------------------
# Run both populations & write workbook
# ---------------------------------------------------------------------------
results = {}
for label, pop in [("AllCT", all_ct), ("PureCT", pure_ct)]:
    results[label] = run_checks(pop, label)

out_path = os.path.join(OUT, "nni_scan_priority_location_check.xlsx")
print(f"\nWriting {out_path} ...")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    for label, (nni_missing_priority, misaligned, no_location, summary_rows) in results.items():
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name=f"Summary_{label}", index=False)
        nni_missing_priority.to_excel(writer, sheet_name=f"{label}_Missing_ScanPriority", index=False)
        misaligned.to_excel(writer, sheet_name=f"{label}_Site_Loc_Misaligned", index=False)
        no_location.to_excel(writer, sheet_name=f"{label}_No_Location", index=False)
    for sheet in writer.sheets.values():
        auto_col_width(sheet)

print("Done.")
