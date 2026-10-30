#!/usr/bin/env python3
"""Clean the original KTAS workbook into a de-identified CSV."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


WORKBOOK_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DEFAULT_INPUT = Path("data/originaldata.xlsx")
DEFAULT_OUTPUT = Path("data/originaldata_cleaned.csv")
DEFAULT_SHEET = "N=1267 data"
COLUMNS_TO_REMOVE = ("Patients number per hour", "Length of stay_min")
HANGUL_RE = re.compile(r"[\u3131-\u318E\uAC00-\uD7A3]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove identifying columns and rows containing Korean text from "
            "the original KTAS workbook."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input workbook path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
        help=f"Worksheet name to read (default: {DEFAULT_SHEET})",
    )
    return parser.parse_args()


def excel_column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return index - 1


def load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []

    values: list[str] = []
    for item in root.findall("a:si", WORKBOOK_NS):
        text = "".join(node.text or "" for node in item.iterfind(".//a:t", WORKBOOK_NS))
        values.append(text)
    return values


def resolve_sheet_path(archive: zipfile.ZipFile, target_sheet: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}

    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("r:Relationship", rel_ns)
    }

    for sheet in workbook.findall("a:sheets/a:sheet", WORKBOOK_NS):
        if sheet.attrib.get("name") == target_sheet:
            rel_id = sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            target = rel_map[rel_id]
            return f"xl/{target}" if not target.startswith("xl/") else target

    available = [sheet.attrib.get("name", "") for sheet in workbook.findall("a:sheets/a:sheet", WORKBOOK_NS)]
    raise ValueError(
        f"Worksheet '{target_sheet}' not found. Available sheets: {', '.join(available)}"
    )


def iter_sheet_rows(
    archive: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]
) -> Iterable[list[str]]:
    sheet = ET.fromstring(archive.read(sheet_path))
    sheet_data = sheet.find("a:sheetData", WORKBOOK_NS)
    if sheet_data is None:
        return

    for row in sheet_data.findall("a:row", WORKBOOK_NS):
        cells: dict[int, str] = {}
        max_index = -1
        for cell in row.findall("a:c", WORKBOOK_NS):
            cell_ref = cell.attrib.get("r", "")
            col_index = excel_column_index(cell_ref) if cell_ref else len(cells)
            max_index = max(max_index, col_index)
            raw_value = cell.find("a:v", WORKBOOK_NS)
            value = ""
            if raw_value is not None and raw_value.text is not None:
                if cell.attrib.get("t") == "s":
                    value = shared_strings[int(raw_value.text)]
                else:
                    value = raw_value.text
            elif cell.attrib.get("t") == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.iterfind(".//a:t", WORKBOOK_NS)
                )
            cells[col_index] = value

        if max_index >= 0:
            yield [cells.get(i, "") for i in range(max_index + 1)]


def normalize_row(row: list[str], width: int) -> list[str]:
    if len(row) < width:
        return row + [""] * (width - len(row))
    return row[:width]


def row_contains_hangul(row: Iterable[str]) -> bool:
    return any(HANGUL_RE.search(value or "") for value in row)


def clean_rows(rows: list[list[str]]) -> tuple[list[str], list[list[str]], int]:
    if not rows:
        raise ValueError("The worksheet is empty.")

    header = rows[0]
    missing = [column for column in COLUMNS_TO_REMOVE if column not in header]
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")

    keep_indices = [i for i, name in enumerate(header) if name not in COLUMNS_TO_REMOVE]
    cleaned_header = [header[i] for i in keep_indices]

    removed_for_korean = 0
    cleaned_rows: list[list[str]] = []
    for raw_row in rows[1:]:
        full_row = normalize_row(raw_row, len(header))
        if row_contains_hangul(full_row):
            removed_for_korean += 1
            continue
        cleaned_rows.append([full_row[i] for i in keep_indices])

    return cleaned_header, cleaned_rows, removed_for_korean


def write_csv(output_path: Path, header: list[str], rows: list[list[str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    if args.input.suffix.lower() != ".xlsx":
        print("Input must be an .xlsx workbook.", file=sys.stderr)
        return 1

    try:
        with zipfile.ZipFile(args.input) as archive:
            shared_strings = load_shared_strings(archive)
            sheet_path = resolve_sheet_path(archive, args.sheet)
            rows = list(iter_sheet_rows(archive, sheet_path, shared_strings))
    except FileNotFoundError:
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1
    except (KeyError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Failed to read workbook: {exc}", file=sys.stderr)
        return 1

    cleaned_header, cleaned_rows, removed_for_korean = clean_rows(rows)
    write_csv(args.output, cleaned_header, cleaned_rows)

    original_row_count = max(len(rows) - 1, 0)
    final_row_count = len(cleaned_rows)

    print(f"Original row count: {original_row_count}")
    print(f"Rows removed for Korean text: {removed_for_korean}")
    print(f"Columns removed: {', '.join(COLUMNS_TO_REMOVE)}")
    print(f"Final row count: {final_row_count}")
    print(f"Output path: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
