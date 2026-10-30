#!/usr/bin/env python3
"""Run full KTAS error analysis for analysis3 wide-format prediction CSV files."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_analysis_module():
    source_path = Path(__file__).resolve().parents[1] / "analysis" / "full_ktas_error_analysis.py"
    spec = importlib.util.spec_from_file_location("analysis_full_ktas_error_analysis", source_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load analysis module from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_analysis_module()
    module.DEFAULT_OUTPUT_DIR = Path("scripts/analysis3/output/full_error_analysis_models")
    module.EXCLUDED_PREDICTION_HINTS = set(module.EXCLUDED_PREDICTION_HINTS) | {
        "clinicalsummary",
        "ktasjustification",
    }
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
