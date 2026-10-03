"""Unit tests for Agent 4 Controlled Transformation Engine (SRS 9.1, C-01, C-02, C-03)."""

from __future__ import annotations

from pathlib import Path
import openpyxl
import pytest

from backend.agents.transform.engine import (
    apply_op_to_cell,
    run_transformation_pipeline,
)
from backend.state.sov_state import SOVState, SourceInfo, SheetManifestEntry
from backend.state.target_schema import TARGET_FIELD_NAMES


def test_apply_op_to_cell_currency():
    val, text = apply_op_to_cell("strip_currency_to_float", "$2,500,000.00")
    assert val == 2500000.0
    assert text == "2500000"


def test_apply_op_to_cell_zip5():
    val, text = apply_op_to_cell("zip5", "2134")
    assert val == "02134"
    assert text == "02134"

    val2, text2 = apply_op_to_cell("zip5", "02134-1234")
    assert val2 == "02134"
    assert text2 == "02134"


def test_apply_op_to_cell_state_abbrev():
    val, text = apply_op_to_cell("state_to_abbrev", "New York")
    assert val == "NY"
    assert text == "NY"

    val2, text2 = apply_op_to_cell("state_to_abbrev", "tx")
    assert val2 == "TX"
    assert text2 == "TX"


def test_apply_op_to_cell_set_null():
    val, text = apply_op_to_cell("set_null", "1900")
    assert val is None
    assert text == ""


def test_transformation_pipeline_execution(tmp_path):
    sample_file = Path("samples/sample1_basic.csv")
    if not sample_file.exists():
        pytest.skip("sample1_basic.csv not found")

    state = SOVState(
        source=SourceInfo(
            file_name="sample1_basic.csv",
            sha256="a" * 64,
            file_path=str(sample_file),
        ),
        manifest=[
            SheetManifestEntry(
                sheet="sample1_basic",
                class_="Primary",
                confidence=1.0,
                header_row=0,
                data_start_row=1,
                data_rows=12,
                headers=[
                    "Loc #", "Street Address", "City", "St", "Zip",
                    "Bldg Repl Cost", "Contents", "BI/EE", "Occ", "Const",
                    "Stories", "# Bldgs", "Yr Blt", "Sprk"
                ]
            )
        ],
        mapping={
            "mappings": [
                {"source_column": "Street Address", "source_index": 1, "target": "Address"},
                {"source_column": "City", "source_index": 2, "target": "City"},
                {"source_column": "St", "source_index": 3, "target": "State"},
                {"source_column": "Zip", "source_index": 4, "target": "Zip"},
                {"source_column": "Bldg Repl Cost", "source_index": 5, "target": "Building Value"},
                {"source_column": "Contents", "source_index": 6, "target": "Contents"},
            ]
        },
        quality={"intake_quality_score": 95.0},
        recommendations=[
            {
                "id": "R-001",
                "op": "zip5",
                "field": "Zip",
                "source_column": "Zip",
                "rows": [1, 2],
                "status": "approved",
                "rationale": "Padded zip code to 5 digits",
            },
            {
                "id": "R-002",
                "op": "state_to_abbrev",
                "field": "State",
                "source_column": "St",
                "rows": [1],
                "status": "approved",
                "rationale": "Normalised state abbreviation",
            }
        ],
        decisions=[]
    )

    result, audit = run_transformation_pipeline(state, output_dir=tmp_path)

    assert result["status"] == "completed"
    assert result["self_check"]["exact_order_matched"] is True

    cleaned_path = tmp_path / "Cleaned_SOV.xlsx"
    assert cleaned_path.exists()

    wb = openpyxl.load_workbook(cleaned_path)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]
    assert tuple(headers) == TARGET_FIELD_NAMES
