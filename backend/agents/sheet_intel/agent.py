"""Agent 1: Sheet Intelligence (SRS 4.2)."""

from __future__ import annotations

from typing import Any

from backend.agents.sheet_intel.loader import load_workbook_file
from backend.agents.sheet_intel.scoring import classify_sheets, score_sheet
from backend.state.gates import contract_gate
from backend.state.sov_state import SheetManifestEntry, SOVState, SourceInfo


@contract_gate("sheet_intelligence")
def sheet_intelligence(state: SOVState) -> dict[str, Any]:
    workbook = load_workbook_file(state.source.file_path, state.source.file_name)
    classified = classify_sheets([score_sheet(sheet) for sheet in workbook.sheets])

    manifest = [
        SheetManifestEntry(
            sheet=item.sheet.name,
            **{"class": sheet_class},
            confidence=confidence,
            header_row=item.detection.header_row if item.detection else None,
            reasons=item.reasons,
            score=item.score,
            factor_scores=item.factor_scores,
            headers=item.detection.headers if item.detection else [],
            data_start_row=item.detection.data_start_row if item.detection else None,
            data_rows=item.data_rows,
            composite_header=bool(item.detection and item.detection.composite),
        )
        for item, sheet_class, confidence in classified
    ]

    source = SourceInfo(
        file_name=workbook.file_name,
        sha256=workbook.sha256,
        sheets=len(workbook.sheets),
        file_path=state.source.file_path,
    )
    return {"source": source, "manifest": manifest}
