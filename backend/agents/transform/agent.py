"""Agent 4: Controlled Transformation (SRS 4.6, FR-TRN-01 to FR-TRN-06).

Applies human-approved recommendations deterministically on a copy of the input file.
Zero LLM calls during execution (C-01, C-02, C-03).
"""

from __future__ import annotations

from typing import Any

from backend.agents.transform.engine import run_transformation_pipeline
from backend.state.gates import contract_gate
from backend.state.sov_state import SOVState


@contract_gate("transformation")
def transformation(state: SOVState) -> dict[str, Any]:
    """Execute Agent 4 transformation pipeline and record audit trail."""
    result_metadata, audit_entries = run_transformation_pipeline(state)

    audit_dicts = [a.model_dump() if hasattr(a, "model_dump") else a for a in audit_entries]

    return {
        "audit": audit_dicts,
        "transformation_result": result_metadata,
    }
