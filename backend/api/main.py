"""FastAPI backend application for ASSAY: SOV Cleansing System.

REST API + Server-Sent Events (SSE) streaming + Human-in-the-Loop review endpoints.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel

from backend.graph import COMPILED_GRAPH, initial_state, run_pipeline
from backend.ingest.loader import IngestError
from backend.state.gates import ContractViolation
from backend.state.sov_state import SOVState

UPLOAD_DIR = Path(os.getenv("ASSAY_UPLOAD_DIR", "data/uploads"))
OUTPUT_DIR = Path("data/outputs")
FRONTEND_DIR = Path("frontend")

app = FastAPI(title="ASSAY", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-load local embedding model during server startup so file uploads run instantly."""
    try:
        from backend.agents.mapping.channels import semantic
        semantic.is_available()
    except Exception:
        pass


class DecisionItem(BaseModel):
    rec_id: str
    action: str  # accept | reject | edit
    note: Optional[str] = ""
    by: Optional[str] = "human_reviewer"
    edited_op: Optional[str] = None


class DecisionsPayload(BaseModel):
    decisions: list[DecisionItem]


# Store active graph states by run_id in memory
RUN_STATES: dict[str, SOVState] = {}


@app.get("/", response_class=FileResponse)
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")


@app.post("/api/runs")
async def create_run(file: UploadFile = File(...)):
    """Upload SOV workbook and start pipeline up to Human Review Gate."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix.lower()
    payload = await file.read()
    if not payload:
        raise HTTPException(
            status_code=422,
            detail={"code": "ingest_error", "message": f"'{file.filename}' is empty (0 bytes)."},
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as handle:
        handle.write(payload)
        stored = Path(handle.name)

    try:
        init_st = initial_state(stored, file.filename)
        run_id = init_st.run_id
        config = {"configurable": {"thread_id": run_id}}

        # Invoke graph until interrupt or completion
        raw_result = COMPILED_GRAPH.invoke(init_st, config=config)

        state_snapshot = COMPILED_GRAPH.get_state(config)
        next_nodes = state_snapshot.next if state_snapshot else ()
        is_interrupted = bool(next_nodes)

        current_val = state_snapshot.values if (state_snapshot and state_snapshot.values) else raw_result
        if isinstance(current_val, SOVState):
            state = current_val
        else:
            state = SOVState.model_validate(current_val)

        RUN_STATES[run_id] = state

    except IngestError as exc:
        raise HTTPException(status_code=422, detail={"code": "ingest_error", "message": str(exc)})
    except ContractViolation as exc:
        raise HTTPException(status_code=422, detail={"code": "contract_violation", "message": str(exc)})

    return _serialize_state(state, is_interrupted=is_interrupted)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Fetch current state and recommendations for a run."""
    config = {"configurable": {"thread_id": run_id}}
    state_snapshot = COMPILED_GRAPH.get_state(config)

    if not state_snapshot or not state_snapshot.values:
        if run_id in RUN_STATES:
            return _serialize_state(RUN_STATES[run_id], is_interrupted=False)
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    current_val = state_snapshot.values
    state = current_val if isinstance(current_val, SOVState) else SOVState.model_validate(current_val)
    is_interrupted = bool(state_snapshot.next)

    return _serialize_state(state, is_interrupted=is_interrupted)


@app.post("/api/runs/{run_id}/decisions")
async def submit_decisions(run_id: str, payload: DecisionsPayload = Body(...)):
    """Submit reviewer decisions and resume graph execution to completion (Agent 4)."""
    config = {"configurable": {"thread_id": run_id}}
    state_snapshot = COMPILED_GRAPH.get_state(config)

    if not state_snapshot:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    decisions_data = [d.model_dump() for d in payload.decisions]

    try:
        # Resume graph execution with decisions
        raw_result = COMPILED_GRAPH.invoke(Command(resume=decisions_data), config=config)
        state_snapshot = COMPILED_GRAPH.get_state(config)

        current_val = state_snapshot.values if (state_snapshot and state_snapshot.values) else raw_result
        if isinstance(current_val, dict) and "audit" in current_val and current_val["audit"]:
            current_val["audit"] = [a.model_dump() if hasattr(a, "model_dump") else a for a in current_val["audit"]]

        state = current_val if isinstance(current_val, SOVState) else SOVState.model_validate(current_val)
        is_interrupted = bool(state_snapshot.next if state_snapshot else ())

        RUN_STATES[run_id] = state

    except ContractViolation as exc:
        raise HTTPException(status_code=422, detail={"code": "contract_violation", "message": str(exc)})

    return _serialize_state(state, is_interrupted=is_interrupted)


@app.get("/api/runs/{run_id}/preview")
async def get_run_data_preview(run_id: str):
    """Fetch raw uploaded file rows and sheet preview for the data viewer."""
    config = {"configurable": {"thread_id": run_id}}
    state_snapshot = COMPILED_GRAPH.get_state(config)

    state = None
    if state_snapshot and state_snapshot.values:
        val = state_snapshot.values
        if isinstance(val, dict) and "audit" in val and val["audit"]:
            val["audit"] = [a.model_dump() if hasattr(a, "model_dump") else a for a in val["audit"]]
        state = val if isinstance(val, SOVState) else SOVState.model_validate(val)
    elif run_id in RUN_STATES:
        state = RUN_STATES[run_id]

    if not state or not state.source.file_path:
        raise HTTPException(status_code=404, detail=f"Preview data for run '{run_id}' not found.")

    file_path = Path(state.source.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Source file no longer exists on disk.")

    primary = state.primary_sheet()
    sheet_name = primary.sheet if primary else None
    header_row = primary.header_row if (primary and primary.header_row is not None) else 0

    try:
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path, header=header_row)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name or 0, header=header_row)

        df = df.fillna("")
        headers = [str(c) for c in df.columns]
        raw_rows = df.head(100).to_dict(orient="records")
        sample_rows = [
            {str(k): (str(v) if v is not None else "") for k, v in row.items()}
            for row in raw_rows
        ]
        total_rows = len(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse preview data: {exc}")

    mapping_dict = state.mapping.model_dump() if hasattr(state.mapping, "model_dump") else (state.mapping if isinstance(state.mapping, dict) else {})
    mappings_list = mapping_dict.get("mappings", []) or mapping_dict.get("mapping", {}).get("mappings", [])
    if not mappings_list and isinstance(mapping_dict.get("mappings"), list):
        mappings_list = mapping_dict["mappings"]

    return {
        "run_id": run_id,
        "sheet_name": sheet_name,
        "total_rows": total_rows,
        "headers": headers,
        "sample_rows": sample_rows,
        "mappings": mappings_list,
    }


@app.get("/api/runs/{run_id}/stream")
async def stream_run_trace(run_id: str):
    """Server-Sent Events (SSE) endpoint streaming execution trace and status."""
    async def event_generator() -> AsyncGenerator[str, None]:
        config = {"configurable": {"thread_id": run_id}}
        state_snapshot = COMPILED_GRAPH.get_state(config)

        if state_snapshot and state_snapshot.values:
            val = state_snapshot.values
            state = val if isinstance(val, SOVState) else SOVState.model_validate(val)
            for trace_event in state.trace:
                yield f"data: {json.dumps(trace_event.model_dump())}\n\n"
                await asyncio.sleep(0.05)

        yield "data: {\"event\": \"stream_end\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}/download/sov")
async def download_cleaned_sov(run_id: str):
    """Download output Cleaned_SOV.xlsx."""
    file_path = OUTPUT_DIR / "Cleaned_SOV.xlsx"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Cleaned_SOV.xlsx has not been generated yet.")
    return FileResponse(file_path, filename="Cleaned_SOV.xlsx")


@app.get("/api/runs/{run_id}/download/audit")
async def download_audit_log(run_id: str):
    """Download output Audit_Log.xlsx."""
    file_path = OUTPUT_DIR / "Audit_Log.xlsx"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audit_Log.xlsx has not been generated yet.")
    return FileResponse(file_path, filename="Audit_Log.xlsx")


def _serialize_state(state: SOVState, is_interrupted: bool = False) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "version": state.version,
        "status": "awaiting_review" if is_interrupted else "completed",
        "is_interrupted": is_interrupted,
        "source": state.source.model_dump(exclude={"file_path"}),
        "manifest": [entry.model_dump(by_alias=True) for entry in state.manifest],
        "mapping": state.mapping,
        "quality": state.quality,
        "recommendations": state.recommendations,
        "decisions": [d.model_dump() if hasattr(d, "model_dump") else d for d in state.decisions],
        "issues": [issue.model_dump() for issue in state.issues],
        "audit": [a.model_dump() if hasattr(a, "model_dump") else a for a in state.audit],
        "trace": [event.model_dump() for event in state.trace],
    }
