"""Minimal FastAPI entrypoint: upload a file, run the graph, return the manifest."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.agents.sheet_intel.loader import IngestError
from backend.graph import run_pipeline
from backend.state.gates import ContractViolation

UPLOAD_DIR = Path(os.getenv("ASSAY_UPLOAD_DIR", "data/uploads"))

app = FastAPI(title="ASSAY", version="0.1.0")


@app.post("/api/runs")
async def create_run(file: UploadFile = File(...)):
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
        state = run_pipeline(stored, file.filename)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail={"code": "ingest_error", "message": str(exc)})
    except ContractViolation as exc:
        raise HTTPException(
            status_code=422, detail={"code": "contract_violation", "message": str(exc)}
        )

    return {
        "run_id": state.run_id,
        "version": state.version,
        "source": state.source.model_dump(exclude={"file_path"}),
        "manifest": [entry.model_dump(by_alias=True) for entry in state.manifest],
        "trace": [event.model_dump() for event in state.trace],
    }
