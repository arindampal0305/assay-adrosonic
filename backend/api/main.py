"""Minimal FastAPI entrypoint: upload a file, run the graph, return the manifest."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.ingest.loader import IngestError
from backend.graph import run_pipeline
from backend.state.gates import ContractViolation

UPLOAD_DIR = Path(os.getenv("ASSAY_UPLOAD_DIR", "data/uploads"))
FRONTEND_DIR = Path("frontend")

app = FastAPI(title="ASSAY", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", response_class=FileResponse)
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")



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
        "mapping": state.mapping,
        "issues": [issue.model_dump() for issue in state.issues],
        # Exposed so a reviewer can see every LLM proposal that was discarded, not
        # just the mappings that survived.
        "audit": state.audit,
        "trace": [event.model_dump() for event in state.trace],
    }
