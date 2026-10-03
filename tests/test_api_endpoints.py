"""Unit tests for FastAPI endpoints (REST, Human Review, Download, SSE)."""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_serve_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "ASSAY" in response.text


def test_create_run_empty_file():
    response = client.post(
        "/api/runs",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "ingest_error"


def test_create_run_and_human_review_flow(tmp_path):
    sample_file = Path("samples/sample1_basic.csv")
    if not sample_file.exists():
        pytest.skip("sample1_basic.csv not found")

    with sample_file.open("rb") as fh:
        response = client.post(
            "/api/runs",
            files={"file": ("sample1_basic.csv", fh, "text/csv")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["status"] in ("awaiting_review", "completed")
    run_id = data["run_id"]

    # Retrieve run state
    get_resp = client.get(f"/api/runs/{run_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["run_id"] == run_id

    # If recommendations are pending, submit decisions
    recs = data.get("recommendations") or []
    if recs:
        decisions = [
            {
                "rec_id": r.get("id", "R-001"),
                "action": "accept",
                "note": "Unit test approval",
                "by": "test_suite",
            }
            for r in recs
        ]

        dec_resp = client.post(
            f"/api/runs/{run_id}/decisions",
            json={"decisions": decisions},
        )
        assert dec_resp.status_code == 200
        final_data = dec_resp.json()
        assert final_data["status"] == "completed"

        # Test preview data endpoint
        prev_resp = client.get(f"/api/runs/{run_id}/preview")
        assert prev_resp.status_code == 200
        prev_data = prev_resp.json()
        assert "headers" in prev_data
        assert "sample_rows" in prev_data
        assert len(prev_data["sample_rows"]) > 0

        # Test download links
        sov_dl = client.get(f"/api/runs/{run_id}/download/sov")
        assert sov_dl.status_code == 200

        audit_dl = client.get(f"/api/runs/{run_id}/download/audit")
        assert audit_dl.status_code == 200
