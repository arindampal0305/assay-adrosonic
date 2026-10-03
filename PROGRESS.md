# ASSAY Progress Log

## Milestone Summary

| Milestone | Description | Status | Date Completed |
|---|---|---|---|
| M1 | Project skeleton, SOVState, LangGraph skeleton, contract gates, FastAPI base, Agent 1 | ✅ Completed | 2026-10-02 |
| M2 | Agent 2: three-channel mapping, Hungarian solver, LLM adjudicator | ✅ Completed | 2026-10-03 |
| M3 | Agent 3: quality rules (DQ-01 to DQ-18), recommendations, rationale verifier | ✅ Completed | 2026-10-04 |
| M4a | LangGraph interrupt gate, SQLite checkpointer, REST API & decision endpoints, SSE | ✅ Completed | 2026-10-04 |
| M4b | Agent 4: Controlled transformation, audit log, output schema verification, ChromaDB memory | ✅ Completed | 2026-10-04 |
| M5 | Review UI integration, mutation harness, performance, Docker, demo script | ✅ Completed | 2026-10-04 |

---

## Progress Details

### M3: Agent 3 — Data Quality & Reasoning (Completed & Verified)
- Implemented full DQ-01 to DQ-18 rule catalogue in `backend/agents/quality/`.
- Strict Pydantic models for `Recommendation` and `QualityBlock`.
- Rule-based detection and engine-computed worked before/after examples (C-02 compliant).
- LLM rationale verifier with strict evidence check (`reasoner.py`).

### M4: Human Gate & Controlled Transformation (Completed & Verified)
- LangGraph `interrupt()` gate preventing unapproved transformations (C-01 compliant).
- Full REST & SSE FastAPI backend (`/api/runs`, `/api/runs/{id}/decisions`, `/api/runs/{id}/stream`).
- Agent 4 transformation engine with 12 whitelisted DSL ops and 0 LLM calls (C-01, C-02, C-03).
- Outputs exact 17-field SRS 5.1 target schema (`Cleaned_SOV.xlsx`) and `Audit_Log.xlsx`.
- Pandera schema self-check verification post-export.
- Local persistent ChromaDB vector memory for positive/negative decision feedback.
- Interactive HTML/JS review UI with side-by-side before/after preview, intake quality card, and downloads.

### M5: Packaging, Evaluation Harness & Final Integration (Completed & Verified)
- Containerized deployment with `Dockerfile` and `docker-compose.yml`.
- End-to-end evaluation harness (`scripts/run_mutation_harness.py`) evaluating Mapping Accuracy (≥ 85%), Quality Recall (≥ 95%), and Transformation Correctness (100%).
- Practice SOV generation script (`scripts/make_samples.py`) and multi-file runner (`scripts/run_samples.py`).
- Updated project documentation and judge-ready setup instructions in `README.md`.
