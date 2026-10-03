# ASSAY

**Agentic SOV Cleansing & Intelligence System**

Commercial property insurance runs on the Statement of Values — a spreadsheet listing every insured location and what it is worth. In practice every broker sends a different shape: headers buried under title blocks, merged banner rows, `BI/EE` instead of `Business Income`, four-digit ZIPs that lost a leading zero, `1900` standing in for "we don't know when it was built". Analysts normalise these by hand, and it is slow, dull, and error-prone.

ASSAY reads a messy SOV and produces a clean 17-column one — but it never silently rewrites a customer's numbers. The governing principle is:

> **The LLM proposes, code disposes.**

A model may suggest that `Bldg Repl Cost` means `Building Value`. It may not perform the rename. Every change is a typed, reviewable recommendation that deterministic code applies only after it passes validation and, where it matters, human approval. That is what makes the output defensible to an underwriter.

---

## Architecture

Four specialised agents run in sequence over one shared, typed state object.

```
                        ┌──────────────────────────────────────────┐
   upload .xlsx/.csv    │               SOVState                   │
          │             │  run_id · version · source · manifest    │
          ▼             │  mapping · issues · recommendations      │
   ┌─────────────┐      │  decisions · audit · trace               │
   │   FastAPI   │      └──────────────────────────────────────────┘
   │ POST /runs  │                        ▲
   └──────┬──────┘              every agent reads and writes
          │                         this one object
          ▼
   ╔══════════════╗   ╔══════════════╗   ╔══════════════╗   ╔══════════════╗
═══║    gate      ║═══║    gate      ║═══║    gate      ║═══║    gate      ║═══
   ╚══════╤═══════╝   ╚══════╤═══════╝   ╚══════╤═══════╝   ╚══════╤═══════╝
          ▼                  ▼                  ▼                  ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │   Agent 1   │    │   Agent 2   │    │   Agent 3   │    │   Agent 4   │
   │    Sheet    │ →  │   Schema    │ →  │    Data     │ →  │ Controlled  │
   │Intelligence │    │   Mapping   │    │   Quality   │    │Transformation│
   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       REAL              stub               stub               stub
```

**Agent 1 — Sheet Intelligence.** Opens the workbook, scores every sheet, finds the header row wherever it hides, rebuilds merged banner headers, and decides which sheet actually holds the location schedule.

**Agent 2 — Schema Mapping.** Maps the source columns onto the 17 target fields using an insurance glossary and fuzzy matching, proposing a confidence per column.

**Agent 3 — Data Quality.** Applies validation rules row by row — ZIP length, impossible years, negative values, state codes, placeholder detection — and raises typed issues with recommendations.

**Agent 4 — Controlled Transformation.** Applies only the approved recommendations to a *copy* of the source and writes the cleaned workbook plus an audit log.

### Contract gates

The gates in that diagram are the heart of the design. Before any agent runs, its gate:

1. re-validates the **entire** state against the `SOVState` schema and checks the version, and
2. asserts the preconditions that *this* agent depends on.

Either failure raises `ContractViolation` and the run stops with a readable message. No agent ever executes against malformed state, which means a bug in one agent cannot quietly corrupt the next. This is also what lets Agents 2–4 exist as stubs today without weakening the skeleton — the contracts are already enforced around them.

### The state object

One Pydantic model, `SOVState`, threaded through the whole graph. Agents never import each other and never talk directly; they communicate only by reading and writing this object. Adding a real agent means replacing a stub — no rewiring.

---

## Workflow

```
1. Upload            An analyst uploads an .xlsx or .csv through the web UI or POST /api/runs.
2. Ingest            The file is hashed (sha256), stored locally, and read — openpyxl for Excel
                     so merged cells survive, stdlib csv for CSV so ragged title rows parse.
3. Sheet selection   Agent 1 scores every sheet on five factors and names exactly one Primary.
                     Header rows are located anywhere in the first 30 rows; merged banners are
                     folded into composite names like "Values › Building".
4. Mapping           Agent 2 proposes a source-column → target-field mapping with confidences.
5. Quality           Agent 3 runs validation rules and raises issues with recommended fixes.
6. Review            Low-confidence mappings and flagged rows go to a human, who accepts,
                     rejects or edits each recommendation. Decisions are recorded in state.
7. Transform         Agent 4 applies only approved changes to a copy of the source.
8. Output            Cleaned_SOV.xlsx plus Audit_Log.xlsx — every change traceable to a rule
                     and a decision.
```

Steps 1–3 and 8's API surface are implemented. Steps 4–7 are the next milestone.

---

## Setting up locally

### Prerequisites

- **Python 3.11+** (developed on 3.12.10)
- **git**

### 1. Clone

```bash
git clone https://github.com/arindampal0305/assay.git
cd assay
```

### 2. Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Git Bash)**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env     # Windows: copy .env.example .env
```

No LLM is called in the current milestone, so the placeholder keys can stay empty. `.env` is gitignored — never commit it.

### 5. Generate the sample SOVs

```bash
python scripts/make_samples.py
```

These are hand-built practice files reproducing the mess real SOVs contain. Drop real files into `samples/` and everything below picks them up unchanged.

---

## Running

### Web interface

From the **repository root** (the server resolves `frontend/` relative to the working directory):

```bash
python -m uvicorn backend.api.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. Upload an SOV and the manifest comes back as JSON.
Interactive API docs are at <http://127.0.0.1:8000/docs>.

### Command line

Run every sample through the graph and print its manifest:

```bash
python scripts/run_samples.py
```

### API directly

```bash
curl -X POST http://127.0.0.1:8000/api/runs -F "file=@samples/sample3_multisheet_merged.xlsx"
```

Response:

```json
{
  "run_id": "223f822d-…",
  "version": 7,
  "source": { "file_name": "sample3_multisheet_merged.xlsx", "sha256": "a2770b…", "sheets": 4 },
  "manifest": [
    {
      "sheet": "Location Schedule",
      "class": "Primary",
      "confidence": 0.969,
      "header_row": 2,
      "composite_header": true,
      "headers": ["Loc Ref", "Street Address", "…", "Values › Building", "…"],
      "factor_scores": { "header_density": 0.918, "vocabulary_overlap": 0.824, "…": 1.0 },
      "reasons": ["highest sheet score 0.935, margin 0.067 over the runner-up", "…"]
    }
  ],
  "trace": [{ "agent": "sheet_intelligence", "event": "finished", "ms": 458 }]
}
```

`header_row` is **0-based**, so `pd.read_excel(path, header=header_row)` works directly. The `reasons` strings quote the 1-based Excel row a human would see.

Bad input returns `422` with a message an analyst can act on:

```json
{ "detail": { "code": "ingest_error",
              "message": "'notes.txt' has an unsupported type (.txt). ASSAY accepts .xlsx and .csv only." } }
```

---

## Project structure

```
assay/
├── backend/
│   ├── state/
│   │   ├── sov_state.py        SOVState — the typed object every agent shares
│   │   ├── gates.py            contract gates: type check + per-agent preconditions
│   │   └── target_schema.py    the 17 target fields and the vocabulary that finds them
│   ├── agents/
│   │   ├── sheet_intel/        Agent 1 — fully implemented
│   │   │   ├── loader.py       .xlsx/.csv ingestion, merged-cell forward fill
│   │   │   ├── cells.py        cell typing and target-field matching
│   │   │   ├── header.py       header-row detection, composite headers
│   │   │   ├── scoring.py      five-factor scoring, Primary/Secondary/Reject
│   │   │   └── agent.py        the graph node
│   │   ├── mapping/            Agent 2 — stub
│   │   ├── quality/            Agent 3 — stub
│   │   └── transform/          Agent 4 — stub
│   ├── api/main.py             FastAPI app, POST /api/runs, serves the frontend
│   ├── graph.py                the LangGraph StateGraph
│   ├── llm/                    (empty — no LLM wired yet)
│   └── harness/                (empty — evaluation harness to come)
├── frontend/index.html         single-file UI, posts to /api/runs
├── samples/                    practice SOVs
├── scripts/
│   ├── make_samples.py         regenerates the practice SOVs
│   └── run_samples.py          runs every sample through the graph
├── project documentation/
│   └── MILESTONE-1.md          what shipped in milestone 1, with real output
└── requirements.txt
```

---

## The target schema

Every cleaned SOV is normalised to these 17 columns (SRS 5.1):

`Reference` · `Address` · `City` · `State` · `Zip` · `County` · `Country` · `Building Value` · `Contents` · `BI` · `Occupancy` · `Construction` · `Storeys` · `Number of Buildings` · `Year Built` · `Fire Sprinklers (Y/N)` · `Other`

---

## Status

| Milestone | Scope | Status |
|---|---|---|
| **1** | Skeleton, `SOVState`, contract gates, Agent 1, API | ✅ complete |
| 2 | Agent 2 schema mapping, insurance glossary, LLM | planned |
| 3 | Agent 3 data quality rules and recommendations | planned |
| 4 | Agent 4 transformation, audit log, review UI | planned |

See [`project documentation/MILESTONE-1.md`](project%20documentation/MILESTONE-1.md) for what milestone 1 delivered, including real output and the defects that running real files exposed.

---

## Notes for contributors

- **Agents never import each other.** Shared truth lives in `backend/state/`.
- **An agent is a pure function:** `SOVState` in, a partial dict out. The gate decorator handles validation and tracing.
- **Adding a real agent** means replacing a stub body. The graph and gates do not change.
- **CORS is currently open** (`allow_origins=["*"]`) for local development. Narrow it before any deployment.
