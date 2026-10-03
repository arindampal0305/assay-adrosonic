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
       REAL              REAL               stub               stub
```

**Agent 1 — Sheet Intelligence.** Opens the workbook, scores every sheet, finds the header row wherever it hides, rebuilds merged banner headers, and decides which sheet actually holds the location schedule.

**Agent 2 — Schema Mapping.** Maps the source columns onto the 17 target fields using three independent evidence channels — a 321-phrase insurance glossary matched with RapidFuzz, local `bge-small` embeddings, and a profile of what is actually in the column — fused and then assigned globally with the Hungarian algorithm. Genuinely ambiguous columns are referred to an LLM whose every citation is re-checked against the evidence before it may change anything.

**Agent 3 — Data Quality.** Applies validation rules row by row — ZIP length, impossible years, negative values, state codes, placeholder detection — and raises typed issues with recommendations.

**Agent 4 — Controlled Transformation.** Applies only the approved recommendations to a *copy* of the source and writes the cleaned workbook plus an audit log.

### Contract gates

The gates in that diagram are the heart of the design. Before any agent runs, its gate:

1. re-validates the **entire** state against the `SOVState` schema and checks the version, and
2. asserts the preconditions that *this* agent depends on.

Either failure raises `ContractViolation` and the run stops with a readable message. No agent ever executes against malformed state, which means a bug in one agent cannot quietly corrupt the next. This is also what lets Agents 3–4 exist as stubs today without weakening the skeleton — the contracts are already enforced around them, and they can no longer run against an empty mapping.

### How the LLM is contained

Agent 2 is the first place a model participates in a decision, so it is worth being precise about what "code disposes" means mechanically. The model is consulted only for columns the deterministic scorer could not separate confidently, it is offered exactly two candidates, and it must cite its reasoning: a glossary phrase, a value shape, a sample value from the column. Code then **re-derives all three** from evidence it already holds. A proposal citing a real glossary phrase that belongs to a different target, or a sample value that does not appear in the data, is discarded and recorded.

Every proposal — accepted or rejected — is appended to an append-only `audit` list and returned by the API, so a reviewer can see what was *refused*, not only what survived.

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
4. Mapping           Agent 2 scores every source column against all 17 targets on three
                     channels, fuses them, and assigns globally so no two columns can claim
                     the same target. Each mapping carries a confidence, the deciding
                     evidence, the runner-up it beat, and — below 0.70 — a mandatory
                     statement of what is uncertain about it.
5. Quality           Agent 3 runs validation rules and raises issues with recommended fixes.
6. Review            Low-confidence mappings and flagged rows go to a human, who accepts,
                     rejects or edits each recommendation. Decisions are recorded in state.
7. Transform         Agent 4 applies only approved changes to a copy of the source.
8. Output            Cleaned_SOV.xlsx plus Audit_Log.xlsx — every change traceable to a rule
                     and a decision.
```

Steps 1–4 and 8's API surface are implemented. Steps 5–7 are the next milestones.

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

This pulls `sentence-transformers` and a CPU build of `torch`, which is the bulk of the install. No GPU is needed — the embedding model is small and runs on CPU in milliseconds.

### 4. Configure environment

```bash
cp .env.example .env     # Windows: copy .env.example .env
```

`.env` is gitignored — never commit it.

Adjudication is **optional**, and there are two ways to supply it:

| Config | Result |
| --- | --- |
| `OPENAI_API_KEY=sk-…` | OpenAI, model from `ASSAY_LLM_MODEL` |
| `ASSAY_LLM_PROVIDER=ollama` with the key empty | Local Ollama at `OLLAMA_BASE_URL`, model from `ASSAY_OLLAMA_MODEL` |
| neither | No adjudication |

A non-empty `OPENAI_API_KEY` always wins, so leave it blank to use Ollama; asking for Ollama with a key set logs a warning and bills OpenAI. Ollama is probed once at startup — if the daemon is down or the model is not pulled, the run falls back to no adjudication rather than failing per column. Local inference is slow enough to need `ASSAY_LLM_TIMEOUT` raised well above its 20 s default; llama3.1 on CPU took ~6–7 s per column once warm.

With no provider, mapping runs fully deterministically on its three local channels, ambiguous columns are flagged for review rather than referred to a model, and the run reports an `adjudicator_unavailable` issue plus an audit record naming the columns that went unadjudicated. Nothing else in the pipeline calls out to the network.

### 5. First run downloads the embedding model

The semantic channel uses `BAAI/bge-small-en-v1.5` (~130 MB), fetched from the HuggingFace Hub on first use and cached in `~/.cache/huggingface` thereafter. The first pipeline run therefore takes around 20 seconds; subsequent runs take ~50 ms. If the model cannot be downloaded, mapping still works — the channel reports itself unavailable and fusion reweights onto the other two.

### 6. Generate the sample SOVs

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
  "mapping": {
    "sheet_identified": "Location Schedule",
    "header_row": 2,
    "overall_confidence": 0.8429,
    "unresolved_count": 0,
    "review_required_count": 0,
    "unmapped_targets": ["County", "Country", "Other"],
    "semantic_channel_available": true,
    "low_margin_count": 2,
    "adjudicator_available": false,
    "adjudicator_consulted": 0,
    "adjudicator_accepted": 0,
    "adjudicator_rejected": 0,
    "mappings": [
      {
        "source_column": "Values › Building",
        "source_index": 5,
        "target": "Building Value",
        "confidence": 0.8461,
        "method": "fused",
        "channels": { "lexical": 1.0, "semantic": 0.7138, "fingerprint": 1.0, "fused": 0.8998,
                      "contributing": ["lexical", "semantic", "fingerprint"] },
        "runner_up": { "target": "Contents", "score": 0.7339 },
        "evidence": { "glossary_phrase": "building", "lexical_exact": false,
                      "semantic_cosine": 0.6712, "dominant_value_shape": "currency",
                      "dominant_shape_share": 1.0,
                      "sample_values": ["5600000", "8900000", "1450000", "980000", "3250000"],
                      "fingerprint_abstained": false },
        "rationale": "mapped to 'Building Value' because the header resembles the glossary phrase 'building' (1.00); the values are 100% 'currency', a shape 'Building Value' accepts; embedding similarity is 0.71. It beat 'Contents' by 0.17",
        "uncertainty": null,
        "flag": "review_suggested",
        "adjudicated": false
      }
    ]
  },
  "issues": [],
  "audit": [],
  "trace": [{ "agent": "sheet_intelligence", "event": "finished", "ms": 458 }]
}
```

`header_row` is **0-based**, so `pd.read_excel(path, header=header_row)` works directly. The `reasons` strings quote the 1-based Excel row a human would see.

`audit` holds one record per LLM consultation, including every proposal that was **rejected** and the specific checks it failed. It is empty above because no provider was configured — note `adjudicator_available: false` beside `low_margin_count: 2`. Those two columns *were* eligible: they scored within 0.15 of their runner-up and would have been referred to a model had one been available. When no provider is configured the run says so, via an `adjudicator_unavailable` issue and an `adjudication_skipped` audit record naming the columns. `low_margin_count` is reported separately from `adjudicator_consulted` precisely so "nothing needed adjudicating" and "something did and no model was available" cannot be confused.

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
│   ├── ingest/
│   │   └── loader.py           .xlsx/.csv ingestion, merged-cell forward fill —
│   │                           shared, because Agents 1 and 2 both read the grid
│   ├── agents/
│   │   ├── sheet_intel/        Agent 1 — fully implemented
│   │   │   ├── cells.py        cell typing and target-field matching
│   │   │   ├── header.py       header-row detection, composite headers
│   │   │   ├── scoring.py      five-factor scoring, Primary/Secondary/Reject
│   │   │   └── agent.py        the graph node
│   │   ├── mapping/            Agent 2 — fully implemented
│   │   │   ├── targets.py      321-phrase insurance glossary over the 17 fields
│   │   │   ├── channels/
│   │   │   │   ├── lexical.py      header-string evidence (RapidFuzz)
│   │   │   │   ├── semantic.py     embedding evidence (bge-small, local)
│   │   │   │   └── fingerprint.py  column-value evidence
│   │   │   ├── solver.py       channel weights, fusion, Hungarian assignment
│   │   │   ├── models.py       typed output + enforced confidence disclosure
│   │   │   ├── adjudicator.py  LLM consultation and the evidence verifier
│   │   │   └── agent.py        the graph node
│   │   ├── quality/            Agent 3 — stub
│   │   └── transform/          Agent 4 — stub
│   ├── api/main.py             FastAPI app, POST /api/runs, serves the frontend
│   ├── graph.py                the LangGraph StateGraph
│   ├── llm/client.py           provider wrapper behind a Protocol (stub-swappable)
│   └── harness/                (empty — evaluation harness to come)
├── frontend/index.html         single-file UI, posts to /api/runs
├── samples/                    practice SOVs
├── scripts/
│   ├── make_samples.py         regenerates the practice SOVs
│   └── run_samples.py          runs every sample through the graph
├── tests/
│   └── test_adjudicator_verifier.py   18 tests proving the verifier rejects bad citations
├── project documentation/
│   ├── MILESTONE-1.md          skeleton + Agent 1, with real output
│   └── MILESTONE-2.md          Agent 2, with real output and the four defects it exposed
└── requirements.txt
```

Run the tests with:

```bash
python -m pytest tests/ -v
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
| **2** | Agent 2 schema mapping, insurance glossary, LLM adjudicator + verifier | ✅ complete |
| 3 | Agent 3 data quality rules and recommendations | planned |
| 4 | Agent 4 transformation, audit log, review UI | planned |

Current mapping, measured on two corpora that should never be quoted as one number:

| Corpus | Columns | Given a target | Unresolved | `human_review_required` | Overall confidence |
| --- | --- | --- | --- | --- | --- |
| 4 hand-built practice files | 58 | 58 | 0 | 0 | 0.843 – 0.860 |
| 4 real client SOVs | 100 | 52 | 48 | 56 | 0.623 – 0.807 |

The practice-file figure is **58 of 58 on the correct target** — but those files were built in-house to reproduce the documented mess, so it is evidence the channels work, not a benchmark.

For the four real SOVs the honest figure is **coverage, not accuracy: 52 of 100 columns received a target and 48 were left unresolved.** No accuracy percentage is published because the real files have no ground-truth labelling, and spot-checking found at least one confidently wrong mapping (`Buildings → Number of Buildings` on a column of 26 currency values, at confidence 0.402 and correctly flagged `human_review_required`). Roughly half the real columns are things the frozen 17-field schema has no slot for — flood-zone determinations, inspection dates, roof and HVAC update years, Marshall-Swift valuation summaries — so a high unresolved count is partly the schema being honest about its own boundaries rather than pure failure.

Per-milestone write-ups, each with real executed output and the defects that running real files exposed:

- [`project documentation/MILESTONE-1.md`](project%20documentation/MILESTONE-1.md) — skeleton and Agent 1
- [`project documentation/MILESTONE-2.md`](project%20documentation/MILESTONE-2.md) — Agent 2, the three evidence channels, and the verifier

---

## Notes for contributors

- **Agents never import each other.** Shared truth lives in `backend/state/`; shared *mechanics* (file ingestion) live in `backend/ingest/`.
- **An agent is a pure function:** `SOVState` in, a partial dict out. The gate decorator handles validation and tracing.
- **Adding a real agent** means replacing a stub body. The graph and gates do not change.
- **No LLM output is applied unverified.** If you add a model call anywhere, add the code that re-derives its claims in the same change. `backend/agents/mapping/adjudicator.py` is the reference pattern.
- **Guardrails need tests that make them fire.** A check that has never been seen rejecting anything is indistinguishable from a check that cannot. `tests/test_adjudicator_verifier.py` fabricates one defect per test for exactly this reason.
- **CORS is currently open** (`allow_origins=["*"]`) for local development. Narrow it before any deployment.
