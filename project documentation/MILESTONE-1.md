# Milestone 1 — Running Skeleton with Agent 1

**Status:** complete
**Goal:** prove the multi-agent architecture end to end, with Agent 1 (Sheet Intelligence) fully working and Agents 2–4 as observable stubs.

The point of this milestone is not features. It is to make the *shape* of the system real: a typed state object that flows through a graph of agents, with a gate in front of every agent that refuses malformed input. Once that skeleton holds, each remaining agent is a drop-in replacement for a stub.

---

## 1. Project setup

| Item | Detail |
|---|---|
| Language | Python (developed and tested on **3.12.10**; SRS 2.4 specifies 3.11 — see *Known deviations*) |
| Dependencies | `langgraph`, `fastapi`, `uvicorn`, `python-multipart`, `pandas`, `openpyxl`, `pydantic` |
| Layout | Per Planning Document §8 |
| Secrets | `.env.example` only; no LLM is called anywhere in this milestone |

`backend/llm/` and `backend/harness/` exist as empty packages so the layout matches the plan, but nothing lives in them yet.

---

## 2. `SOVState` — the typed, versioned state object

Implemented in `backend/state/sov_state.py` as a Pydantic v2 model, matching SRS 5.2.

```
run_id    str (uuid4)             issues           list[Issue]        + reducer
version   int  (STATE_VERSION=7)  recommendations  list[dict]
source    SourceInfo              decisions        list[Decision]
manifest  list[SheetManifestEntry] audit           list[dict]
mapping   dict                    trace            list[TraceEvent]   + reducer
```

Notes on specific choices:

- **`class` is a Python keyword.** `SheetManifestEntry` stores it as `class_` with the Pydantic alias `"class"`, so the JSON on the wire matches the SRS exactly. Serialise with `model_dump(by_alias=True)`.
- **`issues` and `trace` carry `operator.add` reducers.** LangGraph appends rather than overwrites, so an agent returns only what it added.
- **`SourceInfo.file_path` is an addition to the SRS shape.** Storage is local disk (SRS 2.4) and agents need to reach the original workbook. It is stripped from the API response so it never leaves the server.
- **`version` is the state *schema* version**, not a revision counter. Gates reject state carrying a different version, which is what stops a stale producer from feeding a newer consumer.

The 17-field target schema from SRS 5.1 lives separately in `backend/state/target_schema.py`, alongside the vocabulary used to recognise those fields in client files. Agents never import each other; shared truth lives in `state/`.

---

## 3. Contract gates (FR-ORC-02, NFR-7)

`backend/state/gates.py`. Every agent node is wrapped in `@contract_gate("<agent>")`, which runs **before** the agent body and does two things:

1. **Type check.** Re-validates the entire state against the `SOVState` schema and confirms the version matches. A gate is the only place malformed state can be caught cheaply, before an agent acts on it.
2. **Preconditions.** Asserts what *this specific agent* depends on. `schema_mapping`, `data_quality` and `transformation` each require a non-empty manifest with exactly one `Primary` sheet. `sheet_intelligence` requires a readable `source.file_path`.

On failure it raises `ContractViolation` carrying every failed check, which the API surfaces as a readable 422. On success it times the agent and appends a `TraceEvent`.

Verified rejections:

```
wrong type for manifest     rejected: state failed schema validation
bad sheet class enum        rejected: state failed schema validation
confidence out of range     rejected: state failed schema validation
truncated sha256            rejected: state failed schema validation
stale state version         rejected: state version 3 != expected 7
missing manifest            rejected: manifest is empty; Agent 1 produced no sheet classification
missing file_path           rejected: source.file_path is required to read the workbook
```

---

## 4. LangGraph skeleton (FR-ORC-01)

`backend/graph.py` builds a `StateGraph(SOVState)` with exactly four nodes, wired linearly:

```
START → sheet_intelligence → schema_mapping → data_quality → transformation → END
          (real)              (stub)           (stub)         (stub)
```

Each stub prints `Agent N (name) stub ran` and returns `{}` — state passes through untouched, so you can see the wiring fire without any logic behind it. Because the gates sit in front of every node, the stubs are already running against validated state.

---

## 5. Agent 1 — Sheet Intelligence (SRS 4.2)

The only agent with real logic. Four modules under `backend/agents/sheet_intel/`:

### `loader.py` — ingestion (FR-ING-01 … 05)

- `.xlsx` through **openpyxl**, not pandas, so merged-cell ranges survive. pandas discards them, and merge information is exactly what composite headers depend on.
- `.csv` through the **stdlib csv reader**. pandas infers column count from the first line, so a one-cell title row above a 14-column table makes it fail to tokenise. The stdlib reader returns a ragged grid which is then padded.
- Merged ranges are **forward-filled** before any analysis: a banner spanning three columns becomes the same value in three cells.
- Trailing empty rows and columns are trimmed (openpyxl routinely overshoots `max_row`).
- Guards: unsupported extension, zero bytes, corrupt/password-protected zip, missing file, and the 5,000-row ceiling. Every one returns a sentence a broker could act on, never a stack trace.

### `cells.py` — cell primitives

Classifies each cell as `empty / number / date / bool / text`, handling currency strings like `$1,250,000.00` and parenthesised negatives. Matches header text to the 17 target fields using token-subset matching plus `difflib` fuzzy ratio. Deliberately shallow — Agent 2 owns real mapping; Agent 1 only needs to know whether a row *looks like* an SOV header.

### `header.py` — header detection (FR-SHT-03, FR-SHT-04)

Scores every row in the first 30 as a header candidate on five factors: label density, fill, uniqueness, type consistency of the data below it, and target-vocabulary overlap.

Two structural behaviours matter:

- **Climbing to the top of the header block.** A sub-header row is sparse, entirely labels, and sits directly above clean numeric data — so it scores *better* than the banner above it. After picking the best row, detection climbs while the row above qualifies as its banner. A sparsity guard (a sub-row must be much thinner than its banner) stops it climbing into a title row.
- **Composite headers.** When a two-row header is found, the banner and sub-row are joined with `›`, producing `Values › Building`, `Values › Contents`, `Values › BI/EE`. Data then starts two rows below the banner.

### `scoring.py` — sheet scoring and classification (FR-SHT-01, FR-SHT-02)

Each sheet is scored on five weighted factors, all reported individually so a reviewer can see *why*:

| Factor | Weight |
|---|---|
| header density | 0.25 |
| type consistency | 0.20 |
| null ratio | 0.15 |
| row continuity | 0.15 |
| target vocabulary overlap | 0.25 |

Exactly one sheet is named `Primary` (the gate downstream depends on it). Others become `Secondary` if they are tabular *and* recognise enough target fields, otherwise `Reject`. Confidence is confidence **in the assigned class** — so a tidy notes tab gets a *high* Reject confidence, not a low one.

---

## 6. API

`POST /api/runs` in `backend/api/main.py` — multipart upload, runs the graph, returns `run_id`, `source`, `manifest` and `trace` as JSON. `IngestError` and `ContractViolation` become 422 with a readable message. No auth, no SSE, no persistence yet.

A single-file frontend (`frontend/index.html`) is served at `/` and posts to this endpoint.

---

## 7. Test results against the sample SOVs

`scripts/run_samples.py` runs every file in `samples/` through the graph.

```
sample1_basic.xlsx / .csv   [Primary]   'Sheet1'              conf 1.0    header_row 0   composite False  rows 12
sample2_title_block.xlsx    [Primary]   'SOV'                 conf 1.0    header_row 3   composite False  rows 8
sample3_multisheet.xlsx     [Primary]   'Location Schedule'   conf 0.969  header_row 2   composite True   rows 7
                            [Secondary] 'Prior Year Values'   conf 0.868  header_row 0   composite False  rows 5
                            [Reject]    'Notes'               conf 0.794  header_row 0   composite False  rows 3
                            [Reject]    'Instructions'        conf 0.85   header_row None                 rows 4
```

Sample 2 located its header under a three-row title block and recognised 16/17 target fields.
Sample 3 produced composite headers from the merged banner:

```
['Loc Ref', 'Street Address', 'City', 'St', 'Zip',
 'Values › Building', 'Values › Contents', 'Values › BI/EE',
 'Occ', 'Const Type', 'Stories', '# Bldgs', 'Yr Blt', 'Sprk']
```

14/17 fields, correctly missing County, Country and Other. Per-factor breakdown for that sheet:
`header_density 0.918, type_consistency 1.0, null_ratio 1.0, row_continuity 1.0, vocabulary_overlap 0.824`.

---

## 8. Defects found by running real files

All three were invisible to reasoning and only appeared when the samples were executed.

1. **A two-row header lost to its own sub-row by 0.003.** `Location Schedule` resolved to the sub-row, yielding `Unnamed_1 … Unnamed_14` and losing eleven real column names. Fixed by climbing to the top of the header block.
2. **A key/value `Notes` tab passed as Secondary** at 0.688 — structurally flawless, but not an SOV. Target-field overlap is the only factor separating SOV data from any tidy table, so it is now a floor for Secondary rather than just a weighted term.
3. **A CSV uploaded over HTTP took the temp-file stem as its sheet name** (`tmpx68j_89m`). Only reachable through the endpoint, not the graph.

---

## 9. Known deviations and open questions

- **Python 3.12.10, not 3.11.** Everything installs and runs clean, but the version should be pinned before Docker work begins.
- **The three real test SOVs were not available.** `samples/` contains hand-built practice files reproducing the documented mess — shifted headers, merged banners, abbreviations, notes tabs, missing columns, currency strings, placeholder years. `scripts/make_samples.py` regenerates them. Drop the real files into `samples/` and the runner picks them up unchanged.
- **`header_row` is 0-based**, so `pd.read_excel(header=header_row)` works directly; the human-readable `reasons` quote the 1-based Excel row. **This needs confirming before the review UI is built against it.**
- **`FRONTEND_DIR` is a relative path**, so the server must be started from the repository root.
- **CORS is currently `allow_origins=["*"]`** — fine for local development, must be narrowed before any deployment.

---

## 10. Explicitly not built in this milestone

Agent 2/3/4 real logic · any LLM call · embeddings or RapidFuzz · the audit log · Angular · Docker · SSE streaming · authentication · persistence.
