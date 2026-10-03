# Milestone 2 — Agent 2, Schema Mapping

**Status:** complete
**Goal:** replace the `schema_mapping` stub with a real agent that maps a source SOV's columns onto the frozen 17 target fields, per SRS 4.3 and FR-MAP-01 … FR-MAP-10.

Milestone 1 proved the skeleton. This milestone fills the first stub, and it is the first place the governing principle has teeth: an LLM participates in mapping, and it is not trusted. Every claim it makes is re-derived from deterministic evidence before it is allowed to move a single column.

> **The LLM proposes, code disposes.**

---

## 1. What changed in the layout

`loader.py` moved out of Agent 1 and into a shared package:

```
backend/agents/sheet_intel/loader.py   →   backend/ingest/loader.py
```

Agent 1 classifies sheets and Agent 2 profiles column values, both from the same loaded grid — and agents never import each other. Ingestion therefore belongs outside `agents/`, next to `state/`. Five import sites were rewritten and Agent 1's output verified byte-identical to Milestone 1 afterwards.

New modules:

```
backend/ingest/loader.py                  (moved) .xlsx/.csv ingestion
backend/agents/mapping/targets.py         the insurance glossary and target catalogue
backend/agents/mapping/channels/
    lexical.py                            header-string evidence (RapidFuzz)
    semantic.py                           embedding evidence (bge-small)
    fingerprint.py                        column-value evidence
backend/agents/mapping/solver.py          fusion + Hungarian assignment
backend/agents/mapping/models.py          typed mapping output, disclosure rules
backend/agents/mapping/adjudicator.py     LLM consultation + evidence verifier
backend/agents/mapping/agent.py           the graph node (was a stub)
backend/llm/client.py                     provider wrapper behind a Protocol
tests/test_adjudicator_verifier.py        18 guardrail tests
```

New dependencies: `rapidfuzz` 3.14.6, `scipy` 1.18.1, `sentence-transformers` 6.1.0 (`torch` 2.14.1+cpu), `openai` 3.24.0, `pytest` 9.1.1.

---

## 2. The target catalogue (FR-MAP-01)

`backend/state/target_schema.py` still owns the frozen 17 fields and the shallow vocabulary Agent 1 needs. `backend/agents/mapping/targets.py` is the mapping-specific enrichment layer on top of it. For each target it carries three things:

- **A glossary of header spellings.** `bldg rc`, `bi/ee value`, `m&e`, `eyb`, `prop damage bldg`, `loss of rents`. **321 phrases** across the 17 targets.
- **A natural-language description** for the semantic channel, written as a sentence a broker would recognise — because cosine similarity against the bare string `"BI"` is meaningless.
- **The accepted value shapes** the fingerprint channel should expect to see in a column that genuinely holds that field.

Nothing in it is model-derived. It is a hand-authored broker glossary, which is why a mapping decision can *cite* it — and why the verifier in §6 can check a citation.

Two import-time drift guards raise `RuntimeError` if the catalogue and the frozen SRS 5.1 schema ever diverge on field names or dtypes, rather than silently mapping onto a field that no longer exists.

---

## 3. Three independent evidence channels (FR-MAP-02 … 04)

The channels are deliberately independent: they consume different inputs and fail in different ways, so one being wrong does not drag the others with it.

### `lexical.py` — the header string

RapidFuzz `token_set_ratio` between the normalised header and every glossary phrase. Headers are normalised (composite `›` names split, both halves kept) and abbreviations expanded against a **42-entry** dictionary (`bldg → building`, `yr → year`, `tiv → total insured value`) with the original retained alongside the expansion.

One property had to be documented rather than fixed: `token_set_ratio` is **subset-generous**, so a short glossary phrase scores 1.0 against a much longer header. That is why `LEXICAL_ATTRIBUTION_FLOOR = 0.75` governs only whether the channel is allowed to *name* the phrase in its citation — the raw score still reaches fusion. An exact match scores 1.0 and marks the hit `exact`.

### `semantic.py` — meaning, not spelling

Local `BAAI/bge-small-en-v1.5` (384-dim, normalised), thread-locked lazy load, the 17 target descriptions embedded once at startup. Cosine similarity between the abbreviation-expanded header and each description, then calibrated onto `[0, 1]` across the band `CALIBRATION_FLOOR = 0.35` … `CALIBRATION_CEILING = 0.80`.

Three findings here were measured, not assumed, and are recorded in the module docstring:

- **The bge query-instruction prefix was tried and rejected.** It compressed the usable cosine band from 0.31–0.72 to 0.43–0.64 and collapsed unrelated headers onto `Contents`/`BI`.
- **Expanding abbreviations before embedding raised rank-1 accuracy from 7/12 to 10/12.** The model has no idea what `Yr Blt` is; it knows exactly what `year built` is.
- **Raw cosines need calibration.** Unrelated text sits around 0.35, not 0.0, so uncalibrated scores read as weak positive evidence for everything.

Two residual failures remain and were left in place rather than tuned away: `# Bldgs` and `Stories` both lean toward `Year Built`. Lexical carries those columns, which is the point of having three channels.

The channel degrades gracefully: if the model cannot load, `is_available()` is false, every hit is zero, and fusion reweights over the surviving channels rather than reading a zero as evidence of mismatch.

### `fingerprint.py` — what is actually in the column

Profiles up to `SAMPLE_SIZE = 200` values into a shape distribution over ten coarse shapes (`currency`, `zip`, `year`, `small_int`, `us_state`, `boolean`, `street_address`, `place_name`, `identifier`, `category`), then scores a target by how much of that distribution lands in the shapes it accepts.

It is coarse **on purpose**. A fingerprint cannot tell `Building Value` from `Contents` — both are currency — and pretending otherwise would be false precision. Its job is to *rule out* nonsense, e.g. that a column of four-digit years is an `Address`.

Two shape pairs are genuinely indistinguishable from values alone and are declared mutually acceptable rather than guessed at:

```
currency                Building Value / Contents / BI / Other
place_name ~ category   City / County / Country / Occupancy / Construction
```

The second was found by running real data, not by reasoning — see §8.

Column-level corrections the per-value classifier cannot make:

- **Low-cardinality short words lean `category`.** `Masonry, Frame, Masonry` and `Houston, Chicago, Houston` are the same shape to any value-level test, so this is applied as a *lean*, guarded by `CARDINALITY_MIN_SAMPLE = 8` so a distinct-ratio is never read as meaningful when the sample is too small to carry one.
- **Bare 3–5 digit integers are deferred, not decided.** `classify_value` returns an internal `_digits` placeholder and `profile_column` resolves the whole column at once. Deciding per value is what made every five-digit dollar amount fingerprint as a ZIP (§8).

---

## 4. Fusion and global assignment (FR-MAP-05, FR-MAP-06)

`solver.py`. Channel weights are a **module-level config constant**, not inlined at the call site:

```python
CHANNEL_WEIGHTS = {"lexical": 0.30, "semantic": 0.35, "fingerprint": 0.35}
```

Fusion renormalises over **participating** channels only, so an abstaining fingerprint (too few values) or an unavailable semantic model shrinks the denominator instead of dragging the score toward zero.

Assignment is **global, not greedy**. The fused score matrix is negated and handed to `scipy.optimize.linear_sum_assignment` (Hungarian), so the solver maximises total score across the whole sheet under a one-to-one constraint. Per-column `argmax` would happily give `Contents` to two columns and starve the correct one; Hungarian cannot. Where the global optimum takes a column away from its own first preference, the agent **says so in the rationale** rather than hiding the trade.

Thresholds:

| Constant | Value | Meaning |
|---|---|---|
| `MIN_ASSIGNMENT_SCORE` | 0.25 | below this a column is left unmapped rather than forced |
| `LOW_MARGIN_THRESHOLD` | 0.15 | below this the LLM adjudicator is consulted |
| `MARGIN_FULL` / `MARGIN_PENALTY` | 0.20 / 0.35 | margin-adjusted confidence band and maximum penalty |
| `EXACT_CONFIDENCE_FLOOR` | 0.92 | an exact glossary hit cannot be dragged below this |

Confidence is **margin-adjusted**: `fused × (1 − MARGIN_PENALTY × closeness)`. A column that beat its runner-up by 0.01 is not confident no matter how high its raw score, and the output has to reflect that.

---

## 5. Confidence, uncertainty and review flags (FR-MAP-07, FR-MAP-08)

`models.py`. Three bands, enforced by a Pydantic `model_validator` rather than left to the agent's good intentions:

| Band | Flag |
|---|---|
| ≥ 0.85 | `auto_accepted` |
| 0.50 – 0.85 | `review_suggested`, and an **uncertainty statement is mandatory** below 0.70 |
| < 0.50 | `human_review_required` |

`_enforce_disclosure` makes a mapping *unconstructible* if it is low-confidence without a stated reason. `_enforce_counts` on the mapping block rejects duplicate target assignment, a wrong `unresolved_count`, and adjudicator arithmetic that does not add up. These are structural guarantees, not conventions — a future bug in the agent fails loudly at construction instead of emitting a confident-looking mapping with no justification.

Every column reports its three channel scores, the fused score, which channels contributed, the runner-up and its score, the cited glossary phrase, the raw (pre-calibration) cosine, the dominant value shape and share, five sample values, and a plain-English rationale naming the deciding evidence.

---

## 6. The LLM adjudicator and the code-side verifier (FR-MAP-09, FR-MAP-10)

The LLM is consulted **only** for columns the deterministic scorer assigned with a margin below 0.15 — the genuinely ambiguous ones. It is offered exactly two candidates: the assignment and the runner-up. The system prompt tells it plainly that its citations will be re-checked by code and that returning `null` is a valid answer.

`backend/llm/client.py` keeps the provider behind a `runtime_checkable` Protocol, which is what makes the whole adjudication path exercisable offline with a stub client. Temperature 0, JSON object mode, two attempts. `as_json()` **raises** on a non-dict response rather than returning `{}`, because an empty object would read downstream as a deliberate model decline.

Then the part that matters. `verify_proposal` re-derives **every** claim from evidence the code already holds, and a proposal is applied only if all eight checks pass:

| Check | What it re-derives |
|---|---|
| `column_identity` | the proposal is about the column we actually asked about |
| `target_exists` | the target is one of the frozen 17 |
| `target_in_candidates` | it is one of the two offered — the model may choose, not invent |
| `glossary_phrase_exists` | the cited phrase is in the glossary **for that target** |
| `glossary_phrase_matches_header` | it resembles the real header at ≥ `CITATION_SIMILARITY_FLOOR` (0.55) |
| `value_shape_observed` | the cited shape covers ≥ `SHAPE_SHARE_FLOOR` (0.20) of the real values |
| `sample_value_present` | the quoted value actually appears in the column |
| `rationale` | a rationale was given at all |

Failures **accumulate** — an audit record shows the full extent of what was wrong, not the first thing noticed. An audit record is written for accepted *and* rejected proposals and appended to `state.audit`, which `operator.add` makes append-only so nothing can later erase the record.

### Proof the guardrail fires

A guardrail that has never been observed rejecting anything is indistinguishable from a guardrail that *cannot* reject anything. So `tests/test_adjudicator_verifier.py` does not wait for a real model to hallucinate — it fabricates exactly one defect per test and drives it through the real `verify_proposal` and the real `adjudicate` path via a stub client. **18 tests, all passing.**

One control (an honest proposal must be accepted, or the rejections prove nothing) plus eleven fabricated defects, including the subtle case where every token of the citation is real but the association is invented — `yr blt` is a genuine glossary phrase, for `Year Built`, cited in support of `Construction`.

Real output from that path, on a synthetic low-margin column (`Const Type`, values `Masonry / Frame / Masonry / Joisted Masonry / Frame`, solver assignment `Construction` at 0.770, margin 0.115):

```
--- verification ---
accepted : False
verified : ['column_identity', 'target_exists', 'target_in_candidates',
            'glossary_phrase_exists', 'rationale']
FAILED   : glossary_phrase_matches_header: 'tenancy' resembles header 'Const Type'
           at only 0.27, below 0.55
FAILED   : value_shape_observed: cited shape 'currency' covers 0% of values;
           the column is actually 'place_name' (100%)
FAILED   : sample_value_present: 'Retail Store' does not appear in column 'Const Type'

--- audit record written to state.audit ---
{
  "agent": "schema_mapping",
  "event": "llm_adjudication",
  "source_column": "Const Type",
  "consulted": true,
  "proposed_target": "Occupancy",
  "proposed_confidence": 0.93,
  "cited_glossary_phrase": "tenancy",
  "cited_value_shape": "currency",
  "cited_sample_value": "Retail Store",
  "accepted": false,
  "applied_target": null,
  "failed_checks": [ ... three entries ... ]
}

applied_target (what the mapping actually used): None
```

The model asked for `Occupancy` at 0.93 self-reported confidence and received an audit entry and nothing else. The deterministic mapping was not touched.

The converse is also tested: a proposal whose citations all hold **is** allowed to override the solver, otherwise adjudication would be decorative. Provider outages, unparseable text, JSON arrays and markdown-fenced JSON are each covered — a code fence is model sloppiness, not a hallucination, and should not cost a valid proposal.

---

## 7. Wiring, gates and API

`schema_mapping` in `backend/agents/mapping/agent.py` now carries real logic under `@contract_gate("schema_mapping")`. It reads the Primary sheet from the **manifest** rather than re-deriving Agent 1's decision, and re-reads the workbook from disk because `SOVState` is the only channel between agents.

`backend/state/gates.py` gained two preconditions, `_mapping_present` and `_mapping_resolves_something`, wired into both the `data_quality` and `transformation` gates. Agents 3 and 4 are still stubs, but they can no longer run against an empty mapping.

`SOVState` gained an append-only `audit` list with an `operator.add` reducer, alongside `issues` and `trace`. Agent 2 records discarded LLM proposals there and Agent 4 will record the ops it applies; neither can erase the other's record.

`POST /api/runs` now returns `mapping`, `issues` and `audit` in addition to the Milestone 1 fields — `audit` specifically so a reviewer can see every proposal that was *discarded*, not just the mappings that survived.

---

## 8. Test results on the sample SOVs

Run through `POST /api/runs` against a live uvicorn, not just through `run_pipeline`. All four returned **HTTP 200** with `mapping`, `issues`, `audit` and `trace` all serialising, at confidences identical to the in-process run.

| File | Primary sheet | Overall | Mapped | Unresolved | Review required |
|---|---|---|---|---|---|
| `sample1_basic.csv` | `sample1_basic` | 0.8461 | 14 | 0 | 0 |
| `sample1_basic.xlsx` | `Sheet1` | 0.8461 | 14 | 0 | 0 |
| `sample2_title_block.xlsx` | `SOV` | 0.8603 | 16 | 0 | 0 |
| `sample3_multisheet_merged.xlsx` | `Location Schedule` | 0.8429 | 14 | 0 | 0 |

**58 of 58 columns mapped to the correct target. No column fell into `human_review_required`.**

`sample2_title_block.xlsx`, the hardest header set — note that not one column shares a word with its target:

```
Location Number            -> Reference              0.736  lex=1.00 sem=0.64 fp=0.60
Property Address           -> Address                0.936  lex=1.00 sem=0.82 fp=1.00
Town                       -> City                   0.921  lex=1.00 sem=0.77 fp=1.00
State Name                 -> State                  0.796  lex=1.00 sem=0.82 fp=0.60
Postal Code                -> Zip                    0.934  lex=1.00 sem=0.81 fp=1.00
County                     -> County                 0.954  lex=1.00 sem=0.87 fp=1.00
Replacement Cost           -> Building Value         0.786  lex=1.00 sem=0.39 fp=1.00
Business Personal Property -> Contents                0.898  lex=1.00 sem=0.71 fp=1.00
Loss of Income             -> BI                     0.912  lex=1.00 sem=0.75 fp=1.00
Other Property             -> Other                  0.800  lex=1.00 sem=0.81 fp=0.62
Use                        -> Occupancy              0.702  lex=1.00 sem=0.55 fp=0.60
Wall Type                  -> Construction           0.731  lex=1.00 sem=0.63 fp=0.60
No of Floors               -> Storeys                0.902  lex=1.00 sem=0.72 fp=1.00
Qty Buildings              -> Number of Buildings    0.911  lex=1.00 sem=0.74 fp=1.00
Year of Construction       -> Year Built             1.000  lex=1.00 sem=1.00 fp=1.00
Auto Sprinkler             -> Fire Sprinklers (Y/N)  0.846  lex=1.00 sem=0.69 fp=0.88
unmapped targets: ['Country']
```

`sample3_multisheet_merged.xlsx` maps straight off Agent 1's composite headers — `Values › Building → Building Value` at 0.846, `Values › Contents → Contents` at 0.902, `Values › BI/EE → BI` at 0.842.

The weakest correct mappings across all four files are `Const Type → Construction` at 0.655 and `Use → Occupancy` at 0.702, both flagged `review_suggested` with a stated reason. `Stories → Storeys` sits at 0.712 carried entirely by lexical, because the semantic channel scores it 0.18 — one of the two known embedding failures from §3, visible in the output exactly as it should be.

`pytest tests/` — **18 passed**.

---

## 9. Defects found by running real files

Four, all invisible to reasoning, all fixed structurally and documented in the code rather than tuned away.

1. **`Const Type` mapped to `Country` at 0.655.** Diagnosed by calling `profile_column(['Masonry','Frame','Masonry'])` directly: it returned `place_name` at 100%, and because `Construction` accepted only `category`, the fingerprint channel scored the *correct* target **0.0** — so Hungarian gave the column away. Cardinality heuristics cannot fix this: `Houston, Chicago` and `Masonry, Frame` are the same shape to any value-level test. Fixed by declaring `place_name` and `category` mutually acceptable for that whole target group, and making the cardinality lean sample-size aware.

2. **The verifier rejected truthful citations.** Four tests failed because `category` cited against a column the profiler had labelled `place_name` was thrown out. An over-strict guardrail is as harmful as a lax one — it teaches you to ignore it. Fixed with `SHAPE_EQUIVALENCE_GROUPS` / `shapes_equivalent` / `equivalent_share`, declared once in `fingerprint.py` and consumed by the verifier so the two modules cannot drift apart.

3. **Every five-digit integer fingerprinted as a ZIP.** `Location Number` (`1001 … 1005`) landed on `Reference` at 0.342 and `Other Property` (`15000, 0, 8000, 0, 42000`) on `Other` at 0.431, both `exact` lexical hits with fingerprint **0.0**. The ZIP patterns were simply matched before the currency branch. Fixed by deferring 3–5 bare digits to a column-level resolver. Confidences rose to 0.736 and 0.800.

4. **The fix for (3) then broke real ZIP columns.** The new resolver read mixed digit widths as "these are magnitudes, not identifiers" and routed them to `currency` — which is precisely wrong for a ZIP column containing `7030` beside `10001`, i.e. a postal code whose leading zero a spreadsheet had eaten. That is the exact defect ASSAY exists to catch, and the rule inverted the intended behaviour: `Zip` dropped to 0.363 and `Postal Code` to 0.380, both wrongly `human_review_required`. Fixed by testing the width *set* against `{4, 5}` — the only significant-digit counts a US postal code can legitimately show — instead of merely counting distinct widths. Both returned to 0.908 and 0.934 with fingerprint 1.00.

Defect 4 is worth naming as a process point: it was introduced by the fix for defect 3 and caught only because the samples were re-run and the output read, not because anything in the test suite failed.

---

## 10. Known limitations and honest caveats

- **The adjudicator has never been run against a live model.** No `OPENAI_API_KEY` is configured, so `adjudicator_consulted` is **0** on every sample run — and in any case no column fell below the 0.15 margin threshold. The verifier's correctness rests entirely on the 18 stub-client tests, which is why those tests fabricate defects rather than waiting for a hallucination.
- **The samples are hand-built practice files**, not the three real SOVs. The 58/58 figure is against inputs built in-house to reproduce the documented mess. It is evidence the channels work, not a benchmark.
- **Two semantic rank-1 failures remain** (`# Bldgs`, `Stories` → `Year Built`), carried by the lexical channel. Recorded rather than tuned away.
- **FR-MAP-11 (vector memory of past mappings) was explicitly out of scope** for this milestone.
- **The fingerprint channel cannot separate same-shape targets.** `Building Value` vs `Contents` vs `BI` vs `Other` is decided by header evidence alone. On a file whose value columns carry no recognisable headers, mapping will be low-confidence — correctly, and with an uncertainty statement saying so.
- **Model download on first run.** `bge-small` is fetched from the HuggingFace Hub on first use (~130 MB) and cached. The first pipeline run is therefore slow — 22 s against ~50 ms warm.
- Carried over from Milestone 1 and untouched: `FRONTEND_DIR` is CWD-relative, and CORS is still `allow_origins=["*"]`.

---

## 11. Explicitly not built in this milestone

Agent 3 real logic · Agent 4 real logic · vector memory (FR-MAP-11) · Angular UI · the review UI · the audit log workbook · Docker · SSE streaming · authentication · persistence.
