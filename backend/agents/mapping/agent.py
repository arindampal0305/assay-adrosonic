"""Agent 2 — Schema Mapping (SRS 4.3, FR-MAP-01 … FR-MAP-10).

Reads the Primary sheet Agent 1 identified, scores every source column against the
17 target fields on three independent evidence channels, resolves them into a
globally optimal one-to-one assignment, consults an LLM only on genuinely close
calls, and verifies anything the LLM claims before letting it change an assignment.

The agent writes `state.mapping` plus issue and audit records, and nothing else. It
performs no renames and touches no customer data: every output is a *proposal* for
Agent 4 to apply after review.

FR-MAP-11 (vector memory of past mappings) is intentionally out of scope here.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Optional

from backend.agents.mapping.adjudicator import AdjudicationOutcome, adjudicate
from backend.agents.mapping.channels import semantic as sem_channel
from backend.agents.mapping.models import (
    AUTO_ACCEPT_AT_OR_ABOVE,
    REVIEW_REQUIRED_BELOW,
    UNCERTAINTY_REQUIRED_BELOW,
    ChannelScores,
    ColumnMapping,
    Evidence,
    MappingBlock,
    RunnerUp,
)
from backend.agents.mapping.solver import (
    LOW_MARGIN_THRESHOLD,
    MIN_ASSIGNMENT_SCORE,
    Assignment,
    build_evidence,
    margin_adjusted_confidence,
    solve,
)
from backend.agents.mapping.targets import TARGETS
from backend.ingest.loader import load_workbook_file
from backend.llm.client import get_client
from backend.state.gates import contract_gate
from backend.state.sov_state import Issue, SOVState

logger = logging.getLogger(__name__)

# Blank headers get a positional placeholder so the column is still addressable and
# still appears in the mapping as an explicit unresolved entry rather than vanishing.
PLACEHOLDER_HEADER = "Unnamed_{index}"


def _read_primary_columns(
    state: SOVState,
) -> tuple[str, Optional[int], list[str], list[list[Any]]]:
    """Load the Primary sheet and return its headers plus column-wise values.

    Agent 1 already determined which sheet and which header row. Re-deriving that
    here would duplicate its logic and risk the two disagreeing, so this reads the
    manifest. The file itself is re-read from disk because `SOVState` is the only
    channel between agents — nothing is handed over in memory.
    """
    entry = state.primary_sheet()
    if entry is None:  # pragma: no cover - the contract gate guarantees this
        raise RuntimeError("no Primary sheet in manifest")

    workbook = load_workbook_file(state.source.file_path, state.source.file_name)
    sheet = next((s for s in workbook.sheets if s.name == entry.sheet), None)
    if sheet is None:
        raise RuntimeError(
            f"sheet '{entry.sheet}' named Primary by Agent 1 is not in the workbook"
        )

    if not entry.headers:
        return entry.sheet, entry.header_row, [], []

    headers = [
        h if (h or "").strip() else PLACEHOLDER_HEADER.format(index=i + 1)
        for i, h in enumerate(entry.headers)
    ]
    start = entry.data_start_row if entry.data_start_row is not None else 0
    body = sheet.grid[start:]
    columns = [
        [row[index] if index < len(row) else None for row in body]
        for index in range(len(headers))
    ]
    return entry.sheet, entry.header_row, headers, columns


def _flag_for(confidence: float, target: Optional[str]) -> str:
    if target is None or confidence < REVIEW_REQUIRED_BELOW:
        return "human_review_required"
    if confidence >= AUTO_ACCEPT_AT_OR_ABOVE:
        return "auto_accepted"
    return "review_suggested"


def _rationale(assignment: Assignment, outcome: Optional[AdjudicationOutcome]) -> str:
    """A sentence naming the evidence that actually decided this column."""
    if assignment.target is None:
        return (
            f"no target assigned: the best candidate '{assignment.preferred_target}' "
            f"scored {assignment.preferred_score:.2f}, below the "
            f"{MIN_ASSIGNMENT_SCORE} floor required to map a column"
        )

    cell = assignment.cell
    parts: list[str] = []
    if cell and cell.lexical_exact:
        parts.append(f"the header matches the glossary phrase '{cell.glossary_phrase}' exactly")
    elif cell and cell.glossary_phrase:
        parts.append(
            f"the header resembles the glossary phrase '{cell.glossary_phrase}' "
            f"({cell.lexical:.2f})"
        )
    if cell and not cell.fingerprint_abstained and cell.fingerprint > 0:
        parts.append(
            f"the values are {cell.dominant_share:.0%} '{cell.dominant_shape}', a shape "
            f"'{assignment.target}' accepts"
        )
    if cell and cell.semantic > 0:
        parts.append(f"embedding similarity is {cell.semantic:.2f}")
    if not parts:
        parts.append(f"fused evidence is {assignment.fused:.2f}")

    text = (
        f"mapped to '{assignment.target}' because " + "; ".join(parts)
        + f". It beat '{assignment.runner_up}' by {assignment.margin:.2f}"
    )

    # When the global assignment overrode this column's own preference, say so.
    # It is the least intuitive thing the solver does and the most likely thing a
    # reviewer will challenge.
    if assignment.preferred_target and assignment.preferred_target != assignment.target:
        text += (
            f". This column's own best match was '{assignment.preferred_target}' "
            f"({assignment.preferred_score:.2f}), but that field fitted another column "
            "better, so the global one-to-one assignment traded it away"
        )

    if outcome and outcome.verification:
        if outcome.applied_target:
            text += (
                f". An LLM adjudicator proposed '{outcome.applied_target}' for this "
                "low-margin column and every citation it made was verified against the "
                "source, so the proposal was applied"
            )
        elif outcome.verification.accepted:
            text += ". An LLM adjudicator was consulted and agreed with the scored result"
        else:
            text += (
                ". An LLM adjudicator was consulted and its proposal was discarded "
                f"({outcome.verification.summary})"
            )
    elif outcome and outcome.error:
        text += f". LLM adjudication was attempted but failed ({outcome.error})"

    return text


def _uncertainty(assignment: Assignment, confidence: float) -> Optional[str]:
    """What specifically is unsure. Required below UNCERTAINTY_REQUIRED_BELOW,
    because "low confidence" on its own gives a reviewer nothing to act on."""
    if confidence >= UNCERTAINTY_REQUIRED_BELOW:
        return None
    if assignment.target is None:
        return (
            f"no candidate target scored above {MIN_ASSIGNMENT_SCORE}; this column may "
            "hold a field outside the 17-column target schema, or its header may be "
            "unreadable. A human must decide whether to map it or drop it."
        )

    reasons: list[str] = []
    if assignment.margin < LOW_MARGIN_THRESHOLD:
        reasons.append(
            f"'{assignment.target}' and '{assignment.runner_up}' are separated by only "
            f"{assignment.margin:.3f}"
        )
    cell = assignment.cell
    if cell and cell.fingerprint_abstained:
        reasons.append(
            "the column had too few readable values to fingerprint, so this rests on "
            "the header text alone"
        )
    if cell and not cell.glossary_phrase:
        reasons.append("no glossary phrase matched the header strongly enough to cite")
    if cell and "semantic" not in cell.contributing:
        reasons.append("the embedding channel was unavailable for this run")
    if not reasons:
        reasons.append(
            f"fused evidence was only {assignment.fused:.2f}, so no channel strongly "
            "supported this target"
        )
    return "; ".join(reasons)


def _empty_mapping(sheet_name: str, header_row: Optional[int]) -> dict[str, Any]:
    block = MappingBlock(
        sheet_identified=sheet_name,
        header_row=header_row,
        unmapped_targets=list(TARGETS),
        semantic_channel_available=sem_channel.is_available(),
    )
    return {
        "mapping": block.model_dump(),
        "issues": [
            Issue(rule="mapping_no_headers", field=None, severity="High")
        ],
        "audit": [
            {
                "agent": "schema_mapping",
                "event": "no_headers",
                "sheet": sheet_name,
                "detail": (
                    f"the Primary sheet '{sheet_name}' has no usable header row, so no "
                    "column could be mapped to the target schema"
                ),
            }
        ],
    }


@contract_gate("schema_mapping")
def schema_mapping(state: SOVState) -> dict[str, Any]:
    sheet_name, header_row, headers, columns = _read_primary_columns(state)
    if not headers:
        return _empty_mapping(sheet_name, header_row)

    evidence = build_evidence(headers, columns)
    assignments = solve(evidence)

    # Adjudicate only low-margin, actually-assigned columns. Spending a model call
    # on a column that matched its glossary phrase exactly would be waste, and
    # would add a regression risk where there is currently none.
    client = get_client()
    outcomes: dict[int, AdjudicationOutcome] = {}
    low_margin = [
        a for a in assignments
        if a.target is not None and a.margin < LOW_MARGIN_THRESHOLD
    ]

    mappings: list[ColumnMapping] = []
    audit: list[dict[str, Any]] = []
    issues: list[Issue] = []

    if client is not None and low_margin:
        with ThreadPoolExecutor(max_workers=min(len(low_margin), 8)) as executor:
            future_to_col = {
                executor.submit(adjudicate, client, a, columns[a.column_index]): a.column_index
                for a in low_margin
            }
            for future, col_idx in future_to_col.items():
                outcomes[col_idx] = future.result()
    elif low_margin:
        # Skipping adjudication is a legitimate degraded mode, but it was previously
        # recorded only as a log line. That made it invisible in the run output: a
        # reader saw `adjudicator_consulted: 0` next to two columns sitting under the
        # documented 0.15 threshold and had no way to tell a missing key from a
        # broken margin check. The semantic channel already reports its own
        # unavailability as an issue plus an audit record; this now matches it.
        logger.info(
            "%d low-margin column(s) present but no LLM is configured; deterministic "
            "assignments stand unadjudicated",
            len(low_margin),
        )
        issues.append(Issue(rule="adjudicator_unavailable", severity="Medium"))
        audit.append(
            {
                "agent": "schema_mapping",
                "event": "adjudication_skipped",
                "reason": "no_llm_configured",
                "low_margin_columns": [
                    {
                        "source_column": a.header,
                        "column_index": a.column_index,
                        "assigned_target": a.target,
                        "runner_up": a.runner_up,
                        "margin": a.margin,
                    }
                    for a in low_margin
                ],
                "detail": (
                    f"{len(low_margin)} column(s) scored within {LOW_MARGIN_THRESHOLD} "
                    "of their runner-up and were eligible for LLM adjudication, but no "
                    "OPENAI_API_KEY is configured. The deterministic assignments stand "
                    "and are reported with margin-adjusted confidence; no second "
                    "opinion was obtained."
                ),
            }
        )

    for assignment in assignments:
        outcome = outcomes.get(assignment.column_index)
        target = assignment.target
        cell = assignment.cell
        method = "unresolved" if target is None else "fused"

        if outcome is not None:
            audit.append(outcome.audit_record())
            if outcome.applied_target:
                target = outcome.applied_target
                method = "llm_adjudicated"
                # Re-read the fused evidence for the target the LLM chose, so the
                # reported channel scores describe the mapping actually recorded
                # rather than the one it replaced.
                cell = evidence[assignment.column_index].cells[target]

        if target is not None and method == "fused":
            if cell and cell.lexical_exact:
                method = "exact"
            elif cell and len(cell.contributing) == 1:
                method = cell.contributing[0]

        confidence = (
            0.0 if target is None
            else margin_adjusted_confidence(
                cell.fused if cell else assignment.fused, assignment.margin
            )
        )
        # An exact glossary match whose values also fit the target is the one case
        # where a thin margin should not drag confidence down: that evidence is
        # categorical, not comparative. `Contents` beside `Building Value` is not
        # genuinely ambiguous just because both are currency.
        if target is not None and cell and cell.lexical_exact and cell.fingerprint >= 0.5:
            confidence = max(confidence, min(cell.fused, 0.995))

        flag = _flag_for(confidence, target)
        # The rationale must describe the mapping as finally recorded, including an
        # adjudicator override, so it is built from a corrected copy.
        final = replace(assignment, target=target, cell=cell)

        mapping = ColumnMapping(
            source_column=assignment.header,
            source_index=assignment.column_index,
            target=target,
            confidence=confidence,
            method=method,
            channels=ChannelScores(
                lexical=cell.lexical if cell else 0.0,
                semantic=cell.semantic if cell else 0.0,
                fingerprint=cell.fingerprint if cell else 0.0,
                fused=cell.fused if cell else 0.0,
                contributing=list(cell.contributing) if cell else [],
            ),
            runner_up=RunnerUp(
                target=assignment.runner_up if assignment.runner_up in TARGETS else None,
                score=assignment.runner_up_score,
            ),
            evidence=Evidence(
                glossary_phrase=(cell.glossary_phrase or None) if cell else None,
                lexical_exact=cell.lexical_exact if cell else False,
                semantic_cosine=cell.semantic_cosine if cell else None,
                dominant_value_shape=cell.dominant_shape if cell else None,
                dominant_shape_share=cell.dominant_share if cell else None,
                sample_values=[
                    str(v).strip()
                    for v in columns[assignment.column_index][:5]
                    if v is not None
                ],
                fingerprint_abstained=cell.fingerprint_abstained if cell else False,
            ),
            rationale=_rationale(final, outcome),
            uncertainty=_uncertainty(final, confidence),
            flag=flag,
            adjudicated=outcome is not None,
        )
        mappings.append(mapping)

        if flag == "human_review_required":
            issues.append(
                Issue(
                    rule="mapping_requires_review",
                    field=target,
                    severity="High" if target is None else "Medium",
                )
            )

    assigned = {m.target for m in mappings if m.target}
    resolved = [m.confidence for m in mappings if m.target is not None]

    block = MappingBlock(
        sheet_identified=sheet_name,
        header_row=header_row,
        mappings=mappings,
        unmapped_targets=[name for name in TARGETS if name not in assigned],
        unresolved_count=sum(1 for m in mappings if m.target is None),
        review_required_count=sum(1 for m in mappings if m.flag == "human_review_required"),
        overall_confidence=round(sum(resolved) / len(resolved), 4) if resolved else 0.0,
        semantic_channel_available=sem_channel.is_available(),
        low_margin_count=len(low_margin),
        adjudicator_available=client is not None,
        adjudicator_consulted=len(outcomes),
        adjudicator_accepted=sum(
            1 for o in outcomes.values() if o.verification and o.verification.accepted
        ),
        adjudicator_rejected=sum(
            1 for o in outcomes.values()
            if o.verification is not None and not o.verification.accepted
        ),
    )

    if not block.semantic_channel_available:
        issues.append(Issue(rule="semantic_channel_unavailable", severity="Medium"))
        audit.append(
            {
                "agent": "schema_mapping",
                "event": "channel_degraded",
                "channel": "semantic",
                "detail": (
                    "the local embedding model could not be loaded, so mappings rest on "
                    f"two channels instead of three: {sem_channel.unavailable_reason()}"
                ),
            }
        )

    return {"mapping": block.model_dump(), "issues": issues, "audit": audit}
