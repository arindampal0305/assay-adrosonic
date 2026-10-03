"""Agent 3 — Data Quality and Reasoning (SRS 4.4, FR-DQ-01 … FR-DQ-09).

Reads the Primary sheet through Agent 2's mapping, measures completeness,
validates every mapped cell against the SRS 5.1 data dictionary and the SRS 9.2
rule catalogue, groups what it finds into reviewable recommendations drawn from
the SRS 9.1 operation whitelist, and optionally asks an LLM to explain the
hardest of them — then verifies what the LLM says before showing it to anyone.

**This agent changes no data.** Not one cell. Every output is a proposal carrying
a before/after example the engine computed itself, and `status="pending"` until a
human accepts it. That is C-01, and it is why DQ-15 reports a totals row rather
than dropping it and DQ-16 reports a TIV mismatch rather than reconciling it.

The order of operations is load for the agent's own purposes, and the reason is
NFR-MNT-01: nothing in this package imports from `backend.agents.mapping`. Agent 3
re-reads the workbook from disk and runs its own cleaner, because `SOVState` is
the only channel between agents and a shared helper would be a second one.

Pipeline:

    build_frame          read the Primary sheet, clean every mapped cell
    schema.validate      pandera over the typed frame        (DQ-02 … DQ-06, DQ-11)
    run_domain_rules     ten independent rule functions      (DQ-01, DQ-07 … DQ-18)
    run_structural_rules sheet-shape arithmetic              (DQ-15, DQ-16)
    reconcile            drop findings superseded by a stronger one on the same cell
    completeness         per-field percentages and the intake score (FR-DQ-01/07)
    build_recommendations one proposal per (rule, field, op)  (FR-DQ-08/09)
    reason_over          LLM prose for the low-confidence ones, verified
    QualityBlock         the typed report, which refuses to be inconsistent

The `QualityBlock` validator is the contract gate on this agent's own output: it
will not construct unless all eighteen SRS 9.2 rules are accounted for as either
evaluated or explicitly not applicable with a reason. A rule that silently never
ran cannot therefore leave this function.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from backend.agents.quality import completeness as comp_mod
from backend.agents.quality import rules as rules_mod
from backend.agents.quality import schema as schema_mod
from backend.agents.quality import structural as structural_mod
from backend.agents.quality.findings import Finding
from backend.agents.quality.frame import build_frame
from backend.agents.quality.models import (
    RULE_CATALOGUE,
    QualityBlock,
    Recommendation,
    dump_recommendations,
)
from backend.agents.quality.reasoner import (
    MAX_REASONED_RECOMMENDATIONS,
    REASON_BELOW_CONFIDENCE,
    ReasoningOutcome,
    reason_over,
)
from backend.agents.quality.recommend import build_recommendations
from backend.llm.client import get_client
from backend.state.gates import contract_gate
from backend.state.sov_state import Issue, SOVState

logger = logging.getLogger(__name__)

# Rules whose absence from the findings needs no per-file explanation because the
# module that owns them reports its own skip reason. Anything else that produced
# no finding is recorded as having run and found nothing, which is a different
# and more useful statement than silence.
_SCHEMA_OWNED = set(schema_mod.SCHEMA_RULES)


def _rule_accounting(
    found: set[str],
    skipped: dict[str, str],
    schema_ran: bool,
) -> tuple[list[str], dict[str, str]]:
    """Split all eighteen SRS 9.2 rules into evaluated and not-applicable.

    `QualityBlock` requires every rule to appear in exactly one of the two, which
    is the mechanism that stops a rule from quietly never running. A rule is
    "evaluated" when its code executed, whether or not it found anything — so a
    file with no zip problems reports DQ-08 as evaluated, not as missing.
    """
    not_applicable = dict(skipped)

    if not schema_ran:
        for rule in sorted(_SCHEMA_OWNED - found - set(not_applicable)):
            not_applicable[rule] = (
                "the pandera schema could not validate this frame, so the "
                "declarative type and range checks did not run"
            )

    evaluated = [r for r in RULE_CATALOGUE if r not in not_applicable]
    return evaluated, not_applicable


def _issues_from(findings: list[Finding]) -> list[Issue]:
    """One `Issue` per (rule, field, severity), carrying its rows.

    Grouped rather than one-per-cell: 863 identical `Issue` objects on
    `SOV_Q8B3`'s empty BI column would make `state.issues` unreadable for every
    downstream consumer, and the rows are preserved on the object anyway. The
    per-cell detail lives in the recommendations, which is where a reviewer acts.
    """
    buckets: dict[tuple[str, Any, str], list[int]] = {}
    for finding in findings:
        buckets.setdefault((finding.rule, finding.field, finding.severity), []).append(
            finding.row
        )
    return [
        Issue(rule=rule, field=field, rows=sorted(set(rows)), severity=severity)  # type: ignore[arg-type]
        for (rule, field, severity), rows in buckets.items()
    ]


def _reasoner_audit(outcomes: list[ReasoningOutcome]) -> list[dict[str, Any]]:
    return [o.audit_record() for o in outcomes]


@contract_gate("data_quality")
def data_quality(state: SOVState) -> dict[str, Any]:
    frame = build_frame(state)

    schema_findings, schema_ran = schema_mod.validate(frame)
    domain_findings, domain_skipped = rules_mod.run_domain_rules(frame)
    structural_findings, structural_skipped = structural_mod.run_structural_rules(frame)

    findings = rules_mod.reconcile(
        schema_findings + domain_findings + structural_findings
    )

    skipped: dict[str, str] = {**domain_skipped, **structural_skipped}
    completeness = comp_mod.field_completeness(frame)
    score, components = comp_mod.intake_quality_score(
        completeness, findings, frame.n_rows
    )

    recommendations: list[Recommendation] = build_recommendations(findings)

    client = get_client()
    audit: list[dict[str, Any]] = []
    issues: list[Issue] = _issues_from(findings)

    eligible = [r for r in recommendations if r.confidence < REASON_BELOW_CONFIDENCE]
    if client is not None:
        recommendations, outcomes = reason_over(client, recommendations, findings)
        audit.extend(_reasoner_audit(outcomes))
    else:
        outcomes = []
        if eligible:
            # The same reasoning as Agent 2's `adjudication_skipped` record: a
            # degraded mode that appears only in a log line is invisible in the
            # run output, and a reader seeing `reasoner_consulted: 0` beside
            # fourteen low-confidence recommendations cannot tell a missing key
            # from a broken threshold.
            logger.info(
                "%d recommendation(s) below %.2f confidence but no LLM is configured; "
                "engine rationales stand unreasoned",
                len(eligible),
                REASON_BELOW_CONFIDENCE,
            )
            issues.append(Issue(rule="reasoner_unavailable", severity="Low"))
            audit.append(
                {
                    "agent": "data_quality",
                    "event": "reasoning_skipped",
                    "reason": "no_llm_configured",
                    "eligible_recommendations": [
                        {"id": r.id, "rule": r.rule, "confidence": r.confidence}
                        for r in eligible[:MAX_REASONED_RECOMMENDATIONS]
                    ],
                    "detail": (
                        f"{len(eligible)} recommendation(s) scored below "
                        f"{REASON_BELOW_CONFIDENCE} and were eligible for LLM "
                        "reasoning, but no provider is configured. Every one of them "
                        "carries the engine's own rationale and uncertainty, which "
                        "satisfy FR-DQ-08 without a model; no second opinion was "
                        "obtained."
                    ),
                }
            )

    evaluated, not_applicable = _rule_accounting(
        {f.rule for f in findings}, skipped, schema_ran
    )

    totals_rows = sorted(
        {f.row for f in findings if f.rule == "DQ-15"}
    )
    gaps = comp_mod.value_field_gaps(completeness)
    if gaps:
        audit.append(
            {
                "agent": "data_quality",
                "event": "value_field_empty",
                "fields": gaps,
                "detail": (
                    "mapped but entirely unpopulated: "
                    + ", ".join(gaps)
                    + ". A quote cannot be produced from this schedule as mapped, and "
                    "the cause is upstream — either the source column is empty or "
                    "Agent 2 mapped the wrong one."
                ),
            }
        )

    block = QualityBlock(
        sheet_identified=frame.sheet_name,
        rows_analysed=frame.n_rows,
        fields_mapped=sum(1 for c in completeness if c.mapped),
        completeness=completeness,
        intake_quality_score=score,
        score_components=components,
        issue_count=len(findings),
        issues_by_severity=dict(Counter(f.severity for f in findings)),
        issues_by_rule=dict(Counter(f.rule for f in findings)),
        rules_evaluated=evaluated,
        rules_not_applicable=not_applicable,
        recommendation_count=len(recommendations),
        recommendations_by_action=dict(
            Counter(r.action_type for r in recommendations)
        ),
        totals_rows=totals_rows,
        schema_validation_ran=schema_ran,
        reasoner_available=client is not None,
        reasoner_consulted=len(outcomes),
        reasoner_accepted=sum(
            1 for o in outcomes if o.verification and o.verification.accepted
        ),
        reasoner_rejected=sum(
            1 for o in outcomes
            if o.verification is not None and not o.verification.accepted
        ),
    )

    audit.append(
        {
            "agent": "data_quality",
            "event": "quality_assessed",
            "rows": frame.n_rows,
            "fields_mapped": block.fields_mapped,
            "issues": block.issue_count,
            "recommendations": block.recommendation_count,
            "intake_quality_score": score,
            "score_components": components,
            "rules_not_applicable": not_applicable,
            "detail": (
                f"{block.issue_count} finding(s) across "
                f"{len(block.issues_by_rule)} rule(s) grouped into "
                f"{block.recommendation_count} recommendation(s). No data was "
                "changed: every recommendation is pending review (C-01)."
            ),
        }
    )

    return {
        "quality": block.model_dump(),
        "recommendations": dump_recommendations(recommendations),
        "issues": issues,
        "audit": audit,
    }
