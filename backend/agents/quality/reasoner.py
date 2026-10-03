"""LLM reasoning over finished recommendations, and the verifier that polices it.

This is the M2 adjudicator pattern applied to a different question, deliberately
reusing its shape rather than restating it:

    propose -> parse -> VERIFY AGAINST EVIDENCE -> accept or discard

What differs is the size of the hole the model is allowed to fill. In Agent 2 an
accepted proposal could change a column's target. Here it can change **prose and
nothing else**: `rationale` and `uncertainty`. The op, the field, the rows and
every `after` value were computed by code from real cells before this module is
called, and `apply_reasoning` constructs the replacement with `model_copy` over
exactly those two text fields, so there is no code path by which a model response
can alter a proposed transformation. That is not a policy, it is the function
signature.

The model is still not trusted about what it *says*, because a plausible sentence
naming a value that does not exist in the file is precisely the failure C-02
exists to prevent. `verify_reasoning` re-derives every claim from the `Finding`
evidence behind the recommendation:

  1. the response concerns the recommendation that was actually asked about
  2. a cited row is one of the recommendation's own affected rows
  3. a cited cell value genuinely appears as a `before` on one of its findings
  4. a cited corrected value equals the engine's own `after` for that row —
     the model may describe the fix, never redefine it
  5. an echoed op equals the engine's op, so prose cannot promote a
     flag_for_review into a correction
  6. every number in the rationale is traceable: it appears in the evidence, is
     one of the affected rows, is the row count, or is written as a percentage
  7. a rationale is present and is prose rather than a fragment
  8. an uncertainty statement is present whenever SRS 5.4 requires one

Check 6 is the one with teeth and the one specific to this agent. The dangerous
hallucination here is not a wrong target, it is a *confident replacement value* in
a sentence a reviewer might act on — "this was probably built in 1985" about a
cell the engine deliberately refused to fill. Numbers are the part of a sentence
that can be checked against a spreadsheet mechanically, so they are checked, and
a rationale carrying an untraceable one is discarded whole.

Rejection costs nothing: the engine's own rationale from `recommend.py` already
satisfies FR-DQ-08, so a discarded response leaves a complete recommendation
standing and writes an audit record naming the failed check. Per NFR-REL-02 the
same is true when no model is configured at all.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.agents.quality.findings import Finding
from backend.agents.quality.models import (
    UNCERTAINTY_REQUIRED_BELOW,
    RULE_CATALOGUE,
    Recommendation,
)
from backend.llm.client import LLMClient, LLMUnavailable

logger = logging.getLogger(__name__)

# The model is consulted only where its output can matter: recommendations below
# the SRS 5.4 uncertainty threshold are the ones a human must decide, and a
# clearer statement of *what* to decide is worth a model call. A high-confidence
# `zip5` on five rows is already fully explained by arithmetic.
REASON_BELOW_CONFIDENCE = UNCERTAINTY_REQUIRED_BELOW

# A bound on model calls per run. Recommendations are reasoned in review-queue
# order, so the budget is spent on the top of the queue rather than on whatever
# happens to be first alphabetically.
MAX_REASONED_RECOMMENDATIONS = 6

# A rationale shorter than this is a fragment, not an explanation.
MIN_RATIONALE_CHARS = 40
# And one longer than this is no longer the one-paragraph artefact SRS 5.4
# describes. Both bounds exist because a local model asked for prose will
# occasionally return either "Missing." or nine hundred words.
MAX_RATIONALE_CHARS = 900

# Numbers written as a share of something are self-evidently not cell values, so
# they do not need to appear in the evidence. Everything else does.
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

SYSTEM_PROMPT = (
    """You are an insurance data analyst writing the reviewer-facing explanation \
for one data-quality finding in a commercial property Statement of Values.

A deterministic engine has already decided everything that matters: which rule \
fired, which rows it affects, whether a fix is possible, and — if one is — exactly \
what the corrected value is. You are not being asked to re-decide any of that, and \
you cannot change it. You are being asked to explain it to an underwriter who has \
to approve or reject it.

Rules you must follow:
- Do NOT propose a replacement value for any cell. If the engine supplied no \
corrected value, that is a deliberate refusal: the data does not determine one. \
Saying what the value "should probably be" is the single most damaging thing you \
can do here, and it is checked for.
- Every number you write must either appear in the evidence shown to you, be one \
of the affected row numbers, be the count of affected rows, or be written as a \
percentage. A number that is none of those discards your whole answer.
- "cited_row" must be one of the "affected_rows" values. "cited_before" must be \
copied character-for-character from the evidence for that row. "cited_after" must \
be the engine's own corrected value for that row, copied exactly, or null if the \
engine supplied none.
- "op" must be repeated back exactly as given, or null if it was null. You may not \
substitute a different operation.
- Use null for any citation you cannot copy directly from the evidence. A guessed \
citation is worse than no citation.
- "uncertainty" must state what a human still has to determine, in the engine's \
terms. If the engine proposed no fix, this is where you say what the reviewer must \
establish. Do not use it to suggest a value.

Every citation is independently re-checked against the source data by code. An \
answer containing any unverifiable claim is discarded in full, and the engine's \
own explanation is used instead.

Respond with a JSON object only:
{
  "recommendation_id": "<the id you were given, copied exactly>",
  "op": "<the op you were given, copied exactly, or null>",
  "cited_row": <one of the affected row numbers, or null>,
  "cited_before": "<that row's value, copied verbatim, or null>",
  "cited_after": "<the engine's corrected value for that row, copied verbatim, or null>",
  "rationale": "<two or three sentences explaining the issue and what approving this would do>",
  "uncertainty": "<what a human must still determine, or null if nothing>"
}"""
)


@dataclass
class Reasoning:
    """A parsed, not-yet-trusted explanation."""

    recommendation_id: Optional[str]
    op: Optional[str]
    cited_row: Optional[int]
    cited_before: Optional[str]
    cited_after: Optional[str]
    rationale: str
    uncertainty: Optional[str]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    accepted: bool
    failures: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.accepted:
            return "verified: " + ", ".join(self.verified)
        return "rejected: " + "; ".join(self.failures)


def parse_reasoning(payload: dict[str, Any]) -> Reasoning:
    """Coerce a response body into a `Reasoning` without judging it."""

    def text_or_none(key: str) -> Optional[str]:
        value = payload.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    row = payload.get("cited_row")
    if isinstance(row, str):
        row = row.strip()
    try:
        cited_row = int(row) if row not in (None, "") else None
    except (TypeError, ValueError):
        # A non-integer row is not silently dropped: it is carried through as a
        # sentinel so check 2 reports it rather than treating the model as having
        # declined to cite anything.
        cited_row = -1

    return Reasoning(
        recommendation_id=text_or_none("recommendation_id"),
        op=text_or_none("op"),
        cited_row=cited_row,
        cited_before=text_or_none("cited_before"),
        cited_after=text_or_none("cited_after"),
        rationale=str(payload.get("rationale") or "").strip(),
        uncertainty=text_or_none("uncertainty"),
        raw=payload,
    )


def _evidence_numbers(rec: Recommendation, group: list[Finding]) -> set[str]:
    """Every number a rationale is permitted to contain, as written.

    Drawn from the data rather than from a tolerance: the affected rows, the size
    of the group, and every numeric token appearing in a cell value, a computed
    `after`, a rule detail or a rule's evidence dictionary. A number outside this
    set did not come from the file.
    """
    allowed: set[str] = {str(r) for r in rec.rows}
    allowed.add(str(len(rec.rows)))

    def harvest(text: str) -> None:
        for match in _NUMBER.findall(text):
            allowed.add(match)
            # "1,010,000" reaches a rationale as either "1010000" or "1,010,000",
            # and "28,399,660" the same way, so both renderings are allowed once
            # either is present in the evidence.
            allowed.add(match.lstrip("0") or "0")
            if "." in match:
                allowed.add(match.split(".")[0])

    for finding in group:
        harvest(finding.before)
        if finding.after is not None:
            harvest(finding.after)
        harvest(finding.detail)
        harvest(str(finding.row))
        for value in finding.evidence.values():
            harvest(str(value))
    for example in rec.before_after:
        harvest(example.before)
        if example.after is not None:
            harvest(example.after)
    harvest(rec.rationale)
    if rec.uncertainty:
        harvest(rec.uncertainty)
    # The rule id itself ("DQ-13") and the SRS clauses the engine cites are not
    # data, but a rationale that names them is quoting the catalogue, not the file.
    harvest(rec.rule)
    harvest(RULE_CATALOGUE[rec.rule].check)
    return allowed


def _untraceable_numbers(text: str, allowed: set[str]) -> list[str]:
    """Numbers in `text` that are neither percentages nor present in `allowed`."""
    without_percentages = _PERCENT.sub(" ", text)
    # Thousands separators are a presentation choice, not a different number.
    normalised = re.sub(r"(?<=\d),(?=\d{3}\b)", "", without_percentages)
    out: list[str] = []
    for token in _NUMBER.findall(normalised):
        if token in allowed or (token.lstrip("0") or "0") in allowed:
            continue
        out.append(token)
    return out


def verify_reasoning(
    reasoning: Reasoning,
    *,
    rec: Recommendation,
    group: list[Finding],
) -> VerificationResult:
    """Re-derive every claim in `reasoning` from the findings behind `rec`.

    Never consults the model, never mutates anything. Takes a claim and the
    source of truth and returns whether the claim survives.
    """
    failures: list[str] = []
    verified: list[str] = []

    # 1. The answer must be about the recommendation we asked about.
    if reasoning.recommendation_id != rec.id:
        failures.append(
            f"recommendation_identity: response names '{reasoning.recommendation_id}' "
            f"but the recommendation under review is '{rec.id}'"
        )
    else:
        verified.append("recommendation_identity")

    # 2. A cited row must be one this recommendation actually covers.
    if reasoning.cited_row is not None:
        if reasoning.cited_row not in set(rec.rows):
            shown = sorted(rec.rows)
            failures.append(
                f"cited_row_affected: row {reasoning.cited_row} is not among the "
                f"{len(rec.rows)} affected row(s) "
                f"({shown[0]}–{shown[-1]} for this recommendation)"
            )
        else:
            verified.append("cited_row_affected")

    # 3. A cited cell value must genuinely be in the file.
    if reasoning.cited_before:
        present = {f.before for f in group}
        if reasoning.cited_before not in present:
            failures.append(
                f"cited_before_present: '{reasoning.cited_before}' is not a value "
                f"recorded for {rec.field or 'this row'} by rule {rec.rule}"
            )
        else:
            verified.append("cited_before_present")

    # 4. A cited corrected value must be the engine's, not the model's. This is
    #    where a well-meaning "1985" for an unparseable year gets caught.
    if reasoning.cited_after:
        engine_afters = {f.after for f in group if f.after is not None}
        if not engine_afters:
            failures.append(
                f"cited_after_computed: the engine proposed no corrected value for "
                f"{rec.rule}, so '{reasoning.cited_after}' was supplied by the model"
            )
        elif reasoning.cited_after not in engine_afters:
            failures.append(
                f"cited_after_computed: '{reasoning.cited_after}' is not a value the "
                f"engine computed for this group ({sorted(engine_afters)[:4]})"
            )
        else:
            verified.append("cited_after_computed")

    # 5. Prose may not promote a flag into a correction.
    if reasoning.op is not None and reasoning.op != rec.op:
        failures.append(
            f"op_unchanged: response names op '{reasoning.op}' but this "
            f"recommendation's op is '{rec.op}'"
        )
    elif reasoning.op is not None:
        verified.append("op_unchanged")

    # 6. Every quantity in the prose must be traceable to the file.
    allowed = _evidence_numbers(rec, group)
    for label, text in (("rationale", reasoning.rationale),
                        ("uncertainty", reasoning.uncertainty or "")):
        if not text:
            continue
        stray = _untraceable_numbers(text, allowed)
        if stray:
            failures.append(
                f"numbers_traceable: {label} contains {', '.join(sorted(set(stray))[:4])}, "
                "which appear nowhere in this group's evidence"
            )
        else:
            verified.append(f"numbers_traceable_{label}")

    # 7. An explanation has to be an explanation.
    if len(reasoning.rationale) < MIN_RATIONALE_CHARS:
        failures.append(
            f"rationale_present: {len(reasoning.rationale)} characters is too short "
            f"to explain a finding (minimum {MIN_RATIONALE_CHARS})"
        )
    elif len(reasoning.rationale) > MAX_RATIONALE_CHARS:
        failures.append(
            f"rationale_length: {len(reasoning.rationale)} characters exceeds the "
            f"{MAX_RATIONALE_CHARS}-character limit for a reviewer-facing note"
        )
    else:
        verified.append("rationale_present")

    # 8. SRS 5.4 requires an uncertainty statement below the threshold. Accepting a
    #    response that dropped it would strip a required field off the output, so
    #    the response is rejected rather than half-applied.
    if rec.confidence < UNCERTAINTY_REQUIRED_BELOW and not (reasoning.uncertainty or "").strip():
        failures.append(
            f"uncertainty_required: confidence {rec.confidence:.2f} is below "
            f"{UNCERTAINTY_REQUIRED_BELOW} and SRS 5.4 requires a stated uncertainty, "
            "but the response gives none"
        )
    elif rec.confidence < UNCERTAINTY_REQUIRED_BELOW:
        verified.append("uncertainty_required")

    return VerificationResult(accepted=not failures, failures=failures, verified=verified)


def build_user_prompt(rec: Recommendation, group: list[Finding]) -> str:
    """The evidence for one recommendation, as JSON.

    Only engine-computed facts go in. Note that `engine_rationale` is included:
    the model is rewriting an explanation that already exists and is already
    correct, which is a smaller and more checkable task than writing one from
    nothing, and it gives the verifier's number check a legitimate vocabulary of
    quantities to draw on.
    """
    spec = RULE_CATALOGUE[rec.rule]
    rows = sorted(rec.rows)
    payload = {
        "recommendation_id": rec.id,
        "rule": rec.rule,
        "srs_check": spec.check,
        "severity": rec.severity,
        "target_field": rec.field,
        "source_column": rec.source_column,
        "op": rec.op,
        "action_type": rec.action_type,
        "engine_confidence": rec.confidence,
        "affected_rows": rows if len(rows) <= 20 else rows[:20],
        "affected_row_count": len(rows),
        "evidence": [
            {
                "row": f.row,
                "value_in_file": f.before,
                "engine_corrected_value": f.after,
                "engine_detail": f.detail,
            }
            for f in group[:6]
        ],
        "engine_rationale": rec.rationale,
        "engine_uncertainty": rec.uncertainty,
    }
    return json.dumps(payload, indent=2, default=str)


@dataclass
class ReasoningOutcome:
    recommendation_id: str
    rule: str
    consulted: bool
    reasoning: Optional[Reasoning]
    verification: Optional[VerificationResult]
    applied: bool = False
    error: Optional[str] = None

    def audit_record(self) -> dict[str, Any]:
        """Written for accepted and rejected responses alike — "we asked and
        discarded the answer" is as important to an underwriter as the reverse."""
        return {
            "agent": "data_quality",
            "event": "llm_reasoning",
            "recommendation_id": self.recommendation_id,
            "rule": self.rule,
            "consulted": self.consulted,
            "cited_row": self.reasoning.cited_row if self.reasoning else None,
            "cited_before": self.reasoning.cited_before if self.reasoning else None,
            "cited_after": self.reasoning.cited_after if self.reasoning else None,
            "accepted": bool(self.verification and self.verification.accepted),
            "applied": self.applied,
            "verification": self.verification.summary if self.verification else None,
            "failed_checks": list(self.verification.failures) if self.verification else [],
            "error": self.error,
        }


def apply_reasoning(rec: Recommendation, reasoning: Reasoning) -> Recommendation:
    """A copy of `rec` carrying the model's prose and nothing else.

    The explicit field list is the guarantee. `op`, `rows`, `before_after`,
    `field`, `confidence` and `action_type` are not in it and cannot be reached
    from here, so an accepted response changes what a reviewer reads and never
    what Agent 4 would execute.
    """
    return rec.model_copy(
        update={
            "rationale": reasoning.rationale,
            "uncertainty": reasoning.uncertainty or rec.uncertainty,
            "llm_reasoned": True,
        }
    )


def reason_once(
    client: LLMClient,
    rec: Recommendation,
    group: list[Finding],
) -> tuple[Recommendation, ReasoningOutcome]:
    """Consult the model about one recommendation, verify, then accept or discard."""
    try:
        response = client.complete_json(
            system=SYSTEM_PROMPT, user=build_user_prompt(rec, group)
        )
        payload = response.as_json()
    except (LLMUnavailable, ValueError, json.JSONDecodeError) as exc:
        # An unusable response costs the run nothing: the engine's rationale is
        # already a complete FR-DQ-08 explanation.
        return rec, ReasoningOutcome(
            recommendation_id=rec.id,
            rule=rec.rule,
            consulted=True,
            reasoning=None,
            verification=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    reasoning = parse_reasoning(payload)
    verification = verify_reasoning(reasoning, rec=rec, group=group)

    if not verification.accepted:
        logger.info("discarded LLM reasoning for %s: %s", rec.id, verification.summary)
        return rec, ReasoningOutcome(
            recommendation_id=rec.id,
            rule=rec.rule,
            consulted=True,
            reasoning=reasoning,
            verification=verification,
        )

    return apply_reasoning(rec, reasoning), ReasoningOutcome(
        recommendation_id=rec.id,
        rule=rec.rule,
        consulted=True,
        reasoning=reasoning,
        verification=verification,
        applied=True,
    )


def _group_for(rec: Recommendation, findings: list[Finding]) -> list[Finding]:
    """The findings that produced `rec`, recovered by its own grouping key."""
    return [
        f for f in findings
        if f.rule == rec.rule and f.field == rec.field and f.op == rec.op
    ]


def reason_over(
    client: Optional[LLMClient],
    recommendations: list[Recommendation],
    findings: list[Finding],
    *,
    limit: int = MAX_REASONED_RECOMMENDATIONS,
) -> tuple[list[Recommendation], list[ReasoningOutcome]]:
    """Reason over the recommendations where it can make a difference.

    Selection is deliberate rather than exhaustive, for the same reason Agent 2
    adjudicates only low-margin columns: a high-confidence mechanical fix is
    already fully explained by the arithmetic that produced it, and spending a
    model call on it buys nothing while adding a way to go wrong. What benefits
    from explanation is the recommendation a human has to decide — which is
    exactly the set SRS 5.4 requires an uncertainty statement for.

    `recommendations` arrives in review-queue order, so the budget is spent at the
    top of the queue. Returns the list with the same ids in the same order.
    """
    if client is None or not recommendations:
        return recommendations, []

    eligible = [
        r for r in recommendations if r.confidence < REASON_BELOW_CONFIDENCE
    ][:limit]
    chosen = {r.id for r in eligible}

    outcomes: list[ReasoningOutcome] = []
    out: list[Recommendation] = []
    for rec in recommendations:
        if rec.id not in chosen:
            out.append(rec)
            continue
        group = _group_for(rec, findings)
        if not group:  # pragma: no cover - the grouping key is the one that built it
            out.append(rec)
            continue
        updated, outcome = reason_once(client, rec, group)
        out.append(updated)
        outcomes.append(outcome)
    return out, outcomes


__all__ = [
    "MAX_REASONED_RECOMMENDATIONS",
    "REASON_BELOW_CONFIDENCE",
    "Reasoning",
    "ReasoningOutcome",
    "VerificationResult",
    "apply_reasoning",
    "build_user_prompt",
    "parse_reasoning",
    "reason_once",
    "reason_over",
    "verify_reasoning",
]
