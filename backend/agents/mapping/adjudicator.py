"""LLM adjudication for low-margin columns (FR-MAP-08, FR-MAP-09), and the
code-side verifier that decides whether to believe it.

This module is where ASSAY's governing principle is actually implemented rather
than merely asserted. The model is given the evidence for a genuinely close call
and asked which target is right. It is then **not** trusted:

    propose -> parse -> VERIFY AGAINST EVIDENCE -> accept or discard

`verify_proposal` re-derives every claim from the deterministic evidence already
computed by the channels. A proposal survives only if all of these hold:

  1. the column is one that was actually submitted for adjudication
  2. the target is one of the 17 frozen schema fields
  3. the target is one of the two candidates offered (the assignment or its
     runner-up) — the model may choose between them, not invent a third
  4. any cited glossary phrase genuinely exists in that target's glossary
  5. any cited glossary phrase genuinely resembles the real source header
  6. any cited value shape matches the shape actually observed in the column
  7. any quoted sample value genuinely appears in the column
  8. a rationale is present

A failure on any one of these discards the proposal, keeps the deterministic
assignment, and writes an audit record naming the failed check and quoting the
rejected claim. A hallucinated citation therefore cannot change a single cell of
the customer's data — it can only generate an audit entry.

The verifier is unit-tested against deliberately fabricated proposals
(`tests/test_adjudicator_verifier.py`), because a guardrail that has never been
observed rejecting anything is not evidence of a working guardrail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from rapidfuzz import fuzz

from backend.agents.mapping.channels.fingerprint import (
    classify_value,
    equivalent_share,
    profile_column,
)
from backend.agents.mapping.channels.lexical import normalise_header, prepare
from backend.agents.mapping.solver import Assignment
from backend.agents.mapping.targets import GLOSSARY, TARGETS
from backend.llm.client import LLMClient, LLMUnavailable

logger = logging.getLogger(__name__)

# A cited glossary phrase must resemble the real header at least this much. Set
# below the lexical matching floor on purpose: the model is allowed to cite a
# loose-but-real connection, it is not allowed to cite a phrase with no
# relationship to the header at all.
CITATION_SIMILARITY_FLOOR = 0.55

# A cited value shape must account for at least this share of the column's
# observed values to count as a true description of it.
SHAPE_SHARE_FLOOR = 0.20

SYSTEM_PROMPT = """You are an insurance data analyst adjudicating a single ambiguous \
column mapping in a commercial property Statement of Values.

You will be given one source column — its header, sample values, and the two \
candidate target fields that deterministic scoring could not separate — and you must \
decide which candidate is correct.

Rules you must follow:
- Choose ONLY from the two candidate targets given. Do not propose any other field.
- Cite only evidence that is actually present in the data shown to you. Every \
citation is independently re-checked against the source by code, and a proposal \
containing any unverifiable citation is discarded in full.
- If the evidence genuinely does not separate the two candidates, say so by setting \
"target" to null. Declining is a valid and useful answer; guessing is not.

Respond with a JSON object only:
{
  "target": "<one of the two candidate target names, or null>",
  "glossary_phrase": "<the glossary phrase supporting your choice, or null>",
  "value_shape": "<the dominant shape of the sample values, or null>",
  "sample_value": "<one sample value you are relying on, or null>",
  "rationale": "<one or two sentences explaining the choice>",
  "confidence": <float between 0 and 1>
}"""


@dataclass
class Proposal:
    """A parsed, not-yet-trusted LLM claim."""

    source_column: str
    target: Optional[str]
    glossary_phrase: Optional[str]
    value_shape: Optional[str]
    sample_value: Optional[str]
    rationale: str
    confidence: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    accepted: bool
    # Every check that failed, named. The audit record carries these verbatim so a
    # reviewer can see precisely which claim was unsupportable.
    failures: list[str] = field(default_factory=list)
    # Checks that passed, so an accepted proposal is also explainable.
    verified: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.accepted:
            return "verified: " + ", ".join(self.verified)
        return "rejected: " + "; ".join(self.failures)


def parse_proposal(payload: dict[str, Any], source_column: str) -> Proposal:
    """Coerce a raw response body into a Proposal without judging it."""
    target = payload.get("target")
    if isinstance(target, str):
        target = target.strip() or None

    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    def text_or_none(key: str) -> Optional[str]:
        value = payload.get(key)
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    return Proposal(
        source_column=source_column,
        target=target,
        glossary_phrase=text_or_none("glossary_phrase"),
        value_shape=text_or_none("value_shape"),
        sample_value=text_or_none("sample_value"),
        rationale=str(payload.get("rationale") or "").strip(),
        confidence=min(max(confidence, 0.0), 1.0),
        raw=payload,
    )


def verify_proposal(
    proposal: Proposal,
    *,
    header: str,
    values: list[Any],
    candidates: tuple[Optional[str], Optional[str]],
) -> VerificationResult:
    """Re-derive every claim in `proposal` from the deterministic evidence.

    This function never consults the model and never mutates state. It takes a
    claim and the source of truth, and returns whether the claim survives.
    """
    failures: list[str] = []
    verified: list[str] = []

    # 1. The column must be the one we asked about. Guards against a batched or
    #    confused response being applied to the wrong column.
    if normalise_header(proposal.source_column) != normalise_header(header):
        failures.append(
            f"column_identity: proposal names column '{proposal.source_column}' "
            f"but the column under adjudication is '{header}'"
        )
    else:
        verified.append("column_identity")

    # A null target is an explicit, legitimate decline. Nothing further to verify.
    if proposal.target is None:
        if not proposal.rationale:
            failures.append("rationale: declined without stating why")
        return VerificationResult(
            accepted=False,
            failures=failures or ["declined: model reported insufficient evidence"],
            verified=verified,
        )

    # 2. The target must exist in the frozen 17-field schema.
    if proposal.target not in TARGETS:
        failures.append(
            f"target_exists: '{proposal.target}' is not one of the 17 target fields"
        )
        # Every remaining check is defined relative to a real target.
        return VerificationResult(accepted=False, failures=failures, verified=verified)
    verified.append("target_exists")

    # 3. The target must be one of the two candidates offered.
    allowed = {c for c in candidates if c}
    if proposal.target not in allowed:
        failures.append(
            f"target_in_candidates: '{proposal.target}' was not offered; "
            f"candidates were {sorted(allowed)}"
        )
    else:
        verified.append("target_in_candidates")

    # 4 & 5. A cited glossary phrase must be real, and must relate to the header.
    if proposal.glossary_phrase:
        phrase = proposal.glossary_phrase.lower().strip()
        glossary = {p.lower() for p in GLOSSARY[proposal.target]}
        if phrase not in glossary:
            failures.append(
                f"glossary_phrase_exists: '{proposal.glossary_phrase}' is not in the "
                f"glossary for '{proposal.target}'"
            )
        else:
            verified.append("glossary_phrase_exists")
            similarity = (
                fuzz.token_set_ratio(prepare(header), normalise_header(phrase)) / 100.0
            )
            if similarity < CITATION_SIMILARITY_FLOOR:
                failures.append(
                    f"glossary_phrase_matches_header: '{proposal.glossary_phrase}' "
                    f"resembles header '{header}' at only {similarity:.2f}, below "
                    f"{CITATION_SIMILARITY_FLOOR}"
                )
            else:
                verified.append("glossary_phrase_matches_header")

    # 6. A cited value shape must describe the column as actually observed.
    if proposal.value_shape:
        fingerprint = profile_column(values)
        distribution = fingerprint.distribution
        # Equivalent shapes count: this channel cannot separate `category` from
        # `place_name`, so rejecting a citation of one because the profiler
        # happened to label the column the other would discard a true statement.
        share = equivalent_share(distribution, proposal.value_shape)
        if not distribution:
            failures.append(
                f"value_shape_observed: cited shape '{proposal.value_shape}' cannot be "
                "checked because the column has no readable values"
            )
        elif share < SHAPE_SHARE_FLOOR:
            observed = max(distribution, key=lambda s: distribution[s])
            failures.append(
                f"value_shape_observed: cited shape '{proposal.value_shape}' covers "
                f"{share:.0%} of values; the column is actually '{observed}' "
                f"({distribution[observed]:.0%})"
            )
        else:
            verified.append("value_shape_observed")

    # 7. A quoted sample value must genuinely appear in the column.
    if proposal.sample_value:
        present = {str(v).strip().lower() for v in values if v is not None}
        if proposal.sample_value.lower() not in present:
            failures.append(
                f"sample_value_present: '{proposal.sample_value}' does not appear in "
                f"column '{header}'"
            )
        else:
            verified.append("sample_value_present")

    # 8. An unexplained decision is not reviewable, so it is not acceptable.
    if not proposal.rationale:
        failures.append("rationale: proposal states no rationale")
    else:
        verified.append("rationale")

    return VerificationResult(accepted=not failures, failures=failures, verified=verified)


def _sample_values(values: list[Any], limit: int = 8) -> list[str]:
    out: list[str] = []
    for value in values:
        if value is None or classify_value(value) is None:
            continue
        out.append(str(value).strip())
        if len(out) >= limit:
            break
    return out


def build_user_prompt(assignment: Assignment, values: list[Any]) -> str:
    candidates = [c for c in (assignment.target or assignment.preferred_target,
                              assignment.runner_up) if c]
    payload = {
        "source_column": assignment.header,
        "sample_values": _sample_values(values),
        "deterministic_scores": {
            "assigned": assignment.target,
            "assigned_score": assignment.fused,
            "runner_up": assignment.runner_up,
            "runner_up_score": assignment.runner_up_score,
            "margin": assignment.margin,
        },
        "candidate_targets": [
            {
                "name": name,
                "description": TARGETS[name].description,
                "glossary_phrases": list(GLOSSARY[name][:12]),
            }
            for name in candidates
        ],
    }
    return json.dumps(payload, indent=2, default=str)


@dataclass
class AdjudicationOutcome:
    header: str
    column_index: int
    consulted: bool
    proposal: Optional[Proposal]
    verification: Optional[VerificationResult]
    # Only set when the proposal was accepted AND differs from the deterministic
    # assignment. This is the sole channel by which the LLM can alter a mapping.
    applied_target: Optional[str] = None
    error: Optional[str] = None

    def audit_record(self) -> dict[str, Any]:
        """The record written to `SOVState.audit`. Deliberately written for both
        accepted and rejected proposals: "we asked and discarded the answer" is
        exactly as important to an underwriter as "we asked and took it"."""
        return {
            "agent": "schema_mapping",
            "event": "llm_adjudication",
            "source_column": self.header,
            "column_index": self.column_index,
            "consulted": self.consulted,
            "proposed_target": self.proposal.target if self.proposal else None,
            "proposed_confidence": self.proposal.confidence if self.proposal else None,
            "cited_glossary_phrase": self.proposal.glossary_phrase if self.proposal else None,
            "cited_value_shape": self.proposal.value_shape if self.proposal else None,
            "cited_sample_value": self.proposal.sample_value if self.proposal else None,
            "accepted": bool(self.verification and self.verification.accepted),
            "applied_target": self.applied_target,
            "verification": self.verification.summary if self.verification else None,
            "failed_checks": list(self.verification.failures) if self.verification else [],
            "error": self.error,
        }


def adjudicate(
    client: LLMClient,
    assignment: Assignment,
    values: list[Any],
) -> AdjudicationOutcome:
    """Consult the model about one low-margin column, then verify its answer."""
    candidates = (assignment.target or assignment.preferred_target, assignment.runner_up)

    try:
        response = client.complete_json(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(assignment, values),
        )
        payload = response.as_json()
    except (LLMUnavailable, ValueError, json.JSONDecodeError) as exc:
        # An unusable response leaves the deterministic assignment untouched. The
        # run does not fail; it simply loses the second opinion.
        return AdjudicationOutcome(
            header=assignment.header,
            column_index=assignment.column_index,
            consulted=True,
            proposal=None,
            verification=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    proposal = parse_proposal(payload, assignment.header)
    verification = verify_proposal(
        proposal, header=assignment.header, values=values, candidates=candidates
    )

    applied = None
    if verification.accepted and proposal.target != assignment.target:
        applied = proposal.target

    if not verification.accepted:
        logger.info(
            "discarded LLM proposal for '%s': %s", assignment.header, verification.summary
        )

    return AdjudicationOutcome(
        header=assignment.header,
        column_index=assignment.column_index,
        consulted=True,
        proposal=proposal,
        verification=verification,
        applied_target=applied,
    )
