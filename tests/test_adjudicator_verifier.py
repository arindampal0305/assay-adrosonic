"""Proof that the code-side evidence verifier rejects bad LLM citations.

A guardrail that has never been observed rejecting anything is not evidence of a
working guardrail — it is equally consistent with a guardrail that cannot fire. So
these tests do not wait for a real model to hallucinate. They construct proposals
that are deliberately wrong in one specific way each, drive them through the real
`verify_proposal` and the real `adjudicate` path via a stub client, and assert both
that the proposal is discarded and that the deterministic mapping is untouched.

Each test below fabricates exactly one defect so that a failure names the check
that broke, rather than "something was rejected".
"""

from __future__ import annotations

import json
from typing import get_args

import pytest

from backend.agents.mapping.adjudicator import (
    SYSTEM_PROMPT,
    VALUE_SHAPES,
    AdjudicationOutcome,
    Proposal,
    adjudicate,
    parse_proposal,
    verify_proposal,
)
from backend.agents.mapping.solver import Assignment, Cell
from backend.agents.mapping.targets import ValueShape
from backend.llm.client import LLMResponse, LLMUnavailable

# A realistic low-margin column: the header is ambiguous and the values are a
# controlled vocabulary, which is exactly the situation adjudication exists for.
HEADER = "Const Type"
VALUES = ["Masonry", "Frame", "Masonry", "Joisted Masonry", "Frame"]
CANDIDATES = ("Construction", "Occupancy")


def _cell() -> Cell:
    return Cell(
        lexical=1.0, semantic=0.74, fingerprint=0.6, fused=0.77,
        contributing=["lexical", "semantic", "fingerprint"],
        glossary_phrase="construction type", lexical_exact=False,
        semantic_cosine=0.684, dominant_shape="category", dominant_share=0.6,
        fingerprint_abstained=False,
    )


def _assignment() -> Assignment:
    return Assignment(
        column_index=9, header=HEADER, target="Construction", fused=0.77,
        margin=0.115, runner_up="Occupancy", runner_up_score=0.655, cell=_cell(),
        preferred_target="Construction", preferred_score=0.77,
        rejected_below_floor=False,
    )


class StubClient:
    """An LLMClient that returns whatever body a test hands it. No key, no network.

    This is the whole reason `LLMClient` is a Protocol rather than a concrete
    OpenAI dependency: the adjudication path is fully exercisable offline.
    """

    def __init__(self, payload: object, *, raise_error: Exception | None = None) -> None:
        self.payload = payload
        self.raise_error = raise_error
        self.calls = 0

    def complete_json(self, *, system: str, user: str) -> LLMResponse:
        self.calls += 1
        if self.raise_error:
            raise self.raise_error
        body = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return LLMResponse(text=body, model="stub")


def _verify(**overrides) -> object:
    """Build an otherwise-valid proposal with specific fields overridden."""
    base = {
        "source_column": HEADER,
        "target": "Construction",
        "glossary_phrase": "construction type",
        "value_shape": "category",
        "sample_value": "Masonry",
        "rationale": "The values are ISO construction classes, not building uses.",
        "confidence": 0.88,
    }
    base.update(overrides)
    return verify_proposal(
        parse_proposal(base, base["source_column"]),
        header=HEADER,
        values=VALUES,
        candidates=CANDIDATES,
    )


# --------------------------------------------------------------------------
# The control: a fully honest proposal must be accepted, otherwise the
# rejections below prove nothing.
# --------------------------------------------------------------------------


def test_honest_proposal_is_accepted():
    result = _verify()
    assert result.accepted, result.failures
    assert "glossary_phrase_exists" in result.verified
    assert "glossary_phrase_matches_header" in result.verified
    assert "value_shape_observed" in result.verified
    assert "sample_value_present" in result.verified


# --------------------------------------------------------------------------
# One fabricated defect per test.
# --------------------------------------------------------------------------


def test_rejects_glossary_phrase_that_does_not_exist():
    """The model cites a phrase that sounds plausible but is not in the glossary."""
    result = _verify(glossary_phrase="building fabric classification")
    assert not result.accepted
    assert any(f.startswith("glossary_phrase_exists") for f in result.failures), result.failures


def test_rejects_real_glossary_phrase_belonging_to_another_target():
    """`yr blt` is a genuine glossary phrase — for Year Built, not Construction.

    This is the subtle hallucination: every token of the citation is real, but the
    association it asserts is invented.
    """
    result = _verify(glossary_phrase="yr blt")
    assert not result.accepted
    assert any(f.startswith("glossary_phrase_exists") for f in result.failures), result.failures


def test_rejects_glossary_phrase_unrelated_to_the_actual_header():
    """A phrase real and correctly owned by Construction, but bearing no
    resemblance to the header actually in the file."""
    result = _verify(glossary_phrase="roof and wall")
    assert not result.accepted
    assert any(
        f.startswith("glossary_phrase_matches_header") for f in result.failures
    ), result.failures


def test_rejects_value_shape_the_column_does_not_have():
    """The model claims the column is currency. It is a controlled vocabulary."""
    result = _verify(value_shape="currency")
    assert not result.accepted
    failure = next(f for f in result.failures if f.startswith("value_shape_observed"))
    # The message must name both the rejected claim and what the column actually
    # is. The profiler labels this column 'place_name', which is interchangeable
    # with 'category' here; 'currency' belongs to neither, which is why it fails.
    assert "currency" in failure
    assert "place_name" in failure or "category" in failure


@pytest.mark.parametrize("prose", ["nominal", "small whole number", "text"])
def test_prose_value_shape_fails_as_vocabulary_not_as_a_false_citation(prose):
    """Both strings here are verbatim real llama3.1 output on sample1.

    The model named the correct target for `Const` and `Stories` and described the
    shape in its own words. The verifier rejected both — correctly, since an
    unrecognised token cannot be checked — but reported it as "covers 0% of
    values; the column is actually 'place_name'", which reads as a caught
    hallucination about the data. It was nothing of the kind.

    So this asserts the distinction, not just the rejection: a token outside the
    vocabulary must fail `value_shape_vocabulary` and must NOT claim the column
    looks like something else.
    """
    result = _verify(value_shape=prose)
    assert not result.accepted
    failure = next(f for f in result.failures if f.startswith("value_shape_vocabulary"))
    assert prose in failure
    assert not any(f.startswith("value_shape_observed") for f in result.failures)
    # The message has to be actionable: it names the vocabulary it wanted.
    assert "small_int" in failure and "category" in failure


def test_system_prompt_states_every_legal_value_shape():
    """The drift guard for the bug above.

    `VALUE_SHAPES` comes from `get_args(ValueShape)`, so a new shape added to the
    enum reaches the prompt automatically. This test fails if someone replaces the
    derived list with a hand-written one, which is how the prompt and the verifier
    disagreed in the first place.
    """
    assert set(VALUE_SHAPES) == set(get_args(ValueShape))
    for shape in VALUE_SHAPES:
        assert shape in SYSTEM_PROMPT, f"prompt never names the legal shape '{shape}'"


def test_rejects_sample_value_not_present_in_the_column():
    """A quoted value that never appears in the data — the most directly checkable
    hallucination there is."""
    result = _verify(sample_value="Reinforced Concrete")
    assert not result.accepted
    assert any(f.startswith("sample_value_present") for f in result.failures), result.failures


def test_rejects_target_outside_the_seventeen_fields():
    result = _verify(target="Square Footage")
    assert not result.accepted
    assert any(f.startswith("target_exists") for f in result.failures), result.failures


def test_rejects_target_not_among_the_offered_candidates():
    """`Year Built` is a real target, just not one of the two under adjudication.
    The model may choose between the candidates; it may not introduce a third."""
    result = _verify(target="Year Built", glossary_phrase=None, value_shape=None,
                     sample_value=None)
    assert not result.accepted
    assert any(f.startswith("target_in_candidates") for f in result.failures), result.failures


def test_rejects_proposal_about_a_different_column():
    result = _verify(source_column="Yr Blt")
    assert not result.accepted
    assert any(f.startswith("column_identity") for f in result.failures), result.failures


def test_rejects_proposal_with_no_rationale():
    result = _verify(rationale="")
    assert not result.accepted
    assert any(f.startswith("rationale") for f in result.failures), result.failures


def test_explicit_decline_is_not_accepted_as_a_mapping():
    result = _verify(target=None)
    assert not result.accepted


def test_accumulates_every_failed_check():
    """Multiple defects must all be reported, so an audit record shows the full
    extent of what was wrong rather than the first thing noticed."""
    result = _verify(
        glossary_phrase="building fabric classification",
        value_shape="currency",
        sample_value="Reinforced Concrete",
    )
    assert not result.accepted
    assert len(result.failures) >= 3, result.failures


# --------------------------------------------------------------------------
# End to end through adjudicate(): a rejected proposal must leave the
# deterministic assignment untouched and must still be audited.
# --------------------------------------------------------------------------


def test_rejected_proposal_does_not_change_the_mapping_and_is_audited():
    """The guardrail's actual contract: a hallucinated citation can produce an
    audit entry and nothing else. It cannot move a single cell."""
    assignment = _assignment()
    client = StubClient(
        {
            "target": "Occupancy",
            "glossary_phrase": "tenancy",
            "value_shape": "currency",
            "sample_value": "Retail Store",
            "rationale": "These look like building uses.",
            "confidence": 0.93,
        }
    )
    outcome = adjudicate(client, assignment, VALUES)

    assert client.calls == 1
    assert outcome.consulted is True
    assert outcome.proposal is not None
    assert outcome.proposal.target == "Occupancy"
    # The proposal asked for Occupancy at 0.93 confidence and got nothing.
    assert outcome.verification is not None and not outcome.verification.accepted
    assert outcome.applied_target is None

    record = outcome.audit_record()
    assert record["accepted"] is False
    assert record["proposed_target"] == "Occupancy"
    assert record["applied_target"] is None
    assert record["failed_checks"], "a rejection must name the checks that failed"
    assert record["event"] == "llm_adjudication"


def test_accepted_proposal_can_override_the_deterministic_assignment():
    """The converse: a fully verifiable proposal *is* allowed to change the
    mapping, otherwise adjudication would be decorative.

    A separate scenario is needed here, because an override is only acceptable
    when its citations hold. `Occ Class` over warehouse/office values is a column
    the scorer can plausibly hand to Construction while the glossary phrase
    'occ class' genuinely matches the header and the values genuinely are a
    category — so every check can pass on a target the solver did not choose.
    """
    header = "Occ Class"
    values = ["Warehouse", "Office", "Retail", "Warehouse", "Office"]
    assignment = Assignment(
        column_index=8, header=header, target="Construction", fused=0.64,
        margin=0.04, runner_up="Occupancy", runner_up_score=0.60, cell=_cell(),
        preferred_target="Construction", preferred_score=0.64,
        rejected_below_floor=False,
    )
    client = StubClient(
        {
            "target": "Occupancy",
            "glossary_phrase": "occ class",
            "value_shape": "category",
            "sample_value": "Warehouse",
            "rationale": "Warehouse, Office and Retail are building uses, not materials.",
            "confidence": 0.86,
        }
    )
    outcome = adjudicate(client, assignment, values)
    assert outcome.verification is not None and outcome.verification.accepted, (
        outcome.verification.failures if outcome.verification else None
    )
    assert outcome.applied_target == "Occupancy"
    assert outcome.audit_record()["accepted"] is True


def test_unparseable_response_leaves_the_assignment_untouched():
    outcome = adjudicate(StubClient("this is not json at all"), _assignment(), VALUES)
    assert outcome.applied_target is None
    assert outcome.proposal is None
    assert outcome.error is not None
    assert outcome.audit_record()["accepted"] is False


def test_json_array_response_is_treated_as_a_failure_not_an_empty_object():
    """A response of the wrong *shape* must surface as an error rather than being
    coerced into an empty proposal, which would read as a model decline."""
    outcome = adjudicate(StubClient("[1, 2, 3]"), _assignment(), VALUES)
    assert outcome.error is not None
    assert outcome.applied_target is None


def test_provider_failure_is_absorbed_not_raised():
    outcome = adjudicate(
        StubClient(None, raise_error=LLMUnavailable("connection reset")),
        _assignment(),
        VALUES,
    )
    assert outcome.applied_target is None
    assert outcome.error is not None and "connection reset" in outcome.error


def test_markdown_fenced_json_is_still_parsed():
    """Models wrap JSON in code fences constantly; that is sloppiness, not a
    hallucination, and should not cost a valid proposal."""
    fenced = (
        "```json\n"
        + json.dumps(
            {
                "target": "Construction",
                "glossary_phrase": "construction type",
                "value_shape": "category",
                "sample_value": "Frame",
                "rationale": "ISO construction classes.",
                "confidence": 0.9,
            }
        )
        + "\n```"
    )
    outcome = adjudicate(StubClient(fenced), _assignment(), VALUES)
    assert outcome.proposal is not None
    assert outcome.verification is not None and outcome.verification.accepted, (
        outcome.verification.failures if outcome.verification else None
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
