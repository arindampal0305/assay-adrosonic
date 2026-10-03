"""Proof that Agent 3's reasoning verifier rejects bad LLM claims.

The counterpart of `test_adjudicator_verifier.py`, and written for the same
reason: three live llama3.1 responses on `sample1_basic.csv` were all accepted, and
a verifier that has only ever accepted is indistinguishable from a verifier that
cannot reject. So these tests fabricate one specific defect each, drive it through
the real `verify_reasoning` and the real `reason_once` path via a stub client, and
assert both that the response is discarded and that the recommendation reaches the
other side unchanged.

The defect that matters most here is check 6, `numbers_traceable`. Agent 2's
adjudicator could be fooled into choosing the wrong column; Agent 3's reasoner can
be fooled into *writing a value into a reviewer's head* for a cell the engine
deliberately refused to fill. "The year was probably 1985" is a C-02 violation
wearing the clothes of an explanation, and it is checkable only because a
spreadsheet's numbers are finite and knowable.
"""

from __future__ import annotations

import json

import pytest

from backend.agents.quality.findings import Finding
from backend.agents.quality.models import Recommendation
from backend.agents.quality.reasoner import (
    MAX_REASONED_RECOMMENDATIONS,
    REASON_BELOW_CONFIDENCE,
    SYSTEM_PROMPT,
    Reasoning,
    apply_reasoning,
    parse_reasoning,
    reason_once,
    reason_over,
    verify_reasoning,
)
from backend.llm.client import LLMResponse, LLMUnavailable

# A real finding shape: DQ-05 on sample1_basic.csv, the implausible year that the
# engine refuses to correct because 2031 does not determine the intended value.
FINDINGS = [
    Finding(
        rule="DQ-05",
        row=13,
        field="Year Built",
        severity="High",
        detail="Year Built 2031 is outside 1700-2026, so it cannot be a construction year",
        before="2031",
        after=None,
        op=None,
        source_column="Yr Blt",
        evidence={"value": 2031, "earliest": 1700, "latest": 2026},
    )
]

# And an actionable one, where the engine did compute an `after`.
ZIP_FINDINGS = [
    Finding(
        rule="DQ-08",
        row=3,
        field="Zip",
        severity="Low",
        detail="Zip '2110' has four digits because a leading zero was lost. MA zips "
        "begin with 0. The five-digit code is 02110",
        before="2110",
        after="02110",
        op="zip5",
        source_column="Zip",
        evidence={"state": "MA", "digits": 4},
    )
]


def _flag_rec() -> Recommendation:
    return Recommendation(
        id="R-002",
        action_type="flag_for_review",
        op=None,
        rule="DQ-05",
        field="Year Built",
        source_column="Yr Blt",
        rows=[13],
        before_after=[],
        rationale="DQ-05: Year Built on row 13. Year Built 2031 is outside 1700-2026.",
        confidence=0.65,
        uncertainty="The intended year cannot be derived from 2031",
        severity="High",
    )


def _zip_rec() -> Recommendation:
    return Recommendation(
        id="R-005",
        action_type="data_correction",
        op="zip5",
        rule="DQ-08",
        field="Zip",
        source_column="Zip",
        rows=[3],
        before_after=[{"row": 3, "before": "2110", "after": "02110"}],
        rationale="DQ-08: Zip on row 3. A leading zero was lost; the code is 02110.",
        confidence=0.95,
        uncertainty=None,
        severity="Low",
    )


def _honest() -> Reasoning:
    return Reasoning(
        recommendation_id="R-002",
        op=None,
        cited_row=13,
        cited_before="2031",
        cited_after=None,
        rationale=(
            "Year Built on row 13 reads 2031, which is in the future and therefore "
            "cannot be a construction date. The engine proposes no correction."
        ),
        uncertainty="A reviewer must obtain the real construction year from the broker.",
    )


class StubClient:
    """Returns a scripted body, so the whole verified path runs with no network."""

    def __init__(self, payload, *, raises: Exception | None = None, raw: str | None = None):
        self.payload = payload
        self.raises = raises
        self.raw = raw
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, *, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        if self.raises:
            raise self.raises
        text = self.raw if self.raw is not None else json.dumps(self.payload)
        return LLMResponse(text=text, model="stub")


def _verify(reasoning: Reasoning, rec=None, group=None):
    return verify_reasoning(
        reasoning, rec=rec or _flag_rec(), group=group or FINDINGS
    )


# --------------------------------------------------------------------------
# The honest case, so a rejection below means something.
# --------------------------------------------------------------------------
def test_honest_reasoning_is_accepted():
    result = _verify(_honest())
    assert result.accepted, result.failures
    assert "cited_before_present" in result.verified
    assert "numbers_traceable_rationale" in result.verified


# --------------------------------------------------------------------------
# Check 6 — the invented value. The reason this module exists.
# --------------------------------------------------------------------------
def test_rejects_rationale_that_invents_a_replacement_year():
    reasoning = _honest()
    reasoning.rationale = (
        "Year Built on row 13 reads 2031. This was most likely meant to be 1985, "
        "since the building appears in an older part of the schedule."
    )
    result = _verify(reasoning)
    assert not result.accepted
    assert any("numbers_traceable" in f for f in result.failures)
    assert any("1985" in f for f in result.failures)


def test_rejects_invented_value_in_the_uncertainty_field_too():
    """The uncertainty text reaches the reviewer just as the rationale does, so it
    cannot be the unchecked half of an otherwise checked response."""
    reasoning = _honest()
    reasoning.uncertainty = "The year is probably 2013, a digit transposition of 2031."
    result = _verify(reasoning)
    assert not result.accepted
    assert any("uncertainty contains 2013" in f for f in result.failures)


def test_numbers_present_in_the_evidence_are_allowed():
    """The check must not be so strict that honest prose cannot quote the file.
    1700 and 2026 are in the engine's own detail, and 13 is the affected row."""
    reasoning = _honest()
    reasoning.rationale = (
        "Row 13 holds 2031 in Year Built, outside the 1700 to 2026 window the "
        "engine accepts as a possible construction year."
    )
    assert _verify(reasoning).accepted


def test_percentages_are_allowed_without_appearing_in_the_evidence():
    """A share of something is self-evidently not a cell value."""
    reasoning = _honest()
    reasoning.rationale = (
        "Row 13 is one of the rows affected, roughly 8% of the schedule, and its "
        "Year Built of 2031 cannot be a construction date."
    )
    assert _verify(reasoning).accepted


def test_thousands_separators_do_not_count_as_new_numbers():
    """'1,010,000' and '1010000' are the same figure written two ways, and
    rejecting the comma form would discard a true statement."""
    group = [
        Finding(
            rule="DQ-18", row=4, field="Contents", severity="Medium",
            detail="Contents 1010000 is 10.7x Building Value 94471",
            before="1010000", after=None, op=None, source_column="Contents",
            evidence={"ratio": 10.69, "building_value": 94471},
        )
    ]
    rec = Recommendation(
        id="R-009", action_type="flag_for_review", op=None, rule="DQ-18",
        field="Contents", source_column="Contents", rows=[4], before_after=[],
        rationale="DQ-18: Contents on row 4 is 10.7x Building Value.",
        confidence=0.52, uncertainty="Which of the two figures is wrong is not derivable",
        severity="Medium",
    )
    reasoning = Reasoning(
        recommendation_id="R-009", op=None, cited_row=4, cited_before="1010000",
        cited_after=None,
        rationale=(
            "Contents on row 4 is 1,010,000 against a Building Value of 94,471, a "
            "ratio no ordinary schedule produces."
        ),
        uncertainty="A reviewer must establish which of the two figures is wrong.",
    )
    result = verify_reasoning(reasoning, rec=rec, group=group)
    assert result.accepted, result.failures


# --------------------------------------------------------------------------
# Checks 1-5 — citation integrity.
# --------------------------------------------------------------------------
def test_rejects_response_about_a_different_recommendation():
    reasoning = _honest()
    reasoning.recommendation_id = "R-014"
    result = _verify(reasoning)
    assert not result.accepted
    assert any("recommendation_identity" in f for f in result.failures)


def test_rejects_cited_row_outside_the_affected_rows():
    reasoning = _honest()
    reasoning.cited_row = 7
    result = _verify(reasoning)
    assert not result.accepted
    assert any("cited_row_affected" in f for f in result.failures)


def test_rejects_cell_value_not_present_in_the_file():
    reasoning = _honest()
    reasoning.cited_before = "2013"
    result = _verify(reasoning)
    assert not result.accepted
    assert any("cited_before_present" in f for f in result.failures)


def test_rejects_corrected_value_the_engine_never_computed():
    """The sharp end of C-02: the engine proposed no fix for DQ-05, so any `after`
    in the response was authored by the model."""
    reasoning = _honest()
    reasoning.cited_after = "1985"
    result = _verify(reasoning)
    assert not result.accepted
    assert any("cited_after_computed" in f for f in result.failures)


def test_rejects_corrected_value_that_disagrees_with_the_engines_own():
    reasoning = Reasoning(
        recommendation_id="R-005", op="zip5", cited_row=3, cited_before="2110",
        cited_after="21100",
        rationale=(
            "Zip on row 3 reads 2110 and the engine restores the lost leading zero "
            "to give a five-digit code."
        ),
        uncertainty=None,
    )
    result = verify_reasoning(reasoning, rec=_zip_rec(), group=ZIP_FINDINGS)
    assert not result.accepted
    assert any("cited_after_computed" in f for f in result.failures)


def test_accepts_the_engines_own_corrected_value():
    reasoning = Reasoning(
        recommendation_id="R-005", op="zip5", cited_row=3, cited_before="2110",
        cited_after="02110",
        rationale=(
            "Zip on row 3 reads 2110, four digits for a Massachusetts address, and "
            "restoring the leading zero gives 02110."
        ),
        uncertainty=None,
    )
    result = verify_reasoning(reasoning, rec=_zip_rec(), group=ZIP_FINDINGS)
    assert result.accepted, result.failures
    assert "cited_after_computed" in result.verified


def test_rejects_an_op_the_engine_did_not_propose():
    """Prose cannot promote a flag_for_review into a correction."""
    reasoning = _honest()
    reasoning.op = "set_null"
    result = _verify(reasoning)
    assert not result.accepted
    assert any("op_unchanged" in f for f in result.failures)


# --------------------------------------------------------------------------
# Checks 7-8 — the response has to be usable as output.
# --------------------------------------------------------------------------
def test_rejects_a_rationale_too_short_to_explain_anything():
    reasoning = _honest()
    reasoning.rationale = "Bad year."
    result = _verify(reasoning)
    assert not result.accepted
    assert any("rationale_present" in f for f in result.failures)


def test_rejects_a_rationale_longer_than_a_reviewer_will_read():
    reasoning = _honest()
    reasoning.rationale = "The year on row 13 is wrong. " * 60
    result = _verify(reasoning)
    assert not result.accepted
    assert any("rationale_length" in f for f in result.failures)


def test_rejects_dropping_an_uncertainty_srs_54_requires():
    """Accepting this would strip a required field off the recommendation, so the
    response is discarded rather than half-applied."""
    reasoning = _honest()
    reasoning.uncertainty = None
    result = _verify(reasoning)
    assert not result.accepted
    assert any("uncertainty_required" in f for f in result.failures)


def test_high_confidence_recommendation_needs_no_uncertainty():
    reasoning = Reasoning(
        recommendation_id="R-005", op="zip5", cited_row=3, cited_before="2110",
        cited_after="02110",
        rationale=(
            "Zip on row 3 lost its leading zero to a numeric cell format; the "
            "five-digit Massachusetts code is 02110."
        ),
        uncertainty=None,
    )
    assert verify_reasoning(reasoning, rec=_zip_rec(), group=ZIP_FINDINGS).accepted


def test_accumulates_every_failed_check():
    """A response wrong in four ways must report four failures, so the audit record
    tells a reviewer what happened rather than that something happened."""
    reasoning = Reasoning(
        recommendation_id="R-099",
        op="set_null",
        cited_row=99,
        cited_before="1066",
        cited_after="1985",
        rationale="It should be 1942.",
        uncertainty=None,
    )
    result = _verify(reasoning)
    assert not result.accepted
    names = " ".join(result.failures)
    for check in (
        "recommendation_identity",
        "cited_row_affected",
        "cited_before_present",
        "cited_after_computed",
        "op_unchanged",
        "rationale_present",
        "uncertainty_required",
    ):
        assert check in names, f"{check} not reported in {result.failures}"


# --------------------------------------------------------------------------
# The parser, which must not quietly turn a defect into a decline.
# --------------------------------------------------------------------------
def test_non_integer_row_is_reported_not_silently_dropped():
    reasoning = parse_reasoning({"cited_row": "the thirteenth", "rationale": "x"})
    assert reasoning.cited_row == -1
    result = _verify(reasoning)
    assert any("cited_row_affected" in f for f in result.failures)


def test_blank_strings_parse_to_none_rather_than_empty_citations():
    reasoning = parse_reasoning(
        {"recommendation_id": "  ", "cited_before": "", "cited_after": None}
    )
    assert reasoning.recommendation_id is None
    assert reasoning.cited_before is None


# --------------------------------------------------------------------------
# The end-to-end path: a rejected response must leave the recommendation intact.
# --------------------------------------------------------------------------
def test_rejected_response_leaves_the_recommendation_untouched_and_audits_why():
    original = _flag_rec()
    client = StubClient(
        {
            "recommendation_id": "R-002",
            "op": None,
            "cited_row": 13,
            "cited_before": "2031",
            "cited_after": None,
            "rationale": "Year Built on row 13 is wrong and was probably 1985 given "
            "the age of the surrounding buildings.",
            "uncertainty": "A reviewer should confirm 1985 with the broker.",
        }
    )
    updated, outcome = reason_once(client, original, FINDINGS)

    assert updated == original
    assert updated.rationale == original.rationale
    assert updated.llm_reasoned is False
    assert outcome.applied is False
    record = outcome.audit_record()
    assert record["accepted"] is False
    assert any("1985" in f for f in record["failed_checks"])


def test_accepted_response_replaces_prose_and_only_prose():
    original = _flag_rec()
    client = StubClient(
        {
            "recommendation_id": "R-002",
            "op": None,
            "cited_row": 13,
            "cited_before": "2031",
            "cited_after": None,
            "rationale": "Row 13 gives a Year Built of 2031, a date in the future, so "
            "the cell cannot be a construction year as written.",
            "uncertainty": "The true year must come from the broker who filed the schedule.",
        }
    )
    updated, outcome = reason_once(client, original, FINDINGS)

    assert outcome.applied is True
    assert updated.llm_reasoned is True
    assert updated.rationale != original.rationale
    # Everything Agent 4 would execute is identical.
    assert updated.op == original.op
    assert updated.rows == original.rows
    assert updated.before_after == original.before_after
    assert updated.action_type == original.action_type
    assert updated.confidence == original.confidence
    assert updated.field == original.field


def test_apply_reasoning_cannot_reach_an_executable_field():
    """Belt and braces on the above: even a Reasoning carrying an op and an after
    changes nothing but text, because `apply_reasoning` names the fields it sets."""
    rec = _zip_rec()
    hostile = Reasoning(
        recommendation_id="R-005", op="exclude_row", cited_row=3,
        cited_before="2110", cited_after="99999",
        rationale="A long enough rationale to pass the length floor check here.",
        uncertainty=None,
    )
    updated = apply_reasoning(rec, hostile)
    assert updated.op == "zip5"
    assert updated.before_after[0].after == "02110"
    assert updated.action_type == "data_correction"


def test_provider_failure_leaves_every_recommendation_standing():
    """NFR-REL-02: the engine's own rationale already satisfies FR-DQ-08."""
    original = _flag_rec()
    client = StubClient(None, raises=LLMUnavailable("ollama is not running"))
    updated, outcome = reason_once(client, original, FINDINGS)

    assert updated == original
    assert outcome.error is not None and "LLMUnavailable" in outcome.error
    assert outcome.verification is None
    assert outcome.audit_record()["accepted"] is False


def test_unparseable_body_is_a_failure_not_an_empty_decline():
    client = StubClient(None, raw="I'm afraid I can't help with that.")
    updated, outcome = reason_once(client, _flag_rec(), FINDINGS)
    assert updated == _flag_rec()
    assert outcome.error is not None


# --------------------------------------------------------------------------
# Selection: who gets consulted, and what happens with no client.
# --------------------------------------------------------------------------
def test_no_client_means_no_calls_and_no_changes():
    recs = [_flag_rec(), _zip_rec()]
    out, outcomes = reason_over(None, recs, FINDINGS + ZIP_FINDINGS)
    assert out == recs
    assert outcomes == []


def test_only_low_confidence_recommendations_are_consulted():
    """A 0.95 mechanical zip fix is already fully explained by the arithmetic that
    produced it; spending a model call on it adds a way to go wrong and nothing else."""
    client = StubClient(
        {
            "recommendation_id": "R-002", "op": None, "cited_row": 13,
            "cited_before": "2031", "cited_after": None,
            "rationale": "Row 13 gives a Year Built of 2031, which is in the future "
            "and cannot be a construction year.",
            "uncertainty": "The real year must come from the broker.",
        }
    )
    out, outcomes = reason_over(
        client, [_flag_rec(), _zip_rec()], FINDINGS + ZIP_FINDINGS
    )

    assert len(client.calls) == 1
    assert [o.recommendation_id for o in outcomes] == ["R-002"]
    assert out[0].llm_reasoned is True
    assert out[1].llm_reasoned is False
    assert [r.id for r in out] == ["R-002", "R-005"]


def test_consultation_is_bounded_and_spent_on_the_top_of_the_queue():
    recs = []
    for index in range(MAX_REASONED_RECOMMENDATIONS + 4):
        rec = _flag_rec().model_copy(update={"id": f"R-{index + 1:03d}"})
        recs.append(rec)
    client = StubClient({"recommendation_id": "nope", "rationale": "short"})

    out, outcomes = reason_over(client, recs, FINDINGS)
    assert len(client.calls) == MAX_REASONED_RECOMMENDATIONS
    assert [o.recommendation_id for o in outcomes] == [
        f"R-{i + 1:03d}" for i in range(MAX_REASONED_RECOMMENDATIONS)
    ]
    assert len(out) == len(recs)


def test_reason_threshold_matches_the_srs_uncertainty_threshold():
    """These are the same number for a reason: the recommendations that need a
    stated uncertainty are exactly the ones a human has to decide."""
    from backend.agents.quality.models import UNCERTAINTY_REQUIRED_BELOW

    assert REASON_BELOW_CONFIDENCE == UNCERTAINTY_REQUIRED_BELOW


def test_system_prompt_forbids_proposing_a_value():
    """The prompt and check 4 have to agree. If the prompt ever stops saying this,
    the model will start doing it and the verifier will reject everything."""
    assert "Do NOT propose a replacement value" in SYSTEM_PROMPT
    assert "percentage" in SYSTEM_PROMPT
