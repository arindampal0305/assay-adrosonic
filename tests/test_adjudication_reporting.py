"""Proof that a skipped adjudication is reported rather than silently dropped.

These guard a defect found by reading real output: sample1 has two columns whose
margin falls under `LOW_MARGIN_THRESHOLD` (`Const` at 0.123, `Stories` at 0.1334),
and the run reported `adjudicator_consulted: 0` with no issue and no audit record.
That number is also what you get when no column was close enough to need a second
opinion, so the output could not distinguish "nothing to adjudicate" from "two
things needed adjudicating and no LLM was configured".

The margin check itself was never broken, which is the point: the bug was entirely
in what the run said about itself, so the tests assert on the reported block rather
than on the mapping decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import backend.agents.mapping.agent as agent_mod
from backend.agents.mapping.solver import LOW_MARGIN_THRESHOLD, build_evidence, solve
from backend.agents.sheet_intel.agent import sheet_intelligence
from backend.graph import initial_state
from backend.llm.client import LLMResponse

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "sample1_basic.csv"

# The two columns sample1 is known to leave genuinely contested, with the targets
# the deterministic solver gives them. Named explicitly so that a scoring change
# which quietly removes the ambiguity fails here instead of passing vacuously.
EXPECTED_LOW_MARGIN = {"Const": "Construction", "Stories": "Storeys"}


@pytest.fixture
def mapped_state():
    state = initial_state(SAMPLE)
    patch = sheet_intelligence(state)
    return state.model_copy(update={k: v for k, v in patch.items() if k != "issues"})


def test_sample1_still_has_two_genuinely_low_margin_columns(mapped_state):
    """The premise of every other test here. If this breaks, they prove nothing."""
    _, _, headers, columns = agent_mod._read_primary_columns(mapped_state)
    assignments = solve(build_evidence(headers, columns))
    low = {
        a.header: a.target
        for a in assignments
        if a.target is not None and a.margin < LOW_MARGIN_THRESHOLD
    }
    assert low == EXPECTED_LOW_MARGIN


def test_skipped_adjudication_is_reported(monkeypatch, mapped_state):
    monkeypatch.setattr(agent_mod, "get_client", lambda: None)
    out = agent_mod.schema_mapping(mapped_state)
    block = out["mapping"]

    # The eligible count must survive into the output even though nothing was asked.
    assert block["low_margin_count"] == len(EXPECTED_LOW_MARGIN)
    assert block["adjudicator_available"] is False
    assert block["adjudicator_consulted"] == 0

    assert "adjudicator_unavailable" in {i.rule for i in out["issues"]}

    skipped = [r for r in out["audit"] if r.get("event") == "adjudication_skipped"]
    assert len(skipped) == 1
    record = skipped[0]
    assert record["reason"] == "no_llm_configured"
    # The record must name the columns. "2 columns were skipped" is not actionable;
    # knowing it was Const and Stories is.
    named = {c["source_column"]: c["assigned_target"] for c in record["low_margin_columns"]}
    assert named == EXPECTED_LOW_MARGIN
    for entry in record["low_margin_columns"]:
        assert entry["margin"] < LOW_MARGIN_THRESHOLD
        assert entry["runner_up"]


def test_no_skip_record_when_adjudicator_is_available(monkeypatch, mapped_state):
    """The skip report must be conditional, not unconditional."""

    class Decliner:
        """Declines every column, which is a valid answer the verifier rejects."""

        def complete_json(self, *, system: str, user: str) -> LLMResponse:
            return LLMResponse(
                text=json.dumps(
                    {"target": None, "rationale": "The two candidates are not separable."}
                ),
                model="stub",
            )

    monkeypatch.setattr(agent_mod, "get_client", lambda: Decliner())
    out = agent_mod.schema_mapping(mapped_state)
    block = out["mapping"]

    assert block["adjudicator_available"] is True
    assert block["adjudicator_consulted"] == len(EXPECTED_LOW_MARGIN)
    assert block["low_margin_count"] == len(EXPECTED_LOW_MARGIN)
    assert not [r for r in out["audit"] if r.get("event") == "adjudication_skipped"]
    assert "adjudicator_unavailable" not in {i.rule for i in out["issues"]}


def test_adjudicator_is_asked_about_exactly_the_low_margin_columns(
    monkeypatch, mapped_state
):
    """A model call on a column that matched its glossary exactly is wasted spend,
    so the selection must be the low-margin set and nothing wider."""
    asked: list[str] = []

    class Recorder:
        def complete_json(self, *, system: str, user: str) -> LLMResponse:
            payload = json.loads(user)
            asked.append(payload["source_column"])
            # Every candidate offered must be a real target the column could hold.
            assert 1 <= len(payload["candidate_targets"]) <= 2
            assert payload["deterministic_scores"]["margin"] < LOW_MARGIN_THRESHOLD
            return LLMResponse(
                text=json.dumps({"target": None, "rationale": "Not separable."}),
                model="stub",
            )

    monkeypatch.setattr(agent_mod, "get_client", lambda: Recorder())
    agent_mod.schema_mapping(mapped_state)
    assert asked == list(EXPECTED_LOW_MARGIN)
