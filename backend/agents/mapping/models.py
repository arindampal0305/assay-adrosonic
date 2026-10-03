"""The typed shape of `SOVState.mapping` (SRS 5.3).

These models are the contract between Agent 2 and everything downstream. They are
strict on purpose: a mapping that cannot explain itself should fail validation
rather than reach a reviewer. Three invariants are enforced by validators rather
than by convention, because "the LLM proposes, code disposes" is only true if the
disposing code refuses malformed proposals:

- every mapping carries a non-empty `rationale`
- confidence below `UNCERTAINTY_REQUIRED_BELOW` requires a stated `uncertainty`
- confidence below `REVIEW_REQUIRED_BELOW`, or no target at all, forces
  `flag="human_review_required"`
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.state.target_schema import TARGET_FIELD_NAMES

TargetName = Literal[
    "Reference", "Address", "City", "State", "Zip", "County", "Country",
    "Building Value", "Contents", "BI", "Occupancy", "Construction", "Storeys",
    "Number of Buildings", "Year Built", "Fire Sprinklers (Y/N)", "Other",
]

MappingMethod = Literal[
    "exact", "lexical", "semantic", "fingerprint", "fused", "llm_adjudicated", "unresolved"
]

MappingFlag = Literal["auto_accepted", "review_suggested", "human_review_required"]

UNCERTAINTY_REQUIRED_BELOW = 0.70
REVIEW_REQUIRED_BELOW = 0.50
AUTO_ACCEPT_AT_OR_ABOVE = 0.85

# Guard against the Literal above silently drifting from the frozen SRS 5.1 list.
_literal_targets = set(TargetName.__args__)  # type: ignore[attr-defined]
if _literal_targets != set(TARGET_FIELD_NAMES):
    raise RuntimeError(
        "models.TargetName is out of sync with target_schema.TARGET_FIELD_NAMES"
    )


class ChannelScores(BaseModel):
    """Per-channel scores for the chosen target, reported individually so a
    reviewer can see which evidence carried the decision and which dissented."""

    model_config = ConfigDict(extra="forbid")

    lexical: float = Field(ge=0.0, le=1.0)
    semantic: float = Field(ge=0.0, le=1.0)
    fingerprint: float = Field(ge=0.0, le=1.0)
    fused: float = Field(ge=0.0, le=1.0)
    # Which channels actually voted. A channel can abstain (too few values) or be
    # unavailable (no embedding model), and a zero from an abstention means
    # something completely different from a zero from a disagreement.
    contributing: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    """The citations behind a mapping. Every field here is something a human can
    independently check against the source file."""

    model_config = ConfigDict(extra="forbid")

    glossary_phrase: Optional[str] = None
    lexical_exact: bool = False
    semantic_cosine: Optional[float] = None
    dominant_value_shape: Optional[str] = None
    dominant_shape_share: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sample_values: list[str] = Field(default_factory=list)
    fingerprint_abstained: bool = False


class RunnerUp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Optional[TargetName] = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class ColumnMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source_column: str = Field(min_length=1)
    # 0-based position in the header row, so Agent 4 can address the column
    # positionally even when two source columns share a name.
    source_index: int = Field(ge=0)
    # None is a real, intended outcome: the Hungarian assignment force-matches
    # every column, so a column whose best score is below the floor is explicitly
    # left unmapped rather than given a plausible-looking wrong target.
    target: Optional[TargetName] = None
    confidence: float = Field(ge=0.0, le=1.0)
    method: MappingMethod
    channels: ChannelScores
    runner_up: RunnerUp = Field(default_factory=RunnerUp)
    evidence: Evidence = Field(default_factory=Evidence)
    rationale: str = Field(min_length=1)
    # Required whenever confidence is below UNCERTAINTY_REQUIRED_BELOW: the agent
    # must say what it is unsure about, not merely that it is unsure.
    uncertainty: Optional[str] = None
    flag: MappingFlag
    # Set when the LLM adjudicator was consulted, independently of whether its
    # proposal survived verification.
    adjudicated: bool = False

    @model_validator(mode="after")
    def _enforce_disclosure(self) -> "ColumnMapping":
        if self.target is None and self.flag != "human_review_required":
            raise ValueError(
                f"'{self.source_column}' has no target but is not flagged for review"
            )
        if self.confidence < UNCERTAINTY_REQUIRED_BELOW and not (self.uncertainty or "").strip():
            raise ValueError(
                f"'{self.source_column}' has confidence {self.confidence:.2f} below "
                f"{UNCERTAINTY_REQUIRED_BELOW} but states no uncertainty"
            )
        if self.confidence < REVIEW_REQUIRED_BELOW and self.flag != "human_review_required":
            raise ValueError(
                f"'{self.source_column}' has confidence {self.confidence:.2f} below "
                f"{REVIEW_REQUIRED_BELOW} but is flagged '{self.flag}'"
            )
        return self


class MappingBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheet_identified: str = Field(min_length=1)
    header_row: Optional[int] = Field(default=None, ge=0)
    mappings: list[ColumnMapping] = Field(default_factory=list)
    # Target fields no source column was mapped to. Agent 3 and Agent 4 need this:
    # a missing Building Value is a different problem from a mis-mapped one.
    unmapped_targets: list[TargetName] = Field(default_factory=list)
    unresolved_count: int = Field(default=0, ge=0)
    review_required_count: int = Field(default=0, ge=0)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_channel_available: bool = True
    # Columns whose margin fell below LOW_MARGIN_THRESHOLD, i.e. those *eligible*
    # for adjudication. Reported separately from `adjudicator_consulted` so the two
    # reasons that count can be zero stay distinguishable: no column was close
    # enough to need a second opinion, versus columns needed one and did not get it
    # because no LLM was configured. Those were previously identical in the output.
    low_margin_count: int = Field(default=0, ge=0)
    adjudicator_available: bool = False
    adjudicator_consulted: int = Field(default=0, ge=0)
    adjudicator_accepted: int = Field(default=0, ge=0)
    adjudicator_rejected: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _enforce_counts(self) -> "MappingBlock":
        actual_unresolved = sum(1 for m in self.mappings if m.target is None)
        if actual_unresolved != self.unresolved_count:
            raise ValueError(
                f"unresolved_count={self.unresolved_count} but {actual_unresolved} "
                "mappings have no target"
            )
        assigned = [m.target for m in self.mappings if m.target is not None]
        duplicates = {t for t in assigned if assigned.count(t) > 1}
        if duplicates:
            raise ValueError(
                f"target(s) {sorted(duplicates)} assigned to more than one source column"
            )
        if self.adjudicator_accepted + self.adjudicator_rejected > self.adjudicator_consulted:
            raise ValueError("adjudicator accept+reject exceeds consulted count")
        if self.adjudicator_consulted > self.low_margin_count:
            raise ValueError(
                f"adjudicator_consulted={self.adjudicator_consulted} exceeds "
                f"low_margin_count={self.low_margin_count}: a column was adjudicated "
                "without being eligible for it"
            )
        if self.adjudicator_consulted and not self.adjudicator_available:
            raise ValueError(
                "adjudicator_consulted is non-zero but adjudicator_available is False"
            )
        return self
