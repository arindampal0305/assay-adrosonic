"""The typed output of Agent 3, plus the two frozen SRS catalogues it works from.

Three things live here, together on purpose:

- `RULE_CATALOGUE` — SRS 9.2, all eighteen rules with the fields they inspect and
  the severity the SRS assigns them. Transcribed once so a rule implementation
  cannot invent its own severity and the set cannot silently lose a member.
- `OP_WHITELIST` — SRS 9.1. Agent 4 may execute nothing else, so a recommendation
  naming an op outside this tuple must be unconstructible rather than merely
  discouraged.
- `Recommendation` / `QualityBlock` — SRS 5.4 and the report Agent 3 writes to
  state, strict in the same way `mapping/models.py` is strict: a recommendation
  that cannot explain itself, or that claims an op nobody can run, fails
  validation instead of reaching a reviewer.

The validators here are the contract gate for Agent 3's own output. "The LLM
proposes, code disposes" is only true if the disposing code refuses malformed
proposals, and that applies to proposals this agent generates itself just as much
as to the ones it receives from a model.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.state.sov_state import Severity
from backend.state.target_schema import TARGET_FIELD_NAMES

# --------------------------------------------------------------------------
# SRS 9.1 — the operations Agent 4 can execute.
# --------------------------------------------------------------------------
# "Agent 4 can execute only these operations; the LLM may choose one and set its
# parameters, but never supply output values." The second half of that sentence is
# why `set_null` exists at all: a placeholder is replaced with an empty cell and
# never with a guessed value, which is what keeps C-02 intact.
TransformOp = Literal[
    "rename",
    "strip_currency_to_float",
    "to_int",
    "to_float",
    "map_values",
    "state_to_abbrev",
    "zip5",
    "split_column",
    "merge_columns",
    "trim_normalise",
    "set_null",
    "exclude_row",
]

OP_WHITELIST: tuple[str, ...] = get_args(TransformOp)

ActionType = Literal[
    "column_mapping", "data_correction", "standardisation", "flag_for_review"
]

RecommendationStatus = Literal[
    "pending", "approved", "rejected", "edited", "escalated"
]

# Which SRS 5.4 action_type each op belongs to. Derived rather than restated at
# every call site so the two fields cannot disagree.
ACTION_TYPE_FOR_OP: dict[str, ActionType] = {
    "rename": "column_mapping",
    "strip_currency_to_float": "data_correction",
    "to_int": "data_correction",
    "to_float": "data_correction",
    "map_values": "standardisation",
    "state_to_abbrev": "standardisation",
    "zip5": "data_correction",
    "split_column": "data_correction",
    "merge_columns": "data_correction",
    "trim_normalise": "standardisation",
    "set_null": "data_correction",
    "exclude_row": "data_correction",
}

if set(ACTION_TYPE_FOR_OP) != set(OP_WHITELIST):  # pragma: no cover - import guard
    raise RuntimeError("ACTION_TYPE_FOR_OP is out of sync with the SRS 9.1 whitelist")

# Confidence below this requires a stated uncertainty (SRS 5.4).
UNCERTAINTY_REQUIRED_BELOW = 0.70
# SRS 5.4: "attempts | Integer | Re-reasoning count, maximum 2". FR-DQ-10 escalates
# after two rejections. Agent 3 only ever writes 0 here; the ceiling is declared
# now so the review loop cannot exceed it later without failing validation.
MAX_ATTEMPTS = 2
# FR-UI-03 wants a before/after preview for at least 20 rows, so a recommendation
# carries up to this many worked examples. `rows` still lists every affected row;
# capping the examples keeps a 400-row finding reviewable without hiding its size.
MAX_BEFORE_AFTER_EXAMPLES = 20


# --------------------------------------------------------------------------
# SRS 9.2 — the data quality rule catalogue.
# --------------------------------------------------------------------------
class RuleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    check: str
    fields: tuple[str, ...]
    severity: Severity
    # Set where the SRS gives a field-dependent severity, as DQ-01 does
    # ("Medium; High for value fields and Reference").
    elevated_severity: Optional[Severity] = None
    elevated_fields: tuple[str, ...] = ()

    def severity_for(self, field: Optional[str]) -> Severity:
        if field and self.elevated_severity and field in self.elevated_fields:
            return self.elevated_severity
        return self.severity


VALUE_FIELDS: tuple[str, ...] = ("Building Value", "Contents", "BI", "Other")
INTEGER_FIELDS: tuple[str, ...] = (
    "Zip", "Storeys", "Number of Buildings", "Year Built",
)
SPRINKLER_FIELD = "Fire Sprinklers (Y/N)"
# SRS 5.1: "One of Y, N, Y13, Y(13R)".
SPRINKLER_ENUM: tuple[str, ...] = ("Y", "N", "Y13", "Y(13R)")

_SPECS: tuple[RuleSpec, ...] = (
    RuleSpec(
        rule="DQ-01",
        check="Missing value",
        fields=TARGET_FIELD_NAMES,
        severity="Medium",
        elevated_severity="High",
        elevated_fields=VALUE_FIELDS + ("Reference",),
    ),
    RuleSpec(
        rule="DQ-02",
        check="Not numeric after cleaning",
        fields=VALUE_FIELDS,
        severity="High",
    ),
    RuleSpec(
        rule="DQ-03",
        check="Not a whole number",
        fields=INTEGER_FIELDS,
        severity="High",
    ),
    RuleSpec(
        rule="DQ-04", check="Negative amount", fields=VALUE_FIELDS, severity="High"
    ),
    RuleSpec(
        rule="DQ-05",
        check="Year Built after current year, or before 1700",
        fields=("Year Built",),
        severity="High",
    ),
    RuleSpec(
        rule="DQ-06",
        check="Storeys or Number of Buildings below 1",
        fields=("Storeys", "Number of Buildings"),
        severity="High",
    ),
    RuleSpec(
        rule="DQ-07",
        check="4-digit Zip in a state whose zips do not start with 0",
        fields=("Zip", "State"),
        severity="Medium",
    ),
    RuleSpec(rule="DQ-08", check="Zip+4 or text in Zip", fields=("Zip",), severity="Low"),
    RuleSpec(
        rule="DQ-09",
        check="Full state name or non-standard code",
        fields=("State",),
        severity="Low",
    ),
    RuleSpec(
        rule="DQ-10",
        check="Currency symbol or thousands separator in a numeric field",
        fields=VALUE_FIELDS,
        severity="Low",
    ),
    RuleSpec(
        rule="DQ-11",
        check=(
            "Sprinkler value outside Y, N, Y13, Y(13R); ambiguous terms "
            '("Partial") always go to a human'
        ),
        fields=(SPRINKLER_FIELD,),
        severity="Medium",
    ),
    RuleSpec(
        rule="DQ-12",
        check="Placeholder pattern: Year Built 0 or repeated 1900, Storeys 0, value of 1",
        fields=("Year Built", "Storeys") + VALUE_FIELDS,
        severity="Medium",
    ),
    RuleSpec(
        rule="DQ-13",
        check="Duplicate location (same normalised address and zip)",
        fields=("Address", "Zip"),
        severity="Medium",
    ),
    RuleSpec(
        rule="DQ-14",
        check="Wood or frame construction with more than 6 storeys",
        fields=("Construction", "Storeys"),
        severity="Low",
    ),
    RuleSpec(
        rule="DQ-15",
        check="Totals or subtotal row inside the data",
        fields=("__row__",),
        severity="High",
    ),
    RuleSpec(
        rule="DQ-16",
        check="Components disagree with a source TIV column by more than 1%",
        fields=VALUE_FIELDS,
        severity="Medium",
    ),
    RuleSpec(
        rule="DQ-17", check="Mixed date formats in one column", fields=("Year Built",), severity="Low"
    ),
    RuleSpec(
        rule="DQ-18",
        check="Contents or BI above 10x Building Value",
        fields=("Contents", "BI"),
        severity="Low",
    ),
)

RULE_CATALOGUE: dict[str, RuleSpec] = {spec.rule: spec for spec in _SPECS}

# The catalogue is DQ-01 to DQ-18 with no gaps. Asserted at import because a
# missing rule is far easier to notice here than in a run that quietly never
# reports it.
_expected = {f"DQ-{n:02d}" for n in range(1, 19)}
if set(RULE_CATALOGUE) != _expected:  # pragma: no cover - import guard
    raise RuntimeError(
        "SRS 9.2 catalogue is incomplete: "
        f"missing={sorted(_expected - set(RULE_CATALOGUE))} "
        f"unexpected={sorted(set(RULE_CATALOGUE) - _expected)}"
    )


# --------------------------------------------------------------------------
# SRS 5.4 — the Recommendation.
# --------------------------------------------------------------------------
class BeforeAfter(BaseModel):
    """One worked example, computed by the engine from a real row (FR-DQ-09).

    SRS 5.4 describes `before, after` as a "List of pairs". They are carried as
    one list of named triples rather than two parallel lists because parallel
    lists can desynchronise, and because a pair that does not name its row cannot
    be checked against the source file by the reviewer it exists to convince.

    `after=None` means "this cell is left empty", which is the only output
    `set_null` may produce. It never means "unchanged" and never means "a value
    we have not worked out yet".
    """

    model_config = ConfigDict(extra="forbid")

    # 1-based row number as the spreadsheet shows it, so a reviewer can open the
    # source file and look at this row without arithmetic.
    row: int = Field(ge=1)
    before: str
    after: Optional[str] = None


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(pattern=r"^R-\d{3,}$")
    action_type: ActionType
    # None only for flag_for_review (SRS 5.4: "One op from catalogue 9.1, or none
    # for flag_for_review").
    op: Optional[TransformOp] = None
    rule: str
    # The target field, or None when the finding is about a whole row (DQ-15).
    field: Optional[str] = None
    # The source column this field was mapped from, so a reviewer can find it in
    # the original sheet. Agent 3 never renames anything; it reports both names.
    source_column: Optional[str] = None
    rows: list[int] = Field(default_factory=list)
    before_after: list[BeforeAfter] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: Optional[str] = None
    severity: Severity
    status: RecommendationStatus = "pending"
    attempts: int = Field(default=0, ge=0, le=MAX_ATTEMPTS)
    # True when the LLM reasoner's text survived verification and replaced the
    # rule-based rationale. Reported so a reader can tell engine prose from model
    # prose without guessing.
    llm_reasoned: bool = False

    @model_validator(mode="after")
    def _enforce_srs_54(self) -> "Recommendation":
        if self.rule not in RULE_CATALOGUE:
            raise ValueError(f"rule '{self.rule}' is not in the SRS 9.2 catalogue")

        if self.action_type == "flag_for_review":
            if self.op is not None:
                raise ValueError(
                    f"{self.id}: flag_for_review carries op '{self.op}'; SRS 5.4 "
                    "allows an op only on an actionable recommendation"
                )
            if self.before_after:
                raise ValueError(
                    f"{self.id}: flag_for_review proposes no change, so it cannot "
                    "carry before/after examples"
                )
        else:
            if self.op is None:
                raise ValueError(
                    f"{self.id}: action_type '{self.action_type}' requires an op from "
                    "the SRS 9.1 whitelist"
                )
            if ACTION_TYPE_FOR_OP[self.op] != self.action_type:
                raise ValueError(
                    f"{self.id}: op '{self.op}' belongs to action_type "
                    f"'{ACTION_TYPE_FOR_OP[self.op]}', not '{self.action_type}'"
                )
            if not self.before_after:
                raise ValueError(
                    f"{self.id}: op '{self.op}' changes data but shows no worked "
                    "example; FR-DQ-09 requires engine-computed before/after"
                )

        if not self.rows:
            raise ValueError(f"{self.id}: names no affected rows")

        known = set(self.rows)
        stray = [e.row for e in self.before_after if e.row not in known]
        if stray:
            raise ValueError(
                f"{self.id}: before/after examples cite rows {sorted(set(stray))} "
                "that are not in the affected row list"
            )

        if self.confidence < UNCERTAINTY_REQUIRED_BELOW and not (self.uncertainty or "").strip():
            raise ValueError(
                f"{self.id}: confidence {self.confidence:.2f} is below "
                f"{UNCERTAINTY_REQUIRED_BELOW} but states no uncertainty"
            )

        # `set_null` is the one op whose output is fixed by C-02. Enforcing it here
        # rather than trusting the caller means no future rule can smuggle a
        # replacement value in through it.
        if self.op == "set_null":
            filled = [e for e in self.before_after if e.after is not None]
            if filled:
                raise ValueError(
                    f"{self.id}: set_null must leave the cell empty, but row "
                    f"{filled[0].row} proposes '{filled[0].after}'"
                )
        return self


# --------------------------------------------------------------------------
# The report Agent 3 writes to `SOVState.quality`.
# --------------------------------------------------------------------------
class FieldCompleteness(BaseModel):
    """Per-field completeness (FR-DQ-01)."""

    model_config = ConfigDict(extra="forbid")

    field: str
    source_column: Optional[str] = None
    mapped: bool
    rows: int = Field(ge=0)
    non_null: int = Field(ge=0)
    completeness: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _consistent(self) -> "FieldCompleteness":
        if self.non_null > self.rows:
            raise ValueError(
                f"{self.field}: non_null={self.non_null} exceeds rows={self.rows}"
            )
        expected = round(self.non_null / self.rows, 4) if self.rows else 0.0
        if abs(expected - self.completeness) > 1e-4:
            raise ValueError(
                f"{self.field}: completeness {self.completeness} does not match "
                f"{self.non_null}/{self.rows}"
            )
        if self.mapped and self.source_column is None:
            raise ValueError(f"{self.field}: mapped but names no source column")
        return self


class QualityBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheet_identified: str = Field(min_length=1)
    rows_analysed: int = Field(ge=0)
    fields_mapped: int = Field(ge=0, le=len(TARGET_FIELD_NAMES))
    completeness: list[FieldCompleteness] = Field(default_factory=list)
    # FR-DQ-07. A weighted blend, deliberately simple; see `completeness.py` for
    # the formula and why its weights are defaults rather than tuned values.
    intake_quality_score: float = Field(ge=0.0, le=1.0)
    score_components: dict[str, float] = Field(default_factory=dict)

    issue_count: int = Field(default=0, ge=0)
    issues_by_severity: dict[str, int] = Field(default_factory=dict)
    issues_by_rule: dict[str, int] = Field(default_factory=dict)
    # Rules that ran and found nothing, named rather than left to inference. The
    # same lesson as `low_margin_count` in Agent 2: "no issues" and "the rule never
    # ran" are different facts and must not share a representation.
    rules_evaluated: list[str] = Field(default_factory=list)
    rules_not_applicable: dict[str, str] = Field(default_factory=dict)

    recommendation_count: int = Field(default=0, ge=0)
    recommendations_by_action: dict[str, int] = Field(default_factory=dict)
    # Rows a recommendation proposes excluding (DQ-15). Reported, never applied:
    # FR-SHT-05 and the brief both require a totals row to be flagged, not dropped.
    totals_rows: list[int] = Field(default_factory=list)

    schema_validation_ran: bool = True
    reasoner_available: bool = False
    reasoner_consulted: int = Field(default=0, ge=0)
    reasoner_accepted: int = Field(default=0, ge=0)
    reasoner_rejected: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _enforce_counts(self) -> "QualityBlock":
        if self.fields_mapped != sum(1 for c in self.completeness if c.mapped):
            raise ValueError(
                f"fields_mapped={self.fields_mapped} disagrees with the completeness list"
            )
        if sum(self.issues_by_severity.values()) != self.issue_count:
            raise ValueError("issues_by_severity does not sum to issue_count")
        if sum(self.issues_by_rule.values()) != self.issue_count:
            raise ValueError("issues_by_rule does not sum to issue_count")
        if sum(self.recommendations_by_action.values()) != self.recommendation_count:
            raise ValueError(
                "recommendations_by_action does not sum to recommendation_count"
            )
        unknown = set(self.issues_by_rule) - set(RULE_CATALOGUE)
        if unknown:
            raise ValueError(f"issues_by_rule names unknown rules {sorted(unknown)}")
        # Every one of the eighteen rules must be accounted for: it either ran, or
        # it is explicitly recorded as not applicable with a reason.
        accounted = set(self.rules_evaluated) | set(self.rules_not_applicable)
        if accounted != set(RULE_CATALOGUE):
            raise ValueError(
                "every SRS 9.2 rule must be either evaluated or declared "
                "not-applicable; unaccounted="
                f"{sorted(set(RULE_CATALOGUE) - accounted)}"
            )
        overlap = set(self.rules_evaluated) & set(self.rules_not_applicable)
        if overlap:
            raise ValueError(f"rules both evaluated and not applicable: {sorted(overlap)}")
        if self.reasoner_accepted + self.reasoner_rejected > self.reasoner_consulted:
            raise ValueError("reasoner accept+reject exceeds consulted count")
        if self.reasoner_consulted and not self.reasoner_available:
            raise ValueError(
                "reasoner_consulted is non-zero but reasoner_available is False"
            )
        return self


def dump_recommendations(recs: list[Recommendation]) -> list[dict[str, Any]]:
    return [r.model_dump() for r in recs]
