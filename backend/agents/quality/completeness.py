"""Completeness per field (FR-DQ-01) and the intake quality score (FR-DQ-07).

Two things that look like one. Completeness is a measurement: what share of each
mapped field's cells hold a value. The intake score is a *summary*, and summaries
lose information by construction, so this module reports the parts alongside the
whole — `score_components` exists so that a reviewer who distrusts the number can
see which term produced it.

**On the weights.** `FIELD_WEIGHTS` and `SCORE_WEIGHTS` are defaults, not tuned
values, and the brief says so explicitly: thresholds and severity weights get
tuned once Agent 4 and the review loop are wired and the whole pipeline runs end
to end. They are at least *principled* defaults — a field carries weight in
proportion to whether an underwriter can price the risk without it, which is why
Building Value outranks County — but no claim is made that 0.6/0.25/0.15 is
better than 0.5/0.3/0.2. The structure is built so that tuning is a change to
these two tables and to nothing else.

**An unmapped field is 0% complete, not absent.** FR-DQ-01 asks for completeness
"for every mapped field", and the per-field list honours that with `mapped`.
But the score is computed over all seventeen, because a file missing Year Built
entirely is in worse shape than one that merely has it empty on three rows, and a
score that ignored unmapped fields would rank the two the same way.
"""

from __future__ import annotations

from backend.agents.quality.findings import Finding
from backend.agents.quality.frame import MappedFrame
from backend.agents.quality.models import FieldCompleteness, VALUE_FIELDS
from backend.state.target_schema import TARGET_FIELD_NAMES

# How much each target field matters to an underwriter who has to price the risk.
# Default weights (see the module docstring): the four value fields and the
# address that locates them are what a quote cannot be produced without.
FIELD_WEIGHTS: dict[str, float] = {
    "Reference": 0.6,
    "Address": 1.0,
    "City": 0.7,
    "State": 0.9,
    "Zip": 0.9,
    "County": 0.2,
    "Country": 0.2,
    "Building Value": 1.0,
    "Contents": 0.8,
    "BI": 0.7,
    "Occupancy": 0.8,
    "Construction": 0.9,
    "Storeys": 0.6,
    "Number of Buildings": 0.4,
    "Year Built": 0.7,
    "Fire Sprinklers (Y/N)": 0.7,
    "Other": 0.3,
}

if set(FIELD_WEIGHTS) != set(TARGET_FIELD_NAMES):  # pragma: no cover - import guard
    raise RuntimeError(
        "FIELD_WEIGHTS must cover exactly the SRS 5.1 fields; "
        f"missing={sorted(set(TARGET_FIELD_NAMES) - set(FIELD_WEIGHTS))} "
        f"unexpected={sorted(set(FIELD_WEIGHTS) - set(TARGET_FIELD_NAMES))}"
    )

# The three things the score is made of, and how much each counts.
SCORE_WEIGHTS: dict[str, float] = {
    "completeness": 0.60,
    "validity": 0.25,
    "coverage": 0.15,
}

# What one issue of each severity costs the validity term, per affected cell.
# Deliberately asymmetric: a High issue means a figure cannot be trusted, a Low
# one means it is written inconveniently.
SEVERITY_COST: dict[str, float] = {"High": 1.0, "Medium": 0.4, "Low": 0.1}


def field_completeness(frame: MappedFrame) -> list[FieldCompleteness]:
    """Per-field non-null counts, for all seventeen SRS 5.1 fields.

    Counted from `frame.cleaned`'s `empty` status rather than from the typed
    frame, because a cell holding `"N/A"` is a reported absence and a cell
    holding `"1980's"` is a present-but-unreadable value. The typed frame nulls
    both, which would make an unparseable column look like an empty one.
    """
    out: list[FieldCompleteness] = []
    for name in TARGET_FIELD_NAMES:
        cleaned = frame.cleaned.get(name)
        if cleaned is None:
            out.append(
                FieldCompleteness(
                    field=name,
                    source_column=None,
                    mapped=False,
                    rows=frame.n_rows,
                    non_null=0,
                    completeness=0.0,
                    weight=FIELD_WEIGHTS[name],
                )
            )
            continue
        non_null = sum(1 for c in cleaned if c.status != "empty")
        out.append(
            FieldCompleteness(
                field=name,
                source_column=frame.source_column.get(name),
                mapped=True,
                rows=frame.n_rows,
                non_null=non_null,
                completeness=round(non_null / frame.n_rows, 4) if frame.n_rows else 0.0,
                weight=FIELD_WEIGHTS[name],
            )
        )
    return out


def intake_quality_score(
    completeness: list[FieldCompleteness],
    all_findings: list[Finding],
    n_rows: int,
) -> tuple[float, dict[str, float]]:
    """A single 0–1 figure for the state of the file, and its three terms.

    - **completeness** — the weighted share of cells that hold a value.
    - **validity** — how little of the file is under a quality finding, with a
      High finding counting more than a Low one. DQ-01 is excluded here because
      missing values are already the completeness term, and counting them twice
      would make an incomplete file look doubly bad for one reason.
    - **coverage** — the weighted share of the seventeen target fields Agent 2
      managed to map at all. A file with ten mapped fields can be complete and
      valid in those ten and still be a thin schedule.

    Returned with its components so the number is auditable rather than merely
    reportable.
    """
    total_weight = sum(c.weight for c in completeness) or 1.0

    completeness_term = (
        sum(c.completeness * c.weight for c in completeness) / total_weight
    )
    coverage_term = (
        sum(c.weight for c in completeness if c.mapped) / total_weight
    )

    if n_rows:
        weighted_cost = sum(
            SEVERITY_COST.get(f.severity, 0.4)
            * FIELD_WEIGHTS.get(f.field or "", 1.0)
            for f in all_findings
            if f.rule != "DQ-01"
        )
        # Normalised against every cell that could have carried a finding, so a
        # long file is not penalised for being long.
        capacity = n_rows * total_weight
        validity_term = max(0.0, 1.0 - weighted_cost / capacity) if capacity else 1.0
    else:
        validity_term = 1.0

    components = {
        "completeness": round(completeness_term, 4),
        "validity": round(validity_term, 4),
        "coverage": round(coverage_term, 4),
    }
    score = sum(components[k] * w for k, w in SCORE_WEIGHTS.items())
    return round(min(max(score, 0.0), 1.0), 4), components


def value_field_gaps(completeness: list[FieldCompleteness]) -> list[str]:
    """Value fields that are mapped but largely empty.

    Not a rule and not scored — a note for the agent's summary, because a
    Building Value column that is 0% populated (as in SOV_K4T9) is the single
    most consequential thing about a file and is easy to miss in a list of
    seventeen percentages.
    """
    return [
        c.field
        for c in completeness
        if c.field in VALUE_FIELDS and c.mapped and c.completeness == 0.0
    ]


__all__ = [
    "FIELD_WEIGHTS",
    "SCORE_WEIGHTS",
    "SEVERITY_COST",
    "field_completeness",
    "intake_quality_score",
    "value_field_gaps",
]
