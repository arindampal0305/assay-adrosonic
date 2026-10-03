"""The declarative half of validation: SRS 5.1 types and ranges, via pandera.

FR-DQ-02 and FR-DQ-03 ask for type and range validation against the data
dictionary, and that is exactly the kind of statement a schema language says
better than hand-written loops. So DQ-04, DQ-05, DQ-06 and DQ-11 are declared
here as pandera `Check`s and read back out of `SchemaErrors.failure_cases`.

**Why DQ-02 and DQ-03 are not pandera checks.** They could be, via `coerce=True`.
They are not, because a coercion failure in pandera reports that a *column* could
not become `Float64` — it does not report which cell, nor what the cell said, nor
whether the value was merely formatted or outright prose. A reviewer fixing a
spreadsheet needs all three. So typing happens in `frame.py`, cell by cell, and
this module reads its per-cell verdict. Both halves are reported from here so
that "schema validation" means one thing to the caller.

**What the `error=` argument is doing.** Pandera renders a check's `error` string
into the `check` column of `failure_cases`, and ignores `name=` there. Putting the
rule id in `error=` is therefore the only way to get the failing rule back out
without matching on a generated description like `greater_than_or_equal_to(0)`,
which would silently stop matching if a bound ever changed.

Nothing here proposes a correction it cannot compute. A negative building value
does not become its absolute value, and a year of 3015 does not become 2015: no
op in SRS 9.1 flips a sign or edits a digit, and inventing one would breach C-02.
Those findings carry no op and go to a human.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd
import pandera.pandas as pa

from backend.agents.quality import findings
from backend.agents.quality.findings import Finding
from backend.agents.quality.frame import MappedFrame
from backend.agents.quality.models import (
    INTEGER_FIELDS,
    SPRINKLER_ENUM,
    SPRINKLER_FIELD,
    VALUE_FIELDS,
)
from backend.agents.quality.vocab import SPRINKLER_AMBIGUOUS, SPRINKLER_SYNONYMS
from backend.state.target_schema import TARGET_FIELDS

# SRS 9.2 DQ-05: "Year Built after current year or before 1700".
EARLIEST_PLAUSIBLE_YEAR = 1700

# The rules this module owns, declared so the agent can report them as evaluated
# even when they find nothing (a rule that ran clean and a rule that never ran are
# different facts).
SCHEMA_RULES: tuple[str, ...] = ("DQ-02", "DQ-03", "DQ-04", "DQ-05", "DQ-06", "DQ-11")

_PANDERA_DTYPE = {"string": object, "integer": "Int64", "float": "Float64"}


def build_schema(current_year: Optional[int] = None) -> pa.DataFrameSchema:
    """The SRS 5.1 data dictionary as a pandera schema.

    Every column is `required=False` and the schema is non-strict, because the
    frame holds only the fields Agent 2 actually mapped and may also carry none of
    a given one. A missing column is Agent 2's business and DQ-01's; it is not a
    schema violation.

    Every column is `nullable=True`. A null here is a cell that was empty or that
    `frame.py` could not type, and both already have their own rule — DQ-01 and
    DQ-02/DQ-03. Letting pandera also complain would report one problem twice.
    """
    year = current_year or datetime.now().year
    columns: dict[str, pa.Column] = {}

    for target in TARGET_FIELDS:
        checks: list[pa.Check] = []
        if target.name in VALUE_FIELDS:
            # DQ-04, negative amount. `clean_amount` deliberately returns
            # negatives rather than rejecting them so that this is the one place
            # that decides what a negative means.
            checks.append(pa.Check.ge(0, error="DQ-04"))
        if target.name == "Year Built":
            checks.append(
                pa.Check.in_range(EARLIEST_PLAUSIBLE_YEAR, year, error="DQ-05")
            )
        if target.name in ("Storeys", "Number of Buildings"):
            checks.append(pa.Check.ge(1, error="DQ-06"))
        if target.name == SPRINKLER_FIELD:
            checks.append(pa.Check.isin(list(SPRINKLER_ENUM), error="DQ-11"))

        columns[target.name] = pa.Column(
            _PANDERA_DTYPE[target.dtype],
            checks=checks or None,
            nullable=True,
            required=False,
        )

    return pa.DataFrameSchema(columns, strict=False, coerce=False, name="SRS 5.1")


# --------------------------------------------------------------------------
# Translating pandera's verdict into findings.
# --------------------------------------------------------------------------
def _sprinkler_percentage(text: str) -> Optional[tuple[Optional[str], Optional[str], str]]:
    """A numeric sprinkler cell, read as the coverage percentage it is.

    Every sprinkler column in the real corpus is a *percentage* column — the
    headers are `% Sprinklered`, `Sprinkler %` and `%Sprink` — and they hold
    `0`, `1`, `0.5`, `0.07`, `0.01`. So the numbers are not codes to look up in a
    synonym table, they are coverage fractions, and reading them as such is the
    difference between "0.5 matches no known spelling" and "half the building is
    sprinklered, which the SRS says a human must code".

    Full and zero coverage are unambiguous and get an op. Anything in between is
    partial coverage, and DQ-11 sends partial coverage to a human whatever
    notation it arrived in.
    """
    try:
        number = float(text.rstrip("%").replace(",", "").strip())
    except ValueError:
        return None
    # A trailing % means the author wrote 50 for a half, not 0.5.
    scale = 100.0 if text.strip().endswith("%") or number > 1.0 else 1.0
    share = number / scale

    if share == 0:
        return "map_values", "N", (
            f"a coverage figure of {text} is no sprinkler protection"
        )
    if share == 1:
        return "map_values", "Y", (
            f"a coverage figure of {text} is full sprinkler protection"
        )
    if 0 < share < 1:
        hint = (
            f" A bare {text} could also be a reference to an NFPA standard rather "
            "than a percentage, which is a further reason not to decide it here."
            if float(number).is_integer()
            else ""
        )
        return None, None, (
            f"a coverage figure of {text} means roughly {share:.0%} of the "
            "building is sprinklered. SRS 9.2 DQ-11 requires a human to choose "
            "between Y, Y13, Y(13R) and N for partial coverage; mapping it either "
            f"way would state a protection class the source does not give.{hint}"
        )
    return None, None, (
        f"a coverage figure of {text} is not a share between 0 and 100%"
    )


def _sprinkler_fix(raw: Any) -> tuple[Optional[str], Optional[str], str]:
    """A sprinkler cell to (op, after, why).

    Ambiguity is decided before synonymy on purpose. "Partial" has an obvious
    *intent* and no defensible *code*: mapping it to Y overstates the protection
    and mapping it to N understates it, and either way the engine would have
    chosen a rating input the source never gave. SRS 9.2 settles it — those terms
    always go to a human — so this returns no op for them even though a model
    would happily supply one.
    """
    text = findings.render(raw).strip()
    key = text.lower().rstrip(".")
    if key in SPRINKLER_AMBIGUOUS:
        return None, None, (
            f"'{text}' describes partial or undetermined protection. SRS 9.2 DQ-11 "
            "requires a human to choose the code; no mapping is proposed"
        )

    numeric = _sprinkler_percentage(text)
    if numeric is not None:
        return numeric

    canonical = SPRINKLER_SYNONYMS.get(key)
    if canonical:
        return "map_values", canonical, (
            f"'{text}' is an unambiguous spelling of '{canonical}'"
        )
    return None, None, (
        f"'{text}' is outside the permitted set {', '.join(SPRINKLER_ENUM)} and "
        "matches no known spelling of it"
    )


def _describe(rule: str, field_name: str, shown: str) -> tuple[str, Optional[str], Optional[str]]:
    """(detail, op, after) for the rules whose correction code cannot compute."""
    if rule == "DQ-04":
        return (
            f"{field_name} is negative ({shown}). A negative insured value is "
            "either a credit, a sign error, or a bracketed figure that was meant "
            "to be positive; the source does not say which",
            None,
            None,
        )
    if rule == "DQ-05":
        return (
            f"Year Built {shown} is outside {EARLIEST_PLAUSIBLE_YEAR}–"
            f"{datetime.now().year}, so it cannot be a construction year as written",
            None,
            None,
        )
    if rule == "DQ-06":
        why = (
            "a building has at least one storey"
            if field_name == "Storeys"
            else "a location with no buildings on it is not a location"
        )
        return f"{field_name} is {shown}, and {why}", None, None
    return f"{field_name} failed {rule} with value {shown}", None, None


def _run_pandera(frame: MappedFrame) -> tuple[list[Finding], bool]:
    """Validate `frame.typed` and translate the failures. Returns (findings, ran)."""
    if frame.typed.empty or not len(frame.typed.columns):
        return [], False

    schema = build_schema()
    try:
        schema.validate(frame.typed, lazy=True)
        return [], True
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
    except Exception:  # pragma: no cover - a schema-level fault, not a data one
        return [], False

    out: list[Finding] = []
    for _, case in cases.iterrows():
        rule = str(case["check"])
        field_name = case["column"]
        # A schema-context failure (a dtype, say) has no cell behind it and no row
        # a reviewer could open. It is a bug in `_build_typed`, not a data issue.
        if rule not in SCHEMA_RULES or not isinstance(field_name, str):
            continue
        try:
            sheet_row = int(case["index"])
        except (TypeError, ValueError):
            continue

        raw = frame.cell(field_name, frame.index_of_row(sheet_row))
        shown = findings.render(raw)

        if rule == "DQ-11":
            op, after, detail = _sprinkler_fix(raw)
        else:
            detail, op, after = _describe(rule, field_name, shown)

        out.append(
            findings.make(
                rule,
                sheet_row,
                field_name,
                detail=detail,
                before=raw,
                after=after,
                op=op,
                source_column=frame.source_column.get(field_name),
                evidence={
                    "value": shown,
                    "typed_value": _jsonable(case["failure_case"]),
                    "check": rule,
                },
            )
        )
    return out, True


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if pd.isna(value):
        return None
    return str(value)


# --------------------------------------------------------------------------
# The per-cell typing verdict from `frame.py` (DQ-02, DQ-03).
# --------------------------------------------------------------------------
def _typing_findings(frame: MappedFrame) -> list[Finding]:
    """DQ-02 and DQ-03, read from the cleaner's per-cell report.

    The two statuses are kept apart because they lead different places. An
    `unparseable` cell holds something that is not a number at all, so no op in
    SRS 9.1 applies. A `fractional` cell holds a real number that is not whole —
    and SRS 9.1 is explicit that `to_int`'s "rounding rule" is to *reject*
    non-whole input, so proposing it would propose an op that must refuse. Both
    go to a human; only the explanation differs, and the explanation is the part
    the reviewer acts on.
    """
    out: list[Finding] = []

    for field_name in VALUE_FIELDS:
        for index, result in enumerate(frame.cleaned.get(field_name, [])):
            if result.status != "unparseable":
                continue
            raw = frame.raw[field_name][index]
            out.append(
                findings.make(
                    "DQ-02",
                    frame.row_label(index),
                    field_name,
                    detail=(
                        f"{field_name} reads '{findings.render(raw)}', which is not a "
                        "number even after currency symbols and separators are removed"
                    ),
                    before=raw,
                    op=None,
                    source_column=frame.source_column.get(field_name),
                    evidence={"value": findings.render(raw), "status": result.status},
                )
            )

    for field_name in INTEGER_FIELDS:
        for index, result in enumerate(frame.cleaned.get(field_name, [])):
            if result.status not in ("unparseable", "fractional"):
                continue
            raw = frame.raw[field_name][index]
            shown = findings.render(raw)
            if result.status == "fractional":
                detail = (
                    f"{field_name} reads '{shown}', which is numeric but not a whole "
                    f"number. SRS 9.1 requires to_int to reject non-whole input, so "
                    "the correct value has to be confirmed rather than rounded"
                )
            else:
                detail = (
                    f"{field_name} reads '{shown}', which is not a whole number and "
                    "cannot be read as a number at all"
                )
            out.append(
                findings.make(
                    "DQ-03",
                    frame.row_label(index),
                    field_name,
                    detail=detail,
                    before=raw,
                    op=None,
                    source_column=frame.source_column.get(field_name),
                    evidence={"value": shown, "status": result.status},
                )
            )

    return out


def validate(frame: MappedFrame) -> tuple[list[Finding], bool]:
    """Every schema-level finding, and whether pandera actually ran.

    The flag is returned rather than inferred from an empty list, for the same
    reason `rules_not_applicable` exists: a clean sheet and a skipped validator
    must not look identical in the report.
    """
    pandera_found, ran = _run_pandera(frame)
    return _typing_findings(frame) + pandera_found, ran


__all__ = [
    "EARLIEST_PLAUSIBLE_YEAR",
    "SCHEMA_RULES",
    "build_schema",
    "validate",
]
