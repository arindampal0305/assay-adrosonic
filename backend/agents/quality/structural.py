"""The two rules about the shape of the sheet rather than the content of a cell.

**DQ-15, totals rows.** A label test alone is unusable, and that is a measured
claim rather than a cautious one: searching the real corpus for the words
"total", "subtotal", "sum" and "grand" matched "Rio Grande Blvd", "Grand
Prairie", "Grand Place - Exhibition Hall", "Samuel Grand" and "828 S. Carrier
Pkwy, Suite 100 (Grand Prairie)" — every one a genuine location. So the rule is
arithmetic first: a row is a totals row when its own figures equal the sum of
every other row's, in at least two value columns independently. A label is
corroboration, never grounds on its own.

The row is **flagged and never dropped**. FR-SHT-05 and the brief both require
that, and `exclude_row` reaches the output only if a reviewer approves it.

**DQ-16, TIV reconciliation.** Validation only, per FR-DQ-06 and C-02: when the
components disagree with a source total, Agent 3 says so and fills nothing. The
TIV column is found among the headers Agent 2 left unmapped, because a total has
no target field of its own in SRS 5.1 — which is exactly why `MappedFrame` keeps
the unmapped columns instead of discarding them.

A caveat worth stating plainly: in all four real sample files the TIV column is
an Excel `=SUM(...)` formula whose cached result is absent from the saved
workbook, so every candidate reads as entirely empty and DQ-16 correctly finds
nothing to reconcile. The rule is therefore exercised by a constructed frame in
the tests, and that construction is labelled as such.
"""

from __future__ import annotations

from typing import Any, Optional

from backend.agents.quality import findings
from backend.agents.quality.findings import Finding
from backend.agents.quality.frame import MappedFrame, clean_amount
from backend.agents.quality.models import VALUE_FIELDS
from backend.agents.quality.vocab import TIV_HEADER_HINTS, TIV_TOLERANCE

# A row must reconcile in at least this many value columns to be called a totals
# row on arithmetic alone. One column reconciling is a coincidence: on SOV_B4ID a
# genuine location row matched in its single populated value column, which is the
# false positive this threshold exists to exclude.
MIN_RECONCILING_COLUMNS = 2
# With a corroborating label in the row, one reconciling column is enough.
MIN_RECONCILING_COLUMNS_WITH_LABEL = 1
TOTAL_WORDS: tuple[str, ...] = ("total", "subtotal", "sub-total", "grand total", "sum of")
# Relative tolerance when comparing a row against the sum of the others. Floats
# that came from currency strings do not add up exactly.
SUM_TOLERANCE = 0.01


def _labelled_total(frame: MappedFrame, index: int) -> Optional[str]:
    """The cell in this row that calls it a total, if any.

    Deliberately narrow. "Grand Prairie" is a city and "Rio Grande Blvd" is a
    street, so a bare "grand" does not count; the phrase has to be a totalling
    phrase in its own right, and it has to be most of what the cell says. A cell
    that merely *contains* "total" inside an address is not a label.
    """
    for field_name in frame.mapped_fields:
        raw = frame.cell(field_name, index)
        if not isinstance(raw, str):
            continue
        text = raw.strip().lower()
        if not text or len(text) > 32:
            continue
        for word in TOTAL_WORDS:
            if word in text:
                return raw.strip()
    return None


def _column_sums(frame: MappedFrame) -> dict[str, list[Optional[float]]]:
    """Cleaned numbers per value field, so the arithmetic runs once."""
    out: dict[str, list[Optional[float]]] = {}
    for field_name in VALUE_FIELDS:
        if field_name not in frame.cleaned:
            continue
        out[field_name] = [c.value for c in frame.cleaned[field_name]]
    return out


def dq15_totals_rows(frame: MappedFrame) -> list[Finding]:
    """A totals or subtotal row sitting inside the data (SRS 9.2 DQ-15).

    Carries `exclude_row` with an explicit `after=None`, meaning the row leaves
    the cleaned output. Nothing is excluded here: the recommendation is pending
    until a reviewer approves it, and `QualityBlock.totals_rows` reports the rows
    so a reader can see what was flagged without reading the recommendations.
    """
    columns = _column_sums(frame)
    if not columns or frame.n_rows < 3:
        # Two rows cannot distinguish a total from its single component.
        return []

    totals = {
        name: sum(v for v in values if v is not None)
        for name, values in columns.items()
    }

    out: list[Finding] = []
    for index in range(frame.n_rows):
        label = _labelled_total(frame, index)
        reconciling: list[str] = []
        checked: list[str] = []

        for name, values in columns.items():
            here = values[index]
            if here is None or here == 0:
                continue
            others = totals[name] - here
            if others <= 0:
                continue
            checked.append(name)
            if abs(here - others) <= max(1.0, abs(others) * SUM_TOLERANCE):
                reconciling.append(name)

        needed = (
            MIN_RECONCILING_COLUMNS_WITH_LABEL if label else MIN_RECONCILING_COLUMNS
        )
        if not checked or len(reconciling) < needed or len(reconciling) < len(checked):
            continue

        sheet_row = frame.row_label(index)
        shown = ", ".join(
            f"{name} {columns[name][index]:,.0f}" for name in reconciling
        )
        out.append(
            findings.make(
                "DQ-15",
                sheet_row,
                None,
                detail=(
                    f"Row {sheet_row} is a totals row, not a location: its figures "
                    f"equal the sum of every other row in "
                    f"{len(reconciling)} value column(s) ({shown})"
                    + (f", and it is labelled '{label}'" if label else "")
                    + ". Left in the data it would double the schedule's value"
                ),
                before=label or f"row {sheet_row}",
                after=None,
                op="exclude_row",
                source_column=None,
                evidence={
                    "sheet_row": sheet_row,
                    "reconciling_columns": reconciling,
                    "checked_columns": checked,
                    "label": label,
                },
            )
        )
    return out


# --------------------------------------------------------------------------
# DQ-16 — TIV reconciliation
# --------------------------------------------------------------------------
def find_tiv_column(frame: MappedFrame) -> Optional[tuple[str, list[Any]]]:
    """The unmapped source column that reads like a total insured value.

    Only unmapped headers are considered: a column Agent 2 resolved to a target
    field is a component, and reconciling the components against one of
    themselves would always agree. The column must also contain at least one
    number, which is what rules out the `=SUM()` columns whose cached values the
    workbook does not carry.
    """
    best: Optional[tuple[int, str, list[Any]]] = None
    for header, values in frame.unmapped.items():
        lowered = header.strip().lower()
        rank = next(
            (i for i, hint in enumerate(TIV_HEADER_HINTS) if hint in lowered), None
        )
        if rank is None:
            continue
        if not any(clean_amount(v).usable for v in values):
            continue
        if best is None or rank < best[0]:
            best = (rank, header, values)
    return (best[1], best[2]) if best else None


def dq16_tiv_reconciliation(
    frame: MappedFrame, totals_rows: Optional[set[int]] = None
) -> list[Finding]:
    """Components against a source TIV column (SRS 9.2 DQ-16, FR-DQ-06).

    No op, and that is the whole point of the rule: FR-DQ-06 says "Mismatch
    flagged; no value is filled", so a disagreement never results in the engine
    writing the total into the components or the components into the total. The
    finding names both figures and the difference, and stops.

    Rows DQ-15 identified as totals are skipped — a totals row's own TIV
    legitimately differs from its own components' arithmetic, and flagging it
    would be reporting the consequence of a problem already reported.
    """
    found = find_tiv_column(frame)
    if found is None:
        return []
    header, values = found

    components = [f for f in VALUE_FIELDS if f in frame.cleaned]
    if not components:
        return []

    skip = totals_rows or set()
    out: list[Finding] = []
    for index in range(min(frame.n_rows, len(values))):
        sheet_row = frame.row_label(index)
        if sheet_row in skip:
            continue
        stated = clean_amount(values[index])
        if not stated.usable or stated.value is None or stated.value == 0:
            continue

        parts = {
            name: frame.cleaned[name][index].value
            for name in components
            if frame.cleaned[name][index].value is not None
        }
        if not parts:
            continue
        computed = sum(parts.values())
        if computed == 0:
            continue

        difference = computed - stated.value
        if abs(difference) <= abs(stated.value) * TIV_TOLERANCE:
            continue

        breakdown = " + ".join(f"{k} {v:,.0f}" for k, v in parts.items())
        out.append(
            findings.make(
                "DQ-16",
                sheet_row,
                None,
                detail=(
                    f"The components do not add up to the '{header}' column: "
                    f"{breakdown} = {computed:,.0f}, but the row states "
                    f"{stated.value:,.0f}, a difference of {difference:,.0f} "
                    f"({abs(difference) / abs(stated.value):.1%}). Reported for "
                    "checking only; no value is filled in either direction"
                ),
                before=findings.render(values[index]),
                op=None,
                source_column=header,
                evidence={
                    "tiv_column": header,
                    "stated": stated.value,
                    "computed": computed,
                    "difference": difference,
                    "components": {k: v for k, v in parts.items()},
                },
            )
        )
    return out


def run_structural_rules(frame: MappedFrame) -> tuple[list[Finding], dict[str, str]]:
    """DQ-15 and DQ-16, plus a reason for either that could not run."""
    skipped: dict[str, str] = {}

    totals = dq15_totals_rows(frame)
    if frame.n_rows < 3:
        skipped["DQ-15"] = (
            f"{frame.n_rows} data row(s); a totals row cannot be told from its "
            "components without at least three"
        )
    if not any(f in frame.cleaned for f in VALUE_FIELDS):
        skipped["DQ-15"] = "no value field was mapped, so no row arithmetic is possible"

    tiv_rows = {f.row for f in totals}
    reconciliation = dq16_tiv_reconciliation(frame, totals_rows=tiv_rows)
    if find_tiv_column(frame) is None:
        skipped["DQ-16"] = (
            "no unmapped source column holds a usable total insured value; in these "
            "workbooks the total columns are =SUM() formulas with no cached result"
        )

    out = totals + reconciliation
    return [f for f in out if f.rule not in skipped], skipped


__all__ = [
    "MIN_RECONCILING_COLUMNS",
    "TOTAL_WORDS",
    "dq15_totals_rows",
    "dq16_tiv_reconciliation",
    "find_tiv_column",
    "run_structural_rules",
]
