"""Findings to recommendations: FR-DQ-08 and FR-DQ-09.

FR-DQ-08 asks for "one recommendation per issue group", so the grouping rule is
the design decision here. Findings are grouped by **(rule, field, op)**: one
recommendation per thing-that-is-wrong per place-it-is-wrong per
fix-that-would-work. That is what makes 623 sprinkler findings on one file into
three reviewable decisions — map "No" to N, map "Yes" to Y, and look at the
twenty-eight cells nobody can code automatically — rather than 623 identical
questions or one question that hides the distinction.

FR-DQ-09 requires the before/after examples to be computed by the engine from
real rows. They are: every `after` in this module was produced by `frame.py`'s
cleaner or by a rule's own arithmetic at detection time, carried on the finding,
and copied here unchanged. Nothing is formatted into existence at this stage, and
the examples prefer *distinct* value pairs so a reviewer sees the variety of a
column rather than the same row twenty times.

**Confidence is about the fix, not about the finding.** A `zip5` on "2110" in
Massachusetts is certain — the zip is five digits with a lost leading zero and
there is exactly one way to restore it. A duplicate address is a real
observation with no certain remedy, so it scores low and must state why, which
SRS 5.4 enforces below 0.70. The numbers in `OP_CONFIDENCE` are defaults; per the
brief they are reasonable rather than tuned, and tuning them is a change to this
one table.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from backend.agents.quality.findings import Finding
from backend.agents.quality.models import (
    ACTION_TYPE_FOR_OP,
    MAX_BEFORE_AFTER_EXAMPLES,
    RULE_CATALOGUE,
    UNCERTAINTY_REQUIRED_BELOW,
    BeforeAfter,
    Recommendation,
)

# How much the engine trusts its own proposed fix, by op. Mechanical conversions
# score high because the output follows from the input with no judgement;
# `exclude_row` scores lower because dropping a row is destructive and the
# arithmetic that identified it, though strong, is still inference.
OP_CONFIDENCE: dict[str, float] = {
    "zip5": 0.95,
    "state_to_abbrev": 0.93,
    "strip_currency_to_float": 0.95,
    "to_float": 0.90,
    "to_int": 0.85,
    "map_values": 0.88,
    "trim_normalise": 0.90,
    "set_null": 0.85,
    "exclude_row": 0.78,
    "rename": 0.80,
    "split_column": 0.70,
    "merge_columns": 0.70,
}

# A finding with no op is an observation a human has to act on. These sit below
# the SRS 5.4 threshold on purpose: the engine is confident the cell is wrong and
# explicitly not confident about what to do, and the uncertainty text is where it
# says which of the two it means.
FLAG_CONFIDENCE: dict[str, float] = {
    "DQ-01": 0.60,
    "DQ-02": 0.65,
    "DQ-03": 0.65,
    "DQ-04": 0.62,
    "DQ-05": 0.65,
    "DQ-06": 0.62,
    "DQ-07": 0.60,
    "DQ-08": 0.55,
    "DQ-09": 0.55,
    "DQ-11": 0.60,
    "DQ-13": 0.50,
    "DQ-14": 0.55,
    "DQ-16": 0.60,
    "DQ-17": 0.58,
    "DQ-18": 0.52,
}
DEFAULT_FLAG_CONFIDENCE = 0.55

# What the engine cannot settle, per rule, stated for the SRS 5.4 uncertainty
# field. Written as the reviewer's question rather than as an apology.
UNCERTAINTY_TEXT: dict[str, str] = {
    "DQ-01": (
        "The engine cannot supply a missing value and will not invent one (C-02). "
        "Whether these cells can be filled from another source is a question for "
        "the broker who produced the schedule"
    ),
    "DQ-02": (
        "The cell holds text that is not a number. Whether it is a note that "
        "belongs in a comments column, a value in the wrong row, or a genuine "
        "absence cannot be read from the cell"
    ),
    "DQ-03": (
        "No operation in SRS 9.1 may round a non-whole number to fit an integer "
        "field, so the intended whole value has to be confirmed rather than derived"
    ),
    "DQ-04": (
        "A negative insured value may be a credit, a sign error, or an accounting "
        "bracket that was meant to be positive. The engine will not flip the sign, "
        "because the three readings give three different premiums"
    ),
    "DQ-05": (
        "The year is not a possible construction date, but the intended year "
        "cannot be derived from it: 173 could be 1730 or 1973 and the cell does "
        "not say which"
    ),
    "DQ-06": (
        "A count below one is wrong but its correct value is unknown; the engine "
        "will not substitute 1 for a figure the source never gave"
    ),
    "DQ-07": (
        "The zip and the state contradict each other. Either could be the wrong "
        "one, and correcting the wrong half would move the risk to another state"
    ),
    "DQ-08": (
        "No five-digit code can be read out of this cell, so there is nothing to "
        "normalise it to"
    ),
    "DQ-09": (
        "The value is neither a USPS code nor a state name in the engine's "
        "lookup, so no abbreviation can be derived from it"
    ),
    "DQ-11": (
        "SRS 9.2 DQ-11 requires a human to code ambiguous sprinkler protection. "
        "Partial coverage sits between Y and N and the choice changes the rate, so "
        "the engine proposes neither"
    ),
    "DQ-13": (
        "A repeated address is sometimes a double-entered row and sometimes "
        "several buildings on one site. Excluding the wrong one removes real "
        "exposure, so the engine identifies the group and leaves the decision open"
    ),
    "DQ-14": (
        "Either the construction class or the storey count is wrong, and the row "
        "does not indicate which. Both are underwriting inputs"
    ),
    "DQ-16": (
        "FR-DQ-06 restricts this rule to validation: the engine reports that the "
        "components and the stated total disagree and fills neither, because it "
        "cannot know which of the two is authoritative"
    ),
    "DQ-17": (
        "A decade such as \"1920's\" names no single year, so there is no value to "
        "convert it to without choosing one"
    ),
    "DQ-18": (
        "The ratio shows one of the two figures is probably wrong, but which one, "
        "and by what factor, is not derivable from the row"
    ),
}


def _group_key(finding: Finding) -> tuple[str, Optional[str], Optional[str]]:
    return (finding.rule, finding.field, finding.op)


def _examples(group: list[Finding]) -> list[BeforeAfter]:
    """Up to `MAX_BEFORE_AFTER_EXAMPLES` worked examples, variety first.

    Distinct `(before, after)` pairs come first so a reviewer sees that a column
    holds "Yes", "No" and "1" rather than twenty rows of "No". The remaining
    slots are filled in row order, which shows the scale of the change.
    """
    picked: list[Finding] = []
    seen: set[tuple[str, Optional[str]]] = set()

    for finding in group:
        key = (finding.before, finding.after)
        if key in seen:
            continue
        seen.add(key)
        picked.append(finding)
        if len(picked) >= MAX_BEFORE_AFTER_EXAMPLES:
            break

    if len(picked) < MAX_BEFORE_AFTER_EXAMPLES:
        chosen = {id(f) for f in picked}
        for finding in group:
            if id(finding) in chosen:
                continue
            picked.append(finding)
            if len(picked) >= MAX_BEFORE_AFTER_EXAMPLES:
                break

    picked.sort(key=lambda f: f.row)
    return [
        BeforeAfter(row=f.row, before=f.before, after=f.after) for f in picked
    ]


def _rationale(group: list[Finding], op: Optional[str], field: Optional[str]) -> str:
    """Plain-English prose for the group, built from the findings themselves.

    The first finding's own `detail` carries the specific explanation a rule
    wrote at detection time, so it is quoted rather than paraphrased; the count
    and the row range are added because a reviewer's first question about any
    finding is how much of the file it touches.
    """
    spec = RULE_CATALOGUE[group[0].rule]
    rows = sorted({f.row for f in group})
    where = (
        f"row {rows[0]}"
        if len(rows) == 1
        else f"{len(rows)} rows ({rows[0]}–{rows[-1]})"
    )
    subject = f"{field} on {where}" if field else where

    lead = f"{spec.rule} ({spec.check}): {subject}. {group[0].detail}."
    if len(rows) > 1:
        distinct = len({f.before for f in group})
        lead += (
            f" The same problem appears on {len(rows)} rows"
            + (f", across {distinct} distinct values" if distinct > 1 else
               f", all reading '{group[0].before}'")
            + "."
        )

    if op:
        lead += (
            f" Applying {op} would produce the values shown, which the engine "
            "computed from these rows; nothing is changed until this is approved."
        )
    else:
        lead += (
            " No operation in the SRS 9.1 whitelist can fix this safely, so it is "
            "raised for review rather than corrected."
        )
    return lead


def build_recommendations(all_findings: list[Finding]) -> list[Recommendation]:
    """One `Recommendation` per (rule, field, op) group, in review-queue order.

    Severity decides the order, because SRS 9.2 says severity "sets the order of
    the review queue; it never triggers an automatic change" — so it is used for
    exactly that and for nothing else. Ids are assigned after sorting, so R-001
    is the first thing a reviewer should look at.
    """
    groups: dict[tuple[str, Optional[str], Optional[str]], list[Finding]] = defaultdict(list)
    for finding in all_findings:
        groups[_group_key(finding)].append(finding)

    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            severity_order.get(item[1][0].severity, 3),
            item[0][0],                      # rule id
            item[0][1] or "",                # field
            -len(item[1]),                   # biggest group first
        ),
    )

    out: list[Recommendation] = []
    for position, ((rule, field, op), group) in enumerate(ordered, start=1):
        group.sort(key=lambda f: f.row)
        rows = sorted({f.row for f in group})

        if op:
            confidence = OP_CONFIDENCE.get(op, 0.75)
            examples = _examples(group)
        else:
            confidence = FLAG_CONFIDENCE.get(rule, DEFAULT_FLAG_CONFIDENCE)
            # SRS 5.4: a flag_for_review proposes no change, so it carries no
            # before/after. The cell values still reach the reviewer through the
            # rationale, which quotes them.
            examples = []

        # A duplicate group whose rows carry their own location references is
        # probably a campus rather than a double entry, so the engine says it is
        # less sure rather than reporting both situations identically.
        if rule == "DQ-13" and group[0].evidence.get("distinct_references"):
            confidence = min(confidence, 0.40)

        uncertainty = (
            UNCERTAINTY_TEXT.get(rule) if confidence < UNCERTAINTY_REQUIRED_BELOW else None
        )
        if confidence < UNCERTAINTY_REQUIRED_BELOW and not uncertainty:
            uncertainty = (
                "The engine is not confident enough in this correction to propose "
                "it without review"
            )

        out.append(
            Recommendation(
                id=f"R-{position:03d}",
                action_type=ACTION_TYPE_FOR_OP[op] if op else "flag_for_review",
                op=op,  # type: ignore[arg-type]
                rule=rule,
                field=field,
                source_column=group[0].source_column,
                rows=rows,
                before_after=examples,
                rationale=_rationale(group, op, field),
                confidence=round(confidence, 2),
                uncertainty=uncertainty,
                severity=group[0].severity,
            )
        )
    return out


__all__ = [
    "FLAG_CONFIDENCE",
    "OP_CONFIDENCE",
    "UNCERTAINTY_TEXT",
    "build_recommendations",
]
