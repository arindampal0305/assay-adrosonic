"""The SRS 9.2 domain rules: DQ-01, DQ-07 to DQ-14, DQ-17 and DQ-18.

One function per rule, each taking the frame and returning findings, each
importable and testable on its own. They share no mutable state and run in any
order, so a rule can be read in isolation and a failing one can be isolated
without running the rest.

**Raw text, not the typed frame.** Several of these rules are *about* notation, so
they read `frame.raw`. A zip of `"00802"` becomes the integer `802` in the typed
frame, and a rule that checked the integer would report a three-digit zip and
propose a correction to a cell that was already correct. The typed frame is for
range arithmetic; the raw column is for anything that cares how the cell is
written.

**What code may compute, and what it may not.** `state_to_abbrev` on "V.I.",
`zip5` on a lost leading zero, `strip_currency_to_float` on "$2,100,000.00" and
`set_null` on a placeholder are all mechanical: the output follows from the input
with no guessing, so these rules carry an op and a worked after-value. A
duplicate address, a wood-frame tower, a Contents figure forty times Building
Value — nobody can compute the fix from the cell alone, so those carry no op and
go to a reviewer. That split is C-02 in practice, not an afterthought.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Optional

from backend.agents.quality import findings
from backend.agents.quality.findings import Finding
from backend.agents.quality.frame import MappedFrame, is_null
from backend.agents.quality.models import VALUE_FIELDS
from backend.agents.quality.vocab import (
    COMPONENT_RATIO_LIMIT,
    FRAME_EXCLUSIONS,
    FRAME_CONSTRUCTION_TERMS,
    FULL_STATE_NAMES,
    LEADING_ZERO_STATES,
    MAX_PLAUSIBLE_FRAME_STOREYS,
    PLACEHOLDER_REPEAT_MIN,
    PLACEHOLDER_STOREYS,
    PLACEHOLDER_VALUES,
    PLACEHOLDER_YEARS,
    STATE_ABBREVS,
    STATE_ZIP_PREFIXES,
)

# A zip as a reviewer would recognise one, and the Zip+4 form that DQ-08 unpicks.
_ZIP_PLUS_FOUR = re.compile(r"^(\d{5})[-\s]?(\d{4})$")
_DIGITS_ONLY = re.compile(r"^\d+$")
# Date notations DQ-17 looks for in a column that should hold a bare year.
_DATE_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bare year", re.compile(r"^\s*(1[6-9]\d{2}|20\d{2})\s*$")),
    ("ISO date", re.compile(r"^\s*\d{4}-\d{1,2}-\d{1,2}")),
    ("slash date", re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$")),
    ("dotted date", re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{2,4}\s*$")),
    ("decade", re.compile(r"^\s*\d{4}\s*'?s\s*$", re.IGNORECASE)),
    ("month name", re.compile(r"[A-Za-z]{3,9}\s+\d{4}")),
)


def _text_cells(frame: MappedFrame, field_name: str) -> list[tuple[int, int, str]]:
    """(index, sheet_row, text) for every non-null cell of a field."""
    out = []
    for index, raw in enumerate(frame.values(field_name)):
        if is_null(raw):
            continue
        text = str(raw).strip()
        if text:
            out.append((index, frame.row_label(index), text))
    return out


# --------------------------------------------------------------------------
# DQ-01 — missing values
# --------------------------------------------------------------------------
def dq01_missing_values(frame: MappedFrame) -> list[Finding]:
    """A mapped field with an empty cell.

    Reported per cell so the recommendation can name the rows, but no op is
    proposed: the one thing code must never do about a missing value is supply
    one (C-02). An unmapped field is not reported here at all — that is Agent 2's
    finding, and `completeness.py` records it as 0% rather than as N empty cells.
    """
    out: list[Finding] = []
    for field_name in frame.mapped_fields:
        for index, result in enumerate(frame.cleaned[field_name]):
            if result.status != "empty":
                continue
            out.append(
                findings.make(
                    "DQ-01",
                    frame.row_label(index),
                    field_name,
                    detail=f"{field_name} is empty",
                    before=frame.raw[field_name][index],
                    op=None,
                    source_column=frame.source_column.get(field_name),
                    evidence={"field": field_name},
                )
            )
    return out


# --------------------------------------------------------------------------
# DQ-07, DQ-08, DQ-09 — place
# --------------------------------------------------------------------------
def _state_code(frame: MappedFrame, index: int) -> Optional[str]:
    """The normalised two-letter code for a row's state, if one can be read."""
    raw = frame.cell("State", index)
    if is_null(raw):
        return None
    text = str(raw).strip()
    upper = text.upper()
    if upper in STATE_ABBREVS:
        return upper
    return FULL_STATE_NAMES.get(text.lower().strip())


def _prefix_fits(state: str, zip_text: str) -> Optional[bool]:
    """Does a 5-digit zip's 3-digit prefix belong to this state?

    None when there is nothing to compare — an unknown state code, or a zip that
    is not five digits. A rule that treats "I cannot tell" as "mismatch" invents
    findings, so the caller is made to handle the third case.
    """
    ranges = STATE_ZIP_PREFIXES.get(state)
    if not ranges or len(zip_text) != 5 or not _DIGITS_ONLY.match(zip_text):
        return None
    prefix = int(zip_text[:3])
    return any(lo <= prefix <= hi for lo, hi in ranges)


def dq07_zip_state_mismatch(frame: MappedFrame) -> list[Finding]:
    """Zip and State contradict each other (SRS 9.2 DQ-07).

    Two shapes of contradiction. A four-digit zip is a lost leading zero, which
    is only possible where zips actually start with zero; in Texas it is
    something else. And a five-digit zip whose prefix belongs to another state is
    a mismatch outright.

    Neither carries an op. The cell pair is inconsistent but the source does not
    say *which* of the two is wrong, and a rule that picked one would be guessing
    with a reviewer's authority. Agent 3 states the contradiction and stops.
    """
    if "Zip" not in frame.raw or "State" not in frame.raw:
        return []

    out: list[Finding] = []
    for index, sheet_row, zip_text in _text_cells(frame, "Zip"):
        state = _state_code(frame, index)
        if state is None:
            continue
        compact = zip_text.replace(" ", "")

        if _DIGITS_ONLY.match(compact) and len(compact) == 4:
            if state in LEADING_ZERO_STATES:
                continue  # a genuine lost leading zero; DQ-08 proposes zip5
            out.append(
                findings.make(
                    "DQ-07",
                    sheet_row,
                    "Zip",
                    detail=(
                        f"Zip '{zip_text}' has four digits, but {state} has no zip "
                        "codes beginning with 0, so this is not a dropped leading "
                        "zero. Either the zip or the state is wrong"
                    ),
                    before=zip_text,
                    op=None,
                    source_column=frame.source_column.get("Zip"),
                    evidence={"zip": zip_text, "state": state},
                )
            )
            continue

        fits = _prefix_fits(state, compact)
        if fits is False:
            owners = sorted(
                code for code, ranges in STATE_ZIP_PREFIXES.items()
                if any(lo <= int(compact[:3]) <= hi for lo, hi in ranges)
            )
            belongs = f" Prefix {compact[:3]} belongs to {', '.join(owners)}." if owners else ""
            out.append(
                findings.make(
                    "DQ-07",
                    sheet_row,
                    "Zip",
                    detail=(
                        f"Zip '{zip_text}' is not in {state}.{belongs} Either the zip "
                        "or the state is wrong; the row does not say which"
                    ),
                    before=zip_text,
                    op=None,
                    source_column=frame.source_column.get("Zip"),
                    evidence={"zip": zip_text, "state": state, "prefix_owners": owners},
                )
            )
    return out


def dq08_zip_format(frame: MappedFrame) -> list[Finding]:
    """Zip is not a clean five-digit code (SRS 9.2 DQ-08).

    The SRS names "Zip+4 or text in Zip". A dropped leading zero is grouped here
    too, because it is the same kind of problem — the value is right and its
    notation is wrong — and because `zip5` fixes all three mechanically. That
    reading is why this rule and DQ-07 never both fire on one cell: a short zip
    whose state supports a leading zero is a format issue, and a short zip whose
    state does not is a contradiction.
    """
    if "Zip" not in frame.raw:
        return []

    out: list[Finding] = []
    for index, sheet_row, zip_text in _text_cells(frame, "Zip"):
        compact = zip_text.replace(" ", "")
        plus_four = _ZIP_PLUS_FOUR.match(compact)

        if plus_four:
            out.append(
                findings.make(
                    "DQ-08",
                    sheet_row,
                    "Zip",
                    detail=(
                        f"Zip '{zip_text}' is a Zip+4; SRS 5.1 stores the five-digit "
                        f"code, which is {plus_four.group(1)}"
                    ),
                    before=zip_text,
                    after=plus_four.group(1),
                    op="zip5",
                    source_column=frame.source_column.get("Zip"),
                    evidence={"zip": zip_text, "five_digit": plus_four.group(1)},
                )
            )
            continue

        if _DIGITS_ONLY.match(compact) and len(compact) == 5:
            continue  # already correct

        if _DIGITS_ONLY.match(compact) and len(compact) == 4:
            state = _state_code(frame, index)
            if state is not None and state not in LEADING_ZERO_STATES:
                continue  # DQ-07's contradiction, not a format issue
            padded = compact.zfill(5)
            where = f" {state} zips begin with 0." if state else ""
            out.append(
                findings.make(
                    "DQ-08",
                    sheet_row,
                    "Zip",
                    detail=(
                        f"Zip '{zip_text}' has four digits because a leading zero was "
                        f"lost, most likely to a numeric cell format.{where} The "
                        f"five-digit code is {padded}"
                    ),
                    before=zip_text,
                    after=padded,
                    op="zip5",
                    source_column=frame.source_column.get("Zip"),
                    evidence={"zip": zip_text, "five_digit": padded, "state": state},
                )
            )
            continue

        digits = re.sub(r"\D", "", compact)
        if len(digits) == 5:
            out.append(
                findings.make(
                    "DQ-08",
                    sheet_row,
                    "Zip",
                    detail=(
                        f"Zip '{zip_text}' carries text around a five-digit code; the "
                        f"code itself is {digits}"
                    ),
                    before=zip_text,
                    after=digits,
                    op="zip5",
                    source_column=frame.source_column.get("Zip"),
                    evidence={"zip": zip_text, "five_digit": digits},
                )
            )
        else:
            out.append(
                findings.make(
                    "DQ-08",
                    sheet_row,
                    "Zip",
                    detail=(
                        f"Zip '{zip_text}' is not a five-digit code and no five-digit "
                        "code can be read out of it"
                    ),
                    before=zip_text,
                    op=None,
                    source_column=frame.source_column.get("Zip"),
                    evidence={"zip": zip_text, "digits_found": len(digits)},
                )
            )
    return out


def dq09_state_format(frame: MappedFrame) -> list[Finding]:
    """State is a full name or a non-standard code (SRS 9.2 DQ-09).

    `state_to_abbrev` is mechanical for anything in the lookup, so those findings
    carry the exact code they would become. A code that is neither valid nor
    recognisable gets no op — "ZZ" is not a state and nothing in the whitelist can
    make it one.
    """
    if "State" not in frame.raw:
        return []

    out: list[Finding] = []
    for _index, sheet_row, text in _text_cells(frame, "State"):
        if text.upper() in STATE_ABBREVS and text == text.upper() and len(text) == 2:
            continue
        code = FULL_STATE_NAMES.get(text.lower().strip())
        if code is None and text.upper() in STATE_ABBREVS:
            # Right code, wrong case — "tx" for "TX".
            code = text.upper()
        if code:
            out.append(
                findings.make(
                    "DQ-09",
                    sheet_row,
                    "State",
                    detail=(
                        f"State '{text}' is not the two-letter USPS code SRS 5.1 "
                        f"requires; the code for it is {code}"
                    ),
                    before=text,
                    after=code,
                    op="state_to_abbrev",
                    source_column=frame.source_column.get("State"),
                    evidence={"state": text, "code": code},
                )
            )
        else:
            out.append(
                findings.make(
                    "DQ-09",
                    sheet_row,
                    "State",
                    detail=(
                        f"State '{text}' is neither a USPS code nor a state name this "
                        "engine recognises"
                    ),
                    before=text,
                    op=None,
                    source_column=frame.source_column.get("State"),
                    evidence={"state": text},
                )
            )
    return out


# --------------------------------------------------------------------------
# DQ-10 — notation in a numeric field
# --------------------------------------------------------------------------
def dq10_currency_formatting(frame: MappedFrame) -> list[Finding]:
    """A value field written for a human: `$2,100,000.00` (SRS 9.2 DQ-10).

    Only cells that *clean successfully* are reported. A cell the cleaner could
    not parse is DQ-02's, and reporting it here as well would offer
    `strip_currency_to_float` on something stripping will not rescue. So this
    rule sees exactly the cells where the op is guaranteed to work, and its
    after-value is the number the engine already computed.
    """
    out: list[Finding] = []
    for field_name in VALUE_FIELDS:
        for index, result in enumerate(frame.cleaned.get(field_name, [])):
            if result.status != "reformatted" or result.value is None:
                continue
            raw = frame.raw[field_name][index]
            shown = findings.render(raw)
            out.append(
                findings.make(
                    "DQ-10",
                    frame.row_label(index),
                    field_name,
                    detail=(
                        f"{field_name} reads '{shown}', which carries formatting "
                        f"({result.removed}) rather than a bare number"
                    ),
                    before=raw,
                    after=_number_text(result.value),
                    op="strip_currency_to_float",
                    source_column=frame.source_column.get(field_name),
                    evidence={
                        "value": shown,
                        "removed": result.removed,
                        "cleaned": result.value,
                    },
                )
            )
    return out


def _number_text(value: float) -> str:
    """A cleaned number as the output cell would hold it."""
    return str(int(value)) if float(value).is_integer() else f"{value}"


# --------------------------------------------------------------------------
# DQ-12 — placeholders
# --------------------------------------------------------------------------
def dq12_placeholders(frame: MappedFrame) -> list[Finding]:
    """The patterns SRS 9.2 DQ-12 names: Year Built 0 or a repeated 1900,
    Storeys 0, a value of 1.

    Every one of these carries `set_null`, and `set_null` is the only honest
    answer: the cell says nothing, so the cleaned file should say nothing rather
    than repeat a number that will be read as fact. SRS 9.1 fixes its output to an
    empty cell and `Recommendation` refuses any other, so C-02 holds even if a
    future caller tries to be helpful.

    Repetition is required for the suspicious round years because a single
    building really can date from 1900. `PLACEHOLDER_REPEAT_MIN` identical
    occurrences is the point at which it stops being a date and starts being a
    column someone filled in.
    """
    out: list[Finding] = []

    if "Year Built" in frame.raw:
        years: list[tuple[int, int, int]] = []
        for index, result in enumerate(frame.cleaned["Year Built"]):
            if result.value is not None and float(result.value).is_integer():
                years.append((index, frame.row_label(index), int(result.value)))
        counts = Counter(year for _, _, year in years)

        for index, sheet_row, year in years:
            if year not in PLACEHOLDER_YEARS:
                continue
            repeats = counts[year]
            # 0 is never a year at all. The round years need the repetition.
            if year != 0 and repeats < PLACEHOLDER_REPEAT_MIN:
                continue
            if year == 0:
                why = "0 is not a year; it is an unfilled cell"
            else:
                why = (
                    f"{year} appears on {repeats} rows, which reads as a filler value "
                    "rather than {repeats} buildings of that vintage"
                ).replace("{repeats}", str(repeats))
            out.append(
                findings.make(
                    "DQ-12",
                    sheet_row,
                    "Year Built",
                    detail=f"Year Built is a placeholder: {why}",
                    before=frame.raw["Year Built"][index],
                    after=None,
                    op="set_null",
                    source_column=frame.source_column.get("Year Built"),
                    evidence={"year": year, "occurrences": repeats},
                )
            )

    if "Storeys" in frame.raw:
        for index, result in enumerate(frame.cleaned["Storeys"]):
            if result.value is None or not float(result.value).is_integer():
                continue
            if int(result.value) not in PLACEHOLDER_STOREYS:
                continue
            out.append(
                findings.make(
                    "DQ-12",
                    frame.row_label(index),
                    "Storeys",
                    detail=(
                        "Storeys is 0, which is a placeholder rather than a building "
                        "with no floors"
                    ),
                    before=frame.raw["Storeys"][index],
                    after=None,
                    op="set_null",
                    source_column=frame.source_column.get("Storeys"),
                    evidence={"storeys": int(result.value)},
                )
            )

    for field_name in VALUE_FIELDS:
        for index, result in enumerate(frame.cleaned.get(field_name, [])):
            if result.value is None or result.value not in PLACEHOLDER_VALUES:
                continue
            out.append(
                findings.make(
                    "DQ-12",
                    frame.row_label(index),
                    field_name,
                    detail=(
                        f"{field_name} is {_number_text(result.value)}, which is a "
                        "placeholder for an unknown amount rather than a real "
                        "insured value"
                    ),
                    before=frame.raw[field_name][index],
                    after=None,
                    op="set_null",
                    source_column=frame.source_column.get(field_name),
                    evidence={"value": result.value},
                )
            )

    return out


# --------------------------------------------------------------------------
# DQ-13 — duplicate locations
# --------------------------------------------------------------------------
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_STREET_WORDS = {
    "street": "st", "st": "st", "avenue": "ave", "ave": "ave", "road": "rd",
    "rd": "rd", "drive": "dr", "dr": "dr", "boulevard": "blvd", "blvd": "blvd",
    "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "place": "pl", "pl": "pl",
    "parkway": "pkwy", "pkwy": "pkwy", "highway": "hwy", "hwy": "hwy",
    "suite": "ste", "ste": "ste", "north": "n", "south": "s", "east": "e",
    "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se",
    "southwest": "sw",
}


def normalise_address(text: str) -> str:
    """An address reduced to what two spellings of the same place share.

    Case, punctuation and the street-type words that have a dozen abbreviations
    each are the usual reason one location appears twice on a schedule. Nothing
    here drops a number: "100 Main St" and "1000 Main St" must stay different,
    so only the vocabulary is folded, never the digits.
    """
    lowered = _PUNCT.sub(" ", text.lower())
    words = [_STREET_WORDS.get(w, w) for w in _SPACES.sub(" ", lowered).split()]
    return " ".join(words).strip()


def dq13_duplicate_locations(frame: MappedFrame) -> list[Finding]:
    """The same normalised address and zip on more than one row (SRS 9.2 DQ-13).

    No op. A duplicate is sometimes a double-entered row and sometimes two
    genuinely separate policies at one address, and only the client knows which.
    `exclude_row` is in the whitelist but choosing *which* of the duplicates to
    drop is precisely the judgement a reviewer is for.
    """
    if "Address" not in frame.raw:
        return []

    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for index, sheet_row, text in _text_cells(frame, "Address"):
        key_address = normalise_address(text)
        if not key_address:
            continue
        zip_raw = frame.cell("Zip", index)
        key_zip = "" if is_null(zip_raw) else re.sub(r"\D", "", str(zip_raw))[:5]
        groups[(key_address, key_zip)].append((index, sheet_row))

    out: list[Finding] = []
    for (key_address, key_zip), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        rows = [sheet_row for _, sheet_row in members]

        # A schedule for a campus, a school district or a shopping centre lists
        # several buildings at one street address, and each carries its own
        # location reference. That is not a double entry, and saying so is the
        # difference between a useful flag and 454 of them on one file. The
        # finding still stands — SRS 9.2 defines the duplicate by address and zip
        # — but it reports which of the two situations the row is in.
        references = [frame.cell("Reference", i) for i, _ in members]
        named = [str(r).strip() for r in references if not is_null(r)]
        distinct_refs = len(named) == len(members) and len(set(named)) == len(members)

        for index, sheet_row in members:
            others = [r for r in rows if r != sheet_row]
            if distinct_refs:
                reading = (
                    ". Each of those rows carries its own location reference, so "
                    "these are more likely separate buildings at one address than a "
                    "double entry; confirm before excluding any of them"
                )
            else:
                reading = (
                    ". The rows do not have distinct location references, so this "
                    "may be a double entry, which would double the insured value"
                )
            out.append(
                findings.make(
                    "DQ-13",
                    sheet_row,
                    "Address",
                    detail=(
                        f"This location also appears on row(s) "
                        f"{', '.join(str(r) for r in others)}"
                        + (f" with zip {key_zip}" if key_zip else "")
                        + reading
                    ),
                    before=frame.raw["Address"][index],
                    op=None,
                    source_column=frame.source_column.get("Address"),
                    evidence={
                        "normalised": key_address,
                        "zip": key_zip,
                        "duplicate_rows": rows,
                        "distinct_references": distinct_refs,
                    },
                )
            )
    return out


# --------------------------------------------------------------------------
# DQ-14 — frame height
# --------------------------------------------------------------------------
def is_frame_construction(text: str) -> bool:
    """Is this construction description combustible wood frame?

    Exclusions win. "Masonry Non-Combustible (ISO 4)" contains the substring
    "combustible" and appears 154 times in one real file; "Steel Frame" contains
    "frame". Checking the exclusions first is what stops DQ-14 reporting a few
    hundred masonry and steel buildings as wood towers.
    """
    lowered = text.lower()
    if any(term in lowered for term in FRAME_EXCLUSIONS):
        return False
    return any(
        re.search(rf"\b{re.escape(term)}\b", lowered)
        for term in FRAME_CONSTRUCTION_TERMS
    )


def dq14_frame_height(frame: MappedFrame) -> list[Finding]:
    """Wood or frame construction above six storeys (SRS 9.2 DQ-14).

    No op, and the rule does not claim which field is wrong. A fourteen-storey
    frame building is implausible, but the error could be the construction class
    or the storey count, and both are underwriting inputs.
    """
    if "Construction" not in frame.raw or "Storeys" not in frame.raw:
        return []

    out: list[Finding] = []
    for index, sheet_row, text in _text_cells(frame, "Construction"):
        if not is_frame_construction(text):
            continue
        storeys = frame.cleaned["Storeys"][index].value
        if storeys is None or storeys <= MAX_PLAUSIBLE_FRAME_STOREYS:
            continue
        out.append(
            findings.make(
                "DQ-14",
                sheet_row,
                "Construction",
                detail=(
                    f"Construction is '{text}' with {_number_text(storeys)} storeys. "
                    f"Combustible frame is rarely built above "
                    f"{MAX_PLAUSIBLE_FRAME_STOREYS} storeys, so either the "
                    "construction class or the storey count is wrong"
                ),
                before=text,
                op=None,
                source_column=frame.source_column.get("Construction"),
                evidence={"construction": text, "storeys": storeys},
            )
        )
    return out


# --------------------------------------------------------------------------
# DQ-17 — mixed date formats
# --------------------------------------------------------------------------
def _date_shape(text: str) -> Optional[str]:
    for label, pattern in _DATE_SHAPES:
        if pattern.match(text) or (label == "month name" and pattern.search(text)):
            return label
    return None


def dq17_mixed_dates(frame: MappedFrame) -> list[Finding]:
    """One column, more than one date notation (SRS 9.2 DQ-17).

    Only fires when a *mixture* is present: a column written entirely as
    `01/01/1990` is consistent, and a consistent column is a mapping question
    rather than a quality one. The finding goes on the minority notations, since
    those are the cells a reviewer has to look at, and `to_int` carries the year
    the engine already recovered where it could recover one.
    """
    if "Year Built" not in frame.raw:
        return []

    shapes: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for index, sheet_row, text in _text_cells(frame, "Year Built"):
        shape = _date_shape(text)
        if shape:
            shapes[shape].append((index, sheet_row, text))

    if len(shapes) < 2:
        return []

    dominant = max(shapes, key=lambda s: len(shapes[s]))
    out: list[Finding] = []
    for shape, members in sorted(shapes.items()):
        if shape == dominant:
            continue
        for index, sheet_row, text in members:
            result = frame.cleaned["Year Built"][index]
            recovered = (
                str(int(result.value))
                if result.value is not None and float(result.value).is_integer()
                else None
            )
            out.append(
                findings.make(
                    "DQ-17",
                    sheet_row,
                    "Year Built",
                    detail=(
                        f"Year Built reads '{text}', a {shape}, while {len(shapes[dominant])} "
                        f"other rows in this column are written as a {dominant}"
                        + (
                            f". The year in it is {recovered}"
                            if recovered
                            else ". No single year can be read from it"
                        )
                    ),
                    before=frame.raw["Year Built"][index],
                    after=recovered,
                    op="to_int" if recovered else None,
                    source_column=frame.source_column.get("Year Built"),
                    evidence={
                        "value": text,
                        "shape": shape,
                        "dominant_shape": dominant,
                        "dominant_count": len(shapes[dominant]),
                    },
                )
            )
    return out


# --------------------------------------------------------------------------
# DQ-18 — component outliers
# --------------------------------------------------------------------------
def dq18_component_outliers(frame: MappedFrame) -> list[Finding]:
    """Contents or BI more than ten times Building Value (SRS 9.2 DQ-18).

    No op. The ratio says one of the two figures is probably in the wrong column
    or the wrong unit, and code cannot tell which without being told what the
    building is. Rows where Building Value is absent or zero are skipped rather
    than reported, because a ratio against nothing is not a ratio.
    """
    if "Building Value" not in frame.raw:
        return []

    out: list[Finding] = []
    for field_name in ("Contents", "BI"):
        if field_name not in frame.raw:
            continue
        for index, result in enumerate(frame.cleaned[field_name]):
            building = frame.cleaned["Building Value"][index].value
            if result.value is None or building is None or building <= 0:
                continue
            if result.value <= building * COMPONENT_RATIO_LIMIT:
                continue
            ratio = result.value / building
            out.append(
                findings.make(
                    "DQ-18",
                    frame.row_label(index),
                    field_name,
                    detail=(
                        f"{field_name} is {_number_text(result.value)} against a "
                        f"Building Value of {_number_text(building)}, a ratio of "
                        f"{ratio:.1f}x. Above {COMPONENT_RATIO_LIMIT:.0f}x the two "
                        "figures are usually swapped, mis-scaled, or in the wrong "
                        "column"
                    ),
                    before=frame.raw[field_name][index],
                    op=None,
                    source_column=frame.source_column.get(field_name),
                    evidence={
                        field_name: result.value,
                        "Building Value": building,
                        "ratio": round(ratio, 2),
                    },
                )
            )
    return out


# --------------------------------------------------------------------------
# Running them, and saying which could not run.
# --------------------------------------------------------------------------
# Every rule in this module, with the fields it needs. The requirement list is
# what lets the agent distinguish "ran and found nothing" from "could not run",
# which `QualityBlock` insists on for all eighteen rules.
DOMAIN_RULES: tuple[tuple[str, Any, tuple[str, ...]], ...] = (
    ("DQ-01", dq01_missing_values, ()),
    ("DQ-07", dq07_zip_state_mismatch, ("Zip", "State")),
    ("DQ-08", dq08_zip_format, ("Zip",)),
    ("DQ-09", dq09_state_format, ("State",)),
    ("DQ-10", dq10_currency_formatting, ()),
    ("DQ-12", dq12_placeholders, ()),
    ("DQ-13", dq13_duplicate_locations, ("Address",)),
    ("DQ-14", dq14_frame_height, ("Construction", "Storeys")),
    ("DQ-17", dq17_mixed_dates, ("Year Built",)),
    ("DQ-18", dq18_component_outliers, ("Building Value",)),
)


def missing_requirements(frame: MappedFrame, required: tuple[str, ...]) -> list[str]:
    return [name for name in required if name not in frame.raw]


def run_domain_rules(frame: MappedFrame) -> tuple[list[Finding], dict[str, str]]:
    """Every domain rule, plus a reason for each one that could not run."""
    out: list[Finding] = []
    skipped: dict[str, str] = {}
    for rule, fn, required in DOMAIN_RULES:
        absent = missing_requirements(frame, required)
        if absent:
            skipped[rule] = (
                f"needs {', '.join(absent)}, which Agent 2 did not map"
            )
            continue
        out.extend(fn(frame))
    return out, skipped


# DQ-12 reports a cell as an unfilled placeholder and proposes emptying it. DQ-05
# and DQ-06 independently report the same cell as out of range. Both are true, but
# a reviewer does not need to be told twice that Year Built 9999 is wrong — and the
# placeholder finding is the one carrying an op. So a DQ-12 finding supersedes the
# range findings on the same cell; the superseded rules are recorded on the
# survivor's evidence rather than dropped silently.
SUPERSEDES: dict[str, frozenset[str]] = {
    "DQ-12": frozenset({"DQ-05", "DQ-06"}),
}


def reconcile(all_findings: list[Finding]) -> list[Finding]:
    """Apply `SUPERSEDES`, folding the loser's rule id into the winner's evidence."""
    claimed: dict[tuple[Optional[str], int], set[str]] = defaultdict(set)
    for item in all_findings:
        claimed[(item.field, item.row)].add(item.rule)

    kept: list[Finding] = []
    for item in all_findings:
        cell = (item.field, item.row)
        superseded_by = [
            winner
            for winner, losers in SUPERSEDES.items()
            if item.rule in losers and winner in claimed[cell]
        ]
        if superseded_by:
            continue
        also = sorted(
            SUPERSEDES.get(item.rule, frozenset()) & claimed[cell]
        )
        if also:
            kept.append(
                replace(item, evidence={**item.evidence, "also_flagged": also})
            )
        else:
            kept.append(item)
    return kept


__all__ = [
    "DOMAIN_RULES",
    "SUPERSEDES",
    "dq01_missing_values",
    "dq07_zip_state_mismatch",
    "dq08_zip_format",
    "dq09_state_format",
    "dq10_currency_formatting",
    "dq12_placeholders",
    "dq13_duplicate_locations",
    "dq14_frame_height",
    "dq17_mixed_dates",
    "dq18_component_outliers",
    "is_frame_construction",
    "normalise_address",
    "reconcile",
    "run_domain_rules",
]
