"""Fingerprint channel (FR-MAP-03) — evidence from the column's values, not its name.

Profiles the data under a header into a distribution over value *shapes* (currency,
year, ZIP, US state code, boolean, street address, place name, small integer,
identifier, free category) and scores each target by how much of that distribution
lands in the shapes the target accepts.

This is the channel that survives a header being useless. A column called `Col7`
full of `1978, 1995, 2004` is a `Year Built` and nothing else, and no amount of
string matching will tell you that.

Its deliberate limitation, stated rather than hidden: a fingerprint **cannot**
separate `Building Value` from `Contents`, `BI` or `Other` — all four are currency
and all four receive the same score. That is not a defect to be tuned away. It
still constrains the assignment usefully (it rules those four out for every
non-currency column), and lexical and semantic do the separating.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from backend.agents.mapping.targets import TARGETS, ValueShape

# How many non-empty values to profile. Shape distributions stabilise long before
# this; reading 5,000 rows per column to learn "these are dollars" is waste.
SAMPLE_SIZE = 200

# A column must have at least this many non-empty values before its fingerprint is
# treated as evidence. Two values can agree by chance; the channel reports
# zero-confidence rather than guessing.
MIN_SAMPLE = 3

# Shapes the target accepts but does not *prefer* still count, at a discount, so
# "State spelled out in full" is not scored as hard as "State as a 2-letter code".
SECONDARY_SHAPE_WEIGHT = 0.6

# A distinct ratio this low suggests a controlled vocabulary rather than free text,
# but only once there are enough values for the ratio to mean anything.
CATEGORY_DISTINCT_RATIO = 0.4
CARDINALITY_MIN_SAMPLE = 8

# Shapes this channel genuinely cannot tell apart (see the note in targets.py).
# Declared once here because two separate callers need the same truth: the target
# catalogue, which accepts both shapes for the affected fields, and the
# adjudicator's verifier, which must not reject a citation of `category` on a
# column it happened to label `place_name`. Without this, the verifier would
# discard *truthful* LLM citations, which is as damaging as accepting false ones.
SHAPE_EQUIVALENCE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"place_name", "category"}),
    frozenset({"zip", "identifier"}),
)


def shapes_equivalent(claimed: str, observed: str) -> bool:
    claimed, observed = claimed.lower().strip(), observed.lower().strip()
    if claimed == observed:
        return True
    return any(
        claimed in group and observed in group for group in SHAPE_EQUIVALENCE_GROUPS
    )


def equivalent_share(distribution: dict[str, float], claimed: str) -> float:
    """Share of the column's values whose shape is equivalent to `claimed`."""
    return sum(
        share for shape, share in distribution.items() if shapes_equivalent(claimed, shape)
    )

US_STATES = frozenset(
    """AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO
    MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY
    DC PR VI GU AS MP AB BC MB NB NL NS NT NU ON PE QC SK YT""".split()
)

BOOLEAN_TOKENS = frozenset(
    {"y", "n", "yes", "no", "true", "false", "t", "f", "1", "0", "yes.", "none",
     "y/n", "partial", "full", "unsprinklered", "sprinklered"}
)

# Construction/occupancy-style vocabularies are genuinely open-ended, so a
# category is recognised structurally (short, repeating, alphabetic) rather than
# by listing every possible value.
_CURRENCY_CHARS = re.compile(r"^[\s$£€]*\(?-?[\d,]+(\.\d+)?\)?[\s]*$")
_STREET = re.compile(
    r"^\s*\d+[\w\-]*\s+\S+.*?\b("
    r"st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|way|ct|court|"
    r"pl|place|pkwy|parkway|hwy|highway|cir|circle|ter|terrace|trl|trail|sq|square"
    r")\b\.?",
    re.IGNORECASE,
)
_STREET_LOOSE = re.compile(r"^\s*\d+\s+[A-Za-z]")
_ZIP = re.compile(r"^\s*\d{5}(-\d{4})?\s*$")
_ZIP_SHORT = re.compile(r"^\s*\d{3,4}\s*$")
# A currency symbol, thousands separator or decimal part settles the question
# immediately: no postal code is written `$15,000` or `15000.00`.
_AMOUNT_MARKER = re.compile(r"[$£€,]|\.\d")

# Internal placeholder, never returned to callers. profile_column replaces every
# occurrence with a real shape once it can see the whole column.
_AMBIGUOUS_DIGITS = "_digits"

# Significant-digit counts a US postal code can legitimately show once a
# spreadsheet has stripped its leading zero: 5 normally, 4 for the 0xxxx range
# (New England, NJ, PR). A column whose widths fall outside this set is spanning
# magnitudes, which postal codes do not do.
_ZIP_WIDTHS = frozenset({4, 5})

# Largest max/min ratio a column of postal codes can plausibly show. The widest
# real spread is a 4-digit ZIP (leading zero stripped, so ~1000) beside a 5-digit
# one (~99950), which is a factor of about 100; a values column crosses that
# easily. Set generously, because the cost of misreading a ZIP as currency is a
# correct target scored 0.0, while the cost of the reverse is one noisy candidate
# among seventeen.
ZIP_MAGNITUDE_SPREAD = 120.0
_CA_POSTAL = re.compile(r"^\s*[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d\s*$")
_IDENTIFIER = re.compile(r"^[\w\-/\.]+$")
_PLACE = re.compile(r"^[A-Za-z][A-Za-z\s\.'\-]{1,40}$")

# A year that is plausibly a construction date. 1900 and 0 are the classic
# placeholders for "unknown" and are still *shaped* like years, so they count
# here; it is Agent 3's job to flag them as placeholders, not this channel's.
YEAR_MIN, YEAR_MAX = 1700, date.today().year + 2

SMALL_INT_MAX = 500


def _text(value: Any) -> str:
    return str(value).strip()


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _text(value)
    if not _CURRENCY_CHARS.match(text):
        return None
    negative = text.startswith("(") or text.endswith(")") or text.lstrip().startswith("-")
    cleaned = re.sub(r"[^\d.]", "", text)
    if not cleaned or cleaned.count(".") > 1:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return -number if negative else number


def classify_value(value: Any) -> ValueShape | None:
    """The single most specific shape one cell is. None for empty/unrecognisable.

    Order matters: a four-digit 1978 is a year before it is a small integer, and a
    5-digit 90210 is a ZIP before it is currency.
    """
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return "year"
    if isinstance(value, bool):
        return "boolean"

    text = _text(value)
    if not text or text.lower() in {"na", "n/a", "none", "null", "-", "--", "tbd", "unknown"}:
        return None

    lowered = text.lower()
    if lowered in BOOLEAN_TOKENS and not text.isdigit():
        return "boolean"

    upper = text.upper()
    if len(upper) == 2 and upper in US_STATES:
        return "us_state"
    if _CA_POSTAL.match(text):
        return "zip"

    number = _as_number(value)
    if number is not None:
        integral = float(number).is_integer()
        explicit_amount = bool(_AMOUNT_MARKER.search(text))

        if (
            integral
            and not explicit_amount
            and YEAR_MIN <= number <= YEAR_MAX
            and len(text.strip()) == 4
        ):
            return "year"
        if explicit_amount or not integral or number < 0:
            return "currency"
        digits = len(str(int(number)))
        if digits >= 6:
            # No postal code is this long, so the ambiguity below does not arise.
            return "currency"
        if digits >= 3:
            # 3-5 bare digits are genuinely ambiguous between a postal code, a
            # location number and a small currency amount. One cell cannot tell
            # — `15000` is a plausible ZIP and a plausible dollar figure — so the
            # decision is deferred to profile_column, which can see the whole
            # column. Deciding here is what made every 5-digit dollar amount
            # fingerprint as a ZIP.
            return _AMBIGUOUS_DIGITS
        if 0 <= number <= SMALL_INT_MAX:
            return "small_int"
        return "currency"

    if _STREET.search(text) or _STREET_LOOSE.match(text):
        return "street_address"
    if _PLACE.match(text):
        # Multi-word title-case strings read as place names; single short repeating
        # tokens read as categories. The caller refines this using repetition
        # across the whole column, which one cell cannot show.
        return "place_name"
    if _IDENTIFIER.match(text):
        return "identifier"
    return "category"


@dataclass
class ColumnFingerprint:
    sampled: int
    non_empty: int
    shape_counts: Counter = field(default_factory=Counter)
    distinct_ratio: float = 0.0
    numeric_mean: float | None = None

    @property
    def reliable(self) -> bool:
        return self.non_empty >= MIN_SAMPLE

    @property
    def distribution(self) -> dict[str, float]:
        total = sum(self.shape_counts.values())
        if not total:
            return {}
        return {shape: count / total for shape, count in self.shape_counts.items()}

    def dominant(self) -> tuple[str, float] | None:
        dist = self.distribution
        if not dist:
            return None
        shape = max(dist, key=lambda s: dist[s])
        return shape, round(dist[shape], 4)


def _resolve_ambiguous_digits(texts: list[str]) -> str:
    """Decide what a column of bare 3-5 digit integers actually is.

    Postal codes are narrow-range identifiers: four or five significant digits,
    never zero, rarely round, and all of a similar magnitude. Currency amounts
    are none of those things — a values column is full of zeros, round thousands,
    and magnitudes spanning orders.

    Note what is deliberately *not* tested here: how often values repeat. An
    earlier version rejected any column whose distinct ratio fell below 0.6, on
    the assumption that postal codes rarely repeat. Real SOVs say otherwise — a
    multi-building campus is one address, so `29640` appears on all twenty rows,
    and that rule scored a column literally headed `Zip` 0.0 on this channel
    while handing `BI` 1.0. Repetition is not evidence against a postal code; if
    anything a single-site schedule guarantees it.

    Returns 'zip' or 'currency'. It does not attempt to distinguish a 4-digit
    postal code from a 4-digit location number, because nothing in the values can:
    `1001, 1002, 1003` is equally a ZIP that lost its leading zero (the exact
    defect this system exists to catch) and a sequence of location references.
    That ambiguity is handled by `Reference` accepting the `zip` shape, not by
    guessing here.
    """
    numbers = [int(t) for t in texts]
    stripped = [t.strip() for t in texts]
    widths = {len(t.lstrip("0") or "0") for t in stripped}

    # A preserved leading zero is unambiguous positive evidence: `00802` is the
    # US Virgin Islands, and nobody writes an amount of money that way. Checked
    # first because the width test below strips zeros to measure significant
    # digits, which would reduce `00802` to three and reject it as too narrow.
    if any(len(t) in _ZIP_WIDTHS and t.startswith("0") for t in stripped):
        return "zip"

    if any(n == 0 for n in numbers):
        return "currency"
    # Mixed digit widths usually mean magnitudes rather than identifiers — but
    # *not* when the mix is exactly 4-and-5, which is the signature of a ZIP
    # column whose leading zeros were eaten by a spreadsheet (`07030` stored as
    # `7030` beside `10001`). That is the defect ASSAY exists to catch, so an
    # earlier version of this rule — any mixed width means currency — inverted
    # the intended behaviour and scored real `Zip` columns 0.0 on this channel.
    if not widths <= _ZIP_WIDTHS:
        return "currency"
    # Round hundreds across the board reads as money, not as postal codes.
    if len(numbers) >= 3 and all(n % 100 == 0 for n in numbers):
        return "currency"
    # Amounts span magnitudes; postal codes do not. `8000` beside `42000` is a
    # values column, whereas every ZIP in a state shares a prefix and therefore
    # a magnitude. This replaces the distinct-ratio test described above: it
    # measures the thing that actually differs, and an all-identical column —
    # spread 1.0 — now reads as `zip` instead of being disqualified.
    if max(numbers) / min(numbers) > ZIP_MAGNITUDE_SPREAD:
        return "currency"
    return "zip"


def profile_column(values: list[Any]) -> ColumnFingerprint:
    sample = values[:SAMPLE_SIZE]
    shapes: Counter = Counter()
    numbers: list[float] = []
    seen: set[str] = set()
    non_empty = 0
    ambiguous_texts: list[str] = []

    for value in sample:
        shape = classify_value(value)
        if shape is None:
            continue
        non_empty += 1
        shapes[shape] += 1
        seen.add(_text(value).lower())
        if shape is _AMBIGUOUS_DIGITS:
            ambiguous_texts.append(_text(value))
        number = _as_number(value)
        if number is not None:
            numbers.append(number)

    # Resolve the deferred digit ambiguity now that the whole column is visible.
    if ambiguous_texts:
        resolved = _resolve_ambiguous_digits(ambiguous_texts)
        shapes[resolved] += shapes.pop(_AMBIGUOUS_DIGITS)

    fingerprint = ColumnFingerprint(
        sampled=len(sample),
        non_empty=non_empty,
        shape_counts=shapes,
        distinct_ratio=round(len(seen) / non_empty, 4) if non_empty else 0.0,
        numeric_mean=round(sum(numbers) / len(numbers), 2) if numbers else None,
    )

    # A low-cardinality column of short words leans towards a controlled
    # vocabulary (Construction, Occupancy) rather than a list of place names. Only
    # the whole column can show this, so it is corrected here rather than in
    # classify_value.
    #
    # This is a *lean*, not a decision: `place_name` and `category` are accepted
    # interchangeably by every target in that group (see targets.py), because on a
    # 3-row sample `Masonry, Frame, Masonry` has a distinct ratio of 0.67 and is
    # indistinguishable from `Houston, Chicago, Houston`. The sample-size guard
    # stops the ratio being read as meaningful when it cannot be.
    if (
        fingerprint.non_empty >= CARDINALITY_MIN_SAMPLE
        and fingerprint.distinct_ratio <= CATEGORY_DISTINCT_RATIO
        and shapes.get("place_name")
    ):
        moved = shapes.pop("place_name")
        shapes["category"] += moved

    return fingerprint


@dataclass(frozen=True)
class FingerprintHit:
    target: str
    score: float
    dominant_shape: str
    dominant_share: float
    # True when this channel abstained (too few values). Fusion reweights rather
    # than reading the zero as positive evidence of a mismatch.
    abstained: bool


def score_column(values: list[Any]) -> dict[str, FingerprintHit]:
    fingerprint = profile_column(values)
    dominant = fingerprint.dominant()
    shape_name, shape_share = dominant if dominant else ("unknown", 0.0)

    if not fingerprint.reliable:
        return {
            name: FingerprintHit(name, 0.0, shape_name, shape_share, abstained=True)
            for name in TARGETS
        }

    distribution = fingerprint.distribution
    hits: dict[str, FingerprintHit] = {}
    for name, spec in TARGETS.items():
        score = 0.0
        for index, shape in enumerate(spec.shapes):
            weight = 1.0 if index == 0 else SECONDARY_SHAPE_WEIGHT
            score += weight * distribution.get(shape, 0.0)
        hits[name] = FingerprintHit(
            target=name,
            score=round(min(score, 1.0), 4),
            dominant_shape=shape_name,
            dominant_share=shape_share,
            abstained=False,
        )
    return hits
