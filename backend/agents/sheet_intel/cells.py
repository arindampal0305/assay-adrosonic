"""Cell-level primitives shared by header detection and sheet scoring."""

from __future__ import annotations

import re
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any, Iterable

from backend.state.target_schema import TARGET_VOCABULARY

NUMERIC_RE = re.compile(
    r"^[\(\-\+]?\s*[$€£¥]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%?\s*\)?$|"
    r"^[\(\-\+]?\s*[$€£¥]?\s*\d+(?:\.\d+)?\s*%?\s*\)?$"
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

CellKind = str
FUZZY_THRESHOLD = 0.86


def cell_kind(value: Any) -> CellKind:
    if value is None:
        return "empty"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (datetime, date)):
        return "date"
    if isinstance(value, (int, float)):
        return "number"
    text = str(value).strip()
    if not text:
        return "empty"
    if NUMERIC_RE.match(text):
        return "number"
    return "text"


def looks_like_label(value: Any) -> bool:
    """True for cells that read as a column name rather than a data point."""
    return cell_kind(value) == "text"


def normalise(value: Any) -> str:
    if value is None:
        return ""
    return _NON_ALNUM.sub(" ", str(value).lower()).strip()


def tokens(value: Any) -> set[str]:
    return {t for t in normalise(value).split() if t}


def _phrase_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for target, phrases in TARGET_VOCABULARY.items():
        for phrase in phrases:
            index.setdefault(normalise(phrase), target)
        index.setdefault(normalise(target), target)
    return index


PHRASE_INDEX: dict[str, str] = _phrase_index()


def match_target(value: Any) -> str | None:
    """Best-effort target for one header cell. Deliberately shallow: Agent 2 owns
    real mapping; Agent 1 only needs to know whether a row looks like an SOV header."""
    text = normalise(value)
    if not text:
        return None
    if text in PHRASE_INDEX:
        return PHRASE_INDEX[text]

    cell_tokens = tokens(text)
    best: tuple[float, str | None] = (0.0, None)
    for phrase, target in PHRASE_INDEX.items():
        phrase_tokens = tokens(phrase)
        if phrase_tokens and phrase_tokens <= cell_tokens:
            coverage = 0.9 + 0.1 * (len(phrase_tokens) / max(len(cell_tokens), 1))
            if coverage > best[0]:
                best = (coverage, target)
            continue
        ratio = SequenceMatcher(None, text, phrase).ratio()
        if ratio >= FUZZY_THRESHOLD and ratio > best[0]:
            best = (ratio, target)
    return best[1]


def vocabulary_matches(values: Iterable[Any]) -> dict[int, str]:
    """Index -> matched target, for the cells of a candidate header row."""
    matches: dict[int, str] = {}
    for idx, value in enumerate(values):
        target = match_target(value)
        if target:
            matches[idx] = target
    return matches
