"""Lexical channel (FR-MAP-02) — fuzzy string evidence from the header text alone.

RapidFuzz `token_set_ratio` against every glossary phrase for every target. This is
the channel that catches `Bldg Repl Cost` → `Building Value` and `Yr Blt` →
`Year Built`, and it is the only channel whose evidence a human can check instantly:
"this header matched the glossary phrase 'repl cost' at 0.91".

The returned score is the raw best ratio, not a thresholded one. Fusion wants the
real magnitude of the signal; `LEXICAL_ATTRIBUTION_FLOOR` governs only whether
`lexical` is named as a *contributing method* in the citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from backend.agents.mapping.targets import GLOSSARY, TARGETS

# Below this, a lexical match is treated as incidental overlap rather than
# evidence worth naming. The signal still reaches fusion; only attribution stops.
LEXICAL_ATTRIBUTION_FLOOR = 0.75

# An exact glossary hit is qualitatively different from a fuzzy one and is
# reported as such, so a reviewer can tell "identical to glossary" from "close to".
EXACT_SCORE = 1.0

_COMPOSITE_SEPARATOR = "›"
_PUNCT = re.compile(r"[^a-z0-9]+")

# Expanded before matching so abbreviated headers reach their glossary phrases.
# Brokers write these, not the long forms, and token_set_ratio cannot bridge
# 'bldg' to 'building' on its own.
ABBREVIATIONS: dict[str, str] = {
    "bldg": "building",
    "bldgs": "buildings",
    "repl": "replacement",
    "rc": "replacement cost",
    "cost": "cost",
    "val": "value",
    "vals": "values",
    "amt": "amount",
    "tiv": "total insurable value",
    "yr": "year",
    "blt": "built",
    "yob": "year built",
    "eyb": "effective year built",
    "const": "construction",
    "constr": "construction",
    "cons": "construction",
    "occ": "occupancy",
    "sprk": "sprinkler",
    "spkr": "sprinkler",
    "spr": "sprinkler",
    "prot": "protection",
    "st": "state",
    "addr": "address",
    "cnty": "county",
    "ctry": "country",
    "cntry": "country",
    "bi": "business income",
    "ee": "extra expense",
    "bpp": "business personal property",
    "pers": "personal",
    "prop": "property",
    "qty": "quantity",
    "num": "number",
    "no": "number",
    "seq": "sequence",
    "ref": "reference",
    "loc": "location",
    "desc": "description",
    "acct": "account",
    "m&e": "machinery and equipment",
    "sq": "square",
    "ft": "feet",
}


def normalise_header(header: str) -> str:
    """Lowercase, split a composite header on `›`, strip punctuation to spaces.

    A composite header `Values › Building` keeps both halves: the banner carries
    'this is a value' and the sub-row carries 'which value'. Dropping either loses
    the distinction between `Values › Building` and `Values › Contents`.
    """
    text = header.replace(_COMPOSITE_SEPARATOR, " ")
    text = _PUNCT.sub(" ", text.lower())
    return " ".join(text.split())


def expand_abbreviations(text: str) -> str:
    """Rewrite known broker abbreviations to their long forms, keeping the original
    token as well so an exact glossary spelling of the abbreviation still hits."""
    out: list[str] = []
    for token in text.split():
        out.append(token)
        expansion = ABBREVIATIONS.get(token)
        if expansion and expansion != token:
            out.append(expansion)
    return " ".join(out)


def prepare(header: str) -> str:
    return expand_abbreviations(normalise_header(header))


@dataclass(frozen=True)
class LexicalHit:
    target: str
    score: float
    # The glossary phrase that produced the score. This is the citation; a mapping
    # rationale quotes it verbatim.
    matched_phrase: str
    exact: bool

    @property
    def attributable(self) -> bool:
        return self.score >= LEXICAL_ATTRIBUTION_FLOOR


def score_header(header: str) -> dict[str, LexicalHit]:
    """Best lexical hit per target for one source header."""
    prepared = prepare(header)
    bare = normalise_header(header)
    hits: dict[str, LexicalHit] = {}

    for target in TARGETS:
        best_score = 0.0
        best_phrase = ""
        best_exact = False
        for phrase in GLOSSARY[target]:
            phrase_norm = normalise_header(phrase)
            if not phrase_norm:
                continue
            if phrase_norm == bare:
                best_score, best_phrase, best_exact = EXACT_SCORE, phrase, True
                break
            ratio = fuzz.token_set_ratio(prepared, phrase_norm) / 100.0
            if ratio > best_score:
                best_score, best_phrase, best_exact = ratio, phrase, False
        hits[target] = LexicalHit(
            target=target,
            score=round(best_score, 4),
            matched_phrase=best_phrase,
            exact=best_exact,
        )
    return hits
