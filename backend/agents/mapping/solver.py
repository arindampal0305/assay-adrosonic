"""Fusion and global assignment (FR-MAP-04 … FR-MAP-07).

Two steps, deliberately separated.

**Fusion** combines the three channels into one score per (column, target) cell.
Weights live in `CHANNEL_WEIGHTS` as module config, and are *renormalised over the
channels that actually voted* — if the embedding model is missing or the
fingerprint abstained on a near-empty column, the remaining channels carry full
weight instead of the cell being silently penalised.

**Assignment** is global, not greedy. Picking each column's argmax independently
happily maps three different columns onto `Building Value`; the target schema is a
set of 17 distinct fields, so this is a bipartite matching problem and
`scipy.optimize.linear_sum_assignment` solves it exactly. The practical payoff is
that a weak-but-unambiguous column wins the target that a strong-but-contested one
loses, which is the behaviour an underwriter expects.

The one thing Hungarian gets wrong on its own is that it *must* assign every
column. `MIN_ASSIGNMENT_SCORE` is the escape hatch: an assignment below it is
discarded and the column left explicitly unmapped, which is a far more useful
output than a confident-looking wrong target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.agents.mapping.channels import fingerprint as fp_channel
from backend.agents.mapping.channels import lexical as lex_channel
from backend.agents.mapping.channels import semantic as sem_channel
from backend.agents.mapping.targets import TARGETS

# Channel weights. Config, not inline literals, so they can be tuned and version-
# controlled as a single visible decision.
#
# Rationale for these values: semantic and fingerprint are weighted slightly above
# lexical because lexical is the channel most prone to *confident* error — a short
# glossary phrase like 'bi' or 'st' fuzzy-matches far too much — whereas the other
# two fail by being vague rather than by being wrong.
CHANNEL_WEIGHTS: dict[str, float] = {
    "lexical": 0.30,
    "semantic": 0.35,
    "fingerprint": 0.35,
}

# Below this fused score an assignment is rejected and the column left unmapped.
MIN_ASSIGNMENT_SCORE = 0.25

# Evidence admissibility (see `_has_admissible_evidence`). One of these two must
# be met for a column to claim a target at all. Both are set where the measured
# populations actually divide on real files: correct-but-weak mappings sat at
# lexical 1.00, while every misassignment fell below 0.90 lexical and 0.70
# semantic. `ADMISSIBLE_SEMANTIC` is well above the 0.35 calibration floor,
# below which the embedding carries no signal at all.
ADMISSIBLE_LEXICAL = 0.90
ADMISSIBLE_SEMANTIC = 0.70

# Margin = fused score of the chosen target minus that of the runner-up. Below
# this, the decision was close enough that the LLM adjudicator is consulted.
LOW_MARGIN_THRESHOLD = 0.15

# Margin at or above which confidence takes no discount at all.
MARGIN_FULL = 0.20
# Maximum proportion of the fused score a zero margin can remove. At margin 0 a
# fused 0.9 becomes 0.585 — still high, but no longer auto-accepted, which is the
# intent: ambiguity should cost confidence without erasing evidence.
MARGIN_PENALTY = 0.35

# An exact glossary hit on a column whose values also fit the target is the one
# case worth short-circuiting: it is both the most common and the least arguable.
EXACT_CONFIDENCE_FLOOR = 0.92


@dataclass
class Cell:
    """Fused evidence for one (column, target) pair."""

    lexical: float
    semantic: float
    fingerprint: float
    fused: float
    contributing: list[str]
    glossary_phrase: str
    lexical_exact: bool
    semantic_cosine: float
    dominant_shape: str
    dominant_share: float
    fingerprint_abstained: bool


@dataclass
class ColumnEvidence:
    header: str
    index: int
    values: list[Any]
    cells: dict[str, Cell]

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(
            ((name, cell.fused) for name, cell in self.cells.items()),
            key=lambda pair: -pair[1],
        )


def _fuse(lexical: float, semantic: float, fingerprint: float, *,
          semantic_available: bool, fingerprint_abstained: bool) -> tuple[float, list[str]]:
    """Weighted mean over participating channels only.

    A channel that could not form an opinion is removed from the denominator
    rather than contributing a zero, so a missing embedding model degrades the
    *resolution* of the decision without biasing it downward.
    """
    parts: list[tuple[str, float]] = [("lexical", lexical)]
    if semantic_available:
        parts.append(("semantic", semantic))
    if not fingerprint_abstained:
        parts.append(("fingerprint", fingerprint))

    total_weight = sum(CHANNEL_WEIGHTS[name] for name, _ in parts)
    if total_weight <= 0:
        return 0.0, []
    fused = sum(CHANNEL_WEIGHTS[name] * score for name, score in parts) / total_weight
    return round(fused, 4), [name for name, _ in parts]


def build_evidence(
    headers: list[str],
    columns: list[list[Any]],
) -> list[ColumnEvidence]:
    """Score every column against every target on all three channels."""
    semantic_available = sem_channel.is_available()
    semantic_all = sem_channel.score_headers(headers)

    evidence: list[ColumnEvidence] = []
    for index, header in enumerate(headers):
        lexical_hits = lex_channel.score_header(header)
        semantic_hits = semantic_all[header]
        values = columns[index] if index < len(columns) else []
        fingerprint_hits = fp_channel.score_column(values)
        abstained = fingerprint_hits[next(iter(TARGETS))].abstained

        cells: dict[str, Cell] = {}
        for name in TARGETS:
            lex, sem, fin = lexical_hits[name], semantic_hits[name], fingerprint_hits[name]
            fused, contributing = _fuse(
                lex.score, sem.score, fin.score,
                semantic_available=semantic_available,
                fingerprint_abstained=abstained,
            )
            cells[name] = Cell(
                lexical=lex.score,
                semantic=sem.score,
                fingerprint=fin.score,
                fused=fused,
                contributing=contributing,
                glossary_phrase=lex.matched_phrase if lex.attributable else "",
                lexical_exact=lex.exact,
                semantic_cosine=sem.raw_cosine,
                dominant_shape=fin.dominant_shape,
                dominant_share=fin.dominant_share,
                fingerprint_abstained=abstained,
            )
        evidence.append(ColumnEvidence(header=header, index=index, values=values, cells=cells))
    return evidence


@dataclass
class Assignment:
    column_index: int
    header: str
    target: Optional[str]
    fused: float
    margin: float
    runner_up: Optional[str]
    runner_up_score: float
    cell: Optional[Cell]
    # The column's own best target regardless of what the global assignment gave
    # it. When these differ, the Hungarian solver traded this column's preference
    # away, and the rationale says so explicitly.
    preferred_target: Optional[str]
    preferred_score: float
    rejected_below_floor: bool


def _has_admissible_evidence(cell: Cell) -> bool:
    """Whether anything here actually *identifies* the target, as opposed to
    merely failing to contradict it.

    The fused score cannot answer this, and measuring it on real SOVs showed
    why: across six files, correct mappings ran as low as 0.486 fused while
    incorrect ones reached 0.636. No threshold separates those populations,
    because three weak signals average to a number that looks like evidence.

    What *did* separate them was the kind of evidence. Every correct mapping in
    that overlap region had a lexical score of 1.00 — the header genuinely names
    the field. Every incorrect one had lexical below 0.90 and semantic below
    0.70, and was carried by the fingerprint.

    That asymmetry is not a coincidence, it is the fingerprint channel working
    as designed. Shape compatibility is shared: five targets accept
    `place_name`, four accept `currency` (see targets.py). So a fingerprint of
    1.00 says "this could be Other" and never "this is Other" — the channel can
    rule out, not rule in. Left to itself it hands `Longitude` to `Other` and
    `Basement Yes/No` to `Fire Sprinklers` with perfect confidence in the shape
    and none whatsoever in the field.

    So a target is admissible only on header evidence: the glossary recognised
    the header, or the embedding put it near the field's description. The
    fingerprint still contributes to the score and can still veto by scoring
    zero; it just cannot be the sole reason a column is claimed.
    """
    return (
        cell.lexical >= ADMISSIBLE_LEXICAL
        or cell.semantic >= ADMISSIBLE_SEMANTIC
    )


def solve(evidence: list[ColumnEvidence]) -> list[Assignment]:
    """Globally optimal one-to-one column→target assignment."""
    target_names = list(TARGETS)
    if not evidence:
        return []

    # linear_sum_assignment minimises, so costs are negated scores. Rectangular
    # input is fine: with more columns than targets, the surplus columns go
    # unassigned, which is exactly right for a 30-column source sheet.
    #
    # Inadmissible cells are zeroed *here*, before the matching, not filtered out
    # after it. Rejecting them afterwards is too late: on a real file `Bldg #`
    # outscored `Loc #` for `Reference` (0.647 vs 0.410) on the strength of a
    # fingerprint, won the target, and was then rejected for having no header
    # evidence — leaving `Reference` unmapped with the column that genuinely named
    # it sitting unassigned. A cell the floor will refuse must not be allowed to
    # compete for the target in the first place.
    #
    # Zeroed rather than forbidden outright: `linear_sum_assignment` raises on an
    # infeasible matrix, and a column with no admissible target anywhere is a
    # normal case here, not an error.
    cost = np.array(
        [
            [
                -ev.cells[name].fused if _has_admissible_evidence(ev.cells[name]) else 0.0
                for name in target_names
            ]
            for ev in evidence
        ],
        dtype=float,
    )
    row_idx, col_idx = linear_sum_assignment(cost)
    chosen: dict[int, str] = {int(r): target_names[int(c)] for r, c in zip(row_idx, col_idx)}

    assignments: list[Assignment] = []
    for position, ev in enumerate(evidence):
        ranked = ev.ranked()
        preferred, preferred_score = ranked[0]
        target = chosen.get(position)

        if target is None:
            assignments.append(
                Assignment(
                    column_index=ev.index, header=ev.header, target=None, fused=0.0,
                    margin=0.0, runner_up=None, runner_up_score=0.0, cell=None,
                    preferred_target=preferred, preferred_score=preferred_score,
                    rejected_below_floor=False,
                )
            )
            continue

        cell = ev.cells[target]
        fused = cell.fused
        # Runner-up is the best *other* target for this column, which is what
        # "how close was this call" actually means.
        others = [(name, score) for name, score in ranked if name != target]
        runner_up, runner_up_score = others[0] if others else (None, 0.0)
        below_floor = fused < MIN_ASSIGNMENT_SCORE or not _has_admissible_evidence(cell)

        assignments.append(
            Assignment(
                column_index=ev.index,
                header=ev.header,
                target=None if below_floor else target,
                fused=fused,
                margin=round(fused - runner_up_score, 4),
                runner_up=runner_up,
                runner_up_score=runner_up_score,
                cell=ev.cells[target],
                preferred_target=preferred,
                preferred_score=preferred_score,
                rejected_below_floor=below_floor,
            )
        )
    return assignments


def margin_adjusted_confidence(fused: float, margin: float) -> float:
    """Discount the fused score by how contested the decision was.

    A 0.9 that beat its runner-up by 0.4 and a 0.9 that beat it by 0.01 are not
    the same claim, and reporting both as 0.9 would be the single most misleading
    thing this agent could do.
    """
    closeness = 1.0 - min(max(margin, 0.0) / MARGIN_FULL, 1.0)
    return round(fused * (1.0 - MARGIN_PENALTY * closeness), 4)
