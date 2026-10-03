"""Header-row detection and composite header construction (FR-SHT-03, FR-SHT-04)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.agents.sheet_intel.cells import (
    cell_kind,
    looks_like_label,
    match_target,
    normalise,
    vocabulary_matches,
)
from backend.ingest.loader import LoadedSheet

MAX_HEADER_SCAN = 30
# Body rows sampled when measuring type consistency. See `type_consistency_below`.
TYPE_CONSISTENCY_SAMPLE = 400
COMPOSITE_SEPARATOR = " › "
MIN_HEADER_SCORE = 0.38
SUB_HEADER_SPARSITY = 0.6

# Distinct-value share below which a candidate row is rejected outright. A real
# header names different things in every column; a two-column header is the
# smallest legitimate case, so this stays well below 0.5 to avoid excluding
# genuinely repetitive-but-valid headers like 'Bldg / Bldg'.
MIN_HEADER_UNIQUENESS = 0.25

CANDIDATE_WEIGHTS = {
    "label_density": 0.25,
    "fill": 0.10,
    "uniqueness": 0.15,
    "type_consistency_below": 0.25,
    "vocabulary_overlap": 0.25,
}


@dataclass
class HeaderDetection:
    header_row: int
    data_start_row: int
    headers: list[str]
    score: float
    composite: bool
    factor_scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    matched_targets: set[str] = field(default_factory=set)


def _non_null_indexes(row: list[Any]) -> list[int]:
    return [i for i, v in enumerate(row) if v is not None]


def type_consistency_below(
    sheet: LoadedSheet, data_start_row: int, columns: list[int]
) -> float:
    """Mean per-column share of the dominant non-empty cell kind below the header.

    Sampled, not exhaustive. This function is called once per candidate header row
    — up to MAX_HEADER_SCAN times per sheet — and originally scanned the whole body
    each time. On a 5,549-row x 39-column tab that was 12.5 million `cell_kind`
    calls and 25 of the 31 seconds the sheet took to score, which is the real reason
    the loader needed a row ceiling at all.

    What is being measured is a *proportion*: the share of a column that is the
    dominant type. A few hundred rows pins that down to within a couple of percent,
    and the factor is weighted 0.20 against four others, so the extra precision from
    reading every row could never change a classification.

    Strided rather than head-sampled, because the first rows of a schedule are not
    representative — subtotal blocks, a block of one region, or a run of blanks at
    the top would all skew a head sample. A stride spans the whole sheet.
    """
    body = sheet.grid[data_start_row:]
    if not body or not columns:
        return 0.0

    if len(body) > TYPE_CONSISTENCY_SAMPLE:
        stride = -(-len(body) // TYPE_CONSISTENCY_SAMPLE)  # ceil
        body = body[::stride]

    scores: list[float] = []
    for col in columns:
        kinds = [
            kind
            for row in body
            if col < len(row) and (kind := cell_kind(row[col])) != "empty"
        ]
        if not kinds:
            continue
        dominant = Counter(kinds).most_common(1)[0][1]
        scores.append(dominant / len(kinds))
    return sum(scores) / len(scores) if scores else 0.0


def _score_candidate(sheet: LoadedSheet, row_idx: int) -> Optional[dict[str, float]]:
    row = sheet.grid[row_idx]
    filled = _non_null_indexes(row)
    if len(filled) < 2:
        return None

    values = [row[i] for i in filled]
    uniqueness = len({normalise(v) for v in values}) / len(values)

    # A row that is nearly all one repeated string is not a header, whatever else
    # it scores. This is a disqualifier rather than a weighted term because on a
    # real SOV it has to beat four other factors that a merged footnote maxes out:
    # a single sentence spanning 27 merged columns is forward-filled into 27
    # identical cells, giving label_density 1.00, fill 0.96 and
    # type_consistency_below 1.00 against a true header row's 0.839. Uniqueness
    # was the only factor that spotted it, and at weight 0.15 it lost by 0.013.
    if uniqueness < MIN_HEADER_UNIQUENESS:
        return None

    # Deduplicate before measuring label-ness and vocabulary, so forward-filled
    # merged cells contribute one cell's worth of evidence instead of one per
    # column they happen to span.
    distinct: dict[str, Any] = {}
    for value in values:
        distinct.setdefault(normalise(value), value)
    unique_values = list(distinct.values())

    label_density = sum(1 for v in unique_values if looks_like_label(v)) / len(unique_values)
    fill = len(filled) / sheet.n_cols
    vocabulary = len(vocabulary_matches(unique_values)) / len(unique_values)
    consistency = type_consistency_below(sheet, row_idx + 1, filled)

    factors = {
        "label_density": label_density,
        "fill": fill,
        "uniqueness": uniqueness,
        "type_consistency_below": consistency,
        "vocabulary_overlap": vocabulary,
    }
    factors["score"] = sum(CANDIDATE_WEIGHTS[k] * v for k, v in factors.items())
    return factors


def _is_sub_header(sheet: LoadedSheet, header_row: int) -> bool:
    if header_row + 1 >= sheet.n_rows:
        return False

    top_filled = _non_null_indexes(sheet.grid[header_row])
    sub_filled = _non_null_indexes(sheet.grid[header_row + 1])
    if not sub_filled or not top_filled:
        return False
    if len(sub_filled) > SUB_HEADER_SPARSITY * len(top_filled):
        return False
    if not all(looks_like_label(sheet.grid[header_row + 1][i]) for i in sub_filled):
        return False

    spans = sheet.horizontal_spans_on_row(header_row)
    under_banner = any(
        span.col_start <= i <= span.col_end for span in spans for i in sub_filled
    )
    names_a_target = any(match_target(sheet.grid[header_row + 1][i]) for i in sub_filled)
    return under_banner or names_a_target


def _build_headers(sheet: LoadedSheet, header_row: int, composite: bool) -> list[str]:
    top = sheet.grid[header_row]
    sub = sheet.grid[header_row + 1] if composite else [None] * sheet.n_cols

    names: list[str] = []
    for col in range(sheet.n_cols):
        top_value = top[col] if col < len(top) else None
        sub_value = sub[col] if col < len(sub) else None

        if sub_value is not None and top_value is not None:
            if normalise(top_value) == normalise(sub_value):
                name = str(sub_value)
            else:
                name = f"{top_value}{COMPOSITE_SEPARATOR}{sub_value}"
        elif sub_value is not None:
            name = str(sub_value)
        elif top_value is not None:
            name = str(top_value)
        else:
            name = f"Unnamed_{col + 1}"
        names.append(" ".join(str(name).split()))

    seen: Counter[str] = Counter()
    deduped: list[str] = []
    for name in names:
        seen[name] += 1
        deduped.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return deduped


def _first_populated_row(sheet: LoadedSheet, start: int) -> int:
    for idx in range(start, sheet.n_rows):
        if _non_null_indexes(sheet.grid[idx]):
            return idx
    return sheet.n_rows


def detect_header(sheet: LoadedSheet) -> Optional[HeaderDetection]:
    if sheet.n_rows == 0 or sheet.n_cols == 0:
        return None

    candidates: dict[int, dict[str, float]] = {}
    for row_idx in range(min(MAX_HEADER_SCAN, sheet.n_rows)):
        factors = _score_candidate(sheet, row_idx)
        if factors:
            candidates[row_idx] = factors

    if not candidates:
        return None
    best_row = max(candidates, key=lambda r: candidates[r]["score"])
    if candidates[best_row]["score"] < MIN_HEADER_SCORE:
        return None

    # A sub-header row scores well on its own: it is sparse, all labels, and sits
    # directly above clean data. Climb to the top row of the header block so the
    # banner is not mistaken for a title.
    while best_row - 1 in candidates and _is_sub_header(sheet, best_row - 1):
        best_row -= 1

    best_factors = candidates[best_row]
    composite = _is_sub_header(sheet, best_row)
    headers = _build_headers(sheet, best_row, composite)
    data_start_row = _first_populated_row(sheet, best_row + (2 if composite else 1))
    matched = set(vocabulary_matches(headers).values())

    reasons = [
        f"header row detected at row {best_row + 1} (0-based {best_row})"
        + (f", below a {best_row}-row block of title/blank rows" if best_row else ""),
        f"{len(matched)}/17 target fields recognised in the header text",
    ]
    if composite:
        banner_width = sum(s.width for s in sheet.horizontal_spans_on_row(best_row))
        reasons.append(
            "two-row header combined into composite names"
            + (f" across {banner_width} banner columns" if banner_width else "")
        )

    factor_scores = {k: round(best_factors[k], 3) for k in CANDIDATE_WEIGHTS}
    return HeaderDetection(
        header_row=best_row,
        data_start_row=data_start_row,
        headers=headers,
        score=round(best_factors["score"], 3),
        composite=composite,
        factor_scores=factor_scores,
        reasons=reasons,
        matched_targets=matched,
    )
