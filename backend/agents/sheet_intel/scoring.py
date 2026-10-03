"""Sheet scoring and Primary/Secondary/Reject classification (FR-SHT-01, FR-SHT-02)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from backend.agents.sheet_intel.header import (
    HeaderDetection,
    detect_header,
    type_consistency_below,
)
from backend.agents.sheet_intel.loader import LoadedSheet
from backend.state.target_schema import TARGET_FIELD_NAMES

SHEET_WEIGHTS = {
    "header_density": 0.25,
    "type_consistency": 0.20,
    "null_ratio": 0.15,
    "row_continuity": 0.15,
    "vocabulary_overlap": 0.25,
}

PRIMARY_MIN_SCORE = 0.50
SECONDARY_MIN_SCORE = 0.40
SECONDARY_MIN_ROWS = 2
# A clean two-column key/value tab scores well on every structural factor.
# Target-field overlap is the only factor that separates SOV data from any small
# table, so a sheet that recognises no target fields cannot be SOV data.
SECONDARY_MIN_VOCABULARY = 0.15


@dataclass
class SheetScore:
    sheet: LoadedSheet
    detection: Optional[HeaderDetection]
    score: float
    factor_scores: dict[str, float]
    data_rows: int
    reasons: list[str] = field(default_factory=list)


def _data_region_quality(sheet: LoadedSheet, data_start_row: int) -> tuple[float, float, int]:
    """Returns (non-null share, row continuity, populated row count)."""
    body = sheet.grid[data_start_row:]
    if not body:
        return 0.0, 0.0, 0

    cells = 0
    populated_cells = 0
    blank_rows = 0
    for row in body:
        if not any(v is not None for v in row):
            blank_rows += 1
            continue
        for value in row:
            cells += 1
            if value is not None:
                populated_cells += 1

    data_rows = len(body) - blank_rows
    null_ratio = populated_cells / cells if cells else 0.0
    continuity = data_rows / len(body) if body else 0.0
    return null_ratio, continuity, data_rows


def score_sheet(sheet: LoadedSheet) -> SheetScore:
    detection = detect_header(sheet)
    data_start = detection.data_start_row if detection else 0

    null_ratio, continuity, data_rows = _data_region_quality(sheet, data_start)

    if detection:
        header_density = detection.score
        columns = [i for i, name in enumerate(detection.headers) if not name.startswith("Unnamed_")]
        consistency = type_consistency_below(sheet, data_start, columns)
        vocabulary = len(detection.matched_targets) / len(TARGET_FIELD_NAMES)
    else:
        header_density = 0.0
        consistency = type_consistency_below(sheet, data_start, list(range(sheet.n_cols)))
        vocabulary = 0.0

    factors = {
        "header_density": round(header_density, 3),
        "type_consistency": round(consistency, 3),
        "null_ratio": round(null_ratio, 3),
        "row_continuity": round(continuity, 3),
        "vocabulary_overlap": round(vocabulary, 3),
    }
    score = round(sum(SHEET_WEIGHTS[k] * v for k, v in factors.items()), 3)

    reasons = list(detection.reasons) if detection else ["no header row found in the first 30 rows"]
    reasons.append(f"{data_rows} populated data rows, {round(null_ratio * 100)}% of cells filled")
    if detection is None and data_rows:
        reasons.append("content does not look tabular")

    return SheetScore(
        sheet=sheet,
        detection=detection,
        score=score,
        factor_scores=factors,
        data_rows=data_rows,
        reasons=reasons,
    )


def _clamp(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 3)


def classify_sheets(scores: list[SheetScore]) -> list[tuple[SheetScore, str, float]]:
    """Rank sheets and assign exactly one Primary. Confidence is confidence in the
    assigned class, so a clearly non-tabular sheet gets a high Reject confidence."""
    ranked = sorted(scores, key=lambda s: (s.score, s.data_rows), reverse=True)
    eligible = [s for s in ranked if s.data_rows >= 1]
    primary = eligible[0] if eligible else None
    runner_up = next((s for s in eligible[1:]), None)
    margin = (primary.score - runner_up.score) if primary and runner_up else (primary.score if primary else 0.0)

    results: list[tuple[SheetScore, str, float]] = []
    for item in ranked:
        if item is primary:
            item.reasons.insert(0, f"highest sheet score {item.score}, margin {round(margin, 3)} over the runner-up")
            if item.score < PRIMARY_MIN_SCORE:
                item.reasons.append(
                    f"score below the {PRIMARY_MIN_SCORE} Primary threshold; "
                    "best available sheet, reviewer override recommended"
                )
            results.append((item, "Primary", _clamp(item.score + 0.5 * margin)))
        elif (
            item.score >= SECONDARY_MIN_SCORE
            and item.data_rows >= SECONDARY_MIN_ROWS
            and item.detection is not None
            and item.factor_scores["vocabulary_overlap"] >= SECONDARY_MIN_VOCABULARY
        ):
            item.reasons.insert(0, "tabular and SOV-like, but scores below the Primary sheet")
            results.append((item, "Secondary", _clamp(item.score)))
        else:
            if item.factor_scores["vocabulary_overlap"] < SECONDARY_MIN_VOCABULARY:
                item.reasons.insert(0, "recognises too few of the 17 target fields to be SOV data")
            else:
                item.reasons.insert(
                    0, f"score {item.score} is below the {SECONDARY_MIN_SCORE} Secondary threshold"
                )
            # Confidence that this is not SOV data. Driven mainly by the absence
            # of target fields: a tidy key/value tab scores well structurally but
            # we are still sure it is not a location schedule.
            not_sov = 1.0 - (
                0.7 * item.factor_scores["vocabulary_overlap"] + 0.3 * item.score
            )
            results.append((item, "Reject", _clamp(not_sov)))
    return results
