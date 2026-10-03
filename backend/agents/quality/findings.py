"""What a rule reports, before it becomes a recommendation.

A `Finding` is one cell (or one row) that failed one rule. Rules emit them
atomically and never group them, because FR-DQ-08 asks for "one recommendation
per issue *group*" and grouping is a judgement about presentation, not about data
quality. Keeping the two apart means a rule stays a pure function of the frame,
and `recommend.py` owns every decision about how many recommendations a reviewer
sees.

Three things a finding must carry and does:

- **`before`, and `after` when code can compute it.** FR-DQ-09 requires the
  worked example to come from a real row, so the raw cell text is captured at the
  point of detection rather than re-derived later. `after=None` paired with an
  `op` means the cell is deliberately left empty; `op=None` means code has no
  safe correction and the row goes to a human.
- **Severity from the catalogue, never from the rule.** `make` looks the rule up
  in `RULE_CATALOGUE` and asks it, so a rule cannot promote its own findings.
- **`evidence`.** The facts the engine used, in a form the LLM reasoner's
  citations can be checked against. The reasoner may rewrite prose; it may not
  introduce a fact that is not in here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from backend.agents.quality.models import RULE_CATALOGUE, TransformOp
from backend.state.sov_state import Severity


@dataclass(frozen=True)
class Finding:
    """One cell, or one row, that failed one SRS 9.2 rule."""

    rule: str
    # 1-based spreadsheet row, so it can be read straight out of the source file.
    row: int
    # The target field, or None when the finding is about the whole row (DQ-15).
    field: Optional[str]
    severity: Severity
    # Plain English, specific to this cell. Becomes part of the group's rationale.
    detail: str
    # The cell as it reads in the source, verbatim.
    before: str
    # What code worked out the corrected value to be. None means the cell ends up
    # empty, which only `set_null` and `exclude_row` may ask for.
    after: Optional[str] = None
    # The SRS 9.1 op that would fix it, or None when nothing in the whitelist can
    # and a human has to decide.
    op: Optional[TransformOp] = None
    source_column: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        return self.op is not None


def make(
    rule: str,
    row: int,
    field_name: Optional[str],
    *,
    detail: str,
    before: Any,
    after: Optional[str] = None,
    op: Optional[TransformOp] = None,
    source_column: Optional[str] = None,
    evidence: Optional[dict[str, Any]] = None,
) -> Finding:
    """Build a finding, taking severity from the SRS rather than the caller."""
    spec = RULE_CATALOGUE.get(rule)
    if spec is None:  # pragma: no cover - a typo in a rule module
        raise RuntimeError(f"rule '{rule}' is not in the SRS 9.2 catalogue")
    return Finding(
        rule=rule,
        row=row,
        field=field_name,
        severity=spec.severity_for(field_name),
        detail=detail,
        before=render(before),
        after=after,
        op=op,
        source_column=source_column,
        evidence=evidence or {},
    )


def render(raw: Any) -> str:
    """A cell as a reviewer would describe it.

    An empty cell reads as `(empty)` rather than as the empty string, because a
    before/after preview showing nothing on the left is indistinguishable from a
    rendering bug.
    """
    if raw is None:
        return "(empty)"
    text = str(raw)
    return text if text.strip() else "(empty)"


__all__ = ["Finding", "make", "render"]
