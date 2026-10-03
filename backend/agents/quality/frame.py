"""The data Agent 3 validates: the Primary sheet, viewed through Agent 2's mapping.

Two jobs, kept apart on purpose.

**Reading.** `build_frame` re-reads the workbook from disk and arranges it by
*target field* rather than by source header, because every rule in SRS 9.2 is
written about target fields. It reads the sheet itself rather than borrowing
Agent 2's reader: NFR-MNT-01 requires that no agent import another, and
`SOVState` is the only channel between them.

**Cleaning.** `clean_amount` and `clean_integer` turn a raw cell into a typed
value *or* into a named reason it could not be typed. This is the step SRS 9.2
calls "after cleaning" in DQ-02, and it is deliberately separate from validation:
`"$2,100,000.00"` is a formatting issue (DQ-10) on a cell that cleans perfectly,
while `"1980's"` is a cell that does not clean at all (DQ-03). Collapsing the two
would report a reviewer's easiest fix and their hardest problem identically.

Nothing here invents a value. A cell that cannot be cleaned becomes null and is
reported; it is never replaced by a plausible number (C-02).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Optional

import pandas as pd

from backend.agents.quality.models import INTEGER_FIELDS, VALUE_FIELDS
from backend.ingest.loader import load_workbook_file
from backend.state.sov_state import SOVState
from backend.state.target_schema import TARGET_FIELDS, TARGET_FIELD_NAMES

# Strings that mean "no value" rather than a value. A reviewer typing "N/A" into a
# spreadsheet has reported absence, so treating it as text would turn a complete
# column into a column full of type errors.
NULL_TOKENS = frozenset(
    {"", "na", "n/a", "n.a.", "none", "null", "nil", "-", "--", "---", "?",
     "tbd", "tba", "unknown", "unk", "n/k", "not known", "#n/a", "#ref!",
     "#value!", "#div/0!"}
)

# Characters whose presence means the cell was written for a human, not for a
# type system: currency symbols and thousands separators (DQ-10).
_FORMATTING_CHARS = re.compile(r"[$£€¥,\s]")
_PARENS_NEGATIVE = re.compile(r"^\((.*)\)$")
_NUMERIC = re.compile(r"^-?\d+(\.\d+)?$")

CleanStatus = Literal["empty", "ok", "reformatted", "unparseable", "fractional"]


@dataclass(frozen=True)
class CleanResult:
    """A cleaned cell, and the honest account of what cleaning it required."""

    value: Optional[float]
    status: CleanStatus
    # Only set when status is "reformatted": the characters actually removed, so a
    # recommendation can quote them instead of saying "formatting".
    removed: str = ""

    @property
    def usable(self) -> bool:
        return self.value is not None


def is_null(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, float) and pd.isna(raw):
        return True
    if isinstance(raw, str):
        return raw.strip().lower() in NULL_TOKENS
    return False


def clean_amount(raw: Any) -> CleanResult:
    """A currency-ish cell to a float, or a named reason it is not one.

    Accepts the three ways a spreadsheet writes money — `1250000`, `$1,250,000.00`
    and `(45,000)` for a negative — and nothing else. A negative is *returned*
    rather than rejected; DQ-04 decides what a negative amount means, and that is
    not this function's call to make.
    """
    if is_null(raw):
        return CleanResult(None, "empty")
    if isinstance(raw, bool):
        return CleanResult(None, "unparseable")
    if isinstance(raw, (int, float)):
        return CleanResult(float(raw), "ok")
    if isinstance(raw, (datetime, date)):
        return CleanResult(None, "unparseable")

    text = str(raw).strip()
    negative = False
    parens = _PARENS_NEGATIVE.match(text)
    if parens:
        negative, text = True, parens.group(1)

    removed = "".join(sorted(set(_FORMATTING_CHARS.findall(text))))
    stripped = _FORMATTING_CHARS.sub("", text)
    if stripped.startswith("-"):
        negative, stripped = True, stripped[1:]

    if not _NUMERIC.match(stripped):
        return CleanResult(None, "unparseable")

    value = float(stripped)
    if negative:
        value = -value
    # Parentheses are a formatting convention in their own right, so a cell that
    # needed un-bracketing counts as reformatted even if it held no symbols.
    if removed or parens:
        return CleanResult(value, "reformatted", removed=removed + ("()" if parens else ""))
    return CleanResult(value, "ok")


def clean_integer(raw: Any) -> CleanResult:
    """An integer-field cell to a whole number, or a named reason it is not one.

    `3.0` is a whole number written as a float and cleans to 3. `1.5` and `"1+B"`
    do not clean, and are distinguished: the first is fractional (DQ-03 on a value
    the engine *can* read) and the second is unparseable (DQ-03 on one it cannot).
    Both are DQ-03, but only one of them has a `to_int` that would work.
    """
    if isinstance(raw, (datetime, date)) and not isinstance(raw, bool):
        # A date in Year Built is a real pattern: Excel turns a bare year into a
        # date on import. The year is recoverable from the value itself, so this
        # is a formatting problem and not a missing one.
        return CleanResult(float(raw.year), "reformatted", removed="date")

    amount = clean_amount(raw)
    if amount.value is None:
        return amount
    if not float(amount.value).is_integer():
        return CleanResult(amount.value, "fractional", removed=amount.removed)
    return CleanResult(amount.value, amount.status, removed=amount.removed)


def clean_text(raw: Any) -> Optional[str]:
    if is_null(raw):
        return None
    if isinstance(raw, (datetime, date)) and not isinstance(raw, bool):
        return raw.isoformat()
    text = str(raw).strip()
    return text or None


@dataclass
class MappedFrame:
    """The Primary sheet arranged by target field, with a typed companion frame."""

    sheet_name: str
    header_row: Optional[int]
    data_start_row: int
    n_rows: int
    # 1-based spreadsheet row per data row, so every reported row number can be
    # looked up in the source file without arithmetic.
    sheet_rows: list[int]
    # target field -> raw cell values, in row order.
    raw: dict[str, list[Any]] = field(default_factory=dict)
    source_column: dict[str, str] = field(default_factory=dict)
    source_index: dict[str, int] = field(default_factory=dict)
    # Source headers Agent 2 left unmapped, kept because DQ-16 needs to find a TIV
    # column that by definition has no target field of its own.
    unmapped: dict[str, list[Any]] = field(default_factory=dict)
    # Per-field cleaning outcome, parallel to `raw`. Built once; every rule that
    # needs a typed value reads it rather than re-parsing.
    cleaned: dict[str, list[CleanResult]] = field(default_factory=dict)
    # The typed frame the pandera schema validates. Columns are target field names.
    typed: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def mapped_fields(self) -> list[str]:
        """Mapped target fields, in the frozen SRS 5.1 order."""
        return [name for name in TARGET_FIELD_NAMES if name in self.raw]

    def row_label(self, index: int) -> int:
        return self.sheet_rows[index]

    def values(self, field_name: str) -> list[Any]:
        return self.raw.get(field_name, [])

    def cell(self, field_name: str, index: int) -> Any:
        column = self.raw.get(field_name)
        return column[index] if column and index < len(column) else None

    def index_of_row(self, sheet_row: int) -> int:
        return self.sheet_rows.index(sheet_row)


_FIELD_DTYPE = {f.name: f.dtype for f in TARGET_FIELDS}


def _clean_for(field_name: str, raw: Any) -> CleanResult:
    dtype = _FIELD_DTYPE[field_name]
    if dtype == "float":
        return clean_amount(raw)
    if dtype == "integer":
        return clean_integer(raw)
    text = clean_text(raw)
    return CleanResult(None, "empty") if text is None else CleanResult(None, "ok")


def build_frame(state: SOVState) -> MappedFrame:
    """Read the Primary sheet and arrange it by target field.

    The mapping block is the authority on which source column holds which target
    field, and on which rows are data. Re-deriving either here would risk Agent 3
    validating a different sheet from the one Agent 2 mapped.
    """
    entry = state.primary_sheet()
    if entry is None:  # pragma: no cover - the contract gate guarantees this
        raise RuntimeError("no Primary sheet in manifest")

    workbook = load_workbook_file(state.source.file_path, state.source.file_name)
    sheet = next((s for s in workbook.sheets if s.name == entry.sheet), None)
    if sheet is None:
        raise RuntimeError(
            f"sheet '{entry.sheet}' named Primary by Agent 1 is not in the workbook"
        )

    start = entry.data_start_row if entry.data_start_row is not None else 0
    body = sheet.grid[start:]
    width = len(entry.headers)
    columns = [
        [row[i] if i < len(row) else None for row in body] for i in range(width)
    ]
    n_rows = len(body)

    frame = MappedFrame(
        sheet_name=entry.sheet,
        header_row=entry.header_row,
        data_start_row=start,
        n_rows=n_rows,
        # +1 converts the 0-based grid index to the row number a spreadsheet shows.
        sheet_rows=[start + i + 1 for i in range(n_rows)],
    )

    for mapping in state.mapping.get("mappings") or []:
        index = mapping.get("source_index")
        header = mapping.get("source_column") or f"column_{index}"
        if index is None or index >= width:
            continue
        target = mapping.get("target")
        if target is None:
            frame.unmapped[header] = columns[index]
            continue
        frame.raw[target] = columns[index]
        frame.source_column[target] = header
        frame.source_index[target] = index

    for target, values in frame.raw.items():
        frame.cleaned[target] = [_clean_for(target, v) for v in values]

    frame.typed = _build_typed(frame)
    return frame


def frame_from_columns(
    columns: dict[str, list[Any]],
    *,
    sheet_name: str = "constructed",
    first_sheet_row: int = 2,
    unmapped: Optional[dict[str, list[Any]]] = None,
    source_column: Optional[dict[str, str]] = None,
) -> MappedFrame:
    """A `MappedFrame` built from in-memory columns rather than from a workbook.

    Exists for two callers, both legitimate and both deliberately explicit about
    it. The tests use it to drive one rule at a time with a column shaped to
    trigger exactly that rule, and DQ-16 needs it at all because the rule's real
    trigger does not occur in the sample corpus: every TIV column in the four
    real workbooks is an Excel `=SUM(...)` formula whose cached result is absent
    from the saved file, so `find_tiv_column` correctly finds nothing to
    reconcile and the rule cannot be exercised against real data.

    It routes through the same `_clean_for` and `_build_typed` as `build_frame`,
    which is the point: a test harness that re-implemented the cleaner would be
    testing the harness. That is not a hypothetical concern here — the row-
    alignment bug in `_build_typed` was found precisely because a throwaway
    harness built its frame differently from production and the two disagreed.
    """
    lengths = {len(v) for v in columns.values()} | {
        len(v) for v in (unmapped or {}).values()
    }
    if len(lengths) > 1:
        raise ValueError(f"columns have differing lengths: {sorted(lengths)}")
    n_rows = lengths.pop() if lengths else 0

    frame = MappedFrame(
        sheet_name=sheet_name,
        header_row=first_sheet_row - 2 if first_sheet_row >= 2 else None,
        data_start_row=first_sheet_row - 1,
        n_rows=n_rows,
        sheet_rows=[first_sheet_row + i for i in range(n_rows)],
        raw=dict(columns),
        unmapped=dict(unmapped or {}),
        source_column=dict(source_column or {name: name for name in columns}),
        source_index={name: i for i, name in enumerate(columns)},
    )
    for target, values in frame.raw.items():
        frame.cleaned[target] = [_clean_for(target, v) for v in values]
    frame.typed = _build_typed(frame)
    return frame


def _build_typed(frame: MappedFrame) -> pd.DataFrame:
    """The typed frame the pandera schema validates.

    Un-cleanable cells become null here. That is not data loss — the raw cell and
    the reason it failed are both still on `frame.cleaned`, and the rules that care
    read them from there. It keeps the schema's job to range and enum validation
    on values that are genuinely of the declared type.
    """
    data: dict[str, Any] = {}
    for name in frame.mapped_fields:
        dtype = _FIELD_DTYPE[name]
        if dtype == "string":
            data[name] = pd.Series(
                [clean_text(v) for v in frame.raw[name]], dtype=object
            )
            continue
        numbers = [
            c.value if c.status in {"ok", "reformatted"} else None
            for c in frame.cleaned[name]
        ]
        if dtype == "integer":
            data[name] = pd.Series(
                [None if n is None else int(n) for n in numbers], dtype="Int64"
            )
        else:
            data[name] = pd.Series(numbers, dtype="Float64")

    index = pd.Index(frame.sheet_rows, name="sheet_row")
    if not data:  # no mapped fields; the gate makes this unreachable in a real run
        return pd.DataFrame(index=index)

    typed = pd.DataFrame(data)
    # Assigning the index after construction *relabels* the rows. Passing
    # `index=` to the constructor would instead have pandas *align* each Series
    # on its own 0-based index, so sheet row 2 would silently take the value from
    # position 2 and the final rows would become null. The difference is invisible
    # in a dtype check and would have corrupted every range check downstream.
    typed.index = index
    return typed


__all__ = [
    "CleanResult",
    "CleanStatus",
    "INTEGER_FIELDS",
    "MappedFrame",
    "NULL_TOKENS",
    "VALUE_FIELDS",
    "build_frame",
    "clean_amount",
    "clean_integer",
    "clean_text",
    "frame_from_columns",
    "is_null",
]
