"""SOVState: the single typed state object shared by all four agents (SRS 5.2)."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

STATE_VERSION = 8

SheetClass = Literal["Primary", "Secondary", "Reject"]
Severity = Literal["Low", "Medium", "High"]


class SourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_name: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    sheets: int = Field(default=0, ge=0)
    # Local-disk pointer. Storage is local only (SRS 2.4) and Agent 4 must read
    # the untouched source to apply ops on a copy (FR-TRN-02).
    file_path: Optional[str] = None


class SheetManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sheet: str
    class_: SheetClass = Field(alias="class")
    confidence: float = Field(ge=0.0, le=1.0)
    # 0-based index into the raw sheet grid, so pandas can read it directly with
    # header=header_row. None when no header row could be found.
    header_row: Optional[int] = Field(default=None, ge=0)
    reasons: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    factor_scores: dict[str, float] = Field(default_factory=dict)
    headers: list[str] = Field(default_factory=list)
    data_start_row: Optional[int] = Field(default=None, ge=0)
    data_rows: int = Field(default=0, ge=0)
    composite_header: bool = False


class Issue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: str
    field: Optional[str] = None
    rows: list[int] = Field(default_factory=list)
    severity: Severity


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rec_id: str
    action: Literal["accept", "reject", "edit"]
    note: Optional[str] = None
    by: str
    at: str


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str
    event: Literal["started", "finished", "error"]
    ms: int = Field(default=0, ge=0)
    detail: Optional[str] = None


class SOVState(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True)

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    version: int = STATE_VERSION
    source: SourceInfo
    manifest: list[SheetManifestEntry] = Field(default_factory=list)
    mapping: dict[str, Any] = Field(default_factory=dict)
    # Agent 3's report (`QualityBlock`). Replaced rather than appended, like
    # `mapping`: there is one data-quality assessment of a file, and a re-run
    # supersedes the previous one instead of accumulating beside it.
    quality: dict[str, Any] = Field(default_factory=dict)
    issues: Annotated[list[Issue], operator.add] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    # Append-only, like issues and trace. Agent 2 records every LLM proposal it
    # discarded here and Agent 4 will append the ops it applies; neither must be
    # able to erase the other's record, so an agent returning audit entries adds
    # to the log rather than replacing it.
    audit: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
    trace: Annotated[list[TraceEvent], operator.add] = Field(default_factory=list)

    def primary_sheet(self) -> Optional[SheetManifestEntry]:
        return next((e for e in self.manifest if e.class_ == "Primary"), None)
