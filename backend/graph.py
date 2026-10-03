"""The LangGraph StateGraph: 4 agents + 1 human review interrupt gate, one shared SOVState, a gate before each
agent (FR-ORC-01, FR-ORC-02, FR-ORC-03)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from backend.agents.mapping.agent import schema_mapping
from backend.agents.quality.agent import data_quality
from backend.agents.sheet_intel.agent import sheet_intelligence
from backend.agents.transform.agent import transformation
from backend.memory.store import get_memory_store
from backend.state.sov_state import Decision, SOVState, SourceInfo

AGENT_NODES = ("sheet_intelligence", "schema_mapping", "data_quality", "human_review", "transformation")


def human_review(state: SOVState) -> dict[str, Any]:
    """Human-in-the-loop review gate (C-01, FR-ORC-03).

    Pauses graph execution via interrupt() if any recommendations are pending.
    When resumed with decisions, updates recommendation statuses and records decisions.
    """
    recs = state.recommendations or []
    pending = [r for r in recs if r.get("status") == "pending"]
    if not pending:
        return {}

    user_decisions = interrupt({
        "type": "human_review_required",
        "pending_count": len(pending),
        "recommendations": recs,
    })

    decisions_list = list(state.decisions or [])
    updated_recs = [dict(r) for r in recs]
    rec_map = {r["id"]: r for r in updated_recs}
    store = get_memory_store()

    if isinstance(user_decisions, list):
        for d in user_decisions:
            rec_id = d.get("rec_id")
            action = d.get("action")
            note = d.get("note", "")
            by = d.get("by", "human_reviewer")

            if rec_id in rec_map:
                target_rec = rec_map[rec_id]
                if action == "accept":
                    target_rec["status"] = "approved"
                    store.record_decision(
                        source_column=target_rec.get("source_column", ""),
                        target_field=target_rec.get("field", ""),
                        approved=True,
                        notes=note,
                    )
                elif action == "reject":
                    target_rec["status"] = "rejected"
                    store.record_decision(
                        source_column=target_rec.get("source_column", ""),
                        target_field=target_rec.get("field", ""),
                        approved=False,
                        notes=note,
                    )
                elif action == "edit":
                    target_rec["status"] = "edited"
                    if d.get("edited_op"):
                        target_rec["op"] = d["edited_op"]
                    store.record_decision(
                        source_column=target_rec.get("source_column", ""),
                        target_field=target_rec.get("field", ""),
                        approved=True,
                        notes=note,
                    )

                decisions_list.append(
                    Decision(
                        rec_id=rec_id,
                        action=action,
                        note=note,
                        by=by,
                    )
                )

    return {
        "recommendations": updated_recs,
        "decisions": decisions_list,
    }


def build_graph(checkpointer: Any = None):
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(SOVState)
    graph.add_node("sheet_intelligence", sheet_intelligence)
    graph.add_node("schema_mapping", schema_mapping)
    graph.add_node("data_quality", data_quality)
    graph.add_node("human_review", human_review)
    graph.add_node("transformation", transformation)

    graph.set_entry_point("sheet_intelligence")
    graph.add_edge("sheet_intelligence", "schema_mapping")
    graph.add_edge("schema_mapping", "data_quality")
    graph.add_edge("data_quality", "human_review")
    graph.add_edge("human_review", "transformation")
    graph.add_edge("transformation", END)

    return graph.compile(checkpointer=checkpointer)


CHECKPOINTER = MemorySaver()
COMPILED_GRAPH = build_graph(CHECKPOINTER)


def initial_state(path: Path | str, file_name: Optional[str] = None) -> SOVState:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return SOVState(
        source=SourceInfo(
            file_name=file_name or path.name,
            sha256=digest.hexdigest(),
            file_path=str(path),
        )
    )


def run_pipeline(path: Path | str, file_name: Optional[str] = None) -> SOVState:
    config = {"configurable": {"thread_id": "cli_run"}}
    init_st = initial_state(path, file_name)

    result = COMPILED_GRAPH.invoke(init_st, config=config)

    # Auto-approve all recommendations if CLI run
    state_obj = SOVState.model_validate(result) if isinstance(result, dict) else result
    if state_obj.recommendations and any(r.get("status") == "pending" for r in state_obj.recommendations):
        decisions = [
            {"rec_id": r["id"], "action": "accept", "note": "CLI auto-approve", "by": "cli_user"}
            for r in state_obj.recommendations
            if r.get("status") == "pending"
        ]
        result = COMPILED_GRAPH.invoke(Command(resume=decisions), config=config)

    if isinstance(result, SOVState):
        return result
    return SOVState.model_validate(result)
