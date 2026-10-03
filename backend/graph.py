"""The LangGraph StateGraph: 4 agents, one shared SOVState, a gate before each
agent (FR-ORC-01, FR-ORC-02)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from backend.agents.mapping.agent import schema_mapping
from backend.agents.quality.agent import data_quality
from backend.agents.sheet_intel.agent import sheet_intelligence
from backend.agents.transform.agent import transformation
from backend.state.sov_state import SOVState, SourceInfo

AGENT_NODES = ("sheet_intelligence", "schema_mapping", "data_quality", "transformation")


def build_graph():
    graph = StateGraph(SOVState)
    graph.add_node("sheet_intelligence", sheet_intelligence)
    graph.add_node("schema_mapping", schema_mapping)
    graph.add_node("data_quality", data_quality)
    graph.add_node("transformation", transformation)

    graph.set_entry_point("sheet_intelligence")
    graph.add_edge("sheet_intelligence", "schema_mapping")
    graph.add_edge("schema_mapping", "data_quality")
    graph.add_edge("data_quality", "transformation")
    graph.add_edge("transformation", END)
    return graph.compile()


COMPILED_GRAPH = build_graph()


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
    result: Any = COMPILED_GRAPH.invoke(initial_state(path, file_name))
    if isinstance(result, SOVState):
        return result
    return SOVState.model_validate(result)
