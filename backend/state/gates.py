"""Contract gates (FR-ORC-02, NFR-STA-01).

No agent runs on unvalidated state. Every gate does two things before letting an
agent see the state: re-validate the whole object against the SOVState schema
(the type check), then assert the preconditions that agent depends on.
"""

from __future__ import annotations

import functools
from time import perf_counter
from typing import Any, Callable

from backend.state.sov_state import STATE_VERSION, SOVState, TraceEvent


class ContractViolation(Exception):
    """Raised when state reaching a gate is malformed or incomplete."""

    def __init__(self, agent: str, failures: list[str]) -> None:
        self.agent = agent
        self.failures = failures
        joined = "; ".join(failures)
        super().__init__(f"contract gate before '{agent}' rejected the state: {joined}")


def validate_state(state: Any, agent: str = "unknown") -> SOVState:
    if isinstance(state, SOVState):
        payload = state.model_dump(by_alias=True)
    elif isinstance(state, dict):
        payload = state
    else:
        raise ContractViolation(agent, [f"state is {type(state).__name__}, not SOVState"])

    try:
        validated = SOVState.model_validate(payload)
    except Exception as exc:
        raise ContractViolation(agent, [f"state failed schema validation: {exc}"]) from exc

    if validated.version != STATE_VERSION:
        raise ContractViolation(
            agent, [f"state version {validated.version} != expected {STATE_VERSION}"]
        )
    return validated


Precondition = tuple[Callable[[SOVState], bool], str]


def _exactly_one_primary(state: SOVState) -> bool:
    return sum(1 for e in state.manifest if e.class_ == "Primary") == 1


def _mapping_present(state: SOVState) -> bool:
    """Agent 2 is real as of milestone 2, so a downstream agent running against an
    empty mapping is now a bug rather than an expected stub state."""
    return bool(state.mapping) and "mappings" in state.mapping


def _mapping_resolves_something(state: SOVState) -> bool:
    """Validating rows or transforming a sheet with zero resolved columns would
    produce a confidently empty output file, which is worse than stopping."""
    mappings = state.mapping.get("mappings") or []
    return any(m.get("target") for m in mappings)


PRECONDITIONS: dict[str, tuple[Precondition, ...]] = {
    "sheet_intelligence": (
        (lambda s: bool(s.source.file_path), "source.file_path is required to read the workbook"),
        (lambda s: bool(s.source.file_name), "source.file_name is empty"),
    ),
    "schema_mapping": (
        (lambda s: len(s.manifest) > 0, "manifest is empty; Agent 1 produced no sheet classification"),
        (_exactly_one_primary, "manifest must contain exactly one Primary sheet"),
    ),
    "data_quality": (
        (lambda s: len(s.manifest) > 0, "manifest is empty"),
        (_exactly_one_primary, "manifest must contain exactly one Primary sheet"),
        (_mapping_present, "mapping is empty; Agent 2 produced no column mapping"),
        (
            _mapping_resolves_something,
            "mapping resolved no columns to target fields; there is nothing to validate",
        ),
    ),
    "transformation": (
        (lambda s: len(s.manifest) > 0, "manifest is empty"),
        (_exactly_one_primary, "manifest must contain exactly one Primary sheet"),
        (_mapping_present, "mapping is empty; Agent 2 produced no column mapping"),
        (
            _mapping_resolves_something,
            "mapping resolved no columns to target fields; there is nothing to transform",
        ),
    ),
}


def check_preconditions(state: SOVState, agent: str) -> None:
    failures = [msg for predicate, msg in PRECONDITIONS[agent] if not predicate(state)]
    if failures:
        raise ContractViolation(agent, failures)


def contract_gate(agent: str) -> Callable:
    """Wrap an agent node so its gate runs first and its trace entry runs last."""

    def decorator(fn: Callable[[SOVState], dict[str, Any]]) -> Callable:
        @functools.wraps(fn)
        def node(state: Any) -> dict[str, Any]:
            validated = validate_state(state, agent)
            check_preconditions(validated, agent)

            started = perf_counter()
            update = fn(validated)
            elapsed_ms = int((perf_counter() - started) * 1000)

            trace = list(update.get("trace", []))
            trace.append(TraceEvent(agent=agent, event="finished", ms=elapsed_ms))
            update["trace"] = trace
            return update

        node.agent_name = agent
        return node

    return decorator
