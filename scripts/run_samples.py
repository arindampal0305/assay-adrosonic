"""Run every file in samples/ through the graph and print its manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ingest.loader import SUPPORTED_SUFFIXES, IngestError  # noqa: E402
from backend.graph import run_pipeline  # noqa: E402
from backend.state.gates import ContractViolation  # noqa: E402

SAMPLES = ROOT / "samples"


def report(path: Path) -> None:
    print("=" * 78)
    print(f"FILE  {path.name}")
    print("=" * 78)
    try:
        state = run_pipeline(path)
    except (IngestError, ContractViolation) as exc:
        print(f"  BLOCKED: {exc}\n")
        return

    print(f"  run_id {state.run_id}   state version {state.version}   sheets {state.source.sheets}")
    for entry in state.manifest:
        print(f"\n  [{entry.class_}] {entry.sheet!r}  confidence {entry.confidence}  score {entry.score}")
        print(f"    header_row       {entry.header_row}   data_start_row {entry.data_start_row}   data_rows {entry.data_rows}")
        print(f"    composite_header {entry.composite_header}")
        print(f"    factors          {json.dumps(entry.factor_scores)}")
        for reason in entry.reasons:
            print(f"    - {reason}")
        if entry.headers:
            print(f"    headers          {entry.headers}")
    print("\n  trace " + ", ".join(f"{e.agent}:{e.event} {e.ms}ms" for e in state.trace))
    print()


def main() -> None:
    files = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        print(f"no .xlsx/.csv files in {SAMPLES}; run scripts/make_samples.py first")
        return
    for path in files:
        report(path)


if __name__ == "__main__":
    main()
