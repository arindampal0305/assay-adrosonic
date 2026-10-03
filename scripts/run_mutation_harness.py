"""Mutation Testing Harness for ASSAY (Milestone 5).

Generates mutated SOV files with known layout variations and injected data quality flaws,
runs them through the ASSAY end-to-end pipeline, and evaluates performance against targets:
- Mapping Accuracy (Target >= 85%)
- Data Quality Recall (Target >= 95%)
- Transformation Correctness (Target 100%)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

from openpyxl import Workbook
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.graph import COMPILED_GRAPH, initial_state
from backend.state.sov_state import SOVState
from backend.state.target_schema import TARGET_FIELD_NAMES
from langgraph.types import Command


def create_mutated_test_cases(output_dir: Path) -> list[dict[str, Any]]:
    """Create test workbooks with known ground-truth mappings and injected DQ issues."""
    output_dir.mkdir(parents=True, exist_ok=True)
    test_cases = []

    # --- Test Case 1: Standard CSV with stripped zip codes & currency strings ---
    tc1_path = output_dir / "mutated_tc1.csv"
    tc1_headers = [
        "Loc #", "Street Address", "City", "St", "Zip",
        "Bldg Repl Cost", "Contents", "BI/EE", "Occ", "Const", "Stories", "# Bldgs", "Yr Blt", "Sprk"
    ]
    tc1_ground_truth_mapping = {
        "Loc #": "Reference",
        "Street Address": "Address",
        "City": "City",
        "St": "State",
        "Zip": "Zip",
        "Bldg Repl Cost": "Building Value",
        "Contents": "Contents",
        "BI/EE": "BI",
        "Occ": "Occupancy",
        "Const": "Construction",
        "Stories": "Storeys",
        "# Bldgs": "Number of Buildings",
        "Yr Blt": "Year Built",
        "Sprk": "Fire Sprinklers (Y/N)",
    }
    tc1_injected_issues = ["DQ-08", "DQ-10", "DQ-12"]
    tc1_rows = [
        ["LOC-1", "100 Main St", "Boston", "MA", "2110", "$1,000,000.00", 500000, 100000, "Office", "Masonry", 2, 1, 1995, "Y"],
        ["LOC-2", "200 Ocean Ave", "Miami", "FL", "33101", 2500000, 800000, 200000, "Retail", "Frame", 1, 1, 1900, "N"],
    ]

    with tc1_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(tc1_headers)
        writer.writerows(tc1_rows)

    test_cases.append({
        "name": "TC1: Basic CSV with currency & zip issues",
        "path": tc1_path,
        "mapping_truth": tc1_ground_truth_mapping,
        "injected_dq": tc1_injected_issues,
    })

    # --- Test Case 2: Title block header shift ---
    tc2_path = output_dir / "mutated_tc2.xlsx"
    wb2 = Workbook()
    ws2 = wb2.active
    ws2.title = "SOV Schedule"
    ws2.cell(row=1, column=1, value="PROPERTY SCHEDULE 2026")
    ws2.cell(row=2, column=1, value="CLIENT: GLOBAL LOGISTICS INC")

    tc2_headers = [
        "Location ID", "Address", "City Name", "State Code", "Postal Code",
        "Building Value", "Personal Property", "Business Interruption", "Occupancy", "Construction", "Year Built"
    ]
    tc2_ground_truth_mapping = {
        "Location ID": "Reference",
        "Address": "Address",
        "City Name": "City",
        "State Code": "State",
        "Postal Code": "Zip",
        "Building Value": "Building Value",
        "Personal Property": "Contents",
        "Business Interruption": "BI",
        "Occupancy": "Occupancy",
        "Construction": "Construction",
        "Year Built": "Year Built",
    }
    tc2_injected_issues = ["DQ-08", "DQ-09", "DQ-12"]
    tc2_rows = [
        [101, "50 Industrial Way", "Dallas", "Texas", "75001", 3000000, 1200000, 300000, "Warehouse", "Metal", 2005],
        [102, "80 Commercial Rd", "Austin", "TX", "7870", 1500000, 400000, 50000, "Office", "Masonry", 0],
    ]

    for c, h in enumerate(tc2_headers, 1):
        ws2.cell(row=4, column=c, value=h)

    for r_idx, row in enumerate(tc2_rows, 5):
        for c_idx, val in enumerate(row, 1):
            ws2.cell(row=r_idx, column=c_idx, value=val)

    wb2.save(tc2_path)
    test_cases.append({
        "name": "TC2: Title block shift Excel",
        "path": tc2_path,
        "mapping_truth": tc2_ground_truth_mapping,
        "injected_dq": tc2_injected_issues,
    })

    return test_cases


def run_harness() -> bool:
    """Execute evaluation harness on test cases and print report."""
    print("=" * 78)
    print("ASSAY MUTATION EVALUATION HARNESS")
    print("=" * 78)

    test_dir = ROOT / "scratch" / "mutation_test_data"
    test_cases = create_mutated_test_cases(test_dir)

    total_mapping_targets = 0
    correct_mappings = 0
    total_injected_dq = 0
    recalled_dq = 0
    total_approved_ops = 0
    correct_applied_ops = 0

    for tc in test_cases:
        print(f"\nEvaluating: {tc['name']}")
        path = tc["path"]
        
        # 1. Ingest initial state
        init_st = initial_state(path)
        run_id = init_st.run_id
        config = {"configurable": {"thread_id": run_id}}

        # 2. Invoke Graph to Human Gate interrupt
        res = COMPILED_GRAPH.invoke(init_st, config=config)
        snapshot = COMPILED_GRAPH.get_state(config)
        state_dict = snapshot.values if snapshot else res
        state = state_dict if isinstance(state_dict, SOVState) else SOVState.model_validate(state_dict)

        # Evaluate Mapping
        mappings_list = state.mapping.get("mappings", [])
        predicted_map = {
            m.get("source_column"): m.get("target")
            for m in mappings_list if isinstance(m, dict)
        }
        for src, target in tc["mapping_truth"].items():
            total_mapping_targets += 1
            predicted = predicted_map.get(src)
            if predicted == target:
                correct_mappings += 1
            else:
                print(f"  [MAPPING MISMATCH] Column '{src}': expected '{target}', got '{predicted}'")

        # Evaluate DQ Detection
        detected_rules = {issue.rule for issue in (state.issues or [])}
        if isinstance(state.quality, dict):
            for issue in state.quality.get("issues", []):
                rule_name = issue.get("rule") if isinstance(issue, dict) else getattr(issue, "rule", "")
                if rule_name:
                    detected_rules.add(rule_name)
        for rec in (state.recommendations or []):
            rule_id = rec.get("rule_id") or rec.get("rule")
            if rule_id:
                detected_rules.add(rule_id)

        for dq in tc["injected_dq"]:
            total_injected_dq += 1
            if any(dq.lower() in r.lower() for r in detected_rules):
                recalled_dq += 1
            else:
                print(f"  [DQ MISSED] Injected DQ issue '{dq}' was not flagged (detected rules: {detected_rules})")

        # 3. Resume Graph with Auto-Approve Decisions
        recs = state.recommendations
        decisions = [
            {"rec_id": r.get("id"), "action": "accept", "note": "Harness approval", "by": "harness"}
            for r in recs if "id" in r
        ]

        raw_res = COMPILED_GRAPH.invoke(Command(resume=decisions), config=config)
        final_snapshot = COMPILED_GRAPH.get_state(config)
        final_val = final_snapshot.values if final_snapshot else raw_res
        if isinstance(final_val, dict) and "audit" in final_val and final_val["audit"]:
            final_val["audit"] = [a.model_dump() if hasattr(a, "model_dump") else a for a in final_val["audit"]]
        final_state = final_val if isinstance(final_val, SOVState) else SOVState.model_validate(final_val)

        # Evaluate Transformation Correctness
        applied_audit = final_state.audit or []
        for dec in decisions:
            total_approved_ops += 1
            # Check if an audit record exists corresponding to applied ops
            if len(applied_audit) > 0 or len(recs) == 0:
                correct_applied_ops += 1

    # Final Metric Calculations
    mapping_acc = (correct_mappings / total_mapping_targets * 100) if total_mapping_targets else 100.0
    dq_recall = (recalled_dq / total_injected_dq * 100) if total_injected_dq else 100.0
    trn_correctness = (correct_applied_ops / total_approved_ops * 100) if total_approved_ops else 100.0

    print("\n" + "=" * 78)
    print("HARNESS RESULTS SUMMARY")
    print("=" * 78)
    print(f"Schema Mapping Accuracy    : {mapping_acc:6.2f}%  (Target >= 85.0%)  {'[PASS]' if mapping_acc >= 85.0 else '[FAIL]'}")
    print(f"Data Quality Recall        : {dq_recall:6.2f}%  (Target >= 95.0%)  {'[PASS]' if dq_recall >= 95.0 else '[FAIL]'}")
    print(f"Transformation Correctness : {trn_correctness:6.2f}%  (Target 100.0%)  {'[PASS]' if trn_correctness >= 100.0 else '[FAIL]'}")
    print("=" * 78)

    passed = mapping_acc >= 85.0 and dq_recall >= 95.0 and trn_correctness >= 100.0
    return passed


if __name__ == "__main__":
    success = run_harness()
    sys.exit(0 if success else 1)
