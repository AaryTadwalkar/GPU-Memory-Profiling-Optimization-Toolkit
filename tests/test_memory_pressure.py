"""
File: test_memory_pressure.py

1. What this file does:
   Unit tests the Phase 2 static memory pressure analyzer logic.
2. Overall design:
   Uses pytest (or standard unittest if run directly) to evaluate the 
   heuristics and calculations against various edge-case scenarios 
   (empty files, missing columns, zero budgets).
"""

import os
import csv
import tempfile
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from profiler.memory_pressure import evaluate_oom_risk_heuristic, analyze_pressure

def test_oom_risk_heuristic_low():
    # 6GB budget, 1GB used, 0 growth -> Low
    budget = 6 * 1024 * 1024 * 1024
    allocated = 1 * 1024 * 1024 * 1024
    risk = evaluate_oom_risk_heuristic(budget, allocated, 0.0)
    assert risk == "Low"

def test_oom_risk_heuristic_moderate():
    # 6GB budget, 5GB used (headroom < 25%), 0 growth -> Moderate
    budget = 6 * 1024 * 1024 * 1024
    allocated = 5 * 1024 * 1024 * 1024
    risk = evaluate_oom_risk_heuristic(budget, allocated, 0.0)
    assert risk == "Moderate"

def test_oom_risk_heuristic_high():
    # 6GB budget, 5.5GB used (headroom < 10%), 0 growth -> High
    budget = 6 * 1024 * 1024 * 1024
    allocated = int(5.5 * 1024 * 1024 * 1024)
    risk = evaluate_oom_risk_heuristic(budget, allocated, 0.0)
    assert risk == "High"

def test_oom_risk_heuristic_growth_escalation():
    # 6GB budget, 1GB used -> Normally Low. 
    # But growth > 50 MB/s -> escalates to Moderate
    budget = 6 * 1024 * 1024 * 1024
    allocated = 1 * 1024 * 1024 * 1024
    growth = 100 * 1024 * 1024  # 100 MB/s
    risk = evaluate_oom_risk_heuristic(budget, allocated, growth)
    assert risk == "Moderate"

def test_oom_risk_heuristic_exceed_budget():
    # peak equal to budget -> High
    risk = evaluate_oom_risk_heuristic(100, 100, 0.0)
    assert risk == "High"
    # peak exceeding budget -> High
    risk = evaluate_oom_risk_heuristic(100, 150, 0.0)
    assert risk == "High"

def test_oom_risk_zero_budget():
    risk = evaluate_oom_risk_heuristic(0, 100, 0.0)
    assert risk == "Unknown"

# Test analyze_pressure with CSV files
def _write_csv(rows):
    fd, path = tempfile.mkstemp(suffix=".csv")
    with open(path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    os.close(fd)
    return path

def test_analyze_empty_csv():
    path = _write_csv([])
    results = analyze_pressure(path, 100)
    os.remove(path)
    assert len(results) == 0

def test_analyze_one_sample():
    path = _write_csv([
        {"timestamp": "1.0", "allocated_bytes": "50", "peak_memory": "50"}
    ])
    results = analyze_pressure(path, 100)
    os.remove(path)
    assert len(results) == 1
    assert results[0]["growth_rate_bps"] == 0.0
    assert results[0]["budget_headroom"] == 50

def test_analyze_zero_time_interval():
    path = _write_csv([
        {"timestamp": "1.0", "allocated_bytes": "50", "peak_memory": "50"},
        {"timestamp": "1.0", "allocated_bytes": "100", "peak_memory": "100"}
    ])
    results = analyze_pressure(path, 200)
    os.remove(path)
    assert len(results) == 2
    # dt = 0 -> growth rate should remain 0 to avoid DivByZero
    assert results[1]["growth_rate_bps"] == 0.0

def test_analyze_missing_columns():
    path = _write_csv([
        {"timestamp": "1.0"} # no allocated_bytes
    ])
    results = analyze_pressure(path, 100)
    os.remove(path)
    assert len(results) == 0 # skips row due to no allocated_bytes (or assumes 0 and skips if not torch event)

def test_analyze_normal_growth():
    path = _write_csv([
        {"timestamp": "1.0", "allocated_bytes": "50", "peak_memory": "50"},
        {"timestamp": "2.0", "allocated_bytes": "150", "peak_memory": "150"}
    ])
    results = analyze_pressure(path, 200)
    os.remove(path)
    assert len(results) == 2
    assert results[1]["growth_rate_bps"] == 100.0
    assert results[1]["budget_headroom"] == 50
    assert results[1]["peak_to_capacity"] == 0.75
