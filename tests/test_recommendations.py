"""
File: test_recommendations.py

1. What this file does:
   Comprehensive unit tests for the Phase 5 RecommendationEngine and BatchOptimizer.
2. Overall design:
   Uses pytest to evaluate recommendation rules, thresholds, edge cases (empty data,
   extreme memory usage, no churn, missing metadata), and linear batch optimization logic.
3. Elements in the file:
   - Tests for batch size rules (reduction and increase).
   - Tests for mixed precision rule (FP32 triggering vs FP16 suppressing).
   - Tests for cache management rule (empty_cache recommendation).
   - Tests for gradient checkpointing rule.
   - Tests for buffer reuse and churn detection.
   - Tests for BatchOptimizer scaling and selection.
4. Final output of the file:
   Pytest test outcomes validating the optimization engine.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from optimizer.recommendations import RecommendationEngine, OptimizationRecommendation
from optimizer.batch_optimizer import BatchOptimizer, BatchCandidate


def test_empty_events():
    engine = RecommendationEngine(budget_bytes=6 * 1024 * 1024 * 1024)
    report = engine.generate_recommendations([])
    summary = report["summary"]
    assert summary["total_events_analyzed"] == 0
    assert summary["peak_memory_bytes"] == 0
    assert len(report["recommendations"]) == 0


def test_batch_reduction_rule_high_pressure():
    # Budget = 1000 MB, Peak = 900 MB (90% > 85%) -> OPT-BATCH-001
    budget = 1000 * 1024 * 1024
    engine = RecommendationEngine(budget_bytes=budget, safety_margin_ratio=0.15)
    events = [
        {"allocated_bytes": str(900 * 1024 * 1024), "peak_memory": str(900 * 1024 * 1024), "reserved_bytes": str(950 * 1024 * 1024)}
    ]
    report = engine.generate_recommendations(events)
    recs = report["recommendations"]
    rule_ids = [r["rule_id"] for r in recs]
    assert "OPT-BATCH-001" in rule_ids


def test_batch_increase_rule_low_utilization():
    # Budget = 1000 MB, Peak = 200 MB (20% < 35%) -> OPT-BATCH-002
    budget = 1000 * 1024 * 1024
    engine = RecommendationEngine(budget_bytes=budget)
    events = [
        {"allocated_bytes": str(200 * 1024 * 1024), "peak_memory": str(200 * 1024 * 1024), "reserved_bytes": str(250 * 1024 * 1024)}
    ]
    report = engine.generate_recommendations(events)
    recs = report["recommendations"]
    rule_ids = [r["rule_id"] for r in recs]
    assert "OPT-BATCH-002" in rule_ids


def test_mixed_precision_rule_triggered_for_fp32():
    budget = 6 * 1024 * 1024 * 1024
    engine = RecommendationEngine(budget_bytes=budget)
    events = [
        {"allocated_bytes": str(3 * 1024 * 1024 * 1024), "precision": "FP32", "peak_memory": str(3 * 1024 * 1024 * 1024)}
    ]
    report = engine.generate_recommendations(events)
    recs = report["recommendations"]
    rule_ids = [r["rule_id"] for r in recs]
    assert "OPT-PREC-001" in rule_ids


def test_mixed_precision_suppressed_if_already_fp16():
    budget = 6 * 1024 * 1024 * 1024
    engine = RecommendationEngine(budget_bytes=budget)
    events = [
        {"allocated_bytes": str(3 * 1024 * 1024 * 1024), "precision": "FP16", "peak_memory": str(3 * 1024 * 1024 * 1024)}
    ]
    report = engine.generate_recommendations(events)
    recs = report["recommendations"]
    rule_ids = [r["rule_id"] for r in recs]
    assert "OPT-PREC-001" not in rule_ids


def test_cache_management_rule():
    # Reserved = 1000 MB, Allocated = 400 MB -> Unused = 600 MB (60% > 30% and > 100 MB)
    budget = 6 * 1024 * 1024 * 1024
    engine = RecommendationEngine(budget_bytes=budget)
    events = [
        {"allocated_bytes": str(400 * 1024 * 1024), "reserved_bytes": str(1000 * 1024 * 1024), "peak_memory": str(400 * 1024 * 1024)}
    ]
    report = engine.generate_recommendations(events)
    recs = report["recommendations"]
    rule_ids = [r["rule_id"] for r in recs]
    assert "OPT-CACHE-001" in rule_ids


def test_gradient_checkpointing_rule():
    # Peak is 70% of budget (>= 65%)
    budget = 1000 * 1024 * 1024
    engine = RecommendationEngine(budget_bytes=budget)
    events = [
        {"allocated_bytes": str(700 * 1024 * 1024), "peak_memory": str(700 * 1024 * 1024)}
    ]
    report = engine.generate_recommendations(events)
    recs = report["recommendations"]
    rule_ids = [r["rule_id"] for r in recs]
    assert "OPT-CKPT-001" in rule_ids


def test_buffer_reuse_rule():
    budget = 6 * 1024 * 1024 * 1024
    engine = RecommendationEngine(budget_bytes=budget)
    # Generate 40 alternating deltas
    events = []
    for i in range(40):
        delta = 1024 * 1024 if (i % 2 == 0) else -1024 * 1024
        events.append({"allocated_bytes": "1000000", "allocation_delta": str(delta)})

    report = engine.generate_recommendations(events)
    recs = report["recommendations"]
    rule_ids = [r["rule_id"] for r in recs]
    assert "OPT-REUSE-001" in rule_ids


def test_batch_optimizer_classification():
    # Budget = 1000 MB, Safety margin = 15% -> Safe threshold = 850 MB
    # Base = 200 MB, Per-sample = 100 MB
    # b=4: 200 + 400 = 600 MB -> VIABLE
    # b=7: 200 + 700 = 900 MB -> RISKY
    # b=10: 200 + 1000 = 1200 MB -> EXCEEDS_BUDGET
    budget = 1000 * 1024 * 1024
    optimizer = BatchOptimizer(budget_bytes=budget, safety_margin_ratio=0.15)
    base = 200 * 1024 * 1024
    per_sample = 100 * 1024 * 1024

    c4 = optimizer.evaluate_candidate(4, base, per_sample)
    assert c4.status == "VIABLE"

    c7 = optimizer.evaluate_candidate(7, base, per_sample)
    assert c7.status == "RISKY"

    c10 = optimizer.evaluate_candidate(10, base, per_sample)
    assert c10.status == "EXCEEDS_BUDGET"


def test_batch_optimizer_select_best_viable():
    budget = 1000 * 1024 * 1024
    optimizer = BatchOptimizer(budget_bytes=budget, safety_margin_ratio=0.15)
    base = 200 * 1024 * 1024
    per_sample = 100 * 1024 * 1024

    sweep = optimizer.evaluate_sweep(base, per_sample, candidates=[2, 4, 6, 8, 12])
    best = optimizer.select_best_viable_candidate(sweep)
    # At b=6: 200 + 600 = 800 MB <= 850 MB (VIABLE)
    # At b=8: 200 + 800 = 1000 MB > 850 MB (RISKY)
    assert best is not None
    assert best.batch_size == 6
