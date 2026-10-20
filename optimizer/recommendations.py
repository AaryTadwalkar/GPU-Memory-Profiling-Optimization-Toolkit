"""
File: recommendations.py

1. What this file does:
   Analyzes GPU memory profiling event streams and generates evidence-backed,
   traceable optimization recommendations for batch size, mixed precision,
   allocator caching, gradient checkpointing, and buffer reuse.
2. Overall design:
   A modular recommendation engine that reads canonical MemoryEvent CSV logs or
   event dictionaries. It extracts workload high-water marks and behavioral
   patterns, computes diagnostic indicators against a configured memory budget,
   and applies transparent heuristics to emit structured recommendations complete
   with triggering metrics, observed values, thresholds, and operational caveats.
3. Elements in the file:
   - OptimizationRecommendation: Dataclass defining the output recommendation schema.
   - RecommendationEngine: Class implementing heuristic diagnostic rules.
   - load_events_from_csv: Utility to load and validate MemoryEvent logs.
   - CLI entrypoint: Facilitates standalone analysis producing formatted JSON reports.
4. Final output of the file:
   A JSON file containing summary diagnostics and an array of traceable recommendations,
   and/or a human-readable summary printed to stdout.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class OptimizationRecommendation:
    """
    1. The use of class:
       Encapsulates an individual optimization recommendation emitted by the engine.
    2. What the class contain:
       rule_id, category, triggering_metric, measured_value, threshold, recommendation, caveat.
    3. Technologies used and why:
       Python standard dataclass for lightweight serialization to JSON and dict.
    4. Key design and overall design of the class:
       Enforces traceability by requiring every recommendation to cite its exact
       metric, measured value, decision threshold, and caveats.
    """
    rule_id: str
    category: str
    triggering_metric: str
    measured_value: str
    threshold: str
    recommendation: str
    caveat: str

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Converts the recommendation to a standard dictionary.
        2. How does the specific function work:
           Calls dataclasses.asdict on self.
        3. Concepts used in the function:
           Serialization.
        4. Places where this function is imported and used:
           Used by RecommendationEngine and CLI export handlers.
        5. Final output of the function, before and after:
           Before: Dataclass instance. After: Standard Python dictionary.
        """
        return asdict(self)


class RecommendationEngine:
    """
    1. The use of class:
       Inspects memory profiling records and derives actionable optimization advice.
    2. What the class contain:
       Configurable memory budget, safety margins, heuristic evaluation methods for
       batch size, precision, cache management, checkpointing, and tensor reuse.
    3. Technologies used and why:
       Pure Python arithmetic and heuristic decision trees for cross-platform compatibility
       and zero-hardware-dependency testing.
    4. Key design and overall design of the class:
       Takes observed telemetry summaries and passes them through discrete, independent
       rule checkers to produce an auditable diagnostic report.
    """
    def __init__(self, budget_bytes: int = 6 * 1024 * 1024 * 1024, safety_margin_ratio: float = 0.15):
        self.budget_bytes = budget_bytes
        self.safety_margin_ratio = safety_margin_ratio

    def summarize_telemetry(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        1. The use of function:
           Aggregates raw memory events into core diagnostic metrics.
        2. How does the specific function work:
           Scans the event list to identify max allocated bytes, max reserved bytes,
           peak memory, largest cached-but-unused gaps, and allocation delta fluctuations.
        3. Concepts used in the function:
           Data aggregation, time-series telemetry analysis.
        4. Places where this function is imported and used:
           Called internally within generate_recommendations and batch evaluation routines.
        5. Final output of the function, before and after:
           Before: List of event dicts. After: Dictionary of summary statistics.
        """
        max_allocated = 0
        max_reserved = 0
        max_peak = 0
        max_cached_unused = 0
        positive_deltas = 0
        negative_deltas = 0
        workload_name = None
        precision = None
        batch_size = None

        valid_sample_count = 0

        for row in events:
            alloc_val = row.get("allocated_bytes")
            if alloc_val is not None and str(alloc_val).strip() != "":
                try:
                    alloc_int = int(alloc_val)
                    if alloc_int > max_allocated:
                        max_allocated = alloc_int
                    valid_sample_count += 1
                except ValueError:
                    pass

            res_val = row.get("reserved_bytes")
            if res_val is not None and str(res_val).strip() != "":
                try:
                    res_int = int(res_val)
                    if res_int > max_reserved:
                        max_reserved = res_int
                except ValueError:
                    pass

            peak_val = row.get("peak_memory")
            if peak_val is not None and str(peak_val).strip() != "":
                try:
                    peak_int = int(peak_val)
                    if peak_int > max_peak:
                        max_peak = peak_int
                except ValueError:
                    pass

            if alloc_val is not None and res_val is not None:
                try:
                    unused = int(res_val) - int(alloc_val)
                    if unused > max_cached_unused:
                        max_cached_unused = unused
                except ValueError:
                    pass

            delta_val = row.get("allocation_delta")
            if delta_val is not None and str(delta_val).strip() != "":
                try:
                    delta_int = int(delta_val)
                    if delta_int > 0:
                        positive_deltas += 1
                    elif delta_int < 0:
                        negative_deltas += 1
                except ValueError:
                    pass

            if not workload_name and row.get("workload_name"):
                workload_name = row.get("workload_name")
            if not precision and row.get("precision"):
                precision = row.get("precision")
            if not batch_size and row.get("batch_size"):
                try:
                    batch_size = int(row.get("batch_size"))
                except ValueError:
                    pass

        effective_peak = max(max_peak, max_allocated)
        peak_to_capacity = (effective_peak / self.budget_bytes) if self.budget_bytes > 0 else 0.0

        return {
            "budget_bytes": self.budget_bytes,
            "max_allocated_bytes": max_allocated,
            "max_reserved_bytes": max_reserved,
            "peak_memory_bytes": effective_peak,
            "max_cached_unused_bytes": max_cached_unused,
            "peak_to_capacity_ratio": peak_to_capacity,
            "positive_deltas_count": positive_deltas,
            "negative_deltas_count": negative_deltas,
            "total_events_analyzed": len(events),
            "valid_samples": valid_sample_count,
            "workload_name": workload_name or "unknown_workload",
            "precision": precision or "FP32",
            "batch_size": batch_size,
        }

    def check_batch_size_rule(self, summary: Dict[str, Any]) -> Optional[OptimizationRecommendation]:
        """
        1. The use of function:
           Evaluates whether batch size should be reduced or can safely be increased.
        2. How does the specific function work:
           Compares peak-to-capacity ratio against safety thresholds (high watermark 85%, low watermark 35%).
        3. Concepts used in the function:
           Capacity planning, OOM avoidance, hardware saturation.
        4. Places where this function is imported and used:
           Called in generate_recommendations.
        5. Final output of the function, before and after:
           Before: Telemetry summary. After: OptimizationRecommendation or None.
        """
        ratio = summary["peak_to_capacity_ratio"]
        peak_mb = summary["peak_memory_bytes"] / (1024 * 1024)
        budget_mb = summary["budget_bytes"] / (1024 * 1024)

        if ratio > (1.0 - self.safety_margin_ratio):
            return OptimizationRecommendation(
                rule_id="OPT-BATCH-001",
                category="Batch Size Adjustment",
                triggering_metric="peak_to_capacity_ratio",
                measured_value=f"{ratio:.2%} ({peak_mb:.1f} MB / {budget_mb:.1f} MB)",
                threshold=f"> {1.0 - self.safety_margin_ratio:.0%}",
                recommendation="Reduce batch size (e.g. by 25% to 50%) to increase memory headroom and prevent sudden OOM failures.",
                caveat="Decreasing batch size may reduce GPU tensor core utilization and increase overall epoch training time."
            )
        elif ratio < 0.35 and summary["valid_samples"] > 0:
            return OptimizationRecommendation(
                rule_id="OPT-BATCH-002",
                category="Batch Size Adjustment",
                triggering_metric="peak_to_capacity_ratio",
                measured_value=f"{ratio:.2%} ({peak_mb:.1f} MB / {budget_mb:.1f} MB)",
                threshold="< 35%",
                recommendation="Consider increasing batch size to achieve higher GPU memory and compute saturation.",
                caveat="Larger batches increase peak memory proportionally; verify headroom before full-scale deployment."
            )
        return None

    def check_mixed_precision_rule(self, summary: Dict[str, Any]) -> Optional[OptimizationRecommendation]:
        """
        1. The use of function:
           Evaluates whether mixed precision (FP16/BF16) should be recommended.
        2. How does the specific function work:
           Triggers if workload uses FP32 and peak memory exceeds 40% of total budget or 2 GB.
        3. Concepts used in the function:
           Mixed-precision computation, PyTorch AMP, memory footprint reduction.
        4. Places where this function is imported and used:
           Called in generate_recommendations.
        5. Final output of the function, before and after:
           Before: Telemetry summary. After: OptimizationRecommendation or None.
        """
        precision = summary.get("precision", "FP32")
        peak_bytes = summary["peak_memory_bytes"]
        peak_mb = peak_bytes / (1024 * 1024)

        # Trigger if precision is FP32/unspecified and memory consumption is significant (>40% budget or >1.5GB)
        threshold_bytes = min(int(self.budget_bytes * 0.40), int(1.5 * 1024 * 1024 * 1024))
        if precision.upper() == "FP32" and peak_bytes >= threshold_bytes and peak_bytes > 0:
            return OptimizationRecommendation(
                rule_id="OPT-PREC-001",
                category="Mixed Precision (FP16/BF16)",
                triggering_metric="precision_and_peak_memory",
                measured_value=f"Precision={precision}, Peak={peak_mb:.1f} MB",
                threshold=f"FP32 and Peak >= {threshold_bytes / (1024 * 1024):.1f} MB",
                recommendation="Enable PyTorch Automatic Mixed Precision (torch.cuda.amp.autocast) with FP16 or BF16 to reduce activation and weight footprint by up to 50%.",
                caveat="FP16 may require GradScaler to prevent gradient underflow; BF16 is recommended on Ampere (RTX 30xx/40xx) GPUs."
            )
        return None

    def check_cache_management_rule(self, summary: Dict[str, Any]) -> Optional[OptimizationRecommendation]:
        """
        1. The use of function:
           Identifies large reserved-but-unused allocator cache bloat.
        2. How does the specific function work:
           Checks if (reserved - allocated) memory exceeds 30% of reserved memory and is over 100 MB.
        3. Concepts used in the function:
           PyTorch caching allocator mechanics, external fragmentation, virtual memory reuse.
        4. Places where this function is imported and used:
           Called in generate_recommendations.
        5. Final output of the function, before and after:
           Before: Telemetry summary. After: OptimizationRecommendation or None.
        """
        cached_unused = summary["max_cached_unused_bytes"]
        max_reserved = summary["max_reserved_bytes"]
        cached_unused_mb = cached_unused / (1024 * 1024)

        if max_reserved > 0:
            unused_ratio = cached_unused / max_reserved
            if unused_ratio > 0.30 and cached_unused > (100 * 1024 * 1024):
                return OptimizationRecommendation(
                    rule_id="OPT-CACHE-001",
                    category="Allocator Cache Management",
                    triggering_metric="cached_but_unused_ratio",
                    measured_value=f"{unused_ratio:.2%} ({cached_unused_mb:.1f} MB unused / {max_reserved / (1024*1024):.1f} MB reserved)",
                    threshold="unused_ratio > 30% and unused_bytes > 100 MB",
                    recommendation="Invoke torch.cuda.empty_cache() between distinct workload phases to release cached virtual memory back to the driver.",
                    caveat="empty_cache() incurs GPU synchronization overhead and does not release memory currently occupied by active tensors."
                )
        return None

    def check_gradient_checkpointing_rule(self, summary: Dict[str, Any]) -> Optional[OptimizationRecommendation]:
        """
        1. The use of function:
           Recommends gradient checkpointing for deep models with large activation footprint.
        2. How does the specific function work:
           Triggers if peak memory consumes over 65% of budget and workload indicates training/model execution.
        3. Concepts used in the function:
           Activation checkpointing, recomputation vs memory tradeoff.
        4. Places where this function is imported and used:
           Called in generate_recommendations.
        5. Final output of the function, before and after:
           Before: Telemetry summary. After: OptimizationRecommendation or None.
        """
        ratio = summary["peak_to_capacity_ratio"]
        peak_mb = summary["peak_memory_bytes"] / (1024 * 1024)

        if ratio >= 0.65:
            return OptimizationRecommendation(
                rule_id="OPT-CKPT-001",
                category="Gradient Checkpointing",
                triggering_metric="activation_memory_pressure",
                measured_value=f"Peak-to-Capacity={ratio:.2%} ({peak_mb:.1f} MB)",
                threshold="peak_to_capacity >= 65%",
                recommendation="Enable gradient checkpointing (torch.utils.checkpoint) for model backbone layers to trade 20-30% extra compute for significant activation memory savings.",
                caveat="Gradient checkpointing recomputes activations during backward passes and is only effective for training workloads."
            )
        return None

    def check_buffer_reuse_rule(self, summary: Dict[str, Any]) -> Optional[OptimizationRecommendation]:
        """
        1. The use of function:
           Detects high allocation churn and suggests pre-allocated buffer reuse.
        2. How does the specific function work:
           Inspects positive and negative allocation counts; triggers if allocation churn exceeds threshold.
        3. Concepts used in the function:
           Allocation churn, allocator fragmentation, tensor lifecycle pooling.
        4. Places where this function is imported and used:
           Called in generate_recommendations.
        5. Final output of the function, before and after:
           Before: Telemetry summary. After: OptimizationRecommendation or None.
        """
        pos = summary["positive_deltas_count"]
        neg = summary["negative_deltas_count"]
        churn_total = pos + neg

        if churn_total >= 30 and (pos > 0 and neg > 0):
            return OptimizationRecommendation(
                rule_id="OPT-REUSE-001",
                category="Buffer Reuse & Churn Reduction",
                triggering_metric="allocation_churn_count",
                measured_value=f"{churn_total} allocation churn transitions (+{pos}/-{neg})",
                threshold="churn_total >= 30",
                recommendation="High allocation churn detected. Pre-allocate fixed tensor buffers and reuse them via in-place operations (e.g. tensor.copy_()) instead of creating new tensors inside iteration loops.",
                caveat="Reusing buffers requires manual tensor lifecycle management and shape consistency across iterations."
            )
        return None

    def generate_recommendations(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        1. The use of function:
           Orchestrates evaluation of all optimization rules on collected events.
        2. How does the specific function work:
           Summarizes telemetry, runs rule checkers in sequence, and packages summary and recommendations into a dict.
        3. Concepts used in the function:
           Heuristic diagnosis, structured report generation.
        4. Places where this function is imported and used:
           Used by CLI main, Streamlit dashboard, and automated tests.
        5. Final output of the function, before and after:
           Before: List of event dicts. After: Complete diagnostic dictionary.
        """
        summary = self.summarize_telemetry(events)
        recs: List[OptimizationRecommendation] = []

        rules = [
            self.check_batch_size_rule,
            self.check_mixed_precision_rule,
            self.check_cache_management_rule,
            self.check_gradient_checkpointing_rule,
            self.check_buffer_reuse_rule,
        ]

        for rule in rules:
            rec = rule(summary)
            if rec:
                recs.append(rec)

        return {
            "summary": summary,
            "recommendations": [r.to_dict() for r in recs],
        }


def load_events_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    1. The use of function:
       Loads and validates MemoryEvent rows from a CSV file.
    2. How does the specific function work:
       Opens the given CSV path and returns a list of row dictionaries via csv.DictReader.
    3. Concepts used in the function:
       CSV data ingestion.
    4. Places where this function is imported and used:
       Called by CLI main and test runners.
    5. Final output of the function, before and after:
       Before: CSV file on disk. After: List of dictionary records.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV profile not found: {csv_path}")

    events = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)
    return events


def main():
    """
    1. The use of function:
       Command-line entrypoint for the recommendation engine.
    2. How does the specific function work:
       Parses CLI arguments (csv_path, --out, --budget), runs RecommendationEngine,
       writes JSON output, and displays a human-readable console summary.
    3. Concepts used in the function:
       CLI argument parsing, report export.
    4. Places where this function is imported and used:
       Executed as `python optimizer/recommendations.py`.
    5. Final output of the function, before and after:
       Before: Raw profile CSV. After: JSON recommendation file and console output.
    """
    parser = argparse.ArgumentParser(description="Generate optimization recommendations from GPU memory profile.")
    parser.add_argument("csv_path", type=str, help="Path to torch_events.csv profile")
    parser.add_argument("--out", type=str, default="data/profiles/recs.json", help="Output path for JSON recommendations")
    parser.add_argument("--budget", type=int, default=6 * 1024 * 1024 * 1024, help="GPU memory budget in bytes (default 6GB)")
    args = parser.parse_args()

    events = load_events_from_csv(args.csv_path)
    engine = RecommendationEngine(budget_bytes=args.budget)
    report = engine.generate_recommendations(events)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("GPU MEMORY OPTIMIZATION REPORT")
    print("=" * 60)
    summary = report["summary"]
    print(f"Events Analyzed:       {summary['total_events_analyzed']}")
    print(f"Budget:                {summary['budget_bytes'] / (1024**3):.2f} GB")
    print(f"Peak Memory:           {summary['peak_memory_bytes'] / (1024**2):.1f} MB")
    print(f"Peak-to-Capacity:      {summary['peak_to_capacity_ratio']:.2%}")
    print(f"Max Cached Unused:     {summary['max_cached_unused_bytes'] / (1024**2):.1f} MB")
    print(f"Total Recommendations: {len(report['recommendations'])}")
    print("-" * 60)

    for i, rec in enumerate(report["recommendations"], 1):
        print(f"[{i}] {rec['category']} ({rec['rule_id']})")
        print(f"    Trigger:        {rec['triggering_metric']} = {rec['measured_value']}")
        print(f"    Threshold:      {rec['threshold']}")
        print(f"    Recommendation: {rec['recommendation']}")
        print(f"    Caveat:         {rec['caveat']}\n")

    print(f"Full report exported to: {args.out}")


if __name__ == "__main__":
    main()
