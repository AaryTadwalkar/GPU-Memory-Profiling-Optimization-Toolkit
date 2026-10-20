"""
File: batch_optimizer.py

1. What this file does:
   Models and evaluates batch size scaling to determine the largest viable configuration
   under a specified GPU memory budget.
2. Overall design:
   A lightweight batch optimization module that uses linear memory scaling models
   (base weight/context memory + sample activation memory) and empirical benchmark
   metrics to evaluate candidate batch sizes against budget and safety constraints.
3. Elements in the file:
   - BatchCandidate: Dataclass capturing evaluated configuration metrics.
   - BatchOptimizer: Class providing estimation, candidate evaluation, and selection.
4. Final output of the file:
   Ranked candidate configurations and the recommended best-fit batch size under budget.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class BatchCandidate:
    """
    1. The use of class:
       Stores the memory projection and viability status for a specific batch size candidate.
    2. What the class contain:
       batch_size, estimated_peak_bytes, headroom_bytes, peak_to_budget_ratio, status, recommendation_note.
    3. Technologies used and why:
       Python dataclass for straightforward serialization and structure.
    4. Key design and overall design of the class:
       Standardizes configuration evaluation for optimizer analysis and dashboard display.
    """
    batch_size: int
    estimated_peak_bytes: int
    headroom_bytes: int
    peak_to_budget_ratio: float
    status: str  # "VIABLE" | "RISKY" | "EXCEEDS_BUDGET"
    recommendation_note: str

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Converts candidate object to a dictionary.
        2. How does the specific function work:
           Calls dataclasses.asdict on self.
        3. Concepts used in the function:
           Serialization.
        4. Places where this function is imported and used:
           Used by BatchOptimizer and downstream consumers.
        5. Final output of the function, before and after:
           Before: BatchCandidate instance. After: Dictionary.
        """
        return asdict(self)


class BatchOptimizer:
    """
    1. The use of class:
       Evaluates candidate batch sizes against a hardware memory budget.
    2. What the class contain:
       Budget bytes, safety margin percentage, estimation functions, candidate ranking methods.
    3. Technologies used and why:
       Standard library math and data structures for fast, deterministic evaluation.
    4. Key design and overall design of the class:
       Implements cooperative memory estimation: Base_Memory + (Batch_Size * Activation_Per_Sample),
       classifying candidates into VIABLE, RISKY, or EXCEEDS_BUDGET, without claiming a global optimum.
    """
    def __init__(self, budget_bytes: int = 6 * 1024 * 1024 * 1024, safety_margin_ratio: float = 0.15):
        self.budget_bytes = budget_bytes
        self.safety_margin_ratio = safety_margin_ratio

    def evaluate_candidate(
        self,
        batch_size: int,
        base_bytes: int,
        activation_per_sample_bytes: int
    ) -> BatchCandidate:
        """
        1. The use of function:
           Evaluates a single candidate batch size using linear memory projection.
        2. How does the specific function work:
           Calculates projected peak memory, compares against budget, and assigns status.
        3. Concepts used in the function:
           Linear scaling modeling, safety thresholding.
        4. Places where this function is imported and used:
           Called in evaluate_sweep.
        5. Final output of the function, before and after:
           Before: Raw integer inputs. After: Evaluated BatchCandidate instance.
        """
        projected_peak = base_bytes + (batch_size * activation_per_sample_bytes)
        headroom = self.budget_bytes - projected_peak
        ratio = projected_peak / self.budget_bytes if self.budget_bytes > 0 else 0.0

        safe_threshold = 1.0 - self.safety_margin_ratio

        if projected_peak > self.budget_bytes:
            status = "EXCEEDS_BUDGET"
            note = f"Exceeds memory budget by {(projected_peak - self.budget_bytes) / (1024*1024):.1f} MB; high OOM probability."
        elif ratio > safe_threshold:
            status = "RISKY"
            note = f"Consumes {ratio:.1%} of budget (headroom < {self.safety_margin_ratio:.0%}); vulnerable to allocation spikes."
        else:
            status = "VIABLE"
            note = f"Fits within safe budget headroom ({headroom / (1024*1024):.1f} MB free)."

        return BatchCandidate(
            batch_size=batch_size,
            estimated_peak_bytes=projected_peak,
            headroom_bytes=headroom,
            peak_to_budget_ratio=ratio,
            status=status,
            recommendation_note=note
        )

    def evaluate_sweep(
        self,
        base_bytes: int,
        activation_per_sample_bytes: int,
        candidates: Optional[List[int]] = None
    ) -> List[BatchCandidate]:
        """
        1. The use of function:
           Evaluates a series of candidate batch sizes.
        2. How does the specific function work:
           Iterates over candidate list and executes evaluate_candidate for each.
        3. Concepts used in the function:
           Batch sweep evaluation.
        4. Places where this function is imported and used:
           Used by optimizer workflows and experiment comparisons.
        5. Final output of the function, before and after:
           Before: Candidate batch list. After: List of evaluated BatchCandidate objects.
        """
        if candidates is None:
            candidates = [1, 2, 4, 8, 16, 32, 64, 128]

        results = []
        for b in candidates:
            cand = self.evaluate_candidate(b, base_bytes, activation_per_sample_bytes)
            results.append(cand)
        return results

    def select_best_viable_candidate(self, evaluated: List[BatchCandidate]) -> Optional[BatchCandidate]:
        """
        1. The use of function:
           Selects the highest throughput (largest batch size) viable candidate.
        2. How does the specific function work:
           Filters candidates with status == 'VIABLE' and selects the one with maximum batch size.
        3. Concepts used in the function:
           Best-fit selection under constraint.
        4. Places where this function is imported and used:
           Called when determining final recommendation for batch configuration.
        5. Final output of the function, before and after:
           Before: List of BatchCandidates. After: The single highest viable candidate or None.
        """
        viable = [c for c in evaluated if c.status == "VIABLE"]
        if not viable:
            return None
        return max(viable, key=lambda c: c.batch_size)
