"""
File: optimizer/__init__.py

1. What this file does:
   Marks the optimizer package as a Python module and exposes key recommendation, batch
   optimization, tiered activation swapping, and paged KV-cache management utilities.

2. Overall design:
   Standard Python package initializer providing clean, centralized imports for optimization
   recommendations, batch sizing search engines, runtime autograd activation swapping, and
   OS-inspired paged KV-cache virtual token management.

3. Elements in the file:
   - RecommendationEngine: Rule-based recommendation engine for GPU memory bottlenecks.
   - OptimizationRecommendation: Dataclass capturing concrete optimization actions.
   - BatchOptimizer: Search engine optimizing training batch sizes for target memory bounds.
   - SwapEventRecord: Dataclass recording individual activation offload and prefetch events.
   - ActivationSwapSummary: Dataclass summarizing aggregate activation swapping telemetry.
   - TieredActivationManager: Context manager orchestrating tiered activation swapping.
   - PackedActivation: Immutable tuple subclass holding packed activation metadata.
   - PhysicalBlock: Dataclass representing a fixed-capacity physical frame for KV-cache tokens.
   - BlockTable: Logical-to-physical address translation table for generation sequences.
   - SequenceDescriptor: Dataclass modeling an active generation sequence request.
   - WasteComparisonMetrics: Dataclass summarizing fragmentation and savings vs traditional allocation.
   - PagedKVCacheManager: Central OS-inspired virtual memory manager orchestrating paged KV-cache state.

4. Final output of the file:
   Exported classes and utilities for the optimizer package.
"""

from optimizer.recommendations import RecommendationEngine, OptimizationRecommendation
from optimizer.batch_optimizer import BatchOptimizer
from optimizer.activation_swapper import (
    SwapEventRecord,
    ActivationSwapSummary,
    TieredActivationManager,
    PackedActivation,
)
from optimizer.paged_cache import (
    PhysicalBlock,
    BlockTable,
    SequenceDescriptor,
    WasteComparisonMetrics,
    PagedKVCacheManager,
)

__all__ = [
    "RecommendationEngine",
    "OptimizationRecommendation",
    "BatchOptimizer",
    "SwapEventRecord",
    "ActivationSwapSummary",
    "TieredActivationManager",
    "PackedActivation",
    "PhysicalBlock",
    "BlockTable",
    "SequenceDescriptor",
    "WasteComparisonMetrics",
    "PagedKVCacheManager",
]
