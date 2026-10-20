"""
File: __init__.py

1. What this file does:
   Marks the optimizer package as a Python module and exposes key recommendation and optimization utilities.
2. Overall design:
   Standard Python package initializer providing clean imports for recommendation generation and batch optimization.
3. Elements in the file:
   - Package-level imports for RecommendationEngine and BatchOptimizer.
4. Final output of the file:
   Exported classes and functions for the optimizer package.
"""

from optimizer.recommendations import RecommendationEngine, OptimizationRecommendation
from optimizer.batch_optimizer import BatchOptimizer

__all__ = ["RecommendationEngine", "OptimizationRecommendation", "BatchOptimizer"]
