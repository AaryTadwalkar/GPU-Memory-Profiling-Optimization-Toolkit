"""
File: dashboard/__init__.py

1. What this file does:
   Initializes the `dashboard` package and exposes UI helper components.
2. Overall design:
   Standard Python package initializer.
3. Elements in the file:
   Exposes functions from `dashboard.components`.
4. Final output of the file:
   Package namespace for dashboard modules.
"""

from dashboard.components import (
    apply_custom_css,
    create_block_histogram,
    create_contention_stacked_chart,
    create_torch_alloc_chart,
    create_vram_timeline_chart,
    render_admission_decision_badge,
    render_kpi_card,
    render_recommendation_card,
    render_risk_badge,
)

__all__ = [
    "apply_custom_css",
    "create_block_histogram",
    "create_contention_stacked_chart",
    "create_torch_alloc_chart",
    "create_vram_timeline_chart",
    "render_admission_decision_badge",
    "render_kpi_card",
    "render_recommendation_card",
    "render_risk_badge",
]
