"""
File: dashboard/components.py

1. What this file does:
   Provides reusable presentation widgets, charts, styled KPI metric cards,
   and color-coded badge components for the Streamlit GPU memory dashboard.

2. Overall design:
   A modular visualization component library built with Streamlit and Altair.
   Decouples visual styling, theme-aware color palettes, and plot formatting from
   the main dashboard page layout in `dashboard/app.py`. All memory numbers are
   formatted with explicit units (MiB/GiB), and all risk/admission indicators use
   clear semantic coloring (green, amber, red).

3. Elements in the file:
   - `apply_custom_css`: Injects clean typography and responsive layout styles.
   - `render_kpi_card`: Renders a bordered metric card with title, value, and subtitle.
   - `render_risk_badge`: Renders an educational, color-coded OOM-risk indicator.
   - `create_vram_timeline_chart`: Device-wide total vs used vs free timeline (Altair).
   - `create_torch_alloc_chart`: In-process Allocated vs Reserved timeline (Altair).
   - `create_contention_stacked_chart`: Multi-PID memory consumption area chart (Altair).
   - `create_block_histogram`: Allocator free block size distribution histogram (Altair).
   - `render_recommendation_card`: Formatted traceable recommendation card.
   - `render_admission_decision_badge`: Visual pill for ADMIT, DEFER, or REJECT.

4. Final output of the file:
   Interactive, styled UI components embedded into Streamlit tabs.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.normalize import format_bytes, bytes_to_mb, bytes_to_gb


def apply_custom_css():
    """
    1. The use of function:
       Injects custom CSS styling for sleek, modern dashboard presentation.
    2. How does the specific function work:
       Uses `st.markdown` with `unsafe_allow_html=True` to style cards, metrics,
       tables, and badges.
    3. Concepts used in the function:
       CSS layout styling and visual hierarchy.
    4. Places where this function is imported and used:
       Called at the top of `dashboard/app.py`.
    5. Final output of the function, before and after:
       Before: Default unstyled Streamlit theme.
       After: Polished cards with subtle borders, shadows, and status badges.
    """
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .metric-card-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #888888;
            margin-bottom: 6px;
            font-weight: 600;
        }
        .metric-card-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 4px;
            color: #00d2ff;
        }
        .metric-card-subtext {
            font-size: 0.8rem;
            color: #aaaaaa;
        }
        .risk-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 14px;
            font-weight: 700;
            font-size: 0.85rem;
            text-align: center;
        }
        .risk-low {
            background-color: rgba(46, 204, 113, 0.2);
            color: #2ecc71;
            border: 1px solid #2ecc71;
        }
        .risk-moderate {
            background-color: rgba(243, 156, 18, 0.2);
            color: #f39c12;
            border: 1px solid #f39c12;
        }
        .risk-high {
            background-color: rgba(231, 76, 60, 0.2);
            color: #e74c3c;
            border: 1px solid #e74c3c;
        }
        .rec-card {
            background-color: rgba(255, 255, 255, 0.04);
            border-left: 4px solid #00d2ff;
            border-radius: 4px 8px 8px 4px;
            padding: 14px 18px;
            margin-bottom: 14px;
        }
        .rec-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 6px;
        }
        .rec-pill {
            background-color: rgba(0, 210, 255, 0.15);
            color: #00d2ff;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 8px;
        }
        .caveat-box {
            background-color: rgba(243, 156, 18, 0.08);
            border: 1px dashed rgba(243, 156, 18, 0.3);
            border-radius: 6px;
            padding: 8px 12px;
            margin-top: 8px;
            font-size: 0.82rem;
            color: #e0a84c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str,
    value: str,
    subtext: str = "",
    color: str = "#00d2ff"
):
    """
    1. The use of function:
       Renders a styled KPI card in Streamlit.
    2. How does the specific function work:
       Emits HTML with class `metric-card`, title, formatted value, and subtext.
    3. Concepts used in the function:
       HTML embedding inside Streamlit markdown.
    4. Places where this function is imported and used:
       Used across all dashboard tabs for primary summary numbers.
    5. Final output of the function, before and after:
       Before: Plain text or generic st.metric.
       After: Uniform, custom-styled KPI card.
    """
    html = f"""
    <div class="metric-card">
        <div class="metric-card-title">{title}</div>
        <div class="metric-card-value" style="color: {color};">{value}</div>
        <div class="metric-card-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_risk_badge(risk_level: str, headroom_pct: float, growth_rate_mb: float):
    """
    1. The use of function:
       Renders an informative, color-coded OOM risk badge and explanatory text.
    2. How does the specific function work:
       Selects CSS class and icon based on risk_level (Low, Moderate, High),
       displaying headroom and growth rate thresholds.
    3. Concepts used in the function:
       Visual status signaling and rule transparency.
    4. Places where this function is imported and used:
       Used in the Memory Pressure tab.
    5. Final output of the function, before and after:
       Before: Raw string "High".
       After: Styled colored pill with transparent rationale.
    """
    norm_risk = str(risk_level).strip().capitalize()
    if norm_risk == "High":
        css_class = "risk-high"
        icon = "🔴"
        desc = "Headroom critically depleted (< 10%) or rapid memory surge (> 50 MB/s)."
    elif norm_risk == "Moderate":
        css_class = "risk-moderate"
        icon = "🟡"
        desc = "Headroom approaching safety margin (< 25%) or positive allocation growth."
    else:
        css_class = "risk-low"
        icon = "🟢"
        desc = "Headroom comfortable (> 25% of budget) and memory growth stable."

    html = f"""
    <div style="margin: 8px 0 16px 0;">
        <span class="risk-badge {css_class}">{icon} {norm_risk.upper()} OOM RISK</span>
        <div style="font-size: 0.82rem; color: #888888; margin-top: 6px;">
            <strong>Heuristic Policy:</strong> {desc}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def create_vram_timeline_chart(df: pd.DataFrame) -> alt.Chart:
    """
    1. The use of function:
       Builds an interactive Altair chart of device-wide NVML VRAM over time.
    2. How does the specific function work:
       Plots Total VRAM (reference line), Used VRAM (filled area), and Free VRAM.
    3. Concepts used in the function:
       Declarative multi-series time-series visualization.
    4. Places where this function is imported and used:
       Used in Tab 1 (Device Overview).
    5. Final output of the function, before and after:
       Before: Raw DataFrame.
       After: Altair Chart object ready for `st.altair_chart`.
    """
    if df.empty or not any(col in df.columns for col in ["device_used_mb", "device_free_mb"]):
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(text="No device VRAM timeline data available")

    time_col = "elapsed_s" if "elapsed_s" in df.columns else ("timestamp" if "timestamp" in df.columns else df.columns[0])

    # Filter to NVML source rows or rows with device_used_mb
    plot_df = df[df["device_used_mb"].notna()].copy() if "device_used_mb" in df.columns else df.copy()
    if plot_df.empty:
        plot_df = df.copy()

    val_cols = [col for col in ["device_used_mb", "device_free_mb"] if col in plot_df.columns]
    if not val_cols:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(text="No device VRAM timeline data available")

    melted = plot_df.melt(
        id_vars=[time_col],
        value_vars=val_cols,
        var_name="Metric",
        value_name="Memory_MiB",
    )
    melted["Metric"] = melted["Metric"].map({
        "device_used_mb": "Used VRAM (MiB)",
        "device_free_mb": "Free VRAM (MiB)",
    }).fillna("Memory (MiB)")

    color_scale = alt.Scale(
        domain=["Used VRAM (MiB)", "Free VRAM (MiB)"],
        range=["#e74c3c", "#2ecc71"],
    )

    chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(f"{time_col}:Q", title="Elapsed Time (seconds)"),
            y=alt.Y("Memory_MiB:Q", title="Device Memory (MiB)"),
            color=alt.Color("Metric:N", scale=color_scale),
            tooltip=[
                alt.Tooltip(f"{time_col}:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Memory_MiB:Q", title="Memory (MiB)", format=".1f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    return chart


def create_torch_alloc_chart(df: pd.DataFrame) -> alt.Chart:
    """
    1. The use of function:
       Builds an interactive Altair chart comparing in-process Allocated vs Reserved memory.
    2. How does the specific function work:
       Renders lines for Allocated bytes (live tensors) and Reserved bytes (allocator pool).
       Shades the area between Reserved and Allocated to illustrate cached-but-unused memory.
    3. Concepts used in the function:
       Caching allocator visualization, OS virtual memory vs resident memory analogy.
    4. Places where this function is imported and used:
       Used in Tab 2 (In-Process Allocator & Pressure).
    5. Final output of the function, before and after:
       Before: Raw DataFrame.
       After: Layered Altair Chart object.
    """
    if df.empty or not any(col in df.columns for col in ["allocated_mb", "reserved_mb"]):
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(text="No in-process allocation timeline data available")

    time_col = "elapsed_s" if "elapsed_s" in df.columns else ("timestamp" if "timestamp" in df.columns else df.columns[0])
    plot_df = df[df["allocated_mb"].notna()].copy() if "allocated_mb" in df.columns else df.copy()
    if plot_df.empty:
        plot_df = df.copy()

    val_cols = [col for col in ["allocated_mb", "reserved_mb"] if col in plot_df.columns]
    if not val_cols:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(text="No in-process allocation timeline data available")

    melted = plot_df.melt(
        id_vars=[time_col],
        value_vars=val_cols,
        var_name="Metric",
        value_name="Memory_MiB",
    )
    melted["Metric"] = melted["Metric"].map({
        "allocated_mb": "Allocated Memory (Live Tensors)",
        "reserved_mb": "Reserved Memory (Allocator Pool)",
    }).fillna("Memory (MiB)")

    color_scale = alt.Scale(
        domain=["Allocated Memory (Live Tensors)", "Reserved Memory (Allocator Pool)"],
        range=["#00d2ff", "#9b59b6"],
    )

    chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(f"{time_col}:Q", title="Elapsed Time (seconds)"),
            y=alt.Y("Memory_MiB:Q", title="Allocator Memory (MiB)"),
            color=alt.Color("Metric:N", scale=color_scale),
            tooltip=[
                alt.Tooltip(f"{time_col}:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Memory_MiB:Q", title="Memory (MiB)", format=".1f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    return chart


def create_contention_stacked_chart(df: pd.DataFrame) -> alt.Chart:
    """
    1. The use of function:
       Renders a stacked area chart showing concurrent per-PID GPU memory consumption.
    2. How does the specific function work:
       Extracts records with positive PIDs, calculates allocated MB, and builds
       a stacked area chart grouped by `PID (Workload Name)`.
    3. Concepts used in the function:
       Multi-process resource contention visualization.
    4. Places where this function is imported and used:
       Used in Tab 4 (Contention & Concurrency).
    5. Final output of the function, before and after:
       Before: Multi-PID DataFrame.
       After: Stacked Altair Chart.
    """
    time_col = "elapsed_s" if "elapsed_s" in df.columns else "timestamp"
    valid = df[(df["pid"] > 0) & (df["allocated_mb"].notna())].copy()
    if valid.empty:
        valid = df[(df["pid"] > 0) & (df["process_memory"].notna())].copy()
        if not valid.empty:
            valid["allocated_mb"] = valid["process_memory"] / (1024.0 * 1024.0)

    if valid.empty:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(text="No per-PID data")

    # Format process label: "PID 10101 (ResNet-50)"
    def make_label(row):
        wl = row.get("workload_name")
        if pd.notna(wl) and str(wl).strip():
            return f"PID {int(row['pid'])} ({wl})"
        return f"PID {int(row['pid'])}"

    valid["Process"] = valid.apply(make_label, axis=1)

    chart = (
        alt.Chart(valid)
        .mark_area(opacity=0.75)
        .encode(
            x=alt.X(f"{time_col}:Q", title="Elapsed Time (seconds)"),
            y=alt.Y("allocated_mb:Q", title="GPU Memory (MiB)", stack="zero"),
            color=alt.Color("Process:N", scale=alt.Scale(scheme="category10")),
            tooltip=[
                alt.Tooltip(f"{time_col}:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("Process:N"),
                alt.Tooltip("allocated_mb:Q", title="Memory (MiB)", format=".1f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )
    return chart


def create_block_histogram(blocks: List[Dict[str, Any]]) -> alt.Chart:
    """
    1. The use of function:
       Generates a bar histogram of allocator block sizes from a PyTorch snapshot.
    2. How does the specific function work:
       Filters blocks, computes block size in MiB, and groups by state
       (active_allocated vs inactive free blocks).
    3. Concepts used in the function:
       Free block size distribution and fragmentation analysis.
    4. Places where this function is imported and used:
       Used in Tab 3 (Allocator Fragmentation).
    5. Final output of the function, before and after:
       Before: List of raw snapshot block dicts.
       After: Altair histogram chart.
    """
    if not blocks:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(text="No block data available")

    rows = []
    for idx, b in enumerate(blocks):
        size_mb = float(b.get("size", 0)) / (1024.0 * 1024.0)
        state = b.get("state", "unknown")
        rows.append({
            "block_index": idx,
            "size_mb": round(size_mb, 2),
            "state": "Active (Allocated)" if "allocated" in state else "Inactive (Free Space)",
        })

    bdf = pd.DataFrame(rows)
    color_scale = alt.Scale(
        domain=["Active (Allocated)", "Inactive (Free Space)"],
        range=["#3498db", "#e67e22"],
    )

    chart = (
        alt.Chart(bdf)
        .mark_bar()
        .encode(
            x=alt.X("block_index:O", title="Block Index"),
            y=alt.Y("size_mb:Q", title="Block Size (MiB)"),
            color=alt.Color("state:N", scale=color_scale, title="Block State"),
            tooltip=["block_index:O", "state:N", "size_mb:Q"],
        )
        .properties(height=260)
    )
    return chart


def render_recommendation_card(rec: Dict[str, Any]):
    """
    1. The use of function:
       Renders an individual traceable recommendation card in Streamlit.
    2. How does the specific function work:
       Renders HTML with Rule ID, Category, Triggering Metric, Measured Value,
       Threshold, actionable recommendation, and cautionary caveats.
    3. Concepts used in the function:
       Evidence-backed recommendation presentation and traceability.
    4. Places where this function is imported and used:
       Used in Tab 5 (Optimization & Recommendations).
    5. Final output of the function, before and after:
       Before: Raw dictionary.
       After: Styled HTML card on page.
    """
    rule_id = rec.get("rule_id", "RULE-OPT")
    category = rec.get("category", "General Optimization")
    trigger = rec.get("triggering_metric", "N/A")
    measured = rec.get("measured_value", "N/A")
    threshold = rec.get("threshold", "N/A")
    recommendation = rec.get("recommendation", "")
    caveat = rec.get("caveat", "")

    html = f"""
    <div class="rec-card">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <span class="rec-pill">{rule_id}</span>
            <span style="font-weight: 600; font-size: 0.9rem; color: #888888;">{category}</span>
        </div>
        <div class="rec-title">{recommendation}</div>
        <div style="font-size: 0.82rem; color: #cccccc; margin-top: 6px;">
            <strong>Trigger:</strong> <code>{trigger}</code> = <strong>{measured}</strong> (Threshold: {threshold})
        </div>
        <div class="caveat-box">
            <strong>⚠️ Caveat:</strong> {caveat}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_admission_decision_badge(decision: str) -> str:
    """
    1. The use of function:
       Returns an HTML badge string for ADMIT, DEFER, or REJECT decisions.
    2. How does the specific function work:
       Maps decision string to color-coded badge HTML.
    3. Concepts used in the function:
       Visual status encoding.
    4. Places where this function is imported and used:
       Used in Admission Decision table views.
    5. Final output of the function, before and after:
       Before: "ADMIT"
       After: '<span style="...">ADMIT</span>'
    """
    d = str(decision).strip().upper()
    if d == "ADMIT":
        return '<span style="color: #2ecc71; font-weight: 700;">🟢 ADMIT</span>'
    elif d == "DEFER":
        return '<span style="color: #f39c12; font-weight: 700;">🟡 DEFER</span>'
    elif d == "REJECT":
        return '<span style="color: #e74c3c; font-weight: 700;">🔴 REJECT</span>'
    return f'<span style="color: #888888;">{d}</span>'
