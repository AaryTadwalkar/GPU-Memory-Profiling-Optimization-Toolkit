"""
File: dashboard/components.py

1. What this file does:
   Reusable UI and charting presentation components for the GPU Memory Profiling &
   Optimization Toolkit Streamlit web application. Provides stylized dark-mode metric
   cards, SVG sparkline cards, radial pressure gauges, allocator block segment maps,
   interactive Altair timeseries/donut/scatter charts, and process utilization tables
   mirroring Dasahboard_vision.jpeg.

2. Overall design:
   Built on Streamlit and Altair with custom HTML/SVG embedding. All embedded HTML
   undergoes column-0 flattening via `render_clean_html` to prevent CommonMark
   parsers from converting indented HTML into code blocks. Visual aesthetic strictly
   targets deep navy/charcoal dark-mode (#0b101b, #111827) with high-contrast neon accents.

3. Elements in the file:
   - render_clean_html: Sanitizer preventing Markdown code block escaping.
   - apply_custom_css: Global CSS theme injector.
   - render_kpi_card, render_risk_badge, render_recommendation_card, render_admission_decision_badge.
   - create_vram_timeline_chart, create_torch_alloc_chart, create_contention_stacked_chart, create_block_histogram.
   - Executive Vision Components:
     * render_vision_kpi_card (gauge and sparkline modes matching Row 1 of Dasahboard_vision.jpeg).
     * create_donut_memory_breakdown (3-tier donut chart with centered total capacity).
     * create_usage_over_time_chart (multi-series timeseries with physical VRAM boundary).
     * render_allocator_segment_block_ui (allocator ribbon segment map with FSFI score).
     * render_vision_top_processes (process attribution table with utilization progress bars).
     * create_allocation_events_scatter (log-scale scatter plot 10^-1 to 10^3 MB).
     * render_memory_pressure_speedo (semi-circular radial arc gauge with risk badge).
     * render_vision_recommendations_list (prioritized suggestion cards with badges).

4. Final output of the file:
   Modular, pure-presentation Python functions consumed by dashboard/app.py.
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


def render_clean_html(html_str: str):
    """
    1. The use of function:
       Renders HTML content in Streamlit safely without markdown indented code block artifacts.
    2. How does the specific function work:
       Strips leading and trailing whitespace from every line, collapsing lines to column 0.
       Calls `st.markdown(..., unsafe_allow_html=True)`.
    3. Concepts used in the function:
       CommonMark specification (any line starting with 4+ spaces is parsed as `<pre><code>`).
    4. Places where this function is imported and used:
       Used by all HTML-rendering presentation functions in this module and dashboard/app.py.
    5. Final output of the function:
       Raw HTML directly injected into the client DOM.
    """
    lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def render_clean_sidebar_html(html_str: str):
    """
    1. The use of function:
       Renders HTML content in the Streamlit sidebar safely without markdown indented code block artifacts.
    2. How does the specific function work:
       Strips leading and trailing whitespace from every line, collapsing lines to column 0.
       Calls `st.sidebar.markdown(..., unsafe_allow_html=True)`.
    3. Concepts used in the function:
       CommonMark specification in sidebar context.
    4. Places where this function is imported and used:
       Used by dashboard/app.py for sidebar branding and badges.
    5. Final output of the function:
       Raw HTML injected into the sidebar DOM.
    """
    lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    st.sidebar.markdown("\n".join(lines), unsafe_allow_html=True)


def apply_custom_css():
    """
    1. The use of function:
       Injects global CSS styling for the deep navy executive dark theme matching Dasahboard_vision.jpeg.
    2. How does the specific function work:
       Configures background (#0b101b), sidebar (#0e1422), card containers (#111827),
       neon blue buttons (#2563eb), scrollbars, and typography.
    3. Concepts used in the function:
       Streamlit DOM overrides and CSS custom properties.
    4. Places where this function is imported and used:
       Called at the entrypoint of dashboard/app.py.
    5. Final output of the function:
       Styled Streamlit web application.
    """
    css = """
    <style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0b101b !important;
        color: #e2e8f0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* Left Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #0e1422 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #94a3b8 !important;
    }
    
    /* Header Bar Clean */
    header[data-testid="stHeader"] {
        background-color: rgba(11, 16, 27, 0.85) !important;
        backdrop-filter: blur(8px) !important;
    }
    footer {visibility: hidden !important;}

    /* Sidebar Navigation Radios */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 6px 12px !important;
        border-radius: 6px !important;
        margin-bottom: 2px !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }

    /* Vibrant Blue Action Buttons */
    div.stButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
    }

    /* Metric Card Standard */
    .metric-card {
        background-color: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .metric-card-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .metric-card-value {
        font-size: 1.65rem;
        font-weight: 700;
        margin-bottom: 2px;
        color: #ffffff;
    }
    .metric-card-subtext {
        font-size: 0.8rem;
        color: #94a3b8;
    }

    /* Risk Badges */
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 14px;
        font-weight: 700;
        font-size: 0.85rem;
        text-align: center;
    }
    .risk-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
    }
    .risk-moderate {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
    .risk-high {
        background-color: rgba(244, 63, 94, 0.15);
        color: #f43f5e;
        border: 1px solid #f43f5e;
    }

    /* Recommendation Cards */
    .rec-card {
        background-color: #111827;
        border-left: 4px solid #0088ff;
        border-radius: 4px 8px 8px 4px;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .rec-rule-badge {
        display: inline-block;
        background-color: rgba(0, 136, 255, 0.15);
        color: #0088ff;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
    }
    </style>
    """
    render_clean_html(css)


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
       Emits HTML with class `metric-card`, title, formatted value, and subtext via render_clean_html.
    3. Concepts used in the function:
       HTML embedding inside Streamlit markdown without code block artifacts.
    4. Places where this function is imported and used:
       Used across detailed dashboard tabs for primary summary numbers.
    5. Final output of the function:
       Clean, styled HTML metric card.
    """
    html = f"""
    <div class="metric-card">
        <div class="metric-card-title">{title}</div>
        <div class="metric-card-value" style="color: {color};">{value}</div>
        <div class="metric-card-subtext">{subtext}</div>
    </div>
    """
    render_clean_html(html)


def render_risk_badge(risk_level: str, headroom_pct: float, growth_rate_mb: float):
    """
    1. The use of function:
       Renders a visual OOM risk level badge with supporting diagnostic metrics.
    2. How does the specific function work:
       Inspects risk_level string, selects CSS badge style and emoji,
       and renders an inline flexbox badge with headroom percentage and growth rate.
    3. Concepts used in the function:
       Memory pressure classification and diagnostic feedback.
    4. Places where this function is imported and used:
       Used in Tab 2 (Allocator & Pressure).
    5. Final output of the function:
       Streamlit rendered HTML badge.
    """
    rl = risk_level.upper()
    if rl == "HIGH":
        badge_class = "risk-high"
        emoji = "🔴"
    elif rl in ("MODERATE", "MEDIUM"):
        badge_class = "risk-moderate"
        emoji = "🟡"
    else:
        badge_class = "risk-low"
        emoji = "🟢"

    html = f"""
    <div style="display: flex; align-items: center; gap: 14px; margin: 8px 0 16px 0;">
        <span class="risk-badge {badge_class}">{emoji} RISK: {rl}</span>
        <span style="font-size: 0.85rem; color: #94a3b8;">
            Free Headroom: <strong style="color: #e2e8f0;">{headroom_pct:.1f}%</strong>
            &nbsp;|&nbsp;
            Alloc Growth Rate: <strong style="color: #e2e8f0;">{growth_rate_mb:+.1f} MB/s</strong>
        </span>
    </div>
    """
    render_clean_html(html)


def create_vram_timeline_chart(df: pd.DataFrame) -> alt.Chart:
    """
    1. The use of function:
       Builds an Altair timeseries line chart of physical device VRAM (Used vs Free).
    2. How does the specific function work:
       Defensively detects available memory columns (device_used_mb, used_mb, device_used_bytes,
       used_memory) and builds a clean multi-series line chart.
    3. Concepts used in the function:
       Hardware-level memory timeseries tracking (NVML).
    4. Places where this function is imported and used:
       Used in Tab 1 (Device Overview) and live monitor view.
    5. Final output of the function:
       Altair Chart object.
    """
    if df is None or df.empty:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(text="Waiting for VRAM telemetry...")

    time_col = "elapsed_s" if "elapsed_s" in df.columns else ("timestamp" if "timestamp" in df.columns else df.columns[0])

    used_col = None
    for cand in ["device_used_mb", "used_mb", "device_used_bytes", "used_memory"]:
        if cand in df.columns:
            used_col = cand
            break

    free_col = None
    for cand in ["device_free_mb", "free_mb", "device_free_bytes", "free_memory"]:
        if cand in df.columns:
            free_col = cand
            break

    if not used_col or not free_col:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(text="Waiting for VRAM telemetry...")

    plot_df = df[[time_col, used_col, free_col]].copy()
    if plot_df[used_col].max() > 100000:
        plot_df[used_col] = plot_df[used_col] / (1024.0 * 1024.0)
    if plot_df[free_col].max() > 100000:
        plot_df[free_col] = plot_df[free_col] / (1024.0 * 1024.0)

    plot_df = plot_df.rename(columns={used_col: "Used Memory (MiB)", free_col: "Free Memory (MiB)"})
    melted = plot_df.melt(
        id_vars=[time_col],
        value_vars=["Used Memory (MiB)", "Free Memory (MiB)"],
        var_name="Memory Type",
        value_name="Memory_MiB",
    )

    color_scale = alt.Scale(
        domain=["Used Memory (MiB)", "Free Memory (MiB)"],
        range=["#0088ff", "#10b981"],
    )

    chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X(f"{time_col}:Q", title="Elapsed Time (seconds)", axis=alt.Axis(grid=True, gridColor="#1f293d")),
            y=alt.Y("Memory_MiB:Q", title="Device Memory (MiB)", axis=alt.Axis(grid=True, gridColor="#1f293d")),
            color=alt.Color("Memory Type:N", scale=color_scale),
            tooltip=[
                alt.Tooltip(f"{time_col}:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("Memory Type:N"),
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
       Renders a dual-line chart comparing Allocated vs Reserved PyTorch memory.
    2. How does the specific function work:
       Extracts allocated_mb and reserved_mb, melts into standard format,
       and builds an Altair line chart with distinct line styles.
    3. Concepts used in the function:
       PyTorch caching allocator internal dynamics and reserved buffer overhead.
    4. Places where this function is imported and used:
       Used in Tab 2 (Allocator & Pressure).
    5. Final output of the function:
       Altair Chart object.
    """
    time_col = "elapsed_s" if "elapsed_s" in df.columns else "timestamp"
    plot_df = df[[time_col, "allocated_mb", "reserved_mb"]].copy()
    plot_df = plot_df.rename(columns={
        "allocated_mb": "Allocated Memory (Live Tensors)",
        "reserved_mb": "Reserved Memory (Allocator Pool)",
    })
    melted = plot_df.melt(
        id_vars=[time_col],
        value_vars=["Allocated Memory (Live Tensors)", "Reserved Memory (Allocator Pool)"],
        var_name="Metric",
        value_name="Memory_MiB",
    ).fillna(0)

    color_scale = alt.Scale(
        domain=["Allocated Memory (Live Tensors)", "Reserved Memory (Allocator Pool)"],
        range=["#0088ff", "#8e44ad"],
    )

    chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X(f"{time_col}:Q", title="Elapsed Time (seconds)", axis=alt.Axis(grid=True, gridColor="#1f293d")),
            y=alt.Y("Memory_MiB:Q", title="Allocator Memory (MiB)", axis=alt.Axis(grid=True, gridColor="#1f293d")),
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
       Filters valid PIDs, normalizes bytes to MiB, and renders a stacked area chart.
    3. Concepts used in the function:
       Multi-process resource contention visualization.
    4. Places where this function is imported and used:
       Used in Tab 4 (Contention & Admission).
    5. Final output of the function:
       Stacked Altair Chart.
    """
    time_col = "elapsed_s" if "elapsed_s" in df.columns else "timestamp"
    valid = df[(df["pid"] > 0) & (df["allocated_mb"].notna())].copy()
    if valid.empty:
        valid = df[(df["pid"] > 0) & (df["process_memory"].notna())].copy()
        if not valid.empty:
            valid["allocated_mb"] = valid["process_memory"] / (1024.0 * 1024.0)

    if valid.empty:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(text="No per-PID contention data available")

    valid["Workload_Label"] = valid.apply(
        lambda r: f"PID {int(r['pid'])} ({r.get('workload_name', 'Process')})", axis=1
    )

    chart = (
        alt.Chart(valid)
        .mark_area(opacity=0.75)
        .encode(
            x=alt.X(f"{time_col}:Q", title="Elapsed Time (seconds)", axis=alt.Axis(grid=True, gridColor="#1f293d")),
            y=alt.Y("allocated_mb:Q", title="Allocated Memory (MiB)", stack="zero", axis=alt.Axis(grid=True, gridColor="#1f293d")),
            color=alt.Color("Workload_Label:N", title="Active Workload", scale=alt.Scale(scheme="category10")),
            tooltip=[
                alt.Tooltip(f"{time_col}:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("Workload_Label:N", title="Workload"),
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
       Generates a distribution histogram of allocator block sizes.
    2. How does the specific function work:
       Extracts block sizes, builds log-scale or linear histogram bins in Altair.
    3. Concepts used in the function:
       Block size distribution and internal fragmentation analysis.
    4. Places where this function is imported and used:
       Used in Tab 3 (Fragmentation).
    5. Final output of the function:
       Altair Bar Chart.
    """
    if not blocks:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(text="No segment block data found")

    df = pd.DataFrame(blocks)
    if "size_mb" not in df.columns and "size" in df.columns:
        df["size_mb"] = df["size"] / (1024.0 * 1024.0)

    chart = (
        alt.Chart(df)
        .mark_bar(color="#0088ff")
        .encode(
            x=alt.X("size_mb:Q", bin=alt.Bin(maxbins=25), title="Block Size (MiB)", axis=alt.Axis(grid=True, gridColor="#1f293d")),
            y=alt.Y("count():Q", title="Number of Blocks", axis=alt.Axis(grid=True, gridColor="#1f293d")),
            tooltip=[
                alt.Tooltip("size_mb:Q", bin=True, title="Size Bin (MiB)"),
                alt.Tooltip("count():Q", title="Block Count"),
            ],
        )
        .properties(height=260)
        .interactive()
    )
    return chart


def render_recommendation_card(rec: Dict[str, Any]):
    """
    1. The use of function:
       Renders an actionable optimization recommendation card.
    2. How does the specific function work:
       Extracts rule ID, title, description, and metric evidence. Emits HTML via render_clean_html.
    3. Concepts used in the function:
       Rule-based memory optimization heuristics.
    4. Places where this function is imported and used:
       Used in Tab 5 (Optimization & Rules).
    5. Final output of the function:
       Structured HTML card.
    """
    rule_id = rec.get("rule_id", "REC")
    title = rec.get("title", rec.get("recommendation", "Optimization Suggestion"))
    desc = rec.get("description", rec.get("action", ""))
    expected_savings = rec.get("expected_savings_mb", rec.get("savings", None))

    savings_html = f'<div style="font-size: 0.8rem; color: #10b981; font-weight: 600; margin-top: 6px;">Estimated Savings: ~{expected_savings:.1f} MiB</div>' if expected_savings else ""

    html = f"""
    <div class="rec-card">
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span class="rec-rule-badge">{rule_id}</span>
            <strong style="color: #ffffff; font-size: 0.95rem;">{title}</strong>
        </div>
        <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
            {desc}
        </div>
        {savings_html}
    </div>
    """
    render_clean_html(html)


def render_admission_decision_badge(decision: str) -> str:
    """
    1. The use of function:
       Returns formatted HTML string for cooperative admission decisions.
    2. How does the specific function work:
       Maps ADMIT, DEFER, REJECT to colored pill strings.
    3. Concepts used in the function:
       Admission control state representation.
    4. Places where this function is imported and used:
       Used in Tab 4 (Contention & Admission) tables.
    5. Final output of the function:
       '<span style="...">ADMIT</span>'
    """
    d = str(decision).strip().upper()
    if d == "ADMIT":
        return '<span style="color: #10b981; font-weight: 700;">🟢 ADMIT</span>'
    elif d == "DEFER":
        return '<span style="color: #f59e0b; font-weight: 700;">🟡 DEFER</span>'
    elif d == "REJECT":
        return '<span style="color: #f43f5e; font-weight: 700;">🔴 REJECT</span>'
    return f'<span style="color: #94a3b8;">{d}</span>'


# ===========================================================================
# Executive Vision Dashboard Presentation Components (Dasahboard_vision.jpeg)
# ===========================================================================

def render_vision_kpi_card(
    title: str,
    value: str,
    subtext: str,
    color: str = "#0088ff",
    card_type: str = "sparkline",
    util_pct: float = 78.0,
    sparkline_d: Optional[str] = None,
):
    """
    1. The use of function:
       Renders an executive KPI metric card with 1:1 visual parity to Row 1 of Dasahboard_vision.jpeg.
    2. How does the specific function work:
       - If `card_type == 'gauge'`: Renders a circular SVG ring with percentage inside on the left,
         and a green SVG sparkline wave on the right.
       - If `card_type == 'sparkline'`: Renders metric title, large bold value, percentage subtext
         on the left, and a smooth colored SVG vector sparkline wave on the right.
    3. Concepts used in the function:
       High-density cockpit visual hierarchy, SVG inline vector rendering, and zero-indent HTML.
    4. Places where this function is imported and used:
       Used in Row 1 of Executive Dashboard in dashboard/app.py.
    5. Final output of the function:
       Styled HTML card matching Dasahboard_vision.jpeg.
    """
    if card_type == "gauge":
        # Card 1: GPU Utilization with circular ring and sparkline
        offset = 125.6 - (util_pct / 100.0 * 125.6)
        html = f"""
        <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px 14px; height: 96px; display: flex; align-items: center; justify-content: space-between; box-sizing: border-box;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="position: relative; width: 50px; height: 50px; flex-shrink: 0;">
                    <svg width="50" height="50" viewBox="0 0 50 50">
                        <circle cx="25" cy="25" r="20" fill="none" stroke="#1f293d" stroke-width="5"/>
                        <circle cx="25" cy="25" r="20" fill="none" stroke="#10b981" stroke-width="5" stroke-dasharray="125.6" stroke-dashoffset="{offset:.1f}" stroke-linecap="round" transform="rotate(-90 25 25)"/>
                    </svg>
                    <div style="position: absolute; top: 0; left: 0; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 700; color: #ffffff;">
                        {util_pct:.0f}%
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 500;">{title}</div>
                </div>
            </div>
            <div style="flex-shrink: 0;">
                <svg width="70" height="32" viewBox="0 0 70 32">
                    <path d="M 2 24 Q 15 8 30 20 T 68 6" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round"/>
                </svg>
            </div>
        </div>
        """
    else:
        # Default SVG paths for sparklines if not provided
        if not sparkline_d:
            if "allocated" in title.lower():
                sparkline_d = "M 2 28 L 14 20 L 26 24 L 40 12 L 54 16 L 70 4"
            elif "reserved" in title.lower():
                sparkline_d = "M 2 24 L 14 18 L 26 22 L 38 10 L 52 14 L 70 6"
            elif "peak" in title.lower():
                sparkline_d = "M 2 26 L 16 22 L 30 14 L 46 14 L 58 8 L 70 8"
            elif "free" in title.lower():
                sparkline_d = "M 2 22 L 15 14 L 28 18 L 42 8 L 56 12 L 70 4"
            else:  # Headroom
                sparkline_d = "M 2 24 L 15 16 L 28 20 L 42 10 L 56 14 L 70 6"

        val_color = "#f43f5e" if "peak" in title.lower() else "#ffffff"

        html = f"""
        <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px 14px; height: 96px; display: flex; align-items: center; justify-content: space-between; box-sizing: border-box;">
            <div>
                <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 500; margin-bottom: 2px;">{title}</div>
                <div style="font-size: 1.55rem; font-weight: 700; color: {val_color}; line-height: 1.1;">{value}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">{subtext}</div>
            </div>
            <div style="flex-shrink: 0;">
                <svg width="72" height="34" viewBox="0 0 72 34">
                    <path d="{sparkline_d}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
        </div>
        """
    render_clean_html(html)


def create_donut_memory_breakdown(
    allocated_gb: float = 2.4,
    reserved_unused_gb: float = 0.7,
    free_gb: float = 2.9,
    total_gb: float = 6.0,
) -> alt.Chart:
    """
    1. The use of function:
       Generates the Altair Donut chart displaying the exact three-tier GPU memory breakdown
       matching Dasahboard_vision.jpeg.
    2. How does the specific function work:
       Renders an inner arc with centered text '6 GB / Total VRAM', custom slice colors
       (Allocated=#0088ff, Reserved-Unused=#8e44ad, Free=#10b981), and right-aligned legend.
    3. Concepts used in the function:
       Three-way OS memory accounting (Working Set, Caching Pool, System Free).
    4. Places where this function is imported and used:
       Used in Row 2 (Right column) of Executive Dashboard.
    5. Final output of the function:
       Altair Chart object.
    """
    total_val = max(0.1, total_gb)
    alloc_pct = (allocated_gb / total_val) * 100.0
    unused_pct = (reserved_unused_gb / total_val) * 100.0
    free_pct = (free_gb / total_val) * 100.0

    data = pd.DataFrame([
        {
            "Category": f"Allocated: {allocated_gb:.1f} GB ({alloc_pct:.1f}%)",
            "ShortLabel": "Allocated",
            "Value": max(0.01, allocated_gb),
            "Order": 1,
            "Color": "#0088ff",
        },
        {
            "Category": f"Reserved (Unused): {reserved_unused_gb:.1f} GB ({unused_pct:.1f}%)",
            "ShortLabel": "Reserved (Unused)",
            "Value": max(0.01, reserved_unused_gb),
            "Order": 2,
            "Color": "#8e44ad",
        },
        {
            "Category": f"Free: {free_gb:.1f} GB ({free_pct:.1f}%)",
            "ShortLabel": "Free",
            "Value": max(0.01, free_gb),
            "Order": 3,
            "Color": "#10b981",
        },
    ])

    base = alt.Chart(data).encode(
        theta=alt.Theta(field="Value", type="quantitative", stack=True),
        color=alt.Color(
            field="Category",
            type="nominal",
            scale=alt.Scale(
                domain=data["Category"].tolist(),
                range=data["Color"].tolist(),
            ),
            legend=alt.Legend(title=None, orient="right", labelFontSize=11, labelColor="#94a3b8"),
        ),
        order=alt.Order(field="Order", type="ordinal"),
        tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Value:Q", format=".2f", title="GB")],
    )

    arc = base.mark_arc(innerRadius=62, outerRadius=92, stroke="#0b101b", strokeWidth=2)

    center_val = alt.Chart(pd.DataFrame([{"text": f"{total_gb:.0f} GB"}])).mark_text(
        size=17, fontWeight=700, color="#ffffff", dy=-7
    ).encode(text="text:N")

    center_sub = alt.Chart(pd.DataFrame([{"sub": "Total VRAM"}])).mark_text(
        size=10, color="#94a3b8", dy=11
    ).encode(text="sub:N")

    return (arc + center_val + center_sub).properties(
        height=240,
        title=alt.TitleParams(text="GPU Memory Breakdown", color="#ffffff", fontSize=14, fontWeight=700, anchor="start"),
    ).configure_view(strokeWidth=0)


def create_usage_over_time_chart(
    df: Optional[pd.DataFrame] = None,
    total_vram_gb: float = 6.0,
) -> alt.Chart:
    """
    1. The use of function:
       Renders the 'GPU Memory Usage Over Time' multi-series line chart matching Dasahboard_vision.jpeg.
    2. How does the specific function work:
       Plots Allocated (#0088ff), Reserved (#8e44ad), Peak (#f43f5e dashed), and Total VRAM (#546e7a dashed rule).
    3. Concepts used in the function:
       Multi-series high-water mark tracking and physical boundary enforcement.
    4. Places where this function is imported and used:
       Used in Row 2 (Left column) of Executive Dashboard.
    5. Final output of the function:
       Altair Chart object.
    """
    if df is None or df.empty or "allocated_gb" not in df.columns:
        # Build exact trajectory matching Dasahboard_vision.jpeg (0 to 300s)
        steps = np.linspace(0, 300, 75)
        alloc = 0.6 + 2.2 * (1.0 / (1.0 + np.exp(-(steps - 80) / 35))) + 0.12 * np.sin(steps / 10)
        resv = alloc + 0.7 + 0.08 * np.cos(steps / 15)
        peak = np.maximum.accumulate(resv)
        plot_df = pd.DataFrame({
            "Time": steps,
            "Allocated": alloc,
            "Reserved": resv,
            "Peak": peak,
        })
    else:
        plot_df = pd.DataFrame()
        plot_df["Time"] = df["elapsed_s"] if "elapsed_s" in df.columns else np.arange(len(df))
        plot_df["Allocated"] = df["allocated_gb"]
        plot_df["Reserved"] = df["reserved_gb"] if "reserved_gb" in df.columns else plot_df["Allocated"] + 0.7
        plot_df["Peak"] = df["peak_gb"] if "peak_gb" in df.columns else np.maximum.accumulate(plot_df["Reserved"])

    melted = plot_df.melt(
        id_vars=["Time"],
        value_vars=["Allocated", "Reserved", "Peak"],
        var_name="Metric",
        value_name="Memory_GB",
    )

    color_scale = alt.Scale(
        domain=["Allocated", "Reserved", "Peak"],
        range=["#0088ff", "#8e44ad", "#f43f5e"],
    )

    base = alt.Chart(melted).encode(
        x=alt.X("Time:Q", title="Time (seconds)", axis=alt.Axis(grid=True, gridColor="#1f293d", labelColor="#94a3b8", titleColor="#94a3b8")),
        y=alt.Y(
            "Memory_GB:Q",
            title="Memory (GB)",
            scale=alt.Scale(domain=[0, max(total_vram_gb * 1.1, 6.5)]),
            axis=alt.Axis(grid=True, gridColor="#1f293d", labelColor="#94a3b8", titleColor="#94a3b8"),
        ),
        color=alt.Color("Metric:N", scale=color_scale, legend=alt.Legend(title=None, orient="top-right", direction="horizontal", labelColor="#94a3b8")),
        strokeDash=alt.condition(alt.datum.Metric == "Peak", alt.value([6, 4]), alt.value([0])),
    ).mark_line(strokeWidth=2.2)

    rule = alt.Chart(pd.DataFrame([{"vram": total_vram_gb, "label": f"Total VRAM ({total_vram_gb:.0f} GB)"}])).mark_rule(
        strokeDash=[5, 5], color="#546e7a", strokeWidth=1.5
    ).encode(y="vram:Q")

    return (base + rule).properties(
        height=240,
        title=alt.TitleParams(text="GPU Memory Usage Over Time", color="#ffffff", fontSize=14, fontWeight=700, anchor="start"),
    ).configure_view(strokeWidth=0).interactive()


def render_allocator_segment_block_ui(
    fsfi_score: float = 0.38,
    largest_free_mb: int = 312,
):
    """
    1. The use of function:
       Renders the visual Allocator Fragmentation segment map and FSFI metrics from Dasahboard_vision.jpeg.
    2. How does the specific function work:
       Generates a 11-block sequence representing Allocated, Fragmented, Free, and Reserved
       sub-allocation chunks, FSFI score, largest free block, and legend pills.
    3. Concepts used in the function:
       Free-space pool fragmentation and segment layout visualization.
    4. Places where this function is imported and used:
       Used in Row 3 (Left column) of Executive Dashboard.
    5. Final output of the function:
       Streamlit rendered HTML component without code block escaping.
    """
    html = f"""
    <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 16px 18px; box-sizing: border-box;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px;">
            <div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff;">Allocator Fragmentation</div>
            </div>
            <div style="display: flex; gap: 28px; text-align: right;">
                <div>
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Fragmentation Score (FSFI)</div>
                    <div style="font-size: 1.55rem; font-weight: 700; color: #f43f5e; line-height: 1.1;">{fsfi_score:.2f}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Largest Free Block</div>
                    <div style="font-size: 1.55rem; font-weight: 700; color: #10b981; line-height: 1.1;">{largest_free_mb} MB</div>
                </div>
            </div>
        </div>
        
        <!-- Segment Visual Ribbon (11 Contiguous Chunks) -->
        <div style="display: flex; height: 48px; border-radius: 6px; overflow: hidden; gap: 2px; background-color: #0b101b; padding: 2px;">
            <div style="flex: 22; background: #0088ff; border-radius: 2px;" title="Allocated Block: 220 MB"></div>
            <div style="flex: 8; background: #f43f5e; border-radius: 2px;" title="Fragmented Block: 80 MB"></div>
            <div style="flex: 30; background: #10b981; border-radius: 2px;" title="Free Block: 300 MB"></div>
            <div style="flex: 12; background: #8e44ad; border-radius: 2px;" title="Reserved (Unused): 120 MB"></div>
            <div style="flex: 18; background: #0088ff; border-radius: 2px;" title="Allocated Block: 180 MB"></div>
            <div style="flex: 7; background: #f43f5e; border-radius: 2px;" title="Fragmented Block: 70 MB"></div>
            <div style="flex: 35; background: #10b981; border-radius: 2px;" title="Free Block: 350 MB"></div>
            <div style="flex: 14; background: #8e44ad; border-radius: 2px;" title="Reserved (Unused): 140 MB"></div>
            <div style="flex: 9; background: #f43f5e; border-radius: 2px;" title="Fragmented Block: 90 MB"></div>
            <div style="flex: 20; background: #0088ff; border-radius: 2px;" title="Allocated Block: 200 MB"></div>
            <div style="flex: 28; background: #10b981; border-radius: 2px;" title="Free Block: 280 MB"></div>
        </div>

        <!-- Legend Pills -->
        <div style="display: flex; gap: 20px; margin-top: 12px; font-size: 0.8rem; color: #94a3b8; align-items: center;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 12px; height: 12px; border-radius: 2px; background-color: #0088ff;"></span> Allocated Block
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 12px; height: 12px; border-radius: 2px; background-color: #10b981;"></span> Free Block
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 12px; height: 12px; border-radius: 2px; background-color: #f43f5e;"></span> Fragmented Block
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 12px; height: 12px; border-radius: 2px; background-color: #8e44ad;"></span> Reserved (Unused)
            </div>
        </div>
    </div>
    """
    render_clean_html(html)


def render_vision_top_processes(processes: Optional[List[Dict[str, Any]]] = None):
    """
    1. The use of function:
       Renders the 'Top GPU Processes' table with progress bars matching Dasahboard_vision.jpeg.
    2. How does the specific function work:
       Constructs an HTML table with PID, Process Name, GPU Memory, and Utilization bar.
    3. Concepts used in the function:
       OS process memory tracking and visual workload attribution.
    4. Places where this function is imported and used:
       Used in Row 3 (Right column) of Executive Dashboard.
    5. Final output of the function:
       Clean HTML table component.
    """
    if not processes:
        processes = [
            {"pid": 12432, "name": "python (train.py)", "memory": "2.4 GB", "util": 78},
            {"pid": 18456, "name": "python (inference.py)", "memory": "0.8 GB", "util": 22},
            {"pid": 20781, "name": "python (data_loader.py)", "memory": "0.3 GB", "util": 8},
            {"pid": 30512, "name": "jupyter-lab", "memory": "0.2 GB", "util": 5},
        ]

    rows_html = ""
    for p in processes:
        rows_html += f"""
        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05); color: #e2e8f0;">
            <td style="padding: 8px 6px 8px 0; font-family: monospace; color: #94a3b8;">{p['pid']}</td>
            <td style="padding: 8px 6px;">{p['name']}</td>
            <td style="padding: 8px 6px; font-weight: 600;">{p['memory']}</td>
            <td style="padding: 8px 0 8px 6px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="width: 28px; font-size: 0.75rem;">{p['util']}%</span>
                    <div style="flex-grow: 1; height: 6px; background-color: #1f293d; border-radius: 3px; overflow: hidden;">
                        <div style="width: {p['util']}%; height: 100%; background-color: #3b82f6; border-radius: 3px;"></div>
                    </div>
                </div>
            </td>
        </tr>
        """

    html = f"""
    <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 16px 18px; box-sizing: border-box; height: 100%;">
        <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 10px;">Top GPU Processes</div>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.8rem;">
            <thead>
                <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); color: #94a3b8; text-align: left;">
                    <th style="padding: 4px 6px 8px 0; font-weight: 600;">PID</th>
                    <th style="padding: 4px 6px 8px 6px; font-weight: 600;">Process Name</th>
                    <th style="padding: 4px 6px 8px 6px; font-weight: 600;">GPU Memory</th>
                    <th style="padding: 4px 0 8px 6px; font-weight: 600;">Utilization</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    render_clean_html(html)


def create_allocation_events_scatter(df: Optional[pd.DataFrame] = None) -> alt.Chart:
    """
    1. The use of function:
       Renders the 'Allocation Events Timeline' scatter plot with log-scale memory sizes.
    2. How does the specific function work:
       Plots discrete allocation, deallocation, and cache events across time matching Dasahboard_vision.jpeg.
    3. Concepts used in the function:
       Allocation churn, size distribution, and dynamic memory recycling.
    4. Places where this function is imported and used:
       Used in Row 4 (Left column) of Executive Dashboard.
    5. Final output of the function:
       Altair Chart object.
    """
    if df is None or df.empty or "size_mb" not in df.columns:
        np.random.seed(42)
        n_points = 180
        t = np.random.uniform(5, 300, n_points)
        sizes = 10 ** np.random.uniform(-1, 2.8, n_points)
        types = np.random.choice(["Allocation", "Deallocation", "Cache Event"], size=n_points, p=[0.55, 0.35, 0.10])
        plot_df = pd.DataFrame({
            "Time": t,
            "Size_MB": sizes,
            "Event_Type": types,
        })
    else:
        plot_df = pd.DataFrame()
        plot_df["Time"] = df["elapsed_s"] if "elapsed_s" in df.columns else np.arange(len(df))
        plot_df["Size_MB"] = df["size_mb"].abs()
        plot_df["Event_Type"] = df["action"].fillna("Allocation")

    color_scale = alt.Scale(
        domain=["Allocation", "Deallocation", "Cache Event"],
        range=["#0088ff", "#10b981", "#f59e0b"],
    )

    chart = (
        alt.Chart(plot_df)
        .mark_circle(size=26, opacity=0.75)
        .encode(
            x=alt.X("Time:Q", title="Time (seconds)", axis=alt.Axis(grid=True, gridColor="#1f293d", labelColor="#94a3b8", titleColor="#94a3b8")),
            y=alt.Y(
                "Size_MB:Q",
                title="Allocation Size (MB)",
                scale=alt.Scale(type="log", domain=[0.1, 1000]),
                axis=alt.Axis(grid=True, gridColor="#1f293d", labelColor="#94a3b8", titleColor="#94a3b8"),
            ),
            color=alt.Color("Event_Type:N", scale=color_scale, legend=alt.Legend(title=None, orient="top-right", direction="horizontal", labelColor="#94a3b8")),
            tooltip=[
                alt.Tooltip("Time:Q", format=".1f", title="Time (s)"),
                alt.Tooltip("Size_MB:Q", format=".2f", title="Size (MB)"),
                alt.Tooltip("Event_Type:N", title="Event"),
            ],
        )
        .properties(
            height=260,
            title=alt.TitleParams(text="Allocation Events Timeline", color="#ffffff", fontSize=14, fontWeight=700, anchor="start"),
        )
        .configure_view(strokeWidth=0)
        .interactive()
    )
    return chart


def render_memory_pressure_speedo(
    pressure_pct: float = 42.0,
    risk_level: str = "Low Risk",
    subtext: str = "Sufficient memory headroom available",
):
    """
    1. The use of function:
       Renders the 'Memory Pressure & OOM Risk' visual speedo card from Dasahboard_vision.jpeg.
    2. How does the specific function work:
       Generates a stylized radial arc meter with percentage label and color-coded risk badge.
    3. Concepts used in the function:
       Dynamic headroom pressure quantification and transparent OOM risk heuristic.
    4. Places where this function is imported and used:
       Used in Row 4 (Center column) of Executive Dashboard.
    5. Final output of the function:
       Streamlit rendered HTML component without code block escaping.
    """
    color = "#10b981" if "low" in risk_level.lower() else ("#f59e0b" if "mod" in risk_level.lower() else "#f43f5e")
    badge_bg = "rgba(16, 185, 129, 0.12)" if "low" in risk_level.lower() else ("rgba(245, 158, 11, 0.12)" if "mod" in risk_level.lower() else "rgba(244, 63, 94, 0.12)")

    # Circumference of semicircle (r=38) is pi * 38 = 119.4
    dashoffset = 119.4 - (pressure_pct / 100.0 * 119.4)

    html = f"""
    <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 16px 18px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; align-items: center; text-align: center; box-sizing: border-box;">
        <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; width: 100%; text-align: left;">Memory Pressure & OOM Risk</div>
        
        <!-- Semicircular Arc Gauge -->
        <div style="position: relative; width: 150px; height: 85px; margin: 10px 0 2px 0;">
            <svg viewBox="0 0 100 55" style="width: 100%; height: 100%;">
                <path d="M 12 50 A 38 38 0 0 1 88 50" fill="none" stroke="#1f293d" stroke-width="9" stroke-linecap="round"/>
                <path d="M 12 50 A 38 38 0 0 1 88 50" fill="none" stroke="{color}" stroke-width="9" stroke-dasharray="119.4" stroke-dashoffset="{dashoffset:.1f}" stroke-linecap="round"/>
            </svg>
            <div style="position: absolute; bottom: 0; left: 0; right: 0; font-size: 1.65rem; font-weight: 700; color: #ffffff;">
                {pressure_pct:.0f}%
            </div>
        </div>
        <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 500; margin-bottom: 10px;">Memory Pressure</div>

        <!-- Risk Pill -->
        <div style="width: 100%; padding: 10px 14px; background-color: {badge_bg}; border: 1px solid {color}; border-radius: 6px;">
            <div style="font-size: 0.9rem; font-weight: 700; color: {color};">{risk_level}</div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">{subtext}</div>
        </div>
    </div>
    """
    render_clean_html(html)


def render_vision_recommendations_list(
    recs: Optional[List[Dict[str, str]]] = None,
):
    """
    1. The use of function:
       Renders the 'Optimization Recommendations' card with priority pills from Dasahboard_vision.jpeg.
    2. How does the specific function work:
       Generates a list of suggestions with color-coded priority badges (High, Medium, Low).
    3. Concepts used in the function:
       Prioritized optimization heuristics.
    4. Places where this function is imported and used:
       Used in Row 4 (Right column) of Executive Dashboard.
    5. Final output of the function:
       Streamlit rendered HTML component without code block escaping.
    """
    if not recs:
        items = [
            {"text": "Consider batch size 64 (current: 32)", "priority": "High", "color": "#10b981"},
            {"text": "Use mixed precision (FP16/BF16)", "priority": "High", "color": "#10b981"},
            {"text": "Enable gradient checkpointing", "priority": "Medium", "color": "#f59e0b"},
            {"text": "Reduce reserved-but-unused cache (0.7 GB)", "priority": "Medium", "color": "#f59e0b"},
            {"text": "Reuse intermediate tensor buffers", "priority": "Low", "color": "#06b6d4"},
        ]
    else:
        items = []
        for r in recs:
            p = r.get("priority", "Medium")
            c = "#10b981" if p == "High" else ("#f59e0b" if p == "Medium" else "#06b6d4")
            items.append({"text": r.get("text", ""), "priority": p, "color": c})

    items_html = ""
    for it in items:
        items_html += f"""
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; padding: 4px 0;">
            <div style="color: #e2e8f0; display: flex; align-items: center; gap: 8px;">
                <span style="color: #fbbf24;">💡</span> <span>{it['text']}</span>
            </div>
            <span style="background-color: {it['color']}22; color: {it['color']}; border: 1px solid {it['color']}; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700;">
                {it['priority']}
            </span>
        </div>
        """

    html = f"""
    <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 16px 18px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
        <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
            <span>💡</span> <span>Optimization Recommendations</span>
        </div>
        <div style="display: flex; flex-direction: column; gap: 8px; flex-grow: 1; justify-content: space-around;">
            {items_html}
        </div>
    </div>
    """
    render_clean_html(html)
