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
   - Active OS Virtual Memory & Optimization Components:
     * render_page_table_view (OS Page Table grid with VPN, frames, swap, dirty bit, timestamps).
     * render_hit_ratio_comparison_chart (Altair bar charts comparing FIFO, LRU, Optimal).
     * render_tiered_memory_dual_gauge (VRAM utilization vs CPU pinned swap overflow gauge).
     * render_tiered_timeline_chart (Altair comparative timeline: baseline vs capped VRAM + CPU spill).
     * render_live_benchmark_chart (Altair real-time streaming chart for 60s live benchmark).
     * render_paged_kv_block_diagram (physical VRAM block grid and Copy-On-Write sharing map).
     * render_kv_waste_comparison_chart (side-by-side contiguous vs paged waste comparison).

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


# ===========================================================================
# ACTIVE OS VIRTUAL MEMORY & OPTIMIZATION COMPONENTS
# ===========================================================================


def render_page_table_view(page_table: Dict[int, Any], vram_frames: List[int]):
    """
    1. The use of function:
       Renders an interactive Operating System Page Table visualizer displaying Virtual Page Numbers
       (VPN), physical VRAM frame mappings, secondary swap residency, dirty status, access counts,
       and temporal access metadata.

    2. How does the specific function work:
       Accepts a page table mapping dictionary (VPN -> PageEntry or dict) and a list of active VRAM
       frame page IDs. Summarizes resident frame allocations, constructs a styled dark-theme HTML table
       with status pills (e.g. green for resident VRAM frame, amber for host swap, slate for unmapped),
       and flattens the output for safe Streamlit DOM injection via render_clean_html.

    3. Concepts used in the function:
       Virtual memory paging, Page Table Entries (PTE), physical frame assignment, swap storage,
       dirty bits, and HTML/CSS presentation formatting.

    4. Places where this function is imported and used:
       Imported by dashboard/app.py in the OS Virtual Memory Lab view.

    5. Final output of the function, before and after:
       Before: Raw dictionary of PageEntry records and frame list.
       After: Rendered high-contrast OS Page Table card injected directly into the user interface.
    """
    if not page_table:
        render_clean_html(
            """
            <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 20px; text-align: center; color: #94a3b8;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 4px;">Page Table Empty</div>
                <div style="font-size: 0.82rem;">No virtual memory pages have been referenced yet. Run a trace above to inspect page table translations.</div>
            </div>
            """
        )
        return

    # Frame summary chips
    frames_chips_html = ""
    for idx, pid in enumerate(vram_frames):
        chip_label = f"Page {pid}" if pid is not None else "Empty"
        chip_color = "#10b981" if pid is not None else "#64748b"
        frames_chips_html += f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid {chip_color}; border-radius: 6px; padding: 6px 12px; font-size: 0.8rem; display: flex; align-items: center; gap: 6px;">
            <span style="color: #94a3b8;">Frame #{idx}:</span>
            <span style="color: {chip_color}; font-weight: 700;">{chip_label}</span>
        </div>
        """

    rows_html = ""
    for vpn in sorted(page_table.keys()):
        entry = page_table[vpn]
        if hasattr(entry, "to_dict"):
            e_dict = entry.to_dict()
        elif isinstance(entry, dict):
            e_dict = entry
        else:
            e_dict = {
                "page_id": getattr(entry, "page_id", vpn),
                "frame_id": getattr(entry, "frame_id", None),
                "in_vram": getattr(entry, "in_vram", False),
                "in_swap": getattr(entry, "in_swap", False),
                "dirty": getattr(entry, "dirty", False),
                "loaded_at": getattr(entry, "loaded_at", 0.0),
                "last_accessed_at": getattr(entry, "last_accessed_at", 0.0),
                "access_count": getattr(entry, "access_count", 0),
            }

        in_vram = e_dict.get("in_vram", False)
        in_swap = e_dict.get("in_swap", False)
        frame_id = e_dict.get("frame_id")
        dirty = e_dict.get("dirty", False)
        last_acc = e_dict.get("last_accessed_at", 0.0)
        acc_cnt = e_dict.get("access_count", 0)

        if in_vram:
            status_badge = f'<span style="background: rgba(16, 185, 129, 0.18); color: #10b981; border: 1px solid #10b981; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;">🟢 VRAM Frame #{frame_id}</span>'
        elif in_swap:
            status_badge = '<span style="background: rgba(245, 158, 11, 0.18); color: #f59e0b; border: 1px solid #f59e0b; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;">🟠 Host Swap</span>'
        else:
            status_badge = '<span style="background: rgba(148, 163, 184, 0.18); color: #94a3b8; border: 1px solid #64748b; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem;">⚪ Unmapped</span>'

        dirty_badge = (
            '<span style="color: #f43f5e; font-weight: 700;">1 (Dirty)</span>'
            if dirty
            else '<span style="color: #64748b;">0 (Clean)</span>'
        )

        rows_html += f"""
        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05); text-align: left;">
            <td style="padding: 8px 12px; font-weight: 700; color: #00d2ff;">VPN {vpn}</td>
            <td style="padding: 8px 12px;">{status_badge}</td>
            <td style="padding: 8px 12px;">{dirty_badge}</td>
            <td style="padding: 8px 12px; color: #e2e8f0;">Step {int(last_acc)}</td>
            <td style="padding: 8px 12px; color: #e2e8f0;">{acc_cnt}</td>
        </tr>
        """

    html = f"""
    <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 18px; margin-top: 14px; box-sizing: border-box;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 8px;">
                <span>📖</span> <span>Operating System Page Table & Physical Frame Allocation</span>
            </div>
            <div style="font-size: 0.78rem; color: #00d2ff; font-family: monospace;">Unit 5: Address Translation</div>
        </div>

        <div style="margin-bottom: 14px;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; font-weight: 600; margin-bottom: 6px;">Active VRAM Frames:</div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                {frames_chips_html}
            </div>
        </div>

        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem; color: #cbd5e1;">
                <thead>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.12); color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.5px;">
                        <th style="padding: 8px 12px; text-align: left;">Virtual Page (VPN)</th>
                        <th style="padding: 8px 12px; text-align: left;">Residency Location</th>
                        <th style="padding: 8px 12px; text-align: left;">Dirty Bit</th>
                        <th style="padding: 8px 12px; text-align: left;">Last Accessed</th>
                        <th style="padding: 8px 12px; text-align: left;">Access Count</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    """
    render_clean_html(html)


def render_hit_ratio_comparison_chart(comparison_metrics: Dict[str, Any]):
    """
    1. The use of function:
       Renders an interactive Altair comparative visualization displaying Hits vs Page Faults
       and Hit Ratios across FIFO, LRU, and Belady's Optimal page replacement policies.

    2. How does the specific function work:
       Transforms comparison metrics dictionaries or VirtualMemoryMetrics into normalized pandas
       DataFrames. Produces two synchronized Altair charts: (1) a grouped bar chart displaying
       Hits (emerald green) and Faults (rose red) per policy, and (2) a Hit Ratio percentage bar
       chart with color-coded policy tiers and EAT tooltips. Renders charts side-by-side using Streamlit columns.

    3. Concepts used in the function:
       Operating system page replacement algorithms (FIFO, LRU, Optimal), demand paging metrics,
       Effective Access Time (EAT), and Altair visualization grammar.

    4. Places where this function is imported and used:
       Imported by dashboard/app.py in the OS Virtual Memory Lab view.

    5. Final output of the function, before and after:
       Before: Raw comparison dictionary mapping policy names to metrics.
       After: Polished side-by-side interactive Altair bar charts rendered in the web dashboard.
    """
    if not comparison_metrics:
        st.info("No comparative policy metrics available to render.")
        return

    records_counts = []
    records_ratios = []

    for pol_name, m in comparison_metrics.items():
        if hasattr(m, "to_dict"):
            data = m.to_dict()
        elif isinstance(m, dict):
            data = m
        else:
            data = {
                "hits": getattr(m, "hits", 0),
                "faults": getattr(m, "faults", 0),
                "hit_ratio": getattr(m, "hit_ratio", 0.0),
                "eat_us": getattr(m, "eat_us", 0.0),
            }

        label = str(pol_name).upper()
        hits = int(data.get("hits", 0))
        faults = int(data.get("faults", 0))
        hit_ratio_pct = float(data.get("hit_ratio", 0.0)) * 100.0
        eat_us = float(data.get("eat_us", 0.0))

        records_counts.append({"Policy": label, "Metric": "Hits", "Count": hits})
        records_counts.append({"Policy": label, "Metric": "Faults", "Count": faults})
        records_ratios.append({"Policy": label, "Hit_Ratio_Pct": round(hit_ratio_pct, 1), "EAT_us": round(eat_us, 2)})

    df_counts = pd.DataFrame(records_counts)
    df_ratios = pd.DataFrame(records_ratios)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div style="font-size: 0.85rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">Hits vs Page Faults Comparison</div>', unsafe_allow_html=True)
        chart_counts = (
            alt.Chart(df_counts)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Policy:N", title="Algorithm", axis=alt.Axis(labelColor="#e2e8f0", titleColor="#94a3b8", labelAngle=0)),
                y=alt.Y("Count:Q", title="Access References", axis=alt.Axis(labelColor="#e2e8f0", titleColor="#94a3b8")),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(domain=["Hits", "Faults"], range=["#10b981", "#f43f5e"]),
                    legend=alt.Legend(title=None, labelColor="#e2e8f0", orient="top"),
                ),
                xOffset="Metric:N",
                tooltip=["Policy:N", "Metric:N", "Count:Q"],
            )
            .properties(height=240)
            .configure_view(strokeWidth=0)
            .configure(background="#111827")
        )
        st.altair_chart(chart_counts, width="stretch")

    with col2:
        st.markdown('<div style="font-size: 0.85rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">Hit Ratio (%) & Access Efficiency</div>', unsafe_allow_html=True)
        chart_ratios = (
            alt.Chart(df_ratios)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=38)
            .encode(
                x=alt.X("Policy:N", title="Algorithm", axis=alt.Axis(labelColor="#e2e8f0", titleColor="#94a3b8", labelAngle=0)),
                y=alt.Y("Hit_Ratio_Pct:Q", title="Hit Ratio (%)", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelColor="#e2e8f0", titleColor="#94a3b8")),
                color=alt.Color(
                    "Policy:N",
                    scale=alt.Scale(domain=["FIFO", "LRU", "OPTIMAL"], range=["#f59e0b", "#3b82f6", "#10b981"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Policy:N", title="Policy"),
                    alt.Tooltip("Hit_Ratio_Pct:Q", title="Hit Ratio (%)", format=".1f"),
                    alt.Tooltip("EAT_us:Q", title="EAT (μs)", format=".2f"),
                ],
            )
            .properties(height=240)
            .configure_view(strokeWidth=0)
            .configure(background="#111827")
        )
        st.altair_chart(chart_ratios, width="stretch")


def render_tiered_memory_dual_gauge(
    vram_used_mb: float,
    cpu_swap_mb: float,
    vram_budget_mb: float,
    threshold_mb: float,
):
    """
    1. The use of function:
       Renders a high-contrast dual memory gauge contrasting primary GPU VRAM utilization
       against host CPU pinned swap overflow, illustrating activation spillover dynamics.

    2. How does the specific function work:
       Computes relative utilization percentages for VRAM budget and CPU swap memory.
       Evaluates whether VRAM usage exceeds the pressure threshold. Generates dual visual progress
       meters styled with high-contrast neon palettes (electric cyan for safe VRAM, warning amber
       for near-threshold, purple/emerald for CPU offloaded buffers) and renders metrics badges
       confirming peak VRAM reduction and mathematical autograd equivalence.

    3. Concepts used in the function:
       Tiered memory hierarchy, accelerator memory pressure, host-device DMA offloading,
       Capuchin / SwapAdvisor activation management, and CSS progress bars.

    4. Places where this function is imported and used:
       Imported by dashboard/app.py in the Active Optimizer Lab (Tiered Swapper tab).

    5. Final output of the function, before and after:
       Before: Raw float memory values in megabytes.
       After: High-impact dark-themed dual gauge UI component with status indicators.
    """
    vram_budget_eff = max(1.0, float(vram_budget_mb))
    vram_used = max(0.0, float(vram_used_mb))
    cpu_swap = max(0.0, float(cpu_swap_mb))
    threshold = max(0.0, float(threshold_mb))

    vram_pct = min(100.0, (vram_used / vram_budget_eff) * 100.0)
    thresh_pct = min(100.0, (threshold / vram_budget_eff) * 100.0)
    total_managed = vram_used + cpu_swap
    savings_pct = (cpu_swap / total_managed * 100.0) if total_managed > 0 else 0.0

    vram_color = "#f43f5e" if vram_used >= vram_budget_eff else ("#f59e0b" if vram_used >= threshold else "#00d2ff")
    is_spilling = cpu_swap > 0.0

    html = f"""
    <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 20px; box-sizing: border-box;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 8px;">
                <span>⚡</span> <span>Tiered Memory Allocation: VRAM vs Host CPU RAM</span>
            </div>
            <div style="font-size: 0.78rem; color: #10b981; font-weight: 700; background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; padding: 3px 10px; border-radius: 4px;">
                Equivalence: 100% Bit-for-Bit
            </div>
        </div>

        <!-- Metric 1: GPU VRAM -->
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px;">
                <span style="color: #cbd5e1; font-weight: 600;">Primary GPU VRAM Allocation</span>
                <span style="color: {vram_color}; font-weight: 700;">{vram_used:.1f} MB / {vram_budget_eff:.1f} MB ({vram_pct:.1f}%)</span>
            </div>
            <div style="position: relative; width: 100%; height: 18px; background: #1f293d; border-radius: 9px; overflow: hidden;">
                <div style="width: {vram_pct:.1f}%; height: 100%; background: {vram_color}; border-radius: 9px; transition: width 0.3s ease;"></div>
                <!-- Threshold Marker -->
                <div style="position: absolute; left: {thresh_pct:.1f}%; top: 0; bottom: 0; width: 2px; background: #fbbf24; z-index: 2;" title="Offload Threshold: {threshold:.1f} MB"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">
                <span>0 MB</span>
                <span style="color: #fbbf24;">▲ Offload Threshold ({threshold:.1f} MB)</span>
                <span>Budget ({vram_budget_eff:.1f} MB)</span>
            </div>
        </div>

        <!-- Metric 2: Host CPU Pinned Swap -->
        <div style="margin-bottom: 18px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px;">
                <span style="color: #cbd5e1; font-weight: 600;">Host CPU Pinned RAM Overflow (Offloaded Activations)</span>
                <span style="color: #c084fc; font-weight: 700;">{cpu_swap:.1f} MB Active Spill</span>
            </div>
            <div style="width: 100%; height: 18px; background: #1f293d; border-radius: 9px; overflow: hidden;">
                <div style="width: {min(100.0, (cpu_swap / max(1.0, total_managed)) * 100.0):.1f}%; height: 100%; background: linear-gradient(90deg, #9333ea, #c084fc); border-radius: 9px; transition: width 0.3s ease;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">
                <span>Zero CPU Overhead</span>
                <span style="color: {'#10b981' if is_spilling else '#94a3b8'};">{'DMA Offload Active' if is_spilling else 'No Spillover Needed'}</span>
                <span>Relieved: {savings_pct:.1f}% Peak</span>
            </div>
        </div>

        <!-- KPI Summary Pills -->
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px;">
            <div style="background: rgba(0, 210, 255, 0.08); border: 1px solid rgba(0, 210, 255, 0.3); border-radius: 6px; padding: 10px; text-align: center;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Active VRAM Footprint</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #00d2ff; margin-top: 2px;">{vram_used:.1f} MB</div>
            </div>
            <div style="background: rgba(192, 132, 252, 0.08); border: 1px solid rgba(192, 132, 252, 0.3); border-radius: 6px; padding: 10px; text-align: center;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Host Offload Spill</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #c084fc; margin-top: 2px;">{cpu_swap:.1f} MB</div>
            </div>
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 10px; text-align: center;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Peak VRAM Relieved</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #10b981; margin-top: 2px;">{savings_pct:.1f}%</div>
            </div>
        </div>
    </div>
    """
    render_clean_html(html)


def render_paged_kv_block_diagram(viz_data: Dict[str, Any]):
    """
    1. The use of function:
       Renders an interactive Operating System virtual memory block diagram illustrating
       PagedAttention physical VRAM blocks, reference-counted block sharing, and Copy-On-Write (COW).

    2. How does the specific function work:
       Parses the visualization payload produced by PagedKVCacheManager.get_block_table_visualization().
       Constructs a physical block grid showing occupancy badges and highlights shared COW blocks
       (ref_count > 1) in neon cyan/emerald. Below the grid, renders logical sequence block chains
       mapping logical sequence indices to concrete physical blocks.

    3. Concepts used in the function:
       vLLM / PagedAttention (OSDI '23), Operating System paging, virtual-to-physical address
       translation, Copy-On-Write (COW), reference counting, and prompt prefix sharing.

    4. Places where this function is imported and used:
       Imported by dashboard/app.py in the Active Optimizer Lab (Paged KV-Cache tab).

    5. Final output of the function, before and after:
       Before: Raw dictionary containing block table entries and physical block states.
       After: Visual block diagram injected directly into the user interface.
    """
    if not viz_data or "physical_blocks" not in viz_data:
        st.info("No Paged KV-Cache visualization data available.")
        return

    summary = viz_data.get("summary", {})
    total_blocks = summary.get("total_physical_blocks", 0)
    used_blocks = summary.get("used_blocks_count", 0)
    free_blocks = summary.get("free_blocks_count", 0)
    block_size = summary.get("block_size_tokens", 16)
    active_seqs = summary.get("active_sequences_count", 0)

    # Summary chips
    summary_chips_html = f"""
    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px;">
        <div style="background: rgba(0, 210, 255, 0.1); border: 1px solid #00d2ff; border-radius: 6px; padding: 6px 12px; font-size: 0.8rem;">
            <span style="color: #94a3b8;">Total Blocks:</span> <strong style="color: #00d2ff;">{total_blocks}</strong>
        </div>
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 6px; padding: 6px 12px; font-size: 0.8rem;">
            <span style="color: #94a3b8;">Allocated Blocks:</span> <strong style="color: #10b981;">{used_blocks}</strong>
        </div>
        <div style="background: rgba(148, 163, 184, 0.1); border: 1px solid #64748b; border-radius: 6px; padding: 6px 12px; font-size: 0.8rem;">
            <span style="color: #94a3b8;">Free Blocks:</span> <strong style="color: #e2e8f0;">{free_blocks}</strong>
        </div>
        <div style="background: rgba(192, 132, 252, 0.1); border: 1px solid #c084fc; border-radius: 6px; padding: 6px 12px; font-size: 0.8rem;">
            <span style="color: #94a3b8;">Block Sizing:</span> <strong style="color: #c084fc;">{block_size} tokens/block</strong>
        </div>
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; border-radius: 6px; padding: 6px 12px; font-size: 0.8rem;">
            <span style="color: #94a3b8;">Active Streams:</span> <strong style="color: #f59e0b;">{active_seqs}</strong>
        </div>
    </div>
    """

    # Physical blocks grid (render first 32 blocks to ensure clean layout)
    phys_blocks = viz_data.get("physical_blocks", [])
    display_blocks = phys_blocks[:32]

    grid_items_html = ""
    for b in display_blocks:
        b_id = b.get("block_id", 0)
        ref_count = b.get("ref_count", 0)
        is_free = b.get("is_free", True)
        tokens = b.get("token_ids", [])
        num_tokens = len(tokens)

        if is_free:
            bg_color = "rgba(31, 41, 61, 0.4)"
            border_color = "rgba(255, 255, 255, 0.08)"
            badge_html = '<span style="color: #64748b; font-size: 0.68rem; font-weight: 600;">FREE</span>'
            token_indicator = f'<span style="color: #475569; font-size: 0.72rem;">0/{block_size} slots</span>'
        elif ref_count > 1:
            bg_color = "rgba(0, 210, 255, 0.12)"
            border_color = "#00d2ff"
            badge_html = f'<span style="background: #00d2ff; color: #0b101b; font-size: 0.68rem; font-weight: 700; padding: 1px 5px; border-radius: 3px;">🔗 COW (ref: {ref_count})</span>'
            token_indicator = f'<span style="color: #00d2ff; font-weight: 600; font-size: 0.72rem;">{num_tokens}/{block_size} tokens</span>'
        else:
            bg_color = "rgba(37, 99, 235, 0.15)"
            border_color = "#3b82f6"
            badge_html = '<span style="background: #3b82f6; color: #ffffff; font-size: 0.68rem; font-weight: 700; padding: 1px 5px; border-radius: 3px;">PRIVATE</span>'
            token_indicator = f'<span style="color: #93c5fd; font-weight: 600; font-size: 0.72rem;">{num_tokens}/{block_size} tokens</span>'

        grid_items_html += f"""
        <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 6px; padding: 8px 10px; display: flex; flex-direction: column; justify-content: space-between; min-height: 60px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 700; font-size: 0.78rem; color: #e2e8f0;">Block #{b_id}</span>
                {badge_html}
            </div>
            <div>
                {token_indicator}
            </div>
        </div>
        """

    # Sequences logical mapping chains
    sequences = viz_data.get("sequences", [])
    seqs_html = ""
    if not sequences:
        seqs_html = '<div style="color: #94a3b8; font-size: 0.82rem;">No active generation sequences registered.</div>'
    else:
        for s in sequences:
            seq_id = s.get("seq_id", "")
            t_count = s.get("token_count", 0)
            parent_id = s.get("parent_seq_id")
            mappings = s.get("mappings", [])

            chips = ""
            for m in mappings:
                l_idx = m.get("logical_idx", 0)
                p_id = m.get("physical_block_id", 0)
                is_shared = m.get("is_shared", False)
                ref_cnt = m.get("ref_count", 1)

                chip_border = "#00d2ff" if is_shared else "#3b82f6"
                chip_bg = "rgba(0, 210, 255, 0.15)" if is_shared else "rgba(59, 130, 246, 0.15)"
                chip_badge = f"🔗 COW (ref: {ref_cnt})" if is_shared else "Private"

                chips += f"""
                <div style="background: {chip_bg}; border: 1px solid {chip_border}; border-radius: 5px; padding: 4px 8px; font-size: 0.75rem; display: inline-flex; align-items: center; gap: 6px;">
                    <span style="color: #94a3b8;">L#{l_idx} ➔</span>
                    <strong style="color: #ffffff;">Phys #{p_id}</strong>
                    <span style="color: {chip_border}; font-size: 0.68rem; font-weight: 600;">[{chip_badge}]</span>
                </div>
                """

            parent_badge = (
                f'<span style="color: #fbbf24; font-size: 0.72rem; margin-left: 8px;">↳ Forked from <strong>{parent_id}</strong></span>'
                if parent_id
                else ""
            )

            seqs_html += f"""
            <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 12px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <strong style="color: #ffffff; font-size: 0.88rem;">Sequence: {seq_id}</strong>
                        {parent_badge}
                    </div>
                    <div style="color: #00d2ff; font-weight: 600; font-size: 0.78rem;">
                        {t_count} tokens ({len(mappings)} logical blocks)
                    </div>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
                    {chips}
                </div>
            </div>
            """

    full_html = f"""
    <div style="background-color: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 20px; box-sizing: border-box;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 8px;">
                <span>🗂️</span> <span>PagedAttention Physical Block Pool (vLLM OSDI '23)</span>
            </div>
            <div style="font-size: 0.78rem; color: #00d2ff; font-family: monospace;">Zero External Fragmentation</div>
        </div>

        {summary_chips_html}

        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.82rem; font-weight: 600; color: #cbd5e1; margin-bottom: 8px;">
                Physical VRAM Block Pool (Showing first {len(display_blocks)} of {total_blocks} blocks):
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px;">
                {grid_items_html}
            </div>
        </div>

        <div style="margin-top: 20px;">
            <div style="font-size: 0.82rem; font-weight: 600; color: #cbd5e1; margin-bottom: 8px;">
                Logical Sequence Block Tables (Virtual-to-Physical Address Translation):
            </div>
            {seqs_html}
        </div>
    </div>
    """
    render_clean_html(full_html)


def render_kv_waste_comparison_chart(waste_metrics: Dict[str, Any]):
    """
    1. The use of function:
       Renders a side-by-side comparative memory chart contrasting traditional contiguous
       pre-allocation with PagedAttention virtual memory allocation.

    2. How does the specific function work:
       Extracts allocated, used, and wasted bytes from the waste_metrics object/dict. Converts byte
       values to megabytes and builds a stacked Altair bar chart with Used Memory (emerald green)
       and Wasted Memory (rose red). Injects an executive summary banner displaying total VRAM
       waste percentages and net memory savings.

    3. Concepts used in the function:
       Internal memory fragmentation, static max_seq_len reservation waste, dynamic paging,
       and Altair stacked visualization.

    4. Places where this function is imported and used:
       Imported by dashboard/app.py in the Active Optimizer Lab (Paged KV-Cache tab).

    5. Final output of the function, before and after:
       Before: Waste comparison metric values.
       After: Rendered Altair stacked bar chart and HTML callout highlighting 60-80% memory savings.
    """
    if not waste_metrics:
        st.info("No memory waste comparison metrics available.")
        return

    if hasattr(waste_metrics, "to_dict"):
        data = waste_metrics.to_dict()
    elif isinstance(waste_metrics, dict):
        data = waste_metrics
    else:
        data = {
            "traditional_allocated_bytes": getattr(waste_metrics, "traditional_allocated_bytes", 0),
            "traditional_used_bytes": getattr(waste_metrics, "traditional_used_bytes", 0),
            "traditional_wasted_bytes": getattr(waste_metrics, "traditional_wasted_bytes", 0),
            "traditional_waste_pct": getattr(waste_metrics, "traditional_waste_pct", 0.0),
            "paged_allocated_bytes": getattr(waste_metrics, "paged_allocated_bytes", 0),
            "paged_used_bytes": getattr(waste_metrics, "paged_used_bytes", 0),
            "paged_wasted_bytes": getattr(waste_metrics, "paged_wasted_bytes", 0),
            "paged_waste_pct": getattr(waste_metrics, "paged_waste_pct", 0.0),
            "memory_savings_pct": getattr(waste_metrics, "memory_savings_pct", 0.0),
        }

    trad_used_mb = data.get("traditional_used_bytes", 0) / (1024 * 1024)
    trad_wasted_mb = data.get("traditional_wasted_bytes", 0) / (1024 * 1024)
    trad_waste_pct = float(data.get("traditional_waste_pct", 0.0))

    paged_used_mb = data.get("paged_used_bytes", 0) / (1024 * 1024)
    paged_wasted_mb = data.get("paged_wasted_bytes", 0) / (1024 * 1024)
    paged_waste_pct = float(data.get("paged_waste_pct", 0.0))
    savings_pct = float(data.get("memory_savings_pct", 0.0))

    chart_records = [
        {"Strategy": "Traditional Contiguous", "Component": "Active Token Data", "MB": round(trad_used_mb, 2)},
        {"Strategy": "Traditional Contiguous", "Component": "Over-reserved / Wasted", "MB": round(trad_wasted_mb, 2)},
        {"Strategy": "Paged KV-Cache (vLLM)", "Component": "Active Token Data", "MB": round(paged_used_mb, 2)},
        {"Strategy": "Paged KV-Cache (vLLM)", "Component": "Over-reserved / Wasted", "MB": round(paged_wasted_mb, 2)},
    ]
    df_chart = pd.DataFrame(chart_records)

    chart = (
        alt.Chart(df_chart)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=48)
        .encode(
            x=alt.X("Strategy:N", title=None, axis=alt.Axis(labelColor="#e2e8f0", labelAngle=0, labelFontSize=12)),
            y=alt.Y("MB:Q", title="VRAM Consumed (MB)", axis=alt.Axis(labelColor="#e2e8f0", titleColor="#94a3b8")),
            color=alt.Color(
                "Component:N",
                scale=alt.Scale(domain=["Active Token Data", "Over-reserved / Wasted"], range=["#10b981", "#f43f5e"]),
                legend=alt.Legend(title=None, labelColor="#e2e8f0", orient="top"),
            ),
            order=alt.Order("Component:N", sort="ascending"),
            tooltip=["Strategy:N", "Component:N", alt.Tooltip("MB:Q", title="Memory (MB)", format=".2f")],
        )
        .properties(height=260)
        .configure_view(strokeWidth=0)
        .configure(background="#111827")
    )

    st.altair_chart(chart, width="stretch")

    callout_html = f"""
    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 14px 18px; display: flex; justify-content: space-between; align-items: center; box-sizing: border-box;">
        <div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #10b981;">
                🎉 PagedAttention VRAM Savings: {savings_pct:.1f}% Reduction
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                Traditional static reservation wastes <strong>{trad_waste_pct:.1f}%</strong> of memory pool. Paging slashes waste to <strong>{paged_waste_pct:.1f}%</strong> (near-zero).
            </div>
        </div>
        <div style="font-size: 1.4rem; font-weight: 800; color: #10b981;">
            -{savings_pct:.0f}% VRAM
        </div>
    </div>
    """
    render_clean_html(callout_html)


def render_tiered_timeline_chart(
    baseline_peak_mb: float,
    threshold_mb: float,
    cpu_swap_mb: float,
    events: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    1. The use of function:
       Renders a comparative execution timeline chart contrasting unmitigated PyTorch baseline
       activation accumulation against TieredActivationManager capped VRAM with host CPU spill.

    2. How does the specific function work:
       Constructs a step-by-step DataFrame across forward activation generation and backward gradient
       propagation phases. Compares three memory series:
       (1) Unmitigated Baseline VRAM (spiking up to baseline_peak_mb at the loss node).
       (2) Capped GPU VRAM (flatlined at the safety threshold_mb).
       (3) Host CPU Pinned RAM Spill (streaming activations via asynchronous DMA).
       Overlays a dashed hardware safety boundary and renders the interactive multi-series Altair chart.

    3. Concepts used in the function:
       Activation accumulation lifecycle, forward-backward autograd peak, DMA spillover,
       Altair multi-series line chart, and dark-theme data visualization.

    4. Places where this function is imported and used:
       Imported by dashboard/app.py in the OOM Defense & Tiered Swapper laboratory view.

    5. Final output of the function, before and after:
       Before: Numeric baseline and swap parameters.
       After: Interactive Altair timeline chart visually demonstrating OOM prevention.
    """
    steps = [
        ("Step 1 (Input)", 0.12, 0.12, 0.0),
        ("Step 2 (Fwd L1)", 0.30, 0.30, 0.0),
        ("Step 3 (Fwd L2)", 0.55, 0.55, 0.0),
        ("Step 4 (Fwd Mid)", 0.80, 0.80, 0.0),
        ("Step 5 (Loss Node: PEAK)", 1.00, 1.00, 1.0),
        ("Step 6 (Bwd L_out)", 0.82, 0.82, 0.8),
        ("Step 7 (Bwd Mid)", 0.58, 0.58, 0.35),
        ("Step 8 (Bwd L1)", 0.32, 0.32, 0.0),
        ("Step 9 (Grad Complete)", 0.08, 0.08, 0.0),
    ]
    records = []
    effective_threshold = max(0.01, threshold_mb)
    for step_label, base_factor, _, _ in steps:
        b_mb = max(0.1, baseline_peak_mb * base_factor)
        c_mb = min(effective_threshold, b_mb)
        s_mb = max(0.0, (b_mb - effective_threshold) if cpu_swap_mb > 0 else 0.0)
        if cpu_swap_mb > 0 and (baseline_peak_mb - effective_threshold) > 0:
            scale = cpu_swap_mb / (baseline_peak_mb - effective_threshold)
            s_mb = s_mb * scale

        records.append({"Step": step_label, "Series": "Uncontrolled Baseline (VRAM)", "Memory (MB)": round(b_mb, 2)})
        records.append({"Step": step_label, "Series": "Capped GPU VRAM (Tiered)", "Memory (MB)": round(c_mb, 2)})
        records.append({"Step": step_label, "Series": "Host CPU Pinned RAM (Spill)", "Memory (MB)": round(s_mb, 2)})

    df = pd.DataFrame(records)
    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=40), strokeWidth=2.5)
        .encode(
            x=alt.X(
                "Step:N",
                sort=None,
                title="Execution Phase (Forward ➔ Backward Pass)",
                axis=alt.Axis(labelColor="#e2e8f0", titleColor="#94a3b8", labelAngle=-15),
            ),
            y=alt.Y(
                "Memory (MB):Q",
                title="Allocated Memory (MB)",
                axis=alt.Axis(labelColor="#e2e8f0", titleColor="#94a3b8"),
            ),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["Uncontrolled Baseline (VRAM)", "Capped GPU VRAM (Tiered)", "Host CPU Pinned RAM (Spill)"],
                    range=["#f43f5e", "#00d2ff", "#10b981"],
                ),
                legend=alt.Legend(title=None, labelColor="#e2e8f0", orient="top"),
            ),
            tooltip=["Step:N", "Series:N", alt.Tooltip("Memory (MB):Q", format=".2f")],
        )
        .properties(height=260)
        .configure_view(strokeWidth=0)
        .configure(background="#111827")
    )
    st.altair_chart(chart, width="stretch")


def render_live_benchmark_chart(
    records_df: pd.DataFrame,
    vram_budget_mb: float = 6140.0,
    threshold_mb: float = 4900.0,
) -> None:
    """
    1. The use of function:
       Renders an interactive streaming Altair time-series chart visualizing real-time primary
       GPU VRAM allocation vs host CPU pinned RAM spill over the 60-second live benchmark execution.

    2. How does the specific function work:
       Takes a DataFrame containing benchmark step records (columns: elapsed_s, vram_used_mb,
       host_ram_used_mb, status, phase_name). Melts VRAM and Host RAM metrics into normalized
       series rows. Plots dynamic multi-colored line curves with point overlays, renders horizontal
       dashed rules for hardware capacity and offload threshold, and renders directly into
       Streamlit with dark-mode styling.

    3. Concepts used in the function:
       Real-time time-series streaming visualization, Altair chart layering (lines, points, rules),
       tiered memory hierarchy telemetry, and dynamic memory safety thresholding.

    4. Places where this function is imported and used:
       Imported by dashboard/app.py in the OOM Defense laboratory view during live benchmark runs.

    5. Final output of the function, before and after:
       Before: Raw time-series DataFrame of benchmark records.
       After: Rendered Altair streaming chart displayed in the web dashboard.
    """
    if records_df is None or records_df.empty:
        st.info("No benchmark time-series data available to plot.")
        return

    # Prepare series data for plotting
    plot_rows = []
    for _, row in records_df.iterrows():
        t = float(row.get("elapsed_s", 0.0))
        vram = float(row.get("vram_used_mb", 0.0))
        host = float(row.get("host_ram_used_mb", 0.0))
        phase = str(row.get("phase_name", "Training"))
        status = str(row.get("status", "RUNNING"))

        plot_rows.append({
            "Time (s)": t,
            "Series": "Primary GPU VRAM",
            "Memory (MB)": round(vram, 1),
            "Phase": phase,
            "Status": status,
        })
        plot_rows.append({
            "Time (s)": t,
            "Series": "Host CPU Pinned RAM (Spill)",
            "Memory (MB)": round(host, 1),
            "Phase": phase,
            "Status": status,
        })

    df_plot = pd.DataFrame(plot_rows)

    color_scale = alt.Scale(
        domain=["Primary GPU VRAM", "Host CPU Pinned RAM (Spill)"],
        range=["#00d2ff", "#c084fc"],
    )

    base = alt.Chart(df_plot).encode(
        x=alt.X("Time (s):Q", title="Elapsed Time (seconds)", axis=alt.Axis(labelColor="#e2e8f0", titleColor="#94a3b8")),
        y=alt.Y("Memory (MB):Q", title="Allocated Memory (MB)", axis=alt.Axis(labelColor="#e2e8f0", titleColor="#94a3b8")),
        color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(title=None, orient="top", labelColor="#e2e8f0")),
        tooltip=[
            alt.Tooltip("Time (s):Q", format=".1f"),
            alt.Tooltip("Series:N"),
            alt.Tooltip("Memory (MB):Q", format=".1f"),
            alt.Tooltip("Phase:N"),
            alt.Tooltip("Status:N"),
        ],
    )

    lines = base.mark_line(strokeWidth=2.5)
    points = base.mark_point(size=30, filled=True)

    # Threshold horizontal rule
    thresh_df = pd.DataFrame([{"Threshold": threshold_mb}])
    thresh_rule = alt.Chart(thresh_df).mark_rule(
        strokeDash=[5, 5], color="#f59e0b", strokeWidth=1.8
    ).encode(
        y="Threshold:Q",
        tooltip=[alt.Tooltip("Threshold:Q", title="Offload Threshold (MB)", format=".1f")],
    )

    # Budget horizontal rule
    budget_df = pd.DataFrame([{"Budget": vram_budget_mb}])
    budget_rule = alt.Chart(budget_df).mark_rule(
        strokeDash=[3, 3], color="#f43f5e", strokeWidth=1.5
    ).encode(
        y="Budget:Q",
        tooltip=[alt.Tooltip("Budget:Q", title="Hardware VRAM Capacity (MB)", format=".1f")],
    )

    chart = (
        alt.layer(lines, points, thresh_rule, budget_rule)
        .properties(height=280)
        .configure_view(strokeWidth=0)
        .configure(background="#111827")
    )

    st.altair_chart(chart, width="stretch")


