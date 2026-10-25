"""
File: dashboard/app.py

1. What this file does:
   The main interactive Streamlit web dashboard for the GPU Memory Profiling &
   Optimization Toolkit. Visualizes device-level telemetry, in-process allocator
   dynamics, Free-Space Fragmentation Index (FSFI), multi-process contention,
   cooperative admission control, rule-based optimization recommendations, and
   experimental benchmark results.

2. Overall design:
   A modular multi-tab dashboard organized according to the Revised Development Plan
   (Phases 1 through 7) and foundational Operating System resource management
   principles. Supports live profiles from real GPU runs (`data/profiles/`),
   synthetic simulation datasets (`data/mock/`), and user-uploaded telemetry files
   (CSV or JSON). Completely CPU-safe and defensive against missing or polymorphic data.

3. Elements in the file:
   - Sidebar: Data source selection, budget configuration, safety margin tuning.
   - Tab 1: Device Overview (NVML telemetry & timeline).
   - Tab 2: In-Process Allocator & Pressure (Allocated vs Reserved, Headroom, OOM Risk).
   - Tab 3: Allocator Fragmentation (FSFI calculation, segment block distribution).
   - Tab 4: Contention & Admission Control (Multi-PID charts, interactive admission simulator).
   - Tab 5: Optimization & Recommendations (Traceable rules, interactive batch optimizer).
   - Tab 6: Benchmark Matrix (Baseline vs optimized, batch sweeps, precision matrix).
   - Tab 7: OS Concepts & Architecture (Architecture diagrams & OS resource principles).

4. Final output of the file:
   A rich, interactive dashboard running at `http://localhost:8501`.
"""

import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Safe NVML import for real-time hardware telemetry
try:
    import pynvml
except ImportError:
    pynvml = None


# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.normalize import (
    bytes_to_gb,
    bytes_to_mb,
    events_to_dataframe,
    format_bytes,
    load_and_normalize_csv,
)
from pipeline.aggregate import (
    compute_pid_breakdown,
    compute_profile_summary,
    resample_timeseries,
)
from pipeline.validate import validate_csv_file
from profiler.memory_pressure import evaluate_oom_risk_heuristic
from profiler.fragmentation import parse_snapshot_json
from optimizer.batch_optimizer import BatchOptimizer
from optimizer.recommendations import RecommendationEngine
from controller.admission_control import AdmissionController, AdmissionDecision
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



# ---------------------------------------------------------------------------
# Hardware Detection Helpers
# ---------------------------------------------------------------------------

def detect_system_gpu() -> Optional[Dict[str, Any]]:
    """
    1. The use of function:
       Auto-detects the host system's physical NVIDIA GPU via NVML.
    2. How does the specific function work:
       Attempts to initialize pynvml, queries device count, and extracts
       the primary GPU handle, device name, total VRAM capacity, and driver version.
       Returns None if NVML is unavailable or no NVIDIA GPU is present.
    3. Concepts used in the function:
       Hardware introspection, graceful degradation for CPU-only systems.
    4. Places where this function is imported and used:
       Called at dashboard startup to auto-configure live telemetry and default budgets.
    5. Final output of the function, before and after:
       Before: Unknown host hardware environment.
       After: Dictionary with GPU metadata or None if CPU-only.
    """
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes):
                    driver = driver.decode("utf-8", errors="replace")
            except Exception:
                driver = "Unknown"
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            return {
                "available": True,
                "handle": h,
                "name": str(name),
                "driver": str(driver),
                "total_bytes": int(mem.total),
                "total_mb": float(mem.total) / (1024.0 * 1024.0),
                "total_gb": float(mem.total) / (1024.0 * 1024.0 * 1024.0),
                "device_count": int(count),
            }
    except Exception:
        return None
    return None


def get_live_gpu_processes(handle) -> List[Dict[str, Any]]:
    """
    1. The use of function:
       Queries all active processes utilizing the GPU across both graphics and compute modes.
    2. How does the specific function work:
       Calls nvmlDeviceGetGraphicsRunningProcesses and nvmlDeviceGetComputeRunningProcesses,
       resolves each PID to its executable name via nvmlSystemGetProcessName, classifies
       the workload type (Game/Graphics vs CUDA Compute), and safely formats VRAM usage.
    3. Concepts used in the function:
       OS process tracking, WDDM graphics vs compute boundary separation.
    4. Places where this function is imported and used:
       Used in Tab 1 (Device Overview) and Tab 4 (Contention) during live monitoring.
    5. Final output of the function, before and after:
       Before: Raw NVML process structs.
       After: List of normalized process dictionaries ready for DataFrame display.
    """
    if pynvml is None or handle is None:
        return []
    procs: List[Dict[str, Any]] = []
    seen_pids = set()

    # 1. Compute processes (PyTorch, CUDA, etc.)
    try:
        c_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for p in c_procs:
            pid = p.pid
            seen_pids.add(pid)
            p_name = "Unknown"
            try:
                raw_name = pynvml.nvmlSystemGetProcessName(pid)
                if isinstance(raw_name, bytes):
                    raw_name = raw_name.decode("utf-8", errors="replace")
                p_name = Path(str(raw_name)).name
            except Exception:
                p_name = f"Process {pid}"
            vram_bytes = getattr(p, "usedGpuMemory", None)
            vram_str = f"{vram_bytes / (1024.0 * 1024.0):.1f} MiB" if vram_bytes else "Active"
            procs.append({
                "PID": pid,
                "Process Name": p_name,
                "Subsystem": "⚡ CUDA / Compute",
                "VRAM Usage": vram_str,
            })
    except Exception:
        pass

    # 2. Graphics processes (DirectX/Vulkan games, browser, Windows DWM)
    try:
        g_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
        for p in g_procs:
            pid = p.pid
            p_name = "Unknown"
            try:
                raw_name = pynvml.nvmlSystemGetProcessName(pid)
                if isinstance(raw_name, bytes):
                    raw_name = raw_name.decode("utf-8", errors="replace")
                p_name = Path(str(raw_name)).name
            except Exception:
                p_name = f"Process {pid}"
            vram_bytes = getattr(p, "usedGpuMemory", None)
            vram_str = f"{vram_bytes / (1024.0 * 1024.0):.1f} MiB" if vram_bytes else "Shared/OS"
            if pid in seen_pids:
                for item in procs:
                    if item["PID"] == pid:
                        item["Subsystem"] = "🎮 Graphics + Compute"
                        if vram_bytes:
                            item["VRAM Usage"] = vram_str
            else:
                procs.append({
                    "PID": pid,
                    "Process Name": p_name,
                    "Subsystem": "🎮 Graphics / Game / UI",
                    "VRAM Usage": vram_str,
                })
    except Exception:
        pass

    return procs


# ---------------------------------------------------------------------------
# Streamlit Page Setup

# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GPU Memory Profiling & Optimization Toolkit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_css()


# ---------------------------------------------------------------------------
# Sidebar: Configuration & Data Source Selection
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ GPU Profiling Toolkit")
st.sidebar.caption("OS-Oriented GPU Resource Management")

# Hardware Detection
gpu_hardware = detect_system_gpu()

if gpu_hardware:
    mode_options = [
        "🔴 Live GPU (Real-Time NVML)",
        "Recorded Profiles (data/profiles/)",
        "Mock Simulation (data/mock/)",
        "Upload CSV / JSON",
    ]
    default_mode_index = 0
else:
    mode_options = [
        "Recorded Profiles (data/profiles/)",
        "Mock Simulation (data/mock/)",
        "Upload CSV / JSON",
    ]
    default_mode_index = 0

data_source_mode = st.sidebar.radio(
    "Data Source Mode",
    mode_options,
    index=default_mode_index,
)

# Hardware and Memory Budget Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("Hardware & Budget Configuration")

if gpu_hardware:
    st.sidebar.success(f"🟢 **Detected:** {gpu_hardware['name']}")
    st.sidebar.caption(f"Driver: `{gpu_hardware['driver']}` | Hardware VRAM: `{gpu_hardware['total_gb']:.2f} GiB`")
    hardware_choices = [
        f"Auto: {gpu_hardware['name']}",
        "NVIDIA RTX 4050 (6 GB)",
        "NVIDIA RTX 3050 (6 GB)",
        "Custom Budget",
    ]
else:
    st.sidebar.info("ℹ️ Running in CPU-safe mode (no NVIDIA GPU detected).")
    hardware_choices = ["NVIDIA RTX 4050 (6 GB)", "NVIDIA RTX 3050 (6 GB)", "Custom Budget"]

gpu_target = st.sidebar.selectbox("Target Hardware Profile", hardware_choices, index=0)

if gpu_target == "Custom Budget":
    budget_gb = st.sidebar.slider("Configured Memory Budget (GiB)", 1.0, 24.0, 6.0, 0.5)
    budget_bytes = int(budget_gb * 1024 * 1024 * 1024)
elif gpu_hardware and gpu_target.startswith("Auto:"):
    budget_bytes = int(gpu_hardware["total_bytes"])
    budget_gb = float(gpu_hardware["total_gb"])
else:
    budget_gb = 6.0
    budget_bytes = int(budget_gb * 1024 * 1024 * 1024)

safety_margin_pct = st.sidebar.slider("Admission Safety Margin (%)", 5, 30, 15, 5)
safety_margin_bytes = int(budget_bytes * (safety_margin_pct / 100.0))

st.sidebar.markdown(
    f"""
    <div style="font-size: 0.8rem; color: #888888; margin-top: 4px;">
        • <strong>Budget:</strong> {budget_gb:.2f} GiB ({format_bytes(budget_bytes)})<br>
        • <strong>Safety Margin:</strong> {safety_margin_pct}% ({format_bytes(safety_margin_bytes)})<br>
        • <strong>Safe Headroom Cap:</strong> {format_bytes(budget_bytes - safety_margin_bytes)}
    </div>
    """,
    unsafe_allow_html=True,
)

# Live Polling Controls
auto_refresh = True
polling_rate_sec = 1
rolling_window_sec = 60

if data_source_mode.startswith("🔴 Live GPU"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔴 Live Polling Controls")
    auto_refresh = st.sidebar.toggle("Auto-Refresh Telemetry", value=True)
    polling_rate_sec = st.sidebar.selectbox(
        "Polling Frequency",
        [1, 2, 5],
        index=0,
        format_func=lambda s: f"Every {s} second{'s' if s > 1 else ''}",
    )
    rolling_window_sec = st.sidebar.slider("Rolling History Window", 15, 120, 60, 15, format="%d seconds")
    if st.sidebar.button("🧹 Clear Rolling History"):
        st.session_state["live_history"] = []
        st.session_state["live_start_time"] = time.time()
        st.rerun()

# ---------------------------------------------------------------------------
# Data Resolution based on selected mode
# ---------------------------------------------------------------------------
PROFILES_DIR = PROJECT_ROOT / "data" / "profiles"
MOCK_DIR = PROJECT_ROOT / "data" / "mock"

uploaded_df = pd.DataFrame()
uploaded_json_data: Optional[Dict[str, Any]] = None
uploaded_file = None

if data_source_mode.startswith("🔴 Live GPU"):
    baseline_csv = None
    torch_csv = PROFILES_DIR / "torch_events.csv" if (PROFILES_DIR / "torch_events.csv").is_file() else None
    contention_csv = None
    recs_json = PROFILES_DIR / "recs.json" if (PROFILES_DIR / "recs.json").is_file() else None
    snapshot_dir = PROFILES_DIR / "snapshots"
    benchmarks_csv = PROFILES_DIR / "benchmark_results.csv" if (PROFILES_DIR / "benchmark_results.csv").is_file() else None
elif data_source_mode == "Recorded Profiles (data/profiles/)":
    baseline_csv = PROFILES_DIR / "baseline.csv"
    torch_csv = PROFILES_DIR / "torch_events.csv"
    contention_csv = PROFILES_DIR / "contention.csv"
    recs_json = PROFILES_DIR / "recs.json"
    snapshot_dir = PROFILES_DIR / "snapshots"
    benchmarks_csv = PROFILES_DIR / "benchmark_results.csv"
elif data_source_mode == "Mock Simulation (data/mock/)":
    baseline_csv = MOCK_DIR / "mock_baseline.csv"
    torch_csv = MOCK_DIR / "mock_torch_events.csv"
    contention_csv = MOCK_DIR / "mock_contention.csv"
    recs_json = MOCK_DIR / "mock_recs.json"
    snapshot_dir = MOCK_DIR / "snapshots"
    benchmarks_csv = MOCK_DIR / "mock_benchmark_results.csv"
else:  # Upload CSV / JSON
    uploaded_file = st.sidebar.file_uploader(
        "Upload Telemetry CSV or JSON",
        type=["csv", "json"],
        help="Upload a recorded MemoryEvent CSV or snapshot/recommendation JSON."
    )
    baseline_csv = None
    torch_csv = None
    contention_csv = None
    recs_json = None
    snapshot_dir = PROFILES_DIR / "snapshots"
    benchmarks_csv = None

    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        if filename.endswith(".json"):
            try:
                raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
                raw_text = raw_bytes.decode("utf-8-sig", errors="replace") if isinstance(raw_bytes, bytes) else str(raw_bytes)
                uploaded_json_data = json.loads(raw_text)
                st.sidebar.success(f"Loaded JSON file: {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Failed to parse uploaded JSON: {e}")
        else:
            uploaded_df = load_and_normalize_csv(uploaded_file)
            if not uploaded_df.empty:
                st.sidebar.success(f"Loaded {len(uploaded_df)} rows from {uploaded_file.name}")
            else:
                st.sidebar.warning(f"File {uploaded_file.name} appears to be empty or unparseable.")

# Safe dataset routing
df_nvml = pd.DataFrame()
if baseline_csv and baseline_csv.is_file():
    df_nvml = load_and_normalize_csv(baseline_csv)
elif not uploaded_df.empty:
    if "source" in uploaded_df.columns and (uploaded_df["source"] == "nvml").any():
        df_nvml = uploaded_df[uploaded_df["source"] == "nvml"].copy()
    elif "device_used_mb" in uploaded_df.columns and uploaded_df["device_used_mb"].notna().any():
        df_nvml = uploaded_df.copy()

df_torch = pd.DataFrame()
if torch_csv and torch_csv.is_file():
    df_torch = load_and_normalize_csv(torch_csv)
elif not uploaded_df.empty:
    if "source" in uploaded_df.columns and (uploaded_df["source"] == "torch").any():
        df_torch = uploaded_df[uploaded_df["source"] == "torch"].copy()
    elif "allocated_mb" in uploaded_df.columns and uploaded_df["allocated_mb"].notna().any():
        df_torch = uploaded_df.copy()

df_contention = pd.DataFrame()
if contention_csv and contention_csv.is_file():
    df_contention = load_and_normalize_csv(contention_csv)
elif not uploaded_df.empty:
    if "pid" in uploaded_df.columns and (uploaded_df["pid"] > 0).any():
        df_contention = uploaded_df.copy()

bench_df = pd.DataFrame()
if benchmarks_csv and benchmarks_csv.is_file():
    try:
        bench_df = pd.read_csv(benchmarks_csv)
    except Exception:
        bench_df = pd.DataFrame()
elif not uploaded_df.empty:
    if any(col in uploaded_df.columns for col in ["experiment_id", "run_id"]):
        bench_df = uploaded_df.copy()


# ---------------------------------------------------------------------------
# Main Title Banner
# ---------------------------------------------------------------------------
st.title("⚡ GPU Memory Profiling & Optimization Toolkit")
st.markdown(
    """
    **An OS-oriented framework for runtime GPU memory analysis, allocator diagnosis,
    resource contention, and measurement-driven optimization recommendations.**
    """
)

# ---------------------------------------------------------------------------
# Main Tabs Organization
# ---------------------------------------------------------------------------
tab_device, tab_alloc, tab_frag, tab_content, tab_opt, tab_bench, tab_os = st.tabs([
    "🖥️ 1. Device Overview (NVML)",
    "🧠 2. Allocator & Pressure",
    "🧩 3. Allocator Fragmentation",
    "⚖️ 4. Contention & Admission",
    "🚀 5. Optimization & Rules",
    "📊 6. Benchmark Matrix",
    "📚 7. OS Concepts & Architecture",
])


# ===========================================================================
# TAB 1: Device Overview (NVML Monitoring)
# ===========================================================================
with tab_device:
    st.subheader("🖥️ Device-Level GPU Memory Monitoring (NVML)")
    st.caption("Phase 1: Continuous out-of-process device memory headroom and utilization tracking.")

    if data_source_mode.startswith("🔴 Live GPU") and gpu_hardware:
        refresh_interval_str = f"{polling_rate_sec}s" if auto_refresh else None

        @st.fragment(run_every=refresh_interval_str)
        def _render_live_view():
            h = gpu_hardware["handle"]
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
            except Exception as e:
                st.error(f"Error querying live NVML: {e}")
                return

            temp = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, 0)
            except Exception:
                pass

            now = time.time()
            if "live_history" not in st.session_state:
                st.session_state["live_history"] = []
            if "live_start_time" not in st.session_state:
                st.session_state["live_start_time"] = now

            elapsed = round(now - st.session_state["live_start_time"], 1)
            used_mb = round(mem.used / (1024.0 * 1024.0), 1)
            free_mb = round(mem.free / (1024.0 * 1024.0), 1)
            total_mb = round(mem.total / (1024.0 * 1024.0), 1)
            util_pct = util.gpu
            used_pct = round((mem.used / float(mem.total)) * 100.0, 1) if mem.total > 0 else 0.0

            st.session_state["live_history"].append({
                "timestamp": now,
                "elapsed_s": elapsed,
                "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
                "device_used_mb": used_mb,
                "device_free_mb": free_mb,
                "device_utilization_pct": util_pct,
                "temperature_c": temp,
            })
            max_samples = max(10, int(rolling_window_sec / max(1, polling_rate_sec)))
            st.session_state["live_history"] = st.session_state["live_history"][-max_samples:]

            # 5 Live KPI Cards
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                render_kpi_card("Physical GPU", gpu_hardware["name"], f"Capacity: {total_mb/1024:.2f} GiB")
            with col2:
                render_kpi_card("Current Used VRAM", f"{used_mb:.1f} MiB", f"{used_pct}% VRAM load", color="#e74c3c")
            with col3:
                render_kpi_card("Current Free Headroom", f"{free_mb:.1f} MiB", "Available allocatable", color="#2ecc71")
            with col4:
                render_kpi_card("GPU SM Utilization", f"{util_pct}%", "Compute core activity", color="#00d2ff")
            with col5:
                t_str = f"{temp} °C" if temp is not None else "N/A"
                t_col = "#e74c3c" if (temp and temp > 80) else "#2ecc71"
                render_kpi_card("GPU Temperature", t_str, "Thermal reading", color=t_col)

            # Live Timeseries Chart
            st.markdown(f"#### Live Device VRAM Timeseries (Rolling Last {rolling_window_sec}s)")
            df_live = pd.DataFrame(st.session_state["live_history"])
            vram_chart = create_vram_timeline_chart(df_live)
            st.altair_chart(vram_chart, width="stretch")

            # Active Running Processes (Games, UI, CUDA)
            st.markdown("#### Active GPU Processes (DirectX / Vulkan Games & CUDA Compute)")
            procs = get_live_gpu_processes(h)
            if procs:
                st.dataframe(pd.DataFrame(procs), width="stretch")
            else:
                st.caption("No dedicated graphics or compute processes actively reported by NVML.")

            st.markdown(
                """
                > **ℹ️ OS Resource Principle:** NVML communicates directly with the NVIDIA Kernel Driver.
                > It monitors real physical VRAM allocations across **all system processes** (including 3D games,
                > web browser hardware acceleration, Windows DWM, and PyTorch training).
                """
            )

        _render_live_view()
    elif not df_nvml.empty and "device_used_mb" in df_nvml.columns and df_nvml["device_used_mb"].notna().any():
        summary_nvml = compute_profile_summary(df_nvml, budget_bytes)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi_card("Total Device VRAM", f"{budget_gb:.1f} GiB", "Hardware capacity")
        with col2:
            used_mb = df_nvml["device_used_mb"].dropna().iloc[-1] if not df_nvml.empty else 0
            render_kpi_card("Current Used VRAM", f"{used_mb:.1f} MiB", "OS & Process load", color="#e74c3c")
        with col3:
            free_mb = df_nvml["device_free_mb"].dropna().iloc[-1] if "device_free_mb" in df_nvml.columns and df_nvml["device_free_mb"].notna().any() else 0
            render_kpi_card("Current Free Headroom", f"{free_mb:.1f} MiB", "Available allocatable space", color="#2ecc71")
        with col4:
            util = df_nvml["device_utilization_pct"].dropna().iloc[-1] if "device_utilization_pct" in df_nvml.columns and df_nvml["device_utilization_pct"].notna().any() else 0
            render_kpi_card("VRAM Utilization", f"{util:.1f}%", "Memory saturation")

        # Timeline Chart
        st.markdown("#### Device VRAM Timeseries")
        chart = create_vram_timeline_chart(resample_timeseries(df_nvml))
        st.altair_chart(chart, width="stretch")

        # Telemetry metadata table
        with st.expander("🔍 View Raw NVML Telemetry Records"):
            cols = [c for c in ["timestamp", "elapsed_s", "device_used_mb", "device_free_mb", "device_utilization_pct"] if c in df_nvml.columns]
            st.dataframe(df_nvml[cols].tail(50))
    else:
        if data_source_mode == "Upload CSV / JSON":
            if uploaded_file is None:
                st.info("💡 Upload a telemetry CSV file in the sidebar to inspect device VRAM.")
            else:
                st.info(
                    f"💡 The uploaded file '{uploaded_file.name}' does not contain device-level NVML columns (total_memory / free_memory). "
                    "If this is an in-process PyTorch workload trace, check **Tab 2 (Allocator & Pressure)**."
                )
        else:
            st.info(
                f"No baseline NVML records found at `{baseline_csv}`. "
                "Switch to 'Mock Simulation' in the sidebar or run `python collector/nvml_monitor.py` on your GPU machine."
            )


# ===========================================================================
# TAB 2: In-Process Allocator & Memory Pressure
# ===========================================================================
with tab_alloc:
    st.subheader("🧠 In-Process Allocator Introspection & Memory Pressure")
    st.caption("Phase 2: PyTorch caching allocator dynamics, allocation deltas, headroom, and heuristic OOM risk.")

    if not df_torch.empty and "allocated_mb" in df_torch.columns and df_torch["allocated_mb"].notna().any():
        summary_torch = compute_profile_summary(df_torch, budget_bytes)

        # Recent state
        curr_alloc = int(df_torch["allocated_bytes"].dropna().iloc[-1]) if "allocated_bytes" in df_torch.columns and df_torch["allocated_bytes"].notna().any() else 0
        curr_resv = int(df_torch["reserved_bytes"].dropna().iloc[-1]) if "reserved_bytes" in df_torch.columns and df_torch["reserved_bytes"].notna().any() else 0
        curr_cached_unused = max(0, curr_resv - curr_alloc)
        headroom = max(0, budget_bytes - curr_alloc)
        headroom_ratio = headroom / float(budget_bytes) if budget_bytes > 0 else 0

        # Memory Growth Rate calculation (delta over last few seconds)
        growth_rate_bps = 0.0
        if len(df_torch) >= 2 and "allocation_delta" in df_torch.columns and df_torch["allocation_delta"].notna().any():
            recent_deltas = df_torch["allocation_delta"].dropna().tail(5)
            growth_rate_bps = float(recent_deltas.mean()) if not recent_deltas.empty else 0.0
        elif len(df_torch) >= 2 and "allocated_bytes" in df_torch.columns and "elapsed_s" in df_torch.columns:
            alloc_diff = float(df_torch["allocated_bytes"].dropna().iloc[-1] - df_torch["allocated_bytes"].dropna().iloc[0])
            time_diff = float(df_torch["elapsed_s"].dropna().iloc[-1] - df_torch["elapsed_s"].dropna().iloc[0])
            if time_diff > 0:
                growth_rate_bps = alloc_diff / time_diff

        risk_level = evaluate_oom_risk_heuristic(budget_bytes, curr_alloc, growth_rate_bps)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi_card("Peak Allocated (Live)", format_bytes(summary_torch.peak_allocated_bytes), "High-water mark")
        with col2:
            render_kpi_card("Peak Reserved (Pool)", format_bytes(summary_torch.peak_reserved_bytes), "Caching allocator footprint", color="#9b59b6")
        with col3:
            render_kpi_card("Max Cached-Unused", format_bytes(summary_torch.max_cached_unused_bytes), "Reserved minus Allocated", color="#e67e22")
        with col4:
            render_kpi_card("Remaining Headroom", format_bytes(headroom), f"{headroom_ratio * 100:.1f}% of budget")

        # Risk Badge
        render_risk_badge(risk_level, headroom_ratio * 100, bytes_to_mb(growth_rate_bps) or 0.0)

        # Allocated vs Reserved Chart
        st.markdown("#### Allocated vs Reserved Memory Timeline")
        torch_chart = create_torch_alloc_chart(resample_timeseries(df_torch))
        st.altair_chart(torch_chart, width="stretch")

        with st.expander("🔍 View In-Process Allocation Samples"):
            cols = [c for c in ["timestamp", "elapsed_s", "allocated_mb", "reserved_mb", "cached_unused_mb", "peak_memory_mb"] if c in df_torch.columns]
            st.dataframe(df_torch[cols].tail(50))
    else:
        if data_source_mode == "Upload CSV / JSON":
            if uploaded_file is None:
                st.info("💡 Upload a telemetry CSV file in the sidebar to inspect in-process allocator metrics.")
            else:
                st.info(
                    f"💡 The uploaded file '{uploaded_file.name}' does not contain in-process PyTorch allocation columns (allocated_bytes / reserved_bytes). "
                    "If this is a device monitor trace, check **Tab 1 (Device Overview)**."
                )
        else:
            st.info(
                f"No PyTorch allocation events found at `{torch_csv}`. "
                "Switch to 'Mock Simulation' mode or execute `python collector/torch_hooks.py`."
            )


# ===========================================================================
# TAB 3: Allocator-Level Fragmentation (FSFI)
# ===========================================================================
with tab_frag:
    st.subheader("🧩 Allocator-Level Fragmentation Analysis (FSFI)")
    st.caption("Phase 3: Quantifying the distribution of free allocator blocks using the Free-Space Fragmentation Index.")

    st.markdown(
        r"""
        $$\text{FSFI} = 1 - \frac{\text{Largest Free Block}}{\text{Total Free Space in Allocator Pool}}$$
        * **FSFI $\approx 0.00$:** Low fragmentation; free space is concentrated in one large contiguous block.
        * **FSFI $\to 1.00$:** High fragmentation; free memory is splintered into small, non-contiguous blocks.
        """
    )

    # Check for uploaded JSON snapshot first
    snap_data: Optional[Dict[str, Any]] = None
    if uploaded_json_data and (isinstance(uploaded_json_data, list) or "segments" in uploaded_json_data):
        snap_data = uploaded_json_data
        st.success(f"Visualizing uploaded snapshot JSON: {uploaded_file.name}")
    else:
        # Check for available snapshot files on disk
        snapshots = sorted(list(snapshot_dir.glob("*.json"))) if snapshot_dir and snapshot_dir.is_dir() else []
        if not snapshots:
            fallback_dir = MOCK_DIR / "snapshots" if (MOCK_DIR / "snapshots").is_dir() else (PROFILES_DIR / "snapshots")
            if fallback_dir.is_dir():
                snapshots = sorted(list(fallback_dir.glob("*.json")))

        if snapshots:
            selected_snap = st.selectbox(
                "Select PyTorch Memory Snapshot JSON",
                snapshots,
                format_func=lambda p: p.name,
            )
            try:
                with open(selected_snap, "r", encoding="utf-8") as f:
                    snap_data = json.load(f)
            except Exception as e:
                st.error(f"Error reading snapshot: {e}")

    if snap_data is not None:
        # Robust snapshot block parsing
        segments = snap_data.get("segments", []) if isinstance(snap_data, dict) else (snap_data if isinstance(snap_data, list) else [])
        all_blocks = [b for seg in segments if isinstance(seg, dict) for b in seg.get("blocks", []) if isinstance(b, dict)]

        inactive_blocks = [b.get("size", 0) for b in all_blocks if b.get("state") == "inactive"]
        total_free = sum(inactive_blocks)
        largest_free = max(inactive_blocks) if inactive_blocks else 0

        fsfi = 0.0
        if total_free > 0:
            fsfi = 1.0 - (largest_free / float(total_free))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fsfi_color = "#e74c3c" if fsfi > 0.6 else ("#f39c12" if fsfi > 0.3 else "#2ecc71")
            render_kpi_card("FSFI Index", f"{fsfi:.4f}", "Free-Space Fragmentation Index", color=fsfi_color)
        with col2:
            render_kpi_card("Largest Free Block", format_bytes(largest_free), "Max allocatable chunk", color="#00d2ff")
        with col3:
            render_kpi_card("Total Free Allocator Space", format_bytes(total_free), "Sum of inactive blocks", color="#9b59b6")
        with col4:
            render_kpi_card("Free Block Count", str(len(inactive_blocks)), "Splintered inactive blocks")

        # Block Size Distribution
        st.markdown("#### Allocator Segment Block Distribution")
        block_chart = create_block_histogram(all_blocks)
        st.altair_chart(block_chart, width="stretch")

        st.markdown(
            """
            > **⚠️ OS Concept Distinction:** This measures **allocator-level pool fragmentation** within PyTorch's
            > user-space caching allocator. It does not measure physical GPU page fragmentation managed by the NVIDIA driver.
            """
        )
    else:
        st.info("No JSON snapshot files found. Switch to 'Mock Simulation' in the sidebar or upload a snapshot JSON.")


# ===========================================================================
# TAB 4: Multi-Process Contention & Admission Control
# ===========================================================================
with tab_content:
    st.subheader("⚖️ Resource Contention & Cooperative Admission Control")
    st.caption("Phase 4: Multi-process competition for finite VRAM and OS-inspired cooperative workload admission.")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Multi-Process Contention Timeline")
        if not df_contention.empty:
            cont_chart = create_contention_stacked_chart(resample_timeseries(df_contention))
            st.altair_chart(cont_chart, width="stretch")

            # Per-PID statistics table
            pid_df = compute_pid_breakdown(df_contention)
            st.markdown("##### Per-Process Summary")
            st.dataframe(pid_df)
        else:
            st.info("No multi-process contention data found. Switch to Mock Simulation in sidebar to view concurrent workloads.")

    with col_right:
        st.markdown("#### Cooperative Admission Controller Simulator")
        st.caption("Test whether an incoming workload is ADMITTED, DEFERRED, or REJECTED.")

        with st.form("admission_test_form"):
            wl_name_input = st.text_input("Workload Name", value="Vision_Transformer_FineTune")
            wl_req_mb = st.number_input("Estimated VRAM Demand (MiB)", min_value=100, max_value=24000, value=2048, step=256)
            submit_adm = st.form_submit_button("Submit Workload Request")

        if submit_adm:
            wl_req_bytes = int(wl_req_mb * 1024 * 1024)
            controller = AdmissionController(
                budget_bytes=budget_bytes,
                safety_margin_ratio=safety_margin_pct / 100.0,
            )

            current_device_used = 1200 * 1024 * 1024  # baseline 1.2 GB
            if data_source_mode.startswith("🔴 Live GPU") and gpu_hardware and pynvml:
                try:
                    live_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_hardware["handle"])
                    current_device_used = int(live_mem.used)
                except Exception:
                    pass
            elif not df_contention.empty and "device_used_bytes" in df_contention.columns:
                last_used = df_contention["device_used_bytes"].dropna().iloc[-1]
                if last_used > 0:
                    current_device_used = int(last_used)

            decision, event_dict = controller.request(
                workload_id=wl_name_input or "sim-workload-01",
                estimated_bytes=wl_req_bytes,
                observed_device_used_bytes=current_device_used,
            )

            meta = {}
            if "metadata_json" in event_dict and event_dict["metadata_json"]:
                try:
                    meta = json.loads(event_dict["metadata_json"])
                except Exception:
                    pass
            decision_notes = meta.get("reject_reason") or f"Workload '{wl_name_input}' evaluated with decision: {decision}."

            st.markdown(
                f"""
                <div style="padding: 12px; border-radius: 8px; background-color: rgba(255,255,255,0.05); margin-top: 8px;">
                    <h4>Outcome: {render_admission_decision_badge(decision)}</h4>
                    <p style="font-size: 0.85rem; color: #cccccc;">{decision_notes}</p>
                    <div style="font-size: 0.8rem; color: #888888;">
                        • <strong>Requested:</strong> {format_bytes(wl_req_bytes)}<br>
                        • <strong>Current Used:</strong> {format_bytes(current_device_used)}<br>
                        • <strong>Configured Budget:</strong> {format_bytes(budget_bytes)}<br>
                        • <strong>Safety Margin:</strong> {format_bytes(safety_margin_bytes)} ({safety_margin_pct}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Admission Decision History Table
    st.markdown("#### Auditable Admission Log History")
    adm_log_path = MOCK_DIR / "mock_admission_log.csv" if data_source_mode == "Mock Simulation (data/mock/)" else PROFILES_DIR / "admission_log.csv"
    if not adm_log_path.is_file() and (MOCK_DIR / "mock_admission_log.csv").is_file():
        adm_log_path = MOCK_DIR / "mock_admission_log.csv"

    if adm_log_path.is_file():
        adm_df = pd.read_csv(adm_log_path)
        adm_df["Decision"] = adm_df["decision"].apply(render_admission_decision_badge)
        adm_df["Requested"] = adm_df["requested_bytes"].apply(format_bytes)
        adm_df["Observed Peak"] = adm_df["observed_peak_bytes"].apply(format_bytes)
        cols_to_show = [c for c in ["workload_name", "Decision", "Requested", "Observed Peak", "notes"] if c in adm_df.columns]
        st.markdown(adm_df[cols_to_show].to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.caption("No historical admission decisions logged yet.")


# ===========================================================================
# TAB 5: Optimization & Recommendations
# ===========================================================================
with tab_opt:
    st.subheader("🚀 Traceable Optimization Recommendations & Batch Sizing")
    st.caption("Phase 5: Metric-triggered heuristics and linear batch scaling optimizer under a memory budget.")

    col_recs, col_batch = st.columns([3, 2])

    with col_recs:
        st.markdown("#### Active Diagnostic Recommendations")
        recs_data: Optional[Dict[str, Any]] = None

        # Check uploaded JSON
        if uploaded_json_data and "recommendations" in uploaded_json_data:
            recs_data = uploaded_json_data
        elif recs_json and recs_json.is_file():
            try:
                with open(recs_json, "r", encoding="utf-8") as f:
                    recs_data = json.load(f)
            except Exception:
                recs_data = None
        elif not df_torch.empty:
            # Dynamically generate recommendations from live/uploaded torch events!
            try:
                engine = RecommendationEngine(budget_bytes=budget_bytes)
                recs_data = engine.generate_recommendations(df_torch.to_dict("records"))
            except Exception:
                recs_data = None

        if recs_data:
            summary = recs_data.get("summary", {})
            recs_list = recs_data.get("recommendations", [])

            st.markdown(
                f"""
                <div style="font-size: 0.85rem; color: #888888; margin-bottom: 12px;">
                    Analyzed Workload: <strong>{summary.get('workload_name', 'Workload')}</strong> | 
                    Precision: <strong>{summary.get('precision', 'FP32')}</strong> | 
                    Peak Memory: <strong>{format_bytes(summary.get('peak_memory_bytes'))}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if recs_list:
                for rec in recs_list:
                    render_recommendation_card(rec)
            else:
                st.success("No optimization triggers met. Workload memory usage is within balanced operating parameters.")
        else:
            st.info("No recommendations data available. Switch to 'Mock Simulation' or load a PyTorch allocation trace.")

    with col_batch:
        st.markdown("#### Interactive Batch Size Optimizer")
        st.caption("Evaluate candidate batch sizes against budget and safety constraints using linear memory modeling.")

        base_mb = st.slider("Base Weight & Context Memory (MiB)", 200, 3000, 800, 100)
        sample_mb = st.slider("Activation Footprint per Sample (MiB)", 5, 100, 25, 5)

        optimizer = BatchOptimizer(
            budget_bytes=budget_bytes,
            safety_margin_ratio=safety_margin_pct / 100.0,
        )

        candidates = optimizer.evaluate_sweep(
            base_bytes=int(base_mb * 1024 * 1024),
            activation_per_sample_bytes=int(sample_mb * 1024 * 1024),
            candidates=[8, 16, 32, 48, 64, 96, 128],
        )

        best_fit = optimizer.select_best_viable_candidate(candidates)

        c_rows = []
        for c in candidates:
            status_color = "#2ecc71" if c.status == "VIABLE" else ("#f39c12" if c.status == "RISKY" else "#e74c3c")
            c_rows.append({
                "Batch": c.batch_size,
                "Estimated Peak": format_bytes(c.estimated_peak_bytes),
                "Status": f'<span style="color: {status_color}; font-weight: 700;">{c.status}</span>',
                "Note": c.recommendation_note,
            })

        st.markdown(
            pd.DataFrame(c_rows).to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )

        if best_fit:
            st.success(f"⭐ **Recommended Configuration:** Batch Size **{best_fit.batch_size}** ({format_bytes(best_fit.estimated_peak_bytes)})")


# ===========================================================================
# TAB 6: Benchmark Matrix & Experimental Evaluation
# ===========================================================================
with tab_bench:
    st.subheader("📊 Experimental Evaluation: Baseline vs. Optimized")
    st.caption("Phase 7: Validating optimization hypotheses across Model × Batch Size × Precision configurations.")

    if not bench_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi_card("Total Experiments", str(len(bench_df)), "Configurations evaluated")
        with col2:
            succ = len(bench_df[bench_df["status"].astype(str).str.upper() == "SUCCESS"]) if "status" in bench_df.columns else len(bench_df)
            render_kpi_card("Successful Runs", str(succ), f"{succ/len(bench_df)*100:.0f}% completion rate", color="#2ecc71")
        with col3:
            oom_count = len(bench_df[bench_df["oom_occurred"].astype(str).str.lower() == "true"]) if "oom_occurred" in bench_df.columns else 0
            render_kpi_card("OOM Occurrences", str(oom_count), "Controlled stress test captures", color="#e74c3c" if oom_count > 0 else "#00d2ff")
        with col4:
            models_count = bench_df["workload_name"].nunique() if "workload_name" in bench_df.columns else 1
            render_kpi_card("Evaluated Workloads", str(models_count), "Workload models tested")

        # Peak Memory Comparison Bar Chart
        if "peak_memory_bytes" in bench_df.columns and "workload_name" in bench_df.columns:
            st.markdown("#### Memory Footprint: FP32 Baseline vs. FP16 Optimized")
            plot_bench = bench_df.copy()
            plot_bench["peak_mb"] = pd.to_numeric(plot_bench["peak_memory_bytes"], errors="coerce") / (1024.0 * 1024.0)
            plot_bench = plot_bench.dropna(subset=["peak_mb"])

            batch_str = plot_bench["batch_size"].astype(str) if "batch_size" in plot_bench.columns else "N/A"
            plot_bench["label"] = plot_bench["workload_name"] + " (B=" + batch_str + ")"

            prec_col = "precision" if "precision" in plot_bench.columns else "workload_name"
            bar_chart = (
                alt.Chart(plot_bench)
                .mark_bar()
                .encode(
                    x=alt.X("label:N", title="Workload & Batch Size", axis=alt.Axis(labelAngle=-25)),
                    y=alt.Y("peak_mb:Q", title="Peak Memory (MiB)"),
                    color=alt.Color(f"{prec_col}:N", title="Precision", scale=alt.Scale(scheme="category10")),
                    tooltip=["workload_name", "peak_mb"],
                )
                .properties(height=320)
            )
            st.altair_chart(bar_chart, width="stretch")

        st.markdown("#### Complete Benchmark Results Table")
        cols = [c for c in ["experiment_id", "workload_name", "batch_size", "precision", "status", "runtime_s", "peak_memory_bytes", "oom_occurred", "notes"] if c in bench_df.columns]
        st.dataframe(bench_df[cols] if cols else bench_df)
    else:
        st.info("No benchmark matrix file found. Switch to 'Mock Simulation' in the sidebar or load a benchmark CSV.")


# ===========================================================================
# TAB 7: OS Concepts Mapping & Systems Architecture
# ===========================================================================
with tab_os:
    st.subheader("📚 OS Concepts Demonstrated & Framework Architecture")
    st.caption("A teaching-scale systems implementation translating core Operating System principles to GPU workloads.")

    st.markdown(
        """
        | OS Concept | Project Application | Evidence / Metric |
        |---|---|---|
        | **Memory Management** | GPU allocation, deallocation, and caching lifecycle | Allocated vs Reserved timeline; Peak tracking |
        | **Allocation Strategies** | CUDA and PyTorch caching allocator block reuse | Segment block snapshots; Cache clear deltas |
        | **Fragmentation** | Distribution of inactive free allocator blocks | **Free-Space Fragmentation Index (FSFI)** |
        | **Resource Contention** | Concurrent multi-process GPU execution | Per-PID memory tracking; Concurrency sweeps |
        | **Cooperative Scheduling** | Workload budget reservation before launch | **Admission Control Decisions (Admit/Defer/Reject)** |
        | **Resource Safety** | Dynamic headroom and rapid allocation growth alerts | **Transparent Heuristic OOM Risk Badge** |
        | **Optimization Heuristics**| Traceable configuration recommendations | Batch scaling model, Mixed precision AMP |
        """
    )

    st.markdown("---")
    st.markdown("#### System Architecture Flow")
    st.code(
        """
        GPU Workloads (CNN / ResNet / Transformer / Synthetic Stress)
                                     │
                                     ▼
                      Telemetry Collection Layer
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
     NVML Monitor                                      PyTorch Allocator
 (collector/nvml_monitor.py)                        (collector/torch_hooks.py)
   Device VRAM & Util                                  Allocated, Reserved, Peaks
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
                         Unified Event Schema
                            (schema.py)
                                     │
         ┌──────────────────┬────────┴────────┬──────────────────┐
         ▼                  ▼                 ▼                  ▼
  Memory Pressure    Fragmentation       Contention        Optimization
  (profiler/         (profiler/         (profiler/         (optimizer/
   memory_pressure)   fragmentation)     contention)        recommendations)
         │                  │                 │                  │
         └──────────────────┴────────┬────────┴──────────────────┘
                                     ▼
                          Streamlit Dashboard
                           (dashboard/app.py)
        """,
        language="text",
    )
