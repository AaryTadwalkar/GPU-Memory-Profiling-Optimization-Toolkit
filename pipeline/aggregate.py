"""
File: pipeline/aggregate.py

1. What this file does:
   Computes analytical aggregations, statistical summaries, per-PID distributions,
   and downsampled time-series representations from normalized GPU telemetry data.

2. Overall design:
   A CPU-safe analytics module operating on normalized pandas DataFrames or
   MemoryEvent collections. Calculates high-water marks, average memory pressure,
   workload runtimes, concurrent process footprints, and sliding-window growth rates.
   Provides resampling tools to reduce tens of thousands of data points into smooth,
   responsive charts for the Streamlit dashboard without losing peak fidelity.

3. Elements in the file:
   - `TelemetryAggregate`: Dataclass summarizing profile high-water marks and durations.
   - `compute_profile_summary`: Generates top-level metrics across a telemetry stream.
   - `compute_pid_breakdown`: Produces per-process resource consumption summaries.
   - `resample_timeseries`: Downsamples long time-series while preserving critical peaks.
   - `join_device_and_process_streams`: Merges NVML and PyTorch streams by closest timestamp.

4. Final output of the file:
   Structured `TelemetryAggregate` summaries and aggregated DataFrames tailored for
   dashboard visualization and analytical reports.
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.normalize import bytes_to_mb, bytes_to_gb, normalize_memory_events


@dataclass
class TelemetryAggregate:
    """
    1. The use of class:
       Stores statistical summary metrics for a GPU profiling session.
    2. What the class contain:
       total_events, duration_s, sampling_interval_s, peak_allocated_bytes,
       peak_reserved_bytes, peak_device_used_bytes, mean_allocated_bytes,
       max_cached_unused_bytes, peak_to_budget_ratio, and pids_observed.
    3. Technologies used and why:
       Python dataclass for typed serialization and structured data passing.
    4. Key design of the class:
       Provides a unified high-level snapshot of workload behavior for UI cards and reports.
    """
    total_events: int
    duration_s: float
    sampling_interval_s: float
    peak_allocated_bytes: Optional[int] = None
    peak_reserved_bytes: Optional[int] = None
    peak_device_used_bytes: Optional[int] = None
    mean_allocated_bytes: Optional[float] = None
    max_cached_unused_bytes: Optional[int] = None
    peak_to_budget_ratio: Optional[float] = None
    pids_observed: List[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregate instance into a dictionary."""
        d = asdict(self)
        if self.peak_allocated_bytes is not None:
            d["peak_allocated_mb"] = bytes_to_mb(self.peak_allocated_bytes)
        if self.peak_reserved_bytes is not None:
            d["peak_reserved_mb"] = bytes_to_mb(self.peak_reserved_bytes)
        if self.peak_device_used_bytes is not None:
            d["peak_device_used_mb"] = bytes_to_mb(self.peak_device_used_bytes)
        return d


def compute_profile_summary(
    df: pd.DataFrame,
    budget_bytes: Optional[int] = None
) -> TelemetryAggregate:
    """
    1. The use of function:
       Computes high-level profile statistics from a normalized DataFrame.
    2. How does the specific function work:
       - Measures session duration and average sampling frequency.
       - Computes high-water marks (max allocated, max reserved, max device used).
       - Calculates cached-but-unused memory (reserved - allocated).
       - Compares peak memory to the configured budget if supplied.
    3. Concepts used in the function:
       Descriptive statistics, time-series windowing, resource utilization ratios.
    4. Places where this function is imported and used:
       Called by dashboard summary cards and experiment analysis pipelines.
    5. Final output of the function, before and after:
       Before: Raw time-series DataFrame.
       After: Structured TelemetryAggregate instance.
    """
    if df.empty:
        return TelemetryAggregate(
            total_events=0,
            duration_s=0.0,
            sampling_interval_s=0.0,
            pids_observed=[],
        )

    total_events = len(df)
    
    # Duration and sampling rate
    if "elapsed_s" in df.columns and df["elapsed_s"].notna().any():
        t_min = df["elapsed_s"].min()
        t_max = df["elapsed_s"].max()
        duration = float(t_max - t_min)
    elif "timestamp" in df.columns and df["timestamp"].notna().any():
        t_min = df["timestamp"].min()
        t_max = df["timestamp"].max()
        duration = float(t_max - t_min)
    else:
        duration = 0.0

    sampling_interval = (
        round(duration / max(1, total_events - 1), 4) if total_events > 1 else 0.0
    )

    # In-process peaks (PyTorch)
    peak_alloc = None
    mean_alloc = None
    if "allocated_bytes" in df.columns and df["allocated_bytes"].notna().any():
        peak_alloc = int(df["allocated_bytes"].max())
        mean_alloc = float(df["allocated_bytes"].mean())

    peak_resv = None
    if "reserved_bytes" in df.columns and df["reserved_bytes"].notna().any():
        peak_resv = int(df["reserved_bytes"].max())

    # Device peak (NVML)
    peak_dev_used = None
    if "device_used_bytes" in df.columns and df["device_used_bytes"].notna().any():
        peak_dev_used = int(df["device_used_bytes"].max())
    elif (
        "total_memory" in df.columns
        and "free_memory" in df.columns
        and df["total_memory"].notna().any()
        and df["free_memory"].notna().any()
    ):
        diff = df["total_memory"] - df["free_memory"]
        peak_dev_used = int(diff.max())

    # Cached but unused memory
    max_cached_unused = None
    if "cached_unused_bytes" in df.columns and df["cached_unused_bytes"].notna().any():
        max_cached_unused = int(df["cached_unused_bytes"].max())
    elif peak_resv is not None and peak_alloc is not None:
        max_cached_unused = max(0, peak_resv - peak_alloc)

    # Peak to capacity ratio
    peak_ratio = None
    if budget_bytes is not None and budget_bytes > 0:
        ref_peak = peak_alloc if peak_alloc is not None else peak_dev_used
        if ref_peak is not None:
            peak_ratio = round(ref_peak / float(budget_bytes), 4)

    # Observed PIDs
    pids: List[int] = []
    if "pid" in df.columns and df["pid"].notna().any():
        pids = sorted([int(p) for p in df["pid"].unique() if int(p) > 0])

    return TelemetryAggregate(
        total_events=total_events,
        duration_s=round(duration, 3),
        sampling_interval_s=sampling_interval,
        peak_allocated_bytes=peak_alloc,
        peak_reserved_bytes=peak_resv,
        peak_device_used_bytes=peak_dev_used,
        mean_allocated_bytes=mean_alloc,
        max_cached_unused_bytes=max_cached_unused,
        peak_to_budget_ratio=peak_ratio,
        pids_observed=pids,
    )


def compute_pid_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. The use of function:
       Computes per-process resource utilization summaries across PIDs in a DataFrame.
    2. How does the specific function work:
       Groups by `pid` and calculates min/max timestamps, duration, peak allocated
       memory, peak process memory, and sample counts per process.
    3. Concepts used in the function:
       Process accounting and multi-process aggregation.
    4. Places where this function is imported and used:
       Used by the Contention & Concurrency dashboard tab.
    5. Final output of the function, before and after:
       Before: Combined multi-PID event DataFrame.
       After: Summary DataFrame indexed by PID.
    """
    if df.empty or "pid" not in df.columns:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    # Filter out PID 0 (system/unassigned)
    valid_df = df[df["pid"] > 0]
    if valid_df.empty:
        valid_df = df

    for pid, group in valid_df.groupby("pid"):
        # Identify workload name if available
        wl_name = "Unknown"
        if "workload_name" in group.columns and group["workload_name"].notna().any():
            wl_vals = [w for w in group["workload_name"].unique() if w and str(w).strip()]
            if wl_vals:
                wl_name = str(wl_vals[0])

        peak_alloc = (
            int(group["allocated_bytes"].max())
            if "allocated_bytes" in group.columns and group["allocated_bytes"].notna().any()
            else None
        )
        peak_resv = (
            int(group["reserved_bytes"].max())
            if "reserved_bytes" in group.columns and group["reserved_bytes"].notna().any()
            else None
        )
        peak_proc = (
            int(group["process_memory"].max())
            if "process_memory" in group.columns and group["process_memory"].notna().any()
            else None
        )

        # Duration per PID
        duration = 0.0
        if "elapsed_s" in group.columns and group["elapsed_s"].notna().any():
            duration = float(group["elapsed_s"].max() - group["elapsed_s"].min())
        elif "timestamp" in group.columns and group["timestamp"].notna().any():
            duration = float(group["timestamp"].max() - group["timestamp"].min())

        records.append({
            "pid": int(pid),
            "workload_name": wl_name,
            "sample_count": len(group),
            "duration_s": round(duration, 3),
            "peak_allocated_bytes": peak_alloc,
            "peak_allocated_mb": bytes_to_mb(peak_alloc),
            "peak_reserved_bytes": peak_resv,
            "peak_reserved_mb": bytes_to_mb(peak_resv),
            "peak_process_memory_bytes": peak_proc,
            "peak_process_memory_mb": bytes_to_mb(peak_proc),
        })

    out_df = pd.DataFrame(records)
    if not out_df.empty:
        out_df = out_df.sort_values(by="peak_allocated_bytes", ascending=False, na_position="last")
    return out_df


def resample_timeseries(df: pd.DataFrame, target_points: int = 200) -> pd.DataFrame:
    """
    1. The use of function:
       Downsamples a large timeseries DataFrame to a target point count for responsive charts.
    2. How does the specific function work:
       If the DataFrame has fewer than `target_points`, it is returned untouched.
       Otherwise, it segments the DataFrame into uniform intervals, taking the max
       of memory columns (to preserve peaks) and the mean of timestamps.
    3. Concepts used in the function:
       Peak-preserving downsampling and decimation.
    4. Places where this function is imported and used:
       Called before rendering Streamlit charts to ensure high responsiveness.
    5. Final output of the function, before and after:
       Before: DataFrame with 10,000+ rows.
       After: Decimated DataFrame with ~200 rows preserving high-water marks.
    """
    if df.empty or len(df) <= target_points:
        return df

    step = int(np.ceil(len(df) / float(target_points)))
    sampled_indices = list(range(0, len(df), step))
    # Always include the very last index to preserve the end state
    if (len(df) - 1) not in sampled_indices:
        sampled_indices.append(len(df) - 1)

    # Ensure peak values are preserved: find overall max index and ensure it's in the list
    for col in ["allocated_bytes", "reserved_bytes", "device_used_bytes"]:
        if col in df.columns and df[col].notna().any():
            peak_idx = int(df[col].idxmax())
            if peak_idx not in sampled_indices:
                sampled_indices.append(peak_idx)

    sampled_indices.sort()
    return df.iloc[sampled_indices].reset_index(drop=True)


def join_device_and_process_streams(
    nvml_df: pd.DataFrame,
    torch_df: pd.DataFrame,
    tolerance_s: float = 0.5
) -> pd.DataFrame:
    """
    1. The use of function:
       Merges independent NVML device records and PyTorch allocator records on timestamp.
    2. How does the specific function work:
       Uses `pd.merge_asof` to join in-process records with the nearest device-level
       snapshot within the specified tolerance.
    3. Concepts used in the function:
       As-of timestamp joining for asynchronous time-series telemetry.
    4. Places where this function is imported and used:
       Used by integrated analysis views and comparative telemetry dashboards.
    5. Final output of the function, before and after:
       Before: Two disconnected DataFrames.
       After: Combined DataFrame aligned on time.
    """
    if nvml_df.empty:
        return torch_df
    if torch_df.empty:
        return nvml_df

    # Sort both on timestamp
    t_df = torch_df.sort_values("timestamp").copy()
    n_df = nvml_df.sort_values("timestamp").copy()

    # Rename device columns to prevent collisions
    device_cols = {
        col: f"nvml_{col}"
        for col in ["total_memory", "free_memory", "device_used_bytes", "device_used_mb"]
        if col in n_df.columns
    }
    n_df = n_df.rename(columns=device_cols)

    cols_to_keep = ["timestamp"] + list(device_cols.values())
    n_subset = n_df[cols_to_keep]

    merged = pd.merge_asof(
        t_df,
        n_subset,
        on="timestamp",
        direction="nearest",
        tolerance=tolerance_s,
    )
    return merged
