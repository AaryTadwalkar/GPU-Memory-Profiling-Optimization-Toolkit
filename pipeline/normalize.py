"""
File: pipeline/normalize.py

1. What this file does:
   Normalizes raw telemetry CSV files and MemoryEvent streams. Standardizes column
   names, scales byte measurements to human-readable units (MiB/GiB), calculates
   relative elapsed timestamps, and cleanly structures data into pandas DataFrames.

2. Overall design:
   A lightweight, pure-CPU data normalization module. It consumes raw dictionaries
   or CSV files conforming to `schema.py` and produces standardized, typed pandas
   DataFrames. It enforces separation between device-wide NVML metrics and in-process
   PyTorch allocator metrics, ensuring that downstream profilers and the dashboard
   receive clean, consistent data without type errors or missing keys.

3. Elements in the file:
   - `bytes_to_mb`: Converts bytes to megabytes (MiB, 1024^2).
   - `bytes_to_gb`: Converts bytes to gigabytes (GiB, 1024^3).
   - `format_bytes`: Formats raw bytes into a concise, readable string.
   - `normalize_memory_events`: Normalizes a list of MemoryEvent dicts.
   - `events_to_dataframe`: Converts normalized events to a typed pandas DataFrame.
   - `load_and_normalize_csv`: Reads a CSV file from disk and returns a clean DataFrame.

4. Final output of the file:
   Normalized, typed pandas DataFrames with derived helper columns (e.g., elapsed_time_s,
   allocated_mb, reserved_mb, device_used_bytes, cached_unused_bytes) ready for
   dashboard visualization and analytical aggregation.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schema import MemoryEvent


def bytes_to_mb(b: Optional[Union[int, float]]) -> Optional[float]:
    """
    1. The use of function:
       Converts raw byte integer/float values into mebibytes (MiB).
    2. How does the specific function work:
       Divides the input value by 1,048,576 (1024 * 1024). Returns None if input is None.
    3. Concepts used in the function:
       Binary unit conversion (1 MiB = 2^20 bytes).
    4. Places where this function is imported and used:
       Used across the data pipeline, dashboard charts, and metric displays.
    5. Final output of the function, before and after:
       Before: 104857600 (int bytes)
       After: 100.0 (float MiB)
    """
    if b is None or pd.isna(b):
        return None
    try:
        return float(b) / (1024.0 * 1024.0)
    except (ValueError, TypeError):
        return None


def bytes_to_gb(b: Optional[Union[int, float]]) -> Optional[float]:
    """
    1. The use of function:
       Converts raw byte integer/float values into gibibytes (GiB).
    2. How does the specific function work:
       Divides the input value by 1,073,741,824 (1024 * 1024 * 1024). Returns None if input is None.
    3. Concepts used in the function:
       Binary unit conversion (1 GiB = 2^30 bytes).
    4. Places where this function is imported and used:
       Used by dashboard device overview gauges and capacity indicators.
    5. Final output of the function, before and after:
       Before: 6442450944 (int bytes)
       After: 6.0 (float GiB)
    """
    if b is None or pd.isna(b):
        return None
    try:
        return float(b) / (1024.0 * 1024.0 * 1024.0)
    except (ValueError, TypeError):
        return None


def format_bytes(b: Optional[Union[int, float]], precision: int = 2) -> str:
    """
    1. The use of function:
       Formats raw bytes into a human-readable string with appropriate units (B, KB, MB, GB).
    2. How does the specific function work:
       Checks magnitude and divides by 1024 progressively, returning a formatted string.
    3. Concepts used in the function:
       String formatting and threshold comparison.
    4. Places where this function is imported and used:
       Used by UI cards, table cells, and console summaries.
    5. Final output of the function, before and after:
       Before: 52428800
       After: "50.00 MB"
    """
    if b is None or pd.isna(b):
        return "N/A"
    try:
        val = float(b)
    except (ValueError, TypeError):
        return "N/A"

    if val < 0:
        return f"-{format_bytes(abs(val), precision)}"
    if val < 1024:
        return f"{val:.0f} B"
    elif val < 1024 * 1024:
        return f"{val / 1024.0:.{precision}f} KB"
    elif val < 1024 * 1024 * 1024:
        return f"{val / (1024.0 * 1024.0):.{precision}f} MB"
    else:
        return f"{val / (1024.0 * 1024.0 * 1024.0):.{precision}f} GB"


def normalize_memory_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    1. The use of function:
       Normalizes an array of raw event dictionaries into clean, standardized records.
    2. How does the specific function work:
       Iterates over dictionaries, converts numeric fields safely, calculates elapsed
       time from the initial timestamp, and computes derived fields such as device
       used memory and cached-but-unused memory.
    3. Concepts used in the function:
       Time-series normalization and missing-value handling.
    4. Places where this function is imported and used:
       Called by `events_to_dataframe` and the data pipeline ingestion flow.
    5. Final output of the function, before and after:
       Before: List of raw, potentially sparse or string-typed event dicts.
       After: List of consistently typed dicts with derived analytical fields.
    """
    if not events:
        return []

    # Find earliest valid timestamp for relative time calculation
    valid_timestamps = [
        float(e["timestamp"])
        for e in events
        if e.get("timestamp") is not None and str(e["timestamp"]).strip() != ""
    ]
    t0 = min(valid_timestamps) if valid_timestamps else 0.0

    normalized: List[Dict[str, Any]] = []
    for item in events:
        row = dict(item)

        # Parse timestamp and relative elapsed time
        try:
            ts = float(row.get("timestamp", 0.0))
            row["timestamp"] = ts
            row["elapsed_s"] = round(ts - t0, 4) if t0 > 0 else 0.0
        except (ValueError, TypeError):
            row["timestamp"] = 0.0
            row["elapsed_s"] = 0.0

        # Parse PID and GPU ID
        try:
            row["pid"] = int(row.get("pid", 0))
        except (ValueError, TypeError):
            row["pid"] = 0

        try:
            row["gpu_id"] = int(row.get("gpu_id", 0))
        except (ValueError, TypeError):
            row["gpu_id"] = 0

        # Safe parsing for memory quantities (None if missing or invalid)
        for field in [
            "total_memory",
            "free_memory",
            "process_memory",
            "allocated_bytes",
            "reserved_bytes",
            "allocation_delta",
            "peak_memory",
            "batch_size",
        ]:
            val = row.get(field)
            if val is None or val == "" or pd.isna(val):
                row[field] = None
            else:
                try:
                    row[field] = int(float(val))
                except (ValueError, TypeError):
                    row[field] = None

        # Derived metrics: Device used memory (NVML)
        total = row.get("total_memory")
        free = row.get("free_memory")
        if total is not None and free is not None:
            row["device_used_bytes"] = max(0, total - free)
            row["device_used_mb"] = bytes_to_mb(row["device_used_bytes"])
            row["device_free_mb"] = bytes_to_mb(free)
            row["device_total_mb"] = bytes_to_mb(total)
            row["device_utilization_pct"] = (
                round((row["device_used_bytes"] / total) * 100.0, 2)
                if total > 0
                else 0.0
            )
        else:
            row["device_used_bytes"] = None
            row["device_used_mb"] = None
            row["device_free_mb"] = None
            row["device_total_mb"] = None
            row["device_utilization_pct"] = None

        # Derived metrics: In-process allocator (PyTorch)
        alloc = row.get("allocated_bytes")
        resv = row.get("reserved_bytes")
        if alloc is not None:
            row["allocated_mb"] = bytes_to_mb(alloc)
        else:
            row["allocated_mb"] = None

        if resv is not None:
            row["reserved_mb"] = bytes_to_mb(resv)
        else:
            row["reserved_mb"] = None

        if alloc is not None and resv is not None:
            row["cached_unused_bytes"] = max(0, resv - alloc)
            row["cached_unused_mb"] = bytes_to_mb(row["cached_unused_bytes"])
        else:
            row["cached_unused_bytes"] = None
            row["cached_unused_mb"] = None

        if row.get("peak_memory") is not None:
            row["peak_memory_mb"] = bytes_to_mb(row["peak_memory"])
        else:
            row["peak_memory_mb"] = None

        normalized.append(row)

    return normalized


def events_to_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    1. The use of function:
       Converts a list of event dictionaries into a normalized pandas DataFrame.
    2. How does the specific function work:
       Runs `normalize_memory_events` on the list and constructs a DataFrame with
       explicit column typing.
    3. Concepts used in the function:
       Pandas DataFrame construction and typing.
    4. Places where this function is imported and used:
       Used by data ingestion pipelines and the Streamlit dashboard loaders.
    5. Final output of the function, before and after:
       Before: Raw Python list of dicts.
       After: Clean, indexed pandas DataFrame.
    """
    if not events:
        return pd.DataFrame()
    norm = normalize_memory_events(events)
    df = pd.DataFrame(norm)
    return df


def load_and_normalize_csv(filepath: Any) -> pd.DataFrame:
    """
    1. The use of function:
       Loads a MemoryEvent CSV file from disk, an in-memory buffer, or an uploaded
       file object (e.g., Streamlit UploadedFile) and normalizes its content into a DataFrame.
    2. How does the specific function work:
       - If given an object with `.read()` or `.getvalue()` (such as Streamlit UploadedFile
         or io.StringIO), it decodes the text and parses CSV rows via `csv.DictReader`.
       - If given a string or Path object, it safely verifies file existence before opening.
       - Passes records to `events_to_dataframe`, returning a standardized DataFrame.
       - Returns an empty DataFrame if the input is None, unparseable, or empty.
    3. Concepts used in the function:
       Polymorphic file/stream ingestion, CSV parsing, tabular normalization.
    4. Places where this function is imported and used:
       Called by dashboard tab views, test suites, and batch processing scripts.
    5. Final output of the function, before and after:
       Before: Raw CSV file, Path, or Streamlit UploadedFile.
       After: Standardized pandas DataFrame ready for analysis.
    """
    if filepath is None:
        return pd.DataFrame()

    import io

    rows: List[Dict[str, Any]] = []

    # Case 1: Streamlit UploadedFile or file-like buffer (has getvalue or read)
    if hasattr(filepath, "getvalue") or hasattr(filepath, "read"):
        try:
            if hasattr(filepath, "getvalue"):
                raw_data = filepath.getvalue()
                if isinstance(raw_data, bytes):
                    content = raw_data.decode("utf-8-sig", errors="replace")
                else:
                    content = str(raw_data)
            else:
                raw_data = filepath.read()
                if isinstance(raw_data, bytes):
                    content = raw_data.decode("utf-8-sig", errors="replace")
                else:
                    content = str(raw_data)

            # Reset pointer if seekable
            if hasattr(filepath, "seek"):
                filepath.seek(0)

            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                rows.append(row)
            return events_to_dataframe(rows)
        except Exception:
            return pd.DataFrame()

    # Case 2: String or PathLike file path on disk
    try:
        p = Path(filepath)
    except (TypeError, ValueError):
        return pd.DataFrame()

    if not p.is_file():
        return pd.DataFrame()

    try:
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return pd.DataFrame()

    return events_to_dataframe(rows)
