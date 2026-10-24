"""
File: tests/test_pipeline.py

1. What this file does:
   Comprehensive CPU-only unit test suite for Person 3's data pipeline modules
   (`pipeline/normalize.py`, `pipeline/validate.py`, and `pipeline/aggregate.py`).

2. Overall design:
   Uses standard `pytest` test functions with synthetic data dictionaries,
   DataFrames, and in-memory CSV files. Covers edge cases, missing fields, corrupt
   rows, type conversions, summary aggregations, and downsampling peak preservation.

3. Elements in the file:
   - `test_byte_conversions`: Unit conversions for MB, GB, and formatted strings.
   - `test_normalize_events`: Normalization logic and derived column calculations.
   - `test_validation_valid_row`: Clean schema validation.
   - `test_validation_missing_required`: Flags missing required fields.
   - `test_validation_negative_memory`: Flags negative memory fields.
   - `test_validation_alloc_exceeds_resv`: Enforces allocated <= reserved constraint.
   - `test_aggregate_summary`: High-water marks, sampling rate, and ratios.
   - `test_pid_breakdown`: Per-PID grouping and statistics.
   - `test_resample_timeseries`: Decimation while retaining peak memory samples.

4. Final output of the file:
   Pytest test outcomes validating the data pipeline.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

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
    normalize_memory_events,
)
from pipeline.validate import (
    validate_event_list,
    validate_memory_event,
    validate_csv_file,
)
from pipeline.normalize import load_and_normalize_csv
from pipeline.aggregate import (
    compute_pid_breakdown,
    compute_profile_summary,
    resample_timeseries,
)


def test_byte_conversions():
    """
    1. The use of function:
       Tests unit conversions in `pipeline/normalize.py`.
    2. How does the specific function work:
       Asserts conversion of bytes to MiB, GiB, and formatted string thresholds.
    3. Concepts used in the function:
       Binary arithmetic validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if numeric conversions match exact mathematical expectations.
    """
    assert bytes_to_mb(1048576) == 1.0
    assert bytes_to_mb(0) == 0.0
    assert bytes_to_mb(None) is None

    assert bytes_to_gb(1073741824) == 1.0
    assert bytes_to_gb(0) == 0.0
    assert bytes_to_gb(None) is None

    assert format_bytes(512) == "512 B"
    assert format_bytes(1024 * 1024) == "1.00 MB"
    assert format_bytes(1024 * 1024 * 1024 * 2) == "2.00 GB"
    assert format_bytes(None) == "N/A"


def test_normalize_events():
    """
    1. The use of function:
       Tests that `normalize_memory_events` accurately computes derived metrics.
    2. How does the specific function work:
       Passes two raw dicts (NVML and PyTorch), asserting derived fields like
       `elapsed_s`, `device_used_bytes`, `cached_unused_bytes`, and `allocated_mb`.
    3. Concepts used in the function:
       Data normalization and derived metric calculation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if all derived fields match expected formulas.
    """
    events = [
        {
            "timestamp": "100.0",
            "pid": "0",
            "gpu_id": "0",
            "source": "nvml",
            "event_type": "poll",
            "total_memory": "6442450944",
            "free_memory": "4294967296",
        },
        {
            "timestamp": "102.5",
            "pid": "500",
            "gpu_id": "0",
            "source": "torch",
            "event_type": "sample",
            "allocated_bytes": "104857600",
            "reserved_bytes": "209715200",
        },
    ]

    norm = normalize_memory_events(events)
    assert len(norm) == 2

    # Row 1 (NVML)
    r1 = norm[0]
    assert r1["elapsed_s"] == 0.0
    assert r1["device_used_bytes"] == 6442450944 - 4294967296
    assert r1["device_used_mb"] == 2048.0
    assert r1["device_utilization_pct"] == pytest.approx(33.33, 0.01)

    # Row 2 (PyTorch)
    r2 = norm[1]
    assert r2["elapsed_s"] == 2.5
    assert r2["allocated_mb"] == 100.0
    assert r2["reserved_mb"] == 200.0
    assert r2["cached_unused_bytes"] == 104857600
    assert r2["cached_unused_mb"] == 100.0


def test_validation_valid_row():
    """
    1. The use of function:
       Tests validation of a fully compliant MemoryEvent dictionary.
    2. How does the specific function work:
       Supplies a well-formed event dict to `validate_memory_event` and asserts validity.
    3. Concepts used in the function:
       Schema validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if is_valid is True with empty error list.
    """
    event = {
        "timestamp": 1700000000.0,
        "pid": 100,
        "gpu_id": 0,
        "source": "torch",
        "event_type": "memory_sample",
        "allocated_bytes": 1000,
        "reserved_bytes": 2000,
    }
    is_ok, errs = validate_memory_event(event)
    assert is_ok is True
    assert len(errs) == 0


def test_validation_missing_required():
    """
    1. The use of function:
       Tests that missing required fields are caught and flagged by validator.
    2. How does the specific function work:
       Supplies dict missing `source` and `pid`, asserting errors list.
    3. Concepts used in the function:
       Mandatory field checking.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if errors cite missing required fields.
    """
    event = {
        "timestamp": 1700000000.0,
        "gpu_id": 0,
        "event_type": "memory_sample",
    }
    is_ok, errs = validate_memory_event(event)
    assert is_ok is False
    assert any("Missing required field 'pid'" in e for e in errs)
    assert any("Missing required field 'source'" in e for e in errs)


def test_validation_negative_memory():
    """
    1. The use of function:
       Tests that negative memory quantities are flagged as invalid.
    2. How does the specific function work:
       Supplies event with allocated_bytes = -500.
    3. Concepts used in the function:
       Domain constraint checking.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if negative memory violation is reported.
    """
    event = {
        "timestamp": 1700000000.0,
        "pid": 100,
        "gpu_id": 0,
        "source": "torch",
        "event_type": "memory_sample",
        "allocated_bytes": -500,
    }
    is_ok, errs = validate_memory_event(event)
    assert is_ok is False
    assert any("cannot be negative" in e for e in errs)


def test_validation_alloc_exceeds_resv():
    """
    1. The use of function:
       Tests that allocator constraint (allocated <= reserved) is validated.
    2. How does the specific function work:
       Supplies allocated_bytes=5000 and reserved_bytes=2000, asserting failure.
    3. Concepts used in the function:
       Cross-field logical consistency validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if violation is flagged.
    """
    event = {
        "timestamp": 1700000000.0,
        "pid": 100,
        "gpu_id": 0,
        "source": "torch",
        "event_type": "memory_sample",
        "allocated_bytes": 5000,
        "reserved_bytes": 2000,
    }
    is_ok, errs = validate_memory_event(event)
    assert is_ok is False
    assert any("exceeds reserved_bytes" in e for e in errs)


def test_aggregate_summary():
    """
    1. The use of function:
       Tests profile high-water mark and duration calculation in `pipeline/aggregate.py`.
    2. How does the specific function work:
       Builds a DataFrame with 4 timestamped events and asserts aggregated metrics.
    3. Concepts used in the function:
       Statistical aggregation and summary calculation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if peak_allocated, duration, and sampling frequency match expected values.
    """
    df = pd.DataFrame([
        {"timestamp": 100.0, "elapsed_s": 0.0, "pid": 10, "allocated_bytes": 1000, "reserved_bytes": 2000},
        {"timestamp": 101.0, "elapsed_s": 1.0, "pid": 10, "allocated_bytes": 5000, "reserved_bytes": 6000},
        {"timestamp": 102.0, "elapsed_s": 2.0, "pid": 10, "allocated_bytes": 3000, "reserved_bytes": 6000},
    ])
    summary = compute_profile_summary(df, budget_bytes=10000)
    assert summary.total_events == 3
    assert summary.duration_s == 2.0
    assert summary.sampling_interval_s == 1.0
    assert summary.peak_allocated_bytes == 5000
    assert summary.peak_reserved_bytes == 6000
    assert summary.peak_to_budget_ratio == 0.5
    assert summary.pids_observed == [10]


def test_pid_breakdown():
    """
    1. The use of function:
       Tests per-PID grouping and statistics calculation.
    2. How does the specific function work:
       Builds a multi-PID DataFrame and asserts that `compute_pid_breakdown`
       returns one row per PID with accurate peak values.
    3. Concepts used in the function:
       Process accounting and DataFrame grouping.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if per-PID peak memory and sample counts match.
    """
    df = pd.DataFrame([
        {"timestamp": 10.0, "elapsed_s": 0.0, "pid": 101, "allocated_bytes": 1000, "workload_name": "CNN"},
        {"timestamp": 11.0, "elapsed_s": 1.0, "pid": 101, "allocated_bytes": 2500, "workload_name": "CNN"},
        {"timestamp": 12.0, "elapsed_s": 2.0, "pid": 202, "allocated_bytes": 4000, "workload_name": "ResNet"},
    ])
    pids_df = compute_pid_breakdown(df)
    assert len(pids_df) == 2

    # Should be sorted by peak allocated descending
    assert pids_df.iloc[0]["pid"] == 202
    assert pids_df.iloc[0]["peak_allocated_bytes"] == 4000
    assert pids_df.iloc[1]["pid"] == 101
    assert pids_df.iloc[1]["peak_allocated_bytes"] == 2500


def test_resample_timeseries():
    """
    1. The use of function:
       Tests that downsampling reduces row count while preserving peak high-water marks.
    2. How does the specific function work:
       Creates a 1,000-row DataFrame with a sharp spike at row 450.
       Downsamples to 50 target points and asserts that the spike at row 450 is retained.
    3. Concepts used in the function:
       Peak-preserving decimation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if resampled DataFrame has fewer rows but exact peak value.
    """
    n = 1000
    times = list(range(n))
    allocs = [100] * n
    allocs[450] = 999999  # Sharp isolated peak

    df = pd.DataFrame({
        "timestamp": times,
        "elapsed_s": times,
        "allocated_bytes": allocs,
    })

    resampled = resample_timeseries(df, target_points=50)
    assert len(resampled) < 100
    # The max value must still be present!
    assert resampled["allocated_bytes"].max() == 999999


def test_load_and_normalize_csv_file_buffer():
    """
    1. The use of function:
       Tests that `load_and_normalize_csv` correctly handles uploaded file-like buffers
       (such as Streamlit UploadedFile or io.BytesIO) without raising a TypeError.
    2. How does the specific function work:
       Creates an io.BytesIO buffer with CSV text, calls `load_and_normalize_csv`,
       and verifies that records are normalized into a DataFrame.
    3. Concepts used in the function:
       In-memory buffer parsing and polymorphic file handling.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if DataFrame contains expected rows and columns.
    """
    import io

    csv_content = (
        b"timestamp,pid,gpu_id,source,event_type,allocated_bytes,reserved_bytes\n"
        b"100.0,42,0,torch,memory_sample,104857600,209715200\n"
        b"101.0,42,0,torch,memory_sample,157286400,209715200\n"
    )
    buffer = io.BytesIO(csv_content)

    df = load_and_normalize_csv(buffer)
    assert not df.empty
    assert len(df) == 2
    assert "allocated_mb" in df.columns
    assert df["allocated_mb"].iloc[0] == 100.0
    assert df["allocated_mb"].iloc[1] == 150.0


def test_validate_csv_file_buffer():
    """
    1. The use of function:
       Tests that `validate_csv_file` validates in-memory uploaded file buffers.
    2. How does the specific function work:
       Passes io.BytesIO buffer to `validate_csv_file` and verifies `is_valid` is True.
    3. Concepts used in the function:
       Defensive stream validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if ValidationResult reports 2/2 valid rows.
    """
    import io

    csv_content = (
        b"timestamp,pid,gpu_id,source,event_type,total_memory,free_memory\n"
        b"100.0,0,0,nvml,memory_sample,6442450944,4294967296\n"
        b"101.0,0,0,nvml,memory_sample,6442450944,4000000000\n"
    )
    buffer = io.BytesIO(csv_content)

    rows, res = validate_csv_file(buffer)
    assert res.is_valid is True
    assert res.valid_rows == 2
    assert res.invalid_rows == 0

