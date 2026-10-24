"""
File: tests/test_schema.py

1. What this file does:
   Comprehensive CPU-only unit test suite for `schema.py` (Phase 0.5 exit criteria).
   Validates the canonical `MemoryEvent` dataclass, default field assignments,
   dictionary serialization (`to_row`), type integrity, and CSV round-tripping.

2. Overall design:
   Uses standard `unittest` and `pytest` compatible test functions. Requires zero
   CUDA or GPU dependencies, executing purely in memory with synthetic test values.

3. Elements in the file:
   - `test_memory_event_minimal_init`: Verifies initialization with required fields.
   - `test_memory_event_all_fields`: Verifies all optional fields can be populated.
   - `test_to_row_keys_and_values`: Verifies dictionary output matches dataclass fields.
   - `test_to_row_type_preservation`: Ensures integers and floats preserve their types.
   - `test_metadata_json_roundtrip`: Tests JSON serialization in `metadata_json`.
   - `test_csv_roundtrip`: Verifies CSV write and read fidelity.

4. Final output of the file:
   Pytest-compatible test assertions validating schema consistency.
"""

import csv
import io
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schema import MemoryEvent


def test_memory_event_minimal_init():
    """
    1. The use of function:
       Tests that MemoryEvent can be instantiated with only the 5 required fields.
    2. How does the specific function work:
       Creates an instance passing timestamp, pid, gpu_id, source, and event_type.
       Asserts that all optional memory fields default to None.
    3. Concepts used in the function:
       Dataclass default value verification.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if all defaults are None.
    """
    ev = MemoryEvent(
        timestamp=1700000000.0,
        pid=1234,
        gpu_id=0,
        source="nvml",
        event_type="memory_sample",
    )
    assert ev.timestamp == 1700000000.0
    assert ev.pid == 1234
    assert ev.gpu_id == 0
    assert ev.source == "nvml"
    assert ev.event_type == "memory_sample"
    assert ev.total_memory is None
    assert ev.free_memory is None
    assert ev.process_memory is None
    assert ev.allocated_bytes is None
    assert ev.reserved_bytes is None
    assert ev.allocation_delta is None
    assert ev.peak_memory is None
    assert ev.workload_name is None
    assert ev.batch_size is None
    assert ev.precision is None
    assert ev.metadata_json is None
    assert ev.snapshot_path is None


def test_memory_event_all_fields():
    """
    1. The use of function:
       Tests that MemoryEvent correctly stores every single supported field.
    2. How does the specific function work:
       Supplies explicit values for all 17 dataclass fields and asserts equality.
    3. Concepts used in the function:
       Dataclass field assignment and storage verification.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if all supplied values are preserved accurately.
    """
    ev = MemoryEvent(
        timestamp=1700000005.123,
        pid=5678,
        gpu_id=1,
        source="torch",
        event_type="snapshot",
        total_memory=6442450944,
        free_memory=3221225472,
        process_memory=1073741824,
        allocated_bytes=524288000,
        reserved_bytes=629145600,
        allocation_delta=52428800,
        peak_memory=524288000,
        workload_name="ResNet50",
        batch_size=32,
        precision="FP16",
        metadata_json='{"status": "ok"}',
        snapshot_path="data/snapshots/snap1.json",
    )
    assert ev.timestamp == 1700000005.123
    assert ev.pid == 5678
    assert ev.gpu_id == 1
    assert ev.source == "torch"
    assert ev.event_type == "snapshot"
    assert ev.total_memory == 6442450944
    assert ev.free_memory == 3221225472
    assert ev.process_memory == 1073741824
    assert ev.allocated_bytes == 524288000
    assert ev.reserved_bytes == 629145600
    assert ev.allocation_delta == 52428800
    assert ev.peak_memory == 524288000
    assert ev.workload_name == "ResNet50"
    assert ev.batch_size == 32
    assert ev.precision == "FP16"
    assert ev.metadata_json == '{"status": "ok"}'
    assert ev.snapshot_path == "data/snapshots/snap1.json"


def test_to_row_keys_and_values():
    """
    1. The use of function:
       Verifies `to_row()` converts the dataclass into a dict with all expected keys.
    2. How does the specific function work:
       Calls `to_row()` on a MemoryEvent instance and asserts that returned dict has
       all canonical schema keys.
    3. Concepts used in the function:
       Dataclass serialization to dictionary.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if dictionary contains exact schema fields.
    """
    ev = MemoryEvent(
        timestamp=100.0,
        pid=1,
        gpu_id=0,
        source="nvml",
        event_type="poll",
        total_memory=1000,
    )
    row = ev.to_row()
    assert isinstance(row, dict)
    expected_keys = {
        "timestamp",
        "pid",
        "gpu_id",
        "source",
        "event_type",
        "total_memory",
        "free_memory",
        "process_memory",
        "allocated_bytes",
        "reserved_bytes",
        "allocation_delta",
        "peak_memory",
        "workload_name",
        "batch_size",
        "precision",
        "metadata_json",
        "snapshot_path",
    }
    assert set(row.keys()) == expected_keys
    assert row["timestamp"] == 100.0
    assert row["total_memory"] == 1000
    assert row["free_memory"] is None


def test_to_row_type_preservation():
    """
    1. The use of function:
       Tests that numeric and string types are preserved without accidental coercion.
    2. How does the specific function work:
       Asserts `isinstance` checks on the output of `to_row()`.
    3. Concepts used in the function:
       Type fidelity validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if types match python primitives.
    """
    ev = MemoryEvent(
        timestamp=123.456,
        pid=999,
        gpu_id=0,
        source="torch",
        event_type="alloc",
        allocated_bytes=4096,
        workload_name="TestWL",
    )
    row = ev.to_row()
    assert isinstance(row["timestamp"], float)
    assert isinstance(row["pid"], int)
    assert isinstance(row["gpu_id"], int)
    assert isinstance(row["source"], str)
    assert isinstance(row["allocated_bytes"], int)
    assert isinstance(row["workload_name"], str)


def test_metadata_json_roundtrip():
    """
    1. The use of function:
       Tests that JSON payload stored in `metadata_json` can be serialized and deserialized.
    2. How does the specific function work:
       Encodes a Python dictionary to JSON, stores in `metadata_json`, serializes to row,
       and asserts that `json.loads(row["metadata_json"])` restores original dict.
    3. Concepts used in the function:
       JSON metadata embedding.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if dictionary content matches perfectly.
    """
    meta_payload = {"decision": "ADMIT", "headroom_ratio": 0.42, "retries": 0}
    ev = MemoryEvent(
        timestamp=200.0,
        pid=10,
        gpu_id=0,
        source="scheduler",
        event_type="admission",
        metadata_json=json.dumps(meta_payload),
    )
    row = ev.to_row()
    restored = json.loads(row["metadata_json"])
    assert restored["decision"] == "ADMIT"
    assert restored["headroom_ratio"] == 0.42
    assert restored["retries"] == 0


def test_csv_roundtrip():
    """
    1. The use of function:
       Tests serialization to CSV and reading back via DictReader.
    2. How does the specific function work:
       Uses an in-memory `io.StringIO` buffer to write a MemoryEvent row via
       `csv.DictWriter` and parse it back via `csv.DictReader`.
    3. Concepts used in the function:
       CSV serialization round-trip verification.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Passes if CSV parsed string fields correspond to the serialized event.
    """
    ev = MemoryEvent(
        timestamp=1700000000.5,
        pid=42,
        gpu_id=0,
        source="nvml",
        event_type="memory_sample",
        total_memory=6442450944,
        free_memory=3221225472,
    )
    row = ev.to_row()

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)

    buffer.seek(0)
    reader = csv.DictReader(buffer)
    read_rows = list(reader)

    assert len(read_rows) == 1
    r = read_rows[0]
    assert float(r["timestamp"]) == 1700000000.5
    assert int(r["pid"]) == 42
    assert int(r["gpu_id"]) == 0
    assert r["source"] == "nvml"
    assert int(r["total_memory"]) == 6442450944
    assert int(r["free_memory"]) == 3221225472
    assert r["allocated_bytes"] == ""  # None serializes to empty string in CSV
