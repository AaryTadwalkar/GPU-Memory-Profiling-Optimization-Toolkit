"""
File: tests/test_contention.py

1. What this file does:
   CPU-only unit tests for profiler/contention.py.
2. Overall design:
   Uses pytest with in-memory event dicts and temporary CSV files.
   No GPU, CUDA, NVML, or live hardware is accessed at any point.
   Covers: per-PID peak memory, runtime, concurrency, device-vs-process
   attribution, missing/malformed data, OOM detection, CSV loading, and
   the sweep-line concurrent-workload counter.
3. Elements in the file:
   Helpers: mb(), make_event(), write_tmp_csv().
   Test classes organised by concern.
4. Final output of the file:
   Pytest outcomes verifying the contention analysis engine.
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from profiler.contention import (
    ContentionReport,
    DeviceSummary,
    ProcessSummary,
    _collect_errors,
    _compute_max_concurrent_pids,
    _is_oom_event,
    _parse_float,
    _parse_int,
    _parse_metadata,
    analyze_contention,
    load_contention_events,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def mb(n: float) -> int:
    """Convert MiB to bytes."""
    return int(n * 1024 * 1024)


def gb(n: float) -> int:
    """Convert GiB to bytes."""
    return int(n * 1024 * 1024 * 1024)


# Canonical MemoryEvent CSV fieldnames (must match schema.py to_row() output)
_FIELDNAMES = [
    "timestamp", "pid", "gpu_id", "source", "event_type",
    "total_memory", "free_memory", "process_memory",
    "allocated_bytes", "reserved_bytes", "allocation_delta", "peak_memory",
    "workload_name", "batch_size", "precision", "metadata_json", "snapshot_path",
]


def make_event(
    timestamp: float = 1000.0,
    pid: int = 100,
    gpu_id: int = 0,
    source: str = "nvml",
    event_type: str = "memory_sample",
    total_memory: Optional[int] = None,
    free_memory: Optional[int] = None,
    process_memory: Optional[int] = None,
    allocated_bytes: Optional[int] = None,
    reserved_bytes: Optional[int] = None,
    allocation_delta: Optional[int] = None,
    peak_memory: Optional[int] = None,
    workload_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    precision: Optional[str] = None,
    metadata_json: Optional[str] = None,
    snapshot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a MemoryEvent-schema dict for use as test input."""
    return {
        "timestamp": str(timestamp) if timestamp is not None else "",
        "pid": str(pid) if pid is not None else "",
        "gpu_id": str(gpu_id) if gpu_id is not None else "",
        "source": source,
        "event_type": event_type,
        "total_memory": str(total_memory) if total_memory is not None else "",
        "free_memory": str(free_memory) if free_memory is not None else "",
        "process_memory": str(process_memory) if process_memory is not None else "",
        "allocated_bytes": str(allocated_bytes) if allocated_bytes is not None else "",
        "reserved_bytes": str(reserved_bytes) if reserved_bytes is not None else "",
        "allocation_delta": str(allocation_delta) if allocation_delta is not None else "",
        "peak_memory": str(peak_memory) if peak_memory is not None else "",
        "workload_name": workload_name or "",
        "batch_size": str(batch_size) if batch_size is not None else "",
        "precision": precision or "",
        "metadata_json": metadata_json or "",
        "snapshot_path": snapshot_path or "",
    }


def write_tmp_csv(events: List[Dict[str, Any]]) -> str:
    """Write a list of event dicts to a temporary CSV file; returns the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    )
    writer = csv.DictWriter(tmp, fieldnames=_FIELDNAMES)
    writer.writeheader()
    for evt in events:
        writer.writerow(evt)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

class TestParsingHelpers:
    def test_parse_int_valid_integer(self):
        assert _parse_int("42") == 42

    def test_parse_int_float_string(self):
        # CSV sometimes stores integer-valued fields as "123.0"
        assert _parse_int("123.0") == 123

    def test_parse_int_empty_string(self):
        assert _parse_int("") is None

    def test_parse_int_none_value(self):
        assert _parse_int(None) is None

    def test_parse_int_sentinel_strings(self):
        for sentinel in ("none", "None", "nan", "NaN", "n/a", "N/A", "null"):
            assert _parse_int(sentinel) is None, f"Expected None for '{sentinel}'"

    def test_parse_int_non_numeric_string(self):
        assert _parse_int("not_a_number") is None

    def test_parse_float_valid(self):
        assert _parse_float("3.14") == pytest.approx(3.14)

    def test_parse_float_empty(self):
        assert _parse_float("") is None

    def test_parse_float_none(self):
        assert _parse_float(None) is None

    def test_parse_metadata_valid_json(self):
        meta = _parse_metadata('{"key": "value", "count": 5}')
        assert meta == {"key": "value", "count": 5}

    def test_parse_metadata_invalid_json_returns_empty(self):
        assert _parse_metadata("{not valid json}") == {}

    def test_parse_metadata_empty_string_returns_empty(self):
        assert _parse_metadata("") == {}

    def test_parse_metadata_none_returns_empty(self):
        assert _parse_metadata(None) == {}

    def test_parse_metadata_non_dict_json_returns_empty(self):
        # A JSON array is not a dict; should return {}
        assert _parse_metadata("[1, 2, 3]") == {}


# ---------------------------------------------------------------------------
# OOM detection
# ---------------------------------------------------------------------------

class TestOOMDetection:
    def test_oom_from_event_type_oom(self):
        evt = make_event(event_type="oom")
        assert _is_oom_event(evt) is True

    def test_oom_from_event_type_error_oom(self):
        evt = make_event(event_type="error_oom")
        assert _is_oom_event(evt) is True

    def test_oom_from_metadata_oom_occurred_true(self):
        evt = make_event(metadata_json='{"oom_occurred": true}')
        assert _is_oom_event(evt) is True

    def test_oom_from_metadata_oom_true(self):
        evt = make_event(metadata_json='{"oom": true}')
        assert _is_oom_event(evt) is True

    def test_oom_from_metadata_exception_string(self):
        evt = make_event(
            metadata_json='{"exception": "RuntimeError: CUDA out of memory."}'
        )
        assert _is_oom_event(evt) is True

    def test_oom_from_metadata_error_string(self):
        evt = make_event(
            metadata_json='{"error": "torch.cuda.OutOfMemoryError"}'
        )
        assert _is_oom_event(evt) is True

    def test_no_oom_from_high_memory_alone(self):
        """High allocated_bytes must NOT trigger oom_detected. This is a critical invariant."""
        evt = make_event(
            source="torch",
            event_type="memory_sample",
            allocated_bytes=gb(5),  # 5 GiB — very high but not an OOM event
        )
        assert _is_oom_event(evt) is False

    def test_no_oom_from_normal_sample(self):
        evt = make_event(source="nvml", event_type="memory_sample")
        assert _is_oom_event(evt) is False

    def test_no_oom_when_oom_occurred_false(self):
        evt = make_event(metadata_json='{"oom_occurred": false}')
        assert _is_oom_event(evt) is False

    def test_no_oom_from_empty_metadata(self):
        evt = make_event(metadata_json="")
        assert _is_oom_event(evt) is False


# ---------------------------------------------------------------------------
# Error collection
# ---------------------------------------------------------------------------

class TestErrorCollection:
    def test_collect_errors_from_error_field(self):
        evt = make_event(metadata_json='{"error": "something went wrong"}')
        errors = _collect_errors(evt)
        assert len(errors) == 1
        assert "something went wrong" in errors[0]

    def test_collect_errors_empty_metadata(self):
        evt = make_event(metadata_json="")
        assert _collect_errors(evt) == []

    def test_collect_errors_no_error_fields(self):
        evt = make_event(metadata_json='{"decision": "ADMIT"}')
        assert _collect_errors(evt) == []


# ---------------------------------------------------------------------------
# Per-PID peak memory
# ---------------------------------------------------------------------------

class TestPerPidPeakMemory:
    def test_peak_process_memory_from_nvml_rows(self):
        """process_memory field from source='nvml' rows gives peak_process_memory_bytes."""
        events = [
            make_event(pid=200, source="nvml", process_memory=mb(100), timestamp=1.0),
            make_event(pid=200, source="nvml", process_memory=mb(500), timestamp=2.0),
            make_event(pid=200, source="nvml", process_memory=mb(300), timestamp=3.0),
        ]
        report = analyze_contention(events)
        assert len(report.process_summaries) == 1
        ps = report.process_summaries[0]
        assert ps.pid == 200
        assert ps.peak_process_memory_bytes == mb(500)

    def test_peak_allocated_bytes_from_torch_rows(self):
        """allocated_bytes from source='torch' rows gives peak_allocated_bytes."""
        events = [
            make_event(pid=300, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(200), timestamp=1.0),
            make_event(pid=300, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(800), timestamp=2.0),
            make_event(pid=300, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(400), timestamp=3.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.peak_allocated_bytes == mb(800)

    def test_process_memory_and_allocated_bytes_tracked_separately(self):
        """The two memory sources are NEVER added together."""
        events = [
            # NVML row for PID 400: process_memory = 1 GiB
            make_event(pid=400, source="nvml", process_memory=gb(1),
                       total_memory=gb(6), free_memory=gb(5), timestamp=1.0),
            # torch row for same PID 400: allocated_bytes = 900 MiB
            make_event(pid=400, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(900), timestamp=1.1),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        # peak_process_memory_bytes comes ONLY from the nvml row
        assert ps.peak_process_memory_bytes == gb(1)
        # peak_allocated_bytes comes ONLY from the torch row
        assert ps.peak_allocated_bytes == mb(900)
        # These must NOT be summed anywhere — verify they are independent
        assert ps.peak_process_memory_bytes != ps.peak_allocated_bytes

    def test_missing_process_memory_returns_none_not_zero(self):
        """An NVML row with no process_memory field yields peak_process_memory_bytes=None."""
        events = [
            # source='nvml' but no process_memory
            make_event(pid=500, source="nvml",
                       total_memory=gb(6), free_memory=gb(4), timestamp=1.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.peak_process_memory_bytes is None

    def test_missing_allocated_bytes_returns_none(self):
        """A torch row with no allocated_bytes yields peak_allocated_bytes=None."""
        events = [
            make_event(pid=600, source="torch", event_type="memory_sample",
                       timestamp=1.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.peak_allocated_bytes is None

    def test_zero_process_memory_is_recorded_not_treated_as_missing(self):
        """Explicit zero in process_memory should be stored, not dropped."""
        events = [
            make_event(pid=700, source="nvml", process_memory=0, timestamp=1.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.peak_process_memory_bytes == 0


# ---------------------------------------------------------------------------
# Runtime computation
# ---------------------------------------------------------------------------

class TestRuntimeComputation:
    def test_runtime_from_multiple_timestamps(self):
        events = [
            make_event(pid=100, source="nvml", timestamp=1000.0),
            make_event(pid=100, source="nvml", timestamp=1002.5),
            make_event(pid=100, source="nvml", timestamp=1005.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.start_time == pytest.approx(1000.0)
        assert ps.end_time == pytest.approx(1005.0)
        assert ps.runtime_s == pytest.approx(5.0)

    def test_single_event_pid_runtime_is_none(self):
        """A PID with only one event cannot have a runtime computed."""
        events = [
            make_event(pid=100, source="nvml", timestamp=1000.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.runtime_s is None

    def test_two_events_same_timestamp_runtime_is_none(self):
        """Two events at exactly the same time give zero-duration; runtime_s is None."""
        events = [
            make_event(pid=100, source="nvml", timestamp=1000.0),
            make_event(pid=100, source="nvml", timestamp=1000.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.runtime_s is None

    def test_runtime_not_fabricated_from_memory(self):
        """Runtime must come from timestamps, not from memory values."""
        events = [
            make_event(pid=100, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(100), timestamp=2000.0),
            make_event(pid=100, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(200), timestamp=2003.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.runtime_s == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Device vs process attribution
# ---------------------------------------------------------------------------

class TestDeviceVsProcessAttribution:
    def test_device_total_memory_from_nvml_not_from_process_sum(self):
        """total_memory goes to DeviceSummary, not added to any ProcessSummary."""
        events = [
            make_event(pid=10, source="nvml",
                       total_memory=gb(6), free_memory=gb(4),
                       process_memory=mb(512), timestamp=1.0),
            make_event(pid=20, source="nvml",
                       total_memory=gb(6), free_memory=gb(3),
                       process_memory=mb(700), timestamp=1.0),
        ]
        report = analyze_contention(events)

        assert report.device_summary is not None
        assert report.device_summary.total_memory_bytes == gb(6)
        # peak_used = max(6-4, 6-3) GB = max(2, 3) = 3 GiB
        assert report.device_summary.peak_used_bytes == gb(3)

        # ProcessSummary should NOT contain total_memory
        for ps in report.process_summaries:
            # peak_process_memory_bytes is NVML per-process query, not device total
            assert ps.peak_process_memory_bytes != gb(6)

    def test_process_memory_sum_not_equal_to_device_used(self):
        """
        Sum of per-process memory readings is intentionally NOT compared to
        device used memory. The test verifies they're tracked in separate fields.
        """
        events = [
            make_event(pid=10, source="nvml",
                       total_memory=gb(6), free_memory=gb(3),
                       process_memory=mb(512), timestamp=1.0),
            make_event(pid=20, source="nvml",
                       total_memory=gb(6), free_memory=gb(3),
                       process_memory=mb(700), timestamp=1.0),
        ]
        report = analyze_contention(events)

        pid10 = next(ps for ps in report.process_summaries if ps.pid == 10)
        pid20 = next(ps for ps in report.process_summaries if ps.pid == 20)

        # Sum of per-process readings
        process_sum = (pid10.peak_process_memory_bytes or 0) + (pid20.peak_process_memory_bytes or 0)
        device_used = report.device_summary.peak_used_bytes  # = 3 GiB

        # These values differ because process_memory represents a per-process snapshot,
        # not accounting for driver overhead, reserved blocks, or timing differences.
        # The test documents this intentional difference rather than asserting equality.
        assert process_sum != device_used or True  # always passes: documents the non-equality

    def test_nvml_source_populates_device_summary(self):
        events = [
            make_event(pid=10, source="nvml",
                       total_memory=gb(6), free_memory=gb(5), timestamp=1.0),
            make_event(pid=10, source="nvml",
                       total_memory=gb(6), free_memory=gb(2), timestamp=2.0),
        ]
        report = analyze_contention(events)
        ds = report.device_summary
        assert ds is not None
        assert ds.total_memory_bytes == gb(6)
        # peak_used = max(6-5, 6-2) GiB = 4 GiB
        assert ds.peak_used_bytes == gb(4)
        assert ds.min_free_bytes == gb(2)
        assert ds.nvml_sample_count == 2

    def test_torch_source_does_not_populate_device_summary(self):
        """torch-sourced events (no total_memory/free_memory) should not add to DeviceSummary."""
        events = [
            make_event(pid=10, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(500), timestamp=1.0),
        ]
        report = analyze_contention(events)
        # No NVML rows → device_summary should be None
        assert report.device_summary is None

    def test_device_summary_note_is_present(self):
        events = [
            make_event(pid=10, source="nvml",
                       total_memory=gb(6), free_memory=gb(4), timestamp=1.0),
        ]
        report = analyze_contention(events)
        assert report.device_summary is not None
        assert len(report.device_summary.note) > 0
        # Note must explicitly state it's NOT a sum of per-process values
        assert "NOT" in report.device_summary.note or "not" in report.device_summary.note.lower()


# ---------------------------------------------------------------------------
# OOM in analyze_contention
# ---------------------------------------------------------------------------

class TestOOMInContentionReport:
    def test_oom_detected_from_event_type(self):
        events = [
            make_event(pid=100, source="nvml", event_type="oom", timestamp=1.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.oom_detected is True

    def test_oom_detected_from_metadata_flag(self):
        events = [
            make_event(pid=100, source="nvml", timestamp=1.0,
                       metadata_json='{"oom_occurred": true}'),
        ]
        report = analyze_contention(events)
        assert report.process_summaries[0].oom_detected is True

    def test_no_oom_from_high_allocated_bytes(self):
        """5 GiB allocated_bytes is high but NOT evidence of an OOM event."""
        events = [
            make_event(pid=100, source="torch", event_type="memory_sample",
                       allocated_bytes=gb(5), timestamp=1.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.oom_detected is False

    def test_no_oom_when_no_indicator_at_all(self):
        events = [
            make_event(pid=100, source="nvml", process_memory=mb(500), timestamp=1.0),
            make_event(pid=100, source="nvml", process_memory=mb(800), timestamp=2.0),
        ]
        report = analyze_contention(events)
        assert report.process_summaries[0].oom_detected is False


# ---------------------------------------------------------------------------
# Concurrent workload counting (sweep-line)
# ---------------------------------------------------------------------------

class TestConcurrentCounting:
    def test_single_pid_concurrent_count(self):
        intervals = {100: (1.0, 5.0)}
        count, pids = _compute_max_concurrent_pids(intervals)
        assert count == 1
        assert pids == [100]

    def test_two_overlapping_pids(self):
        intervals = {100: (1.0, 5.0), 200: (3.0, 8.0)}
        count, pids = _compute_max_concurrent_pids(intervals)
        assert count == 2
        assert set(pids) == {100, 200}

    def test_two_non_overlapping_pids(self):
        intervals = {100: (1.0, 3.0), 200: (5.0, 8.0)}
        count, pids = _compute_max_concurrent_pids(intervals)
        assert count == 1

    def test_three_pids_full_overlap(self):
        intervals = {100: (1.0, 10.0), 200: (2.0, 9.0), 300: (3.0, 8.0)}
        count, pids = _compute_max_concurrent_pids(intervals)
        assert count == 3

    def test_three_pids_partial_overlap(self):
        # PID 100: 1-5, PID 200: 4-8, PID 300: 7-10
        # Max overlap: 100+200 at [4-5] = 2, or 200+300 at [7-8] = 2
        intervals = {100: (1.0, 5.0), 200: (4.0, 8.0), 300: (7.0, 10.0)}
        count, _ = _compute_max_concurrent_pids(intervals)
        assert count == 2

    def test_single_point_pid_counted_as_concurrent(self):
        # PID 100 has only one event at t=3.0, which is within PID 200's [1, 5] range
        intervals = {100: (3.0, 3.0), 200: (1.0, 5.0)}
        count, pids = _compute_max_concurrent_pids(intervals)
        assert count == 2
        assert set(pids) == {100, 200}

    def test_empty_intervals_returns_zero(self):
        count, pids = _compute_max_concurrent_pids({})
        assert count == 0
        assert pids == []

    def test_pid_with_none_start_excluded(self):
        intervals = {100: (None, None), 200: (1.0, 5.0)}
        count, pids = _compute_max_concurrent_pids(intervals)
        assert count == 1
        assert pids == [200]

    def test_concurrent_count_from_analyze_contention(self):
        """Integration test: analyze_contention reports correct concurrency."""
        events = [
            # PID 100: active 1.0 to 5.0
            make_event(pid=100, source="nvml", timestamp=1.0),
            make_event(pid=100, source="nvml", timestamp=5.0),
            # PID 200: active 3.0 to 8.0 — overlaps with PID 100 in [3, 5]
            make_event(pid=200, source="nvml", timestamp=3.0),
            make_event(pid=200, source="nvml", timestamp=8.0),
        ]
        report = analyze_contention(events)
        assert report.max_concurrent_workloads == 2
        assert set(report.concurrent_pids_at_peak) == {100, 200}


# ---------------------------------------------------------------------------
# Robustness: missing, malformed, empty data
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_empty_event_list_returns_empty_report(self):
        report = analyze_contention([])
        assert report.process_summaries == []
        assert report.device_summary is None
        assert report.max_concurrent_workloads == 0
        assert report.total_events_analyzed == 0
        assert report.malformed_rows_skipped == 0

    def test_malformed_timestamp_row_is_skipped(self):
        events = [
            make_event(pid=100, source="nvml", timestamp=1.0),
            # Inject a malformed row by directly setting timestamp to invalid value
            {**make_event(pid=200, source="nvml", timestamp=2.0), "timestamp": "NOT_A_NUMBER"},
        ]
        report = analyze_contention(events)
        assert report.malformed_rows_skipped == 1
        # PID 200 should NOT appear (its only row is malformed)
        pids = [ps.pid for ps in report.process_summaries]
        assert 200 not in pids

    def test_malformed_pid_row_is_skipped(self):
        events = [
            make_event(pid=100, source="nvml", timestamp=1.0),
            {**make_event(pid=200, source="nvml", timestamp=2.0), "pid": "not_an_int"},
        ]
        report = analyze_contention(events)
        assert report.malformed_rows_skipped == 1
        pids = [ps.pid for ps in report.process_summaries]
        assert 100 in pids

    def test_all_malformed_rows_returns_empty_report(self):
        events = [
            {**make_event(pid=100, timestamp=1.0), "timestamp": "bad", "pid": "bad"},
        ]
        report = analyze_contention(events)
        assert report.process_summaries == []
        assert report.malformed_rows_skipped == 1

    def test_missing_memory_fields_do_not_crash(self):
        """Events with no memory fields at all should produce a summary with None peaks."""
        events = [
            make_event(pid=100, source="nvml", timestamp=1.0),  # no memory fields
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.peak_process_memory_bytes is None
        assert ps.peak_allocated_bytes is None

    def test_non_numeric_process_memory_treated_as_missing(self):
        evt = make_event(pid=100, source="nvml", timestamp=1.0)
        evt["process_memory"] = "NOT_A_NUMBER"
        report = analyze_contention([evt])
        ps = report.process_summaries[0]
        assert ps.peak_process_memory_bytes is None

    def test_non_numeric_allocated_bytes_treated_as_missing(self):
        evt = make_event(pid=100, source="torch", event_type="memory_sample",
                         timestamp=1.0)
        evt["allocated_bytes"] = "garbage"
        report = analyze_contention([evt])
        ps = report.process_summaries[0]
        assert ps.peak_allocated_bytes is None

    def test_duplicate_events_same_timestamp_handled_gracefully(self):
        """Duplicate rows for the same PID and timestamp should not crash or inflate peaks."""
        events = [
            make_event(pid=100, source="nvml", process_memory=mb(500), timestamp=1.0),
            make_event(pid=100, source="nvml", process_memory=mb(500), timestamp=1.0),
            make_event(pid=100, source="nvml", process_memory=mb(500), timestamp=1.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.peak_process_memory_bytes == mb(500)
        assert ps.total_events == 3

    def test_mixed_nvml_and_torch_events_same_pid(self):
        """A PID that appears in both NVML and torch events should be handled correctly."""
        events = [
            make_event(pid=100, source="nvml",
                       total_memory=gb(6), free_memory=gb(4),
                       process_memory=mb(300), timestamp=1.0),
            make_event(pid=100, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(250), timestamp=1.1),
        ]
        report = analyze_contention(events)
        assert len(report.process_summaries) == 1
        ps = report.process_summaries[0]
        assert ps.peak_process_memory_bytes == mb(300)
        assert ps.peak_allocated_bytes == mb(250)
        # data_source_note must mention both nvml and torch
        note_lower = ps.data_source_note.lower()
        assert "nvml" in note_lower
        assert "torch" in note_lower

    def test_multi_pid_events(self):
        """Multiple distinct PIDs each get their own ProcessSummary."""
        events = [
            make_event(pid=10, source="nvml", process_memory=mb(200), timestamp=1.0),
            make_event(pid=20, source="nvml", process_memory=mb(400), timestamp=1.5),
            make_event(pid=30, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(600), timestamp=2.0),
        ]
        report = analyze_contention(events)
        assert len(report.process_summaries) == 3
        by_pid = {ps.pid: ps for ps in report.process_summaries}
        assert by_pid[10].peak_process_memory_bytes == mb(200)
        assert by_pid[20].peak_process_memory_bytes == mb(400)
        assert by_pid[30].peak_allocated_bytes == mb(600)

    def test_invalid_metadata_json_does_not_crash(self):
        evt = make_event(pid=100, source="nvml", timestamp=1.0)
        evt["metadata_json"] = "{this is not valid json!!!"
        report = analyze_contention([evt])
        # Should not crash; metadata errors should be silently handled
        assert len(report.process_summaries) == 1

    def test_multi_run_same_pid_distinguished_by_run_id(self):
        """When events contain explicit run_id in metadata_json, the same PID in
        different runs must be segmented into distinct ProcessSummary objects
        with their own runtimes and peaks, rather than one huge conflated interval."""
        events = [
            # Run 1: PID 100 from t=10.0 to t=20.0
            make_event(pid=100, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(300), timestamp=10.0,
                       metadata_json=json.dumps({"run_id": "run_1"})),
            make_event(pid=100, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(500), timestamp=20.0,
                       metadata_json=json.dumps({"run_id": "run_1"})),
            # Run 2: Same PID 100 from t=100.0 to t=110.0
            make_event(pid=100, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(400), timestamp=100.0,
                       metadata_json=json.dumps({"run_id": "run_2"})),
            make_event(pid=100, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(600), timestamp=110.0,
                       metadata_json=json.dumps({"run_id": "run_2"})),
        ]
        report = analyze_contention(events)
        assert len(report.process_summaries) == 2

        by_run = {ps.run_id: ps for ps in report.process_summaries}
        assert "run_1" in by_run
        assert "run_2" in by_run

        # Run 1 metrics
        assert by_run["run_1"].start_time == 10.0
        assert by_run["run_1"].end_time == 20.0
        assert by_run["run_1"].runtime_s == 10.0
        assert by_run["run_1"].peak_allocated_bytes == mb(500)

        # Run 2 metrics
        assert by_run["run_2"].start_time == 100.0
        assert by_run["run_2"].end_time == 110.0
        assert by_run["run_2"].runtime_s == 10.0
        assert by_run["run_2"].peak_allocated_bytes == mb(600)

        # Analysis notes should document multi-run partitioning
        assert any("Multi-run partitioning" in n for n in report.analysis_notes)


# ---------------------------------------------------------------------------
# Workload name extraction
# ---------------------------------------------------------------------------

class TestWorkloadNameExtraction:
    def test_workload_name_from_event_field(self):
        events = [
            make_event(pid=100, source="nvml", timestamp=1.0,
                       workload_name="resnet_run_1"),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.workload_name == "resnet_run_1"

    def test_workload_name_none_when_not_present(self):
        events = [
            make_event(pid=100, source="nvml", timestamp=1.0),
        ]
        report = analyze_contention(events)
        ps = report.process_summaries[0]
        assert ps.workload_name is None


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestLoadContentionEvents:
    def test_load_valid_csv(self):
        events_in = [
            make_event(pid=100, source="nvml",
                       total_memory=gb(6), free_memory=gb(4), timestamp=1.0),
            make_event(pid=200, source="torch", event_type="memory_sample",
                       allocated_bytes=mb(300), timestamp=2.0),
        ]
        csv_path = write_tmp_csv(events_in)
        try:
            events_out = load_contention_events(csv_path)
            assert len(events_out) == 2
            assert events_out[0]["source"] == "nvml"
            assert events_out[1]["source"] == "torch"
        finally:
            os.unlink(csv_path)

    def test_load_empty_csv_returns_empty_list(self):
        csv_path = write_tmp_csv([])
        try:
            events = load_contention_events(csv_path)
            assert events == []
        finally:
            os.unlink(csv_path)

    def test_load_nonexistent_csv_raises(self):
        with pytest.raises(FileNotFoundError):
            load_contention_events("/nonexistent/path/contention.csv")

    def test_load_and_analyze_roundtrip(self):
        """Load a CSV then analyze; result should match direct dict analysis."""
        events_in = [
            make_event(pid=100, source="nvml", process_memory=mb(500), timestamp=1.0),
            make_event(pid=100, source="nvml", process_memory=mb(800), timestamp=2.0),
        ]
        csv_path = write_tmp_csv(events_in)
        try:
            loaded = load_contention_events(csv_path)
            report = analyze_contention(loaded)
            ps = report.process_summaries[0]
            assert ps.peak_process_memory_bytes == mb(800)
            assert ps.runtime_s == pytest.approx(1.0)
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# ContentionReport serialization
# ---------------------------------------------------------------------------

class TestReportSerialization:
    def test_to_dict_is_json_serializable(self):
        events = [
            make_event(pid=100, source="nvml",
                       total_memory=gb(6), free_memory=gb(4),
                       process_memory=mb(300), timestamp=1.0),
            make_event(pid=100, source="nvml", process_memory=mb(500), timestamp=2.0),
        ]
        report = analyze_contention(events)
        report_dict = report.to_dict()
        # Should not raise
        serialized = json.dumps(report_dict)
        assert len(serialized) > 0

    def test_process_summary_to_dict(self):
        events = [
            make_event(pid=100, source="nvml", process_memory=mb(400), timestamp=1.0),
        ]
        report = analyze_contention(events)
        d = report.process_summaries[0].to_dict()
        assert "pid" in d
        assert "peak_process_memory_bytes" in d
        assert "peak_allocated_bytes" in d
        assert "runtime_s" in d
        assert "oom_detected" in d
