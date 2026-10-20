"""
File: profiler/contention.py

1. What this file does:
   Analyzes recorded GPU memory event CSV files to produce per-process contention
   summaries, device-level statistics, and concurrent-workload counts. All
   analysis is performed on pre-recorded data; this module makes no live GPU,
   NVML, or CUDA calls.

2. Overall design:
   A standalone analysis library and command-line tool following the same pattern
   as profiler/memory_pressure.py. It reads canonical MemoryEvent CSV rows
   (produced by collector/nvml_monitor.py, collector/torch_hooks.py, or any other
   source conforming to schema.py) and returns structured dataclasses.

   Key design constraints enforced explicitly:
   - DEVICE vs PROCESS SEPARATION:
     total_memory and free_memory (source="nvml", device-wide NVML readings)
     are NEVER attributed to an individual process. Device "used" memory is
     computed as max(total_memory - free_memory) across NVML rows only.
     Per-process NVML memory (process_memory field) and in-process PyTorch
     allocator memory (allocated_bytes field) are tracked separately and never
     summed to represent device total, because they measure different things
     from different vantage points and do not account for each other's overhead.
   - OOM DETECTION REQUIRES EXPLICIT EVIDENCE:
     A process is flagged as oom_detected=True ONLY when the recorded event
     data contains an explicit OOM marker (event_type in {"oom","error_oom"}
     or metadata_json contains "oom_occurred":true or an OOM exception message).
     High memory usage alone is never interpreted as evidence of an OOM.
   - NO FABRICATED MEASUREMENTS:
     If a field is absent or unparseable, the corresponding summary value is
     None. Zeros are only recorded when the source data explicitly contains 0.
   - ROBUSTNESS:
     Malformed rows (non-numeric timestamps, non-numeric memory fields, invalid
     JSON metadata) are skipped and counted. Empty files return empty reports.

3. Elements in the file:
   - ProcessSummary: Per-PID analysis result dataclass.
   - DeviceSummary: Device-level NVML statistics dataclass.
   - ContentionReport: Top-level combined result dataclass.
   - _parse_int, _parse_float, _parse_metadata: Robust field parsers.
   - _is_oom_event: Explicit-only OOM detection.
   - _collect_errors: Metadata error extractor.
   - _compute_max_concurrent_pids: Sweep-line concurrency counter.
   - load_contention_events: CSV loader returning list of dicts.
   - analyze_contention: Main analysis entry point.
   - print_contention_report: Console report printer.
   - CLI entry point (analysis mode: reads a recorded CSV).

4. Final output of the file:
   A ContentionReport describing per-PID memory, timing, concurrency,
   and device statistics derived from recorded observations.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Project root — same pattern used by all other modules.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# schema.py is imported for the to_row() method reference only when needed.
# The import is wrapped so this module can be imported on CPU-only machines
# even if schema.py has environment-specific issues.
try:
    from schema import MemoryEvent as _MemoryEvent  # noqa: F401
except ImportError:
    _MemoryEvent = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# OOM detection constants — never change these to infer from high memory.
# ---------------------------------------------------------------------------

_OOM_EVENT_TYPES: frozenset = frozenset({"oom", "error_oom", "cuda_oom"})
_OOM_METADATA_TRUE_FLAGS: Tuple[str, ...] = (
    "oom_occurred",
    "oom",
    "out_of_memory",
)
_OOM_EXCEPTION_KEYWORDS: Tuple[str, ...] = (
    "outofmemory",
    "cuda out of memory",
    "out of memory",
    "runtimeerror: cuda",
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProcessSummary:
    """
    1. The use of class:
       Stores the per-PID analysis result derived from a contention event stream.
    2. What the class contains:
       PID, workload name, peak memory readings from two separate measurement
       sources, timing statistics, OOM indicators, and any error messages found
       in the recorded metadata.
    3. Technologies used and why:
       Python dataclass for lightweight structured storage with dict serialisation
       via asdict().
    4. Key design — measurement separation:
       peak_process_memory_bytes: NVML per-process query (process_memory field,
           source="nvml"). Represents the GPU driver's reported view of this PID's
           memory footprint at a single point in time.
       peak_allocated_bytes: PyTorch allocator (allocated_bytes field,
           source="torch"). Represents the in-process allocator's current
           allocation at a single point in time.
       These two metrics measure memory from different vantage points and are
       NEVER added together or treated as equivalent. Neither alone represents
       the total GPU memory cost of the process.
       oom_detected is True ONLY when the recorded data contains an explicit
       OOM marker; it is never inferred from high memory values.
    """
    pid: int
    workload_name: Optional[str]
    # From source="nvml" rows, process_memory field (NVML per-process query)
    peak_process_memory_bytes: Optional[int]
    # From source="torch" rows, allocated_bytes field (PyTorch allocator)
    peak_allocated_bytes: Optional[int]
    total_events: int
    start_time: Optional[float]    # earliest valid timestamp for this PID
    end_time: Optional[float]      # latest valid timestamp for this PID
    runtime_s: Optional[float]     # end_time - start_time; None if ≤1 distinct ts
    oom_detected: bool             # True ONLY if explicit OOM indicator in data
    errors: List[str]              # error/exception strings from metadata_json
    data_source_note: str          # explains measurement sources and limitations
    run_id: Optional[str] = None   # optional run_id if provided in event metadata

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Converts ProcessSummary to a JSON-serializable dict.
        2. How does the specific function work:
           Uses dataclasses.asdict() for deep conversion.
        3. Concepts used in the function:
           Dataclass serialization.
        4. Places where this function is imported and used:
           Used by ContentionReport.to_dict() and CLI JSON output.
        5. Final output:
           Flat dict suitable for json.dumps().
        """
        return asdict(self)


@dataclass
class DeviceSummary:
    """
    1. The use of class:
       Stores device-level NVML statistics from source="nvml" rows only.
    2. What the class contains:
       GPU capacity, peak usage derived from NVML device-wide readings, and
       an explicit note clarifying attribution boundaries.
    3. Technologies used and why:
       Python dataclass. No live GPU calls; computed entirely from recorded data.
    4. Key design — attribution boundary:
       total_memory is the device capacity reported by NVML (constant per device).
       peak_used_bytes = max(total_memory - free_memory) across all NVML rows.
       This is a DEVICE-WIDE figure, NOT the sum of per-process memory readings.
       Per-process memory (process_memory field) is separately tracked in
       ProcessSummary and is never summed here or equated to device usage.
    """
    gpu_id: Optional[int]
    total_memory_bytes: Optional[int]   # NVML-reported device capacity (bytes)
    peak_used_bytes: Optional[int]      # max(total_memory - free_memory) from NVML rows
    min_free_bytes: Optional[int]       # min(free_memory) from NVML rows (most constrained point)
    nvml_sample_count: int              # number of NVML device-wide rows analyzed
    note: str                           # explicit attribution boundary statement

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Converts DeviceSummary to a JSON-serializable dict.
        2. How does the specific function work:
           Uses dataclasses.asdict().
        3. Concepts used in the function:
           Dataclass serialization.
        4. Places where this function is imported and used:
           Used by ContentionReport.to_dict() and CLI JSON output.
        5. Final output:
           Flat dict.
        """
        return asdict(self)


@dataclass
class ContentionReport:
    """
    1. The use of class:
       Top-level analysis result combining all per-PID summaries, device-level
       statistics, concurrency counts, and data-quality notes.
    2. What the class contains:
       List of ProcessSummary, optional DeviceSummary, maximum concurrent
       workload count and the PIDs active at that peak, analysis caveats,
       and bookkeeping counters for data quality.
    3. Technologies used and why:
       Python dataclass. Produced by analyze_contention() and consumed by the
       CLI printer, JSON serializer, and future dashboard integration.
    4. Key design:
       max_concurrent_workloads is computed from overlapping PID time intervals
       using a sweep-line algorithm. A PID's interval is [min_timestamp,
       max_timestamp] across its events. If a PID has only one observed event,
       its interval is a single point (zero duration).
       analysis_notes accumulates all caveat messages encountered during
       analysis (missing fields, single-event PIDs, mixed sources, etc.)
    """
    process_summaries: List[ProcessSummary]
    device_summary: Optional[DeviceSummary]
    max_concurrent_workloads: int        # max overlapping PID intervals
    concurrent_pids_at_peak: List[int]   # PIDs active when max_concurrent reached
    analysis_notes: List[str]            # caveats and data-quality warnings
    total_events_analyzed: int
    malformed_rows_skipped: int

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Converts ContentionReport to a fully JSON-serializable dict.
        2. How does the specific function work:
           Uses dataclasses.asdict() which recursively converts all nested
           dataclasses and lists.
        3. Concepts used in the function:
           Dataclass serialization, recursive dict conversion.
        4. Places where this function is imported and used:
           Used by the CLI to write JSON output and by tests for assertion.
        5. Final output:
           Fully serializable dict.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------

def _parse_int(value: Any) -> Optional[int]:
    """
    1. The use of function:
       Safely parses a CSV field value to int, returning None for absent values.
    2. How does the specific function work:
       Converts value to str, strips whitespace, checks for empty/sentinel
       strings, then attempts float→int conversion to handle values like "123.0".
    3. Concepts used in the function:
       Defensive parsing, missing-data handling.
    4. Places where this function is imported and used:
       Called throughout analyze_contention() for every memory-field extraction.
    5. Final output:
       Integer or None. Never raises.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "nan", "n/a", "null", "na"):
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError, OverflowError):
        return None


def _parse_float(value: Any) -> Optional[float]:
    """
    1. The use of function:
       Safely parses a CSV field value to float, returning None for absent values.
    2. How does the specific function work:
       Converts value to str, strips whitespace, checks for sentinel strings,
       then attempts float conversion.
    3. Concepts used in the function:
       Defensive parsing, missing-data handling.
    4. Places where this function is imported and used:
       Used for timestamp parsing in analyze_contention().
    5. Final output:
       Float or None. Never raises.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "nan", "n/a", "null", "na"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError, OverflowError):
        return None


def _parse_metadata(value: Any) -> Dict[str, Any]:
    """
    1. The use of function:
       Safely parses a metadata_json field into a dict.
    2. How does the specific function work:
       Converts value to str, checks for empty/null sentinels, then calls
       json.loads(). Returns {} on any parse error.
    3. Concepts used in the function:
       JSON parsing, defensive error handling.
    4. Places where this function is imported and used:
       Called by _is_oom_event(), _collect_errors(), and analyze_contention()
       for reading structured metadata from events.
    5. Final output:
       Dict (possibly empty). Never raises.
    """
    if value is None:
        return {}
    s = str(value).strip()
    if not s or s.lower() in ("none", "null"):
        return {}
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _is_oom_event(event: Dict[str, Any]) -> bool:
    """
    1. The use of function:
       Determines whether a single event record contains an explicit OOM indicator.
    2. How does the specific function work:
       Checks three conditions in order:
       1. event_type is in the hard-coded set of OOM event types.
       2. metadata_json contains a boolean OOM flag set to True.
       3. metadata_json contains an error/exception string with OOM keywords.
       Returns False in all other cases, including when memory usage is high.
    3. Concepts used in the function:
       Conservative OOM detection, explicit-evidence requirement.
    4. Places where this function is imported and used:
       Called per-event in analyze_contention() to build oom_detected per PID.
    5. Final output:
       True only if explicit OOM evidence is present; False otherwise.

    IMPORTANT: This function intentionally does NOT check memory level thresholds
    or any derived metrics. High memory usage is NOT evidence of an OOM event.
    """
    event_type = str(event.get("event_type") or "").strip().lower()
    if event_type in _OOM_EVENT_TYPES:
        return True

    meta = _parse_metadata(event.get("metadata_json"))

    # Check boolean OOM flags
    for flag in _OOM_METADATA_TRUE_FLAGS:
        val = meta.get(flag)
        if val is True:
            return True

    # Check error/exception message strings for OOM keywords
    for key in ("error", "exception", "message", "error_message"):
        msg = str(meta.get(key) or "").lower()
        if any(kw in msg for kw in _OOM_EXCEPTION_KEYWORDS):
            return True

    return False


def _collect_errors(event: Dict[str, Any]) -> List[str]:
    """
    1. The use of function:
       Extracts human-readable error/exception strings from a single event's
       metadata_json.
    2. How does the specific function work:
       Parses metadata_json and checks a set of known error-carrying keys.
       Returns non-empty, non-whitespace string values found under those keys.
    3. Concepts used in the function:
       Metadata inspection, error surfacing.
    4. Places where this function is imported and used:
       Called per-event in analyze_contention() to accumulate errors per PID.
    5. Final output:
       List of formatted error strings. Empty list if none found.
    """
    meta = _parse_metadata(event.get("metadata_json"))
    errors: List[str] = []
    for key in ("error", "exception", "message", "error_message", "reject_reason"):
        val = meta.get(key)
        if val and isinstance(val, str) and val.strip():
            errors.append(f"[{key}] {val.strip()}")
    return errors


# ---------------------------------------------------------------------------
# Concurrent workload counting (sweep-line algorithm)
# ---------------------------------------------------------------------------

def _compute_max_concurrent_pids(
    pid_intervals: Dict[int, Tuple[Optional[float], Optional[float]]],
) -> Tuple[int, List[int]]:
    """
    1. The use of function:
       Computes the maximum number of PIDs that are concurrently active,
       based on their observed time intervals [start_time, end_time].
    2. How does the specific function work:
       Uses a sweep-line algorithm:
       - For each PID with a valid start_time, generates a "start" event at
         start_time and an "end" event at end_time (or start_time if end equals
         start, i.e., a single-observation PID).
       - Events are sorted by (time, tiebreak) where starts sort before ends
         at the same time (tiebreak 0 < 1), ensuring PIDs starting and ending
         simultaneously are counted as concurrent at that moment.
       - Walking the sorted events, the active set is updated and the maximum
         size tracked.
    3. Concepts used in the function:
       Interval scheduling, sweep-line algorithm, concurrency measurement.
    4. Places where this function is imported and used:
       Called by analyze_contention() to compute ContentionReport.max_concurrent.
    5. Final output:
       Tuple (max_count: int, peak_pids: List[int]) where max_count is the
       maximum number of simultaneously active PIDs and peak_pids is the list
       of PIDs that were active when that maximum was first reached.

    Edge cases:
    - PIDs with no valid start_time are excluded.
    - PIDs with only one distinct timestamp (start == end) are included as a
      zero-duration point and counted as concurrent at their single timestamp.
    - An empty pid_intervals dict returns (0, []).
    """
    if not pid_intervals:
        return 0, []

    # Build sweep events: (time, tiebreak, pid, delta)
    # tiebreak 0 = start (process before ends at same time)
    # tiebreak 1 = end   (process after starts at same time)
    sweep: List[Tuple[float, int, int, int]] = []
    for pid, (start, end) in pid_intervals.items():
        if start is None:
            continue
        # A PID with end == start (or end is None) contributes a point interval.
        effective_end = end if (end is not None and end > start) else start
        sweep.append((start, 0, pid, +1))           # start event
        sweep.append((effective_end, 1, pid, -1))   # end event

    if not sweep:
        return 0, []

    sweep.sort(key=lambda x: (x[0], x[1]))

    active_keys: Set[Any] = set()
    max_count = 0
    peak_pids: List[int] = []

    for _time, tiebreak, item_key, delta in sweep:
        if delta > 0:
            active_keys.add(item_key)
            if len(active_keys) > max_count:
                max_count = len(active_keys)
                raw_pids = {k[0] if isinstance(k, tuple) else k for k in active_keys}
                peak_pids = sorted(raw_pids)  # deterministic sort for tests
        else:
            active_keys.discard(item_key)

    return max_count, peak_pids


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_contention_events(csv_path: str) -> List[Dict[str, Any]]:
    """
    1. The use of function:
       Loads a MemoryEvent CSV file into a list of raw row dicts.
    2. How does the specific function work:
       Opens the file with csv.DictReader and returns all rows as dicts.
       Each dict key corresponds to a MemoryEvent CSV column header.
       No parsing is performed here — analyze_contention() handles parsing.
       An empty file (header only) returns an empty list.
    3. Concepts used in the function:
       CSV reading, raw-dict extraction for downstream analysis.
    4. Places where this function is imported and used:
       Called by the CLI entry point and by experiment scripts.
       Tests may call it on temporary CSV fixtures.
    5. Final output of the function, before and after:
       Before: path to a MemoryEvent CSV file.
       After: list of dicts, one per data row (header excluded).
       Raises FileNotFoundError if the path does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Contention CSV not found: {csv_path}")

    events: List[Dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            events.append(dict(row))
    return events


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_contention(events: List[Dict[str, Any]]) -> ContentionReport:
    """
    1. The use of function:
       Analyzes a list of MemoryEvent-schema dicts and produces a ContentionReport.
    2. How does the specific function work:
       Iterates through events, skipping malformed rows (counted in
       malformed_rows_skipped). Groups valid events by PID. For each PID group,
       builds a ProcessSummary. Separately aggregates device-level NVML rows into
       a DeviceSummary. Computes max concurrent workloads via _compute_max_concurrent_pids.
    3. Concepts used in the function:
       Multi-process resource contention analysis, device vs process attribution,
       conservative OOM detection, defensive data handling.
    4. Places where this function is imported and used:
       Called by the CLI entry point and by tests with mock event lists.
    5. Final output of the function, before and after:
       Before: list of raw event dicts (may contain malformed entries).
       After: ContentionReport with all summaries, or empty/minimal report for
       empty or fully-malformed input.

    DEVICE vs PROCESS ATTRIBUTION INVARIANT (enforced here):
    - total_memory and free_memory → DeviceSummary.peak_used_bytes ONLY.
    - process_memory → ProcessSummary.peak_process_memory_bytes ONLY.
    - allocated_bytes → ProcessSummary.peak_allocated_bytes ONLY.
    These three quantities are NEVER combined to represent the same concept.
    """
    # --- Per-PID state accumulators ---
    # Key: (pid, run_id) -> list of valid events
    pid_events: Dict[Tuple[int, Optional[str]], List[Dict[str, Any]]] = {}
    has_explicit_run_ids = False

    # --- Device-level (NVML) accumulators ---
    device_total_memory: Optional[int] = None
    device_peak_used: Optional[int] = None   # max(total - free)
    device_min_free: Optional[int] = None    # min(free)
    nvml_sample_count = 0
    device_gpu_id: Optional[int] = None

    # --- Counters and notes ---
    malformed_rows_skipped = 0
    analysis_notes: List[str] = []

    # --- Pass 1: parse and classify all events ---
    for raw_row in events:
        # A row is malformed if it has no valid timestamp or no valid PID.
        ts = _parse_float(raw_row.get("timestamp"))
        if ts is None:
            malformed_rows_skipped += 1
            continue

        pid = _parse_int(raw_row.get("pid"))
        if pid is None:
            malformed_rows_skipped += 1
            continue

        source = str(raw_row.get("source") or "").strip().lower()
        gpu_id = _parse_int(raw_row.get("gpu_id"))

        # --- Device-level NVML accumulation ---
        if source == "nvml":
            total_mem = _parse_int(raw_row.get("total_memory"))
            free_mem = _parse_int(raw_row.get("free_memory"))

            if total_mem is not None:
                # total_memory should be constant per device; record any value
                if device_total_memory is None:
                    device_total_memory = total_mem
                if gpu_id is not None and device_gpu_id is None:
                    device_gpu_id = gpu_id

            if total_mem is not None and free_mem is not None:
                used = total_mem - free_mem
                device_peak_used = used if device_peak_used is None else max(device_peak_used, used)
                device_min_free = free_mem if device_min_free is None else min(device_min_free, free_mem)
                nvml_sample_count += 1

        # --- Per-PID event accumulation with optional run_id partitioning ---
        meta = _parse_metadata(raw_row.get("metadata_json"))
        run_id: Optional[str] = None
        if "run_id" in meta and str(meta["run_id"]).strip():
            run_id = str(meta["run_id"]).strip()
            has_explicit_run_ids = True
        elif "run_id" in raw_row and raw_row.get("run_id") and str(raw_row["run_id"]).strip():
            run_id = str(raw_row["run_id"]).strip()
            has_explicit_run_ids = True

        group_key = (pid, run_id)
        if group_key not in pid_events:
            pid_events[group_key] = []
        pid_events[group_key].append(raw_row)

    # --- Pass 2: build per-PID summaries ---
    process_summaries: List[ProcessSummary] = []
    pid_intervals: Dict[Tuple[int, Optional[str]], Tuple[Optional[float], Optional[float]]] = {}

    for (pid, run_id), evt_list in pid_events.items():
        timestamps: List[float] = []
        workload_name: Optional[str] = None
        peak_process_mem: Optional[int] = None   # NVML process_memory
        peak_alloc_bytes: Optional[int] = None   # torch allocated_bytes
        oom_detected = False
        errors: List[str] = []
        sources_seen: Set[str] = set()

        for evt in evt_list:
            ts = _parse_float(evt.get("timestamp"))
            if ts is not None:
                timestamps.append(ts)

            source = str(evt.get("source") or "").strip().lower()
            sources_seen.add(source)

            # Workload name: take the first non-empty value found
            if workload_name is None:
                wn = str(evt.get("workload_name") or "").strip()
                if wn:
                    workload_name = wn

            # Per-process NVML memory (source="nvml", process_memory field)
            if source == "nvml":
                pm = _parse_int(evt.get("process_memory"))
                if pm is not None:
                    peak_process_mem = pm if peak_process_mem is None else max(peak_process_mem, pm)

            # In-process PyTorch allocator memory (source="torch", allocated_bytes field)
            if source == "torch":
                ab = _parse_int(evt.get("allocated_bytes"))
                if ab is not None:
                    peak_alloc_bytes = ab if peak_alloc_bytes is None else max(peak_alloc_bytes, ab)

            # OOM detection — explicit evidence only
            if not oom_detected and _is_oom_event(evt):
                oom_detected = True

            # Error collection
            evt_errors = _collect_errors(evt)
            errors.extend(e for e in evt_errors if e not in errors)

        # Timing
        start_time: Optional[float] = min(timestamps) if timestamps else None
        end_time: Optional[float] = max(timestamps) if timestamps else None
        runtime_s: Optional[float] = None
        if (start_time is not None and end_time is not None
                and end_time > start_time):
            runtime_s = end_time - start_time
        elif start_time is not None and end_time is not None and start_time == end_time:
            # Only one distinct timestamp; cannot compute a meaningful duration
            run_label = f" (run {run_id})" if run_id else ""
            analysis_notes.append(
                f"PID {pid}{run_label}: only one distinct timestamp observed; "
                "runtime_s cannot be computed."
            )

        pid_intervals[(pid, run_id)] = (start_time, end_time)

        # Build the data_source_note explaining where measurements came from
        note_parts: List[str] = []
        if "nvml" in sources_seen:
            note_parts.append(
                "peak_process_memory_bytes from NVML per-process query (process_memory field); "
                "represents a snapshot of the GPU driver's view and may not include "
                "all allocator overhead or inactive cached blocks"
            )
        if "torch" in sources_seen:
            note_parts.append(
                "peak_allocated_bytes from PyTorch allocator statistics (allocated_bytes field); "
                "measures only the instrumented process's live allocations at polling time; "
                "reserved (cached) memory may be larger"
            )
        if not note_parts:
            note_parts.append(
                "No recognized source (nvml/torch) found in events for this PID; "
                "memory peak fields are unavailable"
            )
        data_source_note = ". ".join(note_parts) + "."

        process_summaries.append(ProcessSummary(
            pid=pid,
            workload_name=workload_name,
            peak_process_memory_bytes=peak_process_mem,
            peak_allocated_bytes=peak_alloc_bytes,
            total_events=len(evt_list),
            start_time=start_time,
            end_time=end_time,
            runtime_s=runtime_s,
            oom_detected=oom_detected,
            errors=errors,
            data_source_note=data_source_note,
            run_id=run_id,
        ))

    # --- Concurrent workload count ---
    max_concurrent, concurrent_pids_at_peak = _compute_max_concurrent_pids(pid_intervals)

    # --- Build DeviceSummary ---
    device_summary: Optional[DeviceSummary] = None
    if nvml_sample_count > 0 or device_total_memory is not None:
        device_summary = DeviceSummary(
            gpu_id=device_gpu_id,
            total_memory_bytes=device_total_memory,
            peak_used_bytes=device_peak_used,
            min_free_bytes=device_min_free,
            nvml_sample_count=nvml_sample_count,
            note=(
                "peak_used_bytes = max(total_memory - free_memory) from NVML device-wide rows. "
                "This is NOT the sum of per-process process_memory values and does not equal "
                "the sum of per-process peak memory readings. Each measures a different "
                "aspect of device memory from a different vantage point."
            ),
        )

    # --- Cross-section attribution note ---
    if device_summary is not None and process_summaries:
        analysis_notes.append(
            "Device-wide memory (DeviceSummary) and per-process memory (ProcessSummary) "
            "are measured independently and cannot be directly summed or equated. "
            "See DeviceSummary.note and ProcessSummary.data_source_note for details."
        )

    if has_explicit_run_ids:
        analysis_notes.append(
            "Multi-run partitioning: Input events contain explicit run_id metadata; "
            "per-PID intervals are segmented by run_id to avoid concatenating separate executions."
        )
    elif process_summaries:
        analysis_notes.append(
            "Multi-run limitation: Input events lack run_id metadata. If this file contains "
            "multiple concatenated benchmark runs, observations for recycled PIDs are treated "
            "as a single continuous execution interval."
        )

    if not process_summaries and not device_summary:
        analysis_notes.append("No valid events found; report is empty.")

    total_analyzed = sum(len(v) for v in pid_events.values()) + malformed_rows_skipped

    return ContentionReport(
        process_summaries=process_summaries,
        device_summary=device_summary,
        max_concurrent_workloads=max_concurrent,
        concurrent_pids_at_peak=concurrent_pids_at_peak,
        analysis_notes=analysis_notes,
        total_events_analyzed=total_analyzed,
        malformed_rows_skipped=malformed_rows_skipped,
    )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_contention_report(report: ContentionReport) -> None:
    """
    1. The use of function:
       Prints a human-readable summary of a ContentionReport to stdout.
    2. How does the specific function work:
       Iterates through process summaries and device summary, formatting
       each field in human-readable units (MiB).
    3. Concepts used in the function:
       Report formatting, unit conversion.
    4. Places where this function is imported and used:
       Called by the CLI entry point.
    5. Final output:
       Formatted text to stdout. No return value.
    """
    mib = 1024 * 1024

    print("=" * 70)
    print("GPU Memory Contention Analysis Report")
    print("=" * 70)
    print(f"Total events analyzed : {report.total_events_analyzed}")
    print(f"Malformed rows skipped: {report.malformed_rows_skipped}")
    print(f"Max concurrent PIDs   : {report.max_concurrent_workloads}")
    if report.concurrent_pids_at_peak:
        print(f"PIDs at peak overlap  : {report.concurrent_pids_at_peak}")
    print()

    if report.device_summary:
        ds = report.device_summary
        print("--- Device Summary (NVML device-wide rows) ---")
        print(f"  GPU ID              : {ds.gpu_id}")
        total_mib = f"{ds.total_memory_bytes // mib} MiB" if ds.total_memory_bytes is not None else "N/A"
        peak_mib = f"{ds.peak_used_bytes // mib} MiB" if ds.peak_used_bytes is not None else "N/A"
        free_mib = f"{ds.min_free_bytes // mib} MiB" if ds.min_free_bytes is not None else "N/A"
        print(f"  Device capacity     : {total_mib}")
        print(f"  Peak device used    : {peak_mib}  (NOT sum of per-process readings)")
        print(f"  Min device free     : {free_mib}")
        print(f"  NVML samples        : {ds.nvml_sample_count}")
        print()

    if report.process_summaries:
        print("--- Per-Process Summaries ---")
        for ps in report.process_summaries:
            print(f"  PID {ps.pid} | Workload: {ps.workload_name or 'unknown'}")
            proc_mib = (f"{ps.peak_process_memory_bytes // mib} MiB"
                        if ps.peak_process_memory_bytes is not None else "N/A")
            alloc_mib = (f"{ps.peak_allocated_bytes // mib} MiB"
                         if ps.peak_allocated_bytes is not None else "N/A")
            rt = (f"{ps.runtime_s:.3f} s"
                  if ps.runtime_s is not None else "N/A (single timestamp or no data)")
            print(f"    Peak NVML process mem : {proc_mib}")
            print(f"    Peak PyTorch alloc    : {alloc_mib}")
            print(f"    Events                : {ps.total_events}")
            print(f"    Runtime               : {rt}")
            print(f"    OOM detected          : {ps.oom_detected}"
                  + (" (EXPLICIT indicator in data)" if ps.oom_detected else
                     " (no explicit OOM indicator found)"))
            if ps.errors:
                print(f"    Errors recorded       : {ps.errors}")
            print()

    if report.analysis_notes:
        print("--- Analysis Notes ---")
        for note in report.analysis_notes:
            print(f"  * {note}")
        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a recorded MemoryEvent CSV to produce a contention report. "
            "This tool reads pre-recorded data and does not access GPU hardware."
        )
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to a MemoryEvent CSV file (e.g. data/profiles/contention.csv)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the JSON report (default: stdout only)",
    )
    args = parser.parse_args()

    try:
        events = load_contention_events(args.csv_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    report = analyze_contention(events)
    print_contention_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nJSON report saved to: {out_path}")
