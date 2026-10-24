"""
File: pipeline/__init__.py

1. What this file does:
   Initializes the `pipeline` package and exports the core data processing,
   normalization, validation, and analytical aggregation functions.

2. Overall design:
   Provides convenient top-level package imports for downstream modules such as
   `dashboard/app.py`, test runners, and command-line entry points.

3. Elements in the file:
   - Re-exports from `normalize.py`: `bytes_to_mb`, `bytes_to_gb`, `format_bytes`,
     `normalize_memory_events`, `events_to_dataframe`, `load_and_normalize_csv`.
   - Re-exports from `validate.py`: `ValidationResult`, `validate_memory_event`,
     `validate_event_list`, `validate_csv_file`.
   - Re-exports from `aggregate.py`: `TelemetryAggregate`, `compute_profile_summary`,
     `compute_pid_breakdown`, `resample_timeseries`, `join_device_and_process_streams`.

4. Final output of the file:
   A clean namespace `from pipeline import ...`.
"""

from pipeline.normalize import (
    bytes_to_gb,
    bytes_to_mb,
    events_to_dataframe,
    format_bytes,
    load_and_normalize_csv,
    normalize_memory_events,
)
from pipeline.validate import (
    ValidationResult,
    validate_csv_file,
    validate_event_list,
    validate_memory_event,
)
from pipeline.aggregate import (
    TelemetryAggregate,
    compute_pid_breakdown,
    compute_profile_summary,
    join_device_and_process_streams,
    resample_timeseries,
)

__all__ = [
    "bytes_to_mb",
    "bytes_to_gb",
    "format_bytes",
    "normalize_memory_events",
    "events_to_dataframe",
    "load_and_normalize_csv",
    "ValidationResult",
    "validate_memory_event",
    "validate_event_list",
    "validate_csv_file",
    "TelemetryAggregate",
    "compute_profile_summary",
    "compute_pid_breakdown",
    "resample_timeseries",
    "join_device_and_process_streams",
]
