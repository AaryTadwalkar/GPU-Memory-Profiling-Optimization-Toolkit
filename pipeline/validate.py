"""
File: pipeline/validate.py

1. What this file does:
   Validates MemoryEvent records and CSV files against the unified schema rules and
   domain integrity constraints. Identifies malformed rows, missing required fields,
   invalid data types, negative memory quantities, and logical inconsistencies.

2. Overall design:
   A CPU-safe validation module providing both per-record inspection and batch CSV
   auditing. It returns a structured `ValidationResult` dataclass containing status
   flags, valid/invalid row counts, and detailed error messages without throwing
   unhandled exceptions. Can run in lenient mode (filters out invalid rows) or strict
   mode (flags the batch as invalid upon any error).

3. Elements in the file:
   - `ValidationResult`: Dataclass summarizing validation outcomes and diagnostic errors.
   - `validate_memory_event`: Checks an individual record dictionary.
   - `validate_event_list`: Validates an in-memory list of event dictionaries.
   - `validate_csv_file`: Reads and audits a CSV file from disk.

4. Final output of the file:
   A `ValidationResult` detailing schema compliance, along with sanitized lists of
   valid event records ready for downstream processing.
"""

import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schema import MemoryEvent


REQUIRED_FIELDS = {"timestamp", "pid", "gpu_id", "source", "event_type"}
VALID_SOURCES = {"nvml", "torch", "workload", "scheduler", "mock"}


@dataclass
class ValidationResult:
    """
    1. The use of class:
       Encapsulates the outcome of a schema and integrity audit over a dataset.
    2. What the class contain:
       is_valid (bool), total_rows (int), valid_rows (int), invalid_rows (int),
       and errors (list of strings describing specific violations).
    3. Technologies used and why:
       Standard Python dataclass for clean result reporting and inspection.
    4. Key design of the class:
       Maintains explicit counts and non-crashing error logs for auditability.
    """
    is_valid: bool
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a concise human-readable summary of the validation result."""
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"Validation {status}: {self.valid_rows}/{self.total_rows} valid rows "
            f"({self.invalid_rows} errors)."
        )


def validate_memory_event(event: Dict[str, Any], row_idx: Optional[int] = None) -> Tuple[bool, List[str]]:
    """
    1. The use of function:
       Validates a single MemoryEvent dictionary against required fields and domain rules.
    2. How does the specific function work:
       - Verifies that all required fields (timestamp, pid, gpu_id, source, event_type) exist and are not empty.
       - Checks that timestamp is a positive numeric value.
       - Checks that memory values (if present) are non-negative numbers.
       - Verifies that allocated_bytes <= reserved_bytes when both are present.
    3. Concepts used in the function:
       Schema validation, defensive type checking, domain constraint enforcement.
    4. Places where this function is imported and used:
       Called by `validate_event_list` and unit tests.
    5. Final output of the function, before and after:
       Before: Raw dictionary.
       After: Tuple of (is_valid: bool, errors: List[str]).
    """
    errors: List[str] = []
    prefix = f"Row {row_idx}: " if row_idx is not None else ""

    # Check required fields
    for req in REQUIRED_FIELDS:
        if req not in event or event[req] is None or str(event[req]).strip() == "":
            errors.append(f"{prefix}Missing required field '{req}'")

    # If required fields are missing, return early
    if errors:
        return False, errors

    # Validate timestamp
    try:
        ts = float(event["timestamp"])
        if ts < 0:
            errors.append(f"{prefix}Timestamp cannot be negative (got {ts})")
    except (ValueError, TypeError):
        errors.append(f"{prefix}Timestamp must be a valid number (got {event['timestamp']})")

    # Validate pid and gpu_id
    try:
        pid = int(event["pid"])
        if pid < 0:
            errors.append(f"{prefix}PID cannot be negative (got {pid})")
    except (ValueError, TypeError):
        errors.append(f"{prefix}PID must be an integer (got {event['pid']})")

    try:
        gpu_id = int(event["gpu_id"])
        if gpu_id < 0:
            errors.append(f"{prefix}GPU ID cannot be negative (got {gpu_id})")
    except (ValueError, TypeError):
        errors.append(f"{prefix}GPU ID must be an integer (got {event['gpu_id']})")

    # Validate source
    source = str(event.get("source", "")).strip().lower()
    if source not in VALID_SOURCES:
        errors.append(
            f"{prefix}Source '{source}' is unrecognized (must be one of {sorted(VALID_SOURCES)})"
        )

    # Validate memory quantities (must be non-negative if provided)
    memory_fields = [
        "total_memory",
        "free_memory",
        "process_memory",
        "allocated_bytes",
        "reserved_bytes",
        "peak_memory",
    ]
    for mem_field in memory_fields:
        val = event.get(mem_field)
        if val is not None and str(val).strip() != "":
            try:
                num = int(float(val))
                if num < 0:
                    errors.append(f"{prefix}{mem_field} cannot be negative (got {num})")
            except (ValueError, TypeError):
                errors.append(f"{prefix}{mem_field} must be a valid integer (got {val})")

    # Logical consistency check: allocated_bytes <= reserved_bytes
    alloc_val = event.get("allocated_bytes")
    resv_val = event.get("reserved_bytes")
    if alloc_val is not None and resv_val is not None:
        try:
            alloc_num = int(float(alloc_val))
            resv_num = int(float(resv_val))
            if alloc_num > resv_num:
                errors.append(
                    f"{prefix}allocated_bytes ({alloc_num}) exceeds reserved_bytes ({resv_num})"
                )
        except (ValueError, TypeError):
            pass

    return len(errors) == 0, errors


def validate_event_list(
    events: List[Dict[str, Any]],
    strict: bool = False
) -> Tuple[List[Dict[str, Any]], ValidationResult]:
    """
    1. The use of function:
       Validates a collection of in-memory event dictionaries.
    2. How does the specific function work:
       Iterates over the list, checking each record using `validate_memory_event`.
       Separates valid rows from invalid ones and generates a `ValidationResult`.
       In strict mode, any single invalid row sets `is_valid=False`.
    3. Concepts used in the function:
       Batch validation, error collection, dataset hygiene.
    4. Places where this function is imported and used:
       Called during dataset ingestion and integration testing.
    5. Final output of the function, before and after:
       Before: Raw list of dicts.
       After: Tuple of (valid_records, ValidationResult).
    """
    valid_rows: List[Dict[str, Any]] = []
    all_errors: List[str] = []

    for idx, event in enumerate(events):
        is_ok, errs = validate_memory_event(event, row_idx=idx)
        if is_ok:
            valid_rows.append(event)
        else:
            all_errors.extend(errs)

    total = len(events)
    num_valid = len(valid_rows)
    num_invalid = total - num_valid

    is_overall_valid = (num_invalid == 0) if strict else (num_valid > 0 or total == 0)

    result = ValidationResult(
        is_valid=is_overall_valid,
        total_rows=total,
        valid_rows=num_valid,
        invalid_rows=num_invalid,
        errors=all_errors,
    )
    return valid_rows, result


def validate_csv_file(
    filepath: Any,
    strict: bool = False
) -> Tuple[List[Dict[str, Any]], ValidationResult]:
    """
    1. The use of function:
       Audits a CSV file from disk, a file-like buffer, or an uploaded file object
       (e.g., Streamlit UploadedFile) against schema and integrity rules.
    2. How does the specific function work:
       - Accepts file paths or file-like buffers with `.read()` or `.getvalue()`.
       - Decodes and parses CSV rows via `csv.DictReader`.
       - Validates records via `validate_event_list`. Handles empty and malformed cases.
    3. Concepts used in the function:
       Polymorphic auditing, defensive parsing, dataset hygiene.
    4. Places where this function is imported and used:
       Called by dashboard file uploaders, command-line verification, and tests.
    5. Final output of the function, before and after:
       Before: Raw CSV file on disk or Streamlit UploadedFile.
       After: Tuple of (sanitized_valid_records, ValidationResult).
    """
    if filepath is None:
        res = ValidationResult(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            invalid_rows=0,
            errors=["No file provided (input was None)"],
        )
        return [], res

    import io

    rows: List[Dict[str, Any]] = []

    # Case 1: Streamlit UploadedFile or file-like object
    if hasattr(filepath, "getvalue") or hasattr(filepath, "read"):
        try:
            if hasattr(filepath, "getvalue"):
                raw_data = filepath.getvalue()
                content = raw_data.decode("utf-8-sig", errors="replace") if isinstance(raw_data, bytes) else str(raw_data)
            else:
                raw_data = filepath.read()
                content = raw_data.decode("utf-8-sig", errors="replace") if isinstance(raw_data, bytes) else str(raw_data)

            if hasattr(filepath, "seek"):
                filepath.seek(0)

            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                rows.append(row)
            return validate_event_list(rows, strict=strict)
        except Exception as e:
            res = ValidationResult(
                is_valid=False,
                total_rows=0,
                valid_rows=0,
                invalid_rows=0,
                errors=[f"Failed to read uploaded file buffer: {str(e)}"],
            )
            return [], res

    # Case 2: String or PathLike file path on disk
    try:
        p = Path(filepath)
    except (TypeError, ValueError):
        res = ValidationResult(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            invalid_rows=0,
            errors=[f"Invalid file path type: {type(filepath).__name__}"],
        )
        return [], res

    if not p.is_file():
        res = ValidationResult(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            invalid_rows=0,
            errors=[f"File not found: {filepath}"],
        )
        return [], res

    try:
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        res = ValidationResult(
            is_valid=False,
            total_rows=0,
            valid_rows=0,
            invalid_rows=0,
            errors=[f"Failed to read CSV file: {str(e)}"],
        )
        return [], res

    return validate_event_list(rows, strict=strict)
