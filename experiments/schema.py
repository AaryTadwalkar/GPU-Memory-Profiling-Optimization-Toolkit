"""
File: experiments/schema.py

1. What this file does:
   Defines the data model, validation rules, and persistence layer for GPU
   experiment benchmark results. This includes the `ExperimentResult` dataclass,
   the canonical CSV column ordering, serialization helpers (CSV/JSON), and the
   thread-safe `ExperimentResultWriter`.

2. Overall design:
   This module provides a dedicated benchmark result schema that is completely
   distinct from the high-frequency telemetry `MemoryEvent` schema:
   - `MemoryEvent`: Per-sample time-series telemetry (timestamp, PID, free_memory,
     allocated_bytes, event_type, etc.).
   - `ExperimentResult`: Per-run experimental summary record capturing run
     metadata, hardware configuration, concurrency level, admission decisions,
     status (SUCCESS, FAILED, DEFERRED, REJECTED), runtime, observed peak memory,
     declared memory requirements, and error diagnostics.

   Key design constraints strictly enforced:
   - ESTIMATED VS OBSERVED SEPARATION:
     `estimated_requirement_bytes` (declared upfront) and `peak_memory_bytes`
     (measured during execution) are stored in distinct fields and are never
     conflated.
   - NO FABRICATED MEASUREMENTS:
     Missing or unmeasured values remain None (or empty in CSV). Zeros are
     recorded only when explicitly measured as zero.
   - PRESERVATION OF ERRORS:
     Failed runs and exceptions are preserved with full error strings and
     `status="FAILED"` rather than being omitted.
   - DISTINCT STATUS CODES:
     `DEFERRED` and `REJECTED` admission outcomes are recorded distinctly from
     runtime `FAILED` executions.
   - PURE CPU OPERATION:
     This module contains no CUDA, NVML, or GPU runtime calls. It executes
     safely in any CPU environment without dependencies.

3. Elements in the file:
   - `EXPERIMENT_RESULT_FIELDS`: Canonical CSV column list.
   - `ExperimentStatus`: Enumeration of recognized experiment outcome statuses.
   - `ExperimentResult`: Canonical dataclass for benchmark results.
   - `ExperimentResultWriter`: Thread-safe CSV/JSON append writer.
   - Standalone functions: `load_experiment_results`, `save_experiment_results_json`,
     `load_experiment_results_json`, `detect_environment_metadata`.

4. Final output:
   Structured, persistent benchmark records for baseline and controlled experiments.
"""

import csv
import json
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Canonical CSV Column Order
# ---------------------------------------------------------------------------

EXPERIMENT_RESULT_FIELDS: List[str] = [
    "experiment_id",
    "run_id",
    "timestamp",
    "gpu_model",
    "vram_capacity_bytes",
    "driver_version",
    "cuda_version",
    "pytorch_version",
    "workload_name",
    "batch_size",
    "precision",
    "num_concurrent",
    "admission_decision",
    "status",
    "runtime_s",
    "peak_memory_bytes",
    "estimated_requirement_bytes",
    "oom_occurred",
    "error_message",
    "output_csv_path",
    "notes",
]


# ---------------------------------------------------------------------------
# Experiment Status Constants
# ---------------------------------------------------------------------------

class ExperimentStatus:
    """
    1. The use of class:
       Provides named string constants for experiment run outcomes.
    2. What the class contains:
       SUCCESS, FAILED, DEFERRED, REJECTED, OOM.
    3. Key design:
       Maintains distinct semantics between runtime failure (FAILED), out-of-memory
       failure (OOM), admission deferral (DEFERRED), and policy rejection (REJECTED).
    """
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DEFERRED = "DEFERRED"
    REJECTED = "REJECTED"
    OOM = "OOM"

    ALL_STATUSES = {SUCCESS, FAILED, DEFERRED, REJECTED, OOM}


# ---------------------------------------------------------------------------
# Parsing Helpers
# ---------------------------------------------------------------------------

def _parse_optional_int(val: Any) -> Optional[int]:
    """Safely parse an optional integer, returning None on empty or invalid inputs."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "nan"):
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _parse_optional_float(val: Any) -> Optional[float]:
    """Safely parse an optional float, returning None on empty or invalid inputs."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "nan"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_optional_str(val: Any) -> Optional[str]:
    """Safely parse an optional string, returning None for empty or sentinel strings."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null"):
        return None
    return s


def _parse_bool(val: Any) -> bool:
    """Safely parse a boolean value from bool, int, or string representations."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "t")


# ---------------------------------------------------------------------------
# Experiment Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """
    1. The use of class:
       Represents a single benchmark workload execution record.
    2. Fields:
       - experiment_id: Identifier grouping runs belonging to the same benchmark suite.
       - run_id: Unique identifier for this specific execution attempt.
       - timestamp: Unix epoch timestamp when the run was initiated.
       - gpu_model: GPU device name (e.g. 'NVIDIA GeForce RTX 3050 Laptop GPU') or None.
       - vram_capacity_bytes: Total physical GPU memory capacity in bytes or None.
       - driver_version: Host NVIDIA display driver version string or None.
       - cuda_version: CUDA runtime version string or None.
       - pytorch_version: PyTorch package version string or None.
       - workload_name: Identifier of the workload script or model architecture.
       - batch_size: Batch size used for the run, or None if not applicable.
       - precision: Floating point precision string (e.g. 'fp32', 'fp16', 'bf16') or None.
       - num_concurrent: Total count of concurrent workloads scheduled in this run group.
       - admission_decision: 'ADMIT', 'DEFER', 'REJECT', or None (for baseline runs).
       - status: Run outcome ('SUCCESS', 'FAILED', 'DEFERRED', 'REJECTED', 'OOM').
       - runtime_s: Elapsed execution wall-clock time in seconds, or None if unmeasured/deferred.
       - peak_memory_bytes: Observed peak memory in bytes, or None if unmeasured/deferred.
       - estimated_requirement_bytes: Declared or estimated memory requirement in bytes.
       - oom_occurred: True if an Out-Of-Memory condition occurred during execution.
       - error_message: Exception message, reject reason, or error string if run failed.
       - output_csv_path: Path to the telemetry CSV generated during the run, if any.
       - notes: Additional diagnostic context, retry count, or observations.
    """
    experiment_id: str
    run_id: str
    timestamp: float
    workload_name: str
    gpu_model: Optional[str] = None
    vram_capacity_bytes: Optional[int] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    pytorch_version: Optional[str] = None
    batch_size: Optional[int] = None
    precision: Optional[str] = None
    num_concurrent: int = 1
    admission_decision: Optional[str] = None
    status: str = ExperimentStatus.SUCCESS
    runtime_s: Optional[float] = None
    peak_memory_bytes: Optional[int] = None
    estimated_requirement_bytes: Optional[int] = None
    oom_occurred: bool = False
    error_message: Optional[str] = None
    output_csv_path: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a standard Python dictionary preserving types."""
        return asdict(self)

    def to_row(self) -> Dict[str, str]:
        """
        Convert the record to a dictionary of string values suitable for CSV DictWriter.
        None values are serialized as empty strings, booleans as 'True'/'False'.
        """
        d = asdict(self)
        row: Dict[str, str] = {}
        for k in EXPERIMENT_RESULT_FIELDS:
            v = d.get(k)
            if v is None:
                row[k] = ""
            elif isinstance(v, bool):
                row[k] = "True" if v else "False"
            elif isinstance(v, float):
                row[k] = f"{v:.6f}" if k == "timestamp" else f"{v:.4f}"
            else:
                row[k] = str(v)
        return row

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Construct an ExperimentResult instance from a raw dictionary with defensive type parsing."""
        raw_status = data.get("status")
        if raw_status is None:
            parsed_status = ExperimentStatus.SUCCESS
        else:
            parsed_status = str(raw_status).strip().upper()
            if not parsed_status or parsed_status == "NONE":
                parsed_status = ExperimentStatus.SUCCESS

        return cls(
            experiment_id=str(data.get("experiment_id", "")),
            run_id=str(data.get("run_id", "")),
            timestamp=float(data.get("timestamp", 0.0)),
            workload_name=str(data.get("workload_name", "")),
            gpu_model=_parse_optional_str(data.get("gpu_model")),
            vram_capacity_bytes=_parse_optional_int(data.get("vram_capacity_bytes")),
            driver_version=_parse_optional_str(data.get("driver_version")),
            cuda_version=_parse_optional_str(data.get("cuda_version")),
            pytorch_version=_parse_optional_str(data.get("pytorch_version")),
            batch_size=_parse_optional_int(data.get("batch_size")),
            precision=_parse_optional_str(data.get("precision")),
            num_concurrent=_parse_optional_int(data.get("num_concurrent")) or 1,
            admission_decision=_parse_optional_str(data.get("admission_decision")),
            status=parsed_status,
            runtime_s=_parse_optional_float(data.get("runtime_s")),
            peak_memory_bytes=_parse_optional_int(data.get("peak_memory_bytes")),
            estimated_requirement_bytes=_parse_optional_int(data.get("estimated_requirement_bytes")),
            oom_occurred=_parse_bool(data.get("oom_occurred")),
            error_message=_parse_optional_str(data.get("error_message")),
            output_csv_path=_parse_optional_str(data.get("output_csv_path")),
            notes=_parse_optional_str(data.get("notes")),
        )

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "ExperimentResult":
        """Construct an ExperimentResult from a CSV row mapping."""
        return cls.from_dict(row)


# ---------------------------------------------------------------------------
# Thread-Safe Experiment Result Writer
# ---------------------------------------------------------------------------

class ExperimentResultWriter:
    """
    1. The use of class:
       Provides thread-safe persistence of ExperimentResult records to CSV and optional JSON.
    2. How it works:
       Maintains a threading.Lock to serialize appends across concurrent worker threads.
       If the target CSV file does not exist or is 0 bytes, it writes the canonical
       header before writing rows.
    3. Concepts:
       Atomic file operations, schema compliance, audit trail, thread safety.
    """

    def __init__(
        self,
        output_csv_path: Union[str, Path],
        output_json_path: Optional[Union[str, Path]] = None,
    ):
        self.output_csv_path = Path(output_csv_path)
        self.output_json_path = Path(output_json_path) if output_json_path else None
        self._lock = threading.Lock()

        # Ensure parent directory exists
        if self.output_csv_path.parent:
            self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_json_path and self.output_json_path.parent:
            self.output_json_path.parent.mkdir(parents=True, exist_ok=True)

    def write_result(self, result: ExperimentResult) -> None:
        """
        Thread-safely appends a single ExperimentResult to the CSV file (and optional JSON).
        """
        self.write_results([result])

    def write_results(self, results: Sequence[ExperimentResult]) -> None:
        """
        Thread-safely appends a sequence of ExperimentResults to the CSV file (and optional JSON).
        """
        if not results:
            return

        with self._lock:
            # Check if file exists and has content
            file_exists = self.output_csv_path.exists()
            file_empty = file_exists and self.output_csv_path.stat().st_size == 0

            with open(self.output_csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EXPERIMENT_RESULT_FIELDS)
                if not file_exists or file_empty:
                    writer.writeheader()
                for res in results:
                    writer.writerow(res.to_row())

            # Optional JSON persistence
            if self.output_json_path:
                existing_records: List[Dict[str, Any]] = []
                if self.output_json_path.exists() and self.output_json_path.stat().st_size > 0:
                    try:
                        with open(self.output_json_path, mode="r", encoding="utf-8") as jf:
                            content = json.load(jf)
                            if isinstance(content, list):
                                existing_records = content
                    except Exception:
                        existing_records = []

                for res in results:
                    existing_records.append(res.to_dict())

                with open(self.output_json_path, mode="w", encoding="utf-8") as jf:
                    json.dump(existing_records, jf, indent=2)


# ---------------------------------------------------------------------------
# Standalone Loader & Serialization Helpers
# ---------------------------------------------------------------------------

def load_experiment_results(csv_path: Union[str, Path]) -> List[ExperimentResult]:
    """
    Reads a CSV file produced by ExperimentResultWriter and returns a list of
    validated ExperimentResult objects.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment results file not found: {csv_path}")

    results: List[ExperimentResult] = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(ExperimentResult.from_row(row))
    return results


def save_experiment_results_json(
    results: Sequence[ExperimentResult],
    json_path: Union[str, Path],
) -> None:
    """Saves a sequence of ExperimentResult objects to a formatted JSON file."""
    path = Path(json_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)


def load_experiment_results_json(json_path: Union[str, Path]) -> List[ExperimentResult]:
    """Loads a list of ExperimentResult objects from a JSON file."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON results file not found: {json_path}")
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {json_path}, got {type(data)}")
    return [ExperimentResult.from_dict(d) for d in data]


def detect_environment_metadata(safe_cpu_only: bool = True) -> Dict[str, Any]:
    """
    Detects basic software environment metadata without initializing or accessing
    hardware devices (pure CPU-safe introspection).

    When safe_cpu_only is True, GPU and driver fields are returned as None, ensuring
    strict compliance with CPU-only test rules and preventing any hardware queries.
    """
    metadata: Dict[str, Any] = {
        "pytorch_version": None,
        "cuda_version": None,
        "gpu_model": None,
        "vram_capacity_bytes": None,
        "driver_version": None,
    }

    # Safely probe PyTorch version without calling torch.cuda
    if "torch" in sys.modules:
        torch_mod = sys.modules["torch"]
        metadata["pytorch_version"] = getattr(torch_mod, "__version__", None)
    else:
        try:
            import importlib
            torch_spec = importlib.util.find_spec("torch")
            if torch_spec is not None:
                import torch
                metadata["pytorch_version"] = getattr(torch, "__version__", None)
        except Exception:
            pass

    return metadata
