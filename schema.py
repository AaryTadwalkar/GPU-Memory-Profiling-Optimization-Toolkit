"""
File: schema.py

1. What this file does:
   Defines the unified event schema for the GPU Memory Profiling Toolkit.
2. Overall design:
   A single, canonical Python dataclass (`MemoryEvent`) that every module in the
   project imports and uses. It standardizes how both NVML and PyTorch metrics
   are recorded and serialized.
3. Elements in the file:
   - MemoryEvent: Dataclass containing cross-process and in-process fields.
4. Final output of the file:
   A reusable `MemoryEvent` class for consistent log creation and reading.
"""

from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class MemoryEvent:
    """
    1. The use of class:
       Standardizes GPU memory event recording across different collectors.
    2. What the class contain:
       Fields for timestamp, PID, GPU ID, source (nvml/torch), event type,
       cross-process metrics (total, free, process memory), and in-process metrics
       (allocated, reserved, delta, peak).
    3. Technologies used and why:
       Python `dataclass` for lightweight data storage and automatic `__init__`.
       `asdict` for easy conversion to dictionaries to write to CSVs.
    4. Key design and overall design of the class:
       Designed to merge data from two entirely different sources (NVML and PyTorch)
       into a single, unified row schema where N/A fields are just set to None.
    """
    timestamp: float
    pid: int
    gpu_id: int
    source: str
    event_type: str

    total_memory: Optional[int] = None
    free_memory: Optional[int] = None
    process_memory: Optional[int] = None

    allocated_bytes: Optional[int] = None
    reserved_bytes: Optional[int] = None
    allocation_delta: Optional[int] = None
    peak_memory: Optional[int] = None
    workload_name: Optional[str] = None
    batch_size: Optional[int] = None
    precision: Optional[str] = None
    metadata_json: Optional[str] = None
    snapshot_path: Optional[str] = None

    def to_row(self) -> dict:
        """
        1. The use of function:
           Converts the MemoryEvent instance into a dictionary.
        2. How does the specific function work:
           Uses `dataclasses.asdict` to transform the object into a dict.
        3. Concepts used in the function:
           Data serialization.
        4. Places where this function is imported and used:
           Used by collectors (e.g., nvml_monitor.py, torch_hooks.py) when writing
           events to CSV logs.
        5. Final output of the function, before and after:
           Before: MemoryEvent instance.
           After: dict with string keys and corresponding values.
        """
        return asdict(self)