"""
File: nvml_monitor.py

1. What this file does:
   Collects device-level GPU memory statistics using the NVML library.
2. Overall design:
   A standalone script that polls NVML at a regular interval, creates MemoryEvent
   records, and writes them to a CSV file.
3. Elements in the file:
   - collect_memory: Main loop to sample memory and write to CSV.
   - CLI arg parser for execution configuration.
4. Final output of the file:
   A CSV file located in `data/profiles/baseline.csv` containing timeseries 
   memory metrics.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pynvml
from schema import MemoryEvent


def collect_memory(duration: int, output_path: str, interval: float = 1.0):
    """
    1. The use of function:
       Runs a loop to collect NVML metrics and save them.
    2. How does the specific function work:
       Initializes NVML, gets the first GPU handle, and loops until `duration` is reached.
       Sleeps for `interval` seconds between polling. Finally shuts down NVML and dumps
       the collected events to a CSV.
    3. Concepts used in the function:
       Hardware polling via NVML, CSV writing, process ID tracking.
    4. Places where this function is imported and used:
       Executed directly as a script via `if __name__ == "__main__":` block.
    5. Final output of the function, before and after:
       Before: No CSV file or empty events.
       After: Writes all collected events directly to `output_path`.
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    events = []

    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            event = MemoryEvent(
                timestamp=time.time(),
                pid=os.getpid(),
                gpu_id=0,
                source="nvml",
                event_type="memory_sample",
                total_memory=memory.total,
                free_memory=memory.free,
                process_memory=None,
            )
            events.append(event)
            time.sleep(interval)
    finally:
        pynvml.nvmlShutdown()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
                "snapshot_path"
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(event.to_row())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect GPU memory statistics using NVML."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Collection duration in seconds.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/profiles/baseline.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()
    collect_memory(args.duration, args.out)