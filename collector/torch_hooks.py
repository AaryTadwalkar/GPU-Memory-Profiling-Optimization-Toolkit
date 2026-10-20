"""
File: torch_hooks.py

1. What this file does:
   Collects in-process GPU memory statistics using PyTorch's memory stats.
2. Overall design:
   A wrapper script that starts a background polling thread to record PyTorch
   allocator metrics, then executes a provided workload script in the main thread.
3. Elements in the file:
   - TorchHookMonitor: Class handling the background polling.
   - run_workload_and_monitor: Function to orchestrate monitoring and workload execution.
   - CLI arg parser for execution configuration.
4. Final output of the file:
   A CSV file containing the timeseries of in-process memory allocations.
"""

import argparse
import csv
import os
import sys
import time
import threading
import runpy
from pathlib import Path

import json

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from schema import MemoryEvent

_active_monitor = None

def record_snapshot(label: str):
    """
    1. The use of function:
       Saves the current PyTorch allocator snapshot to a JSON file and logs it.
    2. How does the specific function work:
       Calls `torch.cuda.memory_snapshot()`, saves it to disk, and injects a 
       `MemoryEvent` into the active monitor's event stream.
    3. Concepts used in the function:
       PyTorch memory introspection, JSON serialization.
    4. Places where this function is imported and used:
       Imported by workloads (e.g. memory_fragmentation.py) to explicitly 
       mark fragmentation analysis points.
    5. Final output of the function:
       A JSON file on disk and a corresponding CSV event entry.
    """
    global _active_monitor
    if not torch.cuda.is_initialized():
        return
        
    snapshot = torch.cuda.memory_snapshot()
    os.makedirs("data/profiles/snapshots", exist_ok=True)
    filepath = f"data/profiles/snapshots/snapshot_{label}_{int(time.time())}.json"
    
    with open(filepath, "w") as f:
        json.dump(snapshot, f)
        
    if _active_monitor:
        stats = torch.cuda.memory_stats()
        allocated = stats.get("allocated_bytes.all.current", 0)
        reserved = stats.get("reserved_bytes.all.current", 0)
        
        event = MemoryEvent(
            timestamp=time.time(),
            pid=os.getpid(),
            gpu_id=0,
            source="torch",
            event_type="snapshot",
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            allocation_delta=0,
            peak_memory=_active_monitor.peak_memory,
            snapshot_path=filepath
        )
        _active_monitor.events.append(event)


class TorchHookMonitor:
    """
    1. The use of class:
       Manages background polling of PyTorch CUDA memory statistics.
    2. What the class contain:
       Threading constructs, state for peak memory and deltas, and the polling loop.
    3. Technologies used and why:
       Python `threading.Thread` to run concurrently with the workload.
       `torch.cuda.memory_stats()` for fetching allocator metrics.
    4. Key design and overall design of the class:
       Designed to start before the workload, poll at a regular interval,
       and stop cleanly after the workload finishes, logging to a list.
    """
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.events = []
        self._running = False
        self._thread = None
        self.last_allocated = 0
        self.peak_memory = 0
        self.pid = os.getpid()

    def start(self):
        """
        1. The use of function:
           Starts the background monitoring thread.
        2. How does the specific function work:
           Sets the global _active_monitor, running flag to True, and spawns daemon thread.
        3. Concepts used in the function:
           Daemon threads, global state tracking.
        4. Places where this function is imported and used:
           Called in `run_workload_and_monitor`.
        5. Final output of the function, before and after:
           Starts thread and exposes instance to global scope.
        """
        global _active_monitor
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available. Torch hooks will record empty data.")
            return

        _active_monitor = self
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """
        1. The use of function:
           Stops the background monitoring thread.
        2. How does the specific function work:
           Sets running flag to False, joins the thread, clears global.
        3. Concepts used in the function:
           Thread synchronization.
        4. Places where this function is imported and used:
           Called in `run_workload_and_monitor` after the workload finishes.
        5. Final output of the function, before and after:
           Thread terminates, global is cleared.
        """
        global _active_monitor
        self._running = False
        if self._thread:
            self._thread.join()
        _active_monitor = None

    def _poll_loop(self):
        """
        1. The use of function:
           The core loop that periodically fetches memory stats.
        2. How does the specific function work:
           Loops while `_running` is True. Fetches `allocated_bytes.all.current` and
           `reserved_bytes.all.current`. Calculates delta and peak, creates a MemoryEvent,
           and appends it. Sleeps for `interval`.
        3. Concepts used in the function:
           PyTorch CUDA memory stats introspection.
        4. Places where this function is imported and used:
           Run internally by the thread started in `start()`.
        5. Final output of the function, before and after:
           Appends MemoryEvent objects to the `events` list over time.
        """
        while self._running:
            try:
                if not torch.cuda.is_initialized():
                    # Before CUDA is initialized, memory is genuinely zero for this process
                    allocated = 0
                    reserved = 0
                else:
                    stats = torch.cuda.memory_stats()
                    # Use 0 if the key isn't there (e.g. no allocations yet)
                    allocated = stats.get("allocated_bytes.all.current", 0)
                    reserved = stats.get("reserved_bytes.all.current", 0)
            except Exception as e:
                print(f"[TorchHookMonitor] Error fetching memory stats: {e}")
                allocated = None
                reserved = None

            if allocated is not None:
                allocation_delta = allocated - self.last_allocated
                self.last_allocated = allocated
                
                if allocated > self.peak_memory:
                    self.peak_memory = allocated
            else:
                allocation_delta = None

            event = MemoryEvent(
                timestamp=time.time(),
                pid=self.pid,
                gpu_id=0,
                source="torch",
                event_type="memory_sample",
                allocated_bytes=allocated,
                reserved_bytes=reserved,
                allocation_delta=allocation_delta,
                peak_memory=self.peak_memory
            )
            self.events.append(event)
            time.sleep(self.interval)

def run_workload_and_monitor(workload_path: str, output_path: str):
    """
    1. The use of function:
       Coordinates the execution of the workload and the monitoring thread.
    2. How does the specific function work:
       Instantiates `TorchHookMonitor`, starts it, runs the workload using `runpy.run_path`,
       stops the monitor, and writes all collected events to CSV.
    3. Concepts used in the function:
       Dynamic script execution (`runpy`), CSV writing.
    4. Places where this function is imported and used:
       Executed by the main block when the script is run from CLI.
    5. Final output of the function, before and after:
       Before: Output CSV does not exist.
       After: Output CSV is generated containing all memory events.
    """
    monitor = TorchHookMonitor()
    monitor.start()
    
    print(f"Executing workload: {workload_path}")
    try:
        # runpy allows us to execute the script in the current process space.
        # We inject record_snapshot into the globals so the workload uses the 
        # exact same function reference and module state as the wrapper.
        runpy.run_path(workload_path, init_globals={"record_snapshot": record_snapshot}, run_name="__main__")
    except Exception as e:
        print(f"Workload execution failed: {e}")
    finally:
        monitor.stop()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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
        for event in monitor.events:
            writer.writerow(event.to_row())
            
    print(f"Collected {len(monitor.events)} events. Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect in-process GPU memory stats via PyTorch."
    )
    parser.add_argument(
        "--workload",
        type=str,
        required=True,
        help="Path to the workload script to execute.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/profiles/torch_events.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()
    
    run_workload_and_monitor(args.workload, args.out)
