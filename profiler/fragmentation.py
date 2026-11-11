"""
File: profiler/fragmentation.py

1. What this file does:
   Provides allocator-level memory fragmentation profiling and Free-Space Fragmentation
   Index (FSFI) computation for PyTorch caching allocator pools. Supports offline parsing
   of historical JSON snapshots, detailed segment block inspection, and on-demand live
   in-process fragmentation benchmarking on CUDA accelerator hardware with safe tensor
   allocations, inactive block verification, and guaranteed resource cleanup.

2. Overall design:
   The module inspects PyTorch allocator memory snapshots (via torch.cuda.memory_snapshot())
   to quantify internal pool fragmentation without confusing user-space pool inactive blocks
   with physical operating system VRAM fragmentation. When evaluating snapshots, it identifies
   inactive blocks within allocated segments, tracks total inactive free space, and computes
   the canonical FSFI metric:
       FSFI = 1 - (Largest Free Inactive Block / Total Free Inactive Space in Allocator Pool)
   It explicitly distinguishes between live tensor allocated memory, allocator reserved pool
   memory, inactive pool memory, and driver-level free VRAM. The live capture pipeline
   features conservative budgets, verification of inactive block generation, before/after
   empty_cache comparative metrics, and strict try...finally resource cleanup.

3. Elements in the file:
   - parse_snapshot_json: Parses a snapshot JSON file to extract total and largest inactive blocks.
   - parse_snapshot_detailed: Deep parser extracting inactive blocks, block counts, and segment lists.
   - capture_live_fragmentation_snapshot: Executes a safe, controlled allocation sequence on CUDA,
     verifies inactive block generation, captures before/after empty_cache snapshots, and reports
     objective metrics without assuming FSFI must decrease.
   - analyze_snapshots: CLI and offline analytical coordinator processing snapshot CSV events.

4. Final output of the file:
   Returns structured dictionaries, metric tuples, and JSON snapshot dumps consumable by the
   Streamlit dashboard, analytical CLI tools, and unit testing suites.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def parse_snapshot_json(filepath: str) -> Tuple[int, int]:
    """
    1. The use of function:
       Parses a PyTorch memory_snapshot() JSON dump file from disk and extracts cumulative
       inactive free memory and the size of the largest contiguous inactive free block.

    2. How does the specific function work:
       Opens the JSON snapshot, extracts segment collections (supporting both dictionary-based
       and list-based schema variations), iterates over all blocks, filters for blocks with
       state == 'inactive', sums their byte counts, and tracks the maximum inactive block size.

    3. Concepts used in the function:
       PyTorch caching allocator segment structure, inactive block isolation, JSON parsing.

    4. Places where this function is imported and used:
       Imported by dashboard/app.py and called internally by analyze_snapshots.

    5. Final output of the function, before and after:
       Before: JSON snapshot file path on disk.
       After: Tuple of (total_free_memory_bytes, largest_free_block_bytes).
    """
    if not os.path.exists(filepath):
        return 0, 0

    with open(filepath, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    total_free = 0
    largest_free = 0

    segments = snapshot.get("segments", []) if isinstance(snapshot, dict) else snapshot
    if isinstance(segments, list):
        for segment in segments:
            if isinstance(segment, dict):
                for block in segment.get("blocks", []):
                    if isinstance(block, dict) and block.get("state") == "inactive":
                        size = block.get("size", 0)
                        total_free += size
                        if size > largest_free:
                            largest_free = size

    return total_free, largest_free


def parse_snapshot_detailed(
    snapshot_data_or_path: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    1. The use of function:
       Performs a thorough multi-dimensional structural inspection of a PyTorch allocator
       memory snapshot, extracting inactive block counts, active blocks, segment totals, and FSFI.

    2. How does the specific function work:
       Accepts either a filesystem path or an in-memory snapshot data structure. Extracts segments,
       categorizes individual blocks into active and inactive states, calculates total free space,
       identifies the largest free block, computes FSFI, and extracts all block dictionaries for
       histogram rendering.

    3. Concepts used in the function:
       Memory pool introspection, Free-Space Fragmentation Index (FSFI), memory block accounting.

    4. Places where this function is imported and used:
       Called by capture_live_fragmentation_snapshot and dashboard visualizers.

    5. Final output of the function, before and after:
       Before: File path or raw snapshot data structure.
       After: Detailed dictionary containing inactive_bytes, largest_free_block_bytes,
              inactive_block_count, active_block_count, total_blocks, fsfi, and all_blocks list.
    """
    snapshot = None
    if isinstance(snapshot_data_or_path, (str, Path)):
        if os.path.exists(str(snapshot_data_or_path)):
            with open(str(snapshot_data_or_path), "r", encoding="utf-8") as f:
                snapshot = json.load(f)
        else:
            snapshot = []
    elif isinstance(snapshot_data_or_path, (dict, list)):
        snapshot = snapshot_data_or_path
    else:
        snapshot = []

    total_inactive = 0
    largest_inactive = 0
    inactive_count = 0
    active_count = 0
    all_blocks: List[Dict[str, Any]] = []

    segments = snapshot.get("segments", []) if isinstance(snapshot, dict) else snapshot
    if isinstance(segments, list):
        for seg in segments:
            if isinstance(seg, dict):
                for b in seg.get("blocks", []):
                    if isinstance(b, dict):
                        all_blocks.append(b)
                        state = b.get("state")
                        size = int(b.get("size", 0))
                        if state == "inactive":
                            inactive_count += 1
                            total_inactive += size
                            if size > largest_inactive:
                                largest_inactive = size
                        elif state == "active_allocated":
                            active_count += 1

    fsfi = 1.0 - (largest_inactive / float(total_inactive)) if total_inactive > 0 else 0.0
    fsfi = round(max(0.0, min(1.0, fsfi)), 4)

    return {
        "total_inactive_bytes": total_inactive,
        "largest_free_block_bytes": largest_inactive,
        "inactive_block_count": inactive_count,
        "active_block_count": active_count,
        "total_blocks_count": len(all_blocks),
        "fsfi": fsfi,
        "all_blocks": all_blocks,
    }


def get_passive_allocator_status() -> Dict[str, Any]:
    """
    1. The use of function:
       Passively samples PyTorch caching allocator statistics and device memory without
       allocating any artificial tensors or mutating the hardware/driver state.

    2. How does the specific function work:
       Checks PyTorch and CUDA availability. Queries torch.cuda.memory_allocated(),
       torch.cuda.memory_reserved(), torch.cuda.max_memory_allocated(),
       torch.cuda.max_memory_reserved(), and torch.cuda.memory_stats(). Distinguishes
       between an idle allocator pool (0 MB reserved) and an active fragmented pool.
       When the pool is dormant, reports FSFI as None (N/A) with explicit notes,
       preventing false 0.0000 or fabricated fragmentation readings. When the pool is active,
       evaluates inactive_split_bytes and active segments.

    3. Concepts used in the function:
       PyTorch caching allocator introspection, passive telemetry sampling, virtual memory
       pool state distinction (idle vs fragmented).

    4. Places where this function is imported and used:
       Imported by dashboard/app.py in Tab 3 (Mode A: Passive Live Allocator Monitoring)
       and verified in unit tests.

    5. Final output of the function, before and after:
       Before: Raw uninspected allocator runtime.
       After: Structured dictionary with allocated_bytes, reserved_bytes, pool_free_bytes,
              inactive_split_bytes, max_allocated_bytes, max_reserved_bytes,
              allocation_retries, active_segments, is_idle, fsfi, and status_description.
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {
            "available": False,
            "is_idle": True,
            "status_description": "CUDA accelerator not available or PyTorch CUDA runtime absent.",
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "pool_free_bytes": 0,
            "inactive_split_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
            "allocation_retries": 0,
            "active_segments": 0,
            "driver_free_bytes": 0,
            "driver_total_bytes": 0,
            "fsfi": None,
            "distinction_note": "Hardware unavailable.",
        }

    try:
        alloc = torch.cuda.memory_allocated()
        resv = torch.cuda.memory_reserved()
        max_alloc = torch.cuda.max_memory_allocated()
        max_resv = torch.cuda.max_memory_reserved()
        driver_free, driver_total = torch.cuda.mem_get_info()

        stats = {}
        try:
            stats = torch.cuda.memory_stats()
        except Exception:
            pass

        inactive_split = stats.get("inactive_split_bytes.all.current", 0)
        retries = stats.get("num_alloc_retries", 0)
        seg_alloc = stats.get("segment.all.allocated", 0)
        seg_freed = stats.get("segment.all.freed", 0)
        active_segs = max(0, seg_alloc - seg_freed)

        pool_free = max(0, resv - alloc)
        is_idle = (resv == 0)

        if is_idle:
            status_desc = "Current Process Allocator Idle (0 MB reserved)"
            note = (
                "Allocator pool is dormant. No virtual memory segments have been acquired by this "
                "process. FSFI is undefined and reported as N/A (distinguished from 0.0000 fragmentation)."
            )
            fsfi_val = None
        else:
            status_desc = f"Active Allocator Pool ({alloc // (1024*1024)} MB allocated / {resv // (1024*1024)} MB reserved)"
            note = (
                f"Active segments: {active_segs}. Inactive split free space: "
                f"{inactive_split // (1024*1024)} MB of {pool_free // (1024*1024)} MB pool free."
            )
            if pool_free > 0 and inactive_split > 0:
                fsfi_val = round(min(1.0, inactive_split / float(pool_free)), 4)
            else:
                fsfi_val = 0.0

        return {
            "available": True,
            "is_idle": is_idle,
            "status_description": status_desc,
            "allocated_bytes": alloc,
            "reserved_bytes": resv,
            "pool_free_bytes": pool_free,
            "inactive_split_bytes": inactive_split,
            "max_allocated_bytes": max_alloc,
            "max_reserved_bytes": max_resv,
            "allocation_retries": retries,
            "active_segments": active_segs,
            "driver_free_bytes": driver_free,
            "driver_total_bytes": driver_total,
            "fsfi": fsfi_val,
            "distinction_note": note,
        }
    except Exception as exc:
        return {
            "available": False,
            "is_idle": True,
            "status_description": f"Failed to sample allocator: {exc}",
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "pool_free_bytes": 0,
            "inactive_split_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
            "allocation_retries": 0,
            "active_segments": 0,
            "driver_free_bytes": 0,
            "driver_total_bytes": 0,
            "fsfi": None,
            "distinction_note": f"Exception encountered: {exc}",
        }


def capture_live_fragmentation_snapshot(
    tensor_count: int = 16,
    tensor_size_mb: int = 8,
    output_dir: str = "data/profiles/snapshots",
) -> Dict[str, Any]:
    """
    1. The use of function:
       Executes a safe, in-process allocator fragmentation sequence on physical CUDA hardware,
       verifies that inactive blocks were created, captures before/after empty_cache() snapshots,
       and returns objective, measurement-driven telemetry without assuming FSFI must decrease.

    2. How does the specific function work:
       Validates CUDA availability and pre-checks driver free VRAM against required headroom.
       Performs pre-cleanup via gc.collect() and torch.cuda.empty_cache(). Allocates tensor_count
       identically sized float32 tensors, then deletes alternating tensors to poke holes in the
       reserved segments. Captures Phase 1 snapshot, parses inactive blocks, and records actual
       metrics (allocated, reserved, inactive in pool, driver free, FSFI). Calls empty_cache(),
       captures Phase 2 snapshot, and records post-cleanup metrics. Cleans up all remaining
       tensors in a guaranteed finally block and constructs an objective summary explanation.

    3. Concepts used in the function:
       PyTorch caching allocator mechanics, hole-punching allocation patterns, memory snapshotting,
       driver vs allocator memory isolation, resource leak prevention via finally blocks.

    4. Places where this function is imported and used:
       Imported by dashboard/app.py in Tab 3 (Memory Profiling & Fragmentation) and tested in unit tests.

    5. Final output of the function, before and after:
       Before: Dormant GPU allocator state.
       After: Structured telemetry dictionary with saved snapshot paths, before/after metrics,
              inactive block verification flags, and an objective non-prescriptive explanation.
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {
            "success": False,
            "error": "CUDA accelerator not detected or PyTorch CUDA runtime unavailable.",
        }

    # Pre-execution safety headroom check
    req_bytes = tensor_count * tensor_size_mb * 1024 * 1024
    try:
        free_driver_bytes, total_driver_bytes = torch.cuda.mem_get_info()
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to query GPU device memory info: {exc}",
        }

    safe_headroom_bytes = req_bytes * 2 + (256 * 1024 * 1024)
    if free_driver_bytes < safe_headroom_bytes:
        return {
            "success": False,
            "error": (
                f"Insufficient GPU free memory ({free_driver_bytes // (1024*1024)} MB available). "
                f"Required safe headroom: {safe_headroom_bytes // (1024*1024)} MB."
            ),
        }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tensors: List[Optional[torch.Tensor]] = []
    ts = int(time.time())
    snap1_file = out_path / f"snapshot_live_fragmented_{ts}.json"
    snap2_file = out_path / f"snapshot_live_post_empty_cache_{ts}.json"

    try:
        # Pre-cleanup to start from clean allocator state
        gc.collect()
        torch.cuda.empty_cache()

        # Stage 1: Baseline state before experiment
        b_alloc = torch.cuda.memory_allocated()
        b_resv = torch.cuda.memory_reserved()
        b_driver_free, _ = torch.cuda.mem_get_info()

        elements_per_tensor = (tensor_size_mb * 1024 * 1024) // 4  # FP32 = 4 bytes
        for _ in range(tensor_count):
            t = torch.randn(elements_per_tensor, dtype=torch.float32, device="cuda")
            tensors.append(t)
            time.sleep(0.005)

        # Stage 2: Post-allocation state
        a_alloc = torch.cuda.memory_allocated()
        a_resv = torch.cuda.memory_reserved()
        a_driver_free, _ = torch.cuda.mem_get_info()

        # Stage 3: Alternating deletion to punch holes
        kept_tensors: List[torch.Tensor] = []
        for idx, t in enumerate(tensors):
            if idx % 2 == 0 and t is not None:
                kept_tensors.append(t)
            else:
                del t
        tensors = kept_tensors
        time.sleep(0.05)

        # Capture fragmented state (Stage 3)
        snap1_data = torch.cuda.memory_snapshot()
        with open(snap1_file, "w", encoding="utf-8") as f:
            json.dump(snap1_data, f)

        p1_details = parse_snapshot_detailed(snap1_data)
        p1_alloc = torch.cuda.memory_allocated()
        p1_resv = torch.cuda.memory_reserved()
        p1_driver_free, _ = torch.cuda.mem_get_info()

        # Inactive block creation verification
        inactive_blocks_created = p1_details["inactive_block_count"] > 0

        # Stage 4: Call empty_cache() to release inactive segments
        torch.cuda.empty_cache()
        time.sleep(0.05)

        snap2_data = torch.cuda.memory_snapshot()
        with open(snap2_file, "w", encoding="utf-8") as f:
            json.dump(snap2_data, f)

        p2_details = parse_snapshot_detailed(snap2_data)
        p2_alloc = torch.cuda.memory_allocated()
        p2_resv = torch.cuda.memory_reserved()
        p2_driver_free, _ = torch.cuda.mem_get_info()

        # Compute delta metrics
        reserved_delta_mb = (p1_resv - p2_resv) / (1024.0 * 1024.0)
        driver_free_delta_mb = (p2_driver_free - p1_driver_free) / (1024.0 * 1024.0)
        inactive_delta_mb = (p1_details["total_inactive_bytes"] - p2_details["total_inactive_bytes"]) / (1024.0 * 1024.0)

        # OS Concept Explanation: Segment Pinning
        segment_pinning_explanation = (
            "OS Memory Concept (Segment Pinning): In PyTorch's caching allocator, memory is reserved "
            "in contiguous virtual memory segments (typically 20 MB or 512 MB blocks). An allocated segment "
            "can only be returned to the OS / NVIDIA driver via cudaFree if EVERY block inside that segment "
            "is inactive. If alternating deleted tensors leave even a single active tensor in each segment "
            "(segment pinning), empty_cache() cannot free any segment back to the NVIDIA driver, resulting in "
            "0 MB released to the OS driver despite high internal pool fragmentation."
        )

        # Objective, non-prescriptive explanation
        if inactive_blocks_created:
            explanation = (
                f"Allocation of {tensor_count} × {tensor_size_mb} MB tensors followed by alternating deletion "
                f"successfully created {p1_details['inactive_block_count']} inactive blocks in the allocator pool "
                f"({p1_details['total_inactive_bytes'] / (1024*1024):.1f} MB inactive free, FSFI: {p1_details['fsfi']:.4f}). "
                f"Invoking empty_cache() released {reserved_delta_mb:.1f} MB of inactive reserved segments back to the "
                f"NVIDIA driver, increasing driver free VRAM by {driver_free_delta_mb:.1f} MB. "
                f"Post-cleanup FSFI: {p2_details['fsfi']:.4f} across {p2_details['inactive_block_count']} remaining inactive blocks."
            )
        else:
            explanation = (
                f"Allocated {tensor_count} tensors, but the PyTorch caching allocator coalesced segments without "
                f"retaining isolated inactive blocks (inactive count: 0). FSFI remained 0.0000. "
                f"Reserved pool adjusted from {p1_resv / (1024*1024):.1f} MB to {p2_resv / (1024*1024):.1f} MB."
            )

        return {
            "success": True,
            "timestamp": ts,
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
            "tensor_count": tensor_count,
            "tensor_size_mb": tensor_size_mb,
            "inactive_blocks_created": inactive_blocks_created,
            "snapshot_fragmented_path": str(snap1_file),
            "snapshot_post_empty_path": str(snap2_file),
            "stage1_baseline": {
                "allocated_bytes": b_alloc,
                "reserved_bytes": b_resv,
                "driver_free_bytes": b_driver_free,
            },
            "stage2_allocated": {
                "allocated_bytes": a_alloc,
                "reserved_bytes": a_resv,
                "driver_free_bytes": a_driver_free,
                "tensor_count": tensor_count,
            },
            "phase1_fragmented": {
                "allocated_bytes": p1_alloc,
                "reserved_bytes": p1_resv,
                "inactive_bytes": p1_details["total_inactive_bytes"],
                "largest_free_block_bytes": p1_details["largest_free_block_bytes"],
                "inactive_block_count": p1_details["inactive_block_count"],
                "driver_free_bytes": p1_driver_free,
                "fsfi": p1_details["fsfi"],
            },
            "phase2_post_empty_cache": {
                "allocated_bytes": p2_alloc,
                "reserved_bytes": p2_resv,
                "inactive_bytes": p2_details["total_inactive_bytes"],
                "largest_free_block_bytes": p2_details["largest_free_block_bytes"],
                "inactive_block_count": p2_details["inactive_block_count"],
                "driver_free_bytes": p2_driver_free,
                "fsfi": p2_details["fsfi"],
            },
            "reserved_released_mb": round(max(0.0, reserved_delta_mb), 2),
            "driver_free_gained_mb": round(max(0.0, driver_free_delta_mb), 2),
            "inactive_memory_released_mb": round(max(0.0, inactive_delta_mb), 2),
            "explanation": explanation,
            "segment_pinning_explanation": segment_pinning_explanation,
            "all_blocks_phase1": p1_details["all_blocks"],
            "all_blocks_phase2": p2_details["all_blocks"],
        }
    finally:
        # Strict cleanup to prevent memory leaks in the running process
        tensors.clear()
        del tensors
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze_snapshots(csv_path: str) -> None:
    """
    1. The use of function:
       Coordinates the analysis of snapshot events from a recorded CSV timeline, computing FSFI
       and outputting comparative metrics to the console.

    2. How does the specific function work:
       Opens the CSV file, filters for rows with event_type == 'snapshot' and a valid snapshot_path,
       calls parse_snapshot_json, computes FSFI, and prints an objective summary report.

    3. Concepts used in the function:
       CSV parsing, historical trace analysis, FSFI derivation.

    4. Places where this function is imported and used:
       Executed as a CLI script when running profiler/fragmentation.py directly.

    5. Final output of the function, before and after:
       Before: Raw CSV file and snapshot JSONs.
       After: Structured terminal printout summarizing snapshot metrics.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV at {csv_path}")
        return

    snapshots_analyzed = 0
    results: Dict[str, float] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("event_type") == "snapshot" and row.get("snapshot_path"):
                filepath = row["snapshot_path"]
                if not os.path.exists(filepath):
                    print(f"Warning: snapshot file missing: {filepath}")
                    continue

                total_free, largest_free = parse_snapshot_json(filepath)
                fsfi = 1.0 - (largest_free / total_free) if total_free > 0 else 0.0

                label = "unknown"
                if "fragmented" in filepath:
                    label = "fragmented"
                elif "post_empty_cache" in filepath:
                    label = "post_empty_cache"

                results[label] = fsfi

                print("--- Snapshot Analysis ---")
                print(f"File: {os.path.basename(filepath)}")
                print(f"Total Free Memory (Inactive): {total_free} bytes")
                print(f"Largest Free Block:           {largest_free} bytes")
                print(f"FSFI:                         {fsfi:.4f}\n")

                snapshots_analyzed += 1

    if snapshots_analyzed == 0:
        print("No snapshots found in the CSV to analyze.")
    else:
        print("--- Experimental Summary ---")
        if "fragmented" in results and "post_empty_cache" in results:
            print(f"FSFI(fragmented)       = {results['fragmented']:.4f}")
            print(f"FSFI(post_empty_cache) = {results['post_empty_cache']:.4f}")
            delta = results["fragmented"] - results["post_empty_cache"]
            print(f"Observed FSFI Delta    = {delta:+.4f}")
        else:
            print(f"Analyzed {snapshots_analyzed} snapshot(s) successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze FSFI from memory snapshots.")
    parser.add_argument("csv_path", type=str, help="Path to torch_events.csv")
    args = parser.parse_args()

    analyze_snapshots(args.csv_path)
