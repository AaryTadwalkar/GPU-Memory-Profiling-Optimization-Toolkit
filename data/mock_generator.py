"""
File: data/mock_generator.py

1. What this file does:
   Generates realistic synthetic GPU telemetry, multi-process contention datasets,
   admission-control logs, allocator snapshot JSONs, and benchmark experiment results.
   All datasets strictly conform to `schema.py` and `experiments/schema.py`.

2. Overall design:
   A standalone utility enabling full CPU-only development, automated testing, and
   dashboard demonstration without requiring a physical GPU. Simulates realistic
   workload allocation curves (ramp-up, plateau, oscillations, backprop drops),
   multi-workload contention, memory fragmentation, and admission control decisions.

3. Elements in the file:
   - `generate_mock_baseline_nvml`: Device-wide NVML telemetry CSV.
   - `generate_mock_torch_events`: In-process PyTorch allocation trace CSV.
   - `generate_mock_contention_events`: Multi-PID concurrency contention trace CSV.
   - `generate_mock_admission_log`: Cooperative admission controller decision log CSV.
   - `generate_mock_benchmark_results`: Comprehensive experiment matrix CSV.
   - `generate_mock_snapshots`: Fragmented and clean PyTorch allocator JSON snapshots.
   - `generate_all_mock_data`: Convenience driver to generate all assets.
   - CLI entry point: `python data/mock_generator.py --out-dir data/mock/`.

4. Final output of the file:
   Populates `data/mock/` with verified datasets ready for dashboard loading.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schema import MemoryEvent
from experiments.schema import EXPERIMENT_RESULT_FIELDS, ExperimentStatus


# Default 6 GiB VRAM capacity in bytes
DEFAULT_VRAM_CAPACITY = 6 * 1024 * 1024 * 1024  # 6,442,450,944 bytes


def generate_mock_baseline_nvml(
    filepath: Path,
    duration_s: int = 30,
    interval_s: float = 0.5,
    gpu_capacity: int = DEFAULT_VRAM_CAPACITY
) -> Path:
    """
    1. The use of function:
       Generates a mock NVML device monitoring CSV timeseries.
    2. How does the specific function work:
       Simulates baseline desktop OS VRAM usage (~800 MB) with occasional spikes
       and background load, creating canonical MemoryEvent rows with source="nvml".
    3. Concepts used in the function:
       Synthetic timeseries generation and NVML schema compliance.
    4. Places where this function is imported and used:
       Called by `generate_all_mock_data` and dashboard test runners.
    5. Final output of the function, before and after:
       Before: No file on disk.
       After: `mock_baseline.csv` on disk with canonical headers.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time() - duration_s
    steps = int(duration_s / interval_s)
    events: List[Dict[str, Any]] = []

    base_used = 850 * 1024 * 1024  # 850 MB base OS display usage

    for i in range(steps):
        t = t0 + (i * interval_s)
        # Small oscillation to simulate realistic OS memory fluctuations
        oscillation = math.sin(i / 3.0) * (50 * 1024 * 1024)
        used = int(base_used + oscillation)
        free = max(0, gpu_capacity - used)

        ev = MemoryEvent(
            timestamp=round(t, 4),
            pid=0,
            gpu_id=0,
            source="nvml",
            event_type="memory_sample",
            total_memory=gpu_capacity,
            free_memory=free,
        )
        events.append(ev.to_row())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        writer.writeheader()
        writer.writerows(events)

    return filepath


def generate_mock_torch_events(
    filepath: Path,
    workload_name: str = "ResNet50_Training",
    batch_size: int = 32,
    precision: str = "FP32",
    duration_s: int = 25,
    interval_s: float = 0.25,
    gpu_capacity: int = DEFAULT_VRAM_CAPACITY
) -> Path:
    """
    1. The use of function:
       Generates a realistic in-process PyTorch memory allocation timeline.
    2. How does the specific function work:
       Models model loading (warmup), forward pass tensor growth, activation peak,
       backward pass deallocation, and cache-holding behavior.
    3. Concepts used in the function:
       PyTorch caching allocator mechanics (allocated vs reserved).
    4. Places where this function is imported and used:
       Called by `generate_all_mock_data`.
    5. Final output of the function, before and after:
       Before: No file on disk.
       After: `mock_torch_events.csv` with accurate delta and peak records.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time() - duration_s
    steps = int(duration_s / interval_s)
    events: List[Dict[str, Any]] = []

    running_peak = 0
    prev_allocated = 0
    pid = 41280

    for i in range(steps):
        t = t0 + (i * interval_s)
        progress = i / float(steps)

        # Workload phases:
        # Phase 1 (0 - 15%): Model weights load (500 MB)
        # Phase 2 (15 - 55%): Forward activations ramp up to 2.8 GB
        # Phase 3 (55 - 75%): Peak hold during loss calculation (3.2 GB)
        # Phase 4 (75 - 90%): Backprop gradient updates (drops to 1.6 GB)
        # Phase 5 (90 - 100%): Iteration boundary cleanup (drops to weights)
        if progress < 0.15:
            alloc = int((progress / 0.15) * 500 * 1024 * 1024)
            resv = int(alloc * 1.15)
        elif progress < 0.55:
            frac = (progress - 0.15) / 0.40
            alloc = int(500 * 1024 * 1024 + frac * (2300 * 1024 * 1024))
            resv = int(alloc * 1.20)
        elif progress < 0.75:
            alloc = int(2800 * 1024 * 1024 + math.sin(i) * (150 * 1024 * 1024))
            resv = int(3400 * 1024 * 1024)
        elif progress < 0.90:
            frac = (progress - 0.75) / 0.15
            alloc = int(2800 * 1024 * 1024 - frac * (1600 * 1024 * 1024))
            resv = int(3400 * 1024 * 1024)  # PyTorch caching allocator retains segments!
        else:
            alloc = int(600 * 1024 * 1024)
            resv = int(2000 * 1024 * 1024)

        running_peak = max(running_peak, alloc)
        delta = alloc - prev_allocated
        prev_allocated = alloc

        ev = MemoryEvent(
            timestamp=round(t, 4),
            pid=pid,
            gpu_id=0,
            source="torch",
            event_type="memory_sample",
            allocated_bytes=alloc,
            reserved_bytes=resv,
            allocation_delta=delta,
            peak_memory=running_peak,
            workload_name=workload_name,
            batch_size=batch_size,
            precision=precision,
        )
        events.append(ev.to_row())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        writer.writeheader()
        writer.writerows(events)

    return filepath


def generate_mock_contention_events(
    filepath: Path,
    duration_s: int = 35,
    interval_s: float = 0.5,
    gpu_capacity: int = DEFAULT_VRAM_CAPACITY
) -> Path:
    """
    1. The use of function:
       Generates a multi-process GPU resource contention timeseries.
    2. How does the specific function work:
       Simulates 3 distinct processes executing concurrently:
       - Process A (PID 10101, ResNet-50) starts at t=0
       - Process B (PID 20202, Transformer-Bert) arrives at t=8s
       - Process C (PID 30303, CNN-Classifier) arrives at t=16s
       Tracks individual per-process memory, overall device free/used, and peak usage.
    3. Concepts used in the function:
       OS concurrency, resource contention, and PID memory isolation.
    4. Places where this function is imported and used:
       Called by `generate_all_mock_data` to power the Contention dashboard tab.
    5. Final output of the function, before and after:
       Before: No file on disk.
       After: `mock_contention.csv` on disk with multi-PID events.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time() - duration_s
    steps = int(duration_s / interval_s)
    events: List[Dict[str, Any]] = []

    base_os = 800 * 1024 * 1024

    for i in range(steps):
        t = t0 + (i * interval_s)
        rel_t = i * interval_s

        # Process 1: PID 10101 (ResNet-50) runs for entire duration (0 to 35s)
        p1_mem = int((1600 + math.sin(i / 2.0) * 150) * 1024 * 1024)
        events.append(
            MemoryEvent(
                timestamp=round(t, 4),
                pid=10101,
                gpu_id=0,
                source="torch",
                event_type="memory_sample",
                process_memory=p1_mem,
                allocated_bytes=p1_mem,
                reserved_bytes=int(p1_mem * 1.15),
                workload_name="ResNet-50",
            ).to_row()
        )

        # Process 2: PID 20202 (BERT-Tiny) arrives at t=8s and runs until t=28s
        p2_mem = 0
        if 8.0 <= rel_t <= 28.0:
            p2_mem = int((1900 + math.cos(i / 3.0) * 200) * 1024 * 1024)
            events.append(
                MemoryEvent(
                    timestamp=round(t, 4),
                    pid=20202,
                    gpu_id=0,
                    source="torch",
                    event_type="memory_sample",
                    process_memory=p2_mem,
                    allocated_bytes=p2_mem,
                    reserved_bytes=int(p2_mem * 1.12),
                    workload_name="BERT-Tiny",
                ).to_row()
            )

        # Process 3: PID 30303 (CNN-Classifier) arrives at t=16s and runs until t=32s
        p3_mem = 0
        if 16.0 <= rel_t <= 32.0:
            p3_mem = int((1400 + math.sin(i / 1.5) * 100) * 1024 * 1024)
            events.append(
                MemoryEvent(
                    timestamp=round(t, 4),
                    pid=30303,
                    gpu_id=0,
                    source="torch",
                    event_type="memory_sample",
                    process_memory=p3_mem,
                    allocated_bytes=p3_mem,
                    reserved_bytes=int(p3_mem * 1.10),
                    workload_name="CNN-Classifier",
                ).to_row()
            )

        # Device level NVML poll row
        total_used = base_os + p1_mem + p2_mem + p3_mem
        free = max(0, gpu_capacity - total_used)
        events.append(
            MemoryEvent(
                timestamp=round(t + 0.01, 4),
                pid=0,
                gpu_id=0,
                source="nvml",
                event_type="memory_sample",
                total_memory=gpu_capacity,
                free_memory=free,
            ).to_row()
        )

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        writer.writeheader()
        writer.writerows(events)

    return filepath


def generate_mock_admission_log(
    filepath: Path,
    gpu_capacity: int = DEFAULT_VRAM_CAPACITY,
    safety_margin: int = 512 * 1024 * 1024  # 512 MB
) -> Path:
    """
    1. The use of function:
       Generates a mock cooperative admission control decision history log.
    2. How does the specific function work:
       Logs a series of workload requests with estimated requirements, live headroom,
       admission outcomes (ADMIT, DEFER, REJECT), and post-run observed peaks.
    3. Concepts used in the function:
       Resource admission control, safety margins, reservation accounting.
    4. Places where this function is imported and used:
       Called by `generate_all_mock_data`.
    5. Final output of the function, before and after:
       Before: No file on disk.
       After: `mock_admission_log.csv` containing auditable scheduling records.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time() - 3600  # 1 hour ago

    decisions = [
        {
            "timestamp": t0 + 60,
            "workload_id": "wl-001",
            "workload_name": "ResNet18_Inference",
            "requested_bytes": 1073741824,  # 1.0 GB
            "decision": "ADMIT",
            "budget_bytes": gpu_capacity,
            "safety_margin_bytes": safety_margin,
            "available_headroom_bytes": 5120000000,
            "observed_peak_bytes": 1024000000,
            "error_bytes": -49741824,
            "status": "SUCCESS",
            "notes": "Request fits safely within available headroom.",
        },
        {
            "timestamp": t0 + 300,
            "workload_id": "wl-002",
            "workload_name": "BERT_Base_FineTune",
            "requested_bytes": 2684354560,  # 2.5 GB
            "decision": "ADMIT",
            "budget_bytes": gpu_capacity,
            "safety_margin_bytes": safety_margin,
            "available_headroom_bytes": 4046258176,
            "observed_peak_bytes": 2810000000,
            "error_bytes": 125645440,
            "status": "SUCCESS",
            "notes": "Admitted alongside wl-001. Slight underestimation of peak.",
        },
        {
            "timestamp": t0 + 600,
            "workload_id": "wl-003",
            "workload_name": "LLaMA_7B_LoRA",
            "requested_bytes": 3758096384,  # 3.5 GB
            "decision": "DEFER",
            "budget_bytes": gpu_capacity,
            "safety_margin_bytes": safety_margin,
            "available_headroom_bytes": 1361903616,  # ~1.26 GB available
            "observed_peak_bytes": None,
            "error_bytes": None,
            "status": "DEFERRED",
            "notes": "Temporarily deferred: current reservations exceed safe headroom.",
        },
        {
            "timestamp": t0 + 900,
            "workload_id": "wl-004",
            "workload_name": "Gigantic_Diffusion_Model",
            "requested_bytes": 7516192768,  # 7.0 GB
            "decision": "REJECT",
            "budget_bytes": gpu_capacity,
            "safety_margin_bytes": safety_margin,
            "available_headroom_bytes": 5368709120,
            "observed_peak_bytes": None,
            "error_bytes": None,
            "status": "REJECTED",
            "notes": "Rejected: requested memory exceeds total device budget.",
        },
        {
            "timestamp": t0 + 1200,
            "workload_id": "wl-005",
            "workload_name": "Vision_Transformer_Eval",
            "requested_bytes": 1610612736,  # 1.5 GB
            "decision": "ADMIT",
            "budget_bytes": gpu_capacity,
            "safety_margin_bytes": safety_margin,
            "available_headroom_bytes": 4294967296,
            "observed_peak_bytes": 1580000000,
            "error_bytes": -30612736,
            "status": "SUCCESS",
            "notes": "Admitted following release of wl-001 and wl-002.",
        },
    ]

    fieldnames = list(decisions[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)

    return filepath


def generate_mock_benchmark_results(
    filepath: Path,
    gpu_capacity: int = DEFAULT_VRAM_CAPACITY
) -> Path:
    """
    1. The use of function:
       Generates a matrix of benchmark experiment results conforming to `experiments/schema.py`.
    2. How does the specific function work:
       Produces rows covering Model x Batch Size x Precision (FP32 vs FP16)
       comparing baseline unoptimized runs with recommended optimized runs.
    3. Concepts used in the function:
       Benchmarking evaluation, comparative latency/throughput, OOM recording.
    4. Places where this function is imported and used:
       Called by `generate_all_mock_data` to power the Benchmark Evaluation dashboard tab.
    5. Final output of the function, before and after:
       Before: No file on disk.
       After: `mock_benchmark_results.csv` on disk.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time() - 7200

    benchmarks = [
        # Small CNN
        {
            "experiment_id": "EXP-CNN-001",
            "run_id": "run-01",
            "timestamp": t0 + 100,
            "gpu_model": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "vram_capacity_bytes": gpu_capacity,
            "driver_version": "552.22",
            "cuda_version": "12.1",
            "pytorch_version": "2.6.0",
            "workload_name": "Small_CNN",
            "batch_size": 16,
            "precision": "FP32",
            "num_concurrent": 1,
            "admission_decision": "ADMIT",
            "status": "SUCCESS",
            "runtime_s": 4.12,
            "peak_memory_bytes": 450 * 1024 * 1024,
            "estimated_requirement_bytes": 500 * 1024 * 1024,
            "oom_occurred": False,
            "error_message": "",
            "output_csv_path": "data/profiles/exp_cnn_b16.csv",
            "notes": "Baseline Small CNN run.",
        },
        {
            "experiment_id": "EXP-CNN-001",
            "run_id": "run-02",
            "timestamp": t0 + 200,
            "gpu_model": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "vram_capacity_bytes": gpu_capacity,
            "driver_version": "552.22",
            "cuda_version": "12.1",
            "pytorch_version": "2.6.0",
            "workload_name": "Small_CNN",
            "batch_size": 32,
            "precision": "FP32",
            "num_concurrent": 1,
            "admission_decision": "ADMIT",
            "status": "SUCCESS",
            "runtime_s": 6.85,
            "peak_memory_bytes": 820 * 1024 * 1024,
            "estimated_requirement_bytes": 900 * 1024 * 1024,
            "oom_occurred": False,
            "error_message": "",
            "output_csv_path": "data/profiles/exp_cnn_b32.csv",
            "notes": "Batch scale 32.",
        },
        {
            "experiment_id": "EXP-CNN-001",
            "run_id": "run-03",
            "timestamp": t0 + 300,
            "gpu_model": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "vram_capacity_bytes": gpu_capacity,
            "driver_version": "552.22",
            "cuda_version": "12.1",
            "pytorch_version": "2.6.0",
            "workload_name": "Small_CNN",
            "batch_size": 32,
            "precision": "FP16",
            "num_concurrent": 1,
            "admission_decision": "ADMIT",
            "status": "SUCCESS",
            "runtime_s": 4.41,
            "peak_memory_bytes": 480 * 1024 * 1024,
            "estimated_requirement_bytes": 550 * 1024 * 1024,
            "oom_occurred": False,
            "error_message": "",
            "output_csv_path": "data/profiles/exp_cnn_b32_fp16.csv",
            "notes": "Optimized: FP16 cut memory by 41% and reduced runtime by 35%.",
        },
        # ResNet-18
        {
            "experiment_id": "EXP-RESNET-001",
            "run_id": "run-04",
            "timestamp": t0 + 400,
            "gpu_model": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "vram_capacity_bytes": gpu_capacity,
            "driver_version": "551.86",
            "cuda_version": "12.1",
            "pytorch_version": "2.6.0",
            "workload_name": "ResNet18",
            "batch_size": 32,
            "precision": "FP32",
            "num_concurrent": 1,
            "admission_decision": "ADMIT",
            "status": "SUCCESS",
            "runtime_s": 12.30,
            "peak_memory_bytes": 2200 * 1024 * 1024,
            "estimated_requirement_bytes": 2400 * 1024 * 1024,
            "oom_occurred": False,
            "error_message": "",
            "output_csv_path": "data/profiles/exp_resnet_b32.csv",
            "notes": "Baseline ResNet18 FP32.",
        },
        {
            "experiment_id": "EXP-RESNET-001",
            "run_id": "run-05",
            "timestamp": t0 + 500,
            "gpu_model": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "vram_capacity_bytes": gpu_capacity,
            "driver_version": "551.86",
            "cuda_version": "12.1",
            "pytorch_version": "2.6.0",
            "workload_name": "ResNet18",
            "batch_size": 32,
            "precision": "FP16",
            "num_concurrent": 1,
            "admission_decision": "ADMIT",
            "status": "SUCCESS",
            "runtime_s": 8.10,
            "peak_memory_bytes": 1250 * 1024 * 1024,
            "estimated_requirement_bytes": 1400 * 1024 * 1024,
            "oom_occurred": False,
            "error_message": "",
            "output_csv_path": "data/profiles/exp_resnet_b32_fp16.csv",
            "notes": "Optimized: FP16 saved 950 MB VRAM and accelerated throughput.",
        },
        {
            "experiment_id": "EXP-RESNET-001",
            "run_id": "run-06",
            "timestamp": t0 + 600,
            "gpu_model": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "vram_capacity_bytes": gpu_capacity,
            "driver_version": "551.86",
            "cuda_version": "12.1",
            "pytorch_version": "2.6.0",
            "workload_name": "ResNet18",
            "batch_size": 128,
            "precision": "FP32",
            "num_concurrent": 1,
            "admission_decision": "ADMIT",
            "status": "FAILED",
            "runtime_s": 2.10,
            "peak_memory_bytes": 5950 * 1024 * 1024,
            "estimated_requirement_bytes": 6200 * 1024 * 1024,
            "oom_occurred": True,
            "error_message": "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.25 GiB.",
            "output_csv_path": "data/profiles/exp_resnet_b128_oom.csv",
            "notes": "Intentional OOM stress run to evaluate failure capture.",
        },
        # Transformer-2L
        {
            "experiment_id": "EXP-TRANS-001",
            "run_id": "run-07",
            "timestamp": t0 + 700,
            "gpu_model": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "vram_capacity_bytes": gpu_capacity,
            "driver_version": "552.22",
            "cuda_version": "12.1",
            "pytorch_version": "2.6.0",
            "workload_name": "Transformer_2Layer",
            "batch_size": 16,
            "precision": "FP32",
            "num_concurrent": 1,
            "admission_decision": "ADMIT",
            "status": "SUCCESS",
            "runtime_s": 9.45,
            "peak_memory_bytes": 3100 * 1024 * 1024,
            "estimated_requirement_bytes": 3300 * 1024 * 1024,
            "oom_occurred": False,
            "error_message": "",
            "output_csv_path": "data/profiles/exp_trans_b16.csv",
            "notes": "Baseline 2-layer transformer sequence model.",
        },
        {
            "experiment_id": "EXP-TRANS-001",
            "run_id": "run-08",
            "timestamp": t0 + 800,
            "gpu_model": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "vram_capacity_bytes": gpu_capacity,
            "driver_version": "552.22",
            "cuda_version": "12.1",
            "pytorch_version": "2.6.0",
            "workload_name": "Transformer_2Layer",
            "batch_size": 16,
            "precision": "FP16",
            "num_concurrent": 1,
            "admission_decision": "ADMIT",
            "status": "SUCCESS",
            "runtime_s": 5.92,
            "peak_memory_bytes": 1820 * 1024 * 1024,
            "estimated_requirement_bytes": 2000 * 1024 * 1024,
            "oom_occurred": False,
            "error_message": "",
            "output_csv_path": "data/profiles/exp_trans_b16_fp16.csv",
            "notes": "Optimized: Mixed precision enabled 41% memory headroom recovery.",
        },
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(benchmarks)

    return filepath


def generate_mock_snapshots(out_dir: Path) -> Tuple[Path, Path]:
    """
    1. The use of function:
       Generates mock PyTorch allocator JSON snapshots for FSFI fragmentation analysis.
    2. How does the specific function work:
       - Snapshot A (Fragmented): 100 MB free allocator space divided across ten 10 MB inactive blocks (FSFI = 0.90).
       - Snapshot B (Defragmented / Clean): Unused blocks released after cache clear (FSFI = 0.00).
    3. Concepts used in the function:
       PyTorch memory snapshot segment block hierarchy and FSFI formula.
    4. Places where this function is imported and used:
       Called by `generate_all_mock_data` to power the Fragmentation tab.
    5. Final output of the function, before and after:
       Before: No JSON snapshots.
       After: `snapshot_mock_fragmented.json` and `snapshot_mock_clean.json`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frag_path = out_dir / "snapshot_mock_fragmented.json"
    clean_path = out_dir / "snapshot_mock_clean.json"

    # Fragmented segment: 200 MB segment with ten 10 MB active blocks and ten 10 MB inactive blocks
    blocks_frag = []
    for i in range(20):
        state = "active_allocated" if (i % 2 == 0) else "inactive"
        blocks_frag.append({
            "size": 10 * 1024 * 1024,  # 10 MB
            "state": state,
            "requested_size": 10 * 1024 * 1024 if state == "active_allocated" else 0,
        })

    snapshot_frag = {
        "device": 0,
        "segments": [
            {
                "device": 0,
                "address": 140000000000,
                "total_size": 200 * 1024 * 1024,
                "allocated_size": 100 * 1024 * 1024,
                "blocks": blocks_frag,
            }
        ]
    }

    # Clean segment post-cache-clear: only the 100 MB active memory remains; inactive blocks freed
    blocks_clean = [
        {
            "size": 100 * 1024 * 1024,
            "state": "active_allocated",
            "requested_size": 100 * 1024 * 1024,
        }
    ]

    snapshot_clean = {
        "device": 0,
        "segments": [
            {
                "device": 0,
                "address": 140000000000,
                "total_size": 100 * 1024 * 1024,
                "allocated_size": 100 * 1024 * 1024,
                "blocks": blocks_clean,
            }
        ]
    }

    with open(frag_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_frag, f, indent=2)

    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_clean, f, indent=2)

    return frag_path, clean_path


def generate_mock_recommendations(filepath: Path) -> Path:
    """
    1. The use of function:
       Generates a mock recommendations JSON report.
    2. How does the specific function work:
       Produces a structured JSON document with rule citations and evidence metrics.
    3. Concepts used in the function:
       Traceable heuristic recommendations and JSON persistence.
    4. Places where this function is imported and used:
       Called by `generate_all_mock_data`.
    5. Final output of the function, before and after:
       Before: No JSON file.
       After: `mock_recs.json` on disk.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": {
            "budget_bytes": DEFAULT_VRAM_CAPACITY,
            "max_allocated_bytes": 4831838208,  # 4.5 GB
            "max_reserved_bytes": 5905580032,   # 5.5 GB
            "peak_memory_bytes": 4831838208,
            "max_cached_unused_bytes": 1073741824,  # 1.0 GB
            "peak_to_capacity_ratio": 0.75,
            "positive_deltas_count": 84,
            "negative_deltas_count": 42,
            "total_events_analyzed": 140,
            "valid_samples": 140,
            "workload_name": "ResNet50_Training",
            "precision": "FP32",
            "batch_size": 64,
        },
        "recommendations": [
            {
                "rule_id": "OPT-PREC-001",
                "category": "Mixed Precision",
                "triggering_metric": "precision + peak_memory",
                "measured_value": "FP32 with 4608 MB peak (75.0% of 6144 MB budget)",
                "threshold": "FP32 active and peak > 60% of budget",
                "recommendation": "Switch to FP16 mixed precision using torch.cuda.amp.autocast() to reduce activation and weight footprint.",
                "caveat": "Mixed precision may require torch.cuda.amp.GradScaler to prevent gradient underflow during training.",
            },
            {
                "rule_id": "OPT-CACHE-001",
                "category": "Allocator Caching",
                "triggering_metric": "max_cached_unused_bytes",
                "measured_value": "1024.0 MB cached-but-unused (18.2% of reserved)",
                "threshold": "> 512 MB cached-but-unused",
                "recommendation": "Invoke torch.cuda.empty_cache() between validation epochs or major training phases to release unused cached segments.",
                "caveat": "Frequent empty_cache() calls incur CUDA runtime synchronization overhead; use at epoch boundaries only.",
            },
            {
                "rule_id": "OPT-CKPT-001",
                "category": "Gradient Checkpointing",
                "triggering_metric": "peak_to_capacity_ratio",
                "measured_value": "75.0% peak-to-capacity",
                "threshold": "> 65% for training workloads",
                "recommendation": "Apply torch.utils.checkpoint on intermediate feature blocks to trade recomputation for activation memory.",
                "caveat": "Increases training compute time by ~20-30% while reducing activation memory by up to 50%.",
            },
            {
                "rule_id": "OPT-BATCH-001",
                "category": "Batch Size Adjustment",
                "triggering_metric": "peak_to_capacity_ratio",
                "measured_value": "75.0% peak-to-capacity (headroom: 1536 MB)",
                "threshold": "> 70% with safety margin 512 MB",
                "recommendation": "Consider decreasing batch size from 64 to 48 if multi-process contention is anticipated.",
                "caveat": "Lowering batch size may slightly reduce hardware compute utilization.",
            },
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return filepath


def generate_all_mock_data(out_dir: Union[str, Path] = "data/mock") -> Dict[str, Path]:
    """
    1. The use of function:
       Generates the complete set of synthetic files needed for CPU-only execution.
    2. How does the specific function work:
       Calls all individual generators and returns a dictionary of generated paths.
    3. Concepts used in the function:
       Batch data generation and orchestration.
    4. Places where this function is imported and used:
       Called by CLI and the Streamlit dashboard fallback mechanism.
    5. Final output of the function, before and after:
       Before: Empty or partial mock directory.
       After: Complete mock suite ready for visualization.
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    snap_dir = p / "snapshots"

    paths = {
        "baseline": generate_mock_baseline_nvml(p / "mock_baseline.csv"),
        "torch_events": generate_mock_torch_events(p / "mock_torch_events.csv"),
        "contention": generate_mock_contention_events(p / "mock_contention.csv"),
        "admissions": generate_mock_admission_log(p / "mock_admission_log.csv"),
        "benchmarks": generate_mock_benchmark_results(p / "mock_benchmark_results.csv"),
        "recs": generate_mock_recommendations(p / "mock_recs.json"),
    }
    frag_path, clean_path = generate_mock_snapshots(snap_dir)
    paths["snapshot_fragmented"] = frag_path
    paths["snapshot_clean"] = clean_path

    print(f"[MockGenerator] Successfully generated {len(paths)} mock datasets in '{p}'.")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic GPU profiling data.")
    parser.add_argument(
        "--out-dir",
        default="data/mock",
        help="Destination directory for generated files (default: data/mock)",
    )
    args = parser.parse_args()
    generate_all_mock_data(args.out_dir)
