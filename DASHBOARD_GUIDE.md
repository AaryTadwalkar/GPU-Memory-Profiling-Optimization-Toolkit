# ⚡ GPU Memory Profiling & Optimization Toolkit — Dashboard Guide

An in-depth guide explaining every section, control, metric, and visualization in the interactive Streamlit dashboard.

---

## Table of Contents
1. [Overview & Architectural Purpose](#1-overview--architectural-purpose)
2. [Sidebar: Data Source Modes & Configuration](#2-sidebar-data-source-modes--configuration)
   - [Recorded Profiles (`data/profiles/`)](#mode-1-recorded-profiles-dataprofiles)
   - [Mock Simulation (`data/mock/`)](#mode-2-mock-simulation-datamock)
   - [Upload CSV / JSON](#mode-3-upload-csv--json)
   - [Hardware Budget & Safety Margin Controls](#hardware-budget--safety-margin-controls)
3. [The 7 Dashboard Tabs Explained](#3-the-7-dashboard-tabs-explained)
   - [Tab 1: Device Overview (NVML Monitoring)](#tab-1-device-overview-nvml-monitoring)
   - [Tab 2: Allocator Dynamics & Memory Pressure](#tab-2-allocator-dynamics--memory-pressure)
   - [Tab 3: Allocator-Level Fragmentation (FSFI)](#tab-3-allocator-level-fragmentation-fsfi)
   - [Tab 4: Multi-Process Contention & Admission Control](#tab-4-multi-process-contention--admission-control)
   - [Tab 5: Optimization Recommendations & Batch Sizing](#tab-5-optimization-recommendations--batch-sizing)
   - [Tab 6: Benchmark Matrix & Experimental Evaluation](#tab-6-benchmark-matrix--experimental-evaluation)
   - [Tab 7: OS Concepts Mapping & System Architecture](#tab-7-os-concepts-mapping--system-architecture)
4. [Quick Reference Metric Glossary](#4-quick-reference-metric-glossary)

---

## 1. Overview & Architectural Purpose

The **GPU Memory Profiling & Optimization Toolkit Dashboard** (`dashboard/app.py`) is an interactive web interface built with **Streamlit** and **Altair**. 

Its core mission is to bridge high-level deep learning workloads with foundational **Operating System (OS) memory management concepts**, including:
* **Physical memory monitoring** vs. **user-space sub-allocation pools**.
* **Internal and external memory fragmentation** (quantified via the Free-Space Fragmentation Index).
* **Multi-tenant resource contention** and **cooperative admission control** to prevent GPU thrashing and Out-Of-Memory (OOM) crashes.
* **Empirical optimization recommendations** and **working-set batch size estimation**.

---

## 2. Sidebar: Data Source Modes & Configuration

The left sidebar controls what data powers the dashboard and allows live parameter tuning.

```
┌──────────────────────────────────────────────┐
│ Data Source Mode                             │
│  ● 🔴 Live GPU (Real-Time NVML) [Auto]      │
│  ○ Recorded Profiles (data/profiles/)       │
│  ○ Mock Simulation (data/mock/)             │
│  ○ Upload CSV / JSON                         │
├──────────────────────────────────────────────┤
│ Hardware & Budget Configuration              │
│  Target: Auto: NVIDIA GeForce RTX 4050 (6GB) │
│  Admission Safety Margin: 15%                │
│  Safe Headroom Cap: 5.10 GiB                 │
├──────────────────────────────────────────────┤
│ 🔴 Live Polling Controls                     │
│  Auto-Refresh: [ON] (Every 1s)               │
│  Rolling History Window: 60 seconds          │
└──────────────────────────────────────────────┘
```

### Mode 1: 🔴 Live GPU Telemetry (Real-Time NVML)
* **What it means:** Automatically queries your physical NVIDIA GPU in real time via NVML at 1 Hz using Streamlit fragments.
* **Capabilities:**
  * **Auto-Hardware Detection:** Detects GPU name (e.g., `NVIDIA GeForce RTX 4050 Laptop GPU`), driver version, and exact physical VRAM capacity.
  * **Rolling VRAM Timeseries:** Maintains a live rolling time-series graph (last 15s to 120s) showing real physical Used vs Free VRAM.
  * **Live Process Inspector:** Queries `nvmlDeviceGetGraphicsRunningProcesses` and `nvmlDeviceGetComputeRunningProcesses` to track all active processes (DirectX games, Chrome, Discord, and CUDA/PyTorch scripts) with their PIDs and VRAM footprints.
  * **Live Dynamic Admission Control:** Feeds actual live headroom into the Tab 4 Admission Controller simulator.
* **When to use it:** Whenever you are on an NVIDIA GPU machine (RTX 4050 / RTX 3050) and want live operating system resource monitoring.

### Mode 2: Recorded Profiles (`data/profiles/`)
* **What it means:** Loads real-world telemetry files produced by executing actual GPU workloads on NVIDIA GPU machines (e.g., Person 1's RTX 4050 or Person 2's RTX 3050).
* **Files it looks for:**
  * `data/profiles/baseline.csv`: Device-level NVML monitor logs.
  * `data/profiles/torch_events.csv`: In-process PyTorch memory allocation hook traces.
  * `data/profiles/contention.csv`: Multi-PID concurrency traces.
  * `data/profiles/recs.json`: Persisted optimization recommendations.
  * `data/profiles/snapshots/*.json`: PyTorch memory allocator JSON snapshots.
  * `data/profiles/benchmark_results.csv`: Experimental results across models and precisions.
* **When to use it:** When you or your team run workloads on an NVIDIA GPU machine and want to analyze real empirical data from physical experiments.

---

### Mode 3: Mock Simulation (`data/mock/`)
* **What it means:** Loads high-fidelity synthetic simulation datasets generated by `data/mock_generator.py`.
* **Files it looks for:**
  * `data/mock/mock_baseline.csv`
  * `data/mock/mock_torch_events.csv`
  * `data/mock/mock_contention.csv`
  * `data/mock/mock_admission_log.csv`
  * `data/mock/mock_recs.json`
  * `data/mock/mock_benchmark_results.csv`
  * `data/mock/snapshots/*.json`
* **When to use it:** Ideal for **CPU-only laptops** (such as Person 3's machine), continuous integration (CI) tests, presentations, and offline demos. Every tab renders fully populated graphs, KPI cards, tables, and admission simulations without needing an active GPU or pre-recorded profile files.

---

### Mode 4: Upload CSV / JSON
* **What it means:** Provides an interactive drag-and-drop file uploader (`st.sidebar.file_uploader`) allowing you to upload any telemetry CSV or memory snapshot JSON file from your computer.
* **Supported file types:**
  * **Telemetry CSV files:** Conforming to the standardized `MemoryEvent` schema (containing timestamps, allocated/reserved bytes, or device memory metrics).
  * **PyTorch Memory Snapshots (`.json`):** Created via `torch.cuda.memory._dump_snapshot()` containing detailed segment and block allocators.
  * **Benchmark CSV files:** Containing experiment run records.
* **Special capabilities:** 
  * If a PyTorch CSV is uploaded, the dashboard dynamically validates schema bounds, recalculates live KPIs, and runs the rule-based recommendation engine in real time.
  * If a snapshot JSON is uploaded, Tab 3 automatically visualizes its allocator segment block distribution and calculates the Free-Space Fragmentation Index (FSFI).
* **When to use it:** When analyzing ad-hoc profiling logs shared by teammates, exploring newly captured snapshots, or auditing experimental runs.

---

### Hardware Budget & Safety Margin Controls
* **Target Hardware Selector:**
  * `NVIDIA RTX 4050 (6 GB)`: Sets default budget to 6.0 GiB (6,442,450,944 bytes).
  * `NVIDIA RTX 3050 (6 GB)`: Sets default budget to 6.0 GiB.
  * `Custom Budget`: Reveals a slider allowing arbitrary budgets from 1.0 GiB to 24.0 GiB.
* **Admission Safety Margin (%):**
  * Configurable from 5% to 30% (default: 15%).
  * **OS Purpose:** Similar to the operating system reserving a portion of RAM for page cache and kernel buffers, the safety margin guarantees a buffer against sudden allocation spikes before an out-of-memory fault occurs.
* **Safe Headroom Cap Calculation:**
  $$\text{Safe Budget} = \text{Budget} \times (1 - \text{Safety Margin})$$
  Example: For a 6.0 GiB card with a 15% safety margin, the usable headroom is **5.1 GiB**, and the reserved emergency buffer is **0.9 GiB**.

---

## 3. The 7 Dashboard Tabs Explained

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🖥️ 1. Device Overview │ 🧠 2. Allocator │ 🧩 3. Fragmentation │ ⚖️ 4. Contention │ 🚀 5. Optimization │ 📊 6. Benchmarks │ 📚 7. OS Concepts │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Tab 1: Device Overview (NVML Monitoring)
* **What it represents:** **Operating System / Driver-Level Physical Memory Tracking**.
* **Source:** Data gathered out-of-process via the NVIDIA Management Library (`NVML` / `pynvml`).
* **OS Concept:** Global physical memory accounting (analogous to Linux `/proc/meminfo` or Windows Task Manager GPU performance charts). NVML observes total memory occupied across all running processes on the physical card.
* **What it shows:**
  1. **4 KPI Summary Cards:**
     * **Total Device VRAM:** Total physical hardware capacity.
     * **Current Used VRAM:** Memory in active use by the OS display server, window manager, and all background processes.
     * **Current Free Headroom:** Truly unallocated physical memory remaining on the card.
     * **VRAM Utilization (%):** Percentage of physical VRAM currently committed.
  2. **Device VRAM Timeseries (Altair Chart):**
     * Visualizes `device_used_mb` (red line) and `device_free_mb` (green line) plotted against elapsed time.
     * Demonstrates baseline hardware memory footprint and sudden rises when deep learning jobs launch.
  3. **Raw Telemetry Expander:**
     * Collapsible table displaying raw timestamps, elapsed seconds, used/free MB, and GPU utilization metrics.

---

### Tab 2: Allocator Dynamics & Memory Pressure
* **What it represents:** **In-Process User-Space Memory Management**.
* **Source:** In-process hooks tracking PyTorch's caching allocator (`torch.cuda.memory_allocated()`, `torch.cuda.memory_reserved()`).
* **OS Concept:** User-space sub-allocation (analogous to `malloc`/`ptmalloc`/`jemalloc` vs. kernel `brk`/`mmap`). PyTorch avoids the overhead of kernel/driver syscalls for every tensor by allocating large chunks (segments) from the driver and sub-allocating them in user-space.
* **What it shows:**
  1. **4 KPI Summary Cards:**
     * **Peak Allocated (Live):** Maximum high-water mark of memory actually holding live tensor data.
     * **Peak Reserved (Pool):** Maximum memory requested from the NVIDIA driver by PyTorch's caching allocator.
     * **Cached Unused Pool:** Memory held from the OS driver but currently idle/empty ($V_{\text{reserved}} - V_{\text{allocated}}$).
     * **OOM Risk Level (Heuristic):** Dynamic badge (`LOW`, `MODERATE`, `HIGH`) computed from remaining headroom and short-term memory growth rate ($B/s$).
  2. **Allocated vs. Reserved Memory Timeline (Altair Chart):**
     * Two lines comparing active tensor memory (blue) against total reserved pool memory (orange).
     * The gap between the lines reveals **cached unused pool memory**, showing why a process may hold 4 GB of GPU memory even if its tensors only occupy 2 GB.
  3. **Raw In-Process Allocation Samples:**
     * Table displaying sample timestamps, allocated MB, reserved MB, cached unused MB, and peak memory.

---

### Tab 3: Allocator-Level Fragmentation (FSFI)
* **What it represents:** **Internal Allocator Pool Fragmentation**.
* **Source:** PyTorch memory snapshot JSON dumps (`torch.cuda.memory._snapshot()`).
* **OS Concept:** Internal fragmentation within user-space pools (analogous to fragmentation within an OS buddy allocator or slab cache). When tensors of variable shapes are allocated and freed, the pool breaks into non-contiguous blocks.
* **Mathematical Definition:**
  $$\text{FSFI} = 1 - \frac{\text{Largest Free Block}}{\text{Total Free Space in Allocator Pool}}$$
  * **$\text{FSFI} \approx 0.00$:** Zero fragmentation. Free memory is consolidated into one large, contiguous block.
  * **$\text{FSFI} \to 1.00$:** Severe fragmentation. Memory is splintered into small, non-contiguous chunks. An allocation request may fail with an Out-Of-Memory error even when total free memory is plenty, because no single block is large enough.
* **What it shows:**
  1. **4 KPI Summary Cards:**
     * **FSFI Index:** Color-coded fragmentation score ($<0.3$: Green/Healthy, $0.3-0.6$: Orange/Moderate, $>0.6$: Red/Severe).
     * **Largest Free Block:** The single largest contiguous block available in the pool.
     * **Total Free Allocator Space:** The sum of all idle blocks across all segments.
     * **Free Block Count:** The total number of fragmented inactive blocks.
  2. **Allocator Segment Block Distribution (Histogram):**
     * Categorizes blocks into size bins (`< 1 MB`, `1 - 10 MB`, `10 - 50 MB`, `> 50 MB`).
     * Helps developers visualize whether their memory pool consists of useful large blocks or useless small slivers.
  3. **Snapshot Selector Dropdown:** Allows switching between baseline snapshots and peak-load snapshots.

---

### Tab 4: Multi-Process Contention & Admission Control
* **What it represents:** **Multi-Tenancy, Resource Competition, and Scheduling**.
* **Source:** Concurrency traces tracking multiple Process IDs (PIDs) running simultaneously on a single shared GPU.
* **OS Concept:** Multi-process resource contention, thrashing prevention, and cooperative admission control (analogous to an OS scheduling long-running batch jobs vs. real-time tasks to avoid thrashing).
* **What it shows:**
  1. **Multi-Process Contention Timeline (Left Column):**
     * Stacked area chart showing memory consumption partitioned by PID over time.
     * Highlights inter-process competition where multiple workloads simultaneously spike, pushing total usage close to the device budget.
  2. **Per-Process Summary Table:**
     * Displays PID, min memory, mean memory, and max peak memory for every observed workload.
  3. **Interactive Cooperative Admission Controller Simulator (Right Column):**
     * An interactive form simulating an OS admission scheduler.
     * Enter a **Workload Name** and **Estimated VRAM Demand (MiB)**, then click **Submit Workload Request**.
     * Evaluates the request against the configured hardware budget and safety margin:
       * **`ADMIT` (Green):** Safe headroom is sufficient. The workload is allowed to launch.
       * **`DEFER` (Orange):** Request exceeds current headroom but is within total budget; the workload is queued until active jobs release memory.
       * **`REJECT` (Red):** Request violates policy (e.g., exceeds per-workload cap or total budget).
  4. **Auditable Admission Decision Log:**
     * Historical audit table tracking past requests, decisions, requested bytes, observed peaks, and rationale.

---

### Tab 5: Optimization Recommendations & Batch Sizing
* **What it represents:** **Measurement-Driven Optimization & Working-Set Modeling**.
* **Source:** `optimizer/recommendations.py` and `optimizer/batch_optimizer.py`.
* **OS Concept:** Working-set estimation and policy-driven adaptation. Translates profiling data into concrete code changes and optimal batch size configurations.
* **What it shows:**
  1. **Rule-Based Recommendations (Left Column):**
     * Cards generated by deterministic rule checkers auditing the telemetry:
       * **`RULE-BATCH-001` (Batch Size Reduction):** Triggered when peak allocation exceeds safety thresholds ($>85\%$ of budget).
       * **`RULE-PREC-001` (Mixed Precision / AMP):** Triggered when FP32 workloads consume excessive memory. Recommends converting to FP16/BF16 to halve activation footprints.
       * **`RULE-CACHE-001` (Cache Management):** Triggered when cached-unused pool exceeds 1.5 GB. Recommends strategic `torch.cuda.empty_cache()` calls.
       * **`RULE-GRAD-001` (Gradient Checkpointing):** Triggered when activation growth dominates memory. Recommends recomputing activations during backward pass to trade compute for memory.
       * **`RULE-BUF-001` (Buffer Reuse):** Triggered by frequent small allocations causing fragmentation. Recommends pre-allocated scratchpad buffers and in-place operations.
     * Each card includes **Trigger Condition**, **Recommended Action**, and **Operational Caveats** (e.g., runtime overheads).
  2. **Interactive Batch Size Optimizer (Right Column):**
     * Linear memory model:
       $$\text{Projected Peak} = \text{Base Memory} + (\text{Batch Size} \times \text{Activation per Sample})$$
     * Sliders for **Base Weight & Context Memory (MiB)** and **Activation Footprint per Sample (MiB)**.
     * Automatically evaluates candidate batch sizes (`[8, 16, 32, 48, 64, 96, 128]`).
     * Displays a candidate viability table with color-coded badges (`VIABLE`, `RISKY`, `EXCEEDS_BUDGET`) and highlights the recommended maximum safe batch size.

---

### Tab 6: Benchmark Matrix & Experimental Evaluation
* **What it represents:** **Phase 7 Empirical Verification & Trade-Off Analysis**.
* **Source:** `data/profiles/benchmark_results.csv` or `data/mock/mock_benchmark_results.csv`.
* **OS Concept:** Workload characterization and empirical performance evaluation. Validates whether optimization hypotheses (e.g., FP16 vs. FP32) delivered the expected memory reductions and runtime speedups.
* **What it shows:**
  1. **4 KPI Summary Cards:**
     * **Total Experiments:** Total benchmark configurations executed.
     * **Successful Runs:** Percentage and count of completed runs.
     * **OOM Occurrences:** Number of controlled stress tests that triggered Out-Of-Memory.
     * **Evaluated Workloads:** Count of distinct neural network architectures benchmarked (e.g., ResNet, ViT, BERT, GPT-style).
  2. **Peak Memory Comparison (Altair Bar Chart):**
     * Grouped bar chart comparing peak memory across models, batch sizes, and precisions (FP32 baseline vs. FP16 optimized).
  3. **Complete Benchmark Results Table:**
     * Auditable table listing Experiment ID, Workload, Batch Size, Precision, Execution Status (`SUCCESS` / `OOM`), Runtime ($s$), and Peak Memory ($B$).

---

### Tab 7: OS Concepts Mapping & System Architecture
* **What it represents:** **Theoretical & Pedagogical OS Alignment**.
* **Source:** Phase 0 foundational design and the Revised Development Plan.
* **OS Concept:** Explicit conceptual crosswalk connecting GPU systems engineering to classical operating system principles.
* **What it shows:**
  1. **Framework Architecture Diagram:**
     * Mermaid diagram mapping the flow from hardware probes (`NVML`, `PyTorch Hooks`) $\to$ standardization pipeline (`pipeline/`) $\to$ analysis engines $\to$ CLI and Streamlit dashboard.
  2. **OS Concept Mapping Matrix (7 Core Mappings):**
     * **Physical Memory Management:** GPU VRAM tracking vs. OS physical page frames and `/proc/meminfo`.
     * **Dynamic User-Space Allocation:** PyTorch caching allocator vs. glibc `ptmalloc` / `jemalloc` user-space heap pools.
     * **Memory Pool Fragmentation:** FSFI metrics vs. internal/external fragmentation in OS buddy allocators and slab caches.
     * **Multi-Tenant Contention:** Multi-PID profiling vs. OS process scheduling, cgroups, and resource limits.
     * **Admission Control & Thrashing Prevention:** Cooperative admission controller vs. OS page-fault thrashing control and job queues.
     * **Working-Set Modeling:** Linear batch optimizer vs. Denning's Working Set Model for virtual memory.
     * **Empirical Workload Characterization:** Benchmark matrix vs. OS system benchmarking and performance profiling.

---

## 4. Quick Reference Metric Glossary

| Metric Name | Dashboard Location | Unit | Meaning |
| :--- | :--- | :--- | :--- |
| **`device_used_bytes`** | Tab 1, 4 | Bytes / MiB | Total physical GPU memory in use across all processes on the card. |
| **`device_free_bytes`** | Tab 1 | Bytes / MiB | Physical unallocated VRAM remaining on the device. |
| **`allocated_bytes`** | Tab 2, 5 | Bytes / MiB | Memory actively occupied by live PyTorch tensor objects. |
| **`reserved_bytes`** | Tab 2, 5 | Bytes / MiB | Total pool memory requested and held from the NVIDIA driver by PyTorch. |
| **`cached_unused_bytes`**| Tab 2 | Bytes / MiB | Memory held by the allocator pool but not currently assigned to any tensor. |
| **`FSFI`** | Tab 3 | $0.00 - 1.00$ | Free-Space Fragmentation Index. $1.00 - (\text{Largest Free Block} / \text{Total Free})$. |
| **`largest_free_block`** | Tab 3 | Bytes / MiB | The single largest contiguous block available in the allocator's free list. |
| **`Safe Headroom`** | Sidebar, Tab 4 | Bytes / GiB | Budget remaining after deducting active allocations and the safety margin. |
| **`Admission Decision`** | Tab 4 | ADMIT / DEFER / REJECT | Outcome of cooperative admission check based on declared demand vs headroom. |
| **`Peak Memory (Benchmark)`**| Tab 6 | Bytes / MiB | High-water memory usage measured during an experimental workload run. |

---

*This guide was generated for Person 3's deliverables and serves as the official operational documentation for the GPU Memory Profiling & Optimization Toolkit.*
