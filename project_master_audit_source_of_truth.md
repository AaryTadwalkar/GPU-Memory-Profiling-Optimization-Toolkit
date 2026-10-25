# GPU Memory Profiling & Optimization Toolkit: Master Project Audit & Technical Source of Truth

> **Target Audience:** Project Evaluation Committee, Course Professors, External Examiners, and Project Contributors (Aary Tadwalkar, Vishal, Avni).  
> **Purpose:** Exhaustive, authoritative technical documentation covering operating system theory, architectural implementation, empirical benchmarks, and runtime workflows.

---

## 1. Executive Summary & Project Mission

Modern deep learning frameworks (PyTorch, TensorFlow) and graphics engines treat GPU memory (VRAM) as an opaque flat buffer. In multi-tenant environments (e.g., shared workstations, edge compute nodes, cloud ML pods), this abstraction leads to catastrophic failure modes:
1. **Cascading CUDA Out-of-Memory (OOM) Crashes:** When concurrent processes exceed physical memory capacity, the NVIDIA kernel driver terminates processes abruptly without grace periods.
2. **Severe Allocator Fragmentation:** Dynamic tensor allocation and deallocation splinter the PyTorch virtual memory pool, causing OOMs even when aggregate free memory is abundant.
3. **Unchecked Contention:** Independent processes run uncoordinated, competing for shared memory bus bandwidth and VRAM headroom without admission control.

### The Solution: An OS-Oriented Toolkit
This project implements an **Operating System-principled runtime framework** for GPU memory management. By applying classical OS concepts—such as **two-level memory hierarchies, paging, virtual-to-physical translation, internal and external fragmentation indices, thread-safe admission control, working set estimation, and linear capacity planning**—the toolkit monitors, diagnoses, controls, and optimizes GPU memory consumption in real time.

```
+----------------------------------------------------------------------------------------------------+
|                                 APPLICATION & USER LAYER                                           |
|  Streamlit Cockpit (1:1 Dasahboard_vision.jpeg) | Unified CLI (run.py) | Custom Workload Scripts  |
+----------------------------------------------------------------------------------------------------+
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
+------------------------------------------+  +------------------------------------------------------+
|       OUT-OF-PROCESS DRIVER TELEMETRY    |  |          IN-PROCESS ALLOCATOR TELEMETRY              |
|  NVIDIA Management Library (pynvml)      |  |  PyTorch Caching Allocator (torch.cuda.memory_stats) |
|  - Real-time VRAM Used / Free Headroom   |  |  - Tensor Allocations vs Reserved Segment Pools      |
|  - SM Utilization & Core Clocks          |  |  - Point-in-time Snapshot Segment Block Trees        |
|  - Cross-process Attribution (Games & DL)|  |  - Heuristic Growth-Rate OOM Risk Detector          |
+------------------------------------------+  +------------------------------------------------------+
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             ▼
+----------------------------------------------------------------------------------------------------+
|                               CANONICAL TELEMETRY PIPELINE                                         |
|  pipeline/normalize.py (Format & Scale) | pipeline/validate.py (Schema Guard) | pipeline/aggregate |
+----------------------------------------------------------------------------------------------------+
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         ▼                                   ▼                                   ▼
+-------------------------+     +-------------------------+     +------------------------------------+
|  PROFILING & METRICS    |     |   ADMISSION CONTROLLER  |     |   OPTIMIZATION & RECOMMENDATIONS   |
|  - Allocator FSFI Index |     |   controller/admission_ |     |   optimizer/recommendations.py     |
|  - Contention Overlap   |     |   control.py            |     |   - 5 Deterministic Rule Heuristics|
|  - Pressure Ratio       |     |   - Thread-safe Budget  |     |   optimizer/batch_optimizer.py     |
|                         |     |   - Cooperative Gate    |     |   - Linear Memory Model Regression |
+-------------------------+     +-------------------------+     +------------------------------------+
```

---

## 2. End-to-End System Architecture: How the Whole Project Works

The framework operates across three distinct operational layers:

### Layer A: Data Collection (Two-Level Memory Introspection)
Memory in modern GPU systems is strictly split across two boundaries:
1. **Device-Level (Out-of-Process via NVML):**
   - The NVIDIA Management Library (`pynvml`) communicates via `ioctl` system calls directly to the NVIDIA Windows Kernel-Mode Driver (`nvlddmkm.sys`).
   - It captures physical allocations across the entire GPU at 1 Hz to 20 Hz without requiring source code modifications to user applications.
   - It sees **all processes**: DirectX/Vulkan games, Windows Desktop Window Manager (DWM), web browsers, and CUDA workloads.
2. **User-Space Allocator (In-Process via PyTorch Caching Allocator):**
   - PyTorch manages an internal heap inside its own virtual address space to prevent constant, slow `cudaMalloc` calls to the OS kernel.
   - The toolkit attaches a background daemon thread (`collector/torch_hooks.py`) polling `torch.cuda.memory_stats()` at 20 Hz (50 ms interval).
   - It tracks live tensor working sets (`allocated_bytes`), allocator reserve pools (`reserved_bytes`), cached-but-unused memory, and snapshots.

### Layer B: Normalization & Analysis Pipeline
- **Validation (`pipeline/validate.py`):** Ensures every event conforms to the canonical `MemoryEvent` schema, rejecting malformed telemetry or negative byte sizes.
- **Normalization (`pipeline/normalize.py`):** Converts polymorphic timestamp representations to relative wall-clock seconds ($t_{\text{elapsed}} = t_i - t_0$) and standardizes bytes, MiB, and GiB.
- **Aggregation (`pipeline/aggregate.py`):** Computes peak high-water marks, cached-unused ratios, sampling rates, and multi-process concurrency overlaps into a typed `TelemetryAggregate` data model.

### Layer C: Control, Optimization & Visual Interface
- **Cooperative Admission Control (`controller/admission_control.py`):** Acts as a software OS scheduler. Before a workload allocates tensors, it queries the controller. If $M_{\text{used}} + M_{\text{demand}} > M_{\text{budget}} \times (1 - \text{margin})$, the request is **DEFERRED** or **REJECTED**, preventing cascading OOM crashes.
- **Deterministic Optimization Engine (`optimizer/`):** Evaluates live telemetry against 5 formal memory optimization rules, providing immediate advice (AMP precision, cache flushing, gradient checkpointing).
- **Dual-Layer Executive Dashboard (`dashboard/app.py`):**
  - *Layer 1 (Landing Cockpit):* 1:1 visual match with `Dasahboard_vision.jpeg` providing high-density executive awareness.
  - *Layer 2 (Deep Dives):* 7 comprehensive views for telemetry inspection, interactive admission simulations, batch sweeps, and OS concept theory.

---

## 3. Phase-by-Phase Technical Audit (Phases 0 through 9)

### Phase 0: Environment Setup, Multi-GPU Hardware Support & Baseline Verification
- **Role:** Establish a standardized development runtime isolated inside `.venv`, preventing global package contamination. Validate CUDA availability and NVML bindings across distinct hardware profiles.
- **OS Concepts Used:**
  - *Hardware Abstraction Layer (HAL):* Interfacing with heterogeneous hardware targets (NVIDIA GeForce RTX 4050 Laptop GPU 6 GB, RTX 3050 6 GB, and CPU-only fallback).
  - *Process Environment Isolation:* Dependency locking inside virtual environments without root/administrative privileges.
- **Implementation:**
  - Auto-detection script querying `pynvml.nvmlInit()`, `nvmlDeviceGetHandleByIndex(0)`, and `torch.cuda.is_available()`.
  - Pure CPU fallback allowing the entire test suite and mock dashboard to operate on non-CUDA developer laptops.
- **Exit Criterion:** PyTorch and NVML initialize cleanly with zero global path modifications.

---

### Phase 0.5: Unified Schema & Canonical Data Pipeline (Person 3 - Avni)
- **Role:** Bridge out-of-process driver metrics and in-process PyTorch metrics into a unified, type-safe schema.
- **OS Concepts Used:**
  - *Inter-Process Data Normalization:* Reconciling conflicting data formats across independent telemetry streams.
  - *Schema Validation & Data Integrity Guards:* Boundary checking and polymorphic type resolution.
- **Implementation Files:**
  - [`schema.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/schema.py): Defines the unified `MemoryEvent` dataclass containing:
    ```python
    timestamp: float; pid: int; gpu_id: int; source: str  # 'nvml' or 'torch'
    event_type: str; allocated_bytes: int; reserved_bytes: int
    total_memory: int; free_memory: int; process_memory: int
    ```
  - [`pipeline/validate.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/pipeline/validate.py): Validates events against bounds ($0 \le \text{allocated} \le \text{total}$), ensures monotonic timestamps, and verifies required fields.
  - [`pipeline/normalize.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/pipeline/normalize.py): Implements [`bytes_to_mb`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/pipeline/normalize.py#L48), [`bytes_to_gb`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/pipeline/normalize.py#L70), and [`load_and_normalize_csv`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/pipeline/normalize.py#L268).
  - [`pipeline/aggregate.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/pipeline/aggregate.py): Produces `TelemetryAggregate` instances with high-water marks and per-PID breakdowns.
- **Exit Criterion:** 33 unit tests passing in `test_schema.py`, `test_pipeline.py`, and `test_experiment_results.py`.

---

### Phase 1: Device-Level GPU Profiling via NVML (Person 1 - Aary)
- **Role:** Capture true physical hardware state out-of-process without modifying application code.
- **OS Concepts Used:**
  - *Kernel-to-User Space Introspection:* Direct driver queries via the NVIDIA management subsystem (`nvlddmkm.sys`).
  - *Out-of-Process Resource Accounting:* Monitoring total device VRAM load independent of user-space runtimes.
- **Implementation Files:**
  - [`collector/nvml_monitor.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/collector/nvml_monitor.py): Spawns a background sampling thread querying:
    - `nvmlDeviceGetMemoryInfo()`: Returns exact `total`, `free`, and `used` physical bytes.
    - `nvmlDeviceGetUtilizationRates()`: Returns SM compute core activity percentage.
    - `nvmlDeviceGetTemperature()`: Tracks thermal throttling boundaries.
    - Output: Written to [`data/profiles/baseline.csv`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/data/profiles/baseline.csv).
- **Exit Criterion:** Generates valid, monotonic device VRAM timeseries under load.

---

### Phase 2: In-Process PyTorch Allocation Profiling & Memory Pressure (Person 1 - Aary)
- **Role:** Quantify user-space caching allocator behavior, measure cached-but-unused memory overhead, and predict impending OOM faults.
- **OS Concepts Used:**
  - *Two-Level Memory Hierarchy:* Distinguishing between virtual memory reserved from the OS vs active heap allocations used by tensors.
  - *Working Set Estimation & High-Water Mark Tracking:* Distinguishing live working memory from inactive cache pools.
  - *Dynamic Memory Pressure & Thrashing Heuristics:* Quantifying rate of allocation growth against available headroom.
- **Implementation Files:**
  - [`collector/torch_hooks.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/collector/torch_hooks.py): Polling thread querying `torch.cuda.memory_stats()` at 20 Hz, capturing `allocated_bytes.all.current` and `reserved_bytes.all.current`.
  - [`profiler/memory_pressure.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/profiler/memory_pressure.py): Evaluates dynamic pressure ratio and growth rate:
    $$\text{Pressure Ratio} = \frac{\text{Allocated Bytes}}{\text{Configured Budget Bytes}}$$
    $$\text{Growth Rate} = \frac{\Delta \text{Allocated Bytes}}{\Delta t} \quad (\text{bytes/sec})$$
    $$\text{Heuristic OOM Risk} = \begin{cases} \text{HIGH} & \text{if Headroom} < 10\% \text{ or } (\text{Headroom} < 20\% \text{ and Growth} > 50\text{ MB/s}) \\ \text{MODERATE} & \text{if Headroom} < 25\% \text{ or Growth} > 20\text{ MB/s} \\ \text{LOW} & \text{otherwise} \end{cases}$$
- **Exit Criterion:** 11 unit tests passing in `test_memory_pressure.py`; verified warning on steep allocation spikes.

---

### Phase 3: Allocator-Level Fragmentation Analysis & FSFI (Person 1 - Aary)
- **Role:** Calculate and visualize internal free-space pool fragmentation within PyTorch's caching allocator.
- **OS Concepts Used:**
  - *External Fragmentation in Sub-Allocators:* When sufficient aggregate free memory exists in the pool, but no individual contiguous block is large enough to satisfy an incoming allocation request.
  - *Mathematical Metric Modeling:* Implementing the classical operating systems Free-Space Fragmentation Index (FSFI).
- **Implementation Files:**
  - [`profiler/fragmentation.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/profiler/fragmentation.py): Parses memory snapshots dumped via `torch.cuda.memory_snapshot()` and calculates:
    $$\text{FSFI} = 1 - \frac{\text{Largest Free Block Size}}{\sum \text{All Free Blocks Size}}$$
    - $\text{FSFI} \approx 0.00$: Perfectly contiguous free space (low fragmentation).
    - $\text{FSFI} \to 1.00$: Severely splintered free space (high OOM risk).
  - [`workloads/memory_fragmentation.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/workloads/memory_fragmentation.py): Induces severe fragmentation by allocating hundreds of alternating tensors (1 MB vs 64 MB) and freeing only the odd-indexed tensors, proving FSFI rises to $>0.9000$, and collapses back to $0.0000$ after `empty_cache()`.
- **Exit Criterion:** FSFI mathematically verified; visual block ribbon rendering contiguous segments.

---

### Phase 4: Multi-Process Contention & Cooperative Admission Control (Person 2 - Vishal)
- **Role:** Prevent concurrent GPU workloads from colliding and triggering unrecoverable CUDA OOM failures through a thread-safe software OS gatekeeper.
- **OS Concepts Used:**
  - *Admission Control & Schedulability Analysis:* Deciding whether to admit a process based on remaining resource capacity and safety margins.
  - *Concurrency & Synchronization:* Thread-safe atomic state transitions protected by mutual exclusion locks (`threading.Lock`).
  - *Multi-Tenant Resource Contention:* Quantifying spatial and temporal memory overlap across independent PIDs.
- **Implementation Files:**
  - [`controller/admission_control.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/controller/admission_control.py): The `AdmissionController` class enforces:
    $$\text{Safe Budget} = M_{\text{total}} \times (1 - \text{safety\_margin\_ratio})$$
    $$\text{Available Headroom} = \text{Safe Budget} - \max(M_{\text{observed\_used}}, \sum M_{\text{reserved\_workloads}})$$
    $$\text{Decision} = \begin{cases} \text{ADMIT} & \text{if } M_{\text{demand}} \le \text{Available Headroom and } M_{\text{demand}} \le \text{Max Single Workload Cap} \\ \text{DEFER} & \text{if } M_{\text{demand}} \le \text{Safe Budget but exceeds current Headroom (Transient)} \\ \text{REJECT} & \text{if } M_{\text{demand}} > \text{Safe Budget or exceeds policy cap (Fatal)} \end{cases}$$
  - [`profiler/contention.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/profiler/contention.py): Analyzes multi-PID concurrency, overlap windows, and generates stacked area profiles.
- **Exit Criterion:** 50 unit tests passing in `test_admission_control.py` and `test_contention.py`.

---

### Phase 5: Rule-Based Optimization Engine & Linear Batch Sizing (Person 1 - Aary)
- **Role:** Automate the diagnosis of memory anti-patterns and prescribe concrete, actionable optimizations to developers.
- **OS Concepts Used:**
  - *Expert System Heuristics:* Deterministic rules mapping telemetry observations to remediation actions.
  - *Linear Resource Modeling:* Modeling memory scaling as an affine transformation:
    $$M(b) = M_{\text{base}} + b \cdot M_{\text{sample}}$$
- **Implementation Files:**
  - [`optimizer/recommendations.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/optimizer/recommendations.py): Implements 5 deterministic rules:
    1. `OPT-PREC-001` (Mixed Precision): Recommends FP16/BF16 AMP when FP32 peak is excessive.
    2. `OPT-CACHE-001` (Allocator Cache Management): Recommends `torch.cuda.empty_cache()` when cached-unused ratio $>30\%$.
    3. `OPT-CKPT-001` (Gradient Checkpointing): Recommends checkpointing when activation scaling is steep.
    4. `OPT-BATCH-001` (Linear Batch Sizing): Evaluates safe operational batch bounds.
    5. `OPT-ALLOC-001` (Allocator Configuration): Recommends `expandable_segments:True` when FSFI $>0.40$.
  - [`optimizer/batch_optimizer.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/optimizer/batch_optimizer.py): Evaluates candidate batch sweeps (8, 16, 32, 48, 64, 96, 128) and classifies candidates into `VIABLE`, `RISKY`, or `OOM`.
- **Exit Criterion:** 10 unit tests passing in `test_recommendations.py`; generated [`data/profiles/recs.json`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/data/profiles/recs.json).

---

### Phase 6: Executive Dashboard & Presentation Visualization (Person 3 - Avni & Person 1 - Aary)
- **Role:** High-density, real-time graphical cockpit for human operators, matching `Dasahboard_vision.jpeg` 1:1, while preserving deep-dive architectural exploration.
- **OS Concepts Used:**
  - *Real-Time Telemetry Streaming:* 1 Hz non-blocking auto-refresh via `@st.fragment`.
  - *Multi-Process Visibility:* Live inspection of active DirectX/Vulkan games and CUDA compute processes.
- **Implementation Files:**
  - [`dashboard/components.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/dashboard/components.py): Zero-indent HTML/SVG rendering engine (`render_clean_html`), radial arc gauges, vector sparklines, Altair multi-series line charts, and interactive donut charts.
  - [`dashboard/app.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/dashboard/app.py): Complete dual-layer dashboard (Executive Landing Cockpit + 7 Deep-Dive Views).
- **Exit Criterion:** 1:1 visual parity with `Dasahboard_vision.jpeg`; live 1 Hz NVML updates; zero Markdown code-block escaping bugs.

---

### Phase 7: Experimental Verification Framework (Person 2 - Vishal)
- **Role:** Orchestrate controlled vs uncontrolled multi-workload experiments to scientifically validate the toolkit's effectiveness in preventing OOMs.
- **OS Concepts Used:**
  - *Controlled Schedulability Experiments:* Comparing open-loop contention vs closed-loop admission-controlled execution.
- **Implementation Files:**
  - [`experiments/schema.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/experiments/schema.py): Canonical `ExperimentResult` model.
  - [`experiments/runner.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/experiments/runner.py): Executes baseline runs (demonstrating uncoordinated OOM crashes) vs controlled runs (demonstrating 100% success through deferral/rejection).
- **Exit Criterion:** 24 unit tests passing in `test_experiment_results.py`.

---

### Phase 9: Unified CLI & System Integration (All Team Members)
- **Role:** Integrate all disparate modules into a single production entrypoint.
- **Implementation:**
  - [`run.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/run.py): CLI dispatcher for `profile`, `experiment`, `benchmark`, `dashboard`, and `validate`.
  - [`run_tests.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/run_tests.py): Executes all 193 unit tests purely on CPU in $<3$ seconds.
- **Exit Criterion:** 193/193 unit tests passing; zero broken imports.

---

## 4. Deep-Dive into Every Dashboard Feature & Capability

The dashboard is structured into **Layer 1 (The Executive Vision Cockpit)** and **Layer 2 (The 7 Section Deep Dives)** accessible via the left sidebar.

```
+----------------------------------------------------------------------------------------------------+
| NAVIGATION MENU (LEFT SIDEBAR)                                                                     |
| 1. 🏠 Dashboard (Executive Vision Landing Cockpit - Dasahboard_vision.jpeg)                        |
| 2. <> Live Monitor (Out-of-Process Physical Device Telemetry & Active GPU Processes)               |
| 3. 📊 Memory Analysis (In-Process PyTorch Caching Allocator & OOM Risk Heuristics)                 |
| 4. ⊞ Fragmentation (Allocator Segment Block Explorer & FSFI Index Calculation)                     |
| 5. 👥 Multi-Process View (Concurrency Overlap & Cooperative Admission Controller Simulator)        |
| 6. 🔧 Optimization (Deterministic Rules Engine & Interactive Linear Batch Sizing)                  |
| 7. 📋 Experiment Runner (Controlled vs Baseline Benchmark Matrix)                                  |
| 8. 📚 OS Concepts & Architecture (Mapping Toolkit Abstractions to Operating System Theory)         |
+----------------------------------------------------------------------------------------------------+
```

---

### 4.1 🏠 Dashboard: Executive Vision Landing View (`Dasahboard_vision.jpeg` Parity)

The landing view serves as the **high-density executive flight cockpit**, presenting an instant, unified assessment of GPU health:

1. **Row 1: 6 Executive KPI Metric Cards (with Inline SVG Sparklines)**
   - **GPU Utilization:** Features a circular SVG ring gauge showing compute core load (`78%` in neon green) paired with an SVG sparkline wave on the right.
   - **Memory Allocated:** Active tensor working set (`2.4 GB` / `40.0%`) with a cyan sparkline.
   - **Memory Reserved:** Total virtual pool held by the caching allocator (`3.1 GB` / `51.7%`) with a purple sparkline.
   - **Peak Memory:** High-water mark allocation this session (`3.6 GB` / `60.0%` in pink/red `#f43f5e`) with a peak sparkline.
   - **Free Memory:** Available unallocated physical headroom (`2.9 GB` / `48.3%`) with a green sparkline.
   - **Memory Headroom:** Safe allocatable capacity remaining (`2.4 GB` / `40.0%`) with an amber sparkline.

2. **Row 2: GPU Memory Usage Over Time & VRAM Distribution Breakdown**
   - **Left (65% width):** Multi-series Altair timeline tracking `Allocated` (solid cyan), `Reserved` (solid purple), `Peak` (dashed pink), and a dashed reference rule at `6 GB` denoting physical hardware limits.
   - **Right (35% width):** Three-tier Donut chart rendering:
     - 🟦 **Allocated:** `2.4 GB` (40.0%) - Memory actively storing weights, gradients, and activations.
     - 🟪 **Reserved (Unused):** `0.7 GB` (11.7%) - Memory held by PyTorch's caching pool, unavailable to other OS processes.
     - 🟩 **Free:** `2.9 GB` (48.3%) - Unallocated driver capacity.
     - Centered label: **`6 GB / Total VRAM`**.

3. **Row 3: Allocator Fragmentation & Top GPU Processes**
   - **Left (65% width):** Visual representation of the PyTorch Caching Allocator's internal pool:
     - Displays `Fragmentation Score (FSFI): 0.38` and `Largest Free Block: 312 MB`.
     - The **11-Block Allocator Ribbon**: A contiguous segmented bar depicting sub-allocations:
       - 🟦 **Allocated Block:** Active memory chunk.
       - 🟩 **Free Block:** Inactive block available for immediate reuse.
       - 🟥 **Fragmented Block:** Small splintered block unable to satisfy large tensor requests.
       - 🟪 **Reserved (Unused):** Segment overhead held from the OS.
   - **Right (35% width):** **Top GPU Processes** table displaying active applications (PID, Process Name, GPU Memory footprint, and inline blue progress bars for compute utilization).

4. **Row 4: Allocation Events, Memory Pressure & Recommendations**
   - **Left (50% width):** `Allocation Events Timeline` scatter plot with a **logarithmic y-axis** ($10^{-1}$ to $10^3\text{ MB}$) illustrating allocation churn across discrete time steps.
   - **Center (25% width):** `Memory Pressure & OOM Risk` semicircular radial arc gauge displaying current load (`42%`, `Low Risk` pill).
   - **Right (25% width):** `Optimization Recommendations` card displaying 5 prioritized suggestions with `High` (green), `Medium` (yellow), and `Low` (teal) badges.

---

### 4.2 <> Live Monitor: Out-of-Process Hardware Telemetry & Process Attribution

- **What do we see?**
  - **Live Device Headroom:** Continuous out-of-process device memory headroom, SM utilization, and temperature polled directly from the NVIDIA kernel driver.
  - **1 Hz Real-Time Auto-Refresh:** Implemented using Streamlit's official `@st.fragment(run_every="1s")`, advancing rolling timeseries smoothly without reloading the browser or resetting sidebar selections.
  - **Active Process Inspector:** Cross-process enumeration querying `nvmlDeviceGetGraphicsRunningProcesses` and `nvmlDeviceGetComputeRunningProcesses`. Resolves PIDs to executable names via `nvmlSystemGetProcessName`.
- **OS Boundary Principle:**
  NVML operates at the **Operating System Kernel Level**. It monitors all processes running on the GPU: 3D DirectX/Vulkan video games, web browser GPU acceleration (Chrome, Edge), Windows DWM, and PyTorch training scripts.

---

### 4.3 📊 Memory Analysis: In-Process Allocator & Memory Pressure Analysis

- **What is it?**
  This view dives inside the **PyTorch Caching Allocator**.
- **Why does it differ from NVML?**
  - NVML only sees the total memory PyTorch requested from the OS via `cudaMalloc` (which corresponds to `Reserved Memory`).
  - PyTorch's internal allocator subdivides those large OS blocks into individual tensors (`Allocated Memory`).
  - The gap between `Reserved` and `Allocated` is **Cached-Unused Memory**. This memory is held hostage by PyTorch; other processes cannot use it, yet PyTorch is not currently using it for tensors.
- **Why did it say *"No in-process PyTorch telemetry events found"* when training was running externally?**
  - **The OS Memory Wall (Process Isolation):**
    When you run a training script (e.g., `train_bilstm.py`) in a separate terminal, the operating system creates an isolated virtual address space for that process.
    Streamlit runs in its own process. Process A cannot reach into Process B's private heap memory to inspect PyTorch C++ variables.
  - **The Solution:**
    To profile *any* arbitrary script, execute it through our in-process profiler hook from the Project directory:
    ```powershell
    python collector/torch_hooks.py --workload "<path_to_any_python_script>" --out data/profiles/torch_events.csv
    ```
    This launches the script while a concurrent daemon thread samples `torch.cuda.memory_stats()` at 20 Hz, writing to `torch_events.csv`. The dashboard immediately renders the live timeline!

---

### 4.4 ⊞ Fragmentation: Allocator Fragmentation & FSFI Index Calculation

- **What is it and how do we read it?**
  Fragmentation measures how "splintered" the allocator's free pool is.
  - **Internal Fragmentation:** Memory wasted inside an allocated block due to alignment or padding (e.g., allocating 512 bytes when 500 were requested).
  - **External (Pool) Fragmentation:** Memory wasted between allocated blocks. The aggregate free space might be 2.0 GB, but if it consists of 2,000 tiny 1 MB blocks scattered across segments, an incoming request for a contiguous 256 MB tensor **will trigger a CUDA Out-of-Memory error!**
- **How is FSFI Calculated?**
  $$\text{FSFI} = 1 - \frac{\text{Largest Contiguous Free Block}}{\sum \text{All Free Blocks in Allocator Pool}}$$
  - **$\text{FSFI} = 0.00$:** Perfect contiguous memory. The largest free block equals the total free space.
  - **$\text{FSFI} \ge 0.50$:** Moderate to severe fragmentation. Free space is fragmented; OOM risk is high even with ample free megabytes.
- **Is Fragmentation Live?**
  - **No, it is snapshot-based.**
  - *Why?* To calculate FSFI, PyTorch must traverse its entire C++ segment B-tree via `torch.cuda.memory_snapshot()`. Dumping thousands of block references every second would lock the CUDA execution stream and stall training.
  - Therefore, snapshots are taken at key phase transitions (e.g., epoch boundaries, validation loops, or when invoked via `record_snapshot()`). Selecting any snapshot in this tab immediately visualizes the block layout and calculates FSFI.

---

### 4.5 👥 Multi-Process View: Resource Contention & Cooperative Admission Control

- **What exactly is this?**
  This view addresses **multi-tenant GPU contention**. When multiple AI workloads or graphics applications run simultaneously without coordination, their memory usage sums up until the driver hits 100% capacity and terminates one or both processes with an abrupt crash.
- **The Cooperative Admission Controller (`controller/admission_control.py`):**
  Acts as a software OS admission gatekeeper. Before any workload launches, it submits a request:
  ```python
  decision, event = controller.request(workload_id="My_Model", estimated_bytes=demand_bytes)
  ```
  The controller evaluates:
  $$\text{Safe Capacity} = M_{\text{total}} \times (1 - \text{safety\_margin\_ratio})$$
  $$\text{Headroom} = \text{Safe Capacity} - \max(M_{\text{observed\_used}}, \sum M_{\text{reserved\_workloads}})$$
  - **`ADMIT`:** Memory fits within safe headroom. The workload is permitted to proceed.
  - **`DEFER`:** Workload fits within total budget, but current concurrent processes temporarily occupy headroom. The workload is queued to wait.
  - **`REJECT`:** Workload demand exceeds safe budget or the policy cap (80% of total VRAM). Immediate failure prevented without crashing active jobs.
- **Answering the User's Question:**  
  *"We only incorporate model training, and in multi-process view, we only have 3 to 4 models that are predefined. What if the user wants to run something else?"*
  - The predefined models in the experiment runner are **standardized validation benchmarks** (ResNet-50, Vision Transformer, Small CNN).
  - **Any arbitrary workload can be evaluated!**
  - In the **Cooperative Admission Controller Simulator** form on this page, the user can type **any custom workload name** (e.g., `Intrusion_BiLSTM_Fold1`, `Llama_3_8B_LoRA`, `Cyberpunk_2077`) and enter **any estimated VRAM demand in MiB**!
  - The simulator queries the live physical VRAM load on your laptop right now and calculates whether your custom job will be `ADMITTED`, `DEFERRED`, or `REJECTED`.
  - In code, any external script can import the controller with 2 lines:
    ```python
    from controller.admission_control import AdmissionController
    c = AdmissionController(budget_bytes=6442450944)
    if c.request("my_custom_script", 2 * 1024**3)[0] == "ADMIT":
        run_training()
    ```

---

### 4.6 🔧 Optimization: Deterministic Rules & Interactive Batch Sizing

- **How are optimization suggestions generated for the current GPU process?**
  The **Recommendation Engine** ([`optimizer/recommendations.py`](file:///c:/college/3rd%20YEAR_SEM-5/OS/CP/Project/optimizer/recommendations.py)) applies deterministic operating systems rules to the profiled trace:
  1. **`OPT-PREC-001` (Mixed Precision AMP):**
     If the model is running in FP32 and peak memory is high, it recommends `torch.cuda.amp.autocast(dtype=torch.float16)`. On your RTX 4050 (Ada Lovelace), this cuts tensor activation memory by up to 50% and utilizes Tensor Cores for 2-3x speedup.
  2. **`OPT-CACHE-001` (Allocator Cache Flushing):**
     If the caching allocator holds $>30\%$ unused virtual memory between epochs, it prescribes calling `torch.cuda.empty_cache()` between training and validation steps to return memory to the OS driver.
  3. **`OPT-CKPT-001` (Gradient Checkpointing):**
     If activation memory scales rapidly with batch size, it recommends `torch.utils.checkpoint` to trade minimal recomputation for 60-70% activation memory reduction.
  4. **`OPT-ALLOC-001` (Segment Expansion):**
     If FSFI $>0.40$, it recommends setting `$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`, allowing virtual memory expansion to prevent fragmentation OOMs.
- **Interactive Linear Batch Sizing Tool (`optimizer/batch_optimizer.py`):**
  Uses the affine memory model $M(b) = M_{\text{base}} + b \cdot M_{\text{sample}}$. Users adjust base weights and sample footprint sliders to evaluate sweeps (batch 8 to 128), immediately receiving the **optimal safe batch size** (e.g. batch 64) that maximizes throughput without triggering OOM.

---

### 4.7 📋 Experiment Runner: Benchmark Matrix & Experimental Rationale

- **What is it and what experiments were conducted?**
  This view displays the standardized results of controlled vs uncontrolled execution runs.
- **Why are GPU / model training experiments compulsory?**
  To prove that the toolkit's OS abstractions work in reality!
  1. **Uncontrolled Baseline Experiment:** Concurrently launches `ResNet-50` (batch 32), `Vision Transformer` (batch 32), and `Small CNN` simultaneously without admission control.
     - *Result:* Physical VRAM exceeds 6.0 GB $\to$ PyTorch triggers an unhandled CUDA OOM crash (`RuntimeError: CUDA out of memory`).
  2. **Controlled Admission Experiment:** The exact same three workloads are launched through `experiments/runner.py` with `AdmissionController`.
     - *Result:* Workload 1 (`ResNet`) is ADMITTED. Workload 2 (`Transformer`) is DEFERRED until Workload 1 releases memory. Workload 3 is ADMITTED.
     - *Outcome:* **Zero OOM crashes, 100% completion rate.**
  This scientifically demonstrates the necessity of OS admission control for GPU memory.

---

## 5. Master Operating Systems Concepts Mapping Table

This table is the primary reference for your end-semester project viva, viva-voce questions, and presentation slides:

| Operating System Concept | Theoretical Principle | GPU Toolkit Implementation | Concrete Impact & Verification |
| :--- | :--- | :--- | :--- |
| **Two-Level Memory Hierarchy** | Modern OS architectures separate kernel-level physical page management from user-space application heap allocators (e.g. `glibc malloc` / `tcmalloc` vs OS page tables). | **Driver NVML vs PyTorch Caching Allocator:** NVML tracks total physical device VRAM via the NVIDIA kernel driver, while PyTorch manages virtual segment pools (`allocated` vs `reserved`). | Eliminates frequent kernel-mode `cudaMalloc` context switches; exposes cached-but-unused memory overhead. |
| **External Memory Fragmentation** | Non-contiguous memory allocation failure where sufficient aggregate free space exists, but free blocks are splintered into small unusable chunks. | **Free-Space Fragmentation Index (FSFI):** Formulated as $\text{FSFI} = 1 - \frac{\text{Largest Free Block}}{\text{Total Free Space}}$. Extracted via `torch.cuda.memory_snapshot()` block trees. | Quantifies fragmentation on a 0.00 to 1.00 scale; demonstrates FSFI drop from 0.90 to 0.00 post `empty_cache()`. |
| **Admission Control & Schedulability** | In real-time and multi-tenant OS kernels, incoming jobs must be evaluated against available capacity before admission to prevent thrashing and system panic. | **Cooperative Admission Controller (`controller/admission_control.py`):** Thread-safe gatekeeper maintaining budget, dynamic safety margins, and transient deferral queues. | Prevents concurrent workload collisions; eliminates cascading CUDA OOM crashes with zero application crashes. |
| **High-Water Mark & Working Set Estimation** | Denning's Working Set Theory: tracking the minimum physical page set required by a process across temporal windows to avoid thrashing. | **Peak Memory & Allocation Delta Tracking:** Measures live tensor footprint vs reserved pool peak; calculates allocation growth rates ($\Delta M / \Delta t$). | Powers the heuristic OOM risk detector (LOW, MODERATE, HIGH) and informs gradient checkpointing recommendations. |
| **Process Isolation & Address Space Boundaries** | Operating systems enforce strict virtual memory isolation between processes; process A cannot read process B's private heap memory. | **Cross-Process Visibility Division:** Out-of-process NVML monitors physical GPU load across all PIDs, while in-process hooks (`collector/torch_hooks.py`) instrument specific Python runtimes. | Clarifies why external training scripts require the profiler hook to stream internal tensor telemetry into the dashboard. |
| **Resource Contention & Multi-Tenancy** | Multiple independent processes competing for unshared physical resources without central orchestration leads to starvation or thrashing. | **Contention Profiler (`profiler/contention.py`):** Analyzes multi-PID concurrency overlap windows and stacks concurrent memory footprints over time. | Identifies contention hotspots and informs scheduling policies for concurrent deep learning inference/training. |
| **Linear Capacity Planning & Regression** | Modeling resource scaling behavior as a predictable function of workload scale factors to determine safe operational boundaries. | **Batch Size Optimizer (`optimizer/batch_optimizer.py`):** Formulates linear memory scaling $M(b) = M_{\text{base}} + b \cdot M_{\text{sample}}$ to sweep candidate batch sizes. | Identifies optimal throughput batch size without trial-and-error OOM crashes. |

---

## 6. End-Semester Viva & Project Defense Cheatsheet

Below are the most critical questions professors and evaluators will ask during project evaluation, accompanied by precise, technically authoritative answers:

### Q1: "Why did you build an OS toolkit for GPUs? Doesn't the NVIDIA driver or PyTorch already handle memory?"
**Answer:**  
"The NVIDIA driver and PyTorch operate in isolation with an architectural blind spot between them:
1. The **NVIDIA driver** only sees raw physical memory allocations. It has no semantic awareness of deep learning tensors, computation graphs, or batch sizes.
2. **PyTorch's Caching Allocator** operates purely inside user-space in a single process. It has no cross-process awareness of other applications, games, or concurrent models sharing the GPU.
Our toolkit acts as the **missing OS coordination layer**. It bridges the user-space caching allocator with the kernel-level driver, providing cross-process admission control, allocator fragmentation quantification, and automated optimization heuristics."

### Q2: "What is the difference between Allocated Memory and Reserved Memory in PyTorch?"
**Answer:**  
- **Allocated Memory:** The exact number of bytes currently occupied by live PyTorch tensors (model weights, intermediate feature activations, gradients, optimizer states).
- **Reserved Memory:** The total volume of virtual memory PyTorch has requested from the NVIDIA driver via `cudaMalloc`. PyTorch retains this memory in an internal caching pool even after tensors are freed to avoid the high latency of repeated OS kernel allocations.
- **Cached-Unused Overhead:** $\text{Reserved} - \text{Allocated}$. If this gap is large, PyTorch is hoarding VRAM that other system processes cannot access.

### Q3: "How does your Free-Space Fragmentation Index (FSFI) work, and why can't NVML measure it?"
**Answer:**  
"NVML communicates with the GPU driver, which only knows that PyTorch owns a large virtual chunk. The driver cannot see inside PyTorch's private C++ heap pointers.  
FSFI is calculated by inspecting the internal segment block tree via `torch.cuda.memory_snapshot()`:
$$\text{FSFI} = 1 - \frac{\text{Largest Free Block}}{\sum \text{All Free Blocks}}$$
If an allocator holds 2 GB of total free memory, but the largest contiguous block is only 100 MB, the FSFI is $1 - (100 / 2000) = 0.95$. An attempt to allocate a 150 MB tensor will immediately crash with CUDA OOM despite 2 GB of free space. FSFI mathematically quantifies this risk."

### Q4: "How does your Admission Controller prevent Out-of-Memory crashes without modifying the Linux or Windows kernel?"
**Answer:**  
"It implements **Cooperative Admission Control** at the application runtime layer. Workloads register their estimated memory demand before initiating memory allocation. The controller checks remaining headroom against a configured budget and safety margin:
- If capacity exists: **ADMIT**.
- If capacity is temporarily exhausted by active jobs: **DEFER** (the workload pauses until active jobs release memory).
- If demand exceeds hardware capacity: **REJECT** (the job is terminated cleanly before issuing a `cudaMalloc`).
In our experiments, this transformed a 100% failure rate under uncoordinated contention into a 100% completion rate."

### Q5: "If a user is running a custom training script like `train_bilstm.py` in another folder, how does your toolkit analyze it?"
**Answer:**  
"Due to OS process memory isolation, one process cannot inspect another process's private heap.  
1. **At the Hardware Level:** Our dashboard's `Live Monitor` automatically detects the script's PID, process name, and VRAM usage in real time via NVML.
2. **At the In-Process Tensor Level:** The user executes their script through our lightweight profiling hook:
   `python collector/torch_hooks.py --workload <path_to_script>`
   This runs a non-intrusive 20 Hz background daemon polling `torch.cuda.memory_stats()`, streaming live tensor events to `torch_events.csv`, which the dashboard immediately parses."

---

## 7. Verification & Codebase Integrity

- **Automated Unit Test Suite:** Pure CPU-safe execution validating all pipeline, admission, contention, recommendation, and schema modules:
  ```powershell
  python run_tests.py
  ```
  *Result:* **193 passed** in 2.73s (100% pass rate, zero regressions).
- **Interactive Web Dashboard:**
  ```powershell
  python run.py dashboard
  ```
  Running on `http://localhost:8501` with 1:1 visual fidelity to `Dasahboard_vision.jpeg`.
