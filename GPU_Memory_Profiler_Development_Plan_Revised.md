# GPU Memory Profiling & Optimization Toolkit
## Revised Development Plan

**An OS-oriented framework for runtime GPU memory analysis, diagnosis, resource management, and optimization**

**Team:** Vishal Arkalwar, Avni Raich, Aary Tadwalkar  
**Target hardware:** NVIDIA RTX 3050 6GB and RTX 4050 6GB  
**Primary stack:** Python, PyTorch, NVML, Pandas, NumPy, Streamlit  
**Optional stretch stack:** C/CUDA, Linux `LD_PRELOAD`

---

## 1. Project Overview

### Problem statement

Standard GPU monitoring tools such as `nvidia-smi` expose device utilization and memory totals, but do not by themselves explain allocation behavior, allocator-level fragmentation, memory pressure, workload peaks, or the causes of resource contention.

The GPU Memory Profiling & Optimization Toolkit combines monitoring, diagnosis, optimization recommendations, and controlled experiments in a single framework inspired by operating-system resource management.

### Project objective

Build a practical toolkit that:

1. Collects device-level and in-process GPU memory information.
2. Analyzes allocation patterns, memory peaks, allocator fragmentation, and multi-process contention.
3. Estimates memory pressure and warns about potential out-of-memory (OOM) conditions.
4. Recommends and experimentally evaluates memory optimization configurations.
5. Demonstrates cooperative GPU workload admission control.
6. Presents live and historical results through a Streamlit dashboard.

### Research question

> Can allocator-level profiling and lightweight, measurement-driven optimization reduce GPU memory pressure and improve workload execution reliability on resource-constrained GPUs?

The project is a teaching-scale systems framework. It should not claim to match production GPU memory managers or guarantee prevention of all OOM errors.

---

## 2. Core Idea

Treat GPU memory as a finite shared resource, applying OS concepts to GPU workloads:

- **Allocation and deallocation:** who requested memory, how much, and when.
- **Memory fragmentation:** how free allocator space is distributed.
- **Caching:** memory reserved by the framework but not currently occupied by live tensors.
- **Memory pressure:** how much headroom remains and how quickly it is shrinking.
- **Peak usage:** the high-water mark of a workload.
- **Contention:** how concurrent processes compete for the same GPU.
- **Admission control:** deciding whether a workload should be launched under a configured memory budget.
- **Optimization:** selecting workload settings based on measurements rather than guesses.

---

## 3. OS Concepts Demonstrated

| OS concept | Project application | Evidence / metric |
|---|---|---|
| Memory management | GPU allocation, deallocation, and caching | Allocated/reserved memory timeline |
| Allocation strategies | CUDA and PyTorch caching allocator behavior | Allocation deltas and block statistics |
| Fragmentation | Distribution of free allocator blocks | FSFI, largest free block, free-block count |
| Resource management | Continuous GPU monitoring | Memory usage and headroom |
| Resource contention | Concurrent GPU workloads | Per-PID memory, latency, throughput |
| Process management | PID-based tracking | Per-process time series |
| Scheduling/resource sharing | Cooperative admission control | Admission, defer, reject decisions |
| Caching | Reserved-but-unused allocator memory | Reserved minus allocated memory |
| Resource safety | Memory-budget and pressure warnings | Headroom, growth rate, OOM outcomes |

Each module and report result should map to one or more rows in this table.

---

## 4. System Architecture

```text
GPU Workloads
(CNN / ResNet / Transformer / Stress / Concurrent processes)
                         |
                         v
Collection & Tracking Layer
(NVML + PyTorch allocator introspection + PID tracking)
                         |
                         v
Profiling & Diagnosis Engine
(Peaks, rates, allocator fragmentation, pressure, OOM risk)
                         |
              +----------+-----------+
              |                      |
              v                      v
Optimization Engine          Admission-Control Module
(Batch, precision,           (Memory budget, workload
checkpointing, reuse)        admission, defer/reject)
              |                      |
              +----------+-----------+
                         |
                         v
Evaluation & Experiment Runner
(Baseline vs optimized, benchmark matrix, repeatability)
                         |
                         v
Streamlit Dashboard
(Live metrics, charts, process view, recommendations,
experiment comparisons, admission decisions)
```

### 4.1 Collection and Tracking Layer

Collects raw signals from:

- NVML: device-wide memory totals, free/used memory, utilization, GPU identity, and per-process memory where supported.
- PyTorch allocator APIs: allocated and reserved bytes, peaks, allocator statistics, and snapshots from within the profiled process.
- Workload instrumentation: PID, workload name, batch size, precision, timing, and experiment metadata.

**Technical boundary:** NVML reports device- and process-level totals; it does not report individual `cudaMalloc`/`cudaFree` calls. PyTorch allocator introspection sees the allocator state of the process that calls it. Workload scripts must self-report their allocator data into the shared event log.

Use conservative polling first. Treat APIs such as `torch.cuda.memory._record_memory_history()` as version-dependent enhancements, not core dependencies.

### 4.2 Profiling and Diagnosis Engine

Consumes normalized events and computes:

- Allocation/deallocation rates and memory deltas.
- Peak usage and peak-to-capacity ratio.
- Allocator-level fragmentation metrics.
- Memory headroom and growth rate.
- Rule-based memory-pressure/OOM risk.
- Per-process contention summaries and workload correlations.

### 4.3 Optimization Engine

Generates recommendations backed by observed metrics:

- Reduce or adjust batch size.
- Evaluate FP16/BF16 mixed precision where supported.
- Consider gradient checkpointing for training workloads.
- Identify large reserved-but-unused memory.
- Suggest buffer reuse or avoiding repeated allocations.
- Suggest a controlled cache-clearing experiment when appropriate.

Each recommendation must include the triggering metric, measured value, threshold or rule, and a caveat where relevant.

### 4.4 Admission-Control Module

A cooperative controller for workloads launched through the toolkit:

1. A workload declares or estimates its memory requirement.
2. The controller reads current device usage and the configured memory budget.
3. It reserves a safety margin and evaluates whether the request fits.
4. It admits, defers, or rejects the request and logs the decision.
5. It compares estimated demand with actual peak memory after execution.

This is **cooperative admission control**, not a hard GPU memory reservation or enforcement mechanism. NVML alone cannot stop arbitrary external processes from allocating GPU memory. External workloads can invalidate an estimate between the check and launch.

---

## 5. Metrics and Definitions

All memory quantities should be stored in bytes internally. Convert to MiB/GiB only for display.

### 5.1 Core memory metrics

| Metric | Definition / interpretation |
|---|---|
| Allocated memory | Memory currently allocated to live PyTorch tensors, as reported by the process allocator |
| Reserved memory | Memory managed/held by the PyTorch caching allocator |
| Cached-but-unused estimate | `reserved_bytes - allocated_bytes`; interpret according to allocator semantics |
| Peak allocated memory | Maximum allocated memory observed during the measured interval |
| Peak reserved memory | Maximum reserved memory observed during the measured interval |
| Device headroom | Device free memory at the observation time |
| Peak-to-capacity ratio | Workload peak divided by the relevant device or configured budget |
| Allocation rate | Allocation/deallocation events or positive memory deltas per unit time |
| Memory growth rate | Change in observed memory over a defined time window |

Do not equate device free memory with the largest allocatable block or assume that a single process owns all device memory.

### 5.2 Fragmentation metrics

The project defines the **Free-Space Fragmentation Index (FSFI)**:

\[
FSFI = 1 - \frac{L}{F}
\]

where:

- \(L\) = largest free block in the allocator-level snapshot being analyzed.
- \(F\) = total free memory across the relevant allocator-level free blocks.

Handle \(F=0\) explicitly as undefined/not applicable rather than dividing by zero.

Interpretation: values closer to 0 indicate that free allocator space is concentrated in a large block; values closer to 1 indicate that it is distributed among smaller blocks.

**Important limitations:**

- FSFI is a project-defined metric, not an established industry-standard metric.
- Use allocator segment/block information from supported PyTorch snapshots or equivalent introspection.
- NVML totals alone cannot calculate FSFI.
- This measures allocator-level free-space distribution, not necessarily physical GPU-memory fragmentation.
- PyTorch allocator snapshots do not provide a universal map of physically contiguous GPU memory.

Additional metrics:

| Metric | Purpose |
|---|---|
| Largest free allocator block | Largest observed free block |
| Total free allocator space | Sum of free blocks in the selected allocator pool |
| Free-block count | Number of free blocks |
| Free-block size distribution | Histogram or quantiles of free block sizes |
| Internal fragmentation estimate | Report only if the allocator data supports a defensible calculation |
| External fragmentation indicator | FSFI and block distribution, explicitly scoped to allocator-level data |

Do not label internal fragmentation as measured unless the required allocation-size and block-size information is actually available.

### 5.3 Memory pressure and OOM-risk estimate

Add a pressure analyzer that uses:

- Current device free memory and configured workload budget.
- Recent memory growth rate.
- Recent and current workload peaks.
- Workload-specific historical peak measurements, when available.

The initial OOM-risk indicator should be a transparent rule-based warning, not a guaranteed prediction.

Example policy (thresholds configurable and calibrated experimentally):

- **Low:** headroom is comfortably above the configured safety margin and usage is stable.
- **Moderate:** headroom is shrinking or workload peaks are approaching the budget.
- **High:** headroom is below the safety margin, recent growth is rapid, or comparable workloads have approached/exceeded the budget.

The dashboard must label this as an estimate. Sudden allocations, CUDA workspaces, other processes, driver overhead, and external workloads can cause OOM despite a low-risk estimate.

---

## 6. Technology Stack and Environment

| Layer | Tool | Role |
|---|---|---|
| Device monitoring | `pynvml` / NVML | Device and per-process metrics |
| In-process allocation tracking | PyTorch CUDA memory APIs | Allocated/reserved state, peaks, allocator snapshots |
| Analysis | Pandas / NumPy | Time-series and metric calculations |
| Dashboard | Streamlit | Live and historical visualization |
| Testing | pytest | Unit and integration tests |
| Optional low-level interception | C/CUDA, Linux `LD_PRELOAD` | Intercept selected CUDA runtime allocation calls |

Install a local environment on each teammate's machine; do not share a virtual environment between different laptops.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install pynvml torch pandas numpy streamlit pytest
```

Confirm that the installed PyTorch build can use the GPU and is compatible with the installed driver. Record Python, PyTorch, CUDA, driver, GPU model, and OS versions in every benchmark profile.

If `pynvml` has compatibility or permission problems, test the documented `nvidia-smi` query fallback and record the source of each metric.

---

## 7. Repository Structure

```text
gpu-memory-profiler/
├── schema.py
├── collector/
│   ├── nvml_monitor.py
│   ├── torch_hooks.py
│   └── cuda_tracker/              # Optional stretch
│       ├── shim.c
│       └── Makefile
├── profiler/
│   ├── memory_profiler.py
│   ├── fragmentation.py
│   ├── peak_analysis.py
│   ├── contention.py
│   └── memory_pressure.py
├── optimizer/
│   ├── recommendations.py
│   ├── batch_optimizer.py
│   └── experiment_runner.py
├── scheduler/
│   ├── admission_control.py
│   └── memory_budget.py
├── workloads/
│   ├── memory_stress.py
│   ├── cnn_test.py
│   ├── resnet_test.py
│   └── transformer_test.py
├── evaluation/
│   ├── benchmark_runner.py
│   └── results.py
├── dashboard/
│   └── app.py
├── data/
│   └── profiles/
├── tests/
│   ├── test_schema.py
│   ├── test_fragmentation.py
│   ├── test_recommendations.py
│   ├── test_collector.py
│   ├── test_admission.py
│   └── test_memory_pressure.py
├── requirements.txt
├── README.md
└── run.py
```

---

## 8. Development Plan and Exit Criteria

Complete phases in order. Do not mark a phase complete until its exit criterion has been run and its output checked.

### Phase 0 — Environment and Feasibility

**Goal:** Verify the toolchain on each actual machine.

Tasks:
- Install dependencies in a local virtual environment.
- Confirm NVML and PyTorch can see the GPU.
- Record GPU model, driver, CUDA and library versions.

Example checks:

```bash
python -c "import pynvml; pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex(0); i=pynvml.nvmlDeviceGetMemoryInfo(h); print(i.total, i.used, i.free)"
python -c "import torch; x=torch.zeros(10_000_000, device='cuda'); print(torch.cuda.memory_stats()['allocated_bytes.all.current'])"
```

**Exit criterion:** Both commands print plausible, nonzero values without errors on each target machine.

### Phase 0.5 — Unified Event Schema

**Goal:** Define one canonical record format used by every module.

Create `schema.py` with a `MemoryEvent` dataclass containing at minimum:

```python
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class MemoryEvent:
    timestamp: float
    pid: int
    gpu_id: int
    source: str                  # nvml | torch | workload | scheduler
    event_type: str              # poll | alloc | free | peak | admission
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

    def to_row(self) -> dict:
        return asdict(self)
```

Keep schema field types and names consistent. Additional metrics may be stored in a documented metadata field or added through a versioned schema change.

**Exit criterion:** NVML and PyTorch collectors import the same schema and write rows with identical CSV columns. Add schema tests.

### Phase 1 — Baseline GPU Monitor

**Goal:** Provide continuous device-level visibility.

Tasks:
- Poll NVML at a configurable interval.
- Record total, used/free memory, utilization, timestamp and GPU identity.
- Record per-process memory where supported.

**Exit criterion:**

```bash
python collector/nvml_monitor.py --duration 10 --out data/profiles/baseline.csv
```

Produces a CSV with approximately one row per polling interval and the canonical schema.

Cross-check selected readings against `nvidia-smi`.

### Phase 2 — Allocation Tracking and Memory Pressure

**Goal:** Record in-process allocator state and derive pressure indicators.

Tasks:
- Poll `torch.cuda.memory_stats()` at a fixed interval or after a configured number of operations.
- Record allocated/reserved bytes, deltas and running peaks.
- Add headroom, peak-to-capacity ratio and memory-growth calculations.
- Add a transparent, configurable OOM-risk rule.
- Attempt allocator history/snapshot APIs only after basic polling works.

**Exit criterion:**

```bash
python collector/torch_hooks.py --workload workloads/memory_stress.py --out data/profiles/torch_events.csv
```

Produces a timeline with allocated/reserved bytes, deltas, peaks, and pressure fields. Validate risk flags against actual workload outcomes.

### Phase 3 — Fragmentation Analyzer

**Goal:** Quantify allocator-level free-space distribution.

Tasks:
- Parse supported allocator snapshot/segment information.
- Compute FSFI, largest free block, total free allocator space, free-block count and distribution.
- Handle no-free-space and unavailable-snapshot cases explicitly.
- Implement a controlled allocation/free stress workload with varying sizes.
- Compare allocator snapshots before and after cache-clearing experiments.

**Exit criterion:**

```bash
python profiler/fragmentation.py data/profiles/torch_events.csv
```

Produces a report with metric definitions and valid/invalid states. Demonstrate a reproducible change in allocator-level metrics under a controlled workload. Do not assume FSFI must always rise or that `empty_cache()` must always reduce it.

### Phase 4 — Multi-Process Contention and Admission Control

**Goal:** Demonstrate resource contention and implement cooperative workload admission.

Tasks:
- Launch two or more lightweight GPU workloads as separate processes.
- Use NVML per-process metrics to compare memory consumption.
- Record latency and throughput as concurrency changes.
- Implement a configurable memory budget and safety margin.
- Log admission, defer, or reject decisions and compare estimated with actual memory.
- Test on both teammates' GPUs if feasible.

**Exit criterion:**

```bash
python profiler/contention.py --procs 2 --out data/profiles/contention.csv
```

Produces per-PID memory time series for simultaneous processes and a contention report. Exercise admission-control decisions with both synthetic unit tests and real workloads.

A human must inspect actual measurements. Do not claim hard memory isolation or guaranteed admission safety.

### Phase 5 — Optimization and Recommendation Engine

**Goal:** Turn diagnostic results into traceable recommendations.

Tasks:
- Implement configurable rules for batch size, mixed precision, checkpointing, cache behavior and buffer reuse.
- Include the metric and threshold that triggered each recommendation.
- Add a controlled batch-size/precision experiment runner.
- Compare candidate configurations using measured peak memory, latency and throughput.
- Return the best-fitting tested configuration under a user-specified budget, without claiming a global optimum.

**Exit criterion:**

```bash
python optimizer/recommendations.py data/profiles/torch_events.csv --out data/profiles/recs.json
```

Produces readable recommendations with evidence and measured values. Unit tests cover thresholds and edge cases.

### Phase 6 — Streamlit Dashboard

**Goal:** Provide a usable live and historical view.

Dashboard sections:
- Device memory gauge and utilization.
- Allocated versus reserved memory for the instrumented process.
- Memory headroom, growth rate and OOM-risk indicator.
- Allocation rate, peak timeline and FSFI/block-distribution charts.
- Per-process memory table and contention status.
- Recommendation panel with evidence.
- Admission-control decision history.
- Baseline-versus-optimized experiment comparison.

**Exit criterion:**

```bash
streamlit run dashboard/app.py
```

The dashboard starts and visibly updates while a workload runs in another process. A human must inspect the UI and confirm that the displayed data is current and correctly labeled.

### Phase 7 — Evaluation Workloads and Benchmarking

**Goal:** Evaluate behavior using small, controlled models that fit the available hardware.

Workloads:
- Small CNN.
- Small ResNet variant.
- Two-to-four-layer Transformer.
- Memory stress workload.
- Concurrent workload pairs.

For each applicable workload:
- Test two or three batch sizes.
- Compare FP32 and supported FP16/BF16 modes.
- Include warm-up and repeated runs.
- Capture peak allocated/reserved memory, latency, throughput, FSFI where available, OOM outcomes, and recommendations.

**Exit criterion:** Produce a results table covering model × batch size × precision, with measured values and repeatability statistics. Include baseline-versus-optimized comparisons.

### Phase 8 — Stretch: CUDA/C++ Allocation Interceptor

**Goal:** Add a framework-agnostic low-level allocation log.

Tasks:
- On Linux, prototype a shared-library shim intercepting selected CUDA Runtime API calls such as `cudaMalloc` and `cudaFree`.
- Forward calls to the real functions safely and log calls.
- Validate behavior with a small CUDA program before attempting integration.

**Exit criterion:** A test CUDA program run with the shim produces a verified allocation log.

This phase is Linux-specific and may require WSL2 or a Linux environment. It is optional and must not block the core project. Clearly document API coverage and limitations; do not imply every CUDA allocation path is intercepted.

### Phase 9 — Testing, Report and Viva

Tasks:
- Unit-test schema, FSFI calculations, pressure rules, recommendations and admission decisions.
- Integration-test stress workloads and collector output.
- Cross-check NVML data with `nvidia-smi`.
- Prepare architecture diagrams, OS-concept mapping and benchmark tables.
- Rehearse the complete demo at least twice.

**Exit criterion:**

```bash
pytest tests/
```

All automated tests pass, and the end-to-end demo is manually verified.

---

## 9. Testing and Validation Strategy

### Unit tests
Use synthetic data for pure calculations:
- FSFI, including zero-free-memory and missing-data cases.
- Memory headroom and pressure thresholds.
- Recommendation rules.
- Admission decisions at, below, and above the configured budget.

### Integration tests
- Run memory stress workloads and capture allocator data.
- Confirm event schema consistency across collectors.
- Verify that pressure indicators respond to controlled usage changes.
- Exercise two-process contention and admission-control flows.

### Manual validation
- Compare NVML readings with `nvidia-smi` at approximately the same time.
- Check dashboard values against raw profile files.
- Inspect real GPU memory behavior on both target machines.
- Record software versions and hardware details for reproducibility.

---

## 10. Evaluation Plan

| Experiment | Independent variable | Outcomes |
|---|---|---|
| Memory stress | Allocation size/order and release pattern | FSFI, block distribution, peaks |
| Batch-size sweep | Batch size | Peak memory, throughput, latency, OOM |
| Precision comparison | FP32 vs supported mixed precision | Memory, throughput, numerical deviation |
| Contention | Number/type of concurrent workloads | Per-PID memory, latency, throughput |
| Admission control | Requested memory and current load | Admit/defer/reject, wait time, estimation error |
| Optimization comparison | Baseline vs recommended configuration | Memory reduction, throughput change, reliability |

For repeatability:
- Run each benchmark multiple times and report mean and standard deviation.
- Record warm-up policy, run duration, model, batch size, precision, GPU, driver, CUDA and PyTorch versions.
- Separate measured results from estimates and recommendations.
- Report failed runs and OOM outcomes rather than omitting them.

---

## 11. Literature Review Positioning

Use the source presentation's references to position the project as a smaller, educational implementation:

- GPU memory heat-map profiling and allocation-characteristic analysis motivate allocation-pattern visibility.
- Memory handover and kernel-space GPU sharing research provide production-scale context for the contention experiments.
- Backpropagation memory optimization and virtual-memory-based defragmentation research provide context for the toolkit's simpler optimization heuristics.

Do not claim equivalent scale or capabilities to these research systems. Describe what each prior work addresses and what the toolkit demonstrates at a smaller scope. Verify the bibliographic details of all references before final submission.

---

## 12. Suggested Timeline

An eight-week schedule, adjustable to the actual course deadline:

| Week | Focus |
|---|---|
| 1 | Phase 0–0.5: environment, feasibility, schema |
| 2 | Phase 1–2: monitoring, allocation tracking, pressure |
| 3 | Phase 3: fragmentation analysis and stress workloads |
| 4 | Phase 4: contention and admission-control prototype |
| 5 | Phase 5: recommendations and adaptive experiments |
| 6 | Phase 6: dashboard and visual validation |
| 7 | Phase 7: benchmarks and results collection |
| 8 | Phase 8 if feasible; Phase 9 testing, report and viva |

**Deadline fallback:** Phases 0–6 form the core demo. Phase 7 strengthens the evidence. Phase 8 is optional.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| NVML cannot provide individual allocation events | Use in-process PyTorch allocator introspection for instrumented workloads |
| PyTorch APIs differ by version | Start with stable polling; test optional history/snapshot APIs on installed versions |
| FSFI is mistaken for physical fragmentation | Label it allocator-level and document the data source |
| OOM risk estimate is inaccurate | Make it a transparent heuristic and validate against actual runs |
| External processes invalidate admission estimates | Use a safety margin; describe admission control as cooperative and best-effort |
| Large models produce uninformative or unstable demos | Use small, controlled workloads on 6GB GPUs |
| `empty_cache()` does not improve fragmentation | Measure outcomes; do not assume a guaranteed direction |
| Multi-process runs destabilize the GPU | Start with two lightweight processes and provide a safe shutdown procedure |
| Shared virtual environment breaks across machines | Install a local environment on each machine; share code and profile data only |
| Windows blocks the `LD_PRELOAD` phase | Keep interception optional; use Linux/WSL2 if available |
| Time runs out | Finish the core monitoring, profiling, recommendation, and dashboard workflow first |

---

## 14. Viva Preparation

Be prepared to explain which module implements each OS concept and which metric demonstrates it.

### Likely questions

**How is this different from `nvidia-smi`?**  
`nvidia-smi` provides device/process snapshots and utilization data. The toolkit adds instrumented allocator history, allocator-level fragmentation analysis, workload experiments, and recommendations derived from collected measurements.

**Why can fragmentation occur if the driver manages GPU memory?**  
Framework caching allocators retain and reuse blocks to reduce allocation overhead. The resulting allocator pool can contain free space distributed across blocks of different sizes. Explain that this is allocator-level behavior and not automatically proof of physical-memory fragmentation.

**Can the toolkit guarantee that a workload will not OOM?**  
No. The initial pressure indicator is a heuristic, and admission control is cooperative. External allocations, sudden workspace requirements, and runtime overhead can invalidate estimates.

**Does `torch.cuda.empty_cache()` free live tensors?**  
No. It can release unused cached blocks held by the allocator, subject to allocator behavior. It does not free live tensors and does not guarantee that fragmentation will disappear.

**How do you know the optimizer works?**  
Compare baseline and optimized configurations under controlled, repeated workloads using peak memory, throughput, latency, and OOM outcomes.

**What is your research contribution?**  
An integrated, teachable framework that connects allocator-level GPU memory diagnosis with evidence-backed optimization experiments and cooperative resource admission on resource-constrained GPUs.

---

## 15. Future Scope

After the course deadline:

- Package the collector, profiler and optimizer into a standalone application.
- Bundle the Streamlit server with a launcher or wrap it in a native window using a suitable webview framework.
- Explore more advanced workload scheduling and stronger isolation mechanisms.
- Evaluate additional GPU architectures and allocator implementations.
- Explore learned prediction models only after enough reliable, labeled workload data is collected.

These are future directions, not requirements for the course MVP.

---

## Final Scope Statement

The deliverable is a GPU memory profiling and optimization toolkit that monitors device and instrumented-process memory, analyzes allocator-level behavior, identifies memory pressure and contention, recommends tested configuration changes, and demonstrates cooperative admission control through a live dashboard.

The core project is Phases 0–6, supported by controlled evaluation and testing. The CUDA/C++ interceptor remains an optional stretch goal.
