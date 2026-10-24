# GPU Memory Profiling & Optimization Toolkit
## Handover Document: Completed Work, Architecture, and Teammate Blueprint

**Target Audience:** Teammates (Person 2, Person 3) and their respective AI engineering assistants.  
**Reference Authority:** `GPU_Memory_Profiler_Development_Plan_Revised.md` & `work_division.md`  
**Current Milestone Completed:** Phases 0, 0.5, 1, 2, 3, and 5.

---

## 1. Executive Summary

This project applies core **Operating System resource management principles** to GPU memory. Rather than treating GPU memory as a black box where out-of-memory (OOM) errors are unpredictable crashes, our framework treats GPU memory as a managed, auditable, and optimizable shared system resource.

As of this handover, **Person 1 (RTX 4050 6GB)** has completed the core telemetry collection layer, in-process allocation tracking, allocator fragmentation engine (FSFI), memory pressure analyzer, and the rule-based optimization engine (**Phases 0, 0.5, 1, 2, 3, and 5**). All modules adhere strictly to the unified `MemoryEvent` schema, pass automated pytest suites, and are accompanied by empirical test data.

This document explains **what was built, how it was built, where files live, and how Person 2 and Person 3 should build their remaining modules to achieve seamless integration.**

---

## 2. Global Architecture & Communication Model

```text
                                 [ GPU Workloads ]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
       [ NVML Collector ]                           [ PyTorch Allocator Hooks ]
  (collector/nvml_monitor.py)                        (collector/torch_hooks.py)
   Device-wide & per-PID stats                        In-process memory_stats()
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                            [ Canonical MemoryEvent ]
                                  (schema.py)
                                         │
       ┌──────────────────┬──────────────┴───────────────┬──────────────────┐
       ▼                  ▼                              ▼                  ▼
[ Memory Pressure ] [ Fragmentation ]          [ Contention & Adm. ] [ Optimization ]
 (profiler/          (profiler/                  (controller/         (optimizer/
  memory_pressure)    fragmentation.py)           admission_control)   recommendations)
  Headroom / OOM      FSFI / Snapshot Parser     (PERSON 2)           Traceable Rules
       │                  │                              │                  │
       └──────────────────┴──────────────┬───────────────┴──────────────────┘
                                         ▼
                               [ Streamlit Dashboard ]
                                  (dashboard/app.py)
                                     (PERSON 3)
```

### The Communication Standard: `schema.py`
All modules communicate through a single, canonical schema defined in `schema.py`. Whether data originates from NVML, PyTorch hooks, or the admission controller, every event is serialized to or read from this structure:

```python
@dataclass
class MemoryEvent:
    timestamp: float              # Unix timestamp in seconds
    pid: int                      # Process ID
    gpu_id: int                   # Physical GPU index (0, 1, etc.)
    source: str                   # "nvml" | "torch" | "workload" | "scheduler"
    event_type: str               # "poll" | "alloc" | "free" | "peak" | "snapshot" | "admission"
    
    # Device-wide metrics (NVML)
    total_memory: Optional[int] = None
    free_memory: Optional[int] = None
    process_memory: Optional[int] = None
    
    # In-process allocator metrics (PyTorch)
    allocated_bytes: Optional[int] = None
    reserved_bytes: Optional[int] = None
    allocation_delta: Optional[int] = None
    peak_memory: Optional[int] = None
    
    # Workload metadata
    workload_name: Optional[str] = None
    batch_size: Optional[int] = None
    precision: Optional[str] = None
    metadata_json: Optional[str] = None
    snapshot_path: Optional[str] = None

    def to_row(self) -> dict: ...
```

---

## 3. What Was Built & How It Was Implemented (Phases 0–5)

### Phase 0 & 0.5: Environment & Unified Event Schema
* **Files:** `schema.py`
* **What it does:** Standardizes cross-process and in-process memory event records.
* **How it was implemented:** Uses a lightweight Python dataclass with helper methods `to_row()` for CSV serialization and `asdict` mapping. All memory quantities are represented in integer **bytes** internally and converted to MiB/GiB only for display.

---

### Phase 1: Baseline Device GPU Monitor
* **Files:** `collector/nvml_monitor.py`
* **Artifact Generated:** `data/profiles/baseline.csv`
* **What it does:** Provides continuous, out-of-process device-level visibility.
* **How it was implemented:**
  1. Initializes NVML using `pynvml.nvmlInit()`.
  2. Acquires GPU handle `pynvml.nvmlDeviceGetHandleByIndex(0)`.
  3. Periodically samples `nvmlDeviceGetMemoryInfo(handle)` (total, used, free memory) and records timestamped `MemoryEvent` entries.
  4. Flushes rows to CSV according to the canonical schema.
* **Verification Command:**
  ```bash
  python collector/nvml_monitor.py --duration 10 --out data/profiles/baseline.csv
  ```

---

### Phase 2: In-Process Allocation Tracking & Memory Pressure
* **Files:** `collector/torch_hooks.py`, `profiler/memory_pressure.py`, `tests/test_memory_pressure.py`
* **Artifact Generated:** `data/profiles/torch_events.csv`
* **What it does:** Tracks dynamic allocator state within the target Python workload and derives memory pressure, headroom, and heuristic OOM risk.
* **How it was implemented:**
  1. **Background Polling:** `TorchHookMonitor` runs a daemon thread polling `torch.cuda.memory_stats()` every 50ms while executing the user workload in the main thread.
  2. **Allocation Deltas & Peaks:** Records `allocated_bytes.all.current`, `reserved_bytes.all.current`, running peaks, and positive/negative allocation deltas.
  3. **Pressure Analyzer:** `profiler/memory_pressure.py` computes:
     * **Headroom:** `Budget - Allocated_Bytes`.
     * **Headroom Ratio:** `Headroom / Budget`.
     * **Growth Rate:** Memory delta divided by time delta ($\Delta \text{Bytes} / \Delta t$).
     * **Transparent OOM Heuristic:** Classifies risk into **Low**, **Moderate**, or **High** based on whether headroom drops below 25% (Moderate) or 10% (High), or if growth rate exceeds 50 MB/s.
  4. **Testing:** 11 unit tests in `tests/test_memory_pressure.py` covering edge cases, empty CSVs, missing columns, zero budgets, and growth escalations (all passing).
* **Verification Command:**
  ```bash
  pytest tests/test_memory_pressure.py
  python profiler/memory_pressure.py data/profiles/torch_events.csv --budget 6442450944
  ```

---

### Phase 3: Allocator-Level Fragmentation Analyzer
* **Files:** `workloads/memory_fragmentation.py`, `profiler/fragmentation.py`
* **Artifacts Generated:** `data/profiles/snapshots/*.json`
* **What it does:** Measures allocator-level free-space distribution and evaluates the impact of cache clearing.
* **How it was implemented:**
  1. **Controlled Fragmentation Workload:** `workloads/memory_fragmentation.py` allocates 20 tensors (10 MB each = 200 MB total) in a contiguous segment, then deletes alternating tensors to punch inactive "holes" in the allocator pool.
  2. **Snapshot Captures:** Triggers `torch.cuda.memory_snapshot()` and writes the raw segment graph to JSON before and after calling `torch.cuda.empty_cache()`.
  3. **FSFI Calculation:** `profiler/fragmentation.py` iterates through all segments and inactive blocks, computing the project-defined **Free-Space Fragmentation Index (FSFI)**:
     $$FSFI = 1 - \frac{\text{Largest Free Block}}{\text{Total Free Memory Across Inactive Blocks}}$$
  4. **Experimental Proof:** 
     * Pre-cache clear: $\text{FSFI} = 0.9000$ (high fragmentation; 100 MB free space split into ten 10 MB blocks).
     * Post-cache clear (`empty_cache`): $\text{FSFI} = 0.0000$ (unused segments released back to driver).
     * **Phase 3 exit criterion fully met.**
* **Verification Command:**
  ```bash
  python profiler/fragmentation.py data/profiles/torch_events.csv
  ```

---

### Phase 5: Optimization & Recommendation Engine
* **Files:** `optimizer/recommendations.py`, `optimizer/batch_optimizer.py`, `tests/test_recommendations.py`
* **Artifact Generated:** `data/profiles/recs.json`
* **What it does:** Turns raw diagnostic telemetry into traceable, evidence-backed optimization recommendations.
* **How it was implemented:**
  1. **Recommendation Rules (`optimizer/recommendations.py`):**
     * **Batch Size Reduction (`OPT-BATCH-001`):** Triggers if peak memory exceeds $1 - \text{safety\_margin}$ (e.g. > 85% of budget).
     * **Batch Size Increase (`OPT-BATCH-002`):** Triggers if peak memory is under 35% of budget, advising batch scaling for higher hardware saturation.
     * **Mixed Precision (`OPT-PREC-001`):** Triggers if FP32 is used with high memory consumption, recommending `torch.cuda.amp.autocast`.
     * **Cache Management (`OPT-CACHE-001`):** Triggers when reserved memory exceeds allocated memory by > 30% and > 100 MB, suggesting `torch.cuda.empty_cache()`.
     * **Gradient Checkpointing (`OPT-CKPT-001`):** Recommends activation recomputation when peak memory approaches 65% of budget.
     * **Buffer Reuse (`OPT-REUSE-001`):** Detects allocation churn (frequent alloc/free cycles) and advises pre-allocated tensor buffer pools.
     * **Traceability Requirement:** Every recommendation explicitly cites `rule_id`, `triggering_metric`, `measured_value`, `threshold`, actionable `recommendation`, and operational `caveat`.
  2. **Batch Optimizer (`optimizer/batch_optimizer.py`):**
     * Models linear memory scaling: $\text{Peak Memory}(B) = \text{Base Memory} + B \times \text{Activation Per Sample}$.
     * Evaluates candidate batch sizes and tags each as `VIABLE`, `RISKY`, or `EXCEEDS_BUDGET`.
     * Selects the highest-throughput viable candidate under the user budget without claiming global optimality.
  3. **Testing:** 10 unit tests in `tests/test_recommendations.py` covering all rule activations, suppression of rules (e.g. FP16 already active), and batch optimizer classifications (all passing).
* **Verification Command:**
  ```bash
  pytest tests/test_recommendations.py
  python optimizer/recommendations.py data/profiles/torch_events.csv --out data/profiles/recs.json
  ```

---

## 4. Current Repository File Structure

```text
Project/
├── schema.py                             # Unified MemoryEvent dataclass
├── collector/
│   ├── nvml_monitor.py                   # Phase 1: Out-of-process device monitor
│   └── torch_hooks.py                    # Phase 2: In-process allocator hook & snapshot logger
├── profiler/
│   ├── memory_pressure.py                # Phase 2: Headroom, growth rate, OOM risk
│   └── fragmentation.py                  # Phase 3: Allocator segment parser & FSFI calculator
├── optimizer/
│   ├── recommendations.py                # Phase 5: Evidence-backed recommendation engine
│   └── batch_optimizer.py                # Phase 5: Batch scaling model & candidate evaluator
├── workloads/
│   ├── memory_stress.py                  # Allocation stress test workload
│   └── memory_fragmentation.py           # Phase 3: Alternating tensor fragmentation test
├── tests/
│   ├── test_memory_pressure.py           # 11 unit tests for pressure & OOM rules
│   └── test_recommendations.py           # 10 unit tests for optimization rules
├── data/
│   └── profiles/
│       ├── baseline.csv                  # Verified Phase 1 output
│       ├── torch_events.csv              # Verified Phase 2 output
│       ├── recs.json                     # Verified Phase 5 output
│       └── snapshots/                    # Verified Phase 3 JSON snapshot dumps
├── GPU_Memory_Profiler_Development_Plan_Revised.md # Primary project authority
├── work_division.md                      # Team role definitions
└── work_done.md                          # This handover document
```

---

## 5. Teammate Blueprint: Person 2 (RTX 3050 6GB)

### Primary Missions: Phase 4 & Phase 7
Person 2 has the second dedicated GPU and is responsible for **multi-process contention, cooperative admission control, and benchmark evaluation**.

### What Person 2 Must Build:

#### 1. `controller/admission_control.py` (Phase 4)
* **OS Concept:** Cooperative Admission Control & Resource Reservation.
* **Mechanism:**
  ```text
  Workload Submission (Request: 2.0 GB)
                 │
                 ▼
  Current Device Used VRAM (NVML): 3.5 GB
  Device Budget: 6.0 GB | Safety Margin: 0.5 GB
  Available Safe Headroom: 6.0 - 3.5 - 0.5 = 2.0 GB
                 │
         ┌───────┴───────┐
         ▼               ▼
  Request <= Headroom?   Request > Headroom?
         │                       │
      [ ADMIT ]            [ DEFER / REJECT ]
  ```
* **Inputs & Integration:**
  * Import `MemoryEvent` from `schema.py`.
  * Read live free memory from `pynvml` or accept declared requirements.
  * Emit an admission decision event:
    `event_type="admission", metadata_json='{"decision": "ADMIT", "requested": 2147483648, "available": 2684354560}'`.

#### 2. `profiler/contention.py` (Phase 4)
* **OS Concept:** Resource Contention & Concurrency Scheduling.
* **Mechanism:**
  * Launch 2 to 3 concurrent Python subprocesses running workloads (e.g. CNN + ResNet).
  * Use `pynvml.nvmlDeviceGetComputeRunningProcesses(handle)` to record per-PID VRAM consumption over time.
  * Measure degradation in workload latency as concurrency increases ($N=1$ vs $N=2$ vs $N=3$).
* **Exit Criterion Command:**
  ```bash
  python profiler/contention.py --procs 2 --out data/profiles/contention.csv
  ```

#### 3. `workloads/` & `evaluation/benchmark_runner.py` (Phase 7)
* Build small model workloads: `workloads/cnn_test.py`, `workloads/resnet_test.py`, `workloads/transformer_test.py`.
* Implement a benchmark runner to sweep `model × batch_size × precision (FP32 vs FP16)`.
* Save results comparing baseline vs optimized configurations.

---

## 6. Teammate Blueprint: Person 3 (CPU-Only Laptop)

### Primary Missions: Phase 6 & Phase 9
Person 3 does **not** need a GPU to develop! All modules output standard CSV and JSON files, so Person 3 can build the data pipeline, dashboard, and testing suite purely on CPU using recorded or mock profile files.

### What Person 3 Must Build:

#### 1. `dashboard/app.py` (Phase 6 — Streamlit)
Person 3 will create a multi-tab Streamlit dashboard:
* **Tab 1: Device Overview (NVML):**
  * Reads `data/profiles/baseline.csv`.
  * Displays device VRAM gauge, free vs used memory chart over time.
* **Tab 2: In-Process Allocator & Pressure:**
  * Reads `data/profiles/torch_events.csv`.
  * Graphs Allocated vs Reserved bytes over time.
  * Displays Memory Headroom, Growth Rate, and heuristic OOM Risk Badge (Green=Low, Orange=Moderate, Red=High).
* **Tab 3: Allocator Fragmentation (FSFI):**
  * Displays FSFI metric, largest free block vs total free memory.
  * Visualizes allocator segment/block distributions from `data/profiles/snapshots/*.json`.
* **Tab 4: Multi-Process Contention & Admission Control:**
  * Displays per-PID memory stacked area chart from Person 2's `contention.csv`.
  * Table of admission decisions (Admit, Defer, Reject).
* **Tab 5: Optimization & Recommendations:**
  * Reads `data/profiles/recs.json`.
  * Renders actionable cards showing triggering metric, threshold, recommendation text, and caveats.
* **Exit Criterion Command:**
  ```bash
  streamlit run dashboard/app.py
  ```

#### 2. Mock Data Generator (`data/mock_generator.py`) (Phase 6 / CPU Support)
* Person 3 can write a simple script that generates synthetic `MemoryEvent` CSV rows to test all Streamlit charts without waiting for real GPU runs.

#### 3. Test Integration & Final Report (Phase 9)
* Add unit tests for `schema.py` (`tests/test_schema.py`) and admission control (`tests/test_admission.py`).
* Consolidate the test suite so `pytest tests/` runs all tests in < 1 second.
* Compile final report graphs and presentation slides.

---

## 7. Strict Operating & Safety Guidelines for Teammates & AI Agents

1. **Local `.venv` Exclusivity:**
   Each teammate must maintain their own `.venv`. Never commit or share `.venv` across Git.
2. **Hardware Command Safety (CRITICAL):**
   * **Never allow an AI agent to execute GPU allocation scripts, stress tests, or hardware polling autonomously.**
   * Safe for AI: file inspection, static analysis, Python syntax compilation, and unit tests that do not initialize CUDA.
   * Unsafe: Any script initializing `torch.cuda` or NVML hardware polling must be run manually by the teammate on their local terminal.
3. **Internal Bytes Representation:**
   Keep all memory measurements in raw integer bytes in CSV/JSON. Only convert to MB/GB in log printouts and dashboard visualizations.
4. **Code Quality & Docstring Standard:**
   * Every file must include a top-level docstring stating: 1. What the file does, 2. Overall design, 3. Elements in the file, 4. Final output.
   * Every class and function must include detailed docstrings explaining purpose, operation, concepts used, callers, and inputs/outputs.

---

## 8. Summary Table of Completed Deliverables

| Module | File Path | Phase | Status | Unit Tests Passing |
|---|---|---|---|---|
| Unified Schema | `schema.py` | 0.5 | **Done** | Serializer & Dataclass verified |
| Baseline Monitor | `collector/nvml_monitor.py` | 1 | **Done** | `baseline.csv` generated |
| Allocator Hooks | `collector/torch_hooks.py` | 2 | **Done** | `torch_events.csv` generated |
| Memory Pressure | `profiler/memory_pressure.py` | 2 | **Done** | 11/11 tests passing |
| Fragmentation | `profiler/fragmentation.py` | 3 | **Done** | FSFI evaluated (0.9000 → 0.0000) |
| Recommendation Engine | `optimizer/recommendations.py` | 5 | **Done** | `recs.json` generated |
| Batch Optimizer | `optimizer/batch_optimizer.py` | 5 | **Done** | 10/10 tests passing |
| Handover Blueprint | `work_done.md` | - | **Done** | Ready for team distribution |


---

## 9. Person 3 Completed Deliverables & Final System Status

### Milestone Completed: Phases 0.5, 6, and 9 (Person 3 — CPU Laptop)

Person 3 has completed all remaining responsibilities without disturbing any of Person 1 or Person 2's code:

1. **Unified Data Processing Pipeline (`pipeline/`):**
   - `pipeline/normalize.py`: Byte unit conversion (MB/GB/human-readable), timestamp normalization, derived metrics (`device_used_bytes`, `cached_unused_bytes`, `allocated_mb`, `reserved_mb`), and DataFrame generation.
   - `pipeline/validate.py`: Schema validation, non-negative bounds checking, cross-field integrity (`allocated <= reserved`), and non-crashing error reporting via `ValidationResult`.
   - `pipeline/aggregate.py`: Profile-level summary calculation, sliding window growth rates, per-PID grouping, downsampling (`resample_timeseries`), and stream joining (`join_device_and_process_streams`).
   - `pipeline/__init__.py`: Package export interface.

2. **Synthetic Mock Data Generator (`data/mock_generator.py`):**
   - Generates 8 realistic synthetic datasets in `data/mock/` (`mock_baseline.csv`, `mock_torch_events.csv`, `mock_contention.csv`, `mock_admission_log.csv`, `mock_benchmark_results.csv`, `mock_recs.json`, and allocator JSON snapshots).
   - Enables full end-to-end CPU simulation, testing, and dashboard verification without GPU hardware.

3. **Streamlit Interactive Web Dashboard (`dashboard/app.py` & `dashboard/components.py`):**
   - Multi-tab application covering all 7 development plan areas:
     - **Tab 1 (Device Overview):** Live VRAM gauges, used vs free memory timelines, and GPU utilization.
     - **Tab 2 (Allocator & Pressure):** In-process allocated vs reserved memory timeline, cached-unused pool tracking, headroom, and heuristic OOM risk badge.
     - **Tab 3 (Allocator Fragmentation):** FSFI calculation ($FSFI = 1 - L/F$), largest free block metric, and allocator segment block distribution histograms.
     - **Tab 4 (Contention & Admission Control):** Multi-PID stacked memory consumption charts, process statistics table, and an interactive cooperative admission control simulator.
     - **Tab 5 (Optimization & Rules):** Evidence-backed recommendation cards with rule IDs, triggers, and caveats, plus an interactive linear batch size optimizer.
     - **Tab 6 (Benchmark Matrix):** Baseline vs. optimized configuration comparisons, FP32 vs. FP16 memory/runtime sweeps, and throughput analysis.
     - **Tab 7 (OS Concepts & Architecture):** Comprehensive table mapping toolkit metrics to OS memory management principles.

4. **Automated Unit Testing & Orchestration (`tests/` & `run_tests.py`):**
   - `tests/test_schema.py`: 6 unit tests covering `schema.py` dataclass defaults, optional fields, `to_row` dictionary serialization, type fidelity, JSON metadata, and CSV round-tripping.
   - `tests/test_pipeline.py`: 11 unit tests covering byte conversion, normalization, schema validation, negative memory constraints, allocator bounds, summary aggregation, polymorphic buffer/stream loading, and peak-preserving decimation.
   - `run_tests.py`: CPU-safe test runner running all test suites.
   - **Total Test Suite Status:** **193 passed out of 193 unit tests** in ~2.3 seconds.

5. **Unified Toolkit CLI (`run.py`):**
   - Root command-line interface supporting `dashboard`, `test`, `mock`, `pipeline`, `pressure`, `fragmentation`, `recommendations`, and `contention`.
