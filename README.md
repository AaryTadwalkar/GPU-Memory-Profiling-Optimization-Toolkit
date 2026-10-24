# GPU Memory Profiling & Optimization Toolkit

An OS-oriented framework for runtime GPU memory analysis, allocator introspection, resource contention, and measurement-driven optimization recommendations on resource-constrained GPUs.

---

## 👥 Team & Hardware Responsibilities

| Person | Machine | Hardware | Modules & Ownership | Phases Completed |
|---|---|---|---|---|
| **Aary Tadwalkar** (Person 1) | GPU Machine 1 | NVIDIA RTX 4050 (6GB) | Device monitoring, PyTorch allocator tracking, FSFI fragmentation engine, OOM risk heuristics, and optimization engine | Phases 0, 1, 2, 3, 5 |
| **Vishal Arkalwar** (Person 2) | GPU Machine 2 | NVIDIA RTX 3050 (6GB) | Multi-process contention profiler, cooperative admission controller, and benchmark evaluation suite | Phases 4, 7 |
| **Avni Raich** (Person 3) | CPU Machine | CPU-Only Laptop | Unified data pipeline (normalize, validate, aggregate), mock data generator, Streamlit interactive dashboard, automated test consolidation, and unified CLI | Phases 0.5, 6, 9 |

---

## 🚀 Quickstart

### 1. Set Up Virtual Environment
Each teammate maintains a local virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

> **Note on PyTorch (GPUs vs CPU):**
> * On GPU machines (RTX 3050 & RTX 4050), install PyTorch with CUDA support:  
>   `pip install torch --index-url https://download.pytorch.org/whl/cu121`
> * On CPU machines (Person 3), standard `pip install torch` is sufficient.

---

### 2. Run Automated Test Suite
Run all 191 unit tests across all project phases without touching hardware:

```bash
# Via pytest
pytest tests/

# Or via the test runner
python run_tests.py
```

All 191 unit tests run in ~2.5 seconds on CPU.

---

### 3. Launch the Interactive Streamlit Dashboard (Phase 6)
Start the multi-tab interactive web dashboard:

```bash
python run.py dashboard
# or
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501` featuring:
1. **Device Overview (NVML):** Live device VRAM gauges, used vs free memory timelines, and hardware utilization.
2. **Allocator & Pressure:** In-process allocated vs reserved memory timeline, cached-unused pool tracking, headroom, and heuristic OOM risk badge.
3. **Allocator Fragmentation (FSFI):** Free-Space Fragmentation Index calculation, largest free block metric, and allocator segment block distribution histograms.
4. **Contention & Admission Control:** Multi-PID stacked memory consumption charts, process statistics table, and an interactive cooperative admission control simulator.
5. **Optimization & Rules:** Evidence-backed recommendation cards with rule IDs, triggers, and caveats, plus an interactive linear batch size optimizer.
6. **Benchmark Matrix:** Baseline vs. optimized configuration comparisons, FP32 vs. FP16 memory/runtime sweeps, and throughput analysis.
7. **OS Concepts & Architecture:** Comprehensive table mapping toolkit metrics to OS memory management principles.

---

### 4. Unified Command-Line Interface (`run.py`)
Drive all toolkit features from one unified root command:

```bash
# Generate synthetic mock data for CPU testing and simulation
python run.py mock --out-dir data/mock

# Audit, normalize, and summarize any telemetry CSV
python run.py pipeline data/profiles/torch_events.csv

# Run memory pressure and heuristic OOM-risk analysis
python run.py pressure data/profiles/torch_events.csv --budget 6442450944

# Run allocator FSFI fragmentation analysis
python run.py fragmentation data/profiles/torch_events.csv

# Generate traceable optimization recommendations
python run.py recommendations data/profiles/torch_events.csv --budget 6442450944

# Analyze multi-process GPU resource contention
python run.py contention data/profiles/contention.csv
```

---

## 🏛️ OS Concepts Demonstrated

| OS Concept | Project Application | Evidence / Metric |
|---|---|---|
| **Memory Management** | GPU allocation, deallocation, and caching lifecycle | Allocated vs Reserved timeline; Peak tracking |
| **Allocation Strategies** | CUDA and PyTorch caching allocator block reuse | Segment block snapshots; Cache clear deltas |
| **Fragmentation** | Distribution of inactive free allocator blocks | **Free-Space Fragmentation Index (FSFI)** |
| **Resource Contention** | Concurrent multi-process GPU execution | Per-PID memory tracking; Concurrency sweeps |
| **Cooperative Scheduling** | Workload budget reservation before launch | **Admission Control Decisions (Admit/Defer/Reject)** |
| **Resource Safety** | Dynamic headroom and rapid allocation growth alerts | **Transparent Heuristic OOM Risk Badge** |
| **Optimization Heuristics**| Traceable configuration recommendations | Batch scaling model, Mixed precision AMP |

---

## 📁 Repository Structure

```text
GPU-Memory-Profiling-Optimization-Toolkit/
├── requirements.txt                      # Project dependencies
├── schema.py                             # Unified MemoryEvent canonical dataclass
├── run.py                                # Unified CLI entry point
├── run_tests.py                          # Automated test suite orchestrator
├── collector/                            # Telemetry collectors
│   ├── nvml_monitor.py                   # Phase 1: Device-level out-of-process monitor (NVML)
│   └── torch_hooks.py                    # Phase 2: In-process PyTorch allocator hook & snapshot logger
├── profiler/                             # Diagnostic engines
│   ├── memory_pressure.py                # Phase 2: Headroom, growth rate & heuristic OOM risk
│   ├── fragmentation.py                  # Phase 3: Free-Space Fragmentation Index (FSFI)
│   └── contention.py                     # Phase 4: Multi-process resource contention analyzer
├── controller/                           # Resource control
│   └── admission_control.py              # Phase 4: Cooperative GPU workload admission controller
├── optimizer/                            # Optimization engine
│   ├── recommendations.py                # Phase 5: Traceable rule-based recommendation generator
│   └── batch_optimizer.py                # Phase 5: Batch scaling model & candidate classifier
├── pipeline/                             # Phase 6: Unified data processing pipeline (Person 3)
│   ├── normalize.py                      # Byte scaling, elapsed time & dataframe normalization
│   ├── validate.py                       # Schema audit & domain integrity validation
│   └── aggregate.py                      # Peak aggregation, per-PID breakdown & downsampling
├── dashboard/                            # Phase 6: Interactive Streamlit web dashboard (Person 3)
│   ├── app.py                            # Multi-tab Streamlit dashboard application
│   └── components.py                     # Reusable Altair charts, KPI cards & risk badges
├── experiments/                          # Phase 7: Benchmark evaluation runner
│   ├── schema.py                         # Benchmark result data model & persistence
│   └── runner.py                         # Controlled and baseline experiment orchestrator
├── workloads/                            # Synthetic & diagnostic workloads
│   ├── memory_stress.py                  # Allocation stress testing
│   └── memory_fragmentation.py           # Controlled alternating tensor fragmentation test
├── tests/                                # Automated unit test suites (191 tests)
│   ├── test_schema.py                    # Phase 0.5: Schema & serialization tests
│   ├── test_memory_pressure.py           # Phase 2: Pressure & OOM heuristics tests
│   ├── test_admission_control.py         # Phase 4: Admission controller tests
│   ├── test_contention.py                # Phase 4: Contention analyzer tests
│   ├── test_recommendations.py           # Phase 5: Optimization rules & batch optimizer tests
│   ├── test_pipeline.py                  # Phase 6: Normalization, validation & aggregation tests
│   └── test_experiment_results.py        # Phase 7: Benchmark result schema tests
├── data/                                 # Telemetry profiles & snapshots
│   ├── profiles/                         # Real recorded profiles from GPU runs
│   ├── mock/                             # Synthetic datasets for CPU testing & simulation
│   └── mock_generator.py                 # Synthetic data generator script
├── work_done.md                          # Handover document & technical blueprint
├── work_division.md                      # Team responsibilities & module ownership
└── GPU_Memory_Profiler_Development_Plan_Revised.md # Primary development plan
```

---

## 📖 Handover & Documentation
* For an exhaustive technical breakdown of what was built and the architecture blueprint, refer to [`work_done.md`](work_done.md).
* For the full phase-by-phase course specification and research positioning, refer to [`GPU_Memory_Profiler_Development_Plan_Revised.md`](GPU_Memory_Profiler_Development_Plan_Revised.md).
* For team roles and division details, refer to [`work_division.md`](work_division.md).
