# GPU Memory Profiling & Optimization Toolkit

An OS-oriented framework for runtime GPU memory analysis, allocator introspection, resource contention, and measurement-driven optimization recommendations on resource-constrained GPUs.

---

## 👥 Team & Hardware
* **Vishal Arkalwar** — RTX 3050 (6GB)
* **Avni Raich** — CPU-only Laptop
* **Aary Tadwalkar** — RTX 4050 (6GB)

---

## 🚀 Quickstart

### 1. Set Up Virtual Environment
Each teammate must maintain a local virtual environment (never commit `.venv` to Git):

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
> * On CPU machines, standard `pip install torch` is sufficient.

### 2. Run Automated Tests
Verify installation without touching hardware:

```bash
pytest tests/
```

### 3. Generate Diagnostics & Recommendations
Run static analysis on recorded telemetry profiles:

```bash
# Memory pressure & OOM-risk analysis
python profiler/memory_pressure.py data/profiles/torch_events.csv --budget 6442450944

# Allocator fragmentation (FSFI) analysis
python profiler/fragmentation.py data/profiles/torch_events.csv

# Optimization recommendations report
python optimizer/recommendations.py data/profiles/torch_events.csv --out data/profiles/recs.json
```

---

## 📁 Repository Structure

```text
Project/
├── requirements.txt                      # Project dependencies
├── schema.py                             # Canonical MemoryEvent dataclass
├── collector/                            # Telemetry collectors
│   ├── nvml_monitor.py                   # Out-of-process device monitor (NVML)
│   └── torch_hooks.py                    # In-process PyTorch allocator hook & snapshot logger
├── profiler/                             # Diagnostic engines
│   ├── memory_pressure.py                # Headroom, growth rate & heuristic OOM risk
│   └── fragmentation.py                  # Free-Space Fragmentation Index (FSFI)
├── optimizer/                            # Optimization engine
│   ├── recommendations.py                # Traceable rule-based recommendation generator
│   └── batch_optimizer.py                # Batch scaling model & candidate classifier
├── workloads/                            # Synthetic & diagnostic workloads
│   ├── memory_stress.py                  # Allocation stress testing
│   └── memory_fragmentation.py           # Controlled alternating tensor fragmentation
├── tests/                                # Automated pytest suites (21 unit tests)
│   ├── test_memory_pressure.py           # Pressure heuristics tests
│   └── test_recommendations.py           # Recommendation rules & batch optimizer tests
├── data/                                 # Recorded profiles & snapshots for development
│   └── profiles/
├── work_done.md                          # Comprehensive handover blueprint for teammates
├── work_division.md                      # Team responsibilities & module ownership
└── GPU_Memory_Profiler_Development_Plan_Revised.md # Primary development plan
```

---

## 📖 Handover & Documentation
* For an exhaustive technical breakdown of what has been built, how it works, and how to build the remaining modules (Admission Control, Contention, Streamlit Dashboard), refer to [`work_done.md`](work_done.md).
* For the full phase-by-phase course specification, refer to [`GPU_Memory_Profiler_Development_Plan_Revised.md`](GPU_Memory_Profiler_Development_Plan_Revised.md).
