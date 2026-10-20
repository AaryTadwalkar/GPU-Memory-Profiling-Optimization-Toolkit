"""
File: experiments/__init__.py

1. What this file does:
   Exposes the public API for the experiments package, including the experiment
   result dataclass, CSV/JSON result writer, and baseline-versus-controlled
   orchestration runners.

2. Overall design:
   Re-exports the core result schema, status constants, serialization helpers,
   workload specification dataclasses, and execution runners so callers and tests
   can import them cleanly from `experiments`:
       from experiments import (
           ExperimentResult,
           ExperimentResultWriter,
           ExperimentStatus,
           WorkloadSpec,
           run_baseline_experiment,
           run_controlled_experiment,
       )

   COOPERATIVE EXPERIMENT DISCLAIMER:
   This experiment framework benchmarks multi-process workloads using either
   unregulated baseline execution or cooperative admission control. Admission
   control operates on declared estimates and observed telemetry; it does not
   provide hardware-level memory isolation or prevent external processes from
   consuming GPU VRAM.

3. Elements in the file:
   - Re-exports from `experiments.schema`
   - Re-exports from `experiments.runner`

4. Final output:
   Clean package-level namespace for experiment tracking and orchestration.
"""

from experiments.schema import (
    EXPERIMENT_RESULT_FIELDS,
    ExperimentResult,
    ExperimentResultWriter,
    ExperimentStatus,
    detect_environment_metadata,
    load_experiment_results,
    load_experiment_results_json,
    save_experiment_results_json,
)
from experiments.runner import (
    WorkloadExecutionResult,
    WorkloadSpec,
    run_baseline_experiment,
    run_controlled_experiment,
)

__all__ = [
    "EXPERIMENT_RESULT_FIELDS",
    "ExperimentResult",
    "ExperimentResultWriter",
    "ExperimentStatus",
    "WorkloadExecutionResult",
    "WorkloadSpec",
    "detect_environment_metadata",
    "load_experiment_results",
    "load_experiment_results_json",
    "run_baseline_experiment",
    "run_controlled_experiment",
    "save_experiment_results_json",
]
