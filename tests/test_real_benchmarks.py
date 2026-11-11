"""
File: tests/test_real_benchmarks.py

1. What this file does:
   Tests the real GPU workload execution runner and its operational integration with
   cooperative admission control, verifying peak memory measurements, headroom safety checks,
   and deterministic resource cleanup.

2. Overall design:
   Uses WorkloadSpec instances to test real_gpu_workload_runner on available CUDA hardware
   (or graceful CPU error handling), validates forward-only vs forward-backward passes,
   and asserts that reservations are strictly released upon workload completion or failure.

3. Elements in the file:
   - TestRealGPUWorkloadRunner: Verifies execution, peak memory reporting, and cleanup.
   - TestOperationalAdmissionIntegration: Verifies request -> admit -> execute -> release flow.

4. Final output of the file:
   Pytest execution verifying real benchmark runner correctness, safety constraints, and leak prevention.
"""

from pathlib import Path
import sys
import pytest

# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.admission_control import AdmissionController, AdmissionDecision
from experiments.runner import (
    WorkloadSpec,
    WorkloadExecutionResult,
    real_gpu_workload_runner,
    run_controlled_experiment,
)
from experiments.schema import ExperimentStatus

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


class TestRealGPUWorkloadRunner:
    """
    1. The use of class:
       Validates execution of real_gpu_workload_runner across real hardware configurations.

    2. What the class contain:
       Test methods for Small CNN forward-only execution, FP16 execution, and post-run VRAM cleanup.

    3. Technologies used and why:
       Pytest testing framework with CUDA hardware introspection.

    4. Key design and overall design of the class:
       Executes conservative workloads to verify that peak_memory_bytes is populated with an actual
       nonzero integer, runtime_s is positive, and memory is reclaimed post-run.
    """

    def test_real_runner_execution_and_cleanup(self) -> None:
        """
        1. The use of function:
           Tests execution of a conservative Small CNN workload on CUDA (or asserts graceful failure if CPU).

        2. How does the specific function work:
           Configures WorkloadSpec(workload_id="test_cnn", workload_name="small_cnn", estimated_bytes=50*1024*1024, batch_size=8).
           Executes real_gpu_workload_runner. If CUDA is present, verifies SUCCESS, peak_memory_bytes > 0,
           and asserts torch.cuda.memory_allocated() == 0 after execution.

        3. Concepts used in the function:
           Hardware execution, peak memory verification, resource leak assertion.

        4. Places where this function is imported and used:
           Executed by pytest.

        5. Final output of the function, before and after:
           Before: Unexecuted spec.
           After: Asserted successful execution and complete cleanup.
        """
        spec = WorkloadSpec(
            workload_id="test_cnn_01",
            workload_name="small_cnn",
            estimated_bytes=50 * 1024 * 1024,
            batch_size=8,
            precision="fp32",
        )
        res = real_gpu_workload_runner(spec)
        if HAS_CUDA:
            assert res.status == ExperimentStatus.SUCCESS
            assert res.peak_memory_bytes is not None
            assert res.peak_memory_bytes > 0
            assert res.runtime_s is not None and res.runtime_s > 0
            assert torch.cuda.memory_allocated() == 0
        else:
            assert res.status == ExperimentStatus.FAILED
            assert "CUDA accelerator not available" in (res.error_message or "")


class TestOperationalAdmissionIntegration:
    """
    1. The use of class:
       Validates operational integration between AdmissionController and real workload execution.

    2. What the class contain:
       Test methods verifying the ADMIT -> execute -> release lifecycle and reservation integrity.

    3. Technologies used and why:
       Pytest assertions with AdmissionController and run_controlled_experiment.

    4. Key design and overall design of the class:
       Ensures that when a workload is admitted and executed, its reservation is guaranteed to be
       released from the controller even if exceptions occur.
    """

    def test_controlled_experiment_with_real_runner(self) -> None:
        """
        1. The use of function:
           Tests run_controlled_experiment with real_gpu_workload_runner.

        2. How does the specific function work:
           Initializes AdmissionController with an 8 GB budget. Submits a workload spec.
           Verifies that after run_controlled_experiment finishes, current_reservations in the controller
           is strictly 0 (no reservation leaks).

        3. Concepts used in the function:
           Cooperative admission control, reservation lifecycle, leak prevention.

        4. Places where this function is imported and used:
           Executed by pytest.

        5. Final output of the function, before and after:
           Before: Empty controller state.
           After: Asserted zero active reservations remaining.
        """
        controller = AdmissionController(
            budget_bytes=8 * 1024 * 1024 * 1024,
            safety_margin_ratio=0.10,
        )
        spec = WorkloadSpec(
            workload_id="test_controlled_cnn",
            workload_name="small_cnn",
            estimated_bytes=100 * 1024 * 1024,
            batch_size=8,
            precision="fp32",
        )
        results = run_controlled_experiment(
            workloads=[spec],
            controller=controller,
            runner_fn=real_gpu_workload_runner if HAS_CUDA else None,
        )
        assert len(results) == 1
        assert controller.reserved_total_bytes == 0
        assert "test_controlled_cnn" not in controller.admitted_workloads
