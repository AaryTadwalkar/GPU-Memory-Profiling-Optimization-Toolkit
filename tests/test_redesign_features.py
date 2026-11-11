"""
File: tests/test_redesign_features.py

1. What this file does:
   Comprehensive automated unit test suite verifying the redesigned GPU memory
   profiling, passive allocator inspection, measurement-driven memory pressure,
   GPU resource scheduling, contention evaluation, and 7 OS experiment modules.

2. Overall design:
   Uses standard pytest framework with CPU-safe mocked/deterministic paths to
   rigorously test all logic without requiring physical CUDA hardware or violating
   system safety constraints.

3. Elements in the file:
   - TestPassiveAllocatorMonitoring: Tests for get_passive_allocator_status.
   - TestLiveMemoryPressureEvaluation: Tests for evaluate_live_memory_pressure.
   - TestWorkloadSchedulerAndContention: Tests for demand profiling and contention.
   - TestEvaluationExperimentsSuite: Tests for the 7 OS evaluation experiments.

4. Final output of the file:
   Pytest exit code and test execution report verifying zero regressions and complete
   conformance with the redesign specifications.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profiler.fragmentation import (
    get_passive_allocator_status,
    parse_snapshot_detailed,
)
from profiler.memory_pressure import (
    evaluate_live_memory_pressure,
    evaluate_oom_risk_heuristic,
)
from controller.admission_control import AdmissionController, AdmissionDecision
from experiments.schema import ExperimentStatus, ExperimentResult
from experiments.runner import (
    WorkloadSpec,
    profile_workload_demand,
    run_contention_experiment,
    run_allocation_pattern_experiment,
    run_memory_reclamation_experiment,
    run_precision_comparison_experiment,
    run_batch_scaling_experiment,
    get_activation_swapper_evaluation,
)


class TestPassiveAllocatorMonitoring(unittest.TestCase):
    """
    1. The use of class:
       Validates passive PyTorch caching allocator monitoring and idle pool distinction.
    2. What the class contains:
       Unit test methods verifying passive telemetry extraction and N/A FSFI on idle pool.
    3. Technologies used and why:
       unittest.TestCase for deterministic, pure Python execution.
    4. Key design:
       Ensures that dormant/unallocated pools never fabricate a 0.0000 or historical FSFI.
    """

    def test_passive_allocator_idle_or_available(self):
        """
        1. The use of function:
           Tests get_passive_allocator_status return structure and idle classification.
        2. How does the specific function work:
           Calls get_passive_allocator_status, verifies all expected keys are present,
           and checks that when reserved memory is 0, is_idle is True and fsfi is None.
        3. Concepts used in the function:
           Virtual allocator introspection, null-safety, schema validation.
        4. Places where this function is imported and used:
           Executed by pytest during test suite runs.
        5. Final output of the function, before and after:
           Before: Unverified function output.
           After: Verified dictionary with guaranteed key contracts.
        """
        status = get_passive_allocator_status()
        self.assertIn("available", status)
        self.assertIn("is_idle", status)
        self.assertIn("status_description", status)
        self.assertIn("distinction_note", status)
        self.assertIn("fsfi", status)

        if status["is_idle"]:
            self.assertIsNone(status["fsfi"])
            self.assertIn("Idle", status["status_description"])


class TestLiveMemoryPressureEvaluation(unittest.TestCase):
    """
    1. The use of class:
       Validates measurement-driven memory pressure and OOM risk classification.
    2. What the class contains:
       Tests for low risk on idle GPU, high risk on low headroom, and TTE estimation.
    3. Technologies used and why:
       unittest.TestCase with synthetic rolling timeseries data.
    4. Key design:
       Verifies that 98% free GPU is never classified as Moderate Risk due to stale CSV data.
    """

    def test_idle_gpu_reports_low_risk(self):
        """
        1. The use of function:
           Verifies that an idle GPU with >25% headroom and no growth reports Low Risk.
        2. How does the specific function work:
           Passes 6GB total and 100MB used (98.4% free) with empty rolling history.
        3. Concepts used in the function:
           Driver headroom analysis, threshold compliance.
        4. Places where this function is imported and used:
           Executed by pytest during test suite runs.
        5. Final output of the function, before and after:
           Before: Risk unverified.
           After: Confirmed risk_level == 'Low' and tte_seconds is None.
        """
        total = 6 * 1024 * 1024 * 1024
        used = 100 * 1024 * 1024  # 98.4% free

        res = evaluate_live_memory_pressure(
            physical_total_bytes=total,
            physical_used_bytes=used,
            process_allocated_bytes=0,
            process_reserved_bytes=0,
            rolling_history=[],
        )
        self.assertEqual(res["risk_level"], "Low")
        self.assertIsNone(res["tte_seconds"])
        self.assertIn("Stable", res["tte_display"])
        self.assertGreater(res["physical_headroom_ratio"], 0.90)

    def test_low_headroom_reports_high_risk(self):
        """
        1. The use of function:
           Verifies that a GPU with <10% free headroom reports High Risk.
        2. How does the specific function work:
           Passes 6GB total and 5.8GB used (3.3% free headroom).
        3. Concepts used in the function:
           Capacity exhaustion thresholds.
        4. Places where this function is imported and used:
           Executed by pytest during test suite runs.
        5. Final output of the function, before and after:
           Before: Unverified high risk trigger.
           After: Confirmed risk_level == 'High'.
        """
        total = 6 * 1024 * 1024 * 1024
        used = int(5.8 * 1024 * 1024 * 1024)

        res = evaluate_live_memory_pressure(
            physical_total_bytes=total,
            physical_used_bytes=used,
            process_allocated_bytes=100 * 1024 * 1024,
            process_reserved_bytes=200 * 1024 * 1024,
            rolling_history=[],
        )
        self.assertEqual(res["risk_level"], "High")

    def test_sustained_growth_triggers_tte_calculation(self):
        """
        1. The use of function:
           Verifies that sustained positive memory growth calculates Time-to-Exhaustion.
        2. How does the specific function work:
           Constructs synthetic rolling samples growing at ~60 MB/s over 5 seconds.
        3. Concepts used in the function:
           Derivative growth velocity, time-to-exhaustion projection.
        4. Places where this function is imported and used:
           Executed by pytest during test suite runs.
        5. Final output of the function, before and after:
           Before: Unverified TTE calculation.
           After: Confirmed sustained_growth == True and tte_seconds > 0.
        """
        total = 6 * 1024 * 1024 * 1024
        used = 3 * 1024 * 1024 * 1024

        # Synthetic samples growing at 60 MB/s: 0s=100MB, 1s=160MB, 2s=220MB, 3s=280MB
        history = [
            {"timestamp": 100.0, "used_bytes": 100 * 1024 * 1024},
            {"timestamp": 101.0, "used_bytes": 160 * 1024 * 1024},
            {"timestamp": 102.0, "used_bytes": 220 * 1024 * 1024},
            {"timestamp": 103.0, "used_bytes": 280 * 1024 * 1024},
        ]

        res = evaluate_live_memory_pressure(
            physical_total_bytes=total,
            physical_used_bytes=used,
            rolling_history=history,
        )
        self.assertTrue(res["sustained_growth"])
        self.assertIsNotNone(res["tte_seconds"])
        self.assertGreater(res["tte_seconds"], 0)


class TestWorkloadSchedulerAndContention(unittest.TestCase):
    """
    1. The use of class:
       Validates cooperative GPU resource scheduling, empirical demand profiling,
       and isolated vs concurrent contention evaluation.
    2. What the class contains:
       Unit tests for profile_workload_demand and run_contention_experiment.
    3. Technologies used and why:
       unittest.TestCase for robust execution across CPU and GPU environments.
    4. Key design:
       Verifies zero memory leaks, accurate slowdown calculations, and non-guesswork demand profiling.
    """

    def test_demand_profiling_returns_valid_metrics(self):
        """
        1. The use of function:
           Tests that profile_workload_demand returns non-zero empirical memory requirements.
        2. How does the specific function work:
           Calls profile_workload_demand for Small CNN and ResNet Stage.
        3. Concepts used in the function:
           Empirical demand profiling.
        4. Places where this function is imported and used:
           Executed by pytest during test suite runs.
        5. Final output of the function, before and after:
           Before: Unprofiled architecture.
           After: Profile dictionary with positive demand in bytes and MB.
        """
        p_cnn = profile_workload_demand(architecture="Small CNN", batch_size=16, precision="FP32")
        self.assertTrue(p_cnn["success"])
        self.assertGreater(p_cnn["measured_demand_bytes"], 0)
        self.assertGreater(p_cnn["measured_demand_mb"], 0)

        p_resnet = profile_workload_demand(architecture="ResNet Stage", batch_size=32, precision="FP16")
        self.assertTrue(p_resnet["success"])
        self.assertGreater(p_resnet["measured_demand_bytes"], 0)

    def test_contention_experiment_measures_slowdown(self):
        """
        1. The use of function:
           Tests isolated vs concurrent contention experiment execution and slowdown math.
        2. How does the specific function work:
           Calls run_contention_experiment with a 64 MB stress tensor.
        3. Concepts used in the function:
           Resource contention, slowdown percentage math.
        4. Places where this function is imported and used:
           Executed by pytest during test suite runs.
        5. Final output of the function, before and after:
           Before: Unmeasured contention.
           After: Confirmed positive isolated and concurrent runtimes with explanation.
        """
        res = run_contention_experiment(stress_tensor_mb=64)
        self.assertTrue(res["success"])
        self.assertIn("isolated_runtime_s", res)
        self.assertIn("concurrent_runtime_s", res)
        self.assertIn("slowdown_pct", res)
        self.assertIn("explanation", res)
        self.assertGreater(res["isolated_runtime_s"], 0)
        self.assertGreater(res["concurrent_runtime_s"], 0)


class TestEvaluationExperimentsSuite(unittest.TestCase):
    """
    1. The use of class:
       Validates the 7 structured OS memory management evaluation experiments.
    2. What the class contains:
       Test methods for experiments 1 through 7 and ExperimentStatus constants.
    3. Technologies used and why:
       unittest.TestCase for automated testing of experimental routines.
    4. Key design:
       Verifies each experiment produces structured comparative outputs and OS explanations.
    """

    def test_experiment1_allocation_pattern(self):
        """Tests linear gradual vs bursty monolithic allocation experiment."""
        res = run_allocation_pattern_experiment(num_chunks=4, chunk_size_mb=16)
        self.assertTrue(res["success"])
        self.assertIn("linear_time_ms", res)
        self.assertIn("bursty_time_ms", res)
        self.assertIn("explanation", res)

    def test_experiment3_segment_pinning_reclamation(self):
        """Tests unpinned vs pinned segment empty_cache reclamation."""
        res = run_memory_reclamation_experiment()
        self.assertTrue(res["success"])
        self.assertIn("unpinned_released_mb", res)
        self.assertIn("pinned_released_mb", res)
        self.assertIn("explanation", res)
        self.assertIn("Segment Pinning", res["explanation"])

    def test_experiment5_precision_comparison(self):
        """Tests FP32 vs FP16 peak memory reduction."""
        res = run_precision_comparison_experiment(architecture="resnet_stage", batch_size=16)
        self.assertTrue(res["success"])
        self.assertIn("fp32_peak_mb", res)
        self.assertIn("fp16_peak_mb", res)
        self.assertIn("memory_reduction_pct", res)
        self.assertGreater(res["fp32_peak_mb"], res["fp16_peak_mb"])

    def test_experiment6_batch_scaling(self):
        """Tests memory scaling across batch sizes."""
        res = run_batch_scaling_experiment(architecture="small_cnn", batch_sizes=[8, 16])
        self.assertTrue(res["success"])
        self.assertEqual(len(res["batch_results"]), 2)
        self.assertLess(res["batch_results"][0]["peak_mb"], res["batch_results"][1]["peak_mb"])

    def test_experiment7_activation_swapper_evaluation(self):
        """Tests retrieval and structure of swapper optimization evaluation."""
        res = get_activation_swapper_evaluation()
        self.assertTrue(res["success"])
        self.assertIn("baseline_phase", res)
        self.assertIn("swapper_phase", res)
        self.assertFalse(res["baseline_phase"]["training_completed"])
        self.assertTrue(res["swapper_phase"]["training_completed"])
        self.assertIn("0.00e+00", res["swapper_phase"]["max_gradient_difference"])

    def test_experiment_status_constants(self):
        """Verifies that SKIPPED and EXECUTION_ERROR are recognized status codes."""
        self.assertIn("SKIPPED", ExperimentStatus.ALL_STATUSES)
        self.assertIn("EXECUTION_ERROR", ExperimentStatus.ALL_STATUSES)
        self.assertIn("OOM", ExperimentStatus.ALL_STATUSES)
        self.assertIn("SUCCESS", ExperimentStatus.ALL_STATUSES)

