"""
File: tests/test_experiment_results.py

1. What this file does:
   Comprehensive CPU-only unit test suite for the experiment framework:
   - Schema validation and exact canonical CSV column order.
   - Serialization / deserialization roundtrip (CSV and JSON).
   - Robustness with missing or null hardware metadata.
   - Distinct classification of SUCCESS, FAILED, DEFERRED, and REJECTED runs.
   - Separation of estimated memory requirements vs measured peak memory.
   - Preservation of exceptions, tracebacks, and error messages in failed runs.
   - Mocked baseline orchestration (unregulated concurrency).
   - Mocked controlled orchestration (admission control, deferral, rejection, safe release).
   - Clean importing without CUDA/NVML dependencies.

2. Overall design:
   Uses pytest with synthetic workload specifications and in-memory mock runners.
   No live GPU, NVML, or CUDA calls are made. Temporary files are used for CSV/JSON testing.

3. Elements in the file:
   - Helpers: gb(), mb(), make_dummy_spec().
   - TestSchemaAndOrdering: Verifies all 21 canonical fields and their exact order.
   - TestSerialization: Tests roundtrip persistence for CSV and JSON.
   - TestHardwareMetadata: Tests handling of missing or partial hardware metadata.
   - TestStatusAndErrorPreservation: Validates that errors are not swallowed and statuses remain distinct.
   - TestMemorySeparation: Confirms estimated requirement is never conflated with observed peak.
   - TestBaselineOrchestrator: Exercises unregulated multi-workload execution with mock runners.
   - TestControlledOrchestrator: Exercises cooperative admission control, deferral, rejection, and reservation cleanup.
   - TestImportSafety: Confirms no GPU dependencies are triggered upon importing.

4. Final output:
   Automated verification of the experiment framework.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from controller.admission_control import AdmissionController, AdmissionDecision
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gb(n: float) -> int:
    return int(n * 1024 * 1024 * 1024)


def mb(n: float) -> int:
    return int(n * 1024 * 1024)


def make_dummy_spec(
    workload_id: str = "test_wl_1",
    workload_name: str = "cnn_test",
    estimated_bytes: int = 100 * 1024 * 1024,
    batch_size: int = 16,
    precision: str = "fp32",
) -> WorkloadSpec:
    return WorkloadSpec(
        workload_id=workload_id,
        workload_name=workload_name,
        estimated_bytes=estimated_bytes,
        batch_size=batch_size,
        precision=precision,
    )


# ---------------------------------------------------------------------------
# Test Schema & Canonical Field Ordering
# ---------------------------------------------------------------------------

class TestSchemaAndOrdering:
    def test_canonical_field_list_length_and_names(self):
        expected_fields = [
            "experiment_id",
            "run_id",
            "timestamp",
            "gpu_model",
            "vram_capacity_bytes",
            "driver_version",
            "cuda_version",
            "pytorch_version",
            "workload_name",
            "batch_size",
            "precision",
            "num_concurrent",
            "admission_decision",
            "status",
            "runtime_s",
            "peak_memory_bytes",
            "estimated_requirement_bytes",
            "oom_occurred",
            "error_message",
            "output_csv_path",
            "notes",
        ]
        assert EXPERIMENT_RESULT_FIELDS == expected_fields
        assert len(EXPERIMENT_RESULT_FIELDS) == 21

    def test_to_row_contains_all_canonical_fields_in_order(self):
        res = ExperimentResult(
            experiment_id="exp_01",
            run_id="run_01",
            timestamp=1700000000.0,
            workload_name="resnet",
            batch_size=32,
            precision="fp16",
        )
        row = res.to_row()
        assert list(row.keys()) == EXPERIMENT_RESULT_FIELDS

    def test_to_dict_contains_all_fields(self):
        res = ExperimentResult(
            experiment_id="exp_01",
            run_id="run_01",
            timestamp=1700000000.0,
            workload_name="resnet",
        )
        d = res.to_dict()
        for field in EXPERIMENT_RESULT_FIELDS:
            assert field in d


# ---------------------------------------------------------------------------
# Test Serialization & Persistence
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_csv_roundtrip_all_fields_populated(self):
        res = ExperimentResult(
            experiment_id="exp_full",
            run_id="run_101",
            timestamp=1700000123.456789,
            workload_name="transformer",
            gpu_model="NVIDIA GeForce RTX 3050 Laptop GPU",
            vram_capacity_bytes=gb(6),
            driver_version="555.42",
            cuda_version="12.4",
            pytorch_version="2.3.0",
            batch_size=16,
            precision="bf16",
            num_concurrent=3,
            admission_decision="ADMIT",
            status=ExperimentStatus.SUCCESS,
            runtime_s=12.3456,
            peak_memory_bytes=gb(2),
            estimated_requirement_bytes=gb(2),
            oom_occurred=False,
            error_message=None,
            output_csv_path="data/profiles/run_101.csv",
            notes="Completed smoothly",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            writer = ExperimentResultWriter(csv_path)
            writer.write_result(res)

            loaded = load_experiment_results(csv_path)
            assert len(loaded) == 1
            item = loaded[0]

            assert item.experiment_id == "exp_full"
            assert item.run_id == "run_101"
            assert abs(item.timestamp - 1700000123.456789) < 1e-4
            assert item.workload_name == "transformer"
            assert item.gpu_model == "NVIDIA GeForce RTX 3050 Laptop GPU"
            assert item.vram_capacity_bytes == gb(6)
            assert item.driver_version == "555.42"
            assert item.cuda_version == "12.4"
            assert item.pytorch_version == "2.3.0"
            assert item.batch_size == 16
            assert item.precision == "bf16"
            assert item.num_concurrent == 3
            assert item.admission_decision == "ADMIT"
            assert item.status == ExperimentStatus.SUCCESS
            assert abs(item.runtime_s - 12.3456) < 1e-3
            assert item.peak_memory_bytes == gb(2)
            assert item.estimated_requirement_bytes == gb(2)
            assert item.oom_occurred is False
            assert item.error_message is None
            assert item.output_csv_path == "data/profiles/run_101.csv"
            assert item.notes == "Completed smoothly"

    def test_csv_multiple_appends_single_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "multi.csv"
            writer = ExperimentResultWriter(csv_path)

            r1 = ExperimentResult(experiment_id="e1", run_id="r1", timestamp=100.0, workload_name="w1")
            r2 = ExperimentResult(experiment_id="e1", run_id="r2", timestamp=101.0, workload_name="w2")

            writer.write_result(r1)
            writer.write_result(r2)

            with open(csv_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            # Header + 2 data rows = 3 lines total
            assert len(lines) == 3
            assert lines[0].startswith("experiment_id,run_id,timestamp")

            loaded = load_experiment_results(csv_path)
            assert len(loaded) == 2
            assert loaded[0].run_id == "r1"
            assert loaded[1].run_id == "r2"

    def test_json_roundtrip(self):
        results = [
            ExperimentResult(experiment_id="ej", run_id="r1", timestamp=100.0, workload_name="w1"),
            ExperimentResult(experiment_id="ej", run_id="r2", timestamp=101.0, workload_name="w2", oom_occurred=True),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"
            save_experiment_results_json(results, json_path)

            loaded = load_experiment_results_json(json_path)
            assert len(loaded) == 2
            assert loaded[0].run_id == "r1"
            assert loaded[1].oom_occurred is True

    def test_writer_with_both_csv_and_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_p = Path(tmpdir) / "out.csv"
            json_p = Path(tmpdir) / "out.json"
            writer = ExperimentResultWriter(csv_p, json_p)

            res = ExperimentResult(experiment_id="dual", run_id="d1", timestamp=500.0, workload_name="cnn")
            writer.write_result(res)

            assert csv_p.exists()
            assert json_p.exists()
            loaded_csv = load_experiment_results(csv_p)
            loaded_json = load_experiment_results_json(json_p)
            assert len(loaded_csv) == 1
            assert len(loaded_json) == 1


# ---------------------------------------------------------------------------
# Test Missing Hardware Metadata Handling
# ---------------------------------------------------------------------------

class TestHardwareMetadata:
    def test_missing_hardware_metadata_defaults_to_none_in_schema(self):
        res = ExperimentResult(
            experiment_id="e_missing",
            run_id="r_missing",
            timestamp=1000.0,
            workload_name="simple",
        )
        assert res.gpu_model is None
        assert res.vram_capacity_bytes is None
        assert res.driver_version is None
        assert res.cuda_version is None

    def test_detect_environment_metadata_pure_cpu_safety(self):
        meta = detect_environment_metadata(safe_cpu_only=True)
        assert isinstance(meta, dict)
        assert meta["gpu_model"] is None
        assert meta["vram_capacity_bytes"] is None
        assert meta["driver_version"] is None
        assert meta["cuda_version"] is None
        # PyTorch version may be detected if torch is installed, without error
        assert "pytorch_version" in meta

    def test_csv_roundtrip_preserves_none_for_missing_hardware_metadata(self):
        res = ExperimentResult(
            experiment_id="e_nulls",
            run_id="r_nulls",
            timestamp=1000.0,
            workload_name="simple",
            gpu_model=None,
            vram_capacity_bytes=None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "nulls.csv"
            writer = ExperimentResultWriter(csv_path)
            writer.write_result(res)

            loaded = load_experiment_results(csv_path)
            assert loaded[0].gpu_model is None
            assert loaded[0].vram_capacity_bytes is None
            assert loaded[0].driver_version is None


# ---------------------------------------------------------------------------
# Test Status & Error Preservation
# ---------------------------------------------------------------------------

class TestStatusAndErrorPreservation:
    def test_status_constants_are_distinct(self):
        assert ExperimentStatus.SUCCESS == "SUCCESS"
        assert ExperimentStatus.FAILED == "FAILED"
        assert ExperimentStatus.DEFERRED == "DEFERRED"
        assert ExperimentStatus.REJECTED == "REJECTED"
        assert ExperimentStatus.OOM == "OOM"

        statuses = {
            ExperimentStatus.SUCCESS,
            ExperimentStatus.FAILED,
            ExperimentStatus.DEFERRED,
            ExperimentStatus.REJECTED,
            ExperimentStatus.OOM,
        }
        assert len(statuses) == 5

    def test_failed_run_preserves_error_message(self):
        err_msg = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        res = ExperimentResult(
            experiment_id="exp_err",
            run_id="run_err",
            timestamp=200.0,
            workload_name="heavy_resnet",
            status=ExperimentStatus.OOM,
            oom_occurred=True,
            error_message=err_msg,
        )
        assert res.status == "OOM"
        assert res.oom_occurred is True
        assert res.error_message == err_msg

    def test_deferred_status_is_distinct_from_failed(self):
        res_deferred = ExperimentResult(
            experiment_id="exp_def",
            run_id="run_def",
            timestamp=300.0,
            workload_name="queued_wl",
            admission_decision=AdmissionDecision.DEFER,
            status=ExperimentStatus.DEFERRED,
            notes="Deferred due to high active load",
        )
        assert res_deferred.status == "DEFERRED"
        assert res_deferred.status != ExperimentStatus.FAILED
        assert res_deferred.admission_decision == "DEFER"

    def test_rejected_status_is_distinct_from_failed(self):
        res_rejected = ExperimentResult(
            experiment_id="exp_rej",
            run_id="run_rej",
            timestamp=400.0,
            workload_name="oversized_wl",
            admission_decision=AdmissionDecision.REJECT,
            status=ExperimentStatus.REJECTED,
            error_message="estimated_bytes exceeds max_single_workload_ratio cap",
        )
        assert res_rejected.status == "REJECTED"
        assert res_rejected.status != ExperimentStatus.FAILED
        assert res_rejected.admission_decision == "REJECT"

    def test_from_dict_handles_none_and_missing_status(self):
        """Explicit None, empty string, or missing status must default to ExperimentStatus.SUCCESS."""
        d_none = {
            "experiment_id": "exp_test",
            "run_id": "r1",
            "timestamp": 100.0,
            "workload_name": "model_a",
            "status": None,
        }
        res_none = ExperimentResult.from_dict(d_none)
        assert res_none.status == ExperimentStatus.SUCCESS

        d_missing = {
            "experiment_id": "exp_test",
            "run_id": "r2",
            "timestamp": 100.0,
            "workload_name": "model_b",
        }
        res_missing = ExperimentResult.from_dict(d_missing)
        assert res_missing.status == ExperimentStatus.SUCCESS

        d_empty = {
            "experiment_id": "exp_test",
            "run_id": "r3",
            "timestamp": 100.0,
            "workload_name": "model_c",
            "status": "",
        }
        res_empty = ExperimentResult.from_dict(d_empty)
        assert res_empty.status == ExperimentStatus.SUCCESS


# ---------------------------------------------------------------------------
# Test Separation of Estimated vs Measured Memory
# ---------------------------------------------------------------------------

class TestMemorySeparation:
    def test_estimated_and_peak_memory_are_separate_fields(self):
        res = ExperimentResult(
            experiment_id="exp_mem",
            run_id="run_mem",
            timestamp=500.0,
            workload_name="transformer",
            estimated_requirement_bytes=gb(3),
            peak_memory_bytes=gb(2),
        )
        assert res.estimated_requirement_bytes == gb(3)
        assert res.peak_memory_bytes == gb(2)
        assert res.estimated_requirement_bytes != res.peak_memory_bytes

    def test_peak_memory_none_when_unmeasured_or_deferred(self):
        res = ExperimentResult(
            experiment_id="exp_mem",
            run_id="run_def",
            timestamp=500.0,
            workload_name="transformer",
            estimated_requirement_bytes=gb(4),
            peak_memory_bytes=None,  # Not run, no measurement
            status=ExperimentStatus.DEFERRED,
        )
        assert res.estimated_requirement_bytes == gb(4)
        assert res.peak_memory_bytes is None

    def test_csv_serialization_maintains_memory_separation(self):
        res = ExperimentResult(
            experiment_id="exp_mem",
            run_id="run_sep",
            timestamp=600.0,
            workload_name="cnn",
            estimated_requirement_bytes=mb(500),
            peak_memory_bytes=mb(420),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_p = Path(tmpdir) / "sep.csv"
            writer = ExperimentResultWriter(csv_p)
            writer.write_result(res)

            loaded = load_experiment_results(csv_p)[0]
            assert loaded.estimated_requirement_bytes == mb(500)
            assert loaded.peak_memory_bytes == mb(420)


# ---------------------------------------------------------------------------
# Test Baseline Orchestrator
# ---------------------------------------------------------------------------

class TestBaselineOrchestrator:
    def test_baseline_successful_runs(self):
        specs = [
            make_dummy_spec("wl_1", "cnn", mb(200)),
            make_dummy_spec("wl_2", "resnet", mb(400)),
        ]

        def mock_runner(spec: WorkloadSpec) -> WorkloadExecutionResult:
            return WorkloadExecutionResult(
                status=ExperimentStatus.SUCCESS,
                runtime_s=0.05,
                peak_memory_bytes=spec.estimated_bytes + mb(10),
                oom_occurred=False,
            )

        results = run_baseline_experiment(
            workloads=specs,
            experiment_id="exp_base_pass",
            runner_fn=mock_runner,
        )

        assert len(results) == 2
        for r in results:
            assert r.admission_decision is None
            assert r.status == ExperimentStatus.SUCCESS
            assert r.runtime_s == 0.05
            assert r.peak_memory_bytes is not None
            assert r.num_concurrent == 2

    def test_baseline_preserves_runner_exception_without_crashing(self):
        specs = [
            make_dummy_spec("wl_ok", "cnn", mb(100)),
            make_dummy_spec("wl_fail", "bad_model", mb(200)),
        ]

        def mock_runner_with_error(spec: WorkloadSpec) -> WorkloadExecutionResult:
            if spec.workload_id == "wl_fail":
                raise RuntimeError("CUDA out of memory during backward pass")
            return WorkloadExecutionResult(
                status=ExperimentStatus.SUCCESS,
                runtime_s=0.02,
                peak_memory_bytes=spec.estimated_bytes,
            )

        results = run_baseline_experiment(
            workloads=specs,
            experiment_id="exp_base_err",
            runner_fn=mock_runner_with_error,
        )

        assert len(results) == 2
        fail_res = next(r for r in results if r.workload_name == "bad_model")
        assert fail_res.status == ExperimentStatus.OOM
        assert fail_res.oom_occurred is True
        assert "CUDA out of memory" in fail_res.error_message

        ok_res = next(r for r in results if r.workload_name == "cnn")
        assert ok_res.status == ExperimentStatus.SUCCESS
        assert ok_res.oom_occurred is False

    def test_baseline_writes_results_to_writer(self):
        specs = [make_dummy_spec("wl_w1", "cnn", mb(50))]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_p = Path(tmpdir) / "base_writer.csv"
            writer = ExperimentResultWriter(csv_p)
            results = run_baseline_experiment(
                workloads=specs,
                experiment_id="exp_bw",
                writer=writer,
            )
            assert len(results) == 1
            loaded = load_experiment_results(csv_p)
            assert len(loaded) == 1
            assert loaded[0].workload_name == "cnn"

    def test_default_mock_runner_leaves_peak_memory_none(self):
        """When no custom runner is provided, default_mock_runner must leave
        peak_memory_bytes as None (unmeasured) and not fabricate values."""
        specs = [make_dummy_spec("wl_default", "cnn", mb(150))]
        results = run_baseline_experiment(
            workloads=specs,
            experiment_id="exp_default_runner",
        )
        assert len(results) == 1
        res = results[0]
        assert res.peak_memory_bytes is None
        assert res.estimated_requirement_bytes == mb(150)
        assert res.status == ExperimentStatus.SUCCESS


# ---------------------------------------------------------------------------
# Test Controlled Orchestrator
# ---------------------------------------------------------------------------

class TestControlledOrchestrator:
    def test_controlled_admitted_workload_releases_reservation(self):
        controller = AdmissionController(budget_bytes=gb(6), safety_margin_ratio=0.10)
        spec = make_dummy_spec("wl_admit", "cnn", gb(2))

        results = run_controlled_experiment(
            workloads=[spec],
            controller=controller,
            experiment_id="exp_ctrl_ok",
        )

        assert len(results) == 1
        res = results[0]
        assert res.admission_decision == AdmissionDecision.ADMIT
        assert res.status == ExperimentStatus.SUCCESS
        # Controller reservation must be released after completion!
        assert "wl_admit" not in controller.admitted_workloads
        assert controller.reserved_total_bytes == 0

    def test_controlled_records_observed_peak_in_controller(self):
        controller = AdmissionController(budget_bytes=gb(6), safety_margin_ratio=0.10)
        spec = make_dummy_spec("wl_peak", "resnet", gb(1))

        def custom_runner(s: WorkloadSpec) -> WorkloadExecutionResult:
            return WorkloadExecutionResult(
                status=ExperimentStatus.SUCCESS,
                runtime_s=0.01,
                peak_memory_bytes=mb(900),
            )

        results = run_controlled_experiment(
            workloads=[spec],
            controller=controller,
            runner_fn=custom_runner,
        )
        assert len(results) == 1
        assert results[0].peak_memory_bytes == mb(900)
        # Estimation error is tracked in the controller
        err = controller.get_estimation_error("wl_peak")
        assert err == mb(900) - gb(1)

    def test_controlled_rejected_workload(self):
        # Set max_single_workload_ratio = 0.50. Requesting 4GB of 6GB exceeds ratio
        controller = AdmissionController(
            budget_bytes=gb(6),
            safety_margin_ratio=0.10,
            max_single_workload_ratio=0.50,
        )
        spec = make_dummy_spec("wl_oversized", "huge_model", gb(4))

        results = run_controlled_experiment(
            workloads=[spec],
            controller=controller,
            experiment_id="exp_ctrl_rej",
        )

        assert len(results) == 1
        res = results[0]
        assert res.admission_decision == AdmissionDecision.REJECT
        assert res.status == ExperimentStatus.REJECTED
        assert res.peak_memory_bytes is None
        assert "exceeds the per-workload policy cap" in res.error_message
        assert controller.reserved_total_bytes == 0

    def test_controlled_deferred_workload_when_headroom_insufficient(self):
        # 6 GB budget, 10% safety margin -> safe budget 5.4 GB
        controller = AdmissionController(budget_bytes=gb(6), safety_margin_ratio=0.10)
        # Pre-reserve 4.0 GB
        controller.request("existing_job", gb(4))

        # New job requires 2.0 GB -> 4.0 + 2.0 = 6.0 GB > 5.4 GB safe budget -> DEFER
        spec = make_dummy_spec("wl_defer", "secondary_job", gb(2))

        results = run_controlled_experiment(
            workloads=[spec],
            controller=controller,
            retry_deferred=False,
        )

        assert len(results) == 1
        res = results[0]
        assert res.admission_decision == AdmissionDecision.DEFER
        assert res.status == ExperimentStatus.DEFERRED
        assert res.peak_memory_bytes is None
        assert res.status != ExperimentStatus.FAILED

    def test_controlled_releases_reservation_on_runner_exception(self):
        controller = AdmissionController(budget_bytes=gb(6))
        spec = make_dummy_spec("wl_crash", "crash_model", gb(1))

        def crashing_runner(s: WorkloadSpec) -> WorkloadExecutionResult:
            raise RuntimeError("Unexpected simulator crash")

        results = run_controlled_experiment(
            workloads=[spec],
            controller=controller,
            runner_fn=crashing_runner,
        )

        assert len(results) == 1
        res = results[0]
        assert res.admission_decision == AdmissionDecision.ADMIT
        assert res.status == ExperimentStatus.FAILED
        assert "Unexpected simulator crash" in res.error_message
        # Even after crashing, the reservation MUST be released
        assert "wl_crash" not in controller.admitted_workloads
        assert controller.reserved_total_bytes == 0

    def test_controlled_retry_deferred_eventual_admission_deterministic(self):
        """A deferred workload with retry_deferred=True should retry without log flooding,
        and be admitted when an active workload finishes and releases its reservation."""
        import threading
        # 6 GiB budget, 10% safety margin -> safe budget 5.4 GiB
        controller = AdmissionController(budget_bytes=gb(6), safety_margin_ratio=0.10)

        # wl_a: 4.0 GiB (admitted first)
        # wl_b: 2.0 GiB (initially deferred because 4.0 + 2.0 = 6.0 > 5.4 GiB)
        spec_a = make_dummy_spec("wl_a", "heavy_job", gb(4))
        spec_b = make_dummy_spec("wl_b", "secondary_job", gb(2))

        wl_a_running = threading.Event()
        can_release_a = threading.Event()

        def custom_runner(spec: WorkloadSpec) -> WorkloadExecutionResult:
            if spec.workload_id == "wl_a":
                wl_a_running.set()
                # Deterministically wait until the test coordinates release
                can_release_a.wait(timeout=2.0)
            return WorkloadExecutionResult(
                status=ExperimentStatus.SUCCESS,
                runtime_s=0.01,
                peak_memory_bytes=spec.estimated_bytes,
            )

        def run_experiment_thread():
            return run_controlled_experiment(
                workloads=[spec_a, spec_b],
                controller=controller,
                runner_fn=custom_runner,
                max_concurrency=2,
                retry_deferred=True,
                poll_interval_s=0.01,
                max_retry_duration_s=2.0,
            )

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_experiment_thread)
            # Ensure wl_a is actively running and holding reservation
            assert wl_a_running.wait(timeout=2.0)
            # Short sleep to allow wl_b to encounter its initial deferral and enter polling loop
            time.sleep(0.05)
            # Signal wl_a to finish and release its 4.0 GiB reservation
            can_release_a.set()
            results = future.result(timeout=3.0)

        assert len(results) == 2
        res_a = next(r for r in results if r.workload_name == "heavy_job")
        res_b = next(r for r in results if r.workload_name == "secondary_job")

        assert res_a.status == ExperimentStatus.SUCCESS
        assert res_a.admission_decision == AdmissionDecision.ADMIT

        assert res_b.status == ExperimentStatus.SUCCESS
        assert res_b.admission_decision == AdmissionDecision.ADMIT

        # Check decision_log: wl_b should have exactly 1 DEFER event (the initial deferral)
        # and 1 ADMIT event, verifying that intermediate polls did not flood the log.
        wl_b_events = [
            e for e in controller.decision_log
            if e.get("workload_name") == "wl_b"
        ]
        defer_events = [
            e for e in wl_b_events
            if "DEFER" in (e.get("metadata_json") or "")
        ]
        assert len(defer_events) == 1


# ---------------------------------------------------------------------------
# Test Safe Imports (No CUDA / NVML initialization)
# ---------------------------------------------------------------------------

class TestImportSafety:
    def test_import_experiments_modules_does_not_require_cuda(self):
        import experiments
        import experiments.schema
        import experiments.runner

        assert hasattr(experiments, "ExperimentResult")
        assert hasattr(experiments, "run_baseline_experiment")
        assert hasattr(experiments, "run_controlled_experiment")
