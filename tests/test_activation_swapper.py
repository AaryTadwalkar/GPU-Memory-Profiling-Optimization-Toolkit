"""
File: tests/test_activation_swapper.py

1. What this file does:
   Provides comprehensive unit and integration tests for the Active DL Training Tiered
   Activation Swapper (TieredActivationManager, PackedActivation, SwapEventRecord, and
   ActivationSwapSummary). Validates gradient equivalence, threshold filtering, pass-through
   disabled mode, serialization, context manager lifecycles, and event chronology.

2. Overall design:
   Executes 100% on CPU using PyTorch CPU tensors and deterministic seeds, strictly avoiding
   hardware allocation or CUDA operations. Structured into logical test classes covering
   mathematical correctness, memory threshold behavior, event ordering, and serialization.

3. Elements in the file:
   - TestGradientEquivalence: Validates bit-for-bit gradient parity with and without swapping.
   - TestThresholdFiltering: Validates size filtering to prevent offloading small activations.
   - TestDisabledMode: Validates zero-overhead pass-through when disabled.
   - TestSummaryAndSerialization: Validates summary metrics and JSON export.
   - TestContextManagerLifecycle: Validates proper hook activation and unregistration.
   - TestEventChronology: Validates paired OFFLOAD and PREFETCH event pairs and ordering.
   - TestDirectHooksAndReset: Validates manual hook invocations and state resetting.
   - TestLive1MinBenchmark: Validates the multi-phase streaming 60s benchmark engine and OOM defense.

4. Final output of the file:
   Pytest test suite producing verifiable pass/fail assertions for all activation swapper features.
"""

import copy
import json
from pathlib import Path
import sys
import time
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizer.activation_swapper import (
    PackedActivation,
    SwapEventRecord,
    ActivationSwapSummary,
    TieredActivationManager,
)
from workloads.activation_swap_demo import (
    SyntheticMLP,
    BenchmarkStepRecord,
    LiveBenchmarkSummary,
    run_live_1min_benchmark,
)


class TestGradientEquivalence:
    """
    1. The use of class:
       Tests the core mathematical requirement that TieredActivationManager produces
       gradients identical to standard PyTorch autograd execution.

    2. What the class contain:
       Test methods verifying gradient equivalence across multi-layer networks with various
       non-linearities and tensor shapes.

    3. Technologies used and why:
       PyTorch autograd, `torch.allclose`, and deep copying for isolated comparative execution.

    4. Key design and overall design of the class:
       Instantiates two identical networks, executes baseline and swapped passes, and asserts
       strict mathematical equality across all layer weights and biases.
    """

    def test_gradient_equivalence(self) -> None:
        """
        1. The use of function:
           Verifies that training with TieredActivationManager produces gradients that match
           unswapped baseline training within floating-point tolerance (torch.allclose atol=1e-6).

        2. How does the specific function work:
           Initializes two identical networks from fixed random seed. Runs forward and backward
           passes on the same input batch. Compares gradients across all named parameters.

        3. Concepts used in the function:
           Autograd equivalence, gradient matching, numerical tolerance testing.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: unverified gradient tensors from two separate execution passes.
           After: assertion confirming gradients are identical within atol=1e-6 across all layers.
        """
        torch.manual_seed(123)
        model_base = SyntheticMLP(input_dim=64, hidden_dim=128, depth=4)
        model_swap = copy.deepcopy(model_base)

        x_input = torch.randn(16, 64)

        # Baseline execution
        out_base = model_base(x_input)
        loss_base = out_base.sum()
        loss_base.backward()

        # Swapped execution
        manager = TieredActivationManager(
            pressure_threshold_bytes=0,
            min_tensor_size_bytes=1,
            use_pinned_memory=False,
            enabled=True,
            simulate_on_cpu=True,
        )

        with manager:
            out_swap = model_swap(x_input)
            loss_swap = out_swap.sum()
        loss_swap.backward()

        # Assert output equality
        assert torch.allclose(out_base, out_swap, atol=1e-6)
        assert abs(loss_base.item() - loss_swap.item()) < 1e-6

        # Assert gradient equality for all parameters
        for (n1, p1), (n2, p2) in zip(model_base.named_parameters(), model_swap.named_parameters()):
            assert p1.grad is not None and p2.grad is not None, f"Gradient missing for {n1}"
            assert torch.allclose(p1.grad, p2.grad, atol=1e-6), (
                f"Gradient mismatch for {n1}: max diff = {torch.max(torch.abs(p1.grad - p2.grad)).item()}"
            )


class TestThresholdFiltering:
    """
    1. The use of class:
       Tests the minimum tensor size filtering logic of TieredActivationManager.

    2. What the class contain:
       Test methods asserting that small activations stay resident while large activations
       are offloaded.

    3. Technologies used and why:
       PyTorch tensors of deliberate byte sizes to test boundary conditions of `min_tensor_size_bytes`.

    4. Key design and overall design of the class:
       Applies hooks to tensors below and above the size threshold and inspects `was_swapped`
       and `total_tensors_offloaded`.
    """

    def test_size_threshold_filtering(self) -> None:
        """
        1. The use of function:
           Verifies tensors smaller than `min_tensor_size_bytes` stay resident and are not offloaded.

        2. How does the specific function work:
           Configures manager with a 256KB threshold. Packs a small tensor (400 bytes) and a
           large tensor (1MB). Asserts small tensor is not swapped and large tensor is swapped.

        3. Concepts used in the function:
           Byte size thresholding, selective activation offloading, memory optimization heuristics.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: uninspected tensor packing operations.
           After: assertions confirming small tensors remain resident while large tensors offload.
        """
        threshold = 256 * 1024  # 256 KB
        manager = TieredActivationManager(
            pressure_threshold_bytes=0,
            min_tensor_size_bytes=threshold,
            enabled=True,
            simulate_on_cpu=True,
        )

        # Small tensor: 10 x 10 float32 = 400 bytes (< 256 KB)
        small_tensor = torch.randn(10, 10, dtype=torch.float32)
        packed_small = manager.pack_hook(small_tensor)
        assert packed_small[1] is False  # was_swapped == False
        assert manager.total_tensors_offloaded == 0

        # Large tensor: 512 x 512 float32 = 1,048,576 bytes = 1 MB (> 256 KB)
        large_tensor = torch.randn(512, 512, dtype=torch.float32)
        packed_large = manager.pack_hook(large_tensor)
        assert packed_large[1] is True  # was_swapped == True
        assert manager.total_tensors_offloaded == 1


class TestDisabledMode:
    """
    1. The use of class:
       Tests pass-through behavior when TieredActivationManager is initialized with `enabled=False`.

    2. What the class contain:
       Test methods ensuring zero offloading, complete pass-through, and gradient safety when disabled.

    3. Technologies used and why:
       PyTorch neural network forward-backward pass wrapped with a disabled manager.

    4. Key design and overall design of the class:
       Executes training loop with `enabled=False` and verifies that zero tensors are offloaded.
    """

    def test_disabled_mode(self) -> None:
        """
        1. The use of function:
           Verifies that `enabled=False` acts as a pure pass-through with 0 offloaded tensors.

        2. How does the specific function work:
           Instantiates manager with `enabled=False`. Executes forward and backward pass on
           SyntheticMLP. Checks that `total_tensors_offloaded` is 0 and summary reflects 0 bytes.

        3. Concepts used in the function:
           Feature toggle, disabled state pass-through, zero overhead verification.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: training pass with disabled activation manager.
           After: assertion confirming exactly 0 offloads and valid loss gradients.
        """
        manager = TieredActivationManager(
            pressure_threshold_bytes=0,
            min_tensor_size_bytes=1,
            enabled=False,
            simulate_on_cpu=True,
        )

        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
        x = torch.randn(8, 32)

        with manager:
            out = model(x)
            loss = out.sum()
        loss.backward()

        summary = manager.get_summary()
        assert summary.total_tensors_offloaded == 0
        assert summary.total_bytes_offloaded == 0
        assert summary.peak_vram_reduction_pct == 0.0
        assert len(summary.events) == 0


class TestSummaryAndSerialization:
    """
    1. The use of class:
       Tests summary aggregation, dictionary conversion, and JSON serialization of activation metrics.

    2. What the class contain:
       Test methods verifying `ActivationSwapSummary.to_dict()` and `json.dumps()` roundtrip validity.

    3. Technologies used and why:
       Python `json` library and dataclass serialization methods.

    4. Key design and overall design of the class:
       Executes a training pass with offloading, calls `get_summary()`, converts to dictionary,
       and validates JSON serialization and schema completeness.
    """

    def test_summary_and_serialization(self) -> None:
        """
        1. The use of function:
           Verifies `ActivationSwapSummary` calculations and JSON serialization.

        2. How does the specific function work:
           Executes a training step under simulation mode, extracts summary, converts to dictionary,
           verifies key fields (tensors offloaded, durations, reduction pct), and checks JSON serializability.

        3. Concepts used in the function:
           Dataclass serialization, JSON formatting, metrics verification.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: raw manager telemetry.
           After: verified JSON-compliant dictionary with accurate metric values.
        """
        manager = TieredActivationManager(
            pressure_threshold_bytes=0,
            min_tensor_size_bytes=1,
            enabled=True,
            simulate_on_cpu=True,
        )

        model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 10))
        x = torch.randn(8, 32)

        with manager:
            out = model(x)
            loss = out.sum()
        loss.backward()

        summary = manager.get_summary()
        assert isinstance(summary, ActivationSwapSummary)
        assert summary.total_tensors_intercepted > 0
        assert summary.total_tensors_offloaded > 0
        assert summary.total_bytes_offloaded > 0
        assert summary.total_bytes_prefetched == summary.total_bytes_offloaded
        assert summary.peak_vram_reduction_pct > 0.0

        summary_dict = summary.to_dict()
        assert isinstance(summary_dict, dict)
        assert "total_tensors_intercepted" in summary_dict
        assert "events" in summary_dict

        # Verify JSON serializability
        serialized_json = json.dumps(summary_dict)
        assert len(serialized_json) > 0
        deserialized = json.loads(serialized_json)
        assert deserialized["total_tensors_offloaded"] == summary.total_tensors_offloaded


class TestContextManagerLifecycle:
    """
    1. The use of class:
       Tests the registration and unregistration lifecycle of TieredActivationManager context manager.

    2. What the class contain:
       Test methods asserting hooks are actively engaged inside the context and dormant outside.

    3. Technologies used and why:
       PyTorch autograd graph saved tensors hook state tracking.

    4. Key design and overall design of the class:
       Measures tensor interception count during `with` block vs outside the block.
    """

    def test_context_manager_lifecycle(self) -> None:
        """
        1. The use of function:
           Verifies hooks are active within `with` block and properly unregistered after exiting.

        2. How does the specific function work:
           Runs an operation inside `with manager:`, checks intercepted count, then runs an
           identical operation outside `with manager:`, asserting the intercepted count does not increase.

        3. Concepts used in the function:
           Context manager protocol, hook registration, deterministic resource cleanup.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: unverified hook scope bounds.
           After: assertion confirming operations outside context manager do not trigger hooks.
        """
        manager = TieredActivationManager(
            pressure_threshold_bytes=0,
            min_tensor_size_bytes=1,
            enabled=True,
            simulate_on_cpu=True,
        )

        # 1. Operation inside context
        with manager:
            a1 = torch.randn(10, 10, requires_grad=True)
            b1 = torch.relu(a1)
            loss1 = b1.sum()

        count_inside = manager.total_tensors_intercepted
        assert count_inside > 0

        # 2. Operation outside context
        a2 = torch.randn(10, 10, requires_grad=True)
        b2 = torch.relu(a2)
        loss2 = b2.sum()

        count_outside = manager.total_tensors_intercepted
        assert count_outside == count_inside, "Hooks remained active outside context manager!"


class TestEventChronology:
    """
    1. The use of class:
       Tests the chronological pairing and ordering of OFFLOAD and PREFETCH events.

    2. What the class contain:
       Test methods ensuring every offloaded tensor produces an exactly matching prefetch event.

    3. Technologies used and why:
       Event log inspection, tensor ID matching, and timestamp comparisons.

    4. Key design and overall design of the class:
       Extracts all SwapEventRecord objects, groups by event_type, and validates tensor ID parity.
    """

    def test_event_chronology(self) -> None:
        """
        1. The use of function:
           Verifies that every OFFLOAD event has a corresponding PREFETCH event during backward pass.

        2. How does the specific function work:
           Executes forward and backward pass. Inspects event records. Verifies count of OFFLOAD
           events equals count of PREFETCH events, matching tensor IDs, and OFFLOAD precedes PREFETCH.

        3. Concepts used in the function:
           Event tracing, causal chronology validation, bijection between offload and prefetch.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: unordered event history.
           After: assertion confirming pairwise consistency and temporal ordering of offload/prefetch.
        """
        manager = TieredActivationManager(
            pressure_threshold_bytes=0,
            min_tensor_size_bytes=1,
            enabled=True,
            simulate_on_cpu=True,
        )

        model = nn.Sequential(nn.Linear(20, 40), nn.ReLU(), nn.Linear(40, 10))
        x = torch.randn(8, 20)

        with manager:
            out = model(x)
            loss = out.sum()
        loss.backward()

        offload_events = [e for e in manager.events if e.event_type == "OFFLOAD"]
        prefetch_events = [e for e in manager.events if e.event_type == "PREFETCH"]

        assert len(offload_events) > 0, "No offload events recorded"
        assert len(offload_events) == len(prefetch_events), (
            f"Mismatch: {len(offload_events)} offloads vs {len(prefetch_events)} prefetches"
        )

        offload_ids = {e.tensor_id for e in offload_events}
        prefetch_ids = {e.tensor_id for e in prefetch_events}
        assert offload_ids == prefetch_ids, "Offloaded tensor IDs do not match prefetched IDs"

        # Check timestamp ordering for each tensor
        offload_map = {e.tensor_id: e.timestamp for e in offload_events}
        for pe in prefetch_events:
            assert pe.timestamp >= offload_map[pe.tensor_id], (
                f"Prefetch for tensor {pe.tensor_id} occurred before offload!"
            )


class TestDirectHooksAndReset:
    """
    1. The use of class:
       Tests direct invocation of pack_hook / unpack_hook and manager state reset functionality.

    2. What the class contain:
       Test methods asserting `reset()` resets all counters and `PackedActivation` preserves attributes.

    3. Technologies used and why:
       Direct unit testing of hook methods and state mutators without autograd execution.

    4. Key design and overall design of the class:
       Validates state transitions on isolated tensors and verifies manager can be reused.
    """

    def test_packed_activation_subclass(self) -> None:
        """
        1. The use of function:
           Verifies PackedActivation tuple subclass unpacks cleanly as a 5-tuple and retains tensor_id.

        2. How does the specific function work:
           Instantiates PackedActivation, unpacks via 5-tuple destructuring, checks length,
           and checks `tensor_id` attribute.

        3. Concepts used in the function:
           Tuple destructuring, Python object attribute inspection.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: PackedActivation instance.
           After: assertions confirming 5-tuple unpacking and tensor_id access.
        """
        tensor = torch.zeros(4, 4)
        packed = PackedActivation(tensor, True, "cpu", (4, 4), torch.float32, tensor_id=99)

        stored, swapped, dev, shp, dt = packed
        assert stored.shape == (4, 4)
        assert swapped is True
        assert dev == "cpu"
        assert shp == (4, 4)
        assert dt == torch.float32
        assert len(packed) == 5
        assert packed.tensor_id == 99

    def test_reset_functionality(self) -> None:
        """
        1. The use of function:
           Verifies that `manager.reset()` clears all metrics, events, and counters back to zero.

        2. How does the specific function work:
           Populates manager state via pack_hook, verifies nonzero metrics, calls `reset()`,
           and asserts all counters and events are reset to initial empty values.

        3. Concepts used in the function:
           State clearing, idempotency, lifecycle management.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: manager with accumulated telemetry metrics.
           After: clean manager with 0 metrics and empty event list.
        """
        manager = TieredActivationManager(
            pressure_threshold_bytes=0,
            min_tensor_size_bytes=1,
            enabled=True,
            simulate_on_cpu=True,
        )

        tensor = torch.randn(100, 100)
        manager.pack_hook(tensor)
        assert manager.total_tensors_intercepted == 1
        assert manager.total_tensors_offloaded == 1
        assert len(manager.events) == 1

        manager.reset()
        assert manager.total_tensors_intercepted == 0
        assert manager.total_tensors_offloaded == 0
        assert manager.total_bytes_offloaded == 0
        assert manager.total_bytes_prefetched == 0
        assert manager.offload_duration_ms == 0.0
        assert manager.prefetch_duration_ms == 0.0
        assert len(manager.events) == 0

    def test_direct_unpack_unswapped(self) -> None:
        """
        1. The use of function:
           Verifies direct unpack_hook returns unswapped resident tensors without transfer overhead.

        2. How does the specific function work:
           Calls pack_hook on a small tensor below the threshold (not swapped), passes packed
           result to unpack_hook, and asserts returned tensor matches original without prefetch events.

        3. Concepts used in the function:
           Unswapped pass-through, hook bypass testing.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: resident unswapped tensor.
           After: restored tensor returned with zero prefetch events recorded.
        """
        manager = TieredActivationManager(
            pressure_threshold_bytes=0,
            min_tensor_size_bytes=1024 * 1024,
            enabled=True,
            simulate_on_cpu=True,
        )

        tensor = torch.randn(10, 10)
        packed = manager.pack_hook(tensor)
        assert packed[1] is False

        restored = manager.unpack_hook(packed)
        assert torch.equal(tensor, restored)
        assert manager.total_bytes_prefetched == 0
        assert len([e for e in manager.events if e.event_type == "PREFETCH"]) == 0


class TestLive1MinBenchmark:
    """
    1. The use of class:
       Tests the real-time live training benchmark orchestration engine (`run_live_1min_benchmark`),
       telemetry recording (`BenchmarkStepRecord`), and aggregate reporting (`LiveBenchmarkSummary`).

    2. What the class contain:
       Test methods verifying two-phase execution, dynamic OOM crash detection, batch scaling,
       mathematical gradient equivalence, and time-series serialization.

    3. Technologies used and why:
       PyTorch autograd, pytest assertions, and CPU simulation for safe, deterministic testing.

    4. Key design and overall design of the class:
       Validates that Phase 1 appropriately stresses memory capacity and captures crashes,
       Phase 2 engages TieredActivationManager to ensure zero crashes at 2x batch throughput,
       and all telemetry records conform to streaming UI schemas.
    """

    def test_run_live_benchmark_execution(self) -> None:
        """
        1. The use of function:
           Tests normal multi-phase execution of the live training benchmark without forced crash.

        2. How does the specific function work:
           Invokes `run_live_1min_benchmark` with short phase durations and safe threshold,
           collects callback ticks, and verifies records, batch sizes, and gradient equivalence.

        3. Concepts used in the function:
           Multi-phase training execution, time-series callback recording, gradient verification.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unexecuted benchmark.
           After: Populated LiveBenchmarkSummary with verified mathematical gradient match.
        """
        collected_records = []

        summary = run_live_1min_benchmark(
            device_name="cpu",
            simulate=True,
            phase1_duration_s=1.2,
            phase2_duration_s=1.2,
            batch_p1=16,
            batch_p2=32,
            model_type="mlp",
            vram_budget_mb=6140.0,
            offload_threshold_mb=4900.0,
            simulate_oom_on_baseline=False,
            step_interval_s=0.3,
            callback=lambda r: collected_records.append(r),
        )

        assert isinstance(summary, LiveBenchmarkSummary)
        assert summary.total_duration_s > 0.0
        assert len(collected_records) >= 2
        assert summary.batch_speedup_ratio == 2.0
        assert summary.gradient_equivalence is True
        assert summary.max_gradient_diff == 0.0

        p1_records = [r for r in collected_records if r.phase_id == 1]
        p2_records = [r for r in collected_records if r.phase_id == 2]
        assert len(p1_records) >= 1
        assert len(p2_records) >= 1
        assert p1_records[0].batch_size == 16
        assert p2_records[0].batch_size == 32

    def test_run_live_benchmark_oom_crash(self) -> None:
        """
        1. The use of function:
           Tests that the live benchmark engine accurately detects and handles an OOM abort in Phase 1,
           gracefully halts baseline training, and proceeds to Phase 2 with TieredActivationManager.

        2. How does the specific function work:
           Configures `simulate_oom_on_baseline=True` with short duration and fast ramp,
           asserts that Phase 1 records an OOM abort, and confirms Phase 2 completes stably.

        3. Concepts used in the function:
           OOM abort interception, fault recovery, tiered memory protection.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unexecuted benchmark.
           After: Verified crash detection in Phase 1 and successful recovery in Phase 2.
        """
        summary = run_live_1min_benchmark(
            device_name="cpu",
            simulate=True,
            phase1_duration_s=2.5,
            phase2_duration_s=1.2,
            batch_p1=32,
            batch_p2=64,
            model_type="resnet",
            vram_budget_mb=4096.0,
            offload_threshold_mb=3200.0,
            simulate_oom_on_baseline=True,
            step_interval_s=0.3,
        )

        assert summary.phase1_crashed_oom is True
        assert summary.phase2_duration_s > 0.0
        assert summary.batch_speedup_ratio == 2.0
        assert summary.gradient_equivalence is True

    def test_benchmark_step_record_to_dict(self) -> None:
        """
        1. The use of function:
           Tests serialization of BenchmarkStepRecord into a standard dictionary.

        2. How does the specific function work:
           Instantiates BenchmarkStepRecord, calls `to_dict()`, and verifies all dictionary keys.

        3. Concepts used in the function:
           Dataclass serialization and schema conformity.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Typed dataclass.
           After: Verified dictionary representation.
        """
        rec = BenchmarkStepRecord(
            timestamp=1000.0,
            elapsed_s=12.5,
            phase_id=1,
            phase_name="Phase 1: Baseline",
            step=5,
            batch_size=32,
            vram_used_mb=5400.0,
            host_ram_used_mb=0.0,
            status="RUNNING",
            tensors_swapped=0,
            bytes_swapped=0,
            gradient_matched=True,
        )
        d = rec.to_dict()
        assert d["elapsed_s"] == 12.5
        assert d["batch_size"] == 32
        assert d["vram_used_mb"] == 5400.0
        assert d["status"] == "RUNNING"

