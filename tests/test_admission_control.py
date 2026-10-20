"""
File: tests/test_admission_control.py

1. What this file does:
   CPU-only unit tests for controller/admission_control.py.
2. Overall design:
   Uses pytest with synthetic (mock) inputs only — no GPU, no CUDA, no NVML.
   Covers: budget boundary conditions, ADMIT/DEFER/REJECT policy, safety margin,
   multi-workload accounting, double-counting prevention, release/cleanup,
   stale-measurement handling, concurrent thread-safety, MemoryEvent-schema
   compliance, and estimation-vs-observed-peak separation.
3. Elements in the file:
   Fixtures: gb() helper, various controller factories.
   Test functions: one per documented requirement (see below).
4. Final output of the file:
   Pytest outcomes that validate the admission-control engine without touching hardware.
"""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from controller.admission_control import AdmissionController, AdmissionDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gb(n: float) -> int:
    """Convert GiB to bytes."""
    return int(n * 1024 * 1024 * 1024)


def mb(n: float) -> int:
    """Convert MiB to bytes."""
    return int(n * 1024 * 1024)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_negative_budget_raises(self):
        with pytest.raises(ValueError, match="budget_bytes"):
            AdmissionController(budget_bytes=-1)

    def test_safety_margin_out_of_range_high(self):
        with pytest.raises(ValueError, match="safety_margin_ratio"):
            AdmissionController(budget_bytes=gb(6), safety_margin_ratio=1.0)

    def test_safety_margin_out_of_range_low(self):
        with pytest.raises(ValueError, match="safety_margin_ratio"):
            AdmissionController(budget_bytes=gb(6), safety_margin_ratio=-0.1)

    def test_max_single_workload_ratio_zero_raises(self):
        with pytest.raises(ValueError, match="max_single_workload_ratio"):
            AdmissionController(budget_bytes=gb(6), max_single_workload_ratio=0.0)

    def test_max_single_workload_ratio_above_one_raises(self):
        with pytest.raises(ValueError, match="max_single_workload_ratio"):
            AdmissionController(budget_bytes=gb(6), max_single_workload_ratio=1.01)

    def test_zero_budget_allowed_at_construction(self):
        # Zero budget is valid to construct; requests will be REJECTed.
        ctrl = AdmissionController(budget_bytes=0)
        assert ctrl.budget_bytes == 0

    def test_safe_budget_computed_correctly(self):
        # Budget 6 GiB, margin 10% → safe = 6 * 0.9 = 5.4 GiB
        ctrl = AdmissionController(budget_bytes=gb(6), safety_margin_ratio=0.10)
        assert ctrl.safe_budget_bytes == int(gb(6) * 0.90)


# ---------------------------------------------------------------------------
# REJECT policy (no lock needed — parameter-only decisions)
# ---------------------------------------------------------------------------

class TestRejectPolicy:
    def test_reject_zero_estimated_bytes(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        decision, event = ctrl.request("wl", estimated_bytes=0)
        assert decision == AdmissionDecision.REJECT

    def test_reject_negative_estimated_bytes(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        decision, _ = ctrl.request("wl", estimated_bytes=-mb(100))
        assert decision == AdmissionDecision.REJECT

    def test_reject_when_budget_is_zero(self):
        ctrl = AdmissionController(budget_bytes=0)
        decision, _ = ctrl.request("wl", estimated_bytes=mb(100))
        assert decision == AdmissionDecision.REJECT

    def test_reject_exceeds_max_single_workload_ratio(self):
        # Budget 6 GiB, max_single=80% → cap = 4.8 GiB
        # Request 5 GiB → REJECT (policy violation, not just headroom issue)
        ctrl = AdmissionController(
            budget_bytes=gb(6),
            safety_margin_ratio=0.0,
            max_single_workload_ratio=0.80,
        )
        decision, event = ctrl.request("big_wl", estimated_bytes=gb(5))
        assert decision == AdmissionDecision.REJECT
        meta = json.loads(event["metadata_json"])
        assert meta["reject_reason"] is not None
        assert "policy cap" in meta["reject_reason"]

    def test_reject_does_not_record_reservation(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        ctrl.request("wl", estimated_bytes=gb(5))  # over 80% cap with default budget
        # No reservation should be recorded after a reject
        assert "wl" not in ctrl.admitted_workloads

    def test_reject_differs_from_defer_semantically(self):
        """
        REJECT = policy violation (retrying with same params won't help).
        DEFER = temporary capacity issue (retrying after release can help).
        Both appear in different scenarios.
        """
        ctrl = AdmissionController(
            budget_bytes=gb(6),
            safety_margin_ratio=0.0,
            max_single_workload_ratio=0.80,
        )
        # This REJECTS because 5 GiB > 80% * 6 GiB (policy cap)
        dec_reject, _ = ctrl.request("too_big", estimated_bytes=gb(5))
        assert dec_reject == AdmissionDecision.REJECT

        # Admit a large workload to fill headroom, then try a reasonable one:
        ctrl2 = AdmissionController(
            budget_bytes=gb(6),
            safety_margin_ratio=0.0,
            max_single_workload_ratio=1.0,
        )
        ctrl2.request("filler", estimated_bytes=gb(5))  # fills 5 of 6 GiB
        dec_defer, _ = ctrl2.request("small", estimated_bytes=gb(2))  # 5+2=7>6 → DEFER
        assert dec_defer == AdmissionDecision.DEFER


# ---------------------------------------------------------------------------
# DEFER conditions
# ---------------------------------------------------------------------------

class TestDeferPolicy:
    def test_defer_when_headroom_insufficient(self):
        # Budget 6 GiB, margin 0%, reserve 5 GiB, request 2 GiB → only 1 GiB free → DEFER
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.0, max_single_workload_ratio=1.0
        )
        ctrl.request("existing", estimated_bytes=gb(5))
        decision, _ = ctrl.request("new", estimated_bytes=gb(2))
        assert decision == AdmissionDecision.DEFER

    def test_defer_does_not_record_reservation(self):
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.0, max_single_workload_ratio=1.0
        )
        ctrl.request("existing", estimated_bytes=gb(5))
        ctrl.request("new", estimated_bytes=gb(2))
        assert "new" not in ctrl.admitted_workloads

    def test_defer_becomes_admit_after_release(self):
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.0, max_single_workload_ratio=1.0
        )
        ctrl.request("existing", estimated_bytes=gb(5))
        dec_first, _ = ctrl.request("new", estimated_bytes=gb(2))
        assert dec_first == AdmissionDecision.DEFER

        ctrl.release("existing")
        dec_retry, _ = ctrl.request("new", estimated_bytes=gb(2))
        assert dec_retry == AdmissionDecision.ADMIT


# ---------------------------------------------------------------------------
# ADMIT conditions and reservation accounting
# ---------------------------------------------------------------------------

class TestAdmitAndAccounting:
    def test_admit_when_fits_comfortably(self):
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.10
        )
        # Safe = 5.4 GiB, request 1 GiB → fits
        decision, event = ctrl.request("wl_a", estimated_bytes=gb(1))
        assert decision == AdmissionDecision.ADMIT
        assert "wl_a" in ctrl.admitted_workloads

    def test_admit_at_exact_headroom_boundary(self):
        # Budget 6 GiB, margin 0%, safe = 6 GiB, request exactly 6 GiB
        ctrl = AdmissionController(
            budget_bytes=gb(6),
            safety_margin_ratio=0.0,
            max_single_workload_ratio=1.0,
        )
        decision, _ = ctrl.request("wl", estimated_bytes=gb(6))
        assert decision == AdmissionDecision.ADMIT

    def test_safety_margin_reduces_available_headroom(self):
        # Budget 6 GiB, margin 10% → safe = 5.4 GiB
        # Request 5.5 GiB → should DEFER (exceeds safe budget, fits within policy cap)
        ctrl = AdmissionController(
            budget_bytes=gb(6),
            safety_margin_ratio=0.10,
            max_single_workload_ratio=1.0,
        )
        decision, _ = ctrl.request("wl", estimated_bytes=int(gb(6) * 0.92))
        assert decision == AdmissionDecision.DEFER

    def test_multiple_workloads_reservations_sum_correctly(self):
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.0, max_single_workload_ratio=0.5
        )
        ctrl.request("wl_1", estimated_bytes=gb(2))
        ctrl.request("wl_2", estimated_bytes=gb(2))
        assert ctrl.reserved_total_bytes == gb(4)

    def test_third_workload_deferred_when_two_admitted(self):
        ctrl = AdmissionController(
            budget_bytes=gb(6),
            safety_margin_ratio=0.0,
            max_single_workload_ratio=0.5,
        )
        ctrl.request("wl_1", estimated_bytes=gb(2))
        ctrl.request("wl_2", estimated_bytes=gb(2))
        # 4 GiB used, 2 GiB left; request 3 GiB → DEFER
        dec, _ = ctrl.request("wl_3", estimated_bytes=gb(3))
        assert dec == AdmissionDecision.DEFER


# ---------------------------------------------------------------------------
# Release and cleanup
# ---------------------------------------------------------------------------

class TestRelease:
    def test_release_restores_headroom(self):
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.0, max_single_workload_ratio=1.0
        )
        ctrl.request("wl", estimated_bytes=gb(5))
        assert ctrl.reserved_total_bytes == gb(5)
        ctrl.release("wl")
        assert ctrl.reserved_total_bytes == 0

    def test_release_returns_true_for_known_workload(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        ctrl.request("wl", estimated_bytes=gb(1))
        assert ctrl.release("wl") is True

    def test_release_returns_false_for_unknown_workload(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        assert ctrl.release("nonexistent") is False

    def test_release_of_deferred_workload_is_safe(self):
        # Deferred workloads are never recorded; releasing their ID is a no-op.
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.0, max_single_workload_ratio=1.0
        )
        ctrl.request("filler", estimated_bytes=gb(5))
        ctrl.request("small", estimated_bytes=gb(2))  # DEFER
        result = ctrl.release("small")  # was never admitted
        assert result is False
        assert ctrl.reserved_total_bytes == gb(5)  # filler still reserved

    def test_double_release_is_safe(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        ctrl.request("wl", estimated_bytes=gb(1))
        ctrl.release("wl")
        # Second release should not raise or corrupt state
        result = ctrl.release("wl")
        assert result is False


# ---------------------------------------------------------------------------
# Double-counting prevention (same workload_id)
# ---------------------------------------------------------------------------

class TestDoubleCountPrevention:
    def test_same_id_admitted_twice_updates_not_adds(self):
        """Re-submitting an already-admitted workload_id updates the reservation,
        not adds to it. This prevents double-counting."""
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.0, max_single_workload_ratio=1.0
        )
        ctrl.request("wl", estimated_bytes=gb(1))
        ctrl.request("wl", estimated_bytes=gb(2))  # update, not additional
        assert ctrl.reserved_total_bytes == gb(2)  # not gb(3)

    def test_same_id_after_release_is_fresh_admission(self):
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.0, max_single_workload_ratio=1.0
        )
        ctrl.request("wl", estimated_bytes=gb(1))
        ctrl.release("wl")
        dec, _ = ctrl.request("wl", estimated_bytes=gb(1))
        assert dec == AdmissionDecision.ADMIT
        assert ctrl.reserved_total_bytes == gb(1)

    def test_reservation_downscale_admitted_when_headroom_tight(self):
        """When an admitted workload updates with a smaller estimate (downscaling),
        it reduces its reservation and must be ADMITTED even if headroom is tight."""
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.10, max_single_workload_ratio=1.0
        )
        # Safe budget = 5.4 GiB. Admitted workload reserves 5.0 GiB. Headroom = 0.4 GiB.
        dec_init, _ = ctrl.request("wl", estimated_bytes=gb(5))
        assert dec_init == AdmissionDecision.ADMIT
        assert ctrl.reserved_total_bytes == gb(5)

        # Workload shrinks from 5.0 GiB to 3.0 GiB. Delta = -2.0 GiB.
        # Should be admitted even though new estimate (3.0 GiB) > old headroom (0.4 GiB).
        dec_update, _ = ctrl.request("wl", estimated_bytes=gb(3))
        assert dec_update == AdmissionDecision.ADMIT
        assert ctrl.reserved_total_bytes == gb(3)

    def test_reservation_expansion_admitted_when_incremental_delta_fits(self):
        """When an admitted workload expands its reservation and the incremental delta
        fits within available headroom, it must be ADMITTED."""
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.10, max_single_workload_ratio=1.0
        )
        # Safe budget = 5.4 GiB. Workload reserves 3.0 GiB. Headroom = 2.4 GiB.
        ctrl.request("wl", estimated_bytes=gb(3))

        # Workload expands to 4.5 GiB. Delta = +1.5 GiB <= 2.4 GiB headroom.
        dec_update, _ = ctrl.request("wl", estimated_bytes=int(gb(4.5)))
        assert dec_update == AdmissionDecision.ADMIT
        assert ctrl.reserved_total_bytes == int(gb(4.5))

    def test_reservation_expansion_deferred_when_incremental_delta_exceeds_headroom(self):
        """When an admitted workload expands its reservation but the incremental delta
        exceeds available headroom, it must be DEFERRED and retain its previous reservation."""
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.10, max_single_workload_ratio=0.80
        )
        # Safe budget = 5.4 GiB. Max single workload = 4.8 GiB.
        # wl_1 reserves 4.0 GiB. wl_other reserves 1.0 GiB. Total reserved = 5.0 GiB.
        ctrl.request("wl_1", estimated_bytes=gb(4))
        ctrl.request("wl_other", estimated_bytes=gb(1))
        assert ctrl.reserved_total_bytes == gb(5)
        # Headroom = 5.4 - 5.0 = 0.4 GiB.

        # wl_1 requests 4.7 GiB (policy compliant <= 4.8 GiB).
        # Incremental delta = +0.7 GiB > 0.4 GiB headroom -> DEFER.
        dec_update, _ = ctrl.request("wl_1", estimated_bytes=int(gb(4.7)))
        assert dec_update == AdmissionDecision.DEFER
        # wl_1 must retain its existing 4.0 GiB reservation.
        assert ctrl.admitted_workloads["wl_1"] == gb(4)
        assert ctrl.reserved_total_bytes == gb(5)

    def test_request_log_decision_flag_false_does_not_append_to_log(self):
        """Calling request with log_decision=False should not append to decision_log."""
        ctrl = AdmissionController(budget_bytes=gb(6))
        dec, event = ctrl.request("wl", estimated_bytes=gb(1), log_decision=False)
        assert dec == AdmissionDecision.ADMIT
        assert len(ctrl.decision_log) == 0

        # Manually logging the event via log_decision_event should append it
        ctrl.log_decision_event(event)
        assert len(ctrl.decision_log) == 1


# ---------------------------------------------------------------------------
# Stale measurement handling
# ---------------------------------------------------------------------------

class TestStaleMeasurement:
    def test_stale_measurement_ignored_falls_back_to_reservations(self):
        """Stale NVML reading should be treated as unavailable; controller
        falls back to reservation-only headroom accounting."""
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.0, max_single_workload_ratio=1.0
        )
        # Provide an observed reading of 5.5 GiB but mark it as 60 s old (stale)
        dec, event = ctrl.request(
            "wl",
            estimated_bytes=gb(2),
            observed_device_used_bytes=int(gb(6) * 0.92),  # would block if fresh
            measurement_age_s=60.0,
            stale_threshold_s=30.0,
        )
        # With stale reading ignored and no other reservations, headroom = 6 GiB → ADMIT
        assert dec == AdmissionDecision.ADMIT
        meta = json.loads(event["metadata_json"])
        assert meta["measurement_is_stale"] is True

    def test_fresh_measurement_used_in_headroom(self):
        """A fresh NVML reading that exceeds reservations should be used."""
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.0, max_single_workload_ratio=1.0
        )
        # observed = 5 GiB (fresh), request = 2 GiB → headroom = 6-5 = 1 GiB < 2 GiB → DEFER
        dec, event = ctrl.request(
            "wl",
            estimated_bytes=gb(2),
            observed_device_used_bytes=gb(5),
            measurement_age_s=5.0,
            stale_threshold_s=30.0,
        )
        assert dec == AdmissionDecision.DEFER
        meta = json.loads(event["metadata_json"])
        assert meta["measurement_is_stale"] is False
        assert meta["observed_device_used_bytes"] == gb(5)

    def test_no_measurement_provided(self):
        """When no measurement is provided, controller uses reservations only."""
        ctrl = AdmissionController(
            budget_bytes=gb(6), safety_margin_ratio=0.0, max_single_workload_ratio=1.0
        )
        dec, _ = ctrl.request("wl", estimated_bytes=gb(1))
        assert dec == AdmissionDecision.ADMIT


# ---------------------------------------------------------------------------
# MemoryEvent schema compliance
# ---------------------------------------------------------------------------

class TestEventSchema:
    # The canonical MemoryEvent CSV columns (from nvml_monitor.py and schema.py).
    EXPECTED_FIELDS = {
        "timestamp", "pid", "gpu_id", "source", "event_type",
        "total_memory", "free_memory", "process_memory",
        "allocated_bytes", "reserved_bytes", "allocation_delta", "peak_memory",
        "workload_name", "batch_size", "precision", "metadata_json", "snapshot_path",
    }

    def test_admit_event_has_all_schema_fields(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        _, event = ctrl.request("wl", estimated_bytes=gb(1))
        assert set(event.keys()) == self.EXPECTED_FIELDS

    def test_reject_event_has_all_schema_fields(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        _, event = ctrl.request("wl", estimated_bytes=0)
        assert set(event.keys()) == self.EXPECTED_FIELDS

    def test_source_is_scheduler(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        _, event = ctrl.request("wl", estimated_bytes=gb(1))
        assert event["source"] == "scheduler"

    def test_event_type_is_admission(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        _, event = ctrl.request("wl", estimated_bytes=gb(1))
        assert event["event_type"] == "admission"

    def test_metadata_json_is_parseable(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        _, event = ctrl.request("wl", estimated_bytes=gb(1))
        meta = json.loads(event["metadata_json"])
        assert meta["decision"] == AdmissionDecision.ADMIT
        assert meta["workload_id"] == "wl"
        assert meta["estimated_bytes"] == gb(1)

    def test_workload_name_matches_workload_id(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        _, event = ctrl.request("my_workload", estimated_bytes=gb(1))
        assert event["workload_name"] == "my_workload"


# ---------------------------------------------------------------------------
# Estimated vs observed peak separation
# ---------------------------------------------------------------------------

class TestEstimationSeparation:
    def test_record_observed_peak_separate_from_estimate(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        ctrl.request("wl", estimated_bytes=gb(1))
        # Simulate measuring actual peak after execution
        ctrl.record_observed_peak("wl", observed_peak_bytes=gb(2))
        # Estimation error = observed - estimated = 2 GiB - 1 GiB = +1 GiB (underestimate)
        error = ctrl.get_estimation_error("wl")
        assert error == gb(1)

    def test_get_estimation_error_none_if_no_observation(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        ctrl.request("wl", estimated_bytes=gb(1))
        # Never called record_observed_peak
        assert ctrl.get_estimation_error("wl") is None

    def test_get_estimation_error_none_if_not_admitted(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        ctrl.record_observed_peak("wl", gb(1))
        # Never admitted → no ADMIT event in log → error not computable
        assert ctrl.get_estimation_error("wl") is None

    def test_estimated_bytes_not_in_reservations_as_observed(self):
        """The reservation dict stores estimated_bytes, not observed peaks.
        These must never be conflated."""
        ctrl = AdmissionController(budget_bytes=gb(6))
        ctrl.request("wl", estimated_bytes=gb(1))
        ctrl.record_observed_peak("wl", observed_peak_bytes=gb(3))
        # Reservation should still be the original estimate, not the observed peak
        assert ctrl.admitted_workloads.get("wl") == gb(1)


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------

class TestDecisionLog:
    def test_log_accumulates_all_decisions(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        ctrl.request("wl_a", estimated_bytes=gb(1))
        ctrl.request("wl_b", estimated_bytes=gb(1))
        ctrl.request("bad", estimated_bytes=0)
        assert len(ctrl.decision_log) == 3

    def test_log_returned_as_copy(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        ctrl.request("wl", estimated_bytes=gb(1))
        log = ctrl.decision_log
        log.clear()  # mutating the copy
        assert len(ctrl.decision_log) == 1  # original unaffected

    def test_clear_log_empties_log(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        ctrl.request("wl", estimated_bytes=gb(1))
        ctrl.clear_log()
        assert ctrl.decision_log == []

    def test_clear_log_does_not_clear_reservations(self):
        ctrl = AdmissionController(budget_bytes=gb(6))
        ctrl.request("wl", estimated_bytes=gb(1))
        ctrl.clear_log()
        assert ctrl.reserved_total_bytes == gb(1)


# ---------------------------------------------------------------------------
# Concurrent thread-safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_admission_no_over_budget(self):
        """
        Launch N threads that all simultaneously attempt to admit a workload.
        With a budget that can only hold K workloads, at most K should be admitted.
        """
        budget = gb(6)
        margin = 0.0
        per_workload = gb(2)  # 2 GiB each → max 3 fit in 6 GiB
        max_admissible = budget // per_workload  # = 3

        ctrl = AdmissionController(
            budget_bytes=budget,
            safety_margin_ratio=margin,
            max_single_workload_ratio=1.0,
        )

        results = []
        lock = threading.Lock()

        def try_admit(wid):
            dec, _ = ctrl.request(wid, estimated_bytes=per_workload)
            with lock:
                results.append(dec)

        threads = [
            threading.Thread(target=try_admit, args=(f"wl_{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        admitted_count = results.count(AdmissionDecision.ADMIT)
        deferred_count = results.count(AdmissionDecision.DEFER)

        # No more than max_admissible should have been admitted
        assert admitted_count <= max_admissible, (
            f"Over-admitted: {admitted_count} admitted, max allowed = {max_admissible}"
        )
        # Total admitted + deferred = 10 (no requests lost)
        assert admitted_count + deferred_count == 10

        # Reservation total must not exceed budget
        assert ctrl.reserved_total_bytes <= budget

    def test_concurrent_release_and_admit(self):
        """After releasing under concurrent load, the freed headroom should
        eventually allow a previously-deferred workload to be admitted."""
        ctrl = AdmissionController(
            budget_bytes=gb(4),
            safety_margin_ratio=0.0,
            max_single_workload_ratio=1.0,
        )
        # Fill budget
        ctrl.request("filler_a", estimated_bytes=gb(2))
        ctrl.request("filler_b", estimated_bytes=gb(2))

        released = threading.Event()

        def releaser():
            time.sleep(0.05)
            ctrl.release("filler_a")
            released.set()

        t = threading.Thread(target=releaser)
        t.start()
        released.wait(timeout=2.0)

        dec, _ = ctrl.request("newcomer", estimated_bytes=gb(2))
        assert dec == AdmissionDecision.ADMIT
        t.join()
