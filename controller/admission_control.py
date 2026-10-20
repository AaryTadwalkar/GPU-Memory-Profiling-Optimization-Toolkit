"""
File: controller/admission_control.py

1. What this file does:
   Implements a cooperative, best-effort GPU workload admission controller that
   decides whether an incoming workload should be ADMIT, DEFER, or REJECT based
   on a configured memory budget, a safety margin, existing reservations, and an
   optional device-usage observation supplied by the caller.

2. Overall design:
   A single class (AdmissionController) maintains an internal accounting dictionary
   of admitted workload reservations keyed by workload_id. A threading.Lock
   serialises concurrent admission requests to prevent two callers from
   simultaneously observing sufficient headroom and both being admitted beyond the
   budget.

   COOPERATIVE AND BEST-EFFORT DISCLAIMER:
   This controller tracks only reservations made through it. It cannot prevent
   external processes, the CUDA driver, NVML-invisible workloads, or unrelated
   applications from consuming GPU memory. External allocations can invalidate
   any estimate between the headroom check and workload launch. Use this controller
   to coordinate workloads that voluntarily go through it; do not rely on it for
   hard VRAM isolation or guaranteed OOM prevention.

   Decision semantics:
   - ADMIT  : The request fits within the configured safe headroom. A reservation
              is recorded and the workload may proceed.
   - DEFER  : The request is policy-compliant but headroom is temporarily
              insufficient (e.g., another admitted workload is still running).
              The caller should retry later after calling release() on finished
              workloads.
   - REJECT : The request violates a configured policy regardless of current
              capacity. Retrying under the same configuration will not change
              the outcome.

3. Elements in the file:
   - AdmissionDecision: String constants for ADMIT, DEFER, REJECT.
   - AdmissionController: Thread-safe controller with budget, reservations, log.

4. Final output of the file:
   Returns (decision_string, event_dict) from request(). Maintains an auditable
   decision log accessible via the decision_log property.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root on sys.path — same pattern used by all other modules.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schema import MemoryEvent  # noqa: E402  (import after sys.path update)


# ---------------------------------------------------------------------------
# Decision constants
# ---------------------------------------------------------------------------

class AdmissionDecision:
    """
    1. The use of class:
       Provides named string constants for the three admission outcomes.
    2. What the class contains:
       Three class-level string attributes: ADMIT, DEFER, REJECT.
    3. Technologies used and why:
       Plain class (no Enum dependency) to stay consistent with the existing
       codebase style and avoid import overhead.
    4. Key design:
       Constants are compared with == in all code paths; callers should import
       and use these constants rather than raw strings.
    """
    ADMIT = "ADMIT"
    DEFER = "DEFER"
    REJECT = "REJECT"


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class AdmissionController:
    """
    1. The use of class:
       Cooperative, best-effort GPU workload admission controller.
    2. What the class contains:
       A configured memory budget (bytes), safety margin, per-workload maximum,
       a threading.Lock for concurrent-safety, a reservation dict
       {workload_id -> estimated_bytes}, a separate observed-peaks dict
       {workload_id -> observed_peak_bytes}, and a decision log list.
    3. Technologies used and why:
       threading.Lock: Prevents two concurrent calls from both seeing sufficient
           headroom and both being admitted, cumulatively exceeding the budget.
       Pure Python arithmetic: Zero dependency on GPU, CUDA, or NVML at module
           level, keeping the controller importable and testable on CPU-only
           machines.
       MemoryEvent schema: Decision records use source="scheduler",
           event_type="admission" so the dashboard can display them.
    4. Key design and overall design of the class:
       COOPERATIVE AND BEST-EFFORT — see module docstring for the disclaimer.

       ADMIT / DEFER / REJECT semantics:
         REJECT conditions (policy violations — retrying will not change outcome):
           - estimated_bytes <= 0 (invalid request).
           - budget_bytes == 0 (no budget configured).
           - estimated_bytes > budget_bytes * max_single_workload_ratio
             (single request exceeds the per-workload policy cap).

         DEFER conditions (temporary — retry after workloads finish):
           - Request is policy-compliant but available headroom < estimated_bytes.

         ADMIT conditions:
           - Request is policy-compliant and headroom >= estimated_bytes.

       Estimated memory (estimated_bytes from request()) is kept strictly
       separate from observed peak memory (supplied via record_observed_peak()).
       These are stored in different dicts and never conflated. The difference
       can be retrieved via get_estimation_error().
    """

    def __init__(
        self,
        budget_bytes: int,
        safety_margin_ratio: float = 0.10,
        max_single_workload_ratio: float = 0.80,
        gpu_id: int = 0,
    ) -> None:
        """
        1. The use of function:
           Initialises the controller with a configured memory budget and policy.
        2. How does the specific function work:
           Validates parameter ranges, stores budget and policy values, and
           initialises the thread lock, the reservations dict, the observed-peaks
           dict, and the decision log.
        3. Concepts used in the function:
           Resource budget, safety margin, per-workload policy cap.
        4. Places where this function is imported and used:
           Instantiated by workload launchers or experiment scripts before
           submitting any workload. One controller instance per experiment run
           or admission session is typical.
        5. Final output of the function, before and after:
           Before: no controller.
           After: a configured, empty AdmissionController ready to accept requests.

        Args:
            budget_bytes:
                Configured memory budget in bytes. All admission decisions are
                made relative to this value. The caller is responsible for setting
                it at or below the actual device capacity; the controller does not
                verify or enforce this against real hardware.
            safety_margin_ratio:
                Fraction of budget reserved as a safety buffer (default 0.10 = 10%).
                Safe budget = budget_bytes * (1 - safety_margin_ratio).
                Accepted range: [0.0, 1.0).
            max_single_workload_ratio:
                Maximum fraction of the total budget a single workload may request
                (default 0.80 = 80%). Requests exceeding this are REJECTED as a
                policy violation regardless of current headroom.
                Accepted range: (0.0, 1.0].
            gpu_id:
                GPU device index recorded in emitted MemoryEvent records. Does not
                affect any computation; purely a metadata label.
        """
        if budget_bytes < 0:
            raise ValueError(
                f"budget_bytes must be non-negative; got {budget_bytes}"
            )
        if not 0.0 <= safety_margin_ratio < 1.0:
            raise ValueError(
                f"safety_margin_ratio must be in [0.0, 1.0); got {safety_margin_ratio}"
            )
        if not 0.0 < max_single_workload_ratio <= 1.0:
            raise ValueError(
                f"max_single_workload_ratio must be in (0.0, 1.0]; got {max_single_workload_ratio}"
            )

        self._budget_bytes: int = budget_bytes
        self._safety_margin_ratio: float = safety_margin_ratio
        self._max_single_workload_ratio: float = max_single_workload_ratio
        self._gpu_id: int = gpu_id

        # Serialises concurrent admission requests.
        self._lock: threading.Lock = threading.Lock()

        # Primary resource accounting: workload_id -> estimated_bytes.
        # Released workloads are removed from this dict.
        self._reservations: Dict[str, int] = {}

        # Post-execution observed peaks, kept strictly separate from estimates.
        # workload_id -> observed_peak_bytes.
        self._observed_peaks: Dict[str, int] = {}

        # Append-only list of all admission-decision event dicts.
        self._decision_log: List[dict] = []

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def budget_bytes(self) -> int:
        """
        1. The use of function:
           Returns the total configured budget in bytes.
        2. How does the specific function work:
           Returns the stored _budget_bytes value.
        3. Concepts used in the function:
           Property accessor.
        4. Places where this function is imported and used:
           Used by tests and experiment scripts to inspect the controller config.
        5. Final output:
           Integer budget value in bytes.
        """
        return self._budget_bytes

    @property
    def safety_margin_ratio(self) -> float:
        """
        1. The use of function:
           Returns the configured safety margin ratio.
        2. How does the specific function work:
           Returns _safety_margin_ratio.
        3. Concepts used in the function:
           Property accessor.
        4. Places where this function is imported and used:
           Used by tests to verify policy configuration.
        5. Final output:
           Float in [0.0, 1.0).
        """
        return self._safety_margin_ratio

    @property
    def safe_budget_bytes(self) -> int:
        """
        1. The use of function:
           Returns the usable budget after subtracting the safety margin.
        2. How does the specific function work:
           Computes int(budget_bytes * (1 - safety_margin_ratio)).
        3. Concepts used in the function:
           Safety margin, headroom reservation.
        4. Places where this function is imported and used:
           Called by _compute_headroom() and exposed for dashboard display.
        5. Final output:
           Integer safe budget in bytes.
        """
        return int(self._budget_bytes * (1.0 - self._safety_margin_ratio))

    @property
    def reserved_total_bytes(self) -> int:
        """
        1. The use of function:
           Returns the total bytes currently reserved by all admitted workloads.
        2. How does the specific function work:
           Acquires the lock and sums the reservation values.
        3. Concepts used in the function:
           Thread-safe aggregation.
        4. Places where this function is imported and used:
           Used by tests and dashboard to inspect current load.
        5. Final output:
           Integer sum of all active reservations in bytes.
        """
        with self._lock:
            return sum(self._reservations.values())

    @property
    def admitted_workloads(self) -> Dict[str, int]:
        """
        1. The use of function:
           Returns a snapshot copy of the current reservations dict.
        2. How does the specific function work:
           Acquires the lock, returns a shallow copy so callers cannot mutate
           internal state.
        3. Concepts used in the function:
           Defensive copy, thread safety.
        4. Places where this function is imported and used:
           Inspected by experiment scripts and tests.
        5. Final output:
           Dict of {workload_id: estimated_bytes} for all currently admitted workloads.
        """
        with self._lock:
            return dict(self._reservations)

    @property
    def decision_log(self) -> List[dict]:
        """
        1. The use of function:
           Returns a snapshot copy of all decision records emitted so far.
        2. How does the specific function work:
           Acquires the lock, returns a shallow copy of the list.
        3. Concepts used in the function:
           Defensive copy, audit log.
        4. Places where this function is imported and used:
           Inspected by tests and dashboards to display admission history.
        5. Final output:
           List of event dicts in chronological order.
        """
        with self._lock:
            return list(self._decision_log)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _compute_headroom(self, observed_device_used_bytes: Optional[int]) -> int:
        """
        1. The use of function:
           Computes available headroom within the safe budget.
        2. How does the specific function work:
           If an external device-usage observation is provided, the effective
           used memory is max(reservations_total, observed_device_used_bytes).
           Taking the max is conservative: we never optimistically assume the
           device has less in use than our reservations alone already account for.
           If no usable observation is provided, headroom = safe_budget - reserved.
        3. Concepts used in the function:
           Conservative estimation, cooperative headroom accounting.
        4. Places where this function is imported and used:
           Called internally from request() while the lock is held.
        5. Final output:
           Integer headroom in bytes (may be negative if over-budget).

        IMPORTANT: This is a best-effort estimate. External allocations not tracked
        by this controller reduce actual available memory without notice.

        Caller MUST hold self._lock before calling this method.
        """
        safe = self.safe_budget_bytes
        reserved = sum(self._reservations.values())
        if observed_device_used_bytes is not None and observed_device_used_bytes >= 0:
            effective_used = max(reserved, observed_device_used_bytes)
        else:
            effective_used = reserved
        return safe - effective_used

    def request(
        self,
        workload_id: str,
        estimated_bytes: int,
        observed_device_used_bytes: Optional[int] = None,
        measurement_age_s: Optional[float] = None,
        stale_threshold_s: float = 30.0,
        log_decision: bool = True,
    ) -> Tuple[str, dict]:
        """
        1. The use of function:
           Evaluates whether a workload should be ADMIT, DEFER, or REJECT.
        2. How does the specific function work:
           First applies REJECT policy checks (no lock needed — these are
           purely parameter-based). If not rejected, acquires the lock to
           perform the headroom check and record the reservation atomically,
           preventing TOCTOU races between concurrent requests. Emits a
           decision event for every outcome.
        3. Concepts used in the function:
           Thread-safe resource reservation, TOCTOU prevention,
           cooperative admission policy, estimation vs measurement distinction.
        4. Places where this function is imported and used:
           Called by experiment scripts and workload launchers before starting
           a GPU workload. The caller decides whether to proceed, block, or
           discard the workload based on the returned decision.
        5. Final output of the function, before and after:
           Before: pending workload request.
           After: (decision_string, event_dict) where decision_string is
           "ADMIT", "DEFER", or "REJECT" and event_dict conforms to the
           MemoryEvent schema (source="scheduler", event_type="admission").

        Args:
            workload_id:
                Unique string identifier for this workload instance. Used as the
                key for reservation tracking. If the same workload_id is already
                admitted (not yet released), calling request() again overwrites
                its reservation with the new estimated_bytes to avoid double-
                counting. This is the preferred pattern for workloads that update
                their memory estimate before launch.
            estimated_bytes:
                Declared estimated memory requirement in bytes. This is the
                caller's forward-looking intent. It is kept strictly separate
                from any observed peak memory, which is recorded independently
                via record_observed_peak() after execution.
            observed_device_used_bytes:
                Optional current device-wide memory usage in bytes, typically
                obtained from pynvml.nvmlDeviceGetMemoryInfo(). If provided and
                not stale, it is incorporated conservatively into the headroom
                calculation. This controller does not call NVML itself; the
                caller is responsible for obtaining and passing the measurement.
            measurement_age_s:
                Age of the observed_device_used_bytes reading in seconds. If
                this exceeds stale_threshold_s, the observation is treated as
                unavailable and the controller falls back to reservation-only
                headroom accounting. Pass None if age is unknown.
            stale_threshold_s:
                Maximum acceptable measurement age in seconds (default 30 s).
            log_decision:
                If True (default), appends the decision event to decision_log.
                Can be set to False during retry polling loops to avoid log flooding.

        Returns:
            Tuple (decision: str, event: dict)
              decision: "ADMIT", "DEFER", or "REJECT"
              event: MemoryEvent-schema dict with all decision metadata in
                     metadata_json (JSON string).
        """
        # -----------------------------------------------------------
        # Phase A — REJECT policy checks (no lock: parameter-only checks)
        # -----------------------------------------------------------
        reject_reason: Optional[str] = None

        if estimated_bytes <= 0:
            reject_reason = (
                f"estimated_bytes must be > 0; got {estimated_bytes}. "
                "This is a policy violation — retrying with the same value "
                "will not change the outcome."
            )
        elif self._budget_bytes == 0:
            reject_reason = (
                "Controller budget is 0 bytes. No workload can be admitted. "
                "Reconfigure with a positive budget_bytes value."
            )
        else:
            max_allowed_bytes = int(
                self._budget_bytes * self._max_single_workload_ratio
            )
            if estimated_bytes > max_allowed_bytes:
                reject_reason = (
                    f"estimated_bytes ({estimated_bytes:,} B, "
                    f"{estimated_bytes / (1024**3):.3f} GiB) exceeds the "
                    f"per-workload policy cap ({max_allowed_bytes:,} B = "
                    f"{self._max_single_workload_ratio:.0%} of "
                    f"{self._budget_bytes:,} B budget). "
                    "Reduce the workload requirement or increase "
                    "max_single_workload_ratio in the controller configuration. "
                    "Retrying with the same parameters will not help."
                )

        if reject_reason:
            event = self._build_event(
                workload_id=workload_id,
                decision=AdmissionDecision.REJECT,
                estimated_bytes=estimated_bytes,
                headroom_bytes=None,
                effective_observed=None,
                measurement_age_s=measurement_age_s,
                measurement_is_stale=None,
                reject_reason=reject_reason,
                reserved_total_at_decision=0,
            )
            if log_decision:
                with self._lock:
                    self._decision_log.append(event)
            return AdmissionDecision.REJECT, event

        # -----------------------------------------------------------
        # Phase B — Determine whether the device measurement is usable
        # -----------------------------------------------------------
        measurement_is_stale = False
        effective_observed: Optional[int] = None
        if observed_device_used_bytes is not None:
            if (measurement_age_s is not None
                    and measurement_age_s > stale_threshold_s):
                measurement_is_stale = True
                effective_observed = None  # treat as unavailable
            else:
                effective_observed = observed_device_used_bytes

        # -----------------------------------------------------------
        # Phase C — Headroom check + reservation update (lock held)
        # -----------------------------------------------------------
        with self._lock:
            current_reservation = self._reservations.get(workload_id, 0)
            delta = estimated_bytes - current_reservation
            headroom = self._compute_headroom(effective_observed)
            reserved_total = sum(self._reservations.values())

            # Evaluate headroom for new or replacement request:
            # - delta <= 0: Workload is shrinking or renewing its reservation.
            #   No additional headroom is required, so this is always budget-safe.
            # - delta > 0: Workload is new or expanding its reservation.
            #   Requires that available headroom covers the incremental delta.
            if delta <= 0 or headroom >= delta:
                # ADMIT: record or update the reservation.
                # Updating an existing workload_id replaces its old reservation.
                self._reservations[workload_id] = estimated_bytes
                decision = AdmissionDecision.ADMIT
            else:
                # DEFER: policy-compliant but temporarily insufficient headroom.
                decision = AdmissionDecision.DEFER

            event = self._build_event(
                workload_id=workload_id,
                decision=decision,
                estimated_bytes=estimated_bytes,
                headroom_bytes=headroom,
                effective_observed=effective_observed,
                measurement_age_s=measurement_age_s,
                measurement_is_stale=measurement_is_stale,
                reject_reason=None,
                reserved_total_at_decision=reserved_total,
            )
            if log_decision:
                self._decision_log.append(event)

        return decision, event

    def log_decision_event(self, event: dict) -> None:
        """
        1. The use of function:
           Manually appends an admission decision event to the decision log.
        2. How does the specific function work:
           Acquires the lock and appends event to _decision_log.
        3. Concepts used in the function:
           Thread safety, audit logging.
        4. Places where this function is imported and used:
           Used by experiment runners to record final admission decisions
           after silent retry loops.
        5. Final output:
           None.
        """
        with self._lock:
            self._decision_log.append(event)

    def release(self, workload_id: str) -> bool:
        """
        1. The use of function:
           Removes a workload's reservation, freeing its headroom for subsequent
           requests.
        2. How does the specific function work:
           Acquires the lock, removes the workload_id key from _reservations if
           present, and returns True. If the workload_id is not found (e.g.,
           the workload was never admitted or was already released), returns
           False without raising.
        3. Concepts used in the function:
           Resource deallocation, cooperative scheduling, safe release.
        4. Places where this function is imported and used:
           Called by workload launchers after the GPU workload completes, fails,
           or is cancelled. Must be called even on failure to avoid permanently
           consuming headroom.
        5. Final output of the function, before and after:
           Before: workload_id may or may not be in _reservations.
           After: workload_id is removed from _reservations (if it was present).
           Returns True if reservation existed and was removed; False otherwise.
        """
        with self._lock:
            if workload_id in self._reservations:
                del self._reservations[workload_id]
                return True
            return False

    def record_observed_peak(
        self, workload_id: str, observed_peak_bytes: int
    ) -> None:
        """
        1. The use of function:
           Records the actual observed peak memory usage of a workload after
           execution, distinct from the estimated requirement used at admission.
        2. How does the specific function work:
           Acquires the lock and stores observed_peak_bytes in _observed_peaks
           keyed by workload_id. Does not modify reservations or affect future
           admission decisions.
        3. Concepts used in the function:
           Post-execution measurement, estimation error tracking.
        4. Places where this function is imported and used:
           Called by experiment scripts after measuring workload peak via
           torch.cuda.memory_stats() or NVML (caller-executed).
        5. Final output of the function, before and after:
           Before: no observed peak for this workload_id.
           After: observed_peak_bytes stored for retrieval via get_estimation_error().

        Note on separation of concerns:
        estimated_bytes (from request()) captures the caller's forward-looking
        declaration. observed_peak_bytes (from this function) captures what
        actually happened after execution. The two are stored separately so that
        estimation accuracy can be measured and reported.
        """
        with self._lock:
            self._observed_peaks[workload_id] = observed_peak_bytes

    def get_estimation_error(self, workload_id: str) -> Optional[int]:
        """
        1. The use of function:
           Returns the difference between the observed peak and the estimated
           requirement for a workload, in bytes.
        2. How does the specific function work:
           Looks up the observed peak in _observed_peaks. Searches the decision
           log (newest first) for the most recent ADMIT event for this workload_id
           to retrieve the original estimated_bytes. Returns
           observed_peak - estimated (positive = underestimate;
           negative = overestimate). Returns None if either value is unavailable.
        3. Concepts used in the function:
           Estimation accuracy, admission calibration feedback.
        4. Places where this function is imported and used:
           Called post-experiment to evaluate how accurate admission estimates were.
        5. Final output:
           Integer byte difference, or None if data is incomplete.
        """
        with self._lock:
            if workload_id not in self._observed_peaks:
                return None
            observed = self._observed_peaks[workload_id]
            estimated: Optional[int] = None
            for entry in reversed(self._decision_log):
                try:
                    meta = json.loads(entry.get("metadata_json") or "{}")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if (meta.get("workload_id") == workload_id
                        and entry.get("event_type") == "admission"
                        and meta.get("decision") == AdmissionDecision.ADMIT):
                    estimated = meta.get("estimated_bytes")
                    break
            if estimated is None:
                return None
            return observed - estimated

    def clear_log(self) -> None:
        """
        1. The use of function:
           Clears the accumulated decision log.
        2. How does the specific function work:
           Acquires the lock and resets _decision_log to an empty list.
        3. Concepts used in the function:
           Log management, per-experiment isolation.
        4. Places where this function is imported and used:
           Called between experiment runs to avoid mixing decision histories.
        5. Final output:
           Empty decision log; existing reservations are NOT cleared.
        """
        with self._lock:
            self._decision_log.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_event(
        self,
        workload_id: str,
        decision: str,
        estimated_bytes: int,
        headroom_bytes: Optional[int],
        effective_observed: Optional[int],
        measurement_age_s: Optional[float],
        measurement_is_stale: Optional[bool],
        reject_reason: Optional[str],
        reserved_total_at_decision: int,
    ) -> dict:
        """
        1. The use of function:
           Builds a MemoryEvent-compatible dict for an admission decision.
        2. How does the specific function work:
           Constructs a MemoryEvent with source="scheduler" and
           event_type="admission". All decision-specific metadata is serialised
           into the metadata_json field as a JSON string, consistent with the
           pattern described in work_done.md §5.1.
        3. Concepts used in the function:
           MemoryEvent schema compliance, JSON serialisation.
        4. Places where this function is imported and used:
           Called internally from request() for every outcome.
        5. Final output of the function:
           Dict that can be written as a CSV row using the standard MemoryEvent
           fieldnames, or appended to the decision log directly.
        """
        metadata = {
            "decision": decision,
            "workload_id": workload_id,
            "estimated_bytes": estimated_bytes,
            "headroom_bytes": headroom_bytes,
            "safe_budget_bytes": self.safe_budget_bytes,
            "budget_bytes": self._budget_bytes,
            "safety_margin_ratio": self._safety_margin_ratio,
            "max_single_workload_ratio": self._max_single_workload_ratio,
            "reserved_total_bytes": reserved_total_at_decision,
            "observed_device_used_bytes": effective_observed,
            "measurement_age_s": measurement_age_s,
            "measurement_is_stale": measurement_is_stale,
            "reject_reason": reject_reason,
        }
        event = MemoryEvent(
            timestamp=time.time(),
            pid=os.getpid(),
            gpu_id=self._gpu_id,
            source="scheduler",
            event_type="admission",
            workload_name=workload_id,
            metadata_json=json.dumps(metadata),
        )
        return event.to_row()
