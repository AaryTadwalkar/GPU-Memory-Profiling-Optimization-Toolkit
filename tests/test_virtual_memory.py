"""
File: tests/test_virtual_memory.py

1. What this file does:
   Comprehensive test suite validating the Virtual Memory Management and Page Replacement
   subsystem, covering FIFO, LRU, Belady's Optimal algorithms, Belady's anomaly detection,
   page table lifecycle transitions, Effective Access Time (EAT) calculations, edge cases,
   and parameter validation.

2. Overall design:
   Organized into targeted pytest test classes verifying OS textbook examples, mathematical
   invariants, state machine transitions (VRAM <-> Swap), and error handling. Tests execute
   100% in user-space without touching GPU drivers, NVML, or OS kernel facilities.

3. Elements in the file:
   - TestTextbookReferenceString: Verifies textbook reference trace with 3 and 4 frames.
   - TestBeladysAnomaly: Verifies detection of Belady's anomaly on FIFO replacement.
   - TestPageTableTransitions: Tests Hit, Miss with free frame, Eviction, and Swap-in states.
   - TestEffectiveAccessTime: Tests EAT formula across hit ratios and latency configurations.
   - TestEdgeCasesAndValidation: Tests empty trace, single-page repeat, excess frames, and invalid inputs.
   - TestComparePolicies: Verifies multi-policy comparative benchmark output.

4. Final output of the file:
   Pytest execution output verifying full correctness, regression prevention, and edge-case safety.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List
import pytest

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.virtual_memory import (
    FIFOPolicy,
    LRUPolicy,
    OptimalPolicy,
    PageAccessResult,
    PageEntry,
    PageReplacementPolicy,
    VirtualMemoryManager,
    VirtualMemoryMetrics,
)


class TestTextbookReferenceString:
    """
    1. The use of class:
       Validates virtual memory page replacement algorithms against standard operating system
       textbook benchmarks (Silberschatz/Galvin reference string).

    2. What the class contain:
       Contains unit test methods testing FIFO, LRU, and Optimal replacement policies with 3 and 4
       physical frames.

    3. Technologies used and why:
       Pytest testing framework for assertions, fixture isolation, and failure diagnostics.

    4. Key design and overall design of the class:
       Applies the canonical reference string [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
       to verify exact page fault counts matching theoretical derivations.
    """

    def test_textbook_3_frames(self) -> None:
        """
        1. The use of function:
           Tests that with 3 physical frames, the reference string produces exactly 15 faults (FIFO),
           12 faults (LRU), and 9 faults (Optimal).

        2. How does the specific function work:
           Initializes VirtualMemoryManager with vram_frame_count=3. Runs run_trace for FIFO, LRU,
           and Optimal. Asserts that the reported faults and hit counts match textbook values.

        3. Concepts used in the function:
           Textbook paging simulation verification, page fault counting.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unverified simulation results. After: Validated page fault counts (FIFO=15, LRU=12, OPT=9).
        """
        ref_trace = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
        mgr = VirtualMemoryManager(vram_frame_count=3)

        fifo_res = mgr.run_trace(ref_trace, policy_name="fifo")
        assert fifo_res.faults == 15
        assert fifo_res.hits == 5
        assert fifo_res.total_requests == 20

        lru_res = mgr.run_trace(ref_trace, policy_name="lru")
        assert lru_res.faults == 12
        assert lru_res.hits == 8
        assert lru_res.total_requests == 20

        opt_res = mgr.run_trace(ref_trace, policy_name="optimal")
        assert opt_res.faults == 9
        assert opt_res.hits == 11
        assert opt_res.total_requests == 20

    def test_textbook_4_frames(self) -> None:
        """
        1. The use of function:
           Tests that with 4 physical frames, the reference string produces exactly 10 faults (FIFO),
           8 faults (LRU), and 8 faults (Optimal).

        2. How does the specific function work:
           Initializes VirtualMemoryManager with vram_frame_count=4. Executes the reference string
           under FIFO, LRU, and Optimal policies and asserts exact fault counts.

        3. Concepts used in the function:
           Multi-frame paging dynamics, frame allocation scaling.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unchecked 4-frame execution. After: Validated page fault counts (FIFO=10, LRU=8, OPT=8).
        """
        ref_trace = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
        mgr = VirtualMemoryManager(vram_frame_count=4)

        fifo_res = mgr.run_trace(ref_trace, policy_name="fifo")
        assert fifo_res.faults == 10
        assert fifo_res.hits == 10

        lru_res = mgr.run_trace(ref_trace, policy_name="lru")
        assert lru_res.faults == 8
        assert lru_res.hits == 12

        opt_res = mgr.run_trace(ref_trace, policy_name="optimal")
        assert opt_res.faults == 8
        assert opt_res.hits == 12


class TestBeladysAnomaly:
    """
    1. The use of class:
       Validates detection and characterization of Belady's Anomaly where allocating more physical
       memory frames paradoxically increases page faults under FIFO replacement.

    2. What the class contain:
       Contains unit test methods testing manual simulation of anomaly traces and the automated
       detect_beladys_anomaly method.

    3. Technologies used and why:
       Pytest assertions testing algorithmic properties and anomaly reporting structures.

    4. Key design and overall design of the class:
       Evaluates the classic Belady anomaly reference trace [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5],
       proving 3 frames yield 9 faults while 4 frames yield 10 faults.
    """

    def test_belady_manual_comparison(self) -> None:
        """
        1. The use of function:
           Tests manual FIFO simulation for 3 frames versus 4 frames on Belady's trace.

        2. How does the specific function work:
           Instantiates manager with 3 frames, runs trace (9 faults). Instantiates with 4 frames,
           runs trace (10 faults). Asserts that faults(4 frames) > faults(3 frames).

        3. Concepts used in the function:
           Belady's Anomaly, FIFO replacement non-stack algorithm property.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unverified anomaly behavior. After: Proven anomaly with faults increasing from 9 to 10.
        """
        trace = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]

        mgr3 = VirtualMemoryManager(vram_frame_count=3, policy="fifo")
        res3 = mgr3.run_trace(trace)
        assert res3.faults == 9

        mgr4 = VirtualMemoryManager(vram_frame_count=4, policy="fifo")
        res4 = mgr4.run_trace(trace)
        assert res4.faults == 10

        assert res4.faults > res3.faults

    def test_detect_beladys_anomaly_method(self) -> None:
        """
        1. The use of function:
           Tests the automated detect_beladys_anomaly method on VirtualMemoryManager.

        2. How does the specific function work:
           Calls detect_beladys_anomaly() with default trace and frame range (3, 6).
           Asserts detected is True, faults_by_frame[3] == 9, faults_by_frame[4] == 10,
           and anomaly_details contains the transition from 3 to 4 frames.

        3. Concepts used in the function:
           Automated anomaly detection, multi-frame parameter sweeping.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unchecked automated anomaly detector. After: Validated detection dictionary.
        """
        mgr = VirtualMemoryManager()
        analysis = mgr.detect_beladys_anomaly(frame_range=(3, 6))

        assert analysis["detected"] is True
        assert analysis["policy"] == "fifo"
        assert analysis["faults_by_frame"][3] == 9
        assert analysis["faults_by_frame"][4] == 10
        assert len(analysis["anomaly_details"]) >= 1

        detail = analysis["anomaly_details"][0]
        assert detail["frames_lower"] == 3
        assert detail["faults_lower"] == 9
        assert detail["frames_higher"] == 4
        assert detail["faults_higher"] == 10
        assert detail["fault_increase"] == 1


class TestPageTableTransitions:
    """
    1. The use of class:
       Validates fine-grained page table lifecycle transitions: Hit, Fault with free frame allocation,
       Fault with eviction to swap, and Swap-in.

    2. What the class contain:
       Contains unit test methods inspecting PageEntry attributes and PageAccessResult records.

    3. Technologies used and why:
       Pytest testing state machine transitions in the virtual memory engine.

    4. Key design and overall design of the class:
       Walks through sequential access steps, asserting that residency flags (in_vram, in_swap),
       frame assignments, and eviction logs transition exactly according to OS demand paging theory.
    """

    def test_hit_miss_and_eviction_lifecycle(self) -> None:
        """
        1. The use of function:
           Tests the complete sequence of free frame allocation, hit, eviction, and swap-in.

        2. How does the specific function work:
           Configures manager with 2 frames and LRU policy:
           1. Access page 1: Miss, free frame allocated (frame 0), in_vram=True, in_swap=False.
           2. Access page 2: Miss, free frame allocated (frame 1), in_vram=True, in_swap=False.
           3. Access page 1: Hit, frame 0 preserved, access_count=2, is_hit=True.
           4. Access page 3: Miss (frames saturated), page 2 evicted to swap, page 3 takes frame 1.
           5. Access page 2: Miss (swap-in), page 1 evicted to swap, page 2 loaded back to VRAM.

        3. Concepts used in the function:
           Page table state transitions, demand paging, swap-out, swap-in, LRU recency tracking.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unverified lifecycle transitions. After: Fully verified state transitions.
        """
        mgr = VirtualMemoryManager(vram_frame_count=2, policy="lru")

        # Step 0: Page 1 (Miss, free frame 0)
        r0 = mgr.access_page(page_id=1, current_step=0)
        assert not r0.is_hit
        assert r0.frame_id == 0
        assert r0.evicted_page_id is None
        assert mgr.page_table[1].in_vram is True
        assert mgr.page_table[1].in_swap is False
        assert mgr.page_table[1].access_count == 1

        # Step 1: Page 2 (Miss, free frame 1)
        r1 = mgr.access_page(page_id=2, current_step=1)
        assert not r1.is_hit
        assert r1.frame_id == 1
        assert r1.evicted_page_id is None
        assert mgr.page_table[2].in_vram is True
        assert mgr.page_table[2].in_swap is False

        # Step 2: Page 1 (Hit)
        r2 = mgr.access_page(page_id=1, current_step=2)
        assert r2.is_hit
        assert r2.frame_id == 0
        assert r2.evicted_page_id is None
        assert mgr.page_table[1].access_count == 2
        assert mgr.page_table[1].last_accessed_at == 2.0

        # Step 3: Page 3 (Miss, frames full -> page 2 is LRU victim, evicted to swap)
        r3 = mgr.access_page(page_id=3, current_step=3)
        assert not r3.is_hit
        assert r3.evicted_page_id == 2
        assert r3.frame_id == 1
        assert mgr.page_table[2].in_vram is False
        assert mgr.page_table[2].in_swap is True
        assert mgr.page_table[2].frame_id is None
        assert mgr.page_table[3].in_vram is True
        assert mgr.page_table[3].in_swap is False
        assert len(mgr.eviction_log) == 1
        assert mgr.eviction_log[0]["evicted_page"] == 2
        assert mgr.eviction_log[0]["incoming_page"] == 3

        # Step 4: Page 2 (Miss -> Swap-in: page 1 is LRU victim, page 2 restored to VRAM)
        assert mgr.page_table[2].in_swap is True
        r4 = mgr.access_page(page_id=2, current_step=4)
        assert not r4.is_hit
        assert r4.evicted_page_id == 1
        assert mgr.page_table[2].in_vram is True
        assert mgr.page_table[2].in_swap is False
        assert mgr.page_table[1].in_vram is False
        assert mgr.page_table[1].in_swap is True


class TestEffectiveAccessTime:
    """
    1. The use of class:
       Validates Effective Access Time (EAT) calculations under various hit ratios and latency configurations.

    2. What the class contain:
       Contains unit test methods verifying theoretical extremes (100% hit, 100% fault) and intermediate values.

    3. Technologies used and why:
       Pytest assertions testing floating-point precision within reasonable tolerances.

    4. Key design and overall design of the class:
       Applies standard EAT formula: EAT = hit_ratio * t_vram + (1 - hit_ratio) * t_swap,
       ensuring numerical accuracy and proper default/override handling.
    """

    def test_eat_extremes_and_defaults(self) -> None:
        """
        1. The use of function:
           Tests EAT at 100% hit ratio, 0% hit ratio, and typical 80% hit ratio.

        2. How does the specific function work:
           Initializes manager with default latencies (t_vram=1.0us, t_swap=100.0us).
           Asserts calculate_eat(1.0) == 1.0, calculate_eat(0.0) == 100.0,
           and calculate_eat(0.8) == 20.8.

        3. Concepts used in the function:
           Effective Access Time formula, memory hierarchy latency modeling.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unverified EAT values. After: Verified exact mathematical bounds.
        """
        mgr = VirtualMemoryManager(t_vram_us=1.0, t_swap_us=100.0)

        assert pytest.approx(mgr.calculate_eat(1.0), rel=1e-5) == 1.0
        assert pytest.approx(mgr.calculate_eat(0.0), rel=1e-5) == 100.0
        assert pytest.approx(mgr.calculate_eat(0.8), rel=1e-5) == 20.8

    def test_eat_custom_parameter_overrides(self) -> None:
        """
        1. The use of function:
           Tests calculate_eat when caller supplies explicit t_vram_us and t_swap_us overrides.

        2. How does the specific function work:
           Calls calculate_eat(hit_ratio=0.5, t_vram_us=2.0, t_swap_us=50.0).
           Verifies result is 0.5 * 2.0 + 0.5 * 50.0 = 26.0.

        3. Concepts used in the function:
           Parameter override mechanism, dynamic memory access latency calculations.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unchecked override paths. After: Verified custom latency calculation.
        """
        mgr = VirtualMemoryManager(t_vram_us=1.0, t_swap_us=100.0)
        custom_eat = mgr.calculate_eat(0.5, t_vram_us=2.0, t_swap_us=50.0)
        assert pytest.approx(custom_eat, rel=1e-5) == 26.0


class TestEdgeCasesAndValidation:
    """
    1. The use of class:
       Validates system edge cases, boundary conditions, input validation, and object serialization.

    2. What the class contain:
       Contains unit test methods testing empty traces, repetitive traces, oversized frames,
       invalid policies, negative frame counts, serialization to_dict, and empty frame victim errors.

    3. Technologies used and why:
       Pytest exception assertions (pytest.raises) and dictionary comparison.

    4. Key design and overall design of the class:
       Guarantees fault-tolerant execution and defensive bounds checking across all user-facing APIs.
    """

    def test_empty_trace(self) -> None:
        """
        1. The use of function:
           Tests execution behavior when supplied an empty reference string.

        2. How does the specific function work:
           Calls run_trace([]) and checks that metrics reflect zero requests, zero faults, and zero hits.

        3. Concepts used in the function:
           Empty collection boundary handling.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Potential division-by-zero risk. After: Clean zero-initialized VirtualMemoryMetrics.
        """
        mgr = VirtualMemoryManager(vram_frame_count=4)
        m = mgr.run_trace([])
        assert m.total_requests == 0
        assert m.hits == 0
        assert m.faults == 0
        assert m.hit_ratio == 0.0
        assert m.miss_ratio == 0.0
        assert m.eat_us == 0.0
        assert len(m.fault_timeline) == 0

    def test_single_page_repeat(self) -> None:
        """
        1. The use of function:
           Tests repetitive access to the same virtual page [1, 1, 1, 1, 1].

        2. How does the specific function work:
           Runs trace [1, 1, 1, 1, 1] through manager with 3 frames.
           Asserts 1 fault (cold miss) followed by 4 consecutive hits.

        3. Concepts used in the function:
           Temporal locality maximum case, cold miss vs warm hit.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unverified repetitive pattern. After: Verified exactly 1 fault and 4 hits.
        """
        mgr = VirtualMemoryManager(vram_frame_count=3)
        m = mgr.run_trace([1, 1, 1, 1, 1])
        assert m.total_requests == 5
        assert m.faults == 1
        assert m.hits == 4
        assert m.fault_timeline == [1, 0, 0, 0, 0]

    def test_frames_larger_than_working_set(self) -> None:
        """
        1. The use of function:
           Tests behavior when physical frames exceed the number of unique virtual pages referenced.

        2. How does the specific function work:
           Sets vram_frame_count=10 on trace [1, 2, 3, 1, 2, 3].
           Asserts exactly 3 cold faults, 3 hits, and zero evictions.

        3. Concepts used in the function:
           Capacity unconstrained execution, absence of capacity misses.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unchecked oversized frame allocation. After: Confirmed zero evictions occurred.
        """
        mgr = VirtualMemoryManager(vram_frame_count=10)
        m = mgr.run_trace([1, 2, 3, 1, 2, 3])
        assert m.faults == 3
        assert m.hits == 3
        assert len(m.eviction_log) == 0

    def test_invalid_parameters_raise_errors(self) -> None:
        """
        1. The use of function:
           Tests validation errors for invalid frame counts, unsupported policy names, and bad ranges.

        2. How does the specific function work:
           Asserts ValueError when vram_frame_count <= 0, swap_frame_count <= 0,
           policy is unknown, or detect_beladys_anomaly has invalid frame_range.

        3. Concepts used in the function:
           Defensive parameter validation and explicit exception reporting.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unverified parameter gates. After: All invalid parameter inputs reject gracefully.
        """
        with pytest.raises(ValueError, match="vram_frame_count must be positive"):
            VirtualMemoryManager(vram_frame_count=0)

        with pytest.raises(ValueError, match="swap_frame_count must be positive"):
            VirtualMemoryManager(swap_frame_count=-1)

        with pytest.raises(ValueError, match="Unsupported policy"):
            VirtualMemoryManager(policy="non_existent_policy")

        mgr = VirtualMemoryManager()
        with pytest.raises(ValueError, match="Minimum frames in frame_range must be >= 1"):
            mgr.detect_beladys_anomaly(frame_range=(0, 4))

        with pytest.raises(ValueError, match="max_frames .* < min_frames"):
            mgr.detect_beladys_anomaly(frame_range=(5, 3))

    def test_dataclass_serialization(self) -> None:
        """
        1. The use of function:
           Tests to_dict() serialization methods on PageEntry, PageAccessResult, and VirtualMemoryMetrics.

        2. How does the specific function work:
           Instantiates each dataclass, invokes to_dict(), and asserts keys match expected schema.

        3. Concepts used in the function:
           Schema conformance and serialization completeness.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Dataclass instances. After: Validated dictionary representations.
        """
        entry = PageEntry(page_id=42, frame_id=1, in_vram=True)
        d_entry = entry.to_dict()
        assert d_entry["page_id"] == 42
        assert d_entry["in_vram"] is True

        res = PageAccessResult(
            page_id=42,
            step=1,
            is_hit=True,
            evicted_page_id=None,
            frame_id=1,
            policy_used="lru",
            current_vram_pages=[42],
        )
        d_res = res.to_dict()
        assert d_res["page_id"] == 42
        assert d_res["is_hit"] is True

        mgr = VirtualMemoryManager(vram_frame_count=2)
        m = mgr.run_trace([1, 2, 1])
        d_m = m.to_dict()
        assert "hit_ratio" in d_m
        assert "eat_us" in d_m
        assert "fault_timeline" in d_m

    def test_empty_frames_victim_selection_raises_error(self) -> None:
        """
        1. The use of function:
           Tests that select_victim on FIFO, LRU, and Optimal policies raises ValueError when given an empty list.

        2. How does the specific function work:
           Directly calls select_victim([], current_step=0) on each policy and asserts ValueError.

        3. Concepts used in the function:
           Edge case handling on empty candidate pools.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Potential IndexError on empty collection. After: Clear ValueError raised.
        """
        fifo = FIFOPolicy()
        lru = LRUPolicy()
        opt = OptimalPolicy()

        with pytest.raises(ValueError, match="Cannot select victim from an empty frames list"):
            fifo.select_victim([], current_step=0)

        with pytest.raises(ValueError, match="Cannot select victim from an empty frames list"):
            lru.select_victim([], current_step=0)

        with pytest.raises(ValueError, match="Cannot select victim from an empty frames list"):
            opt.select_victim([], current_step=0)


class TestComparePolicies:
    """
    1. The use of class:
       Validates the multi-policy comparison capability of VirtualMemoryManager.

    2. What the class contain:
       Contains unit test methods verifying comparative execution across FIFO, LRU, and Optimal.

    3. Technologies used and why:
       Pytest assertions testing comparative consistency across multiple policy evaluations.

    4. Key design and overall design of the class:
       Executes compare_policies() on a known trace, verifying that metrics for all three policies
       are returned and Optimal page faults <= LRU page faults <= FIFO page faults.
    """

    def test_compare_policies_execution(self) -> None:
        """
        1. The use of function:
           Tests that compare_policies() executes all policies cleanly and returns comparative metrics.

        2. How does the specific function work:
           Runs compare_policies on the textbook reference trace. Asserts presence of "fifo", "lru",
           and "optimal" keys, and verifies that optimal faults (9) <= LRU faults (12) <= FIFO faults (15).

        3. Concepts used in the function:
           Comparative policy benchmarking, policy efficiency hierarchy.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Unchecked batch comparison. After: Validated comparison mapping with verified inequality.
        """
        ref_trace = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
        mgr = VirtualMemoryManager(vram_frame_count=3)
        comparison = mgr.compare_policies(ref_trace)

        assert "fifo" in comparison
        assert "lru" in comparison
        assert "optimal" in comparison

        assert comparison["optimal"].faults <= comparison["lru"].faults
        assert comparison["lru"].faults <= comparison["fifo"].faults
        assert comparison["optimal"].faults == 9
        assert comparison["lru"].faults == 12
        assert comparison["fifo"].faults == 15
