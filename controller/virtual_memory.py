"""
File: controller/virtual_memory.py

1. What this file does:
   Implements the core operating system virtual memory management and page replacement
   simulation engine. It models virtual-to-physical address translation via page tables,
   VRAM frames, secondary swap storage, page fault handling, eviction policies (FIFO,
   LRU, and Belady's Optimal), effective access time (EAT) calculation, and Belady's
   anomaly detection.

2. Overall design:
   Designed as a modular, extensible, and pure user-space virtual memory subsystem.
   A VirtualMemoryManager coordinates physical frames, virtual page table entries (PageEntry),
   and page replacement algorithms inheriting from PageReplacementPolicy. When an accessed
   virtual page is already resident in a physical frame, a cache hit is recorded and
   recency/frequency counters are updated. On a page fault (miss), the engine checks
   for an available free frame; if all physical frames are saturated, an eviction policy
   selects a victim page to evict to secondary swap storage before loading the requested
   page. Comprehensive simulation metrics and time-series traces are produced as
   VirtualMemoryMetrics.

3. Elements in the file:
   - PageEntry: Dataclass representing a virtual page table entry with residence and state flags.
   - PageAccessResult: Dataclass representing the deterministic result of a single page access.
   - VirtualMemoryMetrics: Dataclass summarizing an entire trace simulation with hit/miss ratios and EAT.
   - PageReplacementPolicy: Abstract base class defining the victim selection interface.
   - FIFOPolicy: First-In, First-Out replacement policy based on allocation arrival timestamps.
   - LRUPolicy: Least Recently Used replacement policy based on last-access timestamps.
   - OptimalPolicy: Belady's Optimal replacement algorithm inspecting future memory access references.
   - VirtualMemoryManager: Central simulation coordinator managing page tables, frames, policies, and metrics.

4. Final output of the file:
   Exposes high-level simulation APIs (access_page, run_trace, compare_policies, calculate_eat,
   detect_beladys_anomaly) returning structured dataclasses and dictionaries consumed by test
   suites, analytics modules, and UI visualizers.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class PageEntry:
    """
    1. The use of class:
       Represents an individual page table entry (PTE) tracking the residency, frame mapping,
       and temporal access metadata of a single virtual page.

    2. What the class contain:
       Contains fields for page identifier, physical frame assignment, VRAM residency flag,
       swap residency flag, dirty modification bit, initial load timestamp, last access
       timestamp, total reference count, and a dictionary serialization helper.

    3. Technologies used and why:
       Python standard library `@dataclass` for concise, type-annotated, high-performance
       in-memory state management without third-party dependencies.

    4. Key design and overall design of the class:
       Designed to mirror hardware/OS page table entries with orthogonal flags for VRAM and
       swap residency, enabling faithful emulation of demand paging and swap-in/swap-out.
    """
    page_id: int
    frame_id: Optional[int] = None
    in_vram: bool = False
    in_swap: bool = False
    dirty: bool = False
    loaded_at: float = 0.0
    last_accessed_at: float = 0.0
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Serializes the PageEntry dataclass instance into a standard Python dictionary.

        2. How does the specific function work:
           Invokes dataclasses.asdict on self to produce a shallow key-value dictionary.

        3. Concepts used in the function:
           Dataclass introspection and dictionary serialization.

        4. Places where this function is imported and used:
           Used in logging, status reporting, and export APIs across the memory subsystem.

        5. Final output of the function, before and after:
           Before: PageEntry instance. After: Dictionary of page metadata attributes.
        """
        return asdict(self)


@dataclass
class PageAccessResult:
    """
    1. The use of class:
       Captures the complete outcome of accessing a single virtual page during a simulation step.

    2. What the class contain:
       Contains accessed page ID, discrete step number, hit/miss boolean flag, evicted victim page ID
       (if eviction occurred), assigned physical frame ID, active policy name, and snapshot of currently
       resident VRAM page IDs.

    3. Technologies used and why:
       Python `@dataclass` for lightweight immutable access logging and deterministic inspection.

    4. Key design and overall design of the class:
       Provides a structured audit record per access step, allowing detailed step-by-step UI playback
       and verification of cache behavior.
    """
    page_id: int
    step: int
    is_hit: bool
    evicted_page_id: Optional[int]
    frame_id: int
    policy_used: str
    current_vram_pages: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Serializes the PageAccessResult instance into a standard dictionary.

        2. How does the specific function work:
           Converts internal dataclass fields to a standard dictionary format using dataclasses.asdict.

        3. Concepts used in the function:
           Dataclass serialization and telemetry capture.

        4. Places where this function is imported and used:
           Used by simulation loggers, trace analyzers, and unit tests.

        5. Final output of the function, before and after:
           Before: PageAccessResult object. After: Dictionary representing access step result.
        """
        return asdict(self)


@dataclass
class VirtualMemoryMetrics:
    """
    1. The use of class:
       Encapsulates cumulative analytical metrics and logs for a virtual memory trace execution.

    2. What the class contain:
       Contains policy identifier, physical frame count, total requests count, hit count, page fault
       count, hit ratio, miss ratio, Effective Access Time (EAT in microseconds), per-step binary
       fault timeline, and detailed list of eviction events.

    3. Technologies used and why:
       Python `@dataclass` for standardized aggregation and quantitative comparison across policies.

    4. Key design and overall design of the class:
       Acts as the canonical benchmark result container comparing different paging strategies,
       suitable for visualization in performance dashboards and evaluation against theoretical limits.
    """
    policy: str
    frame_count: int
    total_requests: int
    hits: int
    faults: int
    hit_ratio: float
    miss_ratio: float
    eat_us: float
    fault_timeline: List[int] = field(default_factory=list)
    eviction_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Serializes the simulation metrics into a clean dictionary.

        2. How does the specific function work:
           Invokes dataclasses.asdict to convert metrics, timelines, and logs into a dictionary.

        3. Concepts used in the function:
           Dataclass introspection and metrics export.

        4. Places where this function is imported and used:
           Consumed by experiment loggers, benchmark reports, and dashboard presenters.

        5. Final output of the function, before and after:
           Before: VirtualMemoryMetrics instance. After: Dictionary of cumulative simulation metrics.
        """
        return asdict(self)


class PageReplacementPolicy(ABC):
    """
    1. The use of class:
       Defines the abstract contract for page replacement algorithms selecting a victim page
       when physical frames are exhausted.

    2. What the class contain:
       Contains the policy name string, base constructor, and the abstract method select_victim.

    3. Technologies used and why:
       Python abc module (ABC, abstractmethod) to enforce interface consistency and polymorphism
       across swappable replacement strategies.

    4. Key design and overall design of the class:
       Implements the Strategy Design Pattern allowing the VirtualMemoryManager to remain agnostic
       to specific victim selection heuristics.
    """

    def __init__(self, name: str) -> None:
        """
        1. The use of function:
           Initializes the replacement policy with its canonical identifier.

        2. How does the specific function work:
           Assigns the provided string identifier to the instance variable name.

        3. Concepts used in the function:
           Object-oriented initialization and policy identification.

        4. Places where this function is imported and used:
           Called by child policy classes during instantiation.

        5. Final output of the function, before and after:
           Before: Uninitialized policy instance. After: Initialized policy with name attribute.
        """
        self.name = name

    @abstractmethod
    def select_victim(
        self,
        frames: List[PageEntry],
        current_step: int,
        future_trace: Optional[List[int]] = None,
    ) -> PageEntry:
        """
        1. The use of function:
           Abstract method to select a resident victim PageEntry to be evicted from physical frames.

        2. How does the specific function work:
           Subclasses must inspect the resident frame entries, current step, and optional future trace
           to return the chosen PageEntry victim.

        3. Concepts used in the function:
           Polymorphic method declaration and victim selection heuristic contract.

        4. Places where this function is imported and used:
           Invoked exclusively by VirtualMemoryManager when a page fault occurs and all frames are full.

        5. Final output of the function, before and after:
           Before: Saturated frame entries list. After: Target PageEntry designated for eviction.
        """
        pass


class FIFOPolicy(PageReplacementPolicy):
    """
    1. The use of class:
       Implements the First-In, First-Out (FIFO) page replacement algorithm.

    2. What the class contain:
       Inherits from PageReplacementPolicy and provides a deterministic implementation of select_victim
       favoring the page with the earliest load time.

    3. Technologies used and why:
       Standard Python list operations and min() with multi-key tie-breaking for deterministic execution.

    4. Key design and overall design of the class:
       Evicts the page that has resided in physical frames the longest, determined by minimum loaded_at.
       Ties are broken deterministically by smaller page_id.
    """

    def __init__(self) -> None:
        """
        1. The use of function:
           Initializes the FIFO policy instance with identifier "fifo".

        2. How does the specific function work:
           Invokes super().__init__("fifo").

        3. Concepts used in the function:
           Subclass constructor chaining and naming convention.

        4. Places where this function is imported and used:
           Used by VirtualMemoryManager and policy benchmark suites.

        5. Final output of the function, before and after:
           Before: Uninitialized FIFO instance. After: Initialized FIFOPolicy instance.
        """
        super().__init__(name="fifo")

    def select_victim(
        self,
        frames: List[PageEntry],
        current_step: int,
        future_trace: Optional[List[int]] = None,
    ) -> PageEntry:
        """
        1. The use of function:
           Selects the resident page that was loaded earliest into physical frames.

        2. How does the specific function work:
           Finds the entry in frames with the minimum loaded_at timestamp. Ties are broken
           by page_id.

        3. Concepts used in the function:
           FIFO queue ordering, min() with tuple key comparison.

        4. Places where this function is imported and used:
           Called by VirtualMemoryManager during FIFO demand paging simulations.

        5. Final output of the function, before and after:
           Before: List of candidate resident PageEntry objects. After: Oldest loaded PageEntry.
        """
        if not frames:
            raise ValueError("Cannot select victim from an empty frames list")
        return min(frames, key=lambda p: (p.loaded_at, p.page_id))


class LRUPolicy(PageReplacementPolicy):
    """
    1. The use of class:
       Implements the Least Recently Used (LRU) page replacement algorithm.

    2. What the class contain:
       Inherits from PageReplacementPolicy and implements select_victim choosing the page whose
       most recent access timestamp is the oldest.

    3. Technologies used and why:
       Standard Python min() with tuple comparison over last_accessed_at, loaded_at, and page_id.

    4. Key design and overall design of the class:
       Emulates temporal locality by evicting pages that have not been referenced for the longest
       duration, ensuring pages actively in working sets are preserved.
    """

    def __init__(self) -> None:
        """
        1. The use of function:
           Initializes the LRU policy instance with identifier "lru".

        2. How does the specific function work:
           Invokes super().__init__("lru").

        3. Concepts used in the function:
           Subclass constructor chaining.

        4. Places where this function is imported and used:
           Used by VirtualMemoryManager and benchmark runners.

        5. Final output of the function, before and after:
           Before: Uninitialized LRU instance. After: Initialized LRUPolicy instance.
        """
        super().__init__(name="lru")

    def select_victim(
        self,
        frames: List[PageEntry],
        current_step: int,
        future_trace: Optional[List[int]] = None,
    ) -> PageEntry:
        """
        1. The use of function:
           Selects the resident page with the oldest last_accessed_at timestamp.

        2. How does the specific function work:
           Evaluates all resident frames using min() keyed on last_accessed_at. Ties are broken
           by loaded_at followed by page_id.

        3. Concepts used in the function:
           Temporal locality heuristic, recency tracking, deterministic multi-key ordering.

        4. Places where this function is imported and used:
           Called by VirtualMemoryManager on page faults under LRU configuration.

        5. Final output of the function, before and after:
           Before: Resident frames list. After: PageEntry with oldest access timestamp.
        """
        if not frames:
            raise ValueError("Cannot select victim from an empty frames list")
        return min(frames, key=lambda p: (p.last_accessed_at, p.loaded_at, p.page_id))


class OptimalPolicy(PageReplacementPolicy):
    """
    1. The use of class:
       Implements Belady's Optimal (OPT/MIN) page replacement algorithm, providing theoretical
       minimum page fault upper bounds.

    2. What the class contain:
       Inherits from PageReplacementPolicy and implements select_victim using omniscient forward
       lookahead across the remaining reference trace.

    3. Technologies used and why:
       Python list lookups and max() with tuple keys representing forward reuse distance.

    4. Key design and overall design of the class:
       Inspects future accesses from current_step + 1 onward. Evicts the page that will not be
       used for the longest period of time (or never again). Ties among pages never accessed
       again are broken using FIFO arrival order (earliest loaded_at).
    """

    def __init__(self) -> None:
        """
        1. The use of function:
           Initializes the Belady's Optimal policy instance with identifier "optimal".

        2. How does the specific function work:
           Invokes super().__init__("optimal").

        3. Concepts used in the function:
           Subclass constructor chaining.

        4. Places where this function is imported and used:
           Used by VirtualMemoryManager to compute theoretical efficiency bounds.

        5. Final output of the function, before and after:
           Before: Uninitialized Optimal instance. After: Initialized OptimalPolicy instance.
        """
        super().__init__(name="optimal")

    def select_victim(
        self,
        frames: List[PageEntry],
        current_step: int,
        future_trace: Optional[List[int]] = None,
    ) -> PageEntry:
        """
        1. The use of function:
           Selects the resident page whose next forward reference is furthest in the future.

        2. How does the specific function work:
           Slices future_trace starting from index current_step + 1. For each page in frames,
           computes the index of its next occurrence; if not found, assigns float('inf').
           Uses max() to find the maximum distance. Ties at infinity are broken by choosing the
           page loaded earliest (-loaded_at) and then smaller page ID (-page_id).

        3. Concepts used in the function:
           Belady's Optimal algorithm, lookahead reference inspection, deterministic tie-breaking.

        4. Places where this function is imported and used:
           Called by VirtualMemoryManager during optimal paging simulation.

        5. Final output of the function, before and after:
           Before: Resident frames and future trace. After: PageEntry whose reuse is furthest.
        """
        if not frames:
            raise ValueError("Cannot select victim from an empty frames list")

        future = future_trace[current_step + 1:] if future_trace is not None else []

        def opt_distance(page: PageEntry) -> Tuple[float, float, int]:
            if page.page_id in future:
                dist = float(future.index(page.page_id))
            else:
                dist = float("inf")
            # Maximize distance; if equal, prefer earlier loaded_at, then smaller page_id
            return (dist, -page.loaded_at, -page.page_id)

        return max(frames, key=opt_distance)


class VirtualMemoryManager:
    """
    1. The use of class:
       Central operating system virtual memory controller coordinating page tables, physical frames,
       secondary swap storage, replacement policies, and performance metrics.

    2. What the class contain:
       Contains frame capacities, access latencies (VRAM hit time and swap fault penalty), internal
       page table mapping, policy registry, cumulative performance counters, access history,
       and high-level simulation methods (access_page, run_trace, compare_policies, calculate_eat,
       detect_beladys_anomaly).

    3. Technologies used and why:
       Pure Python data structures (dict, list) for fast, deterministic, cross-platform simulation
       without requiring kernel modules or administrative privileges.

    4. Key design and overall design of the class:
       Encapsulates all aspects of memory translation and paging dynamics. Supports dynamic policy
       switching, isolated repeatable traces, exact Belady anomaly detection across arbitrary frame
       ranges, and effective access time quantification.
    """

    def __init__(
        self,
        vram_frame_count: int = 4,
        swap_frame_count: int = 16,
        policy: str = "lru",
        t_vram_us: float = 1.0,
        t_swap_us: float = 100.0,
    ) -> None:
        """
        1. The use of function:
           Constructs and configures a new VirtualMemoryManager instance.

        2. How does the specific function work:
           Validates configuration parameters (positive frame counts), instantiates the policy
           registry (fifo, lru, optimal), selects the active policy, and invokes reset() to initialize
           state collections.

        3. Concepts used in the function:
           Input validation, object instantiation, policy mapping, state initialization.

        4. Places where this function is imported and used:
           Instantiated by test suites, CLI entry points, and workload execution pipelines.

        5. Final output of the function, before and after:
           Before: Uninitialized memory manager. After: Fully configured and ready simulation engine.
        """
        if vram_frame_count <= 0:
            raise ValueError(f"vram_frame_count must be positive, got {vram_frame_count}")
        if swap_frame_count <= 0:
            raise ValueError(f"swap_frame_count must be positive, got {swap_frame_count}")

        self.vram_frame_count = vram_frame_count
        self.swap_frame_count = swap_frame_count
        self.t_vram_us = float(t_vram_us)
        self.t_swap_us = float(t_swap_us)

        self.policies: Dict[str, PageReplacementPolicy] = {
            "fifo": FIFOPolicy(),
            "lru": LRUPolicy(),
            "optimal": OptimalPolicy(),
            "opt": OptimalPolicy(),
        }

        self.active_policy = self._get_policy(policy)
        self.reset()

    def _get_policy(self, policy_name: str) -> PageReplacementPolicy:
        """
        1. The use of function:
           Resolves a policy name string to a registered PageReplacementPolicy instance.

        2. How does the specific function work:
           Normalizes policy_name to lowercase and checks the self.policies registry.
           Raises ValueError if the policy is unsupported.

        3. Concepts used in the function:
           Registry lookup and string normalization.

        4. Places where this function is imported and used:
           Used internally during __init__, run_trace, and compare_policies.

        5. Final output of the function, before and after:
           Before: Policy name string. After: Matched PageReplacementPolicy instance.
        """
        key = policy_name.strip().lower()
        if key not in self.policies:
            supported = ", ".join(sorted(set(self.policies.keys())))
            raise ValueError(f"Unsupported policy: '{policy_name}'. Supported policies are: {supported}")
        return self.policies[key]

    def reset(self) -> None:
        """
        1. The use of function:
           Resets the manager state, clearing page tables, frame allocations, counters, and logs.

        2. How does the specific function work:
           Empties page_table, resets total_requests, hits, faults, clears fault_timeline,
           eviction_log, and access_history.

        3. Concepts used in the function:
           State clearing for test isolation and clean trace executions.

        4. Places where this function is imported and used:
           Called during __init__, at the beginning of run_trace, and manually between tests.

        5. Final output of the function, before and after:
           Before: Potentially dirty manager state. After: Clean, blank slate ready for execution.
        """
        self.page_table: Dict[int, PageEntry] = {}
        self.total_requests: int = 0
        self.hits: int = 0
        self.faults: int = 0
        self.fault_timeline: List[int] = []
        self.eviction_log: List[Dict[str, Any]] = []
        self.access_history: List[PageAccessResult] = []

    def access_page(
        self,
        page_id: int,
        current_step: int,
        future_trace: Optional[List[int]] = None,
    ) -> PageAccessResult:
        """
        1. The use of function:
           Simulates referencing a single virtual page at a given discrete time step.

        2. How does the specific function work:
           Checks if page_id is present in the page table and currently resident in VRAM.
           If resident (Hit): increments hit counter, updates last_accessed_at and access_count,
           appends 0 to fault_timeline.
           If non-resident (Page Fault / Miss): increments fault counter, appends 1 to fault_timeline.
           If a free frame exists, assigns the free frame. If all frames are occupied, invokes the
           active policy's select_victim() method, evicts the victim to swap, and assigns the victim's
           frame to the incoming page. Updates page table states accordingly.

        3. Concepts used in the function:
           Demand paging, page fault handling, frame allocation, victim eviction, swap transitions.

        4. Places where this function is imported and used:
           Called step-by-step by run_trace, interactive workload simulators, and unit tests.

        5. Final output of the function, before and after:
           Before: Pre-access memory state. After: Updated page table and returned PageAccessResult.
        """
        self.total_requests += 1

        if page_id not in self.page_table:
            entry = PageEntry(page_id=page_id)
            self.page_table[page_id] = entry
        else:
            entry = self.page_table[page_id]

        if entry.in_vram:
            # Cache Hit
            self.hits += 1
            self.fault_timeline.append(0)
            entry.last_accessed_at = float(current_step)
            entry.access_count += 1
            is_hit = True
            evicted_page_id = None
            assigned_frame = entry.frame_id if entry.frame_id is not None else 0
        else:
            # Cache Miss / Page Fault
            self.faults += 1
            self.fault_timeline.append(1)
            is_hit = False

            resident_pages = [p for p in self.page_table.values() if p.in_vram]

            if len(resident_pages) < self.vram_frame_count:
                # Free frame available
                occupied_frames = {p.frame_id for p in resident_pages if p.frame_id is not None}
                assigned_frame = min(f for f in range(self.vram_frame_count) if f not in occupied_frames)
                evicted_page_id = None
            else:
                # Saturation: Eviction required
                victim = self.active_policy.select_victim(resident_pages, current_step, future_trace)
                evicted_page_id = victim.page_id
                assigned_frame = victim.frame_id if victim.frame_id is not None else 0

                victim.in_vram = False
                victim.frame_id = None
                victim.in_swap = True

                self.eviction_log.append({
                    "step": current_step,
                    "evicted_page": evicted_page_id,
                    "incoming_page": page_id,
                    "frame_id": assigned_frame,
                    "policy": self.active_policy.name,
                })

            entry.in_vram = True
            entry.in_swap = False
            entry.frame_id = assigned_frame
            entry.loaded_at = float(current_step)
            entry.last_accessed_at = float(current_step)
            entry.access_count += 1

        current_pages = sorted(
            [p for p in self.page_table.values() if p.in_vram],
            key=lambda p: (p.frame_id if p.frame_id is not None else -1),
        )
        current_vram_pages = [p.page_id for p in current_pages]

        result = PageAccessResult(
            page_id=page_id,
            step=current_step,
            is_hit=is_hit,
            evicted_page_id=evicted_page_id,
            frame_id=assigned_frame,
            policy_used=self.active_policy.name,
            current_vram_pages=current_vram_pages,
        )
        self.access_history.append(result)
        return result

    def calculate_eat(
        self,
        hit_ratio: float,
        t_vram_us: Optional[float] = None,
        t_swap_us: Optional[float] = None,
    ) -> float:
        """
        1. The use of function:
           Calculates the Effective Access Time (EAT) in microseconds given a hit ratio.

        2. How does the specific function work:
           Uses standard OS formula: EAT = (hit_ratio * t_vram) + ((1 - hit_ratio) * t_swap).
           Allows caller overrides for t_vram and t_swap; otherwise falls back to configured defaults.

        3. Concepts used in the function:
           Operating systems memory hierarchy modeling, Effective Access Time equation.

        4. Places where this function is imported and used:
           Used by run_trace, compare_policies, and performance optimization analyzers.

        5. Final output of the function, before and after:
           Before: Hit ratio and latency parameters. After: Computed EAT in microseconds.
        """
        tv = float(t_vram_us) if t_vram_us is not None else self.t_vram_us
        ts = float(t_swap_us) if t_swap_us is not None else self.t_swap_us
        clamped_hit_ratio = max(0.0, min(1.0, float(hit_ratio)))
        return (clamped_hit_ratio * tv) + ((1.0 - clamped_hit_ratio) * ts)

    def run_trace(
        self,
        reference_string: List[int],
        policy_name: Optional[str] = None,
    ) -> VirtualMemoryMetrics:
        """
        1. The use of function:
           Executes a complete page reference string trace and computes summary metrics.

        2. How does the specific function work:
           Optionally switches active policy if policy_name is specified. Calls reset() to ensure
           a pristine state. Iterates through reference_string, calling access_page at each step.
           Finally computes hit_ratio, miss_ratio, EAT, and packages results into VirtualMemoryMetrics.

        3. Concepts used in the function:
           Trace-driven simulation, batch memory access processing, quantitative performance metrics.

        4. Places where this function is imported and used:
           Used in benchmarks, comparative analyses, and test assertions.

        5. Final output of the function, before and after:
           Before: Reference string trace. After: VirtualMemoryMetrics summarizing execution.
        """
        if policy_name is not None:
            self.active_policy = self._get_policy(policy_name)

        self.reset()

        if not reference_string:
            return VirtualMemoryMetrics(
                policy=self.active_policy.name,
                frame_count=self.vram_frame_count,
                total_requests=0,
                hits=0,
                faults=0,
                hit_ratio=0.0,
                miss_ratio=0.0,
                eat_us=0.0,
                fault_timeline=[],
                eviction_log=[],
            )

        for step, page_id in enumerate(reference_string):
            self.access_page(page_id=page_id, current_step=step, future_trace=reference_string)

        total = self.total_requests
        hit_ratio = (self.hits / total) if total > 0 else 0.0
        miss_ratio = (self.faults / total) if total > 0 else 0.0
        eat = self.calculate_eat(hit_ratio)

        return VirtualMemoryMetrics(
            policy=self.active_policy.name,
            frame_count=self.vram_frame_count,
            total_requests=total,
            hits=self.hits,
            faults=self.faults,
            hit_ratio=round(hit_ratio, 6),
            miss_ratio=round(miss_ratio, 6),
            eat_us=round(eat, 6),
            fault_timeline=list(self.fault_timeline),
            eviction_log=list(self.eviction_log),
        )

    def compare_policies(
        self,
        reference_string: List[int],
    ) -> Dict[str, VirtualMemoryMetrics]:
        """
        1. The use of function:
           Executes the same reference string across FIFO, LRU, and Optimal policies for comparison.

        2. How does the specific function work:
           Preserves initial active policy. Sequentially invokes run_trace for "fifo", "lru",
           and "optimal", gathering metrics for each into a dictionary keyed by policy name.
           Restores the original active policy before returning.

        3. Concepts used in the function:
           Comparative policy benchmarking, algorithmic empirical evaluation.

        4. Places where this function is imported and used:
           Used by dashboard analytics, automated reports, and comparative test suites.

        5. Final output of the function, before and after:
           Before: Reference trace. After: Dictionary mapping policy names to VirtualMemoryMetrics.
        """
        original_policy = self.active_policy.name
        results: Dict[str, VirtualMemoryMetrics] = {}

        try:
            for pol in ["fifo", "lru", "optimal"]:
                results[pol] = self.run_trace(reference_string, policy_name=pol)
        finally:
            self.active_policy = self._get_policy(original_policy)

        return results

    def detect_beladys_anomaly(
        self,
        trace: Optional[List[int]] = None,
        frame_range: Tuple[int, int] = (3, 6),
    ) -> Dict[str, Any]:
        """
        1. The use of function:
           Detects Belady's Anomaly where increasing the number of physical frames causes an
           increase in the number of page faults under FIFO replacement.

        2. How does the specific function work:
           Defaults to the canonical reference string [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5] if trace is None.
           Iterates through frame counts in frame_range (min_frames to max_frames inclusive).
           For each frame count, runs a FIFO simulation and records total faults.
           Scans consecutive frame counts to detect any instance where faults(f+1) > faults(f).
           Returns a detailed analysis dictionary.

        3. Concepts used in the function:
           Belady's Anomaly, FIFO replacement pathology, empirical anomaly detection.

        4. Places where this function is imported and used:
           Used in educational demonstrations, OS validation tests, and optimization engines.

        5. Final output of the function, before and after:
           Before: Trace and frame range. After: Dictionary with anomaly detection flag and details.
        """
        target_trace = (
            [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
            if trace is None
            else list(trace)
        )

        min_f, max_f = frame_range
        if min_f < 1:
            raise ValueError(f"Minimum frames in frame_range must be >= 1, got {min_f}")
        if max_f < min_f:
            raise ValueError(f"Invalid frame_range: max_frames ({max_f}) < min_frames ({min_f})")

        faults_by_frame: Dict[int, int] = {}
        original_frames = self.vram_frame_count
        original_policy = self.active_policy.name

        try:
            for frames in range(min_f, max_f + 1):
                self.vram_frame_count = frames
                metrics = self.run_trace(target_trace, policy_name="fifo")
                faults_by_frame[frames] = metrics.faults
        finally:
            self.vram_frame_count = original_frames
            self.active_policy = self._get_policy(original_policy)

        anomaly_details: List[Dict[str, Any]] = []
        frame_keys = sorted(faults_by_frame.keys())
        for i in range(len(frame_keys) - 1):
            f1 = frame_keys[i]
            f2 = frame_keys[i + 1]
            if faults_by_frame[f2] > faults_by_frame[f1]:
                anomaly_details.append({
                    "frames_lower": f1,
                    "faults_lower": faults_by_frame[f1],
                    "frames_higher": f2,
                    "faults_higher": faults_by_frame[f2],
                    "fault_increase": faults_by_frame[f2] - faults_by_frame[f1],
                })

        return {
            "detected": len(anomaly_details) > 0,
            "trace": target_trace,
            "policy": "fifo",
            "faults_by_frame": faults_by_frame,
            "anomaly_details": anomaly_details,
        }
