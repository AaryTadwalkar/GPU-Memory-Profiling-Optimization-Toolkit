"""
File: optimizer/activation_swapper.py

1. What this file does:
   Implements the Active Deep Learning Training Tiered Activation Swapper based on the
   Capuchin / SwapAdvisor research paradigm. Intercepts intermediate activation tensors
   saved for backward autograd passes, dynamically offloads them from accelerator VRAM to
   pinned host CPU memory under memory pressure, and prefetches them back asynchronously
   prior to backward gradient calculation, guaranteeing mathematical gradient equivalence.

2. Overall design:
   Built as a non-invasive, user-space context manager (TieredActivationManager) leveraging
   PyTorch's `torch.autograd.graph.saved_tensors_hooks`. During the forward pass, qualifying
   tensors exceeding a size threshold are detached, copied to host memory (optionally pinned
   for DMA transfer), and replaced with a lightweight metadata tuple (PackedActivation).
   During the backward pass, the hook unpacks the activation and transfers it back to the
   originating device. Telemetry events and aggregate performance summaries are captured via
   SwapEventRecord and ActivationSwapSummary dataclasses. Includes CPU simulation mode for
   safe, deterministic execution without physical GPU hardware.

3. Elements in the file:
   - PackedActivation: Immutable tuple subclass encapsulating packed tensor metadata.
   - SwapEventRecord: Dataclass recording telemetry for individual offload/prefetch events.
   - ActivationSwapSummary: Dataclass summarizing aggregate memory, count, and timing telemetry.
   - TieredActivationManager: Context manager orchestrating graph hooks, offloading, and prefetching.

4. Final output of the file:
   Exposes high-level activation swapping manager and telemetry data structures for integration
   into deep learning training loops, test harnesses, optimization engines, and dashboards.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PackedActivation(tuple):
    """
    1. The use of class:
       Subclasses tuple to represent packed activation tensor data during PyTorch
       autograd forward-to-backward graph retention. It preserves standard 5-tuple
       unpacking semantics while retaining metadata such as tensor ID.

    2. What the class contain:
       Stores stored tensor reference, swap status flag, original device string,
       original tensor shape, dtype, and tensor identifier attribute.

    3. Technologies used and why:
       Python tuple subclassing for zero-overhead, immutable, memory-efficient
       interoperability with PyTorch autograd graph saved tensor hook signatures.

    4. Key design and overall design of the class:
       Enables backward-compatible 5-element tuple destructuring (stored_tensor,
       was_swapped, original_device, shape, dtype) while attaching tensor_id as an
       instance attribute for unambiguous chronology tracking.
    """

    def __new__(
        cls,
        stored_tensor: torch.Tensor,
        was_swapped: bool,
        original_device: Union[str, torch.device],
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        tensor_id: int = 0,
    ) -> "PackedActivation":
        """
        1. The use of function:
           Constructs and initializes an immutable PackedActivation tuple instance.

        2. How does the specific function work:
           Invokes super().__new__ with the canonical 5-element tuple and attaches
           tensor_id directly to the created instance.

        3. Concepts used in the function:
           Python object model `__new__` instantiation for immutable tuple subclassing.

        4. Places where this function is imported and used:
           Used internally by TieredActivationManager.pack_hook to package saved tensors.

        5. Final output of the function, before and after:
           Before: raw activation parameters and metadata.
           After: unified 5-element tuple instance with tensor_id attribute.
        """
        instance = super().__new__(
            cls, (stored_tensor, was_swapped, str(original_device), tuple(shape), dtype)
        )
        instance.tensor_id = tensor_id
        return instance


@dataclass
class SwapEventRecord:
    """
    1. The use of class:
       Records telemetry and timing data for an individual activation offload or prefetch
       lifecycle event between accelerator VRAM and host RAM.

    2. What the class contain:
       Contains timestamp, tensor ID, tensor payload size in bytes, event type ('OFFLOAD'
       or 'PREFETCH'), VRAM usage before the event in MB, VRAM usage after the event in MB,
       transfer duration in milliseconds, source device, target device, and optional tensor shape tuple.

    3. Technologies used and why:
       Standard library `@dataclass` for lightweight, strongly-typed, low-overhead event logging.

    4. Key design and overall design of the class:
       Provides a telemetry record conforming to Capuchin/SwapAdvisor research logging models,
       supporting chronological reconstruction and JSON serialization via `to_dict()`.
    """

    timestamp: float
    tensor_id: int
    size_bytes: int
    event_type: str
    vram_before_mb: float
    vram_after_mb: float
    duration_ms: float
    device_source: str
    device_target: str
    tensor_shape: Optional[Tuple[int, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Serializes the SwapEventRecord instance into a standard Python dictionary.

        2. How does the specific function work:
           Calls `dataclasses.asdict` on self and returns the resulting dictionary.

        3. Concepts used in the function:
           Dataclass reflection and dictionary conversion for JSON export and telemetry analytics.

        4. Places where this function is imported and used:
           Called by TieredActivationManager.get_summary and downstream logging pipelines.

        5. Final output of the function, before and after:
           Before: strongly typed SwapEventRecord dataclass.
           After: standard dictionary containing all event attributes.
        """
        return asdict(self)


@dataclass
class ActivationSwapSummary:
    """
    1. The use of class:
       Aggregates global metrics and telemetry across an entire forward and backward execution pass,
       quantifying tensor counts, transferred memory volumes, transfer latencies, and VRAM reduction.

    2. What the class contain:
       Contains total tensors intercepted, total tensors offloaded, total bytes offloaded, total
       bytes prefetched, cumulative offload latency in ms, cumulative prefetch latency in ms,
       peak VRAM reduction percentage, and a list of serialized swap event dictionaries.

    3. Technologies used and why:
       Python `@dataclass` for concise structure, schema validation, and clean serialization.

    4. Key design and overall design of the class:
       Designed as the primary analytical artifact produced by the tiered activation swapping engine,
       enabling direct comparison against baseline non-swapped training passes.
    """

    total_tensors_intercepted: int
    total_tensors_offloaded: int
    total_bytes_offloaded: int
    total_bytes_prefetched: int
    offload_duration_ms: float
    prefetch_duration_ms: float
    peak_vram_reduction_pct: float
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Serializes the ActivationSwapSummary instance into a JSON-compatible dictionary.

        2. How does the specific function work:
           Converts internal metrics and deep copies the event dictionaries via `asdict`.

        3. Concepts used in the function:
           Dictionary serialization and telemetry aggregation.

        4. Places where this function is imported and used:
           Exported to monitoring tools, unit test assertions, CLI demos, and UI visualizations.

        5. Final output of the function, before and after:
           Before: typed ActivationSwapSummary instance.
           After: JSON-serializable dictionary representing the complete execution summary.
        """
        return asdict(self)


class TieredActivationManager:
    """
    1. The use of class:
       Manages tiered memory offloading and prefetching for deep learning training activations
       following the Capuchin / SwapAdvisor research paradigm. Transparently intercepts saved
       intermediate tensors during the forward pass, evicts qualifying tensors from accelerator
       VRAM to pinned host CPU memory, and prefetches them back just in time for the backward pass.

    2. What the class contain:
       Configuration thresholds (pressure threshold, minimum tensor size, pinned memory flag,
       enabled flag, simulated CPU flag), lifecycle counters, telemetry event storage, context
       manager handles, and hook implementations (pack_hook, unpack_hook).

    3. Technologies used and why:
       PyTorch `torch.autograd.graph.saved_tensors_hooks` for non-invasive, graph-level interception
       without altering model code or custom backward autograd functions; pinned host memory for
       asynchronous DMA PCIe transfers.

    4. Key design and overall design of the class:
       Acts as a re-entrant Python context manager wrapping PyTorch's saved tensors hook mechanism.
       Computes memory pressure dynamically, offloads tensors satisfying size criteria, maintains
       mathematical gradient equivalence, and provides full CPU simulation safety for testing.
    """

    def __init__(
        self,
        pressure_threshold_bytes: int = 4 * 1024**3,
        min_tensor_size_bytes: int = 256 * 1024,
        use_pinned_memory: bool = True,
        enabled: bool = True,
        simulate_on_cpu: bool = False,
    ) -> None:
        """
        1. The use of function:
           Initializes the TieredActivationManager with memory thresholds, flags, and telemetry state.

        2. How does the specific function work:
           Stores user configuration thresholds and initializes internal telemetry counters,
           event histories, and context manager tracking references.

        3. Concepts used in the function:
           Object initialization, configuration encapsulation, and telemetry state setup.

        4. Places where this function is imported and used:
           Instantiated by training loops, workload scripts, test harnesses, and controller modules.

        5. Final output of the function, before and after:
           Before: uninitialized class instance.
           After: fully configured activation manager ready for context entry and hook registration.
        """
        self.pressure_threshold_bytes: int = pressure_threshold_bytes
        self.min_tensor_size_bytes: int = min_tensor_size_bytes
        self.use_pinned_memory: bool = use_pinned_memory
        self.enabled: bool = enabled
        self.simulate_on_cpu: bool = simulate_on_cpu

        # Telemetry metrics
        self.total_tensors_intercepted: int = 0
        self.total_tensors_offloaded: int = 0
        self.total_bytes_offloaded: int = 0
        self.total_bytes_prefetched: int = 0
        self.offload_duration_ms: float = 0.0
        self.prefetch_duration_ms: float = 0.0
        self.events: List[SwapEventRecord] = []

        # Internal state tracking
        self._tensor_counter: int = 0
        self._total_bytes_intercepted: int = 0
        self._current_resident_bytes: int = 0
        self._hook_context: Any = None

    def __enter__(self) -> "TieredActivationManager":
        """
        1. The use of function:
           Activates the context manager and registers PyTorch saved tensors autograd hooks.

        2. How does the specific function work:
           If enabled, instantiates `torch.autograd.graph.saved_tensors_hooks` passing self.pack_hook
           and self.unpack_hook, invokes `__enter__()` on the PyTorch hook context, and returns self.

        3. Concepts used in the function:
           Python context management protocol and PyTorch graph-level hook registration.

        4. Places where this function is imported and used:
           Invoked when entering a `with TieredActivationManager(...):` block around forward pass code.

        5. Final output of the function, before and after:
           Before: dormant manager with inactive hooks.
           After: active hook context actively intercepting autograd saved tensors.
        """
        if self.enabled:
            self._hook_context = torch.autograd.graph.saved_tensors_hooks(
                self.pack_hook, self.unpack_hook
            )
            self._hook_context.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        1. The use of function:
           Deactivates the context manager and unregisters PyTorch saved tensors hooks.

        2. How does the specific function work:
           If `_hook_context` is active, invokes its `__exit__()` method and resets the reference to None.

        3. Concepts used in the function:
           Context management cleanup and deterministic resource unregistration.

        4. Places where this function is imported and used:
           Invoked automatically when exiting a `with TieredActivationManager(...):` block.

        5. Final output of the function, before and after:
           Before: active hook context intercepting new tensor operations.
           After: hooks unregistered, restoring default PyTorch autograd behavior.
        """
        if self._hook_context is not None:
            self._hook_context.__exit__(exc_type, exc_val, exc_tb)
            self._hook_context = None

    def pack_hook(self, tensor: torch.Tensor) -> PackedActivation:
        """
        1. The use of function:
           Intercepts tensors saved by autograd in the forward pass to decide residency or offloading.

        2. How does the specific function work:
           Inspects tensor metadata, increments interception counters, checks size thresholds and
           memory pressure conditions. If eligible for offloading, detaches and transfers tensor data
           to host RAM (pinning memory if CUDA is active), records an OFFLOAD telemetry event, and
           returns a PackedActivation tuple. Otherwise keeps the tensor resident.

        3. Concepts used in the function:
           Autograd hook packing, tensor detachment, host DMA memory transfer, memory pressure evaluation.

        4. Places where this function is imported and used:
           Invoked automatically by PyTorch autograd engine when any forward operator saves activations.

        5. Final output of the function, before and after:
           Before: raw forward activation tensor resident on current device.
           After: PackedActivation 5-tuple holding resident or offloaded tensor data.
        """
        self.total_tensors_intercepted += 1
        self._tensor_counter += 1
        tensor_id = self._tensor_counter

        size_bytes = tensor.numel() * tensor.element_size()
        self._total_bytes_intercepted += size_bytes

        # Pass-through if disabled
        if not self.enabled:
            return PackedActivation(
                tensor, False, str(tensor.device), tuple(tensor.shape), tensor.dtype, tensor_id
            )

        # Filter out small tensors below threshold to avoid transfer overhead
        if size_bytes < self.min_tensor_size_bytes:
            self._current_resident_bytes += size_bytes
            return PackedActivation(
                tensor, False, str(tensor.device), tuple(tensor.shape), tensor.dtype, tensor_id
            )

        # Memory pressure evaluation
        should_swap = False
        if self.simulate_on_cpu:
            should_swap = True
        elif tensor.device.type == "cuda" and torch.cuda.is_available():
            current_allocated = torch.cuda.memory_allocated(tensor.device)
            should_swap = current_allocated >= self.pressure_threshold_bytes
        else:
            should_swap = False

        if not should_swap:
            self._current_resident_bytes += size_bytes
            return PackedActivation(
                tensor, False, str(tensor.device), tuple(tensor.shape), tensor.dtype, tensor_id
            )

        # Execute offload to host memory
        if torch.cuda.is_available() and tensor.device.type == "cuda":
            vram_before_mb = torch.cuda.memory_allocated(tensor.device) / (1024 * 1024)
        else:
            vram_before_mb = self._current_resident_bytes / (1024 * 1024)

        start_time = time.perf_counter()
        if tensor.device.type == "cuda":
            try:
                if self.use_pinned_memory:
                    stored_tensor = tensor.detach().to("cpu", non_blocking=True)
                    if torch.cuda.is_available() and not stored_tensor.is_pinned():
                        try:
                            stored_tensor = stored_tensor.pin_memory()
                        except Exception:
                            pass
                else:
                    stored_tensor = tensor.detach().cpu()
            except Exception:
                stored_tensor = tensor.detach().cpu()
        else:
            # CPU simulation mode
            stored_tensor = tensor.detach().clone()
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        self.total_tensors_offloaded += 1
        self.total_bytes_offloaded += size_bytes
        self.offload_duration_ms += duration_ms

        if torch.cuda.is_available() and tensor.device.type == "cuda":
            vram_after_mb = torch.cuda.memory_allocated(tensor.device) / (1024 * 1024)
        else:
            vram_after_mb = max(0.0, vram_before_mb - (size_bytes / (1024 * 1024)))

        event = SwapEventRecord(
            timestamp=time.time(),
            tensor_id=tensor_id,
            size_bytes=size_bytes,
            event_type="OFFLOAD",
            vram_before_mb=round(vram_before_mb, 4),
            vram_after_mb=round(vram_after_mb, 4),
            duration_ms=round(duration_ms, 4),
            device_source=str(tensor.device),
            device_target="cpu",
            tensor_shape=tuple(tensor.shape),
        )
        self.events.append(event)

        return PackedActivation(
            stored_tensor, True, str(tensor.device), tuple(tensor.shape), tensor.dtype, tensor_id
        )

    def unpack_hook(self, packed_data: Any) -> torch.Tensor:
        """
        1. The use of function:
           Restores activation tensors needed for gradient calculation during backward autograd.

        2. How does the specific function work:
           Unpacks the 5-element PackedActivation tuple. If the activation remained resident, returns
           it directly. If it was swapped to host memory, transfers it back to original device,
           measures prefetch latency, records a PREFETCH telemetry event, and returns the restored tensor.

        3. Concepts used in the function:
           Autograd hook unpacking, host-to-device asynchronous prefetch, telemetry latency profiling.

        4. Places where this function is imported and used:
           Invoked automatically by PyTorch autograd engine as graph backward nodes execute.

        5. Final output of the function, before and after:
           Before: PackedActivation tuple containing host-resident detached tensor.
           After: reconstructed PyTorch tensor resident on target device, ready for backward pass.
        """
        stored_tensor, was_swapped, original_device_str, shape, dtype = packed_data[:5]
        tensor_id = getattr(packed_data, "tensor_id", None)
        if tensor_id is None and len(packed_data) > 5:
            tensor_id = packed_data[5]
        if tensor_id is None:
            tensor_id = 0

        if not was_swapped:
            return stored_tensor

        size_bytes = stored_tensor.numel() * stored_tensor.element_size()
        target_device = (
            torch.device(original_device_str)
            if isinstance(original_device_str, str)
            else original_device_str
        )

        if torch.cuda.is_available() and target_device.type == "cuda":
            vram_before_mb = torch.cuda.memory_allocated(target_device) / (1024 * 1024)
        else:
            vram_before_mb = self._current_resident_bytes / (1024 * 1024)

        start_time = time.perf_counter()
        restored_tensor = stored_tensor.to(target_device, non_blocking=True)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        self.total_bytes_prefetched += size_bytes
        self.prefetch_duration_ms += duration_ms
        self._current_resident_bytes += size_bytes

        if torch.cuda.is_available() and target_device.type == "cuda":
            vram_after_mb = torch.cuda.memory_allocated(target_device) / (1024 * 1024)
        else:
            vram_after_mb = vram_before_mb + (size_bytes / (1024 * 1024))

        event = SwapEventRecord(
            timestamp=time.time(),
            tensor_id=tensor_id,
            size_bytes=size_bytes,
            event_type="PREFETCH",
            vram_before_mb=round(vram_before_mb, 4),
            vram_after_mb=round(vram_after_mb, 4),
            duration_ms=round(duration_ms, 4),
            device_source="cpu",
            device_target=str(target_device),
            tensor_shape=tuple(shape) if shape else None,
        )
        self.events.append(event)

        return restored_tensor

    def get_summary(self) -> ActivationSwapSummary:
        """
        1. The use of function:
           Aggregates and computes overall activation swapping telemetry and memory reduction metrics.

        2. How does the specific function work:
           Calculates peak VRAM reduction percentage based on total offloaded bytes relative to total
           intercepted activation bytes, formats timing totals, converts event records, and returns
           an ActivationSwapSummary instance.

        3. Concepts used in the function:
           Metrics aggregation, memory reduction percentage calculation, and summary dataclass construction.

        4. Places where this function is imported and used:
           Called after forward/backward iterations by evaluation scripts, benchmarkers, and dashboards.

        5. Final output of the function, before and after:
           Before: raw internal manager counters and event lists.
           After: clean ActivationSwapSummary object capturing high-level swap performance.
        """
        if self._total_bytes_intercepted > 0 and self.total_bytes_offloaded > 0:
            reduction_pct = (self.total_bytes_offloaded / self._total_bytes_intercepted) * 100.0
            peak_vram_reduction_pct = round(min(100.0, max(0.0, reduction_pct)), 2)
        else:
            peak_vram_reduction_pct = 0.0

        return ActivationSwapSummary(
            total_tensors_intercepted=self.total_tensors_intercepted,
            total_tensors_offloaded=self.total_tensors_offloaded,
            total_bytes_offloaded=self.total_bytes_offloaded,
            total_bytes_prefetched=self.total_bytes_prefetched,
            offload_duration_ms=round(self.offload_duration_ms, 4),
            prefetch_duration_ms=round(self.prefetch_duration_ms, 4),
            peak_vram_reduction_pct=peak_vram_reduction_pct,
            events=[e.to_dict() for e in self.events],
        )

    def reset(self) -> None:
        """
        1. The use of function:
           Resets all telemetry counters, event logs, and state trackers to initial zero values.

        2. How does the specific function work:
           Clears `events` list, zeroes integer counters, floats, and resident bytes accumulators.

        3. Concepts used in the function:
           State clearing and manager re-initialization between training iterations or test runs.

        4. Places where this function is imported and used:
           Called between training epochs or iterations to produce clean per-step metrics.

        5. Final output of the function, before and after:
           Before: manager with accumulated telemetry from previous runs.
           After: clean manager state ready for fresh observation.
        """
        self.total_tensors_intercepted = 0
        self.total_tensors_offloaded = 0
        self.total_bytes_offloaded = 0
        self.total_bytes_prefetched = 0
        self.offload_duration_ms = 0.0
        self.prefetch_duration_ms = 0.0
        self.events.clear()
        self._tensor_counter = 0
        self._total_bytes_intercepted = 0
        self._current_resident_bytes = 0
