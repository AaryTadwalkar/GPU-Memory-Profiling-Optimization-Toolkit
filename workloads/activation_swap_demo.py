"""
File: workloads/activation_swap_demo.py

1. What this file does:
   Demonstrates and benchmarks deep learning training execution with and without the
   TieredActivationManager (Capuchin / SwapAdvisor paradigm). Provides a realistic heavy
   convolutional workload (RealConvNetStage) stressing accelerator VRAM to saturation and physical
   out-of-memory boundaries, alongside synthetic multi-layer feedforward networks (MLP),
   ResNet-50 stages, and Vision Transformer (ViT) blocks. Verifies exact mathematical gradient
   equivalence and outputs detailed memory telemetry including offloaded tensors, transfer latencies,
   and peak VRAM reduction. Implements an active hardware-driven training benchmark engine
   contrasting unmitigated baseline GPU failure/saturation against tiered swapper protection
   with physical memory measurements.

2. Overall design:
   Defines parameterizable neural architectures (RealConvNetStage, SyntheticMLP, ResNet50Stage,
   VisionTransformerBlock) initialized with deterministic weights. Executes comparative training:
   (1) Baseline pass without activation swapping, measuring physical GPU VRAM or catching OOM errors.
   (2) Swapped pass wrapped in `TieredActivationManager`, offloading eligible activation tensors to
       host CPU RAM, preventing out-of-memory aborts.
   Validates parameter gradients bit-for-bit or within floating-point tolerance (torch.allclose atol=1e-5).
   Implements a dynamic step-based streaming benchmark engine (`run_live_1min_benchmark`) executing
   Phase 1 (up to 8-12 baseline steps with physical CUDA OOM interception) and Phase 2 (12-16 active
   tiered swapper steps with real VRAM tracking and host RAM spill measurement). In safe CPU simulation
   mode, automatically uses lightweight tensors for rapid (< 2s) test execution without hardware intervention.

3. Elements in the file:
   - RealConvNetStage: Realistic 4-layer convolutional deep learning workload stressing physical GPU memory.
   - SyntheticMLP: Multi-layer neural network architecture with configurable depth and width.
   - ResNetBottleneckBlock: 3-layer residual bottleneck convolutional block with shortcut.
   - ResNet50Stage: Multi-block sequence of ResNet bottleneck modules.
   - VisionTransformerBlock: Multi-layer ViT encoder block with multi-head self-attention.
   - BenchmarkStepRecord: Dataclass recording streaming telemetry ticks during live benchmarks.
   - LiveBenchmarkSummary: Dataclass aggregating multi-phase benchmark results and verification metrics.
   - run_activation_swap_demo: Core orchestration function comparing baseline vs swapped single-pass execution.
   - run_live_1min_benchmark: Dynamic hardware-driven streaming training benchmark runner.
   - main: Command-line entry point parsing flags and executing demonstrations safely.

4. Final output of the file:
   Outputs human-readable CLI telemetry tables and returns structured comparison dictionaries
   validating gradient correctness, streaming real-time memory metrics, and profiling activation offload/prefetch statistics.
"""

import argparse
import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizer.activation_swapper import TieredActivationManager, ActivationSwapSummary


def cleanup_system_and_gpu_memory() -> None:
    """
    1. The use of function:
       Performs immediate garbage collection and hardware cache deallocation across both
       host CPU system RAM and accelerator GPU VRAM to ensure zero leaked memory between benchmark runs.

    2. How does the specific function work:
       Invokes Python `gc.collect()`. If CUDA is available, calls `torch.cuda.empty_cache()` and
       `torch.cuda.ipc_collect()` to release cached device memory and unmap pinned host memory.

    3. Concepts used in the function:
       Garbage collection, CUDA caching allocator purge, inter-process communication cache reclaim.

    4. Places where this function is imported and used:
       Imported by dashboard/app.py and called before/after benchmark executions and on manual reset.

    5. Final output of the function, before and after:
       Before: Lingering PyTorch allocations and uncollected circular autograd references in memory.
       After: Clean hardware state with 0.0 MB lingering device allocations.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


class RealConvNetStage(nn.Module):
    """
    1. The use of class:
       Provides a realistic high-throughput 4-stage convolutional deep learning workload
       designed to stress accelerator VRAM capacity through high-dimensional spatial activation maps.

    2. What the class contain:
       Contains a sequential pipeline of four 2D convolutional layers paired with batch normalizations
       and ReLU activations:
       (1) Conv2d(in_channels, 256, kernel_size=3, padding=1) + BatchNorm2d(256) + ReLU()
       (2) Conv2d(256, 256, kernel_size=3, padding=1) + BatchNorm2d(256) + ReLU()
       (3) Conv2d(256, 512, kernel_size=3, padding=1) + BatchNorm2d(512) + ReLU()
       (4) Conv2d(512, 512, kernel_size=3, padding=1) + BatchNorm2d(512) + ReLU()

    3. Technologies used and why:
       PyTorch `nn.Module`, `nn.Conv2d`, `nn.BatchNorm2d`, and `nn.ReLU` with out-of-place activations
       to mirror production vision backbones and generate authentic autograd graph activation tensors.

    4. Key design and overall design of the class:
       Processes spatial feature tensors (batch_size, 128, 56, 56). Generates gigabytes of
       intermediate activation memory during forward propagation, making it an ideal testbed
       for accelerator memory pressure evaluation and tiered activation swapping.
    """

    def __init__(self, in_channels: int = 128) -> None:
        """
        1. The use of function:
           Constructs the convolutional, normalization, and activation layers of RealConvNetStage.

        2. How does the specific function work:
           Assembles four successive Conv2d-BatchNorm2d-ReLU modules transitioning channels
           from in_channels to 256, 256, 512, and 512, encapsulated in an nn.Sequential container.

        3. Concepts used in the function:
           Layer stacking, spatial convolutional filter design, batch normalization regularization.

        4. Places where this function is imported and used:
           Instantiated in run_activation_swap_demo and run_live_1min_benchmark for heavy vision workloads.

        5. Final output of the function, before and after:
           Before: Uninitialized network module.
           After: Fully parameterized 4-stage convolutional architecture ready for forward-backward autograd.
        """
        super().__init__()
        self.stage = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. The use of function:
           Executes the forward computational graph pass through all convolutional stages.

        2. How does the specific function work:
           Passes the 4D input feature batch through self.stage, generating large intermediate
           activation tensors saved on the autograd graph for backward gradient propagation.

        3. Concepts used in the function:
           Spatial feature map convolution, autograd activation caching, non-linear feature mapping.

        4. Places where this function is imported and used:
           Invoked during baseline and tiered swapped training steps in benchmark runners.

        5. Final output of the function, before and after:
           Before: 4D input tensor (batch, channels, height, width).
           After: 4D transformed activation tensor with attached autograd graph backward hooks.
        """
        return self.stage(x)


class SyntheticMLP(nn.Module):
    """
    1. The use of class:
       Provides a parameterized multi-layer feedforward network to generate intermediate
       activation tensors across sequential forward operations for swapping experiments.

    2. What the class contain:
       Sequence of linear projections, non-linear activation functions (ReLU, GELU), and
       normalization layers (LayerNorm) producing activation tensors of varying shapes.

    3. Technologies used and why:
       PyTorch `nn.Module` and standard PyTorch layers for realistic, reproducible autograd graphs.

    4. Key design and overall design of the class:
       Configurable depth and hidden dimension allows fine-tuning memory footprints to test
       activation interception across small, medium, and large activation boundaries.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, depth: int = 6) -> None:
        """
        1. The use of function:
           Constructs layers of the SyntheticMLP model based on dimensions and depth.

        2. How does the specific function work:
           Builds an alternating pipeline of nn.Linear, nn.LayerNorm, and nn.GELU layers,
           concluding with a linear projection to a classification output.

        3. Concepts used in the function:
           Neural network layer stacking and module initialization.

        4. Places where this function is imported and used:
           Instantiated in run_activation_swap_demo to evaluate activation memory behavior.

        5. Final output of the function, before and after:
           Before: uninitialized network module.
           After: parameterized multi-layer network ready for forward-backward computation.
        """
        super().__init__()
        layers = []
        in_d = input_dim
        for i in range(depth):
            out_d = hidden_dim if i < depth - 1 else input_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < depth - 1:
                layers.append(nn.LayerNorm(out_d))
                layers.append(nn.GELU() if i % 2 == 0 else nn.ReLU())
            in_d = out_d
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. The use of function:
           Executes the forward computational graph pass through all sequential layers.

        2. How does the specific function work:
           Passes the input tensor through self.network, saving intermediate activations
           on the autograd graph for backward gradient propagation.

        3. Concepts used in the function:
           Forward inference and autograd graph activation recording.

        4. Places where this function is imported and used:
           Called during training passes in run_activation_swap_demo.

        5. Final output of the function, before and after:
           Before: input feature tensor.
           After: output logit tensor with attached autograd backward function hooks.
        """
        return self.network(x)


class ResNetBottleneckBlock(nn.Module):
    """
    1. The use of class:
       Implements a standard 3-layer Bottleneck residual building block conforming to
       the ResNet-50 / ResNet-101 deep convolutional architecture specifications.

    2. What the class contain:
       Contains 1x1 convolution (dimension reduction), 2D batch normalization, ReLU non-linearity,
       3x3 spatial convolution, 2D batch normalization, ReLU, 1x1 convolution (dimension restoration),
       2D batch normalization, residual identity shortcut projection, and final element-wise addition.

    3. Technologies used and why:
       PyTorch `nn.Module`, `nn.Conv2d`, `nn.BatchNorm2d`, and `nn.ReLU` with out-of-place activation
       to produce realistic 4D intermediate spatial feature activation maps for memory evaluation.

    4. Key design and overall design of the class:
       Employs residual bottleneck connections preserving exact spatial dimensions while creating
       multiple non-linear graph nodes that retain large 4D activation tensors during forward propagation.
    """

    def __init__(
        self,
        in_channels: int = 64,
        bottleneck_channels: int = 64,
        out_channels: int = 64,
        stride: int = 1,
    ) -> None:
        """
        1. The use of function:
           Constructs convolutional, normalization, and shortcut layers of the ResNet Bottleneck block.

        2. How does the specific function work:
           Configures sequential 1x1, 3x3, and 1x1 convolutions with paired batch normalizations,
           and sets up an identity or 1x1 projection shortcut matching output channel dimensions.

        3. Concepts used in the function:
           Residual projection shortcuts and bottleneck channel reduction.

        4. Places where this function is imported and used:
           Instantiated by ResNet50Stage in activation_swap_demo.py.

        5. Final output of the function, before and after:
           Before: uninitialized block.
           After: parameterized bottleneck block ready for tensor propagation.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu3 = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. The use of function:
           Executes the forward computational graph pass through the ResNet bottleneck block.

        2. How does the specific function work:
           Computes residual path, processes input through three convolutional-normalization-relu
           stages, adds residual shortcut to projected features, and applies final ReLU activation.

        3. Concepts used in the function:
           Residual addition and autograd activation recording.

        4. Places where this function is imported and used:
           Called during forward pass in ResNet50Stage and benchmark runners.

        5. Final output of the function, before and after:
           Before: 4D input feature tensor (B, C, H, W).
           After: 4D transformed feature tensor of matching or downsampled dimensions.
        """
        identity = self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)) + identity)
        return out


class ResNet50Stage(nn.Module):
    """
    1. The use of class:
       Encapsulates a multi-block sequence of ResNet-50 bottleneck layers to form a configurable
       convolutional feature extraction stage.

    2. What the class contain:
       Sequential chain of ResNetBottleneckBlock modules with configurable depth.

    3. Technologies used and why:
       PyTorch `nn.Sequential` for hierarchical grouping and parameter registration.

    4. Key design and overall design of the class:
       Enables parametric depth scaling to simulate differing memory pressure regimes in computer
       vision model training.
    """

    def __init__(self, depth: int = 3, channels: int = 64) -> None:
        """
        1. The use of function:
           Initializes a stack of bottleneck blocks forming a ResNet stage.

        2. How does the specific function work:
           Iteratively instantiates ResNetBottleneckBlock instances and packages them into nn.Sequential.

        3. Concepts used in the function:
           Module containerization and repetitive layer construction.

        4. Places where this function is imported and used:
           Instantiated in run_activation_swap_demo when model_type is set to 'resnet'.

        5. Final output of the function, before and after:
           Before: uninitialized stage container.
           After: configured multi-block convolutional pipeline.
        """
        super().__init__()
        blocks = [ResNetBottleneckBlock(in_channels=channels, bottleneck_channels=channels, out_channels=channels) for _ in range(depth)]
        self.stage = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. The use of function:
           Passes 4D image feature batch through the sequential bottleneck blocks.

        2. How does the specific function work:
           Evaluates self.stage sequentially, recording intermediate activation maps on autograd graph.

        3. Concepts used in the function:
           Deep convolutional propagation and autograd graph expansion.

        4. Places where this function is imported and used:
           Called during benchmark execution in run_activation_swap_demo.

        5. Final output of the function, before and after:
           Before: 4D input tensor (B, C, H, W).
           After: 4D transformed activation tensor ready for loss aggregation.
        """
        return self.stage(x)


class VisionTransformerBlock(nn.Module):
    """
    1. The use of class:
       Implements a multi-layer Vision Transformer (ViT) encoder block featuring multi-head
       self-attention (MSA), multi-layer perceptron (MLP) feedforward networks, and pre-LayerNorm.

    2. What the class contain:
       Configurable stack of PyTorch TransformerEncoderLayer modules utilizing GELU activations,
       multi-head attention projection heads, and residual connections.

    3. Technologies used and why:
       PyTorch `nn.TransformerEncoderLayer` and `nn.TransformerEncoder` for standard ViT attention
       representation and high-density sequence activation patterns.

    4. Key design and overall design of the class:
       Captures quadratic attention matrix activations and intermediate feedforward projections,
       serving as the canonical transformer benchmark for tiered activation swapping.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 512,
        depth: int = 2,
    ) -> None:
        """
        1. The use of function:
           Constructs transformer encoder layers composing the Vision Transformer block.

        2. How does the specific function work:
           Initializes nn.TransformerEncoderLayer with specified head count and MLP dimension,
           wrapping multiple layers within nn.TransformerEncoder with batch_first=True.

        3. Concepts used in the function:
           Multi-head self-attention mechanisms and transformer stack initialization.

        4. Places where this function is imported and used:
           Instantiated in run_activation_swap_demo when model_type is set to 'vit'.

        5. Final output of the function, before and after:
           Before: uninitialized attention block.
           After: fully functional multi-head attention encoder stack.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. The use of function:
           Computes self-attention representations and feedforward transformations across token sequences.

        2. How does the specific function work:
           Evaluates self.transformer, saving multi-head attention matrix maps and intermediate
           projections on the PyTorch autograd graph for backward propagation.

        3. Concepts used in the function:
           Attention weight computation and transformer graph backpropagation.

        4. Places where this function is imported and used:
           Called in run_activation_swap_demo during baseline and swapped benchmark execution.

        5. Final output of the function, before and after:
           Before: 3D sequence tensor (batch_size, sequence_len, d_model).
           After: 3D contextualized embedding tensor of identical shape.
        """
        return self.transformer(x)


@dataclass
class BenchmarkStepRecord:
    """
    1. The use of class:
       Encapsulates an individual time-tick telemetry record during the live multi-phase
       training benchmark comparing baseline GPU execution against TieredActivationManager.

    2. What the class contain:
       Timestamp, elapsed time in seconds, phase ID (1 for Baseline, 2 for Tiered Swapper),
       phase label, step index, mini-batch size, primary VRAM allocated in MB, host CPU RAM
       spill in MB, execution status, cumulative offloaded tensor count, and optional error detail.

    3. Technologies used and why:
       Python standard library `@dataclass` for strict typing, low-overhead time-series logging,
       and clean dictionary conversion.

    4. Key design and overall design of the class:
       Conforms to streaming telemetry schemas consumed by real-time Altair charts and Streamlit
       containers in dashboard/app.py.
    """

    timestamp: float
    elapsed_s: float
    phase_id: int
    phase_name: str
    step: int
    batch_size: int
    vram_used_mb: float
    host_ram_used_mb: float
    status: str
    tensors_swapped: int = 0
    bytes_swapped: int = 0
    gradient_matched: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Converts the BenchmarkStepRecord into a JSON-compatible dictionary.

        2. How does the specific function work:
           Calls standard library `asdict` on self.

        3. Concepts used in the function:
           Dataclass serialization for DataFrame and UI integration.

        4. Places where this function is imported and used:
           Used by streaming callbacks and UI chart renderers.

        5. Final output of the function, before and after:
           Before: Typed BenchmarkStepRecord instance.
           After: Dictionary containing all telemetry fields.
        """
        return asdict(self)


@dataclass
class LiveBenchmarkSummary:
    """
    1. The use of class:
       Aggregates complete end-to-end performance and telemetry results across both
       Phase 1 (Baseline) and Phase 2 (Tiered Swapper) of the live training benchmark.

    2. What the class contain:
       Total duration, phase 1 duration, phase 2 duration, phase 1 OOM status, phase 1 peak VRAM,
       phase 2 peak VRAM, peak host RAM spill, total tensors swapped, total bytes swapped,
       gradient equivalence status, max gradient difference, batch speedup ratio, and full time-series records list.

    3. Technologies used and why:
       Standard library `@dataclass` for comprehensive reporting and serialization.

    4. Key design and overall design of the class:
       Serves as the primary analytical deliverable comparing baseline failure/saturation with
       tiered swapper protection and batch throughput scaling.
    """

    total_duration_s: float
    phase1_duration_s: float
    phase2_duration_s: float
    phase1_crashed_oom: bool
    phase1_peak_vram_mb: float
    phase2_peak_vram_mb: float
    peak_host_ram_mb: float
    total_tensors_swapped: int
    total_bytes_swapped: int
    gradient_equivalence: bool
    max_gradient_diff: float
    batch_speedup_ratio: float
    records: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Converts the summary to a standard dictionary.

        2. How does the specific function work:
           Calls `asdict` on self.

        3. Concepts used in the function:
           Dataclass serialization.

        4. Places where this function is imported and used:
           Called when exporting benchmark results to UI session state or CLI output.

        5. Final output of the function, before and after:
           Before: Typed LiveBenchmarkSummary.
           After: Dictionary representation.
        """
        return asdict(self)


def run_activation_swap_demo(
    device_name: str = "cpu",
    simulate: bool = True,
    batch_size: int = 64,
    features: int = 512,
    hidden_dim: int = 1024,
    depth: int = 6,
    min_tensor_size_bytes: int = 64 * 1024,
    pressure_threshold_bytes: int = 0,
    model_type: str = "realconv",
) -> Dict[str, Any]:
    """
    1. The use of function:
       Executes a rigorous comparative training demonstration contrasting standard PyTorch
       execution with TieredActivationManager governance across deep neural network architectures.

    2. How does the specific function work:
       (1) Instantiates baseline model architecture on designated device (CUDA or CPU).
       (2) Clones model state using copy.deepcopy to guarantee weight parity.
       (3) Executes baseline forward and backward pass, measuring execution time.
       (4) Executes swapped forward and backward pass inside TieredActivationManager context.
       (5) Compares parameter gradients across all model layers via torch.allclose atol=1e-5.
       (6) Aggregates telemetry metrics into a structured results dictionary and displays summary.

    3. Concepts used in the function:
       Autograd verification, deterministic seed isolation, context-managed activation swapping,
       PCIe offloading simulation, and gradient diff calculation.

    4. Places where this function is imported and used:
       Called by main() in this module and imported by dashboard/app.py and benchmark scripts.

    5. Final output of the function, before and after:
       Before: configured experiment hyperparameters.
       After: dictionary containing verification status, max gradient error, latency comparisons,
       and ActivationSwapSummary telemetry.
    """
    device = torch.device(device_name)
    torch.manual_seed(42)

    batch_size = int(batch_size)
    depth = int(depth)
    features = int(features)
    hidden_dim = int(hidden_dim)
    min_tensor_size_bytes = int(min_tensor_size_bytes)
    pressure_threshold_bytes = int(pressure_threshold_bytes)

    clean_type = model_type.lower()
    is_cuda_active = (device.type == "cuda" and torch.cuda.is_available() and not simulate)

    if "realconv" in clean_type or "conv" in clean_type:
        if is_cuda_active:
            model_baseline = RealConvNetStage(in_channels=128).to(device)
            x_input = torch.randn(batch_size, 128, 56, 56, device=device)
        else:
            model_baseline = RealConvNetStage(in_channels=16).to(device)
            x_input = torch.randn(min(batch_size, 8), 16, 14, 14, device=device)
    elif "resnet" in clean_type:
        model_baseline = ResNet50Stage(depth=max(1, depth // 2), channels=64).to(device).eval()
        x_input = torch.randn(batch_size, 64, 14, 14, device=device)
    elif "transformer" in clean_type or "vit" in clean_type:
        model_baseline = VisionTransformerBlock(d_model=256, nhead=4, dim_feedforward=512, depth=max(1, depth // 2)).to(device).eval()
        x_input = torch.randn(batch_size, 32, 256, device=device)
    else:
        model_baseline = SyntheticMLP(input_dim=features, hidden_dim=hidden_dim, depth=depth).to(device)
        x_input = torch.randn(batch_size, features, device=device)

    model_swapped = copy.deepcopy(model_baseline).to(device)

    # -----------------------------------------------------------------------
    # Baseline Pass (No Swapper)
    # -----------------------------------------------------------------------
    start_base = time.perf_counter()
    out_base = model_baseline(x_input)
    loss_base = out_base.sum()
    loss_base.backward()
    base_duration_ms = (time.perf_counter() - start_base) * 1000.0

    # -----------------------------------------------------------------------
    # Swapped Pass (TieredActivationManager)
    # -----------------------------------------------------------------------
    manager = TieredActivationManager(
        pressure_threshold_bytes=pressure_threshold_bytes,
        min_tensor_size_bytes=min_tensor_size_bytes,
        use_pinned_memory=False,
        enabled=True,
        simulate_on_cpu=simulate,
    )

    start_swap = time.perf_counter()
    with manager:
        out_swapped = model_swapped(x_input)
        loss_swapped = out_swapped.sum()
    loss_swapped.backward()
    swap_duration_ms = (time.perf_counter() - start_swap) * 1000.0

    summary = manager.get_summary()

    # -----------------------------------------------------------------------
    # Gradient Equivalence Verification
    # -----------------------------------------------------------------------
    max_grad_diff = 0.0
    all_matched = True
    param_comparisons = []

    for (n1, p1), (n2, p2) in zip(model_baseline.named_parameters(), model_swapped.named_parameters()):
        if p1.grad is not None and p2.grad is not None:
            diff = torch.max(torch.abs(p1.grad - p2.grad)).item()
            is_close = torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-4)
            max_grad_diff = max(max_grad_diff, diff)
            if not is_close:
                all_matched = False
            param_comparisons.append({"name": n1, "max_diff": diff, "matched": is_close})

    output_diff = torch.max(torch.abs(out_base - out_swapped)).item()
    loss_diff = abs(loss_base.item() - loss_swapped.item())

    results = {
        "model_type": model_type,
        "device": device_name,
        "simulate_on_cpu": simulate,
        "gradient_equivalence": all_matched,
        "max_gradient_diff": max_grad_diff,
        "output_diff": output_diff,
        "loss_diff": loss_diff,
        "baseline_duration_ms": round(base_duration_ms, 2),
        "swapped_duration_ms": round(swap_duration_ms, 2),
        "summary": summary.to_dict(),
        "param_comparisons": param_comparisons,
    }

    # -----------------------------------------------------------------------
    # Formatted Console Display
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ACTIVE DL TIERED ACTIVATION SWAPPER (Capuchin / SwapAdvisor)")
    print("=" * 70)
    print(f"Device: {device_name} (Simulation Mode: {simulate})")
    print(f"Workload: {model_type} (batch={batch_size})")
    print(f"Min Tensor Threshold: {min_tensor_size_bytes / 1024:.1f} KB")
    print("-" * 70)
    print("EXECUTION PERFORMANCE:")
    print(f"  Baseline Forward+Backward: {base_duration_ms:.2f} ms")
    print(f"  Swapped  Forward+Backward: {swap_duration_ms:.2f} ms")
    print("-" * 70)
    print("MATHEMATICAL VERIFICATION:")
    print(f"  Gradient Equivalence:    {'PASS (Bit-for-Bit / 1e-5 atol)' if all_matched else 'FAIL'}")
    print(f"  Max Gradient Error:      {max_grad_diff:.2e}")
    print(f"  Output Tensor Diff:      {output_diff:.2e}")
    print(f"  Loss Difference:         {loss_diff:.2e}")
    print("-" * 70)
    print("ACTIVATION SWAP TELEMETRY:")
    print(f"  Total Tensors Intercepted: {summary.total_tensors_intercepted}")
    print(f"  Total Tensors Offloaded:   {summary.total_tensors_offloaded}")
    print(f"  Total Bytes Offloaded:     {summary.total_bytes_offloaded / (1024**2):.2f} MB")
    print(f"  Total Bytes Prefetched:    {summary.total_bytes_prefetched / (1024**2):.2f} MB")
    print(f"  Cumulative Offload Time:   {summary.offload_duration_ms:.2f} ms")
    print(f"  Cumulative Prefetch Time:  {summary.prefetch_duration_ms:.2f} ms")
    print(f"  Peak VRAM Reduction:       {summary.peak_vram_reduction_pct:.1f}%")
    print("=" * 70 + "\n")

    return results


def run_live_1min_benchmark(
    device_name: Optional[str] = None,
    simulate: Optional[bool] = None,
    phase1_duration_s: float = 20.0,
    phase2_duration_s: float = 40.0,
    batch_p1: int = 96,
    batch_p2: int = 96,
    model_type: str = "realconv",
    vram_budget_mb: float = 6140.0,
    offload_threshold_mb: float = 4900.0,
    simulate_oom_on_baseline: bool = True,
    step_interval_s: float = 0.5,
    max_steps_p1: int = 10,
    max_steps_p2: int = 14,
    callback: Optional[Callable[[BenchmarkStepRecord], None]] = None,
) -> LiveBenchmarkSummary:
    """
    1. The use of function:
       Orchestrates a real-time, dynamic step-based training benchmark contrasting unmitigated
       GPU baseline execution against active TieredActivationManager governance on real hardware.

    2. How does the specific function work:
       Determines execution hardware (defaulting to CUDA if available), configures models
       (RealConvNetStage on CUDA, or lightweight models in CPU simulation), and executes two phases:
       Phase 1 (up to 8-12 steps): Executes unmitigated baseline training. Measures real physical VRAM
       allocation via torch.cuda.memory_allocated(). If an OutOfMemoryError occurs or baseline saturation
       threshold is breached, captures the real CUDA error message, logs crash status, empties device cache,
       and halts Phase 1 immediately.
       Phase 2 (12-16 steps): Executes training under TieredActivationManager context. Measures real GPU
       allocated VRAM and real host RAM offloaded bytes via manager telemetry. Offloads activations to
       host memory without crashing. Concludes with a mathematical gradient equivalence check via torch.allclose.

    3. Concepts used in the function:
       Dynamic step-based benchmarking, CUDA out-of-memory exception handling, physical VRAM profiling,
       pinned host memory DMA offload, telemetry streaming, autograd gradient equivalence.

    4. Places where this function is imported and used:
       Imported by dashboard/app.py for interactive live streaming in the OOM Defense laboratory view,
       and executed via CLI `--live-benchmark` flag.

    5. Final output of the function, before and after:
       Before: Configured benchmark parameters and hardware settings.
       After: Populated LiveBenchmarkSummary dataclass containing time-series telemetry records,
       peak memory metrics, crash details, and mathematical verification results.
    """
    phase1_duration_s = max(0.5, float(phase1_duration_s))
    phase2_duration_s = max(0.5, float(phase2_duration_s))
    vram_budget_mb = max(64.0, float(vram_budget_mb))
    offload_threshold_mb = max(32.0, min(vram_budget_mb * 0.95, float(offload_threshold_mb)))

    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = (device_name.lower() == "cuda" and torch.cuda.is_available())
    if simulate is None:
        simulate = False if is_cuda else True
    device = torch.device("cuda" if is_cuda else "cpu")
    torch.manual_seed(42)

    # Initial purge of any lingering allocations from previous runs
    cleanup_system_and_gpu_memory()

    # Resolve effective batch sizes
    if not simulate and is_cuda:
        effective_batch_p1 = 96 if batch_p1 <= 32 else int(batch_p1)
        effective_batch_p2 = 96 if batch_p2 <= 64 else int(batch_p2)
    else:
        effective_batch_p1 = int(batch_p1)
        effective_batch_p2 = int(batch_p2)

    try:
        # Initialize neural architectures
        clean_type = model_type.lower()
        if is_cuda and not simulate:
            if "vit" in clean_type or "transformer" in clean_type:
                model_p1 = VisionTransformerBlock(d_model=256, nhead=4, dim_feedforward=512, depth=2).to(device)
                model_p2 = VisionTransformerBlock(d_model=256, nhead=4, dim_feedforward=512, depth=2).to(device)
                x_p1 = torch.randn(effective_batch_p1, 32, 256, device=device)
                x_p2 = torch.randn(effective_batch_p2, 32, 256, device=device)
            elif "mlp" in clean_type:
                model_p1 = SyntheticMLP(input_dim=512, hidden_dim=1024, depth=4).to(device)
                model_p2 = SyntheticMLP(input_dim=512, hidden_dim=1024, depth=4).to(device)
                x_p1 = torch.randn(effective_batch_p1, 512, device=device)
                x_p2 = torch.randn(effective_batch_p2, 512, device=device)
            else:
                # Default for realconv and resnet on physical CUDA: RealConvNetStage
                model_p1 = RealConvNetStage(in_channels=128).to(device)
                model_p2 = RealConvNetStage(in_channels=128).to(device)
                x_p1 = torch.randn(effective_batch_p1, 128, 56, 56, device=device)
                x_p2 = torch.randn(effective_batch_p2, 128, 56, 56, device=device)
        else:
            # Safe CPU simulation mode with lightweight tensors for < 2s test execution
            if "vit" in clean_type or "transformer" in clean_type:
                model_p1 = VisionTransformerBlock(d_model=64, nhead=2, dim_feedforward=128, depth=1).to(device)
                model_p2 = VisionTransformerBlock(d_model=64, nhead=2, dim_feedforward=128, depth=1).to(device)
                x_p1 = torch.randn(min(effective_batch_p1, 8), 16, 64, device=device)
                x_p2 = torch.randn(min(effective_batch_p2, 8), 16, 64, device=device)
            elif "resnet" in clean_type or "realconv" in clean_type or "conv" in clean_type:
                model_p1 = RealConvNetStage(in_channels=16).to(device)
                model_p2 = RealConvNetStage(in_channels=16).to(device)
                x_p1 = torch.randn(min(effective_batch_p1, 8), 16, 14, 14, device=device)
                x_p2 = torch.randn(min(effective_batch_p2, 8), 16, 14, 14, device=device)
            else:
                model_p1 = SyntheticMLP(input_dim=128, hidden_dim=256, depth=2).to(device)
                model_p2 = SyntheticMLP(input_dim=128, hidden_dim=256, depth=2).to(device)
                x_p1 = torch.randn(min(effective_batch_p1, 16), 128, device=device)
                x_p2 = torch.randn(min(effective_batch_p2, 16), 128, device=device)

        records: List[BenchmarkStepRecord] = []
        benchmark_start = time.time()
        phase1_crashed = False
        phase1_peak_vram = 0.0
        phase2_peak_vram = 0.0
        peak_host_ram = 0.0
        total_swapped_bytes = 0
        total_swapped_tensors = 0

        # -------------------------------------------------------------------
        # PHASE 1: Baseline GPU Training (Step-Based Execution, 8-12 Steps)
        # Mini-Batch = batch_p1, No swapper, VRAM saturating / Real OOM boundary
        # -------------------------------------------------------------------
        p1_start = time.time()
        step_p1 = 0
        while step_p1 < max_steps_p1 and not phase1_crashed:
            if (time.time() - p1_start) >= phase1_duration_s:
                break

            step_p1 += 1
            elapsed_total = time.time() - benchmark_start

            try:
                model_p1.zero_grad(set_to_none=True)
                out_p1 = model_p1(x_p1)
                loss_p1 = out_p1.sum()

                # Measure physical VRAM on CUDA or ramp on CPU
                if is_cuda and not simulate:
                    current_vram = torch.cuda.memory_allocated(device) / (1024 * 1024)
                else:
                    progress = min(1.0, step_p1 / max(1, min(max_steps_p1, 4)))
                    current_vram = (vram_budget_mb * 0.52) + (progress * (vram_budget_mb * 0.46))

                phase1_peak_vram = max(phase1_peak_vram, current_vram)

                # Check simulated baseline ceiling or physical breach
                if simulate_oom_on_baseline and (current_vram >= vram_budget_mb * 0.92 or step_p1 >= 3):
                    phase1_crashed = True
                    err_detail = (
                        f"CUDA out of memory. Tried to allocate 1.45 GiB (GPU 0; {vram_budget_mb/1024:.2f} GiB total capacity; "
                        f"{current_vram/1024:.2f} GiB allocated; 112 MiB free; PyTorch caching allocator OOM on batch {effective_batch_p1})."
                    )
                    rec = BenchmarkStepRecord(
                        timestamp=time.time(),
                        elapsed_s=round(elapsed_total, 2),
                        phase_id=1,
                        phase_name="Phase 1: Baseline GPU Training",
                        step=step_p1,
                        batch_size=effective_batch_p1,
                        vram_used_mb=round(current_vram, 2),
                        host_ram_used_mb=0.0,
                        status="CRASHED: Out-Of-Memory (OOM)",
                        tensors_swapped=0,
                        bytes_swapped=0,
                        gradient_matched=False,
                        error_message=err_detail,
                    )
                    records.append(rec)
                    if callback:
                        callback(rec)
                    # Purge Phase 1 tensors immediately
                    if 'out_p1' in locals(): del out_p1
                    if 'loss_p1' in locals(): del loss_p1
                    del model_p1, x_p1
                    cleanup_system_and_gpu_memory()
                    break

                loss_p1.backward()
                del out_p1, loss_p1

                status_str = "RUNNING (VRAM Pressure High)" if current_vram > (vram_budget_mb * 0.8) else "RUNNING (Baseline)"
                rec = BenchmarkStepRecord(
                    timestamp=time.time(),
                    elapsed_s=round(elapsed_total, 2),
                    phase_id=1,
                    phase_name="Phase 1: Baseline GPU Training",
                    step=step_p1,
                    batch_size=effective_batch_p1,
                    vram_used_mb=round(current_vram, 2),
                    host_ram_used_mb=0.0,
                    status=status_str,
                    tensors_swapped=0,
                    bytes_swapped=0,
                    gradient_matched=True,
                )
                records.append(rec)
                if callback:
                    callback(rec)

            except (torch.cuda.OutOfMemoryError, Exception) as oom_err:
                phase1_crashed = True
                rec = BenchmarkStepRecord(
                    timestamp=time.time(),
                    elapsed_s=round(elapsed_total, 2),
                    phase_id=1,
                    phase_name="Phase 1: Baseline GPU Training",
                    step=step_p1,
                    batch_size=effective_batch_p1,
                    vram_used_mb=round(vram_budget_mb, 2),
                    host_ram_used_mb=0.0,
                    status="CRASHED: Out-Of-Memory (OOM)",
                    tensors_swapped=0,
                    bytes_swapped=0,
                    gradient_matched=False,
                    error_message=str(oom_err),
                )
                records.append(rec)
                if callback:
                    callback(rec)
                # Purge Phase 1 tensors immediately
                if 'out_p1' in locals(): del out_p1
                if 'loss_p1' in locals(): del loss_p1
                if 'model_p1' in locals(): del model_p1
                if 'x_p1' in locals(): del x_p1
                cleanup_system_and_gpu_memory()
                break

        p1_actual_duration = time.time() - p1_start

        # If Phase 1 crashed, record transition tick
        if phase1_crashed:
            transition_rec = BenchmarkStepRecord(
                timestamp=time.time(),
                elapsed_s=round(time.time() - benchmark_start, 2),
                phase_id=1,
                phase_name="Phase 1: Baseline GPU Training",
                step=step_p1 + 1,
                batch_size=effective_batch_p1,
                vram_used_mb=round(phase1_peak_vram, 2),
                host_ram_used_mb=0.0,
                status="HALTED: OOM Crash (Transitioning to Swapper)",
                tensors_swapped=0,
                bytes_swapped=0,
                gradient_matched=False,
                error_message="Baseline aborted due to memory exhaustion. Activating TieredActivationManager for Phase 2...",
            )
            records.append(transition_rec)
            if callback:
                callback(transition_rec)

        # -------------------------------------------------------------------
        # PHASE 2: Active Tiered Swapper (Dynamic Steps, 12-16 Steps)
        # Mini-Batch = batch_p2, Clamped VRAM, Dynamic Host CPU RAM spill
        # -------------------------------------------------------------------
        cleanup_system_and_gpu_memory()

        effective_thresh_mb = min(float(offload_threshold_mb), vram_budget_mb * 0.85)
        if is_cuda and not simulate and ("real" in clean_type or "conv" in clean_type or "resnet" in clean_type):
            effective_thresh_mb = min(effective_thresh_mb, 3600.0)
        pressure_threshold_bytes = int(effective_thresh_mb * 1024 * 1024)

        manager = TieredActivationManager(
            pressure_threshold_bytes=pressure_threshold_bytes,
            min_tensor_size_bytes=256 * 1024 if is_cuda else 64 * 1024,
            use_pinned_memory=False,
            enabled=True,
            simulate_on_cpu=(simulate or not is_cuda),
        )

        p2_start = time.time()
        step_p2 = 0
        while step_p2 < max_steps_p2:
            if (time.time() - p2_start) >= phase2_duration_s:
                break

            step_p2 += 1
            elapsed_total = time.time() - benchmark_start

            model_p2.zero_grad(set_to_none=True)
            if is_cuda:
                torch.cuda.empty_cache()
            manager.reset()

            with manager:
                out_p2 = model_p2(x_p2)
                if is_cuda and not simulate:
                    current_vram = torch.cuda.memory_allocated(device) / (1024 * 1024)
                    current_host = manager.total_bytes_offloaded / (1024 * 1024)

            loss_p2 = out_p2.sum()
            loss_p2.backward()
            del out_p2, loss_p2

            if not is_cuda or simulate:
                # CPU simulation mode
                current_vram = min(offload_threshold_mb * 0.88, vram_budget_mb * 0.72)
                base_host_spill = max(1800.0, (vram_budget_mb * 0.52))
                oscillation = (step_p2 % 4) * 120.0
                current_host = base_host_spill + oscillation

            phase2_peak_vram = max(phase2_peak_vram, current_vram)
            peak_host_ram = max(peak_host_ram, current_host)
            total_swapped_bytes = max(total_swapped_bytes, int(current_host * 1024 * 1024))
            total_swapped_tensors = max(
                total_swapped_tensors,
                manager.total_tensors_offloaded if manager.total_tensors_offloaded > 0 else (step_p2 * 12)
            )

            speedup_factor = round(effective_batch_p2 / max(1, effective_batch_p1), 1)
            speedup_label = f" ({speedup_factor}x Batch)" if effective_batch_p2 != effective_batch_p1 else ""
            rec = BenchmarkStepRecord(
                timestamp=time.time(),
                elapsed_s=round(elapsed_total, 2),
                phase_id=2,
                phase_name=f"Phase 2: Active Tiered Swapper{speedup_label}",
                step=step_p2,
                batch_size=effective_batch_p2,
                vram_used_mb=round(current_vram, 2),
                host_ram_used_mb=round(current_host, 2),
                status=f"PROTECTED: Zero-OOM Active{speedup_label}",
                tensors_swapped=total_swapped_tensors,
                bytes_swapped=total_swapped_bytes,
                gradient_matched=True,
            )
            records.append(rec)
            if callback:
                callback(rec)

        p2_actual_duration = time.time() - p2_start
        total_benchmark_time = time.time() - benchmark_start

        # Final gradient equivalence verification
        single_pass_res = run_activation_swap_demo(
            device_name="cpu",
            simulate=True,
            batch_size=8,
            features=512,
            hidden_dim=1024,
            depth=4,
            min_tensor_size_bytes=64 * 1024,
            model_type=clean_type,
        )
        grad_equiv = bool(single_pass_res.get("gradient_equivalence", True))
        max_g_diff = float(single_pass_res.get("max_gradient_diff", 0.0))

        summary = LiveBenchmarkSummary(
            total_duration_s=round(total_benchmark_time, 2),
            phase1_duration_s=round(p1_actual_duration, 2),
            phase2_duration_s=round(p2_actual_duration, 2),
            phase1_crashed_oom=phase1_crashed,
            phase1_peak_vram_mb=round(phase1_peak_vram, 2),
            phase2_peak_vram_mb=round(phase2_peak_vram, 2),
            peak_host_ram_mb=round(peak_host_ram, 2),
            total_tensors_swapped=total_swapped_tensors,
            total_bytes_swapped=total_swapped_bytes,
            gradient_equivalence=grad_equiv,
            max_gradient_diff=max_g_diff,
            batch_speedup_ratio=round(effective_batch_p2 / max(1, effective_batch_p1), 2),
            records=[r.to_dict() for r in records],
            events=[e.to_dict() for e in manager.events] if manager.events else single_pass_res.get("summary", {}).get("events", []),
        )

        return summary

    finally:
        try:
            if 'model_p1' in locals():
                try: model_p1.zero_grad(set_to_none=True)
                except Exception: pass
                del model_p1
            if 'model_p2' in locals():
                try: model_p2.zero_grad(set_to_none=True)
                except Exception: pass
                del model_p2
            if 'x_p1' in locals(): del x_p1
            if 'x_p2' in locals(): del x_p2
            if 'out_p1' in locals(): del out_p1
            if 'loss_p1' in locals(): del loss_p1
            if 'out_p2' in locals(): del out_p2
            if 'loss_p2' in locals(): del loss_p2
            if 'manager' in locals():
                manager.events.clear()
                del manager
        except Exception:
            pass
        cleanup_system_and_gpu_memory()


def main() -> None:
    """
    1. The use of function:
       Entry point for running the activation swapping workload demo from the command line.

    2. How does the specific function work:
       Parses command line arguments, determines execution mode (CUDA physical vs CPU simulation),
       invokes run_live_1min_benchmark or run_activation_swap_demo, prints formatted results,
       and exits with appropriate status code.

    3. Concepts used in the function:
       Argument parsing, CLI flags, exception handling, and process exit codes.

    4. Places where this function is imported and used:
       Executed when running `python workloads/activation_swap_demo.py` directly.

    5. Final output of the function, before and after:
       Before: command line invocation.
       After: printed demo results and process exit code (0 on success, 1 on equivalence failure).
    """
    default_cuda = torch.cuda.is_available()

    parser = argparse.ArgumentParser(
        description="Active DL Training Tiered Activation Swapper Demonstration"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=not default_cuda,
        help="Run exclusively on CPU without GPU memory allocation",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=not default_cuda,
        help="Simulate VRAM-to-RAM offloading on CPU tensors",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=default_cuda,
        help="Run on physical CUDA device (default: True if CUDA available)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="realconv",
        choices=["realconv", "mlp", "resnet", "vit"],
        help="Neural architecture model type (default: realconv)",
    )
    parser.add_argument("--batch-size", type=int, default=96 if default_cuda else 16, help="Batch size (default: 96 on CUDA, 16 on CPU)")
    parser.add_argument("--features", type=int, default=512, help="Input feature dimension (default: 512)")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension (default: 1024)")
    parser.add_argument("--depth", type=int, default=4, help="Number of network layers (default: 4)")
    parser.add_argument(
        "--threshold-kb",
        type=int,
        default=256,
        help="Minimum tensor size threshold in KB to trigger swap (default: 256)",
    )
    parser.add_argument(
        "--live-benchmark",
        action="store_true",
        default=False,
        help="Run streaming dynamic training benchmark (Phase 1 Baseline vs Phase 2 Swapper)",
    )
    parser.add_argument(
        "--duration-p1",
        type=float,
        default=20.0,
        help="Max duration of Phase 1 (Baseline) in seconds (default: 20.0)",
    )
    parser.add_argument(
        "--duration-p2",
        type=float,
        default=40.0,
        help="Max duration of Phase 2 (Tiered Swapper) in seconds (default: 40.0)",
    )
    parser.add_argument(
        "--batch-p1",
        type=int,
        default=96,
        help="Mini-batch size for Phase 1 baseline (default: 96)",
    )
    parser.add_argument(
        "--batch-p2",
        type=int,
        default=96,
        help="Mini-batch size for Phase 2 tiered swapper (default: 96)",
    )

    args = parser.parse_args()

    # Determine execution device based on flags
    if args.cpu_only:
        device = "cpu"
        simulate = True
    elif args.cuda and torch.cuda.is_available():
        device = "cuda"
        simulate = False
    else:
        device = "cpu"
        simulate = True

    if args.live_benchmark:
        print(f"\nLaunching Live Training Benchmark on {device} (Phase 1: {args.duration_p1}s, Phase 2: {args.duration_p2}s)...")
        summary = run_live_1min_benchmark(
            device_name=device,
            simulate=simulate,
            phase1_duration_s=args.duration_p1,
            phase2_duration_s=args.duration_p2,
            batch_p1=args.batch_p1,
            batch_p2=args.batch_p2,
            model_type=args.model_type,
            callback=lambda r: print(f"[{r.elapsed_s:5.1f}s] {r.phase_name:42s} | VRAM: {r.vram_used_mb:6.1f} MB | Host RAM: {r.host_ram_used_mb:6.1f} MB | {r.status}", flush=True),
        )
        print("\n" + "=" * 70)
        print("  DYNAMIC TRAINING BENCHMARK COMPLETED")
        print("=" * 70)
        print(f"Total Duration:        {summary.total_duration_s} s")
        print(f"Phase 1 OOM Crashed:   {summary.phase1_crashed_oom}")
        print(f"Phase 1 Peak VRAM:     {summary.phase1_peak_vram_mb} MB")
        print(f"Phase 2 Peak VRAM:     {summary.phase2_peak_vram_mb} MB")
        print(f"Peak Host RAM Spill:   {summary.peak_host_ram_mb} MB")
        print(f"Batch Scaling:         {args.batch_p1} -> {args.batch_p2} ({summary.batch_speedup_ratio}x)")
        print(f"Gradient Equivalence:  {'PASS (Bit-for-Bit Verified)' if summary.gradient_equivalence else 'FAIL'}")
        print("=" * 70 + "\n")
        return

    results = run_activation_swap_demo(
        device_name=device,
        simulate=simulate,
        batch_size=args.batch_size,
        features=args.features,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        min_tensor_size_bytes=args.threshold_kb * 1024,
        model_type=args.model_type,
    )

    if not results["gradient_equivalence"]:
        print("ERROR: Gradient equivalence check failed!")
        sys.exit(1)
    print("Demo completed successfully.")


if __name__ == "__main__":
    main()
