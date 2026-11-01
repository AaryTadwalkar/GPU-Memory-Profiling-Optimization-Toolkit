"""
File: workloads/activation_swap_demo.py

1. What this file does:
   Demonstrates and benchmarks deep learning training execution with and without the
   TieredActivationManager (Capuchin / SwapAdvisor paradigm). Compares forward-backward
   passes across multi-layer feedforward networks (MLP), ResNet-50 bottleneck stages,
   and Vision Transformer (ViT) blocks. Verifies exact mathematical gradient equivalence
   and outputs detailed memory telemetry including offloaded tensors, transfer latencies,
   and peak VRAM reduction.

2. Overall design:
   Defines parameterizable neural architectures (SyntheticMLP, ResNet50Stage, VisionTransformerBlock)
   initialized with deterministic weights. Executes two identical passes on the same input batch:
   (1) Baseline pass without activation swapping.
   (2) Swapped pass wrapped in `TieredActivationManager`.
   Validates that all parameter gradients match bit-for-bit or within floating-point tolerance
   (torch.allclose) and formats telemetry metrics into a structured report. By default, executes
   in safe CPU simulation mode requiring zero GPU memory allocation or hardware intervention.

3. Elements in the file:
   - SyntheticMLP: Multi-layer neural network architecture with configurable depth and width.
   - ResNetBottleneckBlock: 3-layer residual bottleneck convolutional block with shortcut.
   - ResNet50Stage: Multi-block sequence of ResNet bottleneck modules.
   - VisionTransformerBlock: Multi-layer ViT encoder block with multi-head self-attention.
   - run_activation_swap_demo: Core orchestration function comparing baseline vs swapped execution.
   - main: Command-line entry point parsing flags and executing the demonstration safely.

4. Final output of the file:
   Outputs human-readable CLI telemetry tables and returns structured comparison dictionaries
   validating gradient correctness and profiling activation offload/prefetch statistics.
"""

import argparse
import copy
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizer.activation_swapper import TieredActivationManager, ActivationSwapSummary


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
           Constructs multi-head self-attention transformer encoder layers.

        2. How does the specific function work:
           Initializes nn.TransformerEncoderLayer with batch_first=True, zero dropout (for deterministic
           gradient verification), and wraps multiple layers into an nn.TransformerEncoder.

        3. Concepts used in the function:
           Transformer self-attention architecture and multi-head projection sizing.

        4. Places where this function is imported and used:
           Instantiated in run_activation_swap_demo when model_type is set to 'vit'.

        5. Final output of the function, before and after:
           Before: uninitialized transformer module.
           After: fully instantiated transformer encoder ready for sequence propagation.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. The use of function:
           Propagates sequence token embedding tensors through transformer self-attention layers.

        2. How does the specific function work:
           Invokes self.encoder, saving multi-head attention scores and intermediate linear projections
           on the autograd computational graph.

        3. Concepts used in the function:
           Autoregressive/bidirectional self-attention and autograd graph recording.

        4. Places where this function is imported and used:
           Called in run_activation_swap_demo during baseline and swapped benchmark execution.

        5. Final output of the function, before and after:
           Before: 3D sequence embedding tensor (B, S, D).
           After: 3D context-attended representation tensor.
        """
        return self.encoder(x)


def run_activation_swap_demo(
    device_name: str = "cpu",
    simulate: bool = True,
    batch_size: int = 64,
    features: int = 512,
    hidden_dim: int = 1024,
    depth: int = 6,
    min_tensor_size_bytes: int = 64 * 1024,
    model_type: str = "mlp",
    pressure_threshold_bytes: int = 0,
) -> Dict[str, Any]:
    """
    1. The use of function:
       Orchestrates comparative execution of deep learning training with and without
       TieredActivationManager, validating mathematical equivalence and profiling swap overhead.

    2. How does the specific function work:
       (1) Instantiates two identical copies of target neural architecture (SyntheticMLP, ResNet50Stage,
           or VisionTransformerBlock) with deterministic seeds.
       (2) Generates a synthetic input batch.
       (3) Executes baseline forward and backward pass, recording runtime and gradients.
       (4) Executes swapped forward and backward pass inside TieredActivationManager context.
       (5) Compares parameter gradients across all model layers via torch.allclose.
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
    if "resnet" in clean_type:
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
        use_pinned_memory=False if device.type == "cpu" else True,
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
    print(f"Workload: SyntheticMLP (depth={depth}, hidden={hidden_dim}, batch={batch_size})")
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


def main() -> None:
    """
    1. The use of function:
       Entry point for running the activation swapping workload demo from the command line.

    2. How does the specific function work:
       Parses command line arguments, enforces safe CPU defaults, calls run_activation_swap_demo,
       and exits with appropriate status code.

    3. Concepts used in the function:
       Argument parsing, CLI flags, exception handling, and process exit codes.

    4. Places where this function is imported and used:
       Executed when running `python workloads/activation_swap_demo.py` directly.

    5. Final output of the function, before and after:
       Before: command line invocation.
       After: printed demo results and process exit code (0 on success, 1 on equivalence failure).
    """
    parser = argparse.ArgumentParser(
        description="Active DL Training Tiered Activation Swapper Demonstration"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=True,
        help="Run exclusively on CPU without GPU memory allocation (default: True)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Simulate VRAM-to-RAM offloading on CPU tensors (default: True)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Run on physical CUDA device (MANUAL USER RUN ONLY)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="mlp",
        choices=["mlp", "resnet", "vit"],
        help="Neural architecture model type (default: mlp)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--features", type=int, default=512, help="Input feature dimension (default: 512)")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension (default: 1024)")
    parser.add_argument("--depth", type=int, default=6, help="Number of network layers (default: 6)")
    parser.add_argument(
        "--threshold-kb",
        type=int,
        default=64,
        help="Minimum tensor size threshold in KB to trigger swap (default: 64)",
    )

    args = parser.parse_args()

    # Determine execution device based on flags
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    simulate = True if (device == "cpu" or args.simulate) else False

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
