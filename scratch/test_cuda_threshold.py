import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import gc
from workloads.activation_swap_demo import RealConvNetStage
from optimizer.activation_swapper import TieredActivationManager

print("Testing full end-to-end benchmark steps with cleanup...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(f"Pre-run VRAM: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

# Phase 1: Baseline run
model_p1 = RealConvNetStage(in_channels=128).to(device)
x_p1 = torch.randn(96, 128, 56, 56, device=device)
out_p1 = model_p1(x_p1)
p1_vram = torch.cuda.memory_allocated() / (1024**2)
print(f"Phase 1 Baseline forward VRAM: {p1_vram:.2f} MB")

# Baseline abort / OOM simulation: Clean up Phase 1
del out_p1, model_p1, x_p1
gc.collect()
torch.cuda.empty_cache()
print(f"Post-Phase 1 cleanup VRAM: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

# Phase 2: Tiered Swapper at 3600 MB threshold
threshold_mb = 3600.0
model_p2 = RealConvNetStage(in_channels=128).to(device)
x_p2 = torch.randn(96, 128, 56, 56, device=device)
manager = TieredActivationManager(
    pressure_threshold_bytes=int(threshold_mb * 1024 * 1024),
    min_tensor_size_bytes=256 * 1024,
    use_pinned_memory=False,
    enabled=True,
    simulate_on_cpu=False,
)

for step in range(3):
    model_p2.zero_grad(set_to_none=True)
    manager.reset()
    with manager:
        out_p2 = model_p2(x_p2)
        vram_p2 = torch.cuda.memory_allocated() / (1024**2)
        host_p2 = manager.total_bytes_offloaded / (1024**2)
        print(f"Phase 2 Step {step+1}: VRAM = {vram_p2:.2f} MB, Host Spill = {host_p2:.2f} MB")
    loss_p2 = out_p2.sum()
    loss_p2.backward()
    del out_p2, loss_p2

# Final cleanup
del model_p2, x_p2, manager
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
print(f"Final Post-Run VRAM: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
