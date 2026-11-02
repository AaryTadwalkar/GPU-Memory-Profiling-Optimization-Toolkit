import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import gc
from workloads.activation_swap_demo import run_live_1min_benchmark, cleanup_system_and_gpu_memory

print("=== VERIFYING CLEANUP FUNCTION ===")
# 1. Clean up first
print("Running initial cleanup...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
print(f"Initial VRAM: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

# 2. Run benchmark in simulation mode first to verify no syntax errors
print("\nRunning benchmark in simulation mode...")
summary = run_live_1min_benchmark(
    device_name="cpu",
    simulate=True,
    batch_p1=16,
    batch_p2=32,
    model_type="realconv",
    vram_budget_mb=6140.0,
    offload_threshold_mb=4900.0,
    max_steps_p1=2,
    max_steps_p2=2,
)
print("Simulation complete! Duration:", summary.total_duration_s)
print("Phase 1 Peak VRAM:", summary.phase1_peak_vram_mb)
print("Phase 2 Peak VRAM:", summary.phase2_peak_vram_mb)
print("Peak Host RAM:", summary.peak_host_ram_mb)
