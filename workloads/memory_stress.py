"""
File: memory_stress.py

1. What this file does:
   Generates a rapid sequence of GPU memory allocations and deallocations.
2. Overall design:
   A simple script that allocates PyTorch tensors of varying sizes and then
   frees them, forcing the PyTorch caching allocator to work and potentially
   creating fragmentation.
3. Elements in the file:
   - run_stress_test: The main function performing the allocations.
4. Final output of the file:
   No explicit return value, but modifies the state of the PyTorch CUDA allocator.
"""

import torch
import time

def run_stress_test():
    """
    1. The use of function:
       Stress tests the GPU memory allocator within a safe, strict budget.
    2. How does the specific function work:
       Deterministically loops, allocating tensors of predefined sizes (10-50MB).
       Tracks the total allocated memory in MB. If the running total exceeds a
       hard limit (400 MB), it deletes every other tensor in the list to cause
       memory fragmentation and drops the total budget, ensuring it never OOMs.
    3. Concepts used in the function:
       Safe memory budgeting, deterministic memory fragmentation.
    4. Places where this function is imported and used:
       Executed by the torch_hooks.py wrapper script.
    5. Final output of the function, before and after:
       Before: Empty GPU memory.
       After: Leaves a fragmented state in the allocator bounded to < 400MB.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping stress test.")
        return

    print("Starting deterministic memory stress test: Max budget ~400 MB")
    
    tensors = []
    sizes_mb = [15, 25, 35, 45, 50]  # Deterministic sizes
    max_budget_bytes = 400 * 1024 * 1024  # 400 MiB hard limit
    
    for i in range(100):
        # Clear potentially lingering references from previous loops
        t = None
        
        size_mb = sizes_mb[i % len(sizes_mb)]
        requested_bytes = size_mb * 1024 * 1024
        num_elements = requested_bytes // 4  # 4 bytes per float32
        
        # Source of truth: ask the PyTorch allocator directly
        current_allocated_bytes = torch.cuda.memory_stats().get("allocated_bytes.all.current", 0)
        
        # Enforce hard budget BEFORE allocation using a loop
        while current_allocated_bytes + requested_bytes > max_budget_bytes:
            if not tensors:
                print(f"Iter {i}: All tensors freed, but still exceeded budget. Skipping allocation.")
                break
                
            print(f"Iter {i}: Budget check ({(current_allocated_bytes + requested_bytes) // 1048576} MiB > 400 MiB). Freeing oldest tensor.")
            tensors.pop(0)  # Remove reference
            
            # Re-read truth directly from PyTorch
            current_allocated_bytes = torch.cuda.memory_stats().get("allocated_bytes.all.current", 0)
            
        # Only allocate if the invariant is strictly satisfied
        if current_allocated_bytes + requested_bytes <= max_budget_bytes:
            t = torch.randn(num_elements, device='cuda')
            tensors.append(t)
            # Clear local reference immediately after appending
            t = None
            
        time.sleep(0.05)
        
    print("Stress test completed safely.")

if __name__ == "__main__":
    run_stress_test()
