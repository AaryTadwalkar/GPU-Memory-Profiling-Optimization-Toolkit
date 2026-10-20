"""
File: memory_fragmentation.py

1. What this file does:
   Generates a highly fragmented PyTorch memory state for Phase 3 analysis.
2. Overall design:
   Allocates a series of identically sized tensors to reserve a large memory chunk.
   Then deletes alternating tensors to punch "holes" (inactive blocks) in the
   reserved segment. Triggers a snapshot, clears cache, and triggers a second snapshot.
3. Elements in the file:
   - run_fragmentation_test: The main function performing the experiment.
4. Final output of the file:
   Modifies PyTorch allocator state and triggers two snapshot writes via torch_hooks.
"""

import torch
import time

def run_fragmentation_test():
    """
    1. The use of function:
       Creates a fragmented memory layout and captures snapshot data.
    2. How does the specific function work:
       Allocates 20 tensors of 10 MB each. Deletes every alternating tensor.
       Calls record_snapshot('fragmented'). Calls torch.cuda.empty_cache().
       Calls record_snapshot('post_empty_cache').
    3. Concepts used in the function:
       Memory fragmentation, PyTorch caching allocator, snapshotting.
    4. Places where this function is imported and used:
       Executed by the torch_hooks.py wrapper script.
    5. Final output of the function, before and after:
       Leaves a clean allocator state after executing the snapshots.
    """
    if 'record_snapshot' not in globals():
        raise RuntimeError("record_snapshot not found in globals. This workload must be run via collector/torch_hooks.py")

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping fragmentation test.")
        return

    print("Starting fragmentation test...")
    
    tensors = []
    size_mb = 10
    num_elements = (size_mb * 1024 * 1024) // 4
    
    # 1. Allocate a block of tensors (e.g. 20 tensors of 10 MB = 200 MB)
    for i in range(20):
        t = torch.randn(num_elements, device='cuda')
        tensors.append(t)
        time.sleep(0.01)
        
    print("Allocated 20 tensors (200 MB total requested). Poking holes...")
    
    # 2. Delete alternating tensors to create inactive blocks
    # We must explicitly keep the list but replace dropped ones with None 
    # to avoid shifting indices, or just build a new list with holes.
    kept_tensors = []
    for i, t in enumerate(tensors):
        if i % 2 == 0:
            kept_tensors.append(t)
        else:
            del t  # local ref
            
    tensors = kept_tensors
    
    time.sleep(0.5)  # Wait for polling to catch up if needed
    
    # 3. Snapshot the highly fragmented state
    print("Recording 'fragmented' snapshot...")
    record_snapshot("fragmented")
    
    # 4. Clear the cache
    print("Calling torch.cuda.empty_cache()...")
    torch.cuda.empty_cache()
    time.sleep(0.5)
    
    # 5. Snapshot the cleared state
    print("Recording 'post_empty_cache' snapshot...")
    record_snapshot("post_empty_cache")
    
    print("Fragmentation test completed.")

if __name__ == "__main__":
    run_fragmentation_test()
