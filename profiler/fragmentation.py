"""
File: fragmentation.py

1. What this file does:
   Reads memory event logs and parses PyTorch allocator snapshots to compute FSFI.
2. Overall design:
   A command-line tool that loads a CSV of torch events, filters for "snapshot"
   events, loads the associated JSON snapshot files, parses the allocator segment
   blocks, and computes the Free-Space Fragmentation Index (FSFI).
3. Elements in the file:
   - analyze_snapshots: Core logic reading CSV and computing metrics.
   - parse_snapshot_json: Logic to parse the raw PyTorch segment dump.
4. Final output of the file:
   Prints a summary table to the console comparing fragmented vs post-cache-clear FSFI.
"""

import argparse
import csv
import json
import os
from pathlib import Path

def parse_snapshot_json(filepath: str):
    """
    1. The use of function:
       Parses a PyTorch memory_snapshot() JSON dump.
    2. How does the specific function work:
       Iterates over all segments and all blocks within each segment. 
       Identifies blocks with state == 'inactive'. Sums their sizes for 
       total_free_memory and finds the max for largest_free_block.
    3. Concepts used in the function:
       PyTorch caching allocator structure analysis.
    4. Places where this function is imported and used:
       Called by `analyze_snapshots`.
    5. Final output of the function:
       Returns (total_free_memory_bytes, largest_free_block_bytes).
    """
    with open(filepath, "r") as f:
        snapshot = json.load(f)
        
    total_free = 0
    largest_free = 0
    
    # Handle both PyTorch dict-based snapshots (snapshot["segments"]) and older list-based ones
    segments = snapshot.get("segments", []) if isinstance(snapshot, dict) else snapshot
    
    for segment in segments:
        for block in segment.get("blocks", []):
            if block.get("state") == "inactive":
                size = block.get("size", 0)
                total_free += size
                if size > largest_free:
                    largest_free = size
                    
    return total_free, largest_free

def analyze_snapshots(csv_path: str):
    """
    1. The use of function:
       Coordinates the analysis of snapshot points from a CSV timeline.
    2. How does the specific function work:
       Opens the CSV, filters for rows where `snapshot_path` is populated,
       calls the JSON parser, computes FSFI, and prints the result.
    3. Concepts used in the function:
       Data correlation, metric computation (FSFI).
    4. Places where this function is imported and used:
       Executed as a CLI script.
    5. Final output of the function:
       Console printout of the analysis.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV at {csv_path}")
        return
        
    snapshots_analyzed = 0
    results = {}
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("event_type") == "snapshot" and row.get("snapshot_path"):
                filepath = row["snapshot_path"]
                if not os.path.exists(filepath):
                    print(f"Warning: snapshot file missing: {filepath}")
                    continue
                    
                total_free, largest_free = parse_snapshot_json(filepath)
                
                # FSFI Calculation (Project-Defined Metric)
                if total_free > 0:
                    fsfi = 1.0 - (largest_free / total_free)
                else:
                    fsfi = 0.0
                    
                label = "unknown"
                if "fragmented" in filepath:
                    label = "fragmented"
                elif "post_empty_cache" in filepath:
                    label = "post_empty_cache"
                
                results[label] = fsfi
                    
                print(f"--- Snapshot Analysis ---")
                print(f"File: {os.path.basename(filepath)}")
                print(f"Total Free Memory (Inactive): {total_free} bytes")
                print(f"Largest Free Block:           {largest_free} bytes")
                print(f"Project-Defined FSFI:         {fsfi:.4f}\n")
                
                snapshots_analyzed += 1
                
    if snapshots_analyzed == 0:
        print("No snapshots found in the CSV to analyze.")
    else:
        print("--- Experimental Conclusion ---")
        if "fragmented" in results and "post_empty_cache" in results:
            print(f"FSFI(fragmented) = {results['fragmented']:.4f}")
            print(f"FSFI(post_empty_cache) = {results['post_empty_cache']:.4f}")
            
            if results["fragmented"] > results["post_empty_cache"]:
                print("CRITERION MET: FSFI(fragmented) > FSFI(post_empty_cache)")
            else:
                print("CRITERION FAILED: FSFI did not decrease after empty_cache.")
        else:
            print("Could not find both 'fragmented' and 'post_empty_cache' snapshots for comparison.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze FSFI from memory snapshots.")
    parser.add_argument("csv_path", type=str, help="Path to torch_events.csv")
    args = parser.parse_args()
    
    analyze_snapshots(args.csv_path)
