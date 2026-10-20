"""
File: memory_pressure.py

1. What this file does:
   Statically analyzes a GPU memory event CSV to compute memory pressure,
   headroom, and a heuristic OOM-risk classification.
2. Overall design:
   A standalone command-line tool and library that processes a timeseries of
   MemoryEvent rows. It computes derived metrics (headroom, growth rate, 
   peak-to-capacity ratio) and applies a transparent heuristic rule to label 
   OOM risk as Low, Moderate, or High.
3. Elements in the file:
   - analyze_pressure: Core logic that iterates over CSV rows.
   - evaluate_oom_risk_heuristic: Function applying the transparent risk rules.
4. Final output of the file:
   Prints a timeline of memory pressure and risk estimates to the console.
"""

import argparse
import csv
import os
from typing import List, Dict, Any

# Transparent, configurable risk thresholds
# These are heuristics, not physical guarantees.
RISK_THRESHOLDS = {
    "high_headroom_ratio": 0.10,     # If headroom is < 10% of budget, High risk
    "moderate_headroom_ratio": 0.25, # If headroom is < 25% of budget, Moderate risk
    "high_growth_rate_bps": 50 * 1024 * 1024, # If growing > 50 MB/s, bump risk
}

def evaluate_oom_risk_heuristic(
    budget_bytes: int,
    allocated_bytes: int,
    growth_rate_bps: float
) -> str:
    """
    1. The use of function:
       Classifies OOM risk into Low, Moderate, or High.
    2. How does the specific function work:
       Evaluates headroom ratio and growth rate against transparent thresholds.
       NOTE: This is an ESTIMATE/HEURISTIC, not a prediction or guarantee.
    """
    if budget_bytes <= 0:
        return "Unknown"
        
    headroom = budget_bytes - allocated_bytes
    headroom_ratio = headroom / budget_bytes

    # Start with base risk based on headroom
    risk = "Low"
    if headroom_ratio < RISK_THRESHOLDS["high_headroom_ratio"]:
        risk = "High"
    elif headroom_ratio < RISK_THRESHOLDS["moderate_headroom_ratio"]:
        risk = "Moderate"

    # Escalate risk if memory is growing very rapidly
    if risk != "High" and growth_rate_bps > RISK_THRESHOLDS["high_growth_rate_bps"]:
        if risk == "Low":
            risk = "Moderate"
        elif risk == "Moderate":
            risk = "High"
            
    # Absolute overrun check
    if allocated_bytes >= budget_bytes:
        risk = "High"
        
    return risk

def analyze_pressure(csv_path: str, budget_bytes: int) -> List[Dict[str, Any]]:
    """
    Reads a CSV of memory events and computes pressure metrics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    results = []
    
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        
        prev_time = None
        prev_allocated = None
        
        for idx, row in enumerate(reader):
            try:
                timestamp = float(row.get("timestamp", 0.0))
            except ValueError:
                continue

            # 1. Distinguish missing/empty from allocated_bytes == 0
            alloc_str = row.get("allocated_bytes")
            if alloc_str is None or alloc_str.strip() == "":
                # Missing/empty = unavailable measurement (e.g. nvml-only row).
                # We skip PyTorch pressure analysis for this specific row.
                continue
                
            try:
                allocated = int(alloc_str)
                peak_str = row.get("peak_memory")
                peak = int(peak_str) if peak_str and peak_str.strip() else 0
            except ValueError:
                continue
                
            # 2. Configured-budget headroom (budget - allocated)
            budget_headroom = budget_bytes - allocated
            peak_to_capacity = peak / budget_bytes if budget_bytes > 0 else 0.0
            
            # 3. Memory-growth calculation
            # Formula: (current_allocated - prev_allocated) / time_delta
            # Represents the instantaneous change in allocated bytes per second between consecutive samples.
            growth_rate = 0.0
            if prev_time is not None and prev_allocated is not None:
                dt = timestamp - prev_time
                if dt > 0:
                    growth_rate = (allocated - prev_allocated) / dt
                    
            risk = evaluate_oom_risk_heuristic(budget_bytes, allocated, growth_rate)
            
            results.append({
                "index": idx,
                "timestamp": timestamp,
                "allocated_bytes": allocated,
                "peak_memory": peak,
                "budget_headroom": budget_headroom,
                "peak_to_capacity": peak_to_capacity,
                "growth_rate_bps": growth_rate,
                "oom_risk_estimate": risk
            })
            
            prev_time = timestamp
            prev_allocated = allocated
            
    return results

def print_pressure_report(results: List[Dict[str, Any]], budget_bytes: int):
    print(f"--- Memory Pressure & OOM Risk Analysis ---")
    print(f"Configured Workload Budget: {budget_bytes} bytes (~{budget_bytes // (1024*1024)} MiB)")
    print(f"WARNING: OOM risk is a heuristic ESTIMATE, not a guarantee or prediction.\n")
    
    if not results:
        print("No valid allocation data found to analyze.")
        return
        
    for r in results:
        t = r["timestamp"]
        alloc_mb = r["allocated_bytes"] / (1024*1024)
        head_mb = r["budget_headroom"] / (1024*1024)
        peak_ratio = r["peak_to_capacity"] * 100
        growth_mb_s = r["growth_rate_bps"] / (1024*1024)
        risk = r["oom_risk_estimate"]
        
        print(
            f"T={t:.2f} | Alloc: {alloc_mb:6.1f} MiB | Budget Headroom: {head_mb:6.1f} MiB | "
            f"Peak/Cap: {peak_ratio:5.1f}% | Growth: {growth_mb_s:6.1f} MiB/s | "
            f"ESTIMATED RISK: {risk}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze memory pressure from CSV.")
    parser.add_argument("csv_path", type=str, help="Path to torch_events.csv")
    # Default budget 6GB, documented as a workload budget, not assumed physical capacity
    parser.add_argument(
        "--budget", type=int, default=6*1024*1024*1024, 
        help="Configured memory budget for the workload in bytes (default 6 GiB)"
    )
    args = parser.parse_args()
    
    try:
        report_data = analyze_pressure(args.csv_path, args.budget)
        print_pressure_report(report_data, args.budget)
    except Exception as e:
        print(f"Analysis failed: {e}")
