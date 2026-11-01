"""
File: controller/__init__.py

1. What this file does:
   Marks the controller package as a Python module and exposes the public APIs for
   both GPU admission control and the virtual memory / page replacement management subsystem.

2. Overall design:
   Standard Python package initialiser that re-exports admission control and virtual memory
   components, allowing callers to perform clean high-level imports:
       from controller import VirtualMemoryManager, AdmissionController

3. Elements in the file:
   - AdmissionController, AdmissionDecision (Admission control API)
   - PageEntry, PageAccessResult, VirtualMemoryMetrics (Virtual memory data models)
   - PageReplacementPolicy, FIFOPolicy, LRUPolicy, OptimalPolicy (Paging policies)
   - VirtualMemoryManager (Central virtual memory engine)

4. Final output of the file:
   Exported classes and type structures for the controller package.
"""

from controller.admission_control import AdmissionController, AdmissionDecision
from controller.virtual_memory import (
    FIFOPolicy,
    LRUPolicy,
    OptimalPolicy,
    PageAccessResult,
    PageEntry,
    PageReplacementPolicy,
    VirtualMemoryManager,
    VirtualMemoryMetrics,
)

__all__ = [
    "AdmissionController",
    "AdmissionDecision",
    "PageEntry",
    "PageAccessResult",
    "VirtualMemoryMetrics",
    "PageReplacementPolicy",
    "FIFOPolicy",
    "LRUPolicy",
    "OptimalPolicy",
    "VirtualMemoryManager",
]
