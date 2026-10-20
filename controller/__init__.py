"""
File: controller/__init__.py

1. What this file does:
   Marks the controller package as a Python module and exposes the public
   admission-control API.
2. Overall design:
   Standard Python package initialiser that re-exports AdmissionController and
   AdmissionDecision so callers can write:
       from controller import AdmissionController, AdmissionDecision
3. Elements in the file:
   - Package-level imports for AdmissionController and AdmissionDecision.
4. Final output of the file:
   Exported classes for the controller package.
"""

from controller.admission_control import AdmissionController, AdmissionDecision

__all__ = ["AdmissionController", "AdmissionDecision"]
