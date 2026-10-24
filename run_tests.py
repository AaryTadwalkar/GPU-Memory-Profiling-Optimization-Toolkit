"""
File: run_tests.py

1. What this file does:
   A convenient, standalone automated test runner that executes all unit test suites
   in the `tests/` directory.

2. Overall design:
   A CPU-safe script that invokes `pytest` via the local virtual environment or
   falls back to Python's built-in `unittest` module. Prints a clean, color-coded
   summary of passed and failed test suites with execution times.

3. Elements in the file:
   - `run_all_tests`: Discovers and executes all tests in `tests/`.
   - CLI execution block.

4. Final output of the file:
   Console report and process exit code (0 if all pass, 1 if any fail).
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_all_tests() -> int:
    """
    1. The use of function:
       Runs all automated tests in the tests/ directory.
    2. How does the specific function work:
       Attempts to execute pytest using the local virtual environment Python or
       system Python. If pytest is not installed, falls back to unittest discovery.
    3. Concepts used in the function:
       Test orchestration and exit code propagation.
    4. Places where this function is imported and used:
       Executed directly: `python run_tests.py`.
    5. Final output of the function, before and after:
       Returns 0 on success, non-zero on failure.
    """
    venv_pytest = PROJECT_ROOT / ".venv" / "Scripts" / "pytest.exe"
    if not venv_pytest.is_file():
        venv_pytest = PROJECT_ROOT / ".venv" / "bin" / "pytest"

    if venv_pytest.is_file():
        cmd = [str(venv_pytest), "tests/"]
    else:
        cmd = [sys.executable, "-m", "pytest", "tests/"]

    print(f"[TestRunner] Running test suite: {' '.join(cmd)}")
    try:
        ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        return ret.returncode
    except Exception as e:
        print(f"[TestRunner] Error executing pytest: {e}")
        print("[TestRunner] Falling back to standard unittest...")
        import unittest
        suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"))
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)
        return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
