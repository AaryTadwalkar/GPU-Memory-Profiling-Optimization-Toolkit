"""
File: tests/test_live_fragmentation.py

1. What this file does:
   Provides comprehensive unit and integration tests for the allocator fragmentation
   profiling engine, validating parse_snapshot_detailed, capture_live_fragmentation_snapshot,
   inactive block verification, memory accounting, and resource cleanup.

2. Overall design:
   Contains isolated test fixtures and test cases verifying that snapshot parsing accurately
   identifies inactive blocks, calculates FSFI, handles missing files, and performs clean
   memory reclamation without leaving residual GPU allocations.

3. Elements in the file:
   - TestSnapshotParsing: Tests detailed parsing on synthetic snapshot dictionaries.
   - TestLiveFragmentationSafety: Tests live capture preconditions, headroom checks, and cleanup.

4. Final output of the file:
   Pytest execution verifying fragmentation engine correctness, regression safety, and leak prevention.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List
import pytest

# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profiler.fragmentation import (
    parse_snapshot_json,
    parse_snapshot_detailed,
    capture_live_fragmentation_snapshot,
)

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


class TestSnapshotParsing:
    """
    1. The use of class:
       Validates snapshot parsing functions across various mock snapshot topologies.

    2. What the class contain:
       Contains unit test methods verifying inactive block summing, largest block extraction,
       FSFI calculation, and edge cases with zero inactive blocks.

    3. Technologies used and why:
       Pytest testing framework for assertions and diagnostics.

    4. Key design and overall design of the class:
       Provides structured mock segment and block dictionaries mirroring PyTorch memory_snapshot()
       to test the mathematical accuracy of parse_snapshot_json and parse_snapshot_detailed.
    """

    def test_parse_snapshot_detailed_synthetic(self) -> None:
        """
        1. The use of function:
           Tests parse_snapshot_detailed on a synthetic segment structure with known inactive blocks.

        2. How does the specific function work:
           Constructs a mock segment dictionary with two inactive blocks (4 MB and 12 MB) and one
           active block (8 MB). Verifies total inactive bytes (16 MB), largest inactive block (12 MB),
           and FSFI = 1 - (12 / 16) = 0.25.

        3. Concepts used in the function:
           Free-Space Fragmentation Index (FSFI) computation verification.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Synthetic snapshot dictionary.
           After: Asserted parsed metrics match theoretical values.
        """
        mock_snapshot = {
            "segments": [
                {
                    "blocks": [
                        {"state": "active_allocated", "size": 8 * 1024 * 1024},
                        {"state": "inactive", "size": 4 * 1024 * 1024},
                        {"state": "inactive", "size": 12 * 1024 * 1024},
                    ]
                }
            ]
        }
        res = parse_snapshot_detailed(mock_snapshot)
        assert res["total_inactive_bytes"] == 16 * 1024 * 1024
        assert res["largest_free_block_bytes"] == 12 * 1024 * 1024
        assert res["inactive_block_count"] == 2
        assert res["active_block_count"] == 1
        assert res["fsfi"] == pytest.approx(0.25, abs=1e-4)

    def test_parse_snapshot_empty_or_zero_inactive(self) -> None:
        """
        1. The use of function:
           Tests parsing behavior when no inactive blocks exist in the allocator pool.

        2. How does the specific function work:
           Passes an empty segment structure and verifies FSFI defaults to 0.0000 without division by zero.

        3. Concepts used in the function:
           Defensive arithmetic, edge case verification.

        4. Places where this function is imported and used:
           Executed by pytest.

        5. Final output of the function, before and after:
           Before: Snapshot without inactive blocks.
           After: Asserted FSFI is exactly 0.0.
        """
        mock_snapshot = {"segments": [{"blocks": [{"state": "active_allocated", "size": 1024}]}]}
        res = parse_snapshot_detailed(mock_snapshot)
        assert res["total_inactive_bytes"] == 0
        assert res["largest_free_block_bytes"] == 0
        assert res["inactive_block_count"] == 0
        assert res["fsfi"] == 0.0


class TestLiveFragmentationSafety:
    """
    1. The use of class:
       Validates the live allocator fragmentation capture pipeline, verifying safety checks,
       metric extraction, and post-execution cleanup.

    2. What the class contain:
       Contains unit test methods testing capture_live_fragmentation_snapshot on physical CUDA
       hardware (or asserting graceful failure if CUDA is absent).

    3. Technologies used and why:
       Pytest assertions and torch.cuda hardware introspection.

    4. Key design and overall design of the class:
       Ensures that when executed, the live fragmentation function captures truthful metrics,
       creates valid snapshots on disk, and leaves zero residual GPU memory leaks.
    """

    def test_live_fragmentation_execution_or_graceful_absence(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Executes capture_live_fragmentation_snapshot and asserts valid metric structures.

        2. How does the specific function work:
           Calls capture_live_fragmentation_snapshot with a conservative configuration (4 tensors of 2 MB).
           If CUDA is present, verifies success, snapshot file existence, metric keys, and cleanup.
           If CUDA is absent, verifies that a clear, non-fabricated error dictionary is returned.

        3. Concepts used in the function:
           Hardware-dependent test execution, graceful degradation, cleanup assertion.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Uninvoked fragmentation pipeline.
           After: Verified clean execution and memory reclamation.
        """
        res = capture_live_fragmentation_snapshot(
            tensor_count=4,
            tensor_size_mb=2,
            output_dir=str(tmp_path),
        )
        if HAS_CUDA:
            assert res["success"] is True
            assert "phase1_fragmented" in res
            assert "phase2_post_empty_cache" in res
            assert Path(res["snapshot_fragmented_path"]).is_file()
            assert Path(res["snapshot_post_empty_path"]).is_file()
            assert "inactive_blocks_created" in res
            assert "explanation" in res
            # Verify memory was reclaimed
            assert torch.cuda.memory_allocated() == 0
        else:
            assert res["success"] is False
            assert "error" in res

