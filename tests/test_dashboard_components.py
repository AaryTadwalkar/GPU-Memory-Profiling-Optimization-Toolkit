"""
File: tests/test_dashboard_components.py

1. What this file does:
   Comprehensive test suite validating the presentation rendering functions in
   `dashboard/components.py`, specifically targeting the Active OS Virtual Memory,
   Tiered Activation Swapper, and Paged KV-Cache UI components.

2. Overall design:
   Uses unittest.mock to intercept Streamlit calls (markdown, altair_chart, columns, etc.),
   verifying that rendering functions execute without exceptions, correctly parse input
   dataclasses and dictionaries, and produce well-formed HTML and Altair charts.
   Executes 100% on CPU inside .venv without touching physical hardware or GPU drivers.

3. Elements in the file:
   - TestPageTableView: Tests render_page_table_view across empty and populated page tables.
   - TestHitRatioComparisonChart: Tests render_hit_ratio_comparison_chart with empty and comparative policy metrics.
   - TestTieredMemoryDualGauge: Tests render_tiered_memory_dual_gauge across normal and spillover conditions.
   - TestPagedKVBlockDiagram: Tests render_paged_kv_block_diagram with COW shared, private, and free blocks.
   - TestKVWasteComparisonChart: Tests render_kv_waste_comparison_chart with WasteComparisonMetrics models.
   - TestTieredTimelineChart: Tests render_tiered_timeline_chart multi-series Altair chart generation.

4. Final output of the file:
   Pytest execution output verifying UI component robustness, error tolerance, and data parsing integrity.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.virtual_memory import PageEntry, VirtualMemoryMetrics
from optimizer.paged_cache import WasteComparisonMetrics
from dashboard.components import (
    render_page_table_view,
    render_hit_ratio_comparison_chart,
    render_tiered_memory_dual_gauge,
    render_tiered_timeline_chart,
    render_live_benchmark_chart,
    render_paged_kv_block_diagram,
    render_kv_waste_comparison_chart,
)


class TestPageTableView:
    """
    1. The use of class:
       Tests the render_page_table_view function for correct handling of empty tables,
       dataclass PageEntry objects, dictionary entries, and dirty/swap status badges.

    2. What the class contain:
       Test methods test_render_empty_page_table and test_render_populated_page_table.

    3. Technologies used and why:
       pytest framework and unittest.mock.patch to safely mock streamlit DOM injections.

    4. Key design and overall design of the class:
       Asserts that the component renders without unhandled exceptions and generates expected
       HTML strings containing VPN identifiers and frame mappings.
    """

    @patch("dashboard.components.st.markdown")
    def test_render_empty_page_table(self, mock_markdown: MagicMock):
        """
        1. The use of function:
           Verifies that an empty page table dictionary triggers the fallback informational card.

        2. How does the specific function work:
           Calls render_page_table_view with an empty dict and asserts st.markdown was called.

        3. Concepts used in the function:
           Mock verification and boundary input testing.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Empty dictionary. After: Verified markdown call with empty table notification.
        """
        render_page_table_view({}, [])
        assert mock_markdown.called
        rendered_text = mock_markdown.call_args[0][0]
        assert "Page Table Empty" in rendered_text

    @patch("dashboard.components.st.markdown")
    def test_render_populated_page_table(self, mock_markdown: MagicMock):
        """
        1. The use of function:
           Verifies that a populated page table generates rows with VRAM and Swap status indicators.

        2. How does the specific function work:
           Constructs PageEntry instances (one VRAM resident, one in swap) and passes them to
           render_page_table_view, verifying generated HTML contents.

        3. Concepts used in the function:
           Page table entry inspection and HTML generation verification.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: PageEntry mapping dictionary. After: Verified HTML containing VPN, Frame, and Swap tags.
        """
        entries = {
            1: PageEntry(page_id=1, frame_id=0, in_vram=True, in_swap=False, dirty=False, loaded_at=1.0, last_accessed_at=3.0, access_count=2),
            2: PageEntry(page_id=2, frame_id=None, in_vram=False, in_swap=True, dirty=True, loaded_at=0.0, last_accessed_at=2.0, access_count=1),
        }
        render_page_table_view(entries, [1])
        assert mock_markdown.called
        rendered_text = mock_markdown.call_args[0][0]
        assert "VPN 1" in rendered_text
        assert "VPN 2" in rendered_text
        assert "VRAM Frame #0" in rendered_text
        assert "Host Swap" in rendered_text
        assert "1 (Dirty)" in rendered_text


class TestHitRatioComparisonChart:
    """
    1. The use of class:
       Validates the render_hit_ratio_comparison_chart presentation function.

    2. What the class contain:
       Test methods test_render_empty_metrics and test_render_valid_metrics.

    3. Technologies used and why:
       pytest and unittest.mock.patch to verify Altair chart generation and Streamlit columns.

    4. Key design and overall design of the class:
       Ensures that policy comparison metrics are normalized into DataFrames and rendered into
       two side-by-side charts without syntax errors or missing field crashes.
    """

    @patch("dashboard.components.st.info")
    def test_render_empty_metrics(self, mock_info: MagicMock):
        """
        1. The use of function:
           Tests behavior when comparison metrics dictionary is empty.

        2. How does the specific function work:
           Passes an empty dict to render_hit_ratio_comparison_chart and asserts st.info is called.

        3. Concepts used in the function:
           Empty dictionary defensive guarding.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Empty input. After: Verified graceful fallback message.
        """
        render_hit_ratio_comparison_chart({})
        assert mock_info.called

    @patch("dashboard.components.st.altair_chart")
    @patch("dashboard.components.st.columns")
    def test_render_valid_metrics(self, mock_columns: MagicMock, mock_altair: MagicMock):
        """
        1. The use of function:
           Tests chart rendering when provided with standard VirtualMemoryMetrics objects.

        2. How does the specific function work:
           Configures mock columns, invokes render_hit_ratio_comparison_chart with FIFO, LRU,
           and Optimal metrics, and asserts altair_chart was called for both charts.

        3. Concepts used in the function:
           Multi-column charting and mock column context manager emulation.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Comparison dictionary. After: Verified dual Altair chart rendering calls.
        """
        dummy_col1 = MagicMock()
        dummy_col2 = MagicMock()
        mock_columns.return_value = (dummy_col1, dummy_col2)

        metrics = {
            "fifo": VirtualMemoryMetrics("fifo", 3, 10, 4, 6, 0.4, 0.6, 60.4, [], []),
            "lru": VirtualMemoryMetrics("lru", 3, 10, 6, 4, 0.6, 0.4, 40.6, [], []),
            "optimal": VirtualMemoryMetrics("optimal", 3, 10, 7, 3, 0.7, 0.3, 30.7, [], []),
        }

        render_hit_ratio_comparison_chart(metrics)
        assert mock_altair.called


class TestTieredMemoryDualGauge:
    """
    1. The use of class:
       Tests render_tiered_memory_dual_gauge for proper gauge calculations and HTML rendering.

    2. What the class contain:
       Test methods test_render_safe_state and test_render_spillover_state.

    3. Technologies used and why:
       pytest and unittest.mock.patch.

    4. Key design and overall design of the class:
       Validates math behind VRAM percentage, CPU overflow, and presence of neon gradient bars.
    """

    @patch("dashboard.components.st.markdown")
    def test_render_safe_state(self, mock_markdown: MagicMock):
        """
        1. The use of function:
           Tests dual gauge rendering when memory usage is below threshold with zero CPU spill.

        2. How does the specific function work:
           Calls render_tiered_memory_dual_gauge with vram_used=20.0, cpu_swap=0.0, asserts output HTML.

        3. Concepts used in the function:
           Headroom threshold verification and conditional CSS classes.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Memory parameters. After: Verified HTML rendering containing zero spill label.
        """
        render_tiered_memory_dual_gauge(vram_used_mb=20.0, cpu_swap_mb=0.0, vram_budget_mb=100.0, threshold_mb=50.0)
        assert mock_markdown.called
        rendered = mock_markdown.call_args[0][0]
        assert "20.0 MB" in rendered
        assert "100% Bit-for-Bit" in rendered

    @patch("dashboard.components.st.markdown")
    def test_render_spillover_state(self, mock_markdown: MagicMock):
        """
        1. The use of function:
           Tests dual gauge rendering when activations have spilled to host pinned RAM.

        2. How does the specific function work:
           Calls render_tiered_memory_dual_gauge with active cpu_swap=40.0, asserts spillover badges.

        3. Concepts used in the function:
           Tiered memory spillover and peak relief percentage verification.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Spillover memory values. After: Verified HTML indicating active DMA offload.
        """
        render_tiered_memory_dual_gauge(vram_used_mb=60.0, cpu_swap_mb=40.0, vram_budget_mb=100.0, threshold_mb=50.0)
        assert mock_markdown.called
        rendered = mock_markdown.call_args[0][0]
        assert "40.0 MB Active Spill" in rendered
        assert "DMA Offload Active" in rendered


class TestPagedKVBlockDiagram:
    """
    1. The use of class:
       Tests render_paged_kv_block_diagram across empty states and structured block visualizations.

    2. What the class contain:
       Test methods test_render_empty_viz and test_render_cow_block_diagram.

    3. Technologies used and why:
       pytest and unittest.mock.patch.

    4. Key design and overall design of the class:
       Asserts that shared Copy-On-Write blocks (ref_count > 1) display distinct COW badges
       and logical sequence chains map to translated physical blocks.
    """

    @patch("dashboard.components.st.info")
    def test_render_empty_viz(self, mock_info: MagicMock):
        """
        1. The use of function:
           Tests behavior when visualization payload is empty.

        2. How does the specific function work:
           Passes an empty dict to render_paged_kv_block_diagram and asserts st.info is called.

        3. Concepts used in the function:
           Defensive handling of uninitialized manager state.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Empty dict. After: Verified information message.
        """
        render_paged_kv_block_diagram({})
        assert mock_info.called

    @patch("dashboard.components.st.markdown")
    def test_render_cow_block_diagram(self, mock_markdown: MagicMock):
        """
        1. The use of function:
           Tests rendering of physical blocks including shared COW frames and sequence chains.

        2. How does the specific function work:
           Constructs a payload with one shared block (ref_count=2) and one private block (ref_count=1),
           calls render_paged_kv_block_diagram, and verifies badge contents.

        3. Concepts used in the function:
           Copy-On-Write badge verification and logical sequence mapping check.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Visualization payload dictionary. After: Verified HTML containing COW badges.
        """
        payload = {
            "summary": {
                "total_physical_blocks": 4,
                "free_blocks_count": 2,
                "used_blocks_count": 2,
                "active_sequences_count": 2,
                "block_size_tokens": 16,
                "bytes_per_token": 128,
            },
            "physical_blocks": [
                {"block_id": 0, "block_size_tokens": 16, "ref_count": 2, "is_free": False, "token_ids": list(range(16))},
                {"block_id": 1, "block_size_tokens": 16, "ref_count": 1, "is_free": False, "token_ids": [1, 2]},
                {"block_id": 2, "block_size_tokens": 16, "ref_count": 0, "is_free": True, "token_ids": []},
                {"block_id": 3, "block_size_tokens": 16, "ref_count": 0, "is_free": True, "token_ids": []},
            ],
            "sequences": [
                {
                    "seq_id": "prompt-stream-1",
                    "token_count": 16,
                    "parent_seq_id": None,
                    "mappings": [{"logical_idx": 0, "physical_block_id": 0, "ref_count": 2, "is_shared": True}],
                },
                {
                    "seq_id": "fork-stream-cow",
                    "token_count": 18,
                    "parent_seq_id": "prompt-stream-1",
                    "mappings": [
                        {"logical_idx": 0, "physical_block_id": 0, "ref_count": 2, "is_shared": True},
                        {"logical_idx": 1, "physical_block_id": 1, "ref_count": 1, "is_shared": False},
                    ],
                },
            ],
        }

        render_paged_kv_block_diagram(payload)
        assert mock_markdown.called
        rendered = mock_markdown.call_args[0][0]
        assert "Block #0" in rendered
        assert "COW (ref: 2)" in rendered
        assert "prompt-stream-1" in rendered
        assert "fork-stream-cow" in rendered


class TestKVWasteComparisonChart:
    """
    1. The use of class:
       Tests render_kv_waste_comparison_chart with WasteComparisonMetrics objects and fallback dicts.

    2. What the class contain:
       Test methods test_render_empty_waste and test_render_populated_waste.

    3. Technologies used and why:
       pytest and unittest.mock.patch.

    4. Key design and overall design of the class:
       Validates calculation of megabytes from bytes and presence of high-savings callout banners.
    """

    @patch("dashboard.components.st.info")
    def test_render_empty_waste(self, mock_info: MagicMock):
        """
        1. The use of function:
           Tests behavior when waste metrics parameter is None or empty.

        2. How does the specific function work:
           Passes empty dict to render_kv_waste_comparison_chart and asserts st.info is called.

        3. Concepts used in the function:
           Defensive handling of missing metrics.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Empty input. After: Verified information fallback message.
        """
        render_kv_waste_comparison_chart({})
        assert mock_info.called

    @patch("dashboard.components.st.altair_chart")
    @patch("dashboard.components.st.markdown")
    def test_render_populated_waste(self, mock_markdown: MagicMock, mock_altair: MagicMock):
        """
        1. The use of function:
           Tests waste comparison chart and savings banner rendering from WasteComparisonMetrics.

        2. How does the specific function work:
           Constructs WasteComparisonMetrics instance with 75% savings and verifies both Altair
           chart and markdown callout are triggered.

        3. Concepts used in the function:
           Altair bar chart creation and executive callout banner verification.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Waste metrics instance. After: Verified Altair chart and savings banner calls.
        """
        metrics = WasteComparisonMetrics(
            total_sequences=2,
            max_seq_len=2048,
            block_size_tokens=16,
            bytes_per_token=128,
            traditional_allocated_bytes=2 * 2048 * 128,
            traditional_used_bytes=2 * 64 * 128,
            traditional_wasted_bytes=(2 * 2048 * 128) - (2 * 64 * 128),
            traditional_waste_pct=96.88,
            paged_allocated_bytes=8 * 16 * 128,
            paged_used_bytes=2 * 64 * 128,
            paged_wasted_bytes=0,
            paged_waste_pct=0.0,
            memory_savings_pct=75.0,
        )

        render_kv_waste_comparison_chart(metrics)
        assert mock_altair.called
        assert mock_markdown.called
        rendered = mock_markdown.call_args[0][0]
        assert "75.0% Reduction" in rendered


class TestTieredTimelineChart:
    """
    1. The use of class:
       Tests the render_tiered_timeline_chart presentation component across baseline,
       capped VRAM, and CPU RAM spillover scenarios.

    2. What the class contain:
       Test method test_render_timeline_chart verifying Altair multi-series line chart generation.

    3. Technologies used and why:
       pytest and unittest.mock.patch to safely test Streamlit altair_chart invocations.

    4. Key design and overall design of the class:
       Asserts that the timeline chart executes without exceptions, processes baseline and threshold
       parameters, and triggers st.altair_chart with width="stretch".
    """

    @patch("dashboard.components.st.altair_chart")
    def test_render_timeline_chart(self, mock_altair: MagicMock):
        """
        1. The use of function:
           Tests execution and rendering of the activation timeline chart.

        2. How does the specific function work:
           Calls render_tiered_timeline_chart with baseline peak, threshold, and swap volume,
           asserting st.altair_chart was invoked with expected arguments.

        3. Concepts used in the function:
           Altair chart construction and mock assertion.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: numeric timeline memory values.
           After: verified call to st.altair_chart.
        """
        render_tiered_timeline_chart(baseline_peak_mb=120.0, threshold_mb=40.0, cpu_swap_mb=80.0)
        assert mock_altair.called
        chart_arg = mock_altair.call_args[0][0]
        assert chart_arg is not None


class TestLiveBenchmarkChart:
    """
    1. The use of class:
       Tests the render_live_benchmark_chart presentation component visualizing real-time
       primary GPU VRAM allocation vs host CPU pinned RAM spill over benchmark execution.

    2. What the class contain:
       Test methods verifying empty DataFrame handling and populated time-series rendering.

    3. Technologies used and why:
       pytest, pandas, and unittest.mock.patch to safely mock streamlit DOM and charting operations.

    4. Key design and overall design of the class:
       Asserts that the component renders without exceptions, layers time-series lines, points,
       and horizontal threshold markers, and invokes st.altair_chart with width="stretch".
    """

    @patch("dashboard.components.st.info")
    def test_render_empty_live_benchmark_chart(self, mock_info: MagicMock) -> None:
        """
        1. The use of function:
           Tests that render_live_benchmark_chart displays informational notice when passed an empty DataFrame.

        2. How does the specific function work:
           Invokes render_live_benchmark_chart with an empty DataFrame and asserts st.info was called.

        3. Concepts used in the function:
           Graceful UI degradation on empty telemetry streams.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Empty DataFrame.
           After: Verified call to st.info.
        """
        import pandas as pd
        render_live_benchmark_chart(pd.DataFrame())
        assert mock_info.called

    @patch("dashboard.components.st.altair_chart")
    def test_render_populated_live_benchmark_chart(self, mock_altair: MagicMock) -> None:
        """
        1. The use of function:
           Tests that render_live_benchmark_chart constructs and renders an Altair layered chart for valid telemetry.

        2. How does the specific function work:
           Constructs a multi-row benchmark DataFrame, invokes render_live_benchmark_chart,
           and asserts st.altair_chart was called with width="stretch".

        3. Concepts used in the function:
           Time-series chart generation, Altair layering, and UI presentation testing.

        4. Places where this function is imported and used:
           Executed by pytest test runner.

        5. Final output of the function, before and after:
           Before: Populated DataFrame with elapsed time, VRAM, and Host RAM columns.
           After: Verified invocation of st.altair_chart.
        """
        import pandas as pd
        df = pd.DataFrame([
            {"elapsed_s": 0.0, "vram_used_mb": 2400.0, "host_ram_used_mb": 0.0, "phase_name": "Phase 1", "status": "RUNNING"},
            {"elapsed_s": 10.0, "vram_used_mb": 5200.0, "host_ram_used_mb": 0.0, "phase_name": "Phase 1", "status": "PRESSURE"},
            {"elapsed_s": 25.0, "vram_used_mb": 4800.0, "host_ram_used_mb": 3400.0, "phase_name": "Phase 2", "status": "PROTECTED"},
        ])
        render_live_benchmark_chart(df, vram_budget_mb=6140.0, threshold_mb=4900.0)
        assert mock_altair.called
        chart_arg = mock_altair.call_args[0][0]
        assert chart_arg is not None
        assert mock_altair.call_args[1].get("width") == "stretch"

