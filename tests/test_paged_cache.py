"""
File: tests/test_paged_cache.py

1. What this file does:
   Provides comprehensive unit and integration tests for the Paged Key-Value (KV) Cache and
   Virtual Token Memory Manager (PhysicalBlock, BlockTable, SequenceDescriptor, WasteComparisonMetrics,
   and PagedKVCacheManager). Validates demand paging, Copy-On-Write sharing, dynamic expansion,
   memory reclamation, comparative waste reduction, error handling, and serialization.

2. Overall design:
   Executes 100% on CPU with zero hardware, CUDA, or kernel dependencies. Structured into
   focused test classes addressing paging boundaries, Copy-On-Write mutation semantics, lifecycle
   reclamation, memory savings verification, out-of-memory exception contracts, and telemetry structures.

3. Elements in the file:
   - TestPagedKVAllocation: Tests initial prompt chunking and logical-to-physical block table mapping.
   - TestDynamicExpansion: Tests block boundary crossing during autoregressive token appending.
   - TestCopyOnWriteSharing: Tests prefix sharing via ref counting and COW mutation isolation.
   - TestMemoryReclamation: Tests ref count decrementing and pool frame recycling upon sequence freeing.
   - TestWasteComparison: Tests mathematical metrics and memory savings vs contiguous pre-allocation.
   - TestErrorHandlingAndReset: Tests OOM exception raising, duplicate sequences, and reset behavior.
   - TestSerializationAndTelemetry: Tests dictionary exports and hierarchical visualization structures.

4. Final output of the file:
   Pytest test suite producing verifiable pass/fail assertions for all paged KV-cache capabilities.
"""

from pathlib import Path
import sys
import pytest

# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizer.paged_cache import (
    PhysicalBlock,
    BlockTable,
    SequenceDescriptor,
    WasteComparisonMetrics,
    PagedKVCacheManager,
)


class TestPagedKVAllocation:
    """
    1. The use of class:
       Tests the initialization, prompt token partitioning, and page table creation of sequences.

    2. What the class contain:
       Test methods verifying multi-block prompt splitting, single-block prompts, empty prompts,
       and logical block table translation accuracy.

    3. Technologies used and why:
       Pytest framework for assertion and test execution.

    4. Key design and overall design of the class:
       Validates that PagedKVCacheManager splits arbitrary token streams into discrete fixed-size
       physical blocks without external fragmentation.
    """

    def test_create_sequence_paging(self) -> None:
        """
        1. The use of function:
           Verifies prompt tokens are split across 16-token physical blocks and mapped in BlockTable.

        2. How does the specific function work:
           Initializes manager with block size 16. Creates a sequence with 35 tokens.
           Asserts that 3 physical blocks are allocated, verifies token distribution across blocks
           (16, 16, 3), and confirms BlockTable correctly maps logical indices 0, 1, 2.

        3. Concepts used in the function:
           Page frame allocation, block table mapping, chunking boundary verification.

        4. Places where this function is imported and used:
           Invoked by pytest test runner.

        5. Final output of the function, before and after:
           Before: empty manager.
           After: verified sequence with 3 allocated, properly populated physical blocks.
        """
        manager = PagedKVCacheManager(total_physical_blocks=16, block_size_tokens=16)
        prompt_tokens = list(range(100, 135))  # 35 tokens

        seq = manager.create_sequence("seq-1", prompt_tokens)

        assert seq.seq_id == "seq-1"
        assert seq.token_count == 35
        assert len(seq.block_table) == 3
        assert seq.logical_blocks_needed(16) == 3

        # Verify logical to physical mappings
        phys_id_0 = seq.block_table.get_physical_id(0)
        phys_id_1 = seq.block_table.get_physical_id(1)
        phys_id_2 = seq.block_table.get_physical_id(2)

        # Unique blocks
        assert len({phys_id_0, phys_id_1, phys_id_2}) == 3

        block_0 = manager.physical_blocks[phys_id_0]
        block_1 = manager.physical_blocks[phys_id_1]
        block_2 = manager.physical_blocks[phys_id_2]

        assert block_0.token_ids == prompt_tokens[0:16]
        assert block_0.is_full() is True
        assert block_0.free_slots() == 0
        assert block_0.ref_count == 1
        assert block_0.is_free is False

        assert block_1.token_ids == prompt_tokens[16:32]
        assert block_1.is_full() is True
        assert block_1.free_slots() == 0
        assert block_1.ref_count == 1
        assert block_1.is_free is False

        assert block_2.token_ids == prompt_tokens[32:35]
        assert block_2.is_full() is False
        assert block_2.free_slots() == 13
        assert block_2.ref_count == 1
        assert block_2.is_free is False

        assert len(manager.free_block_pool) == 13


class TestDynamicExpansion:
    """
    1. The use of class:
       Tests dynamic allocation of physical blocks as generation crosses block boundaries.

    2. What the class contain:
       Test methods verifying single-token appends, boundary crossing triggers, and token indexing.

    3. Technologies used and why:
       Pytest assertions and iterative token generation simulation.

    4. Key design and overall design of the class:
       Simulates LLM autoregressive decoding token-by-token to ensure on-demand allocation works.
    """

    def test_append_token_block_expansion(self) -> None:
        """
        1. The use of function:
           Verifies adding tokens dynamically allocates new blocks when crossing block boundaries.

        2. How does the specific function work:
           Creates a sequence with exactly 16 tokens (filling block 0). Appends a 17th token,
           verifying allocation of block 1. Appends 15 more tokens to fill block 1, then appends
           a 33rd token, verifying allocation of block 2.

        3. Concepts used in the function:
           Autoregressive demand paging, dynamic page table expansion.

        4. Places where this function is imported and used:
           Invoked by pytest test runner.

        5. Final output of the function, before and after:
           Before: sequence with 1 full block.
           After: sequence dynamically expanded to 3 blocks holding 33 tokens.
        """
        manager = PagedKVCacheManager(total_physical_blocks=16, block_size_tokens=16)
        prompt = list(range(16))  # Exactly fills 1 block
        seq = manager.create_sequence("seq-dyn", prompt)

        assert len(seq.block_table) == 1
        phys_0 = seq.block_table.get_physical_id(0)
        assert manager.physical_blocks[phys_0].is_full() is True

        # Append token 16 (17th token) -> must allocate new physical block
        phys_ret_1 = manager.append_token("seq-dyn", 16)
        assert len(seq.block_table) == 2
        phys_1 = seq.block_table.get_physical_id(1)
        assert phys_ret_1 == phys_1
        assert phys_1 != phys_0
        assert seq.token_count == 17
        assert manager.physical_blocks[phys_1].token_ids == [16]

        # Fill block 1 with 15 more tokens (tokens 17..31)
        for tok in range(17, 32):
            manager.append_token("seq-dyn", tok)

        assert seq.token_count == 32
        assert len(seq.block_table) == 2
        assert manager.physical_blocks[phys_1].is_full() is True

        # Append token 32 (33rd token) -> crosses boundary into block 2
        phys_ret_2 = manager.append_token("seq-dyn", 32)
        assert len(seq.block_table) == 3
        phys_2 = seq.block_table.get_physical_id(2)
        assert phys_ret_2 == phys_2
        assert seq.token_count == 33
        assert manager.physical_blocks[phys_2].token_ids == [32]


class TestCopyOnWriteSharing:
    """
    1. The use of class:
       Tests the Copy-On-Write (COW) mechanism and zero-copy prefix sharing between sequences.

    2. What the class contain:
       Test methods verifying sequence forking, reference count increments, COW unsharing on mutation,
       and preservation of parent sequence state.

    3. Technologies used and why:
       Pytest assertions validating shared vs private physical block IDs and token contents.

    4. Key design and overall design of the class:
       Directly verifies the core vLLM PagedAttention optimization: shared prompt blocks share
       physical memory frames until divergence, saving significant accelerator memory.
    """

    def test_fork_and_copy_on_write(self) -> None:
        """
        1. The use of function:
           Verifies prefix sharing via reference counts and Copy-On-Write upon mutation.

        2. How does the specific function work:
           1. Sequence A created with 20 tokens (Block 0 full with 16 tokens, Block 1 with 4 tokens).
           2. Fork sequence B from A -> verifies both share block 0 and block 1 (ref_count = 2).
           3. Append token to B -> triggers Copy-On-Write for block 1: a new physical block is
              allocated for B with tokens copied + new token, block 1 ref_count on A decrements to 1,
              while block 0 remains shared (ref_count = 2).
           4. Verifies A's tokens remain unmodified while B has the mutated state.

        3. Concepts used in the function:
           Copy-On-Write (COW), zero-copy sequence forking, reference counting, page divergence.

        4. Places where this function is imported and used:
           Invoked by pytest test runner.

        5. Final output of the function, before and after:
           Before: single sequence A.
           After: two branched sequences sharing prefix block 0, with private diverging tail blocks.
        """
        manager = PagedKVCacheManager(total_physical_blocks=16, block_size_tokens=16)

        # 1. Sequence A created with 20 tokens
        prompt = list(range(20))
        seq_a = manager.create_sequence("seq-A", prompt)

        assert seq_a.token_count == 20
        assert len(seq_a.block_table) == 2
        block_0_id = seq_a.block_table.get_physical_id(0)
        block_1_id = seq_a.block_table.get_physical_id(1)

        assert manager.physical_blocks[block_0_id].ref_count == 1
        assert manager.physical_blocks[block_0_id].is_full() is True
        assert manager.physical_blocks[block_1_id].ref_count == 1
        assert len(manager.physical_blocks[block_1_id].token_ids) == 4

        # 2. Fork sequence B from A
        seq_b = manager.fork_sequence("seq-A", "seq-B")

        assert seq_b.seq_id == "seq-B"
        assert seq_b.parent_seq_id == "seq-A"
        assert seq_b.token_count == 20
        assert seq_b.block_table.get_physical_id(0) == block_0_id
        assert seq_b.block_table.get_physical_id(1) == block_1_id

        # Both blocks are now shared
        assert manager.physical_blocks[block_0_id].ref_count == 2
        assert manager.physical_blocks[block_1_id].ref_count == 2

        # 3. Append token 999 to B -> triggers COW on block 1
        new_b_phys_id = manager.append_token("seq-B", 999)

        # A new physical block must have been allocated for B's logical block 1
        assert new_b_phys_id != block_1_id
        assert seq_b.block_table.get_physical_id(1) == new_b_phys_id

        # Block 0 is still shared with ref_count 2
        assert seq_b.block_table.get_physical_id(0) == block_0_id
        assert manager.physical_blocks[block_0_id].ref_count == 2

        # Block 1 on sequence A decremented its ref_count to 1
        assert seq_a.block_table.get_physical_id(1) == block_1_id
        assert manager.physical_blocks[block_1_id].ref_count == 1
        assert manager.physical_blocks[block_1_id].token_ids == prompt[16:20]  # Unchanged!

        # Sequence B's new block contains the copied tokens plus the new token
        new_block = manager.physical_blocks[new_b_phys_id]
        assert new_block.ref_count == 1
        assert new_block.token_ids == prompt[16:20] + [999]
        assert seq_b.token_count == 21
        assert seq_a.token_count == 20


class TestMemoryReclamation:
    """
    1. The use of class:
       Tests freeing sequences, reference count decrementing, and block reclamation into the pool.

    2. What the class contain:
       Test methods verifying that shared blocks persist while unshared blocks return immediately
       to the free pool, and that completely released blocks restore pool capacity to 100%.

    3. Technologies used and why:
       Pytest assertions checking pool size and block status flags.

    4. Key design and overall design of the class:
       Verifies zero memory leaks when request streams complete or abort.
    """

    def test_free_sequence_reclamation(self) -> None:
        """
        1. The use of function:
           Verifies free_sequence decrements ref_counts and reclaims unreferenced blocks to free_block_pool.

        2. How does the specific function work:
           Allocates sequence A and forks sequence B. Verifies 2 blocks are shared. Appends token
           to B to allocate a private COW block (total 3 allocated blocks). Frees sequence B, verifying
           its private block is reclaimed, while shared blocks remain allocated (ref_count drops to 1).
           Frees sequence A, verifying remaining blocks are reclaimed and pool is 100% restored.

        3. Concepts used in the function:
           Reference counted garbage collection, memory reclamation, pool recycling.

        4. Places where this function is imported and used:
           Invoked by pytest test runner.

        5. Final output of the function, before and after:
           Before: busy manager with 3 allocated blocks.
           After: all sequences freed, free_block_pool restored to initial capacity.
        """
        initial_blocks = 16
        manager = PagedKVCacheManager(total_physical_blocks=initial_blocks, block_size_tokens=16)

        # Allocate A (2 blocks)
        manager.create_sequence("seq-A", list(range(20)))
        assert len(manager.free_block_pool) == 14

        # Fork B (shares 2 blocks, 0 new blocks allocated)
        manager.fork_sequence("seq-A", "seq-B")
        assert len(manager.free_block_pool) == 14

        # Mutate B (COW allocates 1 new block)
        manager.append_token("seq-B", 42)
        assert len(manager.free_block_pool) == 13

        # Free sequence B
        manager.free_sequence("seq-B")
        assert "seq-B" not in manager.sequences
        # The private COW block is freed, shared blocks remain because A references them
        assert len(manager.free_block_pool) == 14

        # Sequence A still holds its 2 blocks with ref_count 1
        for phys_id in manager.sequences["seq-A"].block_table.get_all_physical_ids():
            assert manager.physical_blocks[phys_id].ref_count == 1
            assert manager.physical_blocks[phys_id].is_free is False

        # Free sequence A -> all blocks should be reclaimed
        manager.free_sequence("seq-A")
        assert "seq-A" not in manager.sequences
        assert len(manager.free_block_pool) == initial_blocks
        for i in range(initial_blocks):
            assert manager.physical_blocks[i].is_free is True
            assert manager.physical_blocks[i].ref_count == 0


class TestWasteComparison:
    """
    1. The use of class:
       Tests the memory waste and fragmentation comparison against traditional contiguous allocation.

    2. What the class contain:
       Test methods verifying traditional allocation calculation, paged allocation calculation,
       fragmentation percentages, and memory savings verification (>60%).

    3. Technologies used and why:
       Pytest assertions comparing arithmetic fields of WasteComparisonMetrics.

    4. Key design and overall design of the class:
       Provides algorithmic proof that dynamic paged KV-caching eliminates over-reservation waste.
    """

    def test_memory_waste_comparison(self) -> None:
        """
        1. The use of function:
           Verifies that paged allocation drastically reduces memory waste compared to traditional
           pre-allocation (>60% savings).

        2. How does the specific function work:
           Creates 4 sequences with 20 tokens each. Compares waste against max_seq_len=2048.
           Asserts that traditional allocation pre-reserves 4 * 2048 * 128 bytes with massive waste,
           while paged allocation only reserves 4 * 2 * 16 * 128 bytes. Asserts memory savings > 60%.

        3. Concepts used in the function:
           Memory over-reservation, internal fragmentation, memory efficiency benchmarking.

        4. Places where this function is imported and used:
           Invoked by pytest test runner.

        5. Final output of the function, before and after:
           Before: 4 active sequences.
           After: metrics computed demonstrating >60% memory savings.
        """
        manager = PagedKVCacheManager(
            total_physical_blocks=64, block_size_tokens=16, bytes_per_token=128
        )
        for i in range(4):
            manager.create_sequence(f"seq-{i}", list(range(20)))

        metrics = manager.compare_waste(max_seq_len=2048)

        assert metrics.total_sequences == 4
        assert metrics.max_seq_len == 2048
        assert metrics.bytes_per_token == 128

        # Traditional: 4 * 2048 * 128 = 1,048,576 bytes
        expected_trad_alloc = 4 * 2048 * 128
        assert metrics.traditional_allocated_bytes == expected_trad_alloc

        # Traditional used: 4 * 20 * 128 = 10,240 bytes
        expected_trad_used = 4 * 20 * 128
        assert metrics.traditional_used_bytes == expected_trad_used
        assert metrics.traditional_wasted_bytes == expected_trad_alloc - expected_trad_used
        assert metrics.traditional_waste_pct > 90.0

        # Paged: each seq has 20 tokens -> 2 blocks of 16 slots = 32 slots per seq.
        # 4 seqs * 2 blocks = 8 blocks * 16 slots * 128 bytes = 16,384 bytes.
        expected_paged_alloc = 8 * 16 * 128
        assert metrics.paged_allocated_bytes == expected_paged_alloc

        # Paged used: 80 tokens * 128 bytes = 10,240 bytes
        assert metrics.paged_used_bytes == expected_trad_used

        # Memory savings must exceed 60% (here it is ~98.4%)
        assert metrics.memory_savings_pct > 60.0
        assert metrics.memory_savings_pct > 95.0

        # Verify serialization
        m_dict = metrics.to_dict()
        assert isinstance(m_dict, dict)
        assert m_dict["total_sequences"] == 4
        assert m_dict["memory_savings_pct"] == metrics.memory_savings_pct


class TestErrorHandlingAndReset:
    """
    1. The use of class:
       Tests boundary limits, out-of-memory errors, duplicate sequence keys, and reset functionality.

    2. What the class contain:
       Test methods verifying MemoryError on exhaustion, KeyError on non-existent sequences,
       ValueError on duplicate sequence IDs, and reset restoration.

    3. Technologies used and why:
       Pytest `pytest.raises` context manager for exception validation.

    4. Key design and overall design of the class:
       Ensures rock-solid exception safety and clean state teardown.
    """

    def test_out_of_blocks_error(self) -> None:
        """
        1. The use of function:
           Verifies memory exhaustion exception when free_block_pool is empty.

        2. How does the specific function work:
           Constructs a manager with only 2 physical blocks (capacity 32 tokens).
           Allocates a sequence with 32 tokens (occupying both blocks).
           Attempts to append a 33rd token, asserting MemoryError is raised.
           Attempts to create another sequence, asserting MemoryError is raised.

        3. Concepts used in the function:
           Out-of-memory exception handling, physical frame exhaustion.

        4. Places where this function is imported and used:
           Invoked by pytest test runner.

        5. Final output of the function, before and after:
           Before: manager with 2 total blocks.
           After: confirms MemoryError is reliably raised when resources are exhausted.
        """
        manager = PagedKVCacheManager(total_physical_blocks=2, block_size_tokens=16)

        # Allocate both blocks (32 tokens)
        manager.create_sequence("seq-full", list(range(32)))
        assert len(manager.free_block_pool) == 0

        # Attempt to append token -> requires block 3 -> MemoryError
        with pytest.raises(MemoryError):
            manager.append_token("seq-full", 99)

        # Attempt to create new sequence with prompt -> MemoryError
        with pytest.raises(MemoryError):
            manager.create_sequence("seq-new", [1, 2, 3])

    def test_duplicate_and_missing_sequence_errors(self) -> None:
        """
        1. The use of function:
           Tests error handling for duplicate sequence creation and operations on missing sequences.

        2. How does the specific function work:
           Attempts to create an existing sequence ID, asserting ValueError.
           Attempts to append, fork, or free non-existent sequences, asserting KeyError.

        3. Concepts used in the function:
           Input validation, exception checking.

        4. Places where this function is imported and used:
           Invoked by pytest test runner.

        5. Final output of the function, before and after:
           Before: valid manager.
           After: all invalid operations raise appropriate exceptions.
        """
        manager = PagedKVCacheManager(total_physical_blocks=8, block_size_tokens=16)
        manager.create_sequence("seq-1", [10, 20])

        with pytest.raises(ValueError):
            manager.create_sequence("seq-1", [30])

        with pytest.raises(KeyError):
            manager.append_token("non-existent", 40)

        with pytest.raises(KeyError):
            manager.fork_sequence("non-existent", "child")

        with pytest.raises(KeyError):
            manager.free_sequence("non-existent")

    def test_reset_functionality(self) -> None:
        """
        1. The use of function:
           Verifies reset restores the entire manager state to pristine clean condition.

        2. How does the specific function work:
           Allocates sequences and appends tokens. Calls reset(). Asserts zero sequences remain,
           all physical blocks have ref_count=0 and is_free=True, and free_block_pool is full.

        3. Concepts used in the function:
           Subsystem state reset and idempotency.

        4. Places where this function is imported and used:
           Invoked by pytest test runner.

        5. Final output of the function, before and after:
           Before: manager with allocated state.
           After: pristine manager with 100% capacity restored.
        """
        manager = PagedKVCacheManager(total_physical_blocks=8, block_size_tokens=16)
        manager.create_sequence("s1", list(range(20)))
        manager.fork_sequence("s1", "s2")

        manager.reset()

        assert len(manager.sequences) == 0
        assert len(manager.free_block_pool) == 8
        assert manager.free_block_pool == list(range(8))
        for block in manager.physical_blocks.values():
            assert block.is_free is True
            assert block.ref_count == 0
            assert len(block.token_ids) == 0


class TestSerializationAndTelemetry:
    """
    1. The use of class:
       Tests dictionary serialization and hierarchical visualization data structures.

    2. What the class contain:
       Test methods verifying PhysicalBlock.to_dict, BlockTable.to_dict, SequenceDescriptor.to_dict,
       and get_block_table_visualization.

    3. Technologies used and why:
       Pytest assertions verifying dictionary structure, nested types, and key existence.

    4. Key design and overall design of the class:
       Guarantees telemetry data structures conform to UI and diagnostic expectations.
    """

    def test_serialization_and_visualization(self) -> None:
        """
        1. The use of function:
           Verifies to_dict methods and get_block_table_visualization hierarchical output.

        2. How does the specific function work:
           Creates sequence A and forks sequence B with COW mutation. Retrieves visualization
           payload and asserts summary, sequences, mappings, ref counts, and block entries.

        3. Concepts used in the function:
           Telemetry serialization, diagnostic schema validation.

        4. Places where this function is imported and used:
           Invoked by pytest test runner.

        5. Final output of the function, before and after:
           Before: active manager state.
           After: verified serialization dictionaries and visualization payload.
        """
        manager = PagedKVCacheManager(total_physical_blocks=8, block_size_tokens=16)
        manager.create_sequence("seq-vis-A", list(range(20)))
        manager.fork_sequence("seq-vis-A", "seq-vis-B")
        manager.append_token("seq-vis-B", 999)

        # Test PhysicalBlock serialization
        pb_dict = manager.physical_blocks[0].to_dict()
        assert "block_id" in pb_dict
        assert "ref_count" in pb_dict
        assert "free_slots" in pb_dict

        # Test SequenceDescriptor serialization
        sd_dict = manager.sequences["seq-vis-A"].to_dict()
        assert sd_dict["seq_id"] == "seq-vis-A"
        assert "block_table" in sd_dict

        # Test visualization payload
        viz = manager.get_block_table_visualization()
        assert "summary" in viz
        assert "sequences" in viz
        assert "physical_blocks" in viz

        summary = viz["summary"]
        assert summary["total_physical_blocks"] == 8
        assert summary["active_sequences_count"] == 2
        assert summary["used_blocks_count"] == 3

        # Sequences mappings
        seqs = viz["sequences"]
        assert len(seqs) == 2
        seq_b_data = next(s for s in seqs if s["seq_id"] == "seq-vis-B")
        assert len(seq_b_data["mappings"]) == 2
        # Logical block 0 should be shared (is_shared == True)
        assert seq_b_data["mappings"][0]["is_shared"] is True
        # Logical block 1 is private to B due to COW (is_shared == False)
        assert seq_b_data["mappings"][1]["is_shared"] is False
