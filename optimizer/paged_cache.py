"""
File: optimizer/paged_cache.py

1. What this file does:
   Implements a Paged Key-Value (KV) Cache and Virtual Token Memory Manager for LLM inference
   workloads based on the vLLM / PagedAttention (OSDI '23 Best Paper) paradigm. Translates
   operating system virtual memory concepts (paging, page tables, page frames, and Copy-On-Write)
   into user-space GPU/CPU KV-cache tensor allocation to eliminate external fragmentation and
   minimize internal fragmentation.

2. Overall design:
   The module models the KV-cache as a collection of fixed-size physical memory blocks
   (PhysicalBlock). Each request sequence (SequenceDescriptor) manages its own logical-to-physical
   page table (BlockTable). When sequences share prefixes (e.g., parallel sampling, system prompt
   sharing, or beam search), physical blocks are shared via reference counting. On mutation
   during autoregressive token generation, Copy-On-Write (COW) allocates private physical blocks
   only when unshared writes occur. The manager (PagedKVCacheManager) coordinates allocation,
   translation, sharing, deallocation, telemetry, and waste comparison against traditional
   contiguous pre-allocation.

3. Elements in the file:
   - PhysicalBlock: Dataclass representing a fixed-capacity physical memory frame for tokens.
   - BlockTable: Logical-to-physical address translation table for request sequences.
   - SequenceDescriptor: Dataclass modeling an active generation sequence (equivalent to an OS PCB).
   - WasteComparisonMetrics: Dataclass summarizing fragmentation and savings vs traditional allocation.
   - PagedKVCacheManager: Central OS-inspired virtual memory manager orchestrating paged KV-cache state.

4. Final output of the file:
   Exposes high-level paged KV-cache management, block translation, Copy-On-Write branching,
   and memory waste diagnostics for integration into LLM serving engines and visualization dashboards.
"""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class PhysicalBlock:
    """
    1. The use of class:
       Represents an individual physical memory frame allocated within the KV-cache pool,
       storing a fixed number of token states with reference counting for page sharing.

    2. What the class contain:
       Stores unique physical block ID, maximum token capacity per block, current reference
       count, allocation availability flag, and the list of resident token IDs.

    3. Technologies used and why:
       Python `@dataclass` for lightweight memory representation, fast attribute access,
       and minimal serialization overhead.

    4. Key design and overall design of the class:
       Emulates an OS physical page frame. It maintains stateful tracking of token occupancy
       and reference counting, allowing multiple logical sequences to safely share identical
       prefix blocks until mutation triggers Copy-On-Write.
    """

    block_id: int
    block_size_tokens: int
    ref_count: int = 0
    is_free: bool = True
    token_ids: List[int] = field(default_factory=list)

    def append_token(self, token_id: int) -> None:
        """
        1. The use of function:
           Appends a generated or prompt token ID into the physical block's slot array.

        2. How does the specific function work:
           Verifies the block is not full, then appends the integer token ID into token_ids.
           Raises RuntimeError if the block has reached its maximum token capacity.

        3. Concepts used in the function:
           Page frame slot insertion, bounded buffer capacity validation.

        4. Places where this function is imported and used:
           Invoked by PagedKVCacheManager.create_sequence and PagedKVCacheManager.append_token.

        5. Final output of the function, before and after:
           Before: block has N resident tokens (where N < block_size_tokens).
           After: block has N + 1 resident tokens including the newly added token_id.
        """
        if self.is_full():
            raise RuntimeError(
                f"PhysicalBlock {self.block_id} is full (capacity {self.block_size_tokens})"
            )
        self.token_ids.append(token_id)

    def is_full(self) -> bool:
        """
        1. The use of function:
           Checks whether the physical block has reached its maximum token capacity.

        2. How does the specific function work:
           Compares the length of resident token_ids against block_size_tokens.

        3. Concepts used in the function:
           Page saturation boundary detection.

        4. Places where this function is imported and used:
           Used across PagedKVCacheManager to determine whether block expansion or allocation is needed.

        5. Final output of the function, before and after:
           Before: unqueried capacity state.
           After: returns boolean True if no slots remain, False otherwise.
        """
        return len(self.token_ids) >= self.block_size_tokens

    def free_slots(self) -> int:
        """
        1. The use of function:
           Calculates the remaining unallocated token slots in the physical block.

        2. How does the specific function work:
           Subtracts the current count of resident token_ids from block_size_tokens.

        3. Concepts used in the function:
           Page capacity arithmetic, internal fragmentation calculation.

        4. Places where this function is imported and used:
           Used by PagedKVCacheManager for waste metrics calculation and block table visualization.

        5. Final output of the function, before and after:
           Before: unknown slot count.
           After: non-negative integer indicating available token slots.
        """
        return max(0, self.block_size_tokens - len(self.token_ids))

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Serializes the physical block state to a Python dictionary for telemetry and UI display.

        2. How does the specific function work:
           Packages block_id, block_size_tokens, ref_count, is_free, resident token count,
           free slots, and a shallow copy of token_ids into a standard dictionary.

        3. Concepts used in the function:
           Dictionary serialization, data encapsulation.

        4. Places where this function is imported and used:
           Used by PagedKVCacheManager.get_block_table_visualization and logging diagnostics.

        5. Final output of the function, before and after:
           Before: PhysicalBlock dataclass instance.
           After: dictionary representing physical block telemetry.
        """
        return {
            "block_id": self.block_id,
            "block_size_tokens": self.block_size_tokens,
            "ref_count": self.ref_count,
            "is_free": self.is_free,
            "token_count": len(self.token_ids),
            "free_slots": self.free_slots(),
            "token_ids": list(self.token_ids),
        }


class BlockTable:
    """
    1. The use of class:
       Provides logical-to-physical address translation for a specific generation sequence,
       mapping sequential virtual block indices to physical memory block IDs.

    2. What the class contain:
       Maintains an internal dictionary mapping logical block indices (0, 1, 2, ...) to
       physical block IDs within the manager's global pool.

    3. Technologies used and why:
       Standard Python dictionary and list mappings for O(1) page lookup, mutation, and cloning.

    4. Key design and overall design of the class:
       Direct analogue to an operating system Page Table. Decouples virtual token sequence
       continuity from contiguous physical memory layout, eliminating external memory fragmentation.
    """

    def __init__(self, entries: Optional[Dict[int, int]] = None) -> None:
        """
        1. The use of function:
           Initializes a BlockTable instance with an optional initial mapping.

        2. How does the specific function work:
           Copies the provided mapping into an internal dictionary or initializes an empty dictionary.

        3. Concepts used in the function:
           Page table instantiation, encapsulation.

        4. Places where this function is imported and used:
           Instantiated in PagedKVCacheManager.create_sequence, clone(), and deserialization.

        5. Final output of the function, before and after:
           Before: uninitialized BlockTable.
           After: BlockTable ready for address translation.
        """
        self._entries: Dict[int, int] = dict(entries) if entries is not None else {}

    def map_block(self, logical_idx: int, physical_id: int) -> None:
        """
        1. The use of function:
           Maps a virtual/logical block index to a concrete physical block ID.

        2. How does the specific function work:
           Assigns the physical_id to the key logical_idx in the internal translation dictionary.

        3. Concepts used in the function:
           Page table entry mapping, virtual-to-physical binding.

        4. Places where this function is imported and used:
           Invoked when allocating new blocks or swapping physical blocks during Copy-On-Write.

        5. Final output of the function, before and after:
           Before: logical_idx is either unmapped or mapped to an older physical block.
           After: logical_idx translates directly to physical_id.
        """
        self._entries[logical_idx] = physical_id

    def get_physical_id(self, logical_idx: int) -> int:
        """
        1. The use of function:
           Translates a logical block index to its mapped physical block ID.

        2. How does the specific function work:
           Retrieves the physical block ID from the internal dictionary. Raises KeyError if unmapped.

        3. Concepts used in the function:
           Virtual address translation, page table lookup.

        4. Places where this function is imported and used:
           Used by PagedKVCacheManager during token appending, COW checking, and visualization.

        5. Final output of the function, before and after:
           Before: logical block index query.
           After: translated physical block ID integer.
        """
        if logical_idx not in self._entries:
            raise KeyError(f"Logical block index {logical_idx} is not mapped in BlockTable")
        return self._entries[logical_idx]

    def get_all_physical_ids(self) -> List[int]:
        """
        1. The use of function:
           Returns all mapped physical block IDs ordered by their logical block indices.

        2. How does the specific function work:
           Sorts the logical index keys and returns the corresponding physical block IDs in order.

        3. Concepts used in the function:
           Page sequence traversal, ordered physical frame gathering.

        4. Places where this function is imported and used:
           Used during sequence forking, sequence deallocation, and waste analysis.

        5. Final output of the function, before and after:
           Before: internal dictionary entries.
           After: ordered list of physical block ID integers.
        """
        return [self._entries[idx] for idx in sorted(self._entries.keys())]

    def clone(self) -> "BlockTable":
        """
        1. The use of function:
           Creates a duplicate BlockTable mapping identical logical indices to physical frames.

        2. How does the specific function work:
           Instantiates a new BlockTable pre-populated with a shallow copy of the current entries.

        3. Concepts used in the function:
           Page table duplication for OS fork() semantics and shared prefix caching.

        4. Places where this function is imported and used:
           Used by PagedKVCacheManager.fork_sequence to spawn child sequences without copying data.

        5. Final output of the function, before and after:
           Before: single BlockTable instance.
           After: independent BlockTable pointing to the same physical blocks.
        """
        return BlockTable(dict(self._entries))

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Serializes the BlockTable mappings into a JSON-compatible dictionary.

        2. How does the specific function work:
           Packages the mapping entries and total block count into a dictionary.

        3. Concepts used in the function:
           State serialization, UI data structuring.

        4. Places where this function is imported and used:
           Used by SequenceDescriptor.to_dict and visualization routines.

        5. Final output of the function, before and after:
           Before: BlockTable object.
           After: dictionary containing block table entries and counts.
        """
        return {
            "entries": dict(self._entries),
            "num_blocks": len(self._entries),
            "physical_ids": self.get_all_physical_ids(),
        }

    def __len__(self) -> int:
        """
        1. The use of function:
           Returns the total number of mapped logical blocks in the table.

        2. How does the specific function work:
           Returns the count of entries in the internal dictionary.

        3. Concepts used in the function:
           Python special method protocol `__len__`.

        4. Places where this function is imported and used:
           Used throughout PagedKVCacheManager for bounds checking and block indexing.

        5. Final output of the function, before and after:
           Before: unmeasured table.
           After: integer length of mapped logical blocks.
        """
        return len(self._entries)

    def __getitem__(self, logical_idx: int) -> int:
        """
        1. The use of function:
           Enables subscript access syntax `block_table[logical_idx]`.

        2. How does the specific function work:
           Delegates directly to get_physical_id(logical_idx).

        3. Concepts used in the function:
           Python special method protocol `__getitem__`.

        4. Places where this function is imported and used:
           Convenience accessor in test suites and manager algorithms.

        5. Final output of the function, before and after:
           Before: index subscript query.
           After: mapped physical block ID integer.
        """
        return self.get_physical_id(logical_idx)


@dataclass
class SequenceDescriptor:
    """
    1. The use of class:
       Represents an active generation request sequence within the virtual memory manager,
       analogous to an Operating System Process Control Block (PCB).

    2. What the class contain:
       Stores sequence ID string, cumulative token count, associated BlockTable, optional
       parent sequence ID for forked lineage, and creation timestamp.

    3. Technologies used and why:
       Standard library `@dataclass` for clear, readable, and structured process modeling.

    4. Key design and overall design of the class:
       Encapsulates all virtual memory metadata of a generation stream. Provides helper
       methods to compute required logical blocks and export serialized dictionaries.
    """

    seq_id: str
    token_count: int
    block_table: BlockTable
    parent_seq_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def logical_blocks_needed(self, block_size_tokens: int) -> int:
        """
        1. The use of function:
           Calculates the exact number of logical blocks required to hold the sequence tokens.

        2. How does the specific function work:
           Uses integer ceiling division: 0 if token_count == 0, else ceil(token_count / block_size).

        3. Concepts used in the function:
           Ceiling division, page allocation sizing.

        4. Places where this function is imported and used:
           Used by PagedKVCacheManager and unit tests to verify block allocation parity.

        5. Final output of the function, before and after:
           Before: token count and block size parameters.
           After: integer count of required logical blocks.
        """
        if self.token_count <= 0:
            return 0
        return math.ceil(self.token_count / block_size_tokens)

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Converts the sequence descriptor and its block table into a dictionary representation.

        2. How does the specific function work:
           Packages seq_id, token_count, parent_seq_id, created_at, and serialized block table.

        3. Concepts used in the function:
           Dataclass serialization, nested object representation.

        4. Places where this function is imported and used:
           Used by PagedKVCacheManager.get_block_table_visualization and logging.

        5. Final output of the function, before and after:
           Before: SequenceDescriptor instance.
           After: structured dictionary representing sequence virtual memory state.
        """
        return {
            "seq_id": self.seq_id,
            "token_count": self.token_count,
            "parent_seq_id": self.parent_seq_id,
            "created_at": self.created_at,
            "block_table": self.block_table.to_dict(),
            "physical_block_ids": self.block_table.get_all_physical_ids(),
        }


@dataclass
class WasteComparisonMetrics:
    """
    1. The use of class:
       Quantifies and compares memory consumption and fragmentation between traditional
       static pre-allocation and OS-inspired paged KV-cache allocation.

    2. What the class contain:
       Stores sequence counts, maximum sequence lengths, block parameters, allocated bytes,
       used bytes, wasted bytes, waste percentages, and overall memory savings percentage.

    3. Technologies used and why:
       Standard library `@dataclass` providing structured, strongly-typed efficiency metrics.

    4. Key design and overall design of the class:
       Formulates concrete mathematical proof of the vLLM / PagedAttention efficiency gains,
       demonstrating how virtual paging eliminates over-reservation waste.
    """

    total_sequences: int
    max_seq_len: int
    block_size_tokens: int
    bytes_per_token: int
    traditional_allocated_bytes: int
    traditional_used_bytes: int
    traditional_wasted_bytes: int
    traditional_waste_pct: float
    paged_allocated_bytes: int
    paged_used_bytes: int
    paged_wasted_bytes: int
    paged_waste_pct: float
    memory_savings_pct: float

    def to_dict(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Serializes the waste comparison metrics into a standard dictionary.

        2. How does the specific function work:
           Invokes `dataclasses.asdict` on the metric fields.

        3. Concepts used in the function:
           Dataclass serialization.

        4. Places where this function is imported and used:
           Used by visualization widgets, telemetry reporters, and optimization benchmarks.

        5. Final output of the function, before and after:
           Before: WasteComparisonMetrics instance.
           After: JSON-compatible dictionary of metrics.
        """
        return asdict(self)


class PagedKVCacheManager:
    """
    1. The use of class:
       Serves as the central Virtual Memory Manager for LLM KV-cache allocations, managing
       a physical block pool, sequence block tables, Copy-On-Write sharing, and fragmentation metrics.

    2. What the class contain:
       Contains physical block dictionary, free block pool, active sequence registry, block size
       configurations, and bytes-per-token constants.

    3. Technologies used and why:
       Pure Python data structures (dicts, lists) enabling zero-dependency, safe, high-speed
       virtual memory simulation without hardware or CUDA dependencies.

    4. Key design and overall design of the class:
       Directly implements the vLLM PagedAttention paradigm: virtual token addresses map via
       block tables to physical blocks. Sequence forking shares blocks via reference counting,
       and token generation on shared blocks triggers Copy-On-Write allocation dynamically.
    """

    def __init__(
        self,
        total_physical_blocks: int = 128,
        block_size_tokens: int = 16,
        bytes_per_token: int = 128,
    ) -> None:
        """
        1. The use of function:
           Constructs and initializes the PagedKVCacheManager with physical memory constraints.

        2. How does the specific function work:
           Validates inputs, initializes total_physical_blocks PhysicalBlock instances, populates
           the free block pool, and sets up an empty sequence registry.

        3. Concepts used in the function:
           Physical memory frame initialization, free pool configuration.

        4. Places where this function is imported and used:
           Instantiated in tests, benchmarks, LLM inference managers, and optimization loops.

        5. Final output of the function, before and after:
           Before: unallocated memory space.
           After: initialized manager ready to allocate sequences and physical blocks.
        """
        if total_physical_blocks <= 0:
            raise ValueError("total_physical_blocks must be positive")
        if block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")
        if bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")

        self.total_physical_blocks: int = total_physical_blocks
        self.block_size_tokens: int = block_size_tokens
        self.bytes_per_token: int = bytes_per_token

        self.physical_blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i, block_size_tokens=block_size_tokens)
            for i in range(total_physical_blocks)
        }
        self.free_block_pool: List[int] = list(range(total_physical_blocks))
        self.sequences: Dict[str, SequenceDescriptor] = {}

    def _allocate_block(self) -> int:
        """
        1. The use of function:
           Allocates a single free physical block from the available pool.

        2. How does the specific function work:
           Checks if free_block_pool is empty (raising MemoryError if so). Pops the lowest available
           block ID, marks it as allocated with ref_count=1, and returns its block ID.

        3. Concepts used in the function:
           Operating system physical frame allocation, out-of-memory exception handling.

        4. Places where this function is imported and used:
           Used internally by create_sequence, append_token, and Copy-On-Write branching.

        5. Final output of the function, before and after:
           Before: free block exists in free_block_pool with is_free=True.
           After: block is marked is_free=False, ref_count=1, and removed from pool.
        """
        if not self.free_block_pool:
            raise MemoryError("Out of physical memory blocks: KV-cache pool exhausted")

        block_id = self.free_block_pool.pop(0)
        block = self.physical_blocks[block_id]
        block.is_free = False
        block.ref_count = 1
        block.token_ids.clear()
        return block_id

    def _free_block(self, block_id: int) -> None:
        """
        1. The use of function:
           Reclaims an unreferenced physical block and returns it to the free block pool.

        2. How does the specific function work:
           Resets the block's attributes (is_free=True, ref_count=0, clears tokens) and appends
           the block ID back to free_block_pool, maintaining sorted order for deterministic reuse.

        3. Concepts used in the function:
           Frame deallocation, memory recycling, fragmentation prevention.

        4. Places where this function is imported and used:
           Used internally by free_sequence and append_token during Copy-On-Write.

        5. Final output of the function, before and after:
           Before: block has ref_count=0.
           After: block is returned to free_block_pool and ready for reuse.
        """
        block = self.physical_blocks[block_id]
        block.is_free = True
        block.ref_count = 0
        block.token_ids.clear()
        if block_id not in self.free_block_pool:
            self.free_block_pool.append(block_id)
            self.free_block_pool.sort()

    def create_sequence(self, seq_id: str, prompt_tokens: List[int]) -> SequenceDescriptor:
        """
        1. The use of function:
           Registers a new generation sequence and partitions prompt tokens across physical blocks.

        2. How does the specific function work:
           Validates uniqueness of seq_id. Chunks prompt_tokens into chunks of block_size_tokens,
           allocates a physical block for each chunk, populates the block table, and creates a
           SequenceDescriptor. If prompt_tokens is empty, creates an empty BlockTable.

        3. Concepts used in the function:
           Paging, virtual memory process creation, chunked token allocation.

        4. Places where this function is imported and used:
           Invoked when initiating a new LLM generation request or prompt prefill.

        5. Final output of the function, before and after:
           Before: sequence does not exist in the manager.
           After: sequence is registered, physical blocks are allocated, and descriptor is returned.
        """
        if seq_id in self.sequences:
            raise ValueError(f"Sequence '{seq_id}' already exists")

        block_table = BlockTable()
        tokens = list(prompt_tokens)
        token_count = len(tokens)

        if token_count > 0:
            num_blocks = math.ceil(token_count / self.block_size_tokens)
            for logical_idx in range(num_blocks):
                start = logical_idx * self.block_size_tokens
                end = min(start + self.block_size_tokens, token_count)
                chunk = tokens[start:end]

                phys_id = self._allocate_block()
                block = self.physical_blocks[phys_id]
                for tok in chunk:
                    block.append_token(tok)
                block_table.map_block(logical_idx, phys_id)

        seq = SequenceDescriptor(
            seq_id=seq_id,
            token_count=token_count,
            block_table=block_table,
            parent_seq_id=None,
        )
        self.sequences[seq_id] = seq
        return seq

    def append_token(self, seq_id: str, token_id: int) -> int:
        """
        1. The use of function:
           Appends a newly sampled token to an active sequence, triggering dynamic block expansion
           or Copy-On-Write (COW) when writing to a shared physical block.

        2. How does the specific function work:
           Retrieves the sequence. If the sequence has no blocks or its last physical block is full,
           allocates a new physical block and maps the next logical index. If the last block is not
           full but is shared (ref_count > 1), triggers Copy-On-Write: decrements ref_count on the
           shared block, allocates a new private block, copies prior tokens plus the new token,
           and updates the block table. If unshared and not full, appends directly.

        3. Concepts used in the function:
           Copy-On-Write (COW), dynamic paging, demand allocation, reference counting.

        4. Places where this function is imported and used:
           Invoked during autoregressive generation step loops for active sequences.

        5. Final output of the function, before and after:
           Before: sequence has N tokens.
           After: sequence has N + 1 tokens; returns the physical block ID where the token was placed.
        """
        if seq_id not in self.sequences:
            raise KeyError(f"Sequence '{seq_id}' not found")

        seq = self.sequences[seq_id]
        table = seq.block_table

        if len(table) == 0:
            # First block allocation
            new_phys_id = self._allocate_block()
            block = self.physical_blocks[new_phys_id]
            block.append_token(token_id)
            table.map_block(0, new_phys_id)
            seq.token_count += 1
            return new_phys_id

        last_logical_idx = len(table) - 1
        current_phys_id = table.get_physical_id(last_logical_idx)
        current_block = self.physical_blocks[current_phys_id]

        if current_block.is_full():
            # Expansion across block boundary
            new_logical_idx = last_logical_idx + 1
            new_phys_id = self._allocate_block()
            new_block = self.physical_blocks[new_phys_id]
            new_block.append_token(token_id)
            table.map_block(new_logical_idx, new_phys_id)
            seq.token_count += 1
            return new_phys_id

        # Current block has remaining slots
        if current_block.ref_count > 1:
            # Copy-On-Write: unshare block and create private replica
            current_block.ref_count -= 1
            cow_phys_id = self._allocate_block()
            cow_block = self.physical_blocks[cow_phys_id]
            for tok in current_block.token_ids:
                cow_block.append_token(tok)
            cow_block.append_token(token_id)
            table.map_block(last_logical_idx, cow_phys_id)
            seq.token_count += 1
            return cow_phys_id

        # Unshared block with free slots
        current_block.append_token(token_id)
        seq.token_count += 1
        return current_phys_id

    def fork_sequence(self, parent_seq_id: str, child_seq_id: str) -> SequenceDescriptor:
        """
        1. The use of function:
           Forks a child sequence from an existing parent, duplicating the block table and
           incrementing physical block reference counts without copying memory data.

        2. How does the specific function work:
           Validates existence of parent and non-existence of child. Clones parent's BlockTable,
           increments ref_count for every mapped physical block, and creates a child SequenceDescriptor
           recording parent lineage.

        3. Concepts used in the function:
           OS fork() copy-on-write semantics, zero-copy memory sharing, reference counting.

        4. Places where this function is imported and used:
           Used in parallel beam search, parallel decoding, speculative decoding, and tree search.

        5. Final output of the function, before and after:
           Before: parent sequence exists with ref_counts R; child sequence does not exist.
           After: child sequence exists sharing physical blocks with ref_counts R + 1.
        """
        if parent_seq_id not in self.sequences:
            raise KeyError(f"Parent sequence '{parent_seq_id}' not found")
        if child_seq_id in self.sequences:
            raise ValueError(f"Child sequence '{child_seq_id}' already exists")

        parent_seq = self.sequences[parent_seq_id]
        child_table = parent_seq.block_table.clone()

        # Increment reference count for all shared physical blocks
        for phys_id in child_table.get_all_physical_ids():
            self.physical_blocks[phys_id].ref_count += 1

        child_seq = SequenceDescriptor(
            seq_id=child_seq_id,
            token_count=parent_seq.token_count,
            block_table=child_table,
            parent_seq_id=parent_seq_id,
        )
        self.sequences[child_seq_id] = child_seq
        return child_seq

    def free_sequence(self, seq_id: str) -> None:
        """
        1. The use of function:
           Terminates an active sequence, decrements reference counts of mapped physical blocks,
           and reclaims unreferenced blocks to the free pool.

        2. How does the specific function work:
           Retrieves sequence. For each physical block mapped in its BlockTable, decrements ref_count.
           If ref_count drops to zero, invokes _free_block to return it to free_block_pool.
           Removes the sequence from the sequence registry.

        3. Concepts used in the function:
           Process termination, reference-counted garbage collection, page reclamation.

        4. Places where this function is imported and used:
           Invoked when a generation sequence finishes generation or is canceled.

        5. Final output of the function, before and after:
           Before: sequence holds physical block references.
           After: sequence is removed; zero-reference physical blocks are returned to free pool.
        """
        if seq_id not in self.sequences:
            raise KeyError(f"Sequence '{seq_id}' not found")

        seq = self.sequences[seq_id]
        for phys_id in seq.block_table.get_all_physical_ids():
            block = self.physical_blocks[phys_id]
            block.ref_count -= 1
            if block.ref_count <= 0:
                self._free_block(phys_id)

        del self.sequences[seq_id]

    def compare_waste(self, max_seq_len: Optional[int] = None) -> WasteComparisonMetrics:
        """
        1. The use of function:
           Computes comprehensive memory waste metrics comparing traditional contiguous
           pre-allocation vs PagedAttention virtual memory allocation.

        2. How does the specific function work:
           Determines max_seq_len (defaults to max sequence token count or 2048).
           Calculates traditional allocated bytes as total_sequences * max_seq_len * bytes_per_token.
           Calculates paged allocated bytes by identifying unique allocated physical blocks.
           Derives wasted bytes and waste percentages for both strategies and computes memory savings.

        3. Concepts used in the function:
           Internal fragmentation, external fragmentation, over-reservation analysis.

        4. Places where this function is imported and used:
           Used by optimization benchmarks, reporting dashboards, and verification tests.

        5. Final output of the function, before and after:
           Before: raw sequence and block state.
           After: WasteComparisonMetrics dataclass summarizing comparative savings.
        """
        total_seqs = len(self.sequences)
        if total_seqs == 0:
            eff_max_len = max_seq_len or 2048
            return WasteComparisonMetrics(
                total_sequences=0,
                max_seq_len=eff_max_len,
                block_size_tokens=self.block_size_tokens,
                bytes_per_token=self.bytes_per_token,
                traditional_allocated_bytes=0,
                traditional_used_bytes=0,
                traditional_wasted_bytes=0,
                traditional_waste_pct=0.0,
                paged_allocated_bytes=0,
                paged_used_bytes=0,
                paged_wasted_bytes=0,
                paged_waste_pct=0.0,
                memory_savings_pct=0.0,
            )

        if max_seq_len is None:
            max_observed = max(seq.token_count for seq in self.sequences.values())
            eff_max_len = max(2048, max_observed)
        else:
            eff_max_len = max_seq_len

        # Traditional contiguous allocation: pre-allocates max_seq_len per sequence
        trad_allocated_bytes = total_seqs * eff_max_len * self.bytes_per_token
        trad_used_bytes = sum(seq.token_count for seq in self.sequences.values()) * self.bytes_per_token
        trad_wasted_bytes = max(0, trad_allocated_bytes - trad_used_bytes)
        trad_waste_pct = (
            (trad_wasted_bytes / trad_allocated_bytes * 100.0) if trad_allocated_bytes > 0 else 0.0
        )

        # Paged KV-cache: allocates only physical blocks in use
        allocated_phys_ids = set()
        for seq in self.sequences.values():
            allocated_phys_ids.update(seq.block_table.get_all_physical_ids())

        paged_allocated_blocks = len(allocated_phys_ids)
        paged_allocated_bytes = (
            paged_allocated_blocks * self.block_size_tokens * self.bytes_per_token
        )
        paged_used_bytes = (
            sum(len(self.physical_blocks[pid].token_ids) for pid in allocated_phys_ids)
            * self.bytes_per_token
        )
        paged_wasted_bytes = (
            sum(self.physical_blocks[pid].free_slots() for pid in allocated_phys_ids)
            * self.bytes_per_token
        )
        paged_waste_pct = (
            (paged_wasted_bytes / paged_allocated_bytes * 100.0)
            if paged_allocated_bytes > 0
            else 0.0
        )

        memory_savings_pct = (
            ((trad_allocated_bytes - paged_allocated_bytes) / trad_allocated_bytes * 100.0)
            if trad_allocated_bytes > 0
            else 0.0
        )

        return WasteComparisonMetrics(
            total_sequences=total_seqs,
            max_seq_len=eff_max_len,
            block_size_tokens=self.block_size_tokens,
            bytes_per_token=self.bytes_per_token,
            traditional_allocated_bytes=trad_allocated_bytes,
            traditional_used_bytes=trad_used_bytes,
            traditional_wasted_bytes=trad_wasted_bytes,
            traditional_waste_pct=round(trad_waste_pct, 2),
            paged_allocated_bytes=paged_allocated_bytes,
            paged_used_bytes=paged_used_bytes,
            paged_wasted_bytes=paged_wasted_bytes,
            paged_waste_pct=round(paged_waste_pct, 2),
            memory_savings_pct=round(memory_savings_pct, 2),
        )

    def get_block_table_visualization(self) -> Dict[str, Any]:
        """
        1. The use of function:
           Generates a structured hierarchical representation of physical blocks, logical
           block tables, and Copy-On-Write sharing for UI rendering and diagnostic inspection.

        2. How does the specific function work:
           Compiles global memory summary stats, iterates sequences to assemble logical-to-physical
           mappings with sharing flags, and serializes all physical block frames.

        3. Concepts used in the function:
           Virtual-to-physical memory visualization, telemetry synthesis.

        4. Places where this function is imported and used:
           Used by front-end dashboards, CLI status monitors, and memory profilers.

        5. Final output of the function, before and after:
           Before: raw internal manager dictionaries.
           After: structured dictionary formatted for visualization components.
        """
        used_blocks_count = self.total_physical_blocks - len(self.free_block_pool)
        sequences_data = []

        for seq in self.sequences.values():
            mappings = []
            for logical_idx in sorted(seq.block_table._entries.keys()):
                phys_id = seq.block_table.get_physical_id(logical_idx)
                block = self.physical_blocks[phys_id]
                mappings.append(
                    {
                        "logical_idx": logical_idx,
                        "physical_block_id": phys_id,
                        "ref_count": block.ref_count,
                        "is_shared": block.ref_count > 1,
                        "token_count": len(block.token_ids),
                        "free_slots": block.free_slots(),
                    }
                )

            sequences_data.append(
                {
                    "seq_id": seq.seq_id,
                    "token_count": seq.token_count,
                    "parent_seq_id": seq.parent_seq_id,
                    "created_at": seq.created_at,
                    "mappings": mappings,
                }
            )

        return {
            "summary": {
                "total_physical_blocks": self.total_physical_blocks,
                "free_blocks_count": len(self.free_block_pool),
                "used_blocks_count": used_blocks_count,
                "active_sequences_count": len(self.sequences),
                "block_size_tokens": self.block_size_tokens,
                "bytes_per_token": self.bytes_per_token,
            },
            "sequences": sequences_data,
            "physical_blocks": [
                self.physical_blocks[i].to_dict()
                for i in range(self.total_physical_blocks)
            ],
        }

    def reset(self) -> None:
        """
        1. The use of function:
           Resets the manager state, releasing all sequences and reclaiming all physical blocks.

        2. How does the specific function work:
           Clears sequences dict, resets each physical block's attributes, and restores
           free_block_pool to a full sequential list of block IDs.

        3. Concepts used in the function:
           Virtual memory subsystem reset, clean state reinitialization.

        4. Places where this function is imported and used:
           Used between benchmark iterations, test teardowns, and server reloads.

        5. Final output of the function, before and after:
           Before: manager with allocated sequences and busy physical blocks.
           After: pristine manager with 100% free physical memory frames.
        """
        self.sequences.clear()
        for i in range(self.total_physical_blocks):
            block = self.physical_blocks[i]
            block.ref_count = 0
            block.is_free = True
            block.token_ids.clear()
        self.free_block_pool = list(range(self.total_physical_blocks))
