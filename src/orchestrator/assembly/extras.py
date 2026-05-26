"""Bookkeeping for extra device buffers a fused kernel pulls in.

Every per-strategy template starts with a fixed base of buffer slots
(matmul: A, B, C; reduction: in, out; elementwise: X, out). Prologue /
epilogue chains may reference additional tensor operands (bias rows,
broadcast vectors, etc.); these need extra slots, env bindings, and
runtime row/col-stride dim values. `Extras` deduplicates by storage
owner (`buffer_key`) so an aliased view binds to the underlying buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orchestrator.ir import Tensor


@dataclass
class Extras:
    buffers: list[str] = field(default_factory=list)
    bindings: list[str] = field(default_factory=list)
    dim_names: list[str] = field(default_factory=list)
    dim_values: list[int] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    base_slots_by_key: dict[str, int] = field(default_factory=dict)

    def add_tensor(self, t: Tensor, base_slot: int) -> int:
        """Register `t` as an input. Returns the buffer slot index. The
        MSL param name and env binding both use `buffer_key` so an
        aliased view binds to its storage owner; already-added
        buffer_keys deduplicate."""
        key = t.buffer_key
        if key in self.base_slots_by_key:
            return self.base_slots_by_key[key]
        if key in self.seen:
            return self.bindings.index(key) + base_slot
        slot = base_slot + len(self.bindings)
        self.buffers.append(f"device const float* {key} [[buffer({slot})]]")
        self.bindings.append(key)
        self.seen.add(key)
        return slot
