"""Threadgroup-memory lifetime analysis.

First pass in the memory-aliasing pipeline. Sits between fusion's
output (`KernelGroup`s) and the scheduler, and answers one question
per kernel: for every named threadgroup-memory object touched anywhere
inside the kernel, which top-level fragment index first writes/reads
it (`birth`), and which top-level fragment is the last to touch it
(`death`)?

The downstream aliasing pass consumes these intervals to decide which
tgmem names can share the same physical threadgroup buffer slot — two
names whose [birth, death] intervals don't overlap are eligible.
That rewriter is not in this module; this pass is observation only.

Walking rules:

  - Iterate `kernel.fragments` in order; the index of the *top-level*
    fragment is what becomes `birth` / `death`. A nested access (e.g.
    a tg-tile read from `MatmulComputeFragment` inside
    `MatmulMainloopFragment`) is attributed to the enclosing
    top-level index — for aliasing purposes the whole loop body is
    "live" at that step.
  - A fragment's `tgmem_accesses` is the canonical declaration. A
    fragment may additionally expose `sub_fragments`; the walker
    recurses into it and merges any accesses found there. The
    matmul mainloop is the existing user of this hook.
  - `size_floats` / `shape` on the output entry record the *max*
    footprint seen across all accesses to that name. Aliasing needs
    the worst-case size to provision the shared slot.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Iterable

from compute.fragments import AliasFragment, TgmemAccess
from orchestrator.kernel_group import KernelGroup
from runtime.program import Kernel


TGMEM_CAP_BYTES = 32_768
"""Apple-silicon threadgroup memory limit. A kernel whose post-alias
footprint exceeds this won't compile under MSL. Fusion uses this as
the hard ceiling for its budget gate; the aliasing pass reports
against it; future autotuner will tune hyperparams to fit under it."""


class TgmemOverflowError(RuntimeError):
    """Raised when a kernel's threadgroup-memory footprint cannot be
    brought under `TGMEM_CAP_BYTES`. Distinct from a Metal compile
    failure so the orchestrator can catch it cleanly and either retry
    (blacklist + re-fuse) or surface it as an actionable error."""


@dataclass(frozen=True)
class TgmemLifetime:
    """Birth/death of one named threadgroup-memory object inside one
    kernel. Indices refer to positions in `kernel.fragments`."""

    name: str
    birth: int
    death: int
    size_floats: int
    shape: tuple[int, int]


def _walk_accesses(fragment: object) -> Iterable[TgmemAccess]:
    """Yield every `TgmemAccess` reachable from `fragment` — its own
    `tgmem_accesses` plus any inside `sub_fragments`, recursively."""
    own = getattr(fragment, "tgmem_accesses", ())
    yield from own
    for sub in getattr(fragment, "sub_fragments", ()):
        yield from _walk_accesses(sub)


def compute_lifetimes(kernel: Kernel) -> tuple[TgmemLifetime, ...]:
    """Birth/death indices over `kernel.fragments`. One entry per
    distinct tgmem name; emission order matches first-touch order."""
    order: list[str] = []
    birth: dict[str, int] = {}
    death: dict[str, int] = {}
    size_floats: dict[str, int] = {}
    shape: dict[str, tuple[int, int]] = {}

    for idx, fragment in enumerate(kernel.fragments):
        for access in _walk_accesses(fragment):
            if access.name not in birth:
                order.append(access.name)
                birth[access.name] = idx
                size_floats[access.name] = access.size_floats
                shape[access.name] = access.shape
            death[access.name] = idx
            if access.size_floats > size_floats[access.name]:
                size_floats[access.name] = access.size_floats
                shape[access.name] = access.shape

    return tuple(
        TgmemLifetime(
            name=name,
            birth=birth[name],
            death=death[name],
            size_floats=size_floats[name],
            shape=shape[name],
        )
        for name in order
    )


def compute_group_lifetimes(group: KernelGroup) -> tuple[TgmemLifetime, ...]:
    """Convenience: lifetimes for the kernel inside a `KernelGroup`."""
    return compute_lifetimes(group.kernel)


@dataclass(frozen=True)
class FusionBudget:
    """Static threadgroup-memory footprint of one kernel as the MSL
    runtime actually allocates it — the bytes that count against the
    32 KiB Apple-silicon cap.

    MSL `threadgroup` decls are alive for the full kernel duration;
    there is no time-varying peak. After the aliasing pass, the
    footprint is exactly `sum(slot.size_floats)` over the alias
    plan's slots: each slot's storage is sized to its largest tenant
    and survives the entire kernel.

      - `n_slots` — number of distinct tg decls emitted (= one per
        `AliasSlot`). Lower bound: the maximum number of tgmem
        objects simultaneously alive at any fragment index.
      - `size_floats` — total fp32 floats allocated across all
        slots. Multiply by 4 for bytes (`size_bytes`).

    This is computed via `compute_alias_map(lifetimes)`; it matches
    the post-alias MSL allocation exactly, by construction."""

    n_slots: int
    size_floats: int

    @property
    def size_bytes(self) -> int:
        return self.size_floats * 4  # fp32 throughout v0


def compute_budget(lifetimes: tuple[TgmemLifetime, ...]) -> FusionBudget:
    """Run the aliasing planner over `lifetimes` and report what the
    rewriter will actually allocate. Equals `compute_alias_map(...)`'s
    `total_size_floats` and `n_slots` — no estimation gap. Use this
    to predict whether a fused kernel will fit under the tg cap."""
    alias_map = compute_alias_map(lifetimes)
    return FusionBudget(
        n_slots=alias_map.n_slots,
        size_floats=alias_map.total_size_floats,
    )


def compute_group_budget(group: KernelGroup) -> FusionBudget:
    """Convenience: budget for the kernel inside a `KernelGroup`."""
    return compute_budget(compute_lifetimes(group.kernel))


# ---------------------------------------------------------------------------
# Aliasing plan: which original tgmem names share a physical slot.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasSlot:
    """One physical threadgroup-memory slot under the alias plan.

    A slot hosts one or more original tgmem objects whose [birth,
    death] intervals are pairwise disjoint — by construction, only
    one tenant is alive at any fragment index, so they can share the
    same physical storage. The slot is sized to the largest tenant
    and shaped accordingly (the rewriter will declare one
    `threadgroup` array per slot and rebind references)."""

    name: str
    tenants: tuple[str, ...]
    size_floats: int
    shape: tuple[int, int]


@dataclass(frozen=True)
class AliasMap:
    """Per-kernel alias plan — the output of `compute_alias_map`. The
    `slots` tuple is the canonical form; `name_to_slot` is the lookup
    a rewriter uses to substitute tgmem references."""

    slots: tuple[AliasSlot, ...]

    @property
    def name_to_slot(self) -> dict[str, str]:
        return {tenant: slot.name for slot in self.slots for tenant in slot.tenants}

    @property
    def total_size_floats(self) -> int:
        """Static tg allocation under this plan — sum of per-slot
        sizes. The aliasing win against the pre-rewrite footprint is
        `sum(lt.size_floats for lt in lifetimes) - total_size_floats`."""
        return sum(s.size_floats for s in self.slots)

    @property
    def n_slots(self) -> int:
        return len(self.slots)


def compute_alias_map(lifetimes: tuple[TgmemLifetime, ...]) -> AliasMap:
    """Birth-order greedy alias matching with size promotion.

    Process lifetimes in birth order. For each `lt`, find every
    disjoint slot (one whose last tenant died before `lt` is born —
    by birth-order processing, this single check guarantees
    pairwise disjointness with the slot's whole tenant chain). Pick
    the one that minimizes the NEW slot size after placement,
    where slot size = `max(tenant.size_floats)` across all tenants:

      - Cost of placing `lt` in slot S = `max(S.size, lt.size) - S.size`.
        Zero when `lt.size <= S.size` (purely free reuse); else `lt`
        grows the slot to its own size.
      - Cost of opening a new slot = `lt.size_floats`.

    Ranking key = `(growth, current, -last.death)` where
    `growth = max(0, lt.size - S.size)`:

      1. Minimize growth — total tgmem only ever grows on insertion,
         and the growth delta on the chosen slot IS the total delta.
         Slots where `lt` already fits (`growth = 0`) always beat
         slots that need to grow.
      2. Tiebreak on smallest current size — in the zero-growth
         case, this picks the tightest fit and preserves larger
         slots for items that genuinely need them. In the
         positive-growth case all candidates have the same growth
         only if their `current` is identical too, so this
         tiebreak is moot there.
      3. Final tiebreak: most-recently-dead last tenant — packs
         slots densely in time.

    Unlike the previous birth-order BFD, this version DOES NOT
    require `owner.size >= lt.size`. The rewriter
    (`apply_alias_map`) synthesizes a flat `threadgroup float
    _<owner>_storage[<max_size>];` per multi-tenant slot and emits
    an `AliasFragment` for EVERY tenant including the owner. So the
    owner's decl no longer has to be big enough to host later
    tenants — the synthesized storage is sized to the largest.

    Slot.name remains the first-born tenant's name (cosmetic
    anchor used to derive the storage symbol). Slot.shape is the
    first-born's shape (cosmetic; the storage is flat). Slot
    .size_floats is `max(tenant.size_floats)` — the actual
    allocation."""
    if not lifetimes:
        return AliasMap(slots=())

    slots: list[list[TgmemLifetime]] = []
    sizes: list[int] = []
    for lt in sorted(lifetimes, key=lambda lt: (lt.birth, lt.death)):
        best_idx: int | None = None
        best_key: tuple[int, int, int] | None = None
        for i, tenants in enumerate(slots):
            last = tenants[-1]
            if last.death >= lt.birth:
                continue
            current = sizes[i]
            growth = lt.size_floats - current if lt.size_floats > current else 0
            key = (growth, current, -last.death)
            if best_key is None or key < best_key:
                best_idx = i
                best_key = key
        if best_idx is not None:
            slots[best_idx].append(lt)
            if lt.size_floats > sizes[best_idx]:
                sizes[best_idx] = lt.size_floats
        else:
            slots.append([lt])
            sizes.append(lt.size_floats)

    return AliasMap(
        slots=tuple(
            AliasSlot(
                name=tenants[0].name,
                tenants=tuple(t.name for t in tenants),
                size_floats=size,
                shape=tenants[0].shape,
            )
            for tenants, size in zip(slots, sizes)
        )
    )


def compute_group_alias_map(group: KernelGroup) -> AliasMap:
    """Convenience: alias map for the kernel inside a `KernelGroup`."""
    return compute_alias_map(compute_lifetimes(group.kernel))


# ---------------------------------------------------------------------------
# Rewriter: apply an AliasMap to a Kernel.
# ---------------------------------------------------------------------------


# Canonical decl form: `threadgroup float <name>[<dim>][<dim>]...;` with
# any whitespace around tokens. One or more bracketed dim groups so 1D
# (scratch) and 2D (tiles) decls both parse cleanly. The match is
# anchored so unexpected formats raise rather than silently slip past.
_DECL_NAME_RE = re.compile(r"^\s*threadgroup\s+float\s+(\w+)\s*(?:\[[^\]]+\])+\s*;\s*$")


def _extract_decl_name(decl: str) -> str:
    """Pull the variable name out of a single tg decl string. Raises
    `ValueError` if the decl doesn't match the canonical form — we
    never silently drop a malformed decl."""
    m = _DECL_NAME_RE.match(decl)
    if m is None:
        raise ValueError(
            f"apply_alias_map: cannot parse threadgroup decl {decl!r}. "
            f"Expected `threadgroup float <name>[N]...;`."
        )
    return m.group(1)


def _is_tgmem_barrier(fragment: object) -> bool:
    """A barrier fragment we can use as the alias's sync predecessor.
    Currently only `BarrierFragment` qualifies (it exposes
    `is_tgmem_barrier=True`); duck-typed via `getattr` so future
    barrier variants slot in without import churn."""
    return getattr(fragment, "is_tgmem_barrier", False) is True


def _storage_name(owner_name: str) -> str:
    """Synthesized flat-storage symbol for a multi-tenant slot. The
    owner's natural name (`<owner_name>`) becomes a pointer alias
    onto this storage at the top of the kernel, freeing the
    storage symbol itself to be the underlying decl. Underscore
    prefix avoids collisions with user tile names (which by
    convention don't start with `_`)."""
    return f"_{owner_name}_storage"


def apply_alias_map(kernel: Kernel, alias_map: AliasMap) -> Kernel:
    """Rewrite `kernel` to apply an aliasing plan.

    Two kinds of slots:

      - **Trivial (1 tenant):** the tenant's original decl is
        kept verbatim; no `AliasFragment` is emitted. No change to
        the kernel's behavior for this name.
      - **Multi-tenant:** drop every tenant's original decl,
        synthesize one flat `threadgroup float
        _<owner>_storage[<max_size>];` sized to the slot's largest
        tenant, then emit an `AliasFragment` for EVERY tenant
        (including the owner) re-typing the storage to each
        tenant's 2D shape via a pointer cast.

    Per-tenant alias placement within a multi-tenant slot:

      - **First tenant (owner).** Prepended to the kernel body
        (insertion index `-1`) with `preceded_by_barrier=True` so
        no barrier is emitted — there is no prior data on the
        storage to sync against.
      - **Subsequent tenants.** Scan the seam (predecessor's
        death, this tenant's birth) for a tg barrier:
          - Found → place alias immediately after the *latest*
            such barrier with `preceded_by_barrier=True` (alias
            sits as close to first use as possible while reusing
            the existing sync).
          - None → place alias right before the new tenant's first
            touching fragment with `preceded_by_barrier=False`
            (the `AliasFragment` emits its own barrier).

    Predecessor for the barrier check is the IMMEDIATE birth-order
    prior tenant in the slot; `old_name` for every tenant is the
    synthesized storage symbol (every tenant casts onto the same
    underlying flat array).

    A decl whose name is neither kept nor dropped is unexpected
    (the lifetime pass missed a tgmem object) and raises loudly.

    Returns a new `Kernel`; the original is untouched."""
    if not alias_map.slots:
        return kernel

    lifetimes = compute_lifetimes(kernel)
    by_name = {lt.name: lt for lt in lifetimes}

    insertions: dict[int, list[AliasFragment]] = {}
    synthesized_decls: list[str] = []
    dropped_names: set[str] = set()
    kept_names: set[str] = set()

    for slot in alias_map.slots:
        if len(slot.tenants) == 1:
            # Trivial slot — leave the original decl in place.
            kept_names.add(slot.name)
            continue

        storage = _storage_name(slot.name)
        synthesized_decls.append(f"threadgroup float {storage}[{slot.size_floats}];")
        dropped_names.update(slot.tenants)

        prev_death = -1
        for i, tenant_name in enumerate(slot.tenants):
            lt = by_name[tenant_name]

            if i == 0:
                # Owner: prepend to kernel body; no barrier needed
                # (storage is freshly declared, holds no live data).
                preceded_by_barrier = True
                insert_idx = -1
            else:
                barrier_idx: int | None = None
                for j in range(prev_death + 1, lt.birth):
                    if _is_tgmem_barrier(kernel.fragments[j]):
                        barrier_idx = j  # keep updating → ends at latest
                if barrier_idx is not None:
                    preceded_by_barrier = True
                    insert_idx = barrier_idx
                else:
                    preceded_by_barrier = False
                    insert_idx = lt.birth - 1

            af = AliasFragment(
                old_name=storage,
                new_name=tenant_name,
                new_shape=lt.shape,
                preceded_by_barrier=preceded_by_barrier,
            )
            insertions.setdefault(insert_idx, []).append(af)
            prev_death = lt.death

    new_fragments: list = list(insertions.get(-1, ()))
    for i, frag in enumerate(kernel.fragments):
        new_fragments.append(frag)
        for af in insertions.get(i, ()):
            new_fragments.append(af)

    new_decls: list[str] = []
    for decl in kernel.ctx.threadgroup_decls:
        name = _extract_decl_name(decl)
        if name in kept_names:
            new_decls.append(decl)
        elif name in dropped_names:
            continue
        else:
            raise ValueError(
                f"apply_alias_map: threadgroup decl {decl!r} declares "
                f"name {name!r} which is not present in any AliasSlot. "
                f"This means lifetime analysis missed the tgmem object."
            )
    new_decls.extend(synthesized_decls)

    new_ctx = dataclasses.replace(kernel.ctx, threadgroup_decls=tuple(new_decls))
    return Kernel(fragments=tuple(new_fragments), ctx=new_ctx)


def apply_group_alias_map(group: KernelGroup, alias_map: AliasMap) -> KernelGroup:
    """Convenience: apply `alias_map` to the kernel inside a
    `KernelGroup`, returning a new group with the rewritten kernel
    and all other fields preserved."""
    return dataclasses.replace(group, kernel=apply_alias_map(group.kernel, alias_map))


def alias_group(group: KernelGroup) -> KernelGroup:
    """End-to-end aliasing pass for one `KernelGroup`: compute
    lifetimes, derive the alias plan, rewrite the kernel. This is
    the single seam the pipeline (`api.run`) plugs into between
    assembly and scheduling — everything else in this module is
    internal machinery exposed for tests and introspection."""
    lifetimes = compute_lifetimes(group.kernel)
    alias_map = compute_alias_map(lifetimes)
    return apply_group_alias_map(group, alias_map)


__all__ = [
    "AliasMap",
    "AliasSlot",
    "FusionBudget",
    "TGMEM_CAP_BYTES",
    "TgmemLifetime",
    "TgmemOverflowError",
    "alias_group",
    "apply_alias_map",
    "apply_group_alias_map",
    "compute_alias_map",
    "compute_budget",
    "compute_group_alias_map",
    "compute_group_budget",
    "compute_group_lifetimes",
    "compute_lifetimes",
]
