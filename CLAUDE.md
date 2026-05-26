# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Metal Kernel Compiler

## What this is

A fragment-composition kernel compiler for Apple Silicon. User writes torch-style expressions; compiler decomposes to primitives, fuses what it can, generates MSL via fragment composition, runs through MLX. Forward-pass inference only. fp32 only. v0 target: WWDC 2026.

## Lineage

- **CUTLASS-inspired implementation:** kernels are composed from fragments (parameterized MSL-emitting Python functions). Templates parameterize over tile shape, SIMD-group layout, dtype, fusion hooks.
- **tinygrad-inspired primitive philosophy:** small set of irreducible primitives (matmul, elementwise, reduction). Surface ops decompose to primitives.
- **Inductor-inspired pipeline:** lazy graph capture → IR → decomposition → simplification → fusion → scheduling → codegen.

## Architecture

User torch-y API (mc.matmul, mc.relu, mc.softmax, ...)
↓ lazy construction
IR1: high-level DAG of Op/Value
↓ decomposition pass
IR2: primitives only (matmul, elementwise, reduction)
↓ algebraic simplification (pattern matching, no CAS)
IR2': simplified
↓ fusion pass (placement-based)
KernelGraph: list of kernels with explicit Op→Kernel assignment
↓ scheduling
DispatchPlan: kernels in topo order with buffer bindings
↓ codegen (fragment composition)
MSL strings → mx.fast.metal_kernel → execution

## Core insight: fusion is placement

Fragments are scope-aware code emitters, not pre-compiled kernels. "Fused" vs "unfused" is the same fragments emitted into different scopes:

- Unfused: matmul kernel writes to C, elementwise kernel reads from C
- Fused: elementwise fragment emitted inside matmul's epilogue scope, operates on register accumulators directly

No fusion-rewrite logic needed. The organizer decides *where* each op's fragments go.

## Fragment contracts

Every fragment is a Python function returning an MSL string. Fragments:

- Take parameters for input/output names (names of buffers/registers in surrounding scope)
- Assume certain variables are pre-declared by the template (flat_tid, sg_id, threadgroup buffers)
- Don't declare what they don't own
- Don't emit their own barriers (template's job)

Fragments compose via name binding. Templates concatenate fragments in order, threading names through. The MSL compiler enforces compatibility at compile time.

## What's built

- Memory fragments: cooperative load/store (device ↔ threadgroup), float4 vectorized
- Matmul: setup + simdgroup_matrix compute + writeback fragments. Template handles tile selection, split-K, aligned/unaligned paths, Hilbert curve traversal
- Elementwise: Op classes (Add, Mul, Relu, Exp, ...) with codegen methods. Template chains ops
- Reductions: last-axis sum/max/min with cooperative SIMD-group reduction
- Organizer: linear-chain fusion via placement decisions

## Design principles

1. **Trust the Apple MSL compiler** for low-level scheduling, register allocation, instruction-level pipelining. Don't hand-schedule.
2. **Own the global decisions:** tile sizes, threadgroup layouts, fusion choices, threadgroup memory layout, dtype precision.
3. **Fragments are dumb, templates are smart.** Fragments emit code; templates decide placement and parameters.
4. **Parameterize by what varies between callers, hardcode the rest.** SIMD group is always 32 (silicon). Tile shape varies.
5. **Verify against MLX as ground truth** for correctness. Benchmark vs MLX for perf.
6. **No premature generality.** Add fragments and parameters only when concrete callers need them.
7. **Fragments assume canonical preamble.** The packaging layer (scaffold or caller-supplied `ctx.preamble`) emits `tg`, `lid`, `flat_tid`, and dim names. Fragments use these directly — they never redeclare `uint global_idx = ...` or restate what the preamble already gave them. See `docs/kernel_style.md` for the full canonical name list.

## Performance posture

Matmul: competitive at small sizes (faster than MLX at 256), within 1.3-1.5x at large sizes. Memory-bandwidth-saturated at large sizes; further gains require different threadgroup memory strategy. Not chasing peak — chasing breadth.

## Pitch

MLX-quality kernels through a torch API by fragment composition. The architecture lets you scale kernel libraries (new ops, new dtypes, new hardware variants) with bounded engineering effort — fragments compose, templates parameterize, primitives decompose. v0 demonstrates the architecture works; v1 fills out coverage.

## Constraints I work under (Claude)

- Spec-driven: don't generate code without a clear spec from the user
- When debugging fails, surface the root cause; don't paper over with patch-fixes
