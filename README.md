# Metellus

Metellus is a fragment-composition kernel compiler for Apple Silicon. It takes
small tensor programs — matmuls, elementwise ops, reductions, and shape/copy ops
— and emits fused Metal Shading Language (MSL) kernels.

Instead of maintaining one giant kernel template per fused-op shape, Metellus
assembles kernels from reusable hand-written fragments:

- cooperative tile loads and stores
- simdgroup matmul mainloops
- tiled elementwise chains and epilogues
- scalar value transforms
- cooperative reductions

Those fragments are selected and composed by a fusion pipeline over a small SSA
IR.

## Scope

- **Target:** Apple Silicon / Metal
- **Precision:** fp32
- **Execution:** forward-only
- **Primitive ops:** matmul, elementwise, reduction, shape/copy
- **Views:** transpose and reshape are represented with tensor metadata when
  possible, not eager kernels

Built with AI-assisted development (Claude Code). The architecture, design
decisions, and verification are my own.

## Quick start

```bash
./scripts/setup.sh     # install deps and build the native extension
./scripts/test.sh      # run the test suite
./scripts/bench.sh     # run correctness/performance benchmarks

uv run demo.py         # play around with operations of your own - reference other examples and the docstring for guidance
```

Requirements:

- macOS on Apple Silicon
- Xcode Command Line Tools
- `uv`
- MLX-compatible environment

Tests verify correctness against MLX, using bit-exact checks where reduction
order matches and fp32 tolerances elsewhere.

## How it works

The runtime path is:

```text
Operations builder
  → graph fusion
  → assembly
  → threadgroup-memory aliasing
  → scheduling
  → Metal execution
```

The end-to-end entrypoint is `src/api/run.py`:

```text
ops.build()
  → fuse(...)
  → assemble(...)
  → alias_group(...)
  → schedule(...)
```

### 1. IR and tensor metadata

Files:

- `src/orchestrator/builder.py`
- `src/orchestrator/ir.py`

`Operations` builds a small SSA program over `Tensor` values. Matmul,
elementwise, reduction, and shape/copy operations are real IR nodes.

Logical views such as transpose and reshape are represented with shape, stride,
and `buffer_key` metadata instead of eager copy kernels when possible. This means
a computation like `Q @ Kᵀ` can compile to one matmul kernel without a separate
transpose pass.

### 2. Vertex graph fusion

Files:

- `src/orchestrator/graph.py`
- `src/orchestrator/fusion.py`

The program starts as one `Vertex` per IR op. Fusion runs priority-ordered rewrite
rules that collapse legal subgraphs into fused vertices.

Current fused shapes include:

- elementwise chains
- matmul and reduction epilogues
- elementwise prologues folded into anchor loads
- multi-producer convergent matmuls
- shared-input diamond matmuls

After each candidate rewrite, the fuser estimates post-alias threadgroup-memory
usage. If a fusion would exceed the 32KB Apple Silicon threadgroup-memory cap,
that fusion signature is blacklisted and fusion is retried deterministically.

### 3. Assembly

Files:

- `src/orchestrator/assembly/`
- `src/orchestrator/assembly/decision.py`

Each fused `Vertex` is classified into a `FusionStrategy`, then lowered by a
strategy-specific assembler.

Assembly handles:

- standalone matmul, reduction, elementwise, and shape kernels
- matmul register epilogues
- matmul threadgroup-tile epilogues
- elementwise prologues into matmul/reduction loads
- pure elementwise chains
- two-matmul multi-anchor kernels

The output is a `KernelGroup`: generated fragments, runtime buffer bindings,
launch dimensions, grid shape, and threadgroup size.

### 4. Fragment codegen

Files:

- `src/compute/`
- `src/memory/`
- `src/runtime/program.py`

Kernels are composed from reusable fragments. The codegen engine renders a
fragment list plus a `CodegenContext` into MSL.

Examples of fragments include:

- `TgLoadFragment` and `TgStoreFragment`
- matmul setup/mainloop/compute/store fragments
- tiled elementwise chain fragments
- reduction setup/compute/store fragments
- alias fragments inserted by the threadgroup-memory aliasing pass

### 5. Threadgroup-memory aliasing

File:

- `src/orchestrator/aliasing.py`

After assembly, the aliasing pass computes lifetimes for named threadgroup-memory
objects such as `A_tile`, `B_tile`, and `C_tile`.

Lifetime-disjoint tiles are rewritten to share one physical `threadgroup`
allocation via typed pointer aliases. This is the pass that lets larger fused
kernels fit under Metal's 32KB threadgroup-memory limit.

### 6. Scheduling and runtime

Files:

- `src/orchestrator/scheduler.py`
- `src/api/run.py`

The scheduler turns `KernelGroup`s into an executable runtime plan:

- upload input arrays
- allocate device buffers
- compile/cache MSL kernels
- dispatch kernels in topological order
- download materialized outputs
- free buffers at last use

Intermediate tensors absorbed by fusion are not materialized unless they survive
as real runtime bindings.

## Good files to read first

- `src/api/run.py` — end-to-end pipeline wiring
- `src/orchestrator/fusion.py` — fusion rules and budget gating
- `src/orchestrator/assembly/__init__.py` — strategy dispatch
- `src/orchestrator/assembly/decision.py` — fused-vertex classification
- `src/orchestrator/aliasing.py` — threadgroup-memory lifetime analysis and aliasing
