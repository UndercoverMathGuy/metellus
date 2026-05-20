# Metal Kernel Style Guide

Generated compute code should be emitted as operative sections first. Full Metal kernels are packaging, not compute APIs.

## Canonical names

Every packaged 2D tile kernel should define these names before invoking compute sections:

- `tg`: `uint2` threadgroup id, equivalent to `threadgroup_position_in_grid`.
- `lid`: `uint2` local thread id, equivalent to `thread_position_in_threadgroup`.
- `flat_tid`: flattened local thread id. Use `lid.y * TG_X + lid.x`.
- `M`: output row count.
- `N`: output column count.
- `Y_rows`, `Y_cols`, `Y_stride`: secondary input shape/stride metadata for elementwise/broadcast sections.
- `C_rows`, `C_cols`, `C_stride`: condition input shape/stride metadata for `where`-style sections.

## Section contract

Compute modules should not emit kernel signatures, Metal includes, or argument buffers unless they are explicitly packaging helpers.

A section may assume:

- Canonical variables above exist.
- Required threadgroup tiles have already been declared.
- Required cooperative loads have already completed.
- `threadgroup_barrier(mem_flags::mem_threadgroup)` is inserted by the packaging/orchestration layer when needed.

## Elementwise tile convention

Elementwise sections operate on threadgroup tiles:

- `X_tile[row][col]`: primary input tile.
- `Y_tile[row][col]`: secondary input tile for binary ops.
- `Cond_tile[row][col]`: condition tile for ternary ops.
- `Out_tile[row][col]`: output tile.

Broadcast modes map as:

- `none`: `[tile_row][tile_col]`
- `scalar`: `[0][0]`
- `row`: `[0][tile_col]`
- `col`: `[tile_row][0]`

## Testing and benchmarking

Tests may package sections into complete kernels using `compute.scaffold`, but compute APIs should remain fragment-first so later fusion can compose them without reverse-engineering a standalone kernel.

## Packaging convention

Use named `KernelBlock`s for every meaningful section. A packaged kernel should read as an ordered list of operations, not as a pile of f-string spaghetti.

Preferred order:

- Declare threadgroup tiles.
- `block("load X_tile", ...)`
- Optional `block("load Y_tile", ...)`
- Optional `block("load Cond_tile", ...)`
- `barrier("inputs ready")`
- `block("apply elementwise", ...)`
- `barrier("output tile ready")`
- `block("store Out_tile", ...)`

Helpers should hide indexing and broadcast decisions behind names such as `primary_input_load`, `secondary_input_load`, `condition_input_load`, `required_tiles`, and `elementwise_sections`.
