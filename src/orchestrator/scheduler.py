"""Schedule a list of KernelGroups into an executable Runtime.

Inputs:

  - `ops`: the `Operations` builder that produced the IR Program. Its
    `uploads` dict (populated via `ops.from_numpy(...)`) supplies the
    numpy payloads for every input tensor; its `tensors` registry names
    every tensor the user declared (inputs, op outputs, and views like
    transpose/reshape).
  - `groups`: the topo-ordered tuple of `KernelGroup`s produced by
    `assemble(fuse(program))`. Each carries everything its `Dispatch`
    needs (bindings, dims, grid, threads, MSL kernel).

Output: a `Runtime` whose fragments, when executed against a fresh env
dict, run the whole program end-to-end:

    [FromNumpy ...] uploads inputs
    [Allocate ...]  one per unique non-input buffer_key, in first-use order
    [Kernel ...]    primes MSL source cache (matches the bench pattern)
    [Dispatch ...]  one per group, in topo order
        ↳ after each Dispatch, any buffer_key whose last use was this
          dispatch is Downloaded (if it backs a named non-input tensor)
          and Freed. Inputs that aren't named outputs are freed too once
          their last reader has run.

Materialization rule: the user gets back numpy for every declared
non-input tensor whose buffer was actually allocated. A tensor whose
producing op was absorbed by fusion (e.g. `C = a@b` followed by
`D = relu(C)` with a register epilogue) is never materialized — its
`buffer_key` never appears in any `group.bindings`, so it has no buffer
to download. View tensors (transpose / reshape views) ARE downloaded
when their backing buffer is materialized — the bytes come from the
storage owner and are reshaped to the view's logical shape. Note: for
a transpose / non-contiguous view, this returns the buffer's bytes
reinterpreted, not a true logical transpose; a stride-aware Download
path is the planned follow-up.

Free is end-of-life per buffer: download (if needed), then free, the
moment no later group touches the key. Eager — no reuse, no aliasing.
"""

from __future__ import annotations

import numpy as np

from orchestrator.builder import Operations
from orchestrator.dag import build_dag
from orchestrator.kernel_group import KernelGroup
from runtime import (
    Allocate,
    Dispatch,
    Download,
    Free,
    FromNumpy,
    Runtime,
    RuntimeFragment,
)


def schedule(
    ops: Operations,
    groups: tuple[KernelGroup, ...],
    *,
    profile: bool = False,
) -> Runtime:
    """Walk `groups` in order and produce a `Runtime` covering upload,
    allocate, dispatch, download, and free.

    `profile=True` plumbs a `time_key=f"t_{i}"` into each Dispatch so the
    caller can read per-kernel ms latencies out of the env. Off by
    default — `Dispatch.time_key=None` is a no-op timing-wise.
    """
    program = ops.build()
    dag = build_dag(program)
    tensors = ops.tensors  # name -> Tensor (includes views)
    uploads = ops.uploads  # name -> np.ndarray (owner-only by from_numpy guard)

    # Validate uploads cover every truly external input buffer.
    #
    # dag.inputs is tensor names that some op reads but no op produces.
    # That includes views (e.g. a metadata-only reshape of a reduction's
    # 1D output) — those alias an internally-produced buffer and do NOT
    # need uploading. Filter to buffer_keys not produced by any op.
    produced_buffer_keys = {op.out.buffer_key for op in program}
    required_keys: set[str] = set()
    for n in dag.inputs:
        bk = tensors[n].buffer_key
        if bk not in produced_buffer_keys:
            required_keys.add(bk)
    provided_keys = set(uploads)
    missing = required_keys - provided_keys
    extra = provided_keys - required_keys
    if missing:
        raise ValueError(
            f"schedule: input buffers {sorted(missing)} have no from_numpy upload"
        )
    if extra:
        raise ValueError(
            f"schedule: uploads {sorted(extra)} do not correspond to any program input"
        )

    # buffer_key -> Tensor used for sizing (Allocate.nbytes) and downloads.
    # Op outputs are owners by construction; inputs are owners by from_numpy
    # guard; views share an owner's buffer_key and never need separate sizing.
    nbytes_for_key: dict[str, int] = {}
    for op in program:
        nbytes_for_key[op.out.buffer_key] = op.out.nbytes
    for name in uploads:
        t = tensors[name]
        nbytes_for_key[t.buffer_key] = t.nbytes

    input_buffer_keys = {tensors[n].buffer_key for n in uploads}

    # Tensors the user wants dumped back: every declared name except the
    # inputs they already have as numpy. Includes views — each view name
    # gets its own Download reading from its `buffer_key` with its
    # `shape`. We only emit a Download for names whose backing buffer is
    # actually allocated (i.e. some group writes to that buffer_key);
    # otherwise the tensor was fused away and there's nothing to read.
    materialized_keys: set[str] = set(input_buffer_keys)
    for g in groups:
        for k in g.bindings:
            materialized_keys.add(k)

    output_names_per_key: dict[str, list[str]] = {}
    for name, t in tensors.items():
        if name in uploads:
            continue
        if t.buffer_key not in materialized_keys:
            continue
        output_names_per_key.setdefault(t.buffer_key, []).append(name)

    # Last-use index per buffer_key across all dispatches. After
    # dispatch i, any binding whose last_use is i is done — download
    # (if named output) and free.
    last_use: dict[str, int] = {}
    for i, g in enumerate(groups):
        for k in g.bindings:
            last_use[k] = i

    fragments: list[RuntimeFragment] = []

    # 1. Uploads (FromNumpy creates the Buffer + writes data).
    for name, arr in uploads.items():
        t = tensors[name]
        fragments.append(FromNumpy(name=t.buffer_key, source=arr))

    # 2. Allocates for every non-input buffer_key, in first-use order.
    #    Also prime each kernel's MSL render cache off the timed path.
    seen: set[str] = set(input_buffer_keys)
    for g in groups:
        for k in g.bindings:
            if k not in seen:
                fragments.append(Allocate(name=k, nbytes=nbytes_for_key[k]))
                seen.add(k)
        fragments.append(g.kernel)  # primes ctx.source via Kernel.execute

    # 3. Dispatches with download+free at each buffer's last use.
    #
    # Download(name=k, into=k) replaces env[k] (the Buffer) with the
    # ndarray; the prior Buffer goes out of scope and releases its
    # MTLBuffer automatically — i.e. the owner-name Download IS the free.
    # So the emission rule per dying buffer_key `k` is:
    #
    #   (a) emit Downloads for every *view* name sharing `k` first
    #       (while the Buffer is still at env[k]),
    #   (b) if `k` itself is a named output (op-output owner), emit its
    #       Download last — that releases the buffer,
    #   (c) otherwise (input being retired after last consumer, or any
    #       buffer with view-only downloads), emit an explicit Free(k).
    freed: set[str] = set()
    for i, g in enumerate(groups):
        time_key = f"t_{i}" if profile else None
        fragments.append(
            Dispatch(
                kernel=g.kernel,
                bindings=g.bindings,
                dims=g.dims,
                grid=g.grid,
                threads=g.threads,
                time_key=time_key,
            )
        )
        for k in set(g.bindings):
            if k in freed:
                continue
            if last_use.get(k) != i:
                continue
            names = output_names_per_key.get(k, ())
            view_names = [n for n in names if n != k]
            owner_download = k in names  # i.e. k is itself a named output
            for vn in view_names:
                t = tensors[vn]
                # Views read the owner's flat row-major storage via the
                # view's strides — gather assembled host-side.
                fragments.append(
                    Download(
                        name=t.buffer_key,
                        shape=t.shape,
                        dtype=np.float32,
                        into=t.name,
                        row_stride=t.row_stride,
                        col_stride=t.col_stride,
                    )
                )
            if owner_download:
                t = tensors[k]
                # Owner is row-major contiguous by construction (op outputs
                # always are); plain memcpy via the no-stride path.
                fragments.append(
                    Download(
                        name=t.buffer_key,
                        shape=t.shape,
                        dtype=np.float32,
                        into=t.name,
                    )
                )
            else:
                fragments.append(Free(name=k))
            freed.add(k)

    return Runtime(tuple(fragments))
