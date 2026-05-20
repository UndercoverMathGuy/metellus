from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

import metal_backend
from compute.fragments import CodegenContext, CodegenEngine
from runtime.buffer import Buffer


Env = dict[str, Any]
"""A program's mutable state: a flat dict keyed by user-chosen names. Holds
Buffers (allocated by Allocate/FromNumpy), downloaded ndarrays (written by
Download), and optional per-dispatch timings (when Dispatch is given a
`time_key`). Fragments share state through this dict; no other plumbing."""


class RuntimeFragment(Protocol):
    """Host-side program step. Mutates the Env when executed."""

    def execute(self, env: Env) -> None: ...


@dataclass(frozen=True)
class Allocate:
    name: str
    nbytes: int

    def execute(self, env: Env) -> None:
        if self.name in env:
            raise KeyError(f"Allocate: key {self.name!r} already in env")
        env[self.name] = Buffer(self.nbytes)


@dataclass(frozen=True)
class Free:
    name: str

    def execute(self, env: Env) -> None:
        env.pop(self.name).release()


@dataclass(frozen=True)
class Upload:
    name: str
    source: np.ndarray

    def execute(self, env: Env) -> None:
        env[self.name].write(self.source)


@dataclass(frozen=True)
class FromNumpy:
    """Allocate sized to the array and upload in one step."""

    name: str
    source: np.ndarray

    def execute(self, env: Env) -> None:
        if self.name in env:
            raise KeyError(f"FromNumpy: key {self.name!r} already in env")
        env[self.name] = Buffer.from_numpy(self.source)


@dataclass(frozen=True)
class Download:
    """Read a buffer into a fresh ndarray; stash it in env[into].

    When `row_stride` / `col_stride` are None (the default), the read is a
    plain memcpy of `shape`-many elements from the owner's row-major
    storage — used for op-output owners and contiguous inputs.

    When strides are provided, the buffer is treated as a flat element
    pool and the destination is gathered via the view's strides — used
    by the scheduler to materialize transpose / reshape views that
    alias a row-major owner. Strides are in *elements* (not bytes).
    1D shapes use `row_stride` only; 2D shapes use both."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype | type
    into: str
    row_stride: int | None = None
    col_stride: int | None = None

    def execute(self, env: Env) -> None:
        buf = env[self.name]
        if self.row_stride is None:
            env[self.into] = buf.numpy(self.shape, self.dtype)
            return
        itemsize = np.dtype(self.dtype).itemsize
        nelem = buf.nbytes // itemsize
        flat = buf.numpy((nelem,), self.dtype)
        strides_bytes: tuple[int, ...]
        if len(self.shape) == 2:
            strides_bytes = (
                self.row_stride * itemsize,
                (self.col_stride or 0) * itemsize,
            )
        elif len(self.shape) == 1:
            strides_bytes = (self.row_stride * itemsize,)
        else:
            raise ValueError(
                f"Download with strides supports 1D/2D shapes only; got {self.shape}"
            )
        viewed = np.lib.stride_tricks.as_strided(
            flat,
            shape=self.shape,
            strides=strides_bytes,
            writeable=False,
        )
        env[self.into] = np.ascontiguousarray(viewed)


@dataclass(frozen=True)
class Fill:
    name: str
    byte_value: int = 0

    def execute(self, env: Env) -> None:
        env[self.name].fill(self.byte_value)


class Kernel:
    """Compile a kernel from fragments + ctx. The Kernel object itself is the
    handle Dispatch refers to (not a string id). Rendering is lazy and cached
    on the instance — putting Kernel in a program just primes the cache."""

    def __init__(self, fragments: tuple[Any, ...], ctx: CodegenContext) -> None:
        self.fragments = tuple(fragments)
        self.ctx = ctx
        self._source: str | None = None
        self._dims_slot_index = (
            ctx.dims_buffer_index
            if ctx.dims_buffer_index is not None
            else len(ctx.buffers)
        )

    @property
    def source(self) -> str:
        if self._source is None:
            self._source = CodegenEngine().render(list(self.fragments), self.ctx)
        return self._source

    @property
    def function_name(self) -> str:
        return self.ctx.function_name

    @property
    def dims_slot_index(self) -> int:
        return self._dims_slot_index

    @property
    def dim_names(self) -> tuple[str, ...]:
        return self.ctx.dims

    def execute(self, env: Env) -> None:
        _ = self.source


@dataclass(frozen=True)
class Dispatch:
    """Launch a Kernel. `bindings` are env keys (buffers) in MSL-slot order,
    matching `kernel.ctx.buffers`. `dims` are runtime values for the kernel's
    declared dim names; they're packed into a fresh dims buffer and inserted
    at `kernel.dims_slot_index`. If `time_key` is set, the GPU time_ms is
    stored at env[time_key] (overwriting any prior value).
    """

    kernel: Kernel
    bindings: tuple[str, ...]
    dims: tuple[int, ...]
    grid: tuple[int, int, int]
    threads: tuple[int, int, int]
    time_key: str | None = None

    def execute(self, env: Env) -> None:
        if len(self.dims) != len(self.kernel.dim_names):
            raise ValueError(
                f"Dispatch: expected {len(self.kernel.dim_names)} dim values "
                f"({self.kernel.dim_names}), got {len(self.dims)}"
            )
        slots: list[Buffer] = [env[name] for name in self.bindings]
        dims_buf = Buffer.from_numpy(np.array(self.dims, dtype=np.uint32))
        slots.insert(self.kernel.dims_slot_index, dims_buf)
        result = metal_backend.run_kernel(
            self.kernel.source,
            self.kernel.function_name,
            slots,
            self.grid,
            self.threads,
        )
        if self.time_key is not None:
            env[self.time_key] = float(result["time_ms"])


@dataclass(frozen=True)
class Runtime:
    """An ordered tuple of RuntimeFragments. Run against a fresh or
    pre-existing Env (a plain dict); the same Env is returned so the caller
    can pull out buffers, downloaded ndarrays, and timings by key."""

    fragments: tuple[RuntimeFragment, ...] = field(default_factory=tuple)

    def run(self, env: Env | None = None) -> Env:
        if env is None:
            env = {}
        for fragment in self.fragments:
            fragment.execute(env)
        return env

    def __add__(self, other: "Runtime") -> "Runtime":
        return Runtime(self.fragments + other.fragments)
