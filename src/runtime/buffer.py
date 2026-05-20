from __future__ import annotations

import numpy as np

import metal_backend


class Buffer:
    """Owning handle to a persistent Metal buffer (shared storage).

    Lifetime is tied to the Python object. When the last reference drops,
    __del__ releases the underlying MTLBuffer through the backend registry.
    Use the .handle attribute to bind into metal_backend.run_kernel.
    """

    __slots__ = ("_handle", "_nbytes", "_released")

    def __init__(self, nbytes: int) -> None:
        if nbytes <= 0:
            raise ValueError(f"Buffer nbytes must be > 0; got {nbytes}")
        self._handle = metal_backend.create_buffer(nbytes)
        self._nbytes = nbytes
        self._released = False

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "Buffer":
        contiguous = np.ascontiguousarray(array)
        buffer = cls(contiguous.nbytes)
        metal_backend.write_buffer(buffer._handle, contiguous)
        return buffer

    @classmethod
    def zeros(cls, nbytes: int) -> "Buffer":
        buffer = cls(nbytes)
        metal_backend.fill_buffer(buffer._handle, 0)
        return buffer

    @property
    def handle(self) -> int:
        if self._released:
            raise RuntimeError("Buffer has been released")
        return self._handle

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def write(self, array: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(array)
        if contiguous.nbytes > self._nbytes:
            raise ValueError(
                f"write: source ({contiguous.nbytes} bytes) exceeds buffer ({self._nbytes} bytes)"
            )
        metal_backend.write_buffer(self.handle, contiguous)

    def read_into(self, out: np.ndarray) -> np.ndarray:
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("read_into: destination must be C-contiguous")
        if out.nbytes > self._nbytes:
            raise ValueError(
                f"read_into: destination ({out.nbytes} bytes) exceeds buffer ({self._nbytes} bytes)"
            )
        metal_backend.read_buffer(self.handle, out)
        return out

    def numpy(self, shape: tuple[int, ...], dtype: np.dtype | type) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        return self.read_into(out)

    def fill(self, byte_value: int) -> None:
        if not 0 <= byte_value <= 255:
            raise ValueError(f"fill byte_value must be in [0, 255]; got {byte_value}")
        metal_backend.fill_buffer(self.handle, byte_value)

    def release(self) -> None:
        if not self._released:
            metal_backend.release_buffer(self._handle)
            self._released = True

    def __del__(self) -> None:
        # __del__ can run during interpreter shutdown; swallow anything the
        # backend might raise so we don't trigger spurious "Exception ignored"
        # noise from already-torn-down state.
        try:
            self.release()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "released" if self._released else f"handle={self._handle}"
        return f"Buffer(nbytes={self._nbytes}, {state})"
