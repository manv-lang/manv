from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryView:
    data: bytes | bytearray
    readonly: bool = False
    shape: tuple[int, ...] | None = None
    strides: tuple[int, ...] | None = None

    def cast(self, _dtype: str) -> "MemoryView":
        return self

    def slice(self, start: int, end: int) -> "MemoryView":
        return MemoryView(self.data[start:end], readonly=self.readonly, shape=self.shape, strides=self.strides)


def memoryview(obj: bytes | bytearray) -> MemoryView:
    return MemoryView(obj, readonly=isinstance(obj, bytes), shape=(len(obj),), strides=(1,))
