from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Array:
    dtype: str
    size: int
    _data: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError("size must be >= 0")
        if not self._data:
            self._data = [0] * self.size
        elif len(self._data) != self.size:
            raise ValueError("data length must equal size")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Any:
        return self._data[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self._data[idx] = value

    def to_list(self) -> list[Any]:
        return list(self._data)

    def to_bytes(self) -> bytes:
        return bytes(str(self._data), encoding="utf-8")


def from_list(values: list[Any], dtype: str = "dynamic") -> Array:
    return Array(dtype=dtype, size=len(values), _data=list(values))
