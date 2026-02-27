from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Tensor:
    data: list[Any]
    shape: tuple[int, ...]
    dtype: str = "f32"
    device: str = "cpu"

    def to_list(self) -> list[Any]:
        return list(self.data)


def elementwise_add(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return Tensor(data=[x + y for x, y in zip(a.data, b.data, strict=True)], shape=a.shape, dtype=a.dtype, device=a.device)


def reduce_sum(x: Tensor) -> Any:
    return sum(x.data)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("matmul expects rank-2 tensors")
    m, k1 = a.shape
    k2, n = b.shape
    if k1 != k2:
        raise ValueError("inner dimension mismatch")
    out: list[Any] = []
    for i in range(m):
        for j in range(n):
            s = 0
            for k in range(k1):
                s += a.data[i * k1 + k] * b.data[k * n + j]
            out.append(s)
    return Tensor(data=out, shape=(m, n), dtype=a.dtype, device=a.device)
