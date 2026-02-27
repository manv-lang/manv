from __future__ import annotations

from itertools import islice as _islice
from typing import Any, Iterable


def chunked(items: Iterable[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("size must be > 0")
    it = iter(items)
    out: list[list[Any]] = []
    while True:
        chunk = list(_islice(it, size))
        if not chunk:
            return out
        out.append(chunk)


def take(items: Iterable[Any], n: int) -> list[Any]:
    return list(_islice(items, max(0, int(n))))
