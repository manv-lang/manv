from __future__ import annotations

import heapq
from typing import Any, Iterable


def heapify(items: list[Any]) -> None:
    heapq.heapify(items)


def heappush(items: list[Any], value: Any) -> None:
    heapq.heappush(items, value)


def heappop(items: list[Any]) -> Any:
    return heapq.heappop(items)


def nlargest(n: int, items: Iterable[Any]) -> list[Any]:
    return heapq.nlargest(n, items)


def nsmallest(n: int, items: Iterable[Any]) -> list[Any]:
    return heapq.nsmallest(n, items)
