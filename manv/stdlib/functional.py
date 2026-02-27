from __future__ import annotations

from functools import reduce
from typing import Any, Callable, Iterable


def map_(fn: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]:
    return [fn(x) for x in items]


def filter_(fn: Callable[[Any], bool], items: Iterable[Any]) -> list[Any]:
    return [x for x in items if fn(x)]


def reduce_(fn: Callable[[Any, Any], Any], items: Iterable[Any], initial: Any | None = None) -> Any:
    if initial is None:
        return reduce(fn, items)
    return reduce(fn, items, initial)
