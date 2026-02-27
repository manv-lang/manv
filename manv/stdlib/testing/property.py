from __future__ import annotations

from typing import Any, Callable, Iterable


def for_all(values: Iterable[Any], predicate: Callable[[Any], bool]) -> bool:
    for value in values:
        if not predicate(value):
            return False
    return True
