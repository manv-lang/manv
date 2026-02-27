from __future__ import annotations

from builtins import all as _all
from builtins import any as _any
from builtins import enumerate as _enumerate
from builtins import max as _max
from builtins import min as _min
from builtins import range as _range
from builtins import sorted as _sorted
from builtins import sum as _sum
from builtins import zip as _zip
from typing import Any, Iterable

__all__ = [
    "print_",
    "repr_",
    "len_",
    "hash_",
    "range_",
    "enumerate_",
    "zip_",
    "min_",
    "max_",
    "sum_",
    "any_",
    "all_",
    "sorted_",
    "int_",
    "float_",
    "bool_",
    "str_",
    "bytes_",
    "iter_",
    "next_",
]


def print_(*parts: Any) -> None:
    print(*parts)


def repr_(value: Any) -> str:
    return repr(value)


def len_(value: Any) -> int:
    return len(value)


def hash_(value: Any) -> int:
    return hash(value)


def range_(*args: int) -> range:
    return _range(*args)


def enumerate_(items: Iterable[Any], start: int = 0):
    return _enumerate(items, start=start)


def zip_(*items: Iterable[Any]):
    return _zip(*items)


def min_(items: Iterable[Any]) -> Any:
    return _min(items)


def max_(items: Iterable[Any]) -> Any:
    return _max(items)


def sum_(items: Iterable[Any], start: Any = 0) -> Any:
    return _sum(items, start)


def any_(items: Iterable[Any]) -> bool:
    return _any(items)


def all_(items: Iterable[Any]) -> bool:
    return _all(items)


def sorted_(items: Iterable[Any], *, reverse: bool = False):
    return _sorted(items, reverse=reverse)


def int_(value: Any = 0) -> int:
    return int(value)


def float_(value: Any = 0.0) -> float:
    return float(value)


def bool_(value: Any = False) -> bool:
    return bool(value)


def str_(value: Any = "") -> str:
    return str(value)


def bytes_(value: Any = b"") -> bytes:
    return bytes(value)


def iter_(value: Any):
    return iter(value)


def next_(it: Any, default: Any = None) -> Any:
    if default is None:
        return next(it)
    return next(it, default)
