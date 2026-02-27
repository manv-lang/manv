from __future__ import annotations

import random as _random
from typing import Any


def seed(value: int) -> None:
    _random.seed(value)


def randint(lo: int, hi: int) -> int:
    return _random.randint(lo, hi)


def random() -> float:
    return _random.random()


def choice(items: list[Any]) -> Any:
    return _random.choice(items)


def shuffle(items: list[Any]) -> None:
    _random.shuffle(items)
