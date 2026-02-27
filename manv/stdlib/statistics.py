from __future__ import annotations

import statistics as _stats
from typing import Iterable


def mean(values: Iterable[float]) -> float:
    return float(_stats.mean(values))


def median(values: Iterable[float]) -> float:
    return float(_stats.median(values))


def variance(values: Iterable[float]) -> float:
    return float(_stats.variance(values))


def stdev(values: Iterable[float]) -> float:
    return float(_stats.stdev(values))
