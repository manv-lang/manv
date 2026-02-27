from __future__ import annotations

import time
from typing import Any, Callable


def benchmark(fn: Callable[..., Any], *args: Any, repeat: int = 5, **kwargs: Any) -> dict[str, float]:
    times: list[float] = []
    for _ in range(max(1, int(repeat))):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return {
        "min_s": min(times),
        "max_s": max(times),
        "avg_s": sum(times) / len(times),
    }
