from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


def defaultmap(factory: Callable[[], Any]):
    return defaultdict(factory)
