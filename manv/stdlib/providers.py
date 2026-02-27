from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import platform
import random
import time
from typing import Protocol


class HostProvider(Protocol):
    name: str

    def now_ms(self) -> int: ...

    def monotonic_ms(self) -> int: ...

    def rand_seed(self, seed: int) -> None: ...

    def rand_float(self) -> float: ...

    def fs_exists(self, path: str) -> bool: ...

    def fs_read_text(self, path: str) -> str: ...

    def fs_write_text(self, path: str, data: str) -> None: ...


@dataclass
class DefaultHostProvider:
    name: str = "default"

    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def monotonic_ms(self) -> int:
        return int(time.monotonic() * 1000)

    def rand_seed(self, seed: int) -> None:
        random.seed(int(seed))

    def rand_float(self) -> float:
        return random.random()

    def fs_exists(self, path: str) -> bool:
        return Path(path).exists()

    def fs_read_text(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    def fs_write_text(self, path: str, data: str) -> None:
        Path(path).write_text(data, encoding="utf-8")


@dataclass
class DeterministicTestProvider(DefaultHostProvider):
    name: str = "deterministic"
    _time_ms: int = 0
    _rand: random.Random = random.Random(0)

    def now_ms(self) -> int:
        return self._time_ms

    def monotonic_ms(self) -> int:
        return self._time_ms

    def advance_ms(self, delta: int) -> None:
        self._time_ms += int(delta)

    def rand_seed(self, seed: int) -> None:
        self._rand.seed(int(seed))

    def rand_float(self) -> float:
        return self._rand.random()


@dataclass
class MockProvider(DefaultHostProvider):
    name: str = "mock"


_CURRENT: HostProvider = DefaultHostProvider()


def current_provider() -> HostProvider:
    return _CURRENT


def set_provider(provider: HostProvider) -> None:
    global _CURRENT
    _CURRENT = provider


def capability_table() -> dict[str, bool]:
    return {
        "fs": True,
        "path": True,
        "time": True,
        "random": True,
        "network": True,
        "threading": True,
        "process": True,
        "compression": True,
        "gpu": True,
        "platform.windows": platform.system().lower().startswith("win"),
        "platform.posix": os.name == "posix",
    }
