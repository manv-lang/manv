"""CUDA-side memory bookkeeping.

Why this exists:
- The runtime needs a stable home for residency tracking and device buffer
  ownership that is separate from the backend selection layer.
- The first version keeps the model intentionally conservative: buffers are
  synchronized at the GPU boundary and residency tracking is advisory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeviceBuffer:
    name: str
    device_ptr: int
    nbytes: int
    dtype: str
    host_shadow: list[Any] = field(default_factory=list)
    dirty_host: bool = False
    dirty_device: bool = False


@dataclass
class ResidencyEntry:
    host_id: int
    buffer: DeviceBuffer


class ResidencyTracker:
    def __init__(self) -> None:
        self._entries: dict[int, ResidencyEntry] = {}

    def get(self, host_value: object) -> ResidencyEntry | None:
        return self._entries.get(id(host_value))

    def remember(self, host_value: object, buffer: DeviceBuffer) -> ResidencyEntry:
        entry = ResidencyEntry(host_id=id(host_value), buffer=buffer)
        self._entries[id(host_value)] = entry
        return entry
