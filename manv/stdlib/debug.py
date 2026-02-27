from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DebugEvent:
    kind: str
    message: str
