from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class TraceEvent:
    ts: float
    dur: float
    cat: str
    name: str
    pid: int
    tid: int
    args: dict[str, Any] = field(default_factory=dict)

    def to_chrome(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cat": self.cat,
            "ph": "X",
            "ts": self.ts,
            "dur": self.dur,
            "pid": self.pid,
            "tid": self.tid,
            "args": self.args,
        }


class TraceRecorder:
    def __init__(self, *, pid: int = 1, tid: int = 1):
        self.pid = pid
        self.tid = tid
        self._events: list[TraceEvent] = []

    def add(self, cat: str, name: str, dur_ms: float, args: dict[str, Any] | None = None) -> None:
        now = time.time() * 1_000_000.0
        self._events.append(
            TraceEvent(
                ts=now,
                dur=dur_ms * 1000.0,
                cat=cat,
                name=name,
                pid=self.pid,
                tid=self.tid,
                args=dict(args or {}),
            )
        )

    def scoped(self, cat: str, name: str, args: dict[str, Any] | None = None):
        return _TraceScope(self, cat=cat, name=name, args=args or {})

    def to_chrome_trace(self) -> dict[str, Any]:
        return {"traceEvents": [e.to_chrome() for e in self._events]}

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)


class _TraceScope:
    def __init__(self, recorder: TraceRecorder, *, cat: str, name: str, args: dict[str, Any]):
        self.recorder = recorder
        self.cat = cat
        self.name = name
        self.args = args
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        end = time.time()
        self.recorder.add(self.cat, self.name, dur_ms=(end - self.start) * 1000.0, args=self.args)
        return False
