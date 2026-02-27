from __future__ import annotations

from typing import Any

from ..gpu_trace import TraceRecorder


def timeline() -> TraceRecorder:
    return TraceRecorder()


def export_chrome_trace(recorder: TraceRecorder) -> dict[str, Any]:
    return recorder.to_chrome_trace()
