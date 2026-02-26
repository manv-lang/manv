from __future__ import annotations

import io
from pathlib import Path
import sys
import threading
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.dap import DAPSession, MessageFramer
from manv.debug_engine import DebugEngine


class _ChunkReader(io.BytesIO):
    def __init__(self, payload: bytes, chunk: int = 3):
        super().__init__(payload)
        self.chunk = chunk

    def read(self, size: int = -1) -> bytes:  # noqa: A003
        if size < 0:
            size = self.chunk
        else:
            size = min(size, self.chunk)
        return super().read(size)


class _CapturingDAPSession(DAPSession):
    def __init__(self, engine: DebugEngine | None = None):
        super().__init__(reader=io.BytesIO(), writer=io.BytesIO(), engine=engine)
        self.events: list[dict[str, object]] = []
        self._events_lock = threading.Lock()
        self._cursor = 0

    def _send_event(self, event: str, body: dict[str, object]) -> None:  # type: ignore[override]
        with self._events_lock:
            self.events.append({"event": event, "body": body})

    def wait_for_event(self, name: str, timeout: float = 2.0) -> dict[str, object]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._events_lock:
                for idx in range(self._cursor, len(self.events)):
                    item = self.events[idx]
                    if item.get("event") == name:
                        self._cursor = idx + 1
                        return item
            time.sleep(0.01)
        raise AssertionError(f"event not received: {name}")


@pytest.fixture()
def sample_program(tmp_path: Path) -> Path:
    src = tmp_path / "main.mv"
    src.write_text(
        "fn main() -> int:\n"
        "    int x = 1\n"
        "    int y = 2\n"
        "    x = x + y\n"
        "    print(x)\n"
        "    return x\n",
        encoding="utf-8",
    )
    return src


def test_message_framer_roundtrip_partial_reads() -> None:
    payload = {"type": "request", "seq": 1, "command": "initialize", "arguments": {"clientID": "vscode"}}

    out = io.BytesIO()
    MessageFramer.write_message(out, payload)

    partial = _ChunkReader(out.getvalue(), chunk=2)
    parsed = MessageFramer.read_message(partial)
    assert parsed == payload


def test_message_framer_malformed_header() -> None:
    bad = io.BytesIO(b"Content-Type: application/json\r\n\r\n{}")
    with pytest.raises(ValueError):
        MessageFramer.read_message(bad)


def test_dap_launch_breakpoint_stack_scopes_and_eval(sample_program: Path) -> None:
    session = _CapturingDAPSession()

    init = session.handle_request({"type": "request", "seq": 1, "command": "initialize", "arguments": {}})
    assert init["success"] is True

    launch = session.handle_request(
        {
            "type": "request",
            "seq": 2,
            "command": "launch",
            "arguments": {"program": str(sample_program), "stopOnEntry": True},
        }
    )
    assert launch["success"] is True
    session.wait_for_event("stopped")

    set_bp = session.handle_request(
        {
            "type": "request",
            "seq": 3,
            "command": "setBreakpoints",
            "arguments": {
                "source": {"path": str(sample_program)},
                "breakpoints": [{"line": 4}],
            },
        }
    )
    assert set_bp["success"] is True
    assert set_bp["body"]["breakpoints"][0]["verified"] is True

    cont = session.handle_request({"type": "request", "seq": 4, "command": "continue", "arguments": {"threadId": 1}})
    assert cont["success"] is True
    stop = session.wait_for_event("stopped")
    assert stop["body"]["reason"] in {"breakpoint", "step"}

    stack = session.handle_request(
        {
            "type": "request",
            "seq": 5,
            "command": "stackTrace",
            "arguments": {"threadId": 1, "startFrame": 0, "levels": 10},
        }
    )
    assert stack["success"] is True
    frames = stack["body"]["stackFrames"]
    assert len(frames) >= 1

    frame_id = frames[0]["id"]
    scopes = session.handle_request(
        {
            "type": "request",
            "seq": 6,
            "command": "scopes",
            "arguments": {"frameId": frame_id},
        }
    )
    assert scopes["success"] is True
    assert len(scopes["body"]["scopes"]) >= 3

    eval_resp = session.handle_request(
        {
            "type": "request",
            "seq": 7,
            "command": "evaluate",
            "arguments": {"frameId": frame_id, "context": "watch", "expression": "x + y"},
        }
    )
    assert eval_resp["success"] is True
    assert eval_resp["body"]["result"] in {"3", "3.0"}

    session.handle_request({"type": "request", "seq": 8, "command": "terminate", "arguments": {}})


def test_dap_attach_existing_session(sample_program: Path) -> None:
    engine = DebugEngine()
    sid = engine.launch(str(sample_program), options={"stopOnEntry": True})

    session = _CapturingDAPSession(engine=engine)
    init = session.handle_request({"type": "request", "seq": 1, "command": "initialize", "arguments": {}})
    assert init["success"] is True

    attach = session.handle_request(
        {
            "type": "request",
            "seq": 2,
            "command": "attach",
            "arguments": {"sessionId": sid},
        }
    )
    assert attach["success"] is True

    threads = session.handle_request({"type": "request", "seq": 3, "command": "threads", "arguments": {}})
    assert threads["success"] is True
    assert threads["body"]["threads"][0]["id"] == 1

    session.handle_request({"type": "request", "seq": 4, "command": "disconnect", "arguments": {}})


def test_dap_trace_compare_exception_stop(tmp_path: Path) -> None:
    src = tmp_path / "main.mv"
    src.write_text(
        "fn main() -> int:\n"
        "    array a = [1, 2, 3]\n"
        "    return 0\n",
        encoding="utf-8",
    )

    session = _CapturingDAPSession()
    session.handle_request({"type": "request", "seq": 1, "command": "initialize", "arguments": {}})
    launch = session.handle_request(
        {
            "type": "request",
            "seq": 2,
            "command": "launch",
            "arguments": {"program": str(src), "traceCompare": True},
        }
    )
    assert launch["success"] is True

    stop = session.wait_for_event("stopped")
    assert stop["body"]["reason"] == "exception"
    assert "Kernel mismatch" in str(stop["body"].get("text", ""))

    session.handle_request({"type": "request", "seq": 3, "command": "continue", "arguments": {"threadId": 1}})
    term = session.wait_for_event("terminated")
    assert term["event"] == "terminated"

