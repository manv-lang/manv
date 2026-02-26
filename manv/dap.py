from __future__ import annotations

import json
import queue
import socket
import sys
import threading
from typing import Any, BinaryIO

from .debug_engine import DebugEngine


class MessageFramer:
    """DAP wire framing (Content-Length + JSON body) for stdio/tcp transports."""

    @staticmethod
    def read_message(stream: BinaryIO) -> dict[str, Any] | None:
        header_bytes = b""
        while b"\r\n\r\n" not in header_bytes:
            chunk = stream.read(1)
            if not chunk:
                if not header_bytes:
                    return None
                raise ValueError("unexpected EOF while reading DAP headers")
            header_bytes += chunk
            if len(header_bytes) > 64 * 1024:
                raise ValueError("DAP header too large")

        header_text = header_bytes.decode("utf-8", errors="replace")
        content_length = None
        for line in header_text.split("\r\n"):
            if not line.strip():
                continue
            key, sep, value = line.partition(":")
            if not sep:
                raise ValueError("invalid DAP header line")
            if key.lower().strip() == "content-length":
                content_length = int(value.strip())
        if content_length is None:
            raise ValueError("missing Content-Length header")

        body = MessageFramer._read_exact(stream, content_length)
        return json.loads(body.decode("utf-8"))

    @staticmethod
    def write_message(stream: BinaryIO, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        stream.write(header)
        stream.write(body)
        stream.flush()

    @staticmethod
    def _read_exact(stream: BinaryIO, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = stream.read(remaining)
            if not chunk:
                raise ValueError("unexpected EOF while reading DAP payload")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


class DAPSession:
    def __init__(self, reader: BinaryIO, writer: BinaryIO, engine: DebugEngine | None = None):
        self.reader = reader
        self.writer = writer
        self.engine = engine or DebugEngine()
        self._write_lock = threading.Lock()
        self._outbound: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._events_thread: threading.Thread | None = None
        self._running = False
        self._active_session_id: str | None = None

    def serve_forever(self) -> None:
        self._running = True
        self._events_thread = threading.Thread(target=self._event_pump, name="manv-dap-events", daemon=True)
        self._events_thread.start()

        while self._running:
            try:
                req = MessageFramer.read_message(self.reader)
            except Exception as err:
                self._send_event("output", {"category": "stderr", "output": f"DAP framing error: {err}\n"})
                break

            if req is None:
                break
            if req.get("type") != "request":
                continue

            resp = self.handle_request(req)
            self._send(resp)

            if req.get("command") in {"disconnect", "terminate"}:
                break

        self._running = False
        self._outbound.put(None)

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any]:
        seq = int(req.get("seq", 0))
        command = str(req.get("command", ""))
        args = req.get("arguments", {}) or {}

        try:
            body = self._dispatch(command, args)
            return {
                "type": "response",
                "request_seq": seq,
                "success": True,
                "command": command,
                "body": body or {},
            }
        except Exception as err:
            return {
                "type": "response",
                "request_seq": seq,
                "success": False,
                "command": command,
                "message": str(err),
                "body": {"error": {"id": 1, "format": str(err)}},
            }

    def emit_event(self, event: str, body: dict[str, Any]) -> None:
        self._send_event(event, body)

    def _dispatch(self, command: str, args: dict[str, Any]) -> dict[str, Any]:
        if command == "initialize":
            self._send_event("initialized", {})
            return {
                "supportsConfigurationDoneRequest": True,
                "supportsFunctionBreakpoints": True,
                "supportsConditionalBreakpoints": True,
                "supportsHitConditionalBreakpoints": True,
                "supportsLogPoints": True,
                "supportsEvaluateForHovers": True,
                "supportsSetVariable": False,
                "supportsCompletionsRequest": False,
                "supportsExceptionInfoRequest": False,
                "supportsTerminateRequest": True,
                "supportsLoadedSourcesRequest": False,
                "supportsRestartRequest": False,
                "supportsSteppingGranularity": False,
            }

        if command == "launch":
            program = str(args.get("program", ""))
            if not program:
                raise RuntimeError("launch requires 'program'")
            options = {
                "stopOnEntry": bool(args.get("stopOnEntry", False)),
                "traceCompare": bool(args.get("traceCompare", False)),
            }
            self._active_session_id = self.engine.launch(program, args=args.get("args") or [], options=options)
            self.engine.add_listener(self._active_session_id, self._on_runtime_event)
            return {}

        if command == "attach":
            sid = str(args.get("sessionId") or args.get("session") or "")
            if not sid:
                raise RuntimeError("attach requires 'sessionId'")
            self._active_session_id = self.engine.attach(sid, options=args)
            self.engine.add_listener(self._active_session_id, self._on_runtime_event)
            return {}

        if command == "configurationDone":
            return {}

        if command == "setBreakpoints":
            sid = self._require_session()
            source = args.get("source", {}) or {}
            path = source.get("path")
            if not path:
                raise RuntimeError("setBreakpoints requires source.path")
            bps = self.engine.set_source_breakpoints(sid, str(path), list(args.get("breakpoints") or []))
            return {
                "breakpoints": [
                    {
                        "id": bp.id,
                        "verified": bp.verified,
                        "line": bp.line,
                        "message": bp.message,
                    }
                    for bp in bps
                ]
            }

        if command == "setFunctionBreakpoints":
            sid = self._require_session()
            raw = list(args.get("breakpoints") or [])
            names = [str(item.get("name", "")) for item in raw if item.get("name")]
            bps = self.engine.set_function_breakpoints(sid, names)
            return {
                "breakpoints": [
                    {
                        "id": bp.id,
                        "verified": bp.verified,
                        "line": bp.line,
                        "message": bp.message,
                    }
                    for bp in bps
                ]
            }

        if command == "setExceptionBreakpoints":
            sid = self._require_session()
            self.engine.set_exception_breakpoints(sid, list(args.get("filters") or []))
            return {}

        if command == "threads":
            sid = self._require_session()
            return {"threads": self.engine.get_threads(sid)}

        if command == "stackTrace":
            sid = self._require_session()
            thread_id = int(args.get("threadId", 1))
            start = int(args.get("startFrame", 0))
            levels = int(args.get("levels", 20))
            frames = self.engine.get_stack(sid, thread_id=thread_id, start=start, levels=levels)
            return {"stackFrames": frames, "totalFrames": len(frames)}

        if command == "scopes":
            sid = self._require_session()
            frame_id = int(args.get("frameId", 0))
            return {"scopes": self.engine.get_scopes(sid, frame_id)}

        if command == "variables":
            sid = self._require_session()
            ref = int(args.get("variablesReference", 0))
            start = int(args.get("start", 0))
            count = int(args.get("count", 0))
            return {"variables": self.engine.get_variables(sid, ref, start=start, count=count)}

        if command == "evaluate":
            sid = self._require_session()
            expr = str(args.get("expression", ""))
            context = str(args.get("context", "watch"))
            frame_id = args.get("frameId")
            f = int(frame_id) if frame_id is not None else None
            return self.engine.evaluate(sid, expr, frame_id=f, context=context)

        if command == "continue":
            sid = self._require_session()
            self.engine.resume(sid, thread_id=int(args.get("threadId", 1)))
            return {"allThreadsContinued": True}

        if command == "next":
            sid = self._require_session()
            self.engine.step_over(sid, thread_id=int(args.get("threadId", 1)))
            return {}

        if command == "stepIn":
            sid = self._require_session()
            self.engine.step_in(sid, thread_id=int(args.get("threadId", 1)))
            return {}

        if command == "stepOut":
            sid = self._require_session()
            self.engine.step_out(sid, thread_id=int(args.get("threadId", 1)))
            return {}

        if command == "pause":
            sid = self._require_session()
            self.engine.pause(sid, thread_id=int(args.get("threadId", 1)))
            return {}

        if command == "source":
            sid = self._require_session()
            source_ref = int(args.get("sourceReference", 0))
            text = self.engine.get_source(sid, source_ref)
            return {"content": text, "mimeType": "text/plain"}

        if command == "disconnect":
            if self._active_session_id:
                self.engine.disconnect(self._active_session_id)
            self._running = False
            return {}

        if command == "terminate":
            if self._active_session_id:
                self.engine.terminate(self._active_session_id)
            self._running = False
            return {}

        raise RuntimeError(f"unsupported DAP request: {command}")

    def _require_session(self) -> str:
        if not self._active_session_id:
            raise RuntimeError("no active debug session")
        return self._active_session_id

    def _on_runtime_event(self, event: dict[str, Any]) -> None:
        self._send_event(str(event.get("event")), dict(event.get("body") or {}))

    def _send(self, payload: dict[str, Any]) -> None:
        with self._write_lock:
            MessageFramer.write_message(self.writer, payload)

    def _send_event(self, event: str, body: dict[str, Any]) -> None:
        self._outbound.put({"type": "event", "event": event, "body": body})

    def _event_pump(self) -> None:
        while True:
            item = self._outbound.get()
            if item is None:
                return
            try:
                self._send(item)
            except Exception:
                return


class DAPServer:
    def __init__(self, engine: DebugEngine | None = None):
        self.engine = engine or DebugEngine()

    def start_stdio(self) -> None:
        session = DAPSession(sys.stdin.buffer, sys.stdout.buffer, engine=self.engine)
        session.serve_forever()

    def start_tcp(self, host: str = "127.0.0.1", port: int = 4711) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(1)
            while True:
                conn, _ = server.accept()
                with conn:
                    reader = conn.makefile("rb")
                    writer = conn.makefile("wb")
                    try:
                        session = DAPSession(reader, writer, engine=self.engine)
                        session.serve_forever()
                    finally:
                        reader.close()
                        writer.close()
