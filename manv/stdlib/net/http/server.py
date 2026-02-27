from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable


def serve(host: str, port: int, handler: Callable[[str, str, bytes], tuple[int, dict[str, str], bytes]]) -> None:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, headers, body = handler("GET", self.path, b"")
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length)
            status, headers, body = handler("POST", self.path, payload)
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _fmt, *_args):  # pragma: no cover
            return

    HTTPServer((host, int(port)), _Handler).serve_forever()
