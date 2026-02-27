from __future__ import annotations

from dataclasses import dataclass
from urllib import request as _request


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding, errors="replace")


class Session:
    def request(self, method: str, url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> Response:
        req = _request.Request(url=url, data=data, method=method.upper(), headers=headers or {})
        with _request.urlopen(req) as resp:
            return Response(status=resp.status, headers=dict(resp.headers.items()), body=resp.read())


def request(method: str, url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> Response:
    return Session().request(method=method, url=url, data=data, headers=headers)
