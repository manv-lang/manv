from __future__ import annotations

import socket as _socket

AF_INET = _socket.AF_INET
SOCK_STREAM = _socket.SOCK_STREAM
SOCK_DGRAM = _socket.SOCK_DGRAM


class Socket(_socket.socket):
    pass


def socket(*, family: int = AF_INET, kind: int = SOCK_STREAM) -> Socket:
    return Socket(family, kind)
