from __future__ import annotations

from io import BytesIO


def open_binary(path: str, mode: str = "rb"):
    return open(path, mode)
