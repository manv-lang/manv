from __future__ import annotations

from io import StringIO


def open_text(path: str, mode: str = "r", encoding: str = "utf-8"):
    return open(path, mode, encoding=encoding)
