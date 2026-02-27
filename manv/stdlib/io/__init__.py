from .buffered import BufferedReader, BufferedWriter
from .binary import BytesIO
from .text import StringIO, open_text


def open(path: str, mode: str = "r", encoding: str | None = "utf-8"):
    if "b" in mode:
        return __builtins__["open"](path, mode)
    return __builtins__["open"](path, mode, encoding=encoding)


__all__ = ["open", "StringIO", "BytesIO", "BufferedReader", "BufferedWriter", "open_text"]
