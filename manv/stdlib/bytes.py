from __future__ import annotations


def to_bytes(text: str, encoding: str = "utf-8") -> bytes:
    return text.encode(encoding)


def from_bytes(data: bytes, encoding: str = "utf-8") -> str:
    return data.decode(encoding)
