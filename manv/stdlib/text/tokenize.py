from __future__ import annotations


def tokenize(text: str, sep: str | None = None) -> list[str]:
    return text.split(sep)
