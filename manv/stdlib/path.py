from __future__ import annotations

from pathlib import Path as _Path

Path = _Path


def join(*parts: str) -> str:
    return str(_Path(parts[0]).joinpath(*parts[1:]))


def resolve(path: str) -> str:
    return str(_Path(path).resolve())


def glob(path: str, pattern: str) -> list[str]:
    return [str(p) for p in _Path(path).glob(pattern)]


def relative_to(path: str, parent: str) -> str:
    return str(_Path(path).resolve().relative_to(_Path(parent).resolve()))
