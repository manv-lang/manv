from __future__ import annotations

import os


def get(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def set(name: str, value: str) -> None:
    os.environ[str(name)] = str(value)


def items() -> dict[str, str]:
    return dict(os.environ)
