from __future__ import annotations

import os as _os

environ = _os.environ


def getcwd() -> str:
    return _os.getcwd()


def chdir(path: str) -> None:
    _os.chdir(path)
