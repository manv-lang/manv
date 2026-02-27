from __future__ import annotations

import platform as _platform


def system() -> str:
    return _platform.system()


def machine() -> str:
    return _platform.machine()


def python_version() -> str:
    return _platform.python_version()
