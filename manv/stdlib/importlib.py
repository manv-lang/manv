from __future__ import annotations

import importlib as _importlib


def import_module(name: str):
    return _importlib.import_module(name)
