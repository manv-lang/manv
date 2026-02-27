from __future__ import annotations

import json as _json
from typing import Any, TextIO


def loads(text: str) -> Any:
    return _json.loads(text)


def dumps(value: Any, *, canonical: bool = False) -> str:
    if canonical:
        return _json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _json.dumps(value)


def load_stream(stream: TextIO) -> Any:
    return _json.load(stream)


def dump_stream(value: Any, stream: TextIO, *, canonical: bool = False) -> None:
    if canonical:
        _json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        return
    _json.dump(value, stream)
