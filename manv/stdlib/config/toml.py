from __future__ import annotations

import tomllib

try:
    import tomli_w as _tomli_w
except Exception:  # pragma: no cover
    _tomli_w = None


def loads(text: str):
    return tomllib.loads(text)


def dumps(value: dict[str, object]) -> str:
    if _tomli_w is None:
        raise RuntimeError("toml serialization requires tomli_w in this runtime")
    return _tomli_w.dumps(value)
