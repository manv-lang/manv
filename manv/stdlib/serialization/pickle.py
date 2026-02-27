from __future__ import annotations

import pickle as _pickle
from typing import Any


def dumps(value: Any, protocol: int = _pickle.HIGHEST_PROTOCOL) -> bytes:
    return _pickle.dumps(value, protocol=protocol)


def loads(data: bytes) -> Any:
    return _pickle.loads(data)
