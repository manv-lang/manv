from __future__ import annotations

from typing import Any, get_type_hints


def hints(obj: Any) -> dict[str, Any]:
    return get_type_hints(obj)
