from __future__ import annotations

from .diagnostics import ManvError, diag


def unsupported_feature(feature: str, file: str, line: int, column: int, detail: str = "") -> ManvError:
    suffix = f": {detail}" if detail else ""
    return ManvError(diag("E9000", f"feature '{feature}' is not implemented{suffix}", file, line, column))
