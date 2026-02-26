from __future__ import annotations

from ..diagnostics import ManvError, diag


def not_implemented(feature: str = "std.memory") -> None:
    raise ManvError(diag("E9101", f"{feature} controls not implemented in v0.1", "<runtime>", 1, 1))
