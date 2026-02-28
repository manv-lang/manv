"""Small cross-platform shared-library probing helpers.

Why this module exists:
- Probe-only backends still need to answer "is the vendor runtime present?"
  without importing a heavy Python binding or crashing module import.
- The resolver needs deterministic skip reasons, not ad-hoc `OSError` strings.

Design notes:
- This helper intentionally does not perform deep runtime initialization.
  Loading and symbol lookup are enough for the early selection/reporting phases.
- Candidate ordering is stable so the resulting skip reason remains stable.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DynamicLoadResult:
    """Result of probing a shared library candidate set."""

    available: bool
    loaded_name: str | None = None
    skip_reason: str | None = None


def try_load_any(
    candidates: Iterable[str],
    *,
    missing_reason: str,
    symbol: str | None = None,
) -> DynamicLoadResult:
    """Try to load one of the provided library names.

    The function returns only deterministic metadata. Callers that need the
    loaded handle can load the same library again once selection has chosen the
    backend. This keeps probing side-effect free.
    """

    ordered = [candidate for candidate in candidates if candidate]
    for candidate in ordered:
        resolved = ctypes.util.find_library(candidate) or candidate
        try:
            handle = ctypes.CDLL(resolved)
        except OSError:
            continue
        if symbol is not None and getattr(handle, symbol, None) is None:
            continue
        return DynamicLoadResult(available=True, loaded_name=resolved, skip_reason=None)
    return DynamicLoadResult(available=False, loaded_name=None, skip_reason=missing_reason)
