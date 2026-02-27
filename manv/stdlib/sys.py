from __future__ import annotations

from .errors import CapabilityError
from .providers import capability_table
from .settings import capabilities as _caps, set_deterministic_mode as _set_det


def set_deterministic_mode(flag: bool) -> None:
    _set_det(flag)


def capabilities() -> dict[str, bool]:
    out = _caps()
    out.update(capability_table())
    return out


def require(capability: str) -> None:
    cap = capabilities()
    if not cap.get(capability, False):
        raise CapabilityError(f"required capability not available: {capability}")
