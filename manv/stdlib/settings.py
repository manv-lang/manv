from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeSettings:
    deterministic_mode: bool = False
    fixed_time_ms: int | None = None
    fixed_seed: int | None = None
    capabilities: dict[str, bool] = field(default_factory=dict)


_SETTINGS = RuntimeSettings(
    capabilities={
        "fs": True,
        "path": True,
        "time": True,
        "random": True,
        "json": True,
        "network": True,
        "threading": True,
        "process": True,
        "compression": True,
        "gpu": True,
    }
)


def settings() -> RuntimeSettings:
    return _SETTINGS


def set_deterministic_mode(flag: bool, *, fixed_time_ms: int | None = None, fixed_seed: int | None = None) -> None:
    _SETTINGS.deterministic_mode = bool(flag)
    _SETTINGS.fixed_time_ms = fixed_time_ms
    _SETTINGS.fixed_seed = fixed_seed


def capabilities() -> dict[str, bool]:
    return dict(_SETTINGS.capabilities)


def set_capability(name: str, enabled: bool) -> None:
    _SETTINGS.capabilities[str(name)] = bool(enabled)
