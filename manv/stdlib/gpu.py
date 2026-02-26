from __future__ import annotations

from typing import Any

from ..gpu_backends import list_backends
from ..gpu_dispatch import backend_capability_table, dispatch_kernel_ir


def backends() -> list[str]:
    return list_backends()


def capabilities() -> dict[str, dict[str, Any]]:
    return backend_capability_table()


def dispatch(
    kernel_ir: dict[str, Any],
    *,
    backend: str = "auto",
    target: str = "generic",
    inputs: dict[str, list[Any]] | None = None,
    launch_override: dict[str, int] | None = None,
) -> dict[str, Any]:
    result = dispatch_kernel_ir(
        kernel_ir,
        backend=backend,
        target=target,
        inputs=inputs,
        launch_override=launch_override,
        strict_verify=False,
    )
    return result.to_dict()
