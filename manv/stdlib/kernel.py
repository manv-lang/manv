from __future__ import annotations

from typing import Any

from .gpu import dispatch


def compile(kernel_ir: dict[str, Any], *, backend: str = "auto", target: str = "generic") -> dict[str, Any]:
    return {
        "kernel_ir": kernel_ir,
        "backend": backend,
        "target": target,
    }


def launch(compiled: dict[str, Any], *, inputs: dict[str, list[Any]] | None = None, launch_override: dict[str, int] | None = None) -> dict[str, Any]:
    return dispatch(
        compiled["kernel_ir"],
        backend=compiled.get("backend", "auto"),
        target=compiled.get("target", "generic"),
        inputs=inputs,
        launch_override=launch_override,
    )


def jit(kernel_ir: dict[str, Any], *, backend: str = "auto", target: str = "generic") -> dict[str, Any]:
    return compile(kernel_ir, backend=backend, target=target)


def to_device(values: list[Any]) -> list[Any]:
    return list(values)


def to_host(values: list[Any]) -> list[Any]:
    return list(values)
