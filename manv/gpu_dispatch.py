"""Kernel-IR backend dispatch and compatibility reporting.

Why this module exists:
- It preserves the existing `dispatch_kernel_ir` entrypoint while the runtime
  selection logic moves into the dedicated `manv.device` package.
- It keeps backend compilation/runtime execution separate from backend probing
  and reporting, which is important for the phased migration plan.

Important invariants:
- Auto selection must delegate to the deterministic device resolver.
- Explicit backend requests normalize aliases but do not silently invent new
  backends.
- `executed_backend` must reflect where work actually happened. Today that
  means CUDA for real GPU execution and `cpu` for every reference fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .device import SelectionRequest, normalize_backend_name, resolve_device_selection
from .device.interfaces import BackendId, SelectionReport
from .gpu_backends import (
    CompiledKernelBundle,
    create_runtime,
    compile_kir_backend,
    get_backend_capabilities,
    list_backends,
)
from .gpu_trace import TraceRecorder
from .kernel_ir import KIRModule, parse_kir_module
from .kir_verify import assert_valid_kir_module
from .vendor_interop import register_default_rules, try_substitute_kernel


@dataclass
class DispatchResult:
    selected_backend: BackendId
    executed_backend: BackendId
    bundle: CompiledKernelBundle
    outputs: dict[str, Any]
    trace: dict[str, Any]
    substitutions: list[dict[str, Any]]
    selection_report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_backend": self.selected_backend,
            "executed_backend": self.executed_backend,
            "bundle": self.bundle.to_dict(),
            "outputs": self.outputs,
            "trace": self.trace,
            "substitutions": self.substitutions,
            "selection_report": self.selection_report,
        }


def backend_selection_report(preferred: str = "auto", *, device: str | None = None, policy: str = "auto") -> SelectionReport:
    requested_backend = preferred
    requested_device = device

    # Environment-based backend selection must apply at the shared dispatch
    # entrypoint so CLI calls, intrinsic calls, and packaged runtimes all see
    # the same override behavior. Explicit non-`auto` values still win.
    if requested_backend == "auto":
        requested_backend = os.getenv("MANV_BACKEND", requested_backend)
    if requested_device is None:
        requested_device = os.getenv("MANV_DEVICE")

    return resolve_device_selection(
        SelectionRequest(
            requested_backend=requested_backend,
            requested_device=requested_device,
            policy=policy,
        )
    )


def select_backend(preferred: str = "auto", *, device: str | None = None) -> BackendId:
    normalized = normalize_backend_name(preferred)
    if normalized == "auto":
        return backend_selection_report(preferred, device=device).selected_backend
    return normalized


def dispatch_kernel_ir(
    kernel_ir: KIRModule | dict[str, Any],
    *,
    backend: str = "auto",
    target: str = "generic",
    inputs: dict[str, Any] | None = None,
    launch_override: dict[str, int] | None = None,
    strict_verify: bool = False,
    allow_cpu_fallback: bool = True,
    device: str | None = None,
) -> DispatchResult:
    module = kernel_ir if isinstance(kernel_ir, KIRModule) else parse_kir_module(kernel_ir)
    assert_valid_kir_module(module, strict=strict_verify)

    selection = backend_selection_report(backend, device=device)
    selected = selection.selected_backend if normalize_backend_name(backend) == "auto" else select_backend(backend, device=device)
    trace = TraceRecorder()

    register_default_rules()
    substitutions: list[dict[str, Any]] = []
    for kernel in module.to_dict().get("kernels", []):
        sub = try_substitute_kernel(kernel)
        if sub is not None:
            substitutions.append({"library": sub.library, "symbol": sub.symbol, "reason": sub.reason})

    with trace.scoped("compile", "backend_compile", {"backend": selected, "target": target}):
        bundle = compile_kir_backend(module, selected, target=target, trace=trace)

    runtime = create_runtime(selected, trace=trace)
    runtime.initialize(device_selector=device or target)

    try:
        with trace.scoped("runtime", "backend_launch", {"backend": selected}):
            result = runtime.execute_kir(
                module,
                inputs=inputs,
                launch_override=launch_override,
                capture_debug=True,
                compiled_bundle=bundle,
                allow_cpu_fallback=allow_cpu_fallback,
            )
    finally:
        runtime.shutdown()

    runtime_meta = result.get("_runtime", {}) if isinstance(result, dict) else {}
    executed_backend = selected
    if isinstance(runtime_meta, dict) and runtime_meta.get("executed_on_gpu") is False:
        executed_backend = "cpu"

    return DispatchResult(
        selected_backend=selected,
        executed_backend=executed_backend,
        bundle=bundle,
        outputs=result,
        trace=trace.to_chrome_trace(),
        substitutions=substitutions,
        selection_report=selection.to_dict(),
    )


def backend_capability_table() -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for backend in list_backends():
        capability = get_backend_capabilities(backend)
        table[backend] = {
            "fp16": capability.fp16,
            "bf16": capability.bf16,
            "int8": capability.int8,
            "atomics": capability.atomics,
            "shared_mem": capability.shared_mem,
            "subgroup_ops": capability.subgroup_ops,
            "barriers": capability.barriers,
            "images": capability.images,
            "max_threads_per_block": capability.max_threads_per_block,
            "max_shared_bytes": capability.max_shared_bytes,
            "async_copy": capability.async_copy,
            "debug_printf": capability.debug_printf,
        }
    return table
