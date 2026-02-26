from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .gpu_backends import (
    BackendId,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_backend": self.selected_backend,
            "executed_backend": self.executed_backend,
            "bundle": self.bundle.to_dict(),
            "outputs": self.outputs,
            "trace": self.trace,
            "substitutions": self.substitutions,
        }


def select_backend(preferred: str = "auto") -> BackendId:
    if preferred == "auto":
        return "cuda"

    normalized = preferred.replace("-", "_").lower()
    alias = {
        "cuda_ptx": "cuda",
        "vulkan": "vulkan_spirv",
        "vulkan_spv": "vulkan_spirv",
        "spirv": "vulkan_spirv",
        "dx": "directx",
        "cpu": "cpu_ref",
        "cpu_reference": "cpu_ref",
    }
    resolved = alias.get(normalized, normalized)
    if resolved not in list_backends():
        raise RuntimeError(f"unsupported backend '{preferred}'")
    return resolved  # type: ignore[return-value]


def dispatch_kernel_ir(
    kernel_ir: KIRModule | dict[str, Any],
    *,
    backend: str = "auto",
    target: str = "generic",
    inputs: dict[str, list[Any]] | None = None,
    launch_override: dict[str, int] | None = None,
    strict_verify: bool = False,
) -> DispatchResult:
    module = kernel_ir if isinstance(kernel_ir, KIRModule) else parse_kir_module(kernel_ir)
    assert_valid_kir_module(module, strict=strict_verify)

    selected = select_backend(backend)
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
    runtime.initialize(device_selector=target)

    with trace.scoped("runtime", "backend_launch", {"backend": selected}):
        result = runtime.execute_kir(module, inputs=inputs, launch_override=launch_override, capture_debug=True)

    runtime.shutdown()

    return DispatchResult(
        selected_backend=selected,
        executed_backend=selected,
        bundle=bundle,
        outputs=result,
        trace=trace.to_chrome_trace(),
        substitutions=substitutions,
    )


def backend_capability_table() -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for backend in list_backends():
        cap = get_backend_capabilities(backend)
        table[backend] = {
            "fp16": cap.fp16,
            "bf16": cap.bf16,
            "int8": cap.int8,
            "atomics": cap.atomics,
            "shared_mem": cap.shared_mem,
            "subgroup_ops": cap.subgroup_ops,
            "barriers": cap.barriers,
            "images": cap.images,
            "max_threads_per_block": cap.max_threads_per_block,
            "max_shared_bytes": cap.max_shared_bytes,
            "async_copy": cap.async_copy,
            "debug_printf": cap.debug_printf,
        }
    return table
