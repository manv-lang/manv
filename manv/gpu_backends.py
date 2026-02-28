"""Compatibility backend layer used during the device-subsystem migration.

Why this module still exists:
- A large part of the compiler/test surface already imports
  `compile_kir_backend`, `create_runtime`, and `list_backends`.
- The new `manv.device` package is the canonical source of backend naming and
  selection, but the compile/runtime shims still need a stable compatibility
  facade while the rest of the codebase migrates incrementally.

Important constraints:
- Backend ids here must match the canonical ids in `manv.device`.
- CUDA is the only backend that attempts a real driver-facing runtime today.
- All non-CUDA runtime execution paths remain explicit CPU-reference shims so
  semantics stay deterministic even before those backends gain real runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .backends.cuda import (
    CudaCacheStore,
    build_cuda_cache_key,
    compile_cuda_source,
    cuda_is_available,
    emit_cuda_cpp,
)
from .device.interfaces import BackendId
from .gpu_trace import TraceRecorder
from .kernel_ir import KIRModule, parse_kir_module
from .kernel_mock import execute_kernel_ir_reference


@dataclass(frozen=True)
class BackendCapabilities:
    fp16: bool
    bf16: bool
    int8: bool
    atomics: bool
    shared_mem: bool
    subgroup_ops: bool
    barriers: bool
    images: bool
    max_threads_per_block: int
    max_shared_bytes: int
    async_copy: bool
    debug_printf: bool


@dataclass
class CompiledKernelBundle:
    backend: BackendId
    target: str
    binaries: dict[str, str]
    reflection: dict[str, Any]
    entrypoints: list[str]
    compile_log: list[str]
    cache_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "target": self.target,
            "binaries": self.binaries,
            "reflection": self.reflection,
            "entrypoints": self.entrypoints,
            "compile_log": self.compile_log,
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True)
class RuntimeHandle:
    backend: BackendId
    device: str


@dataclass
class DeviceBuffer:
    id: str
    size: int
    memory_kind: str
    data: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class KernelHandle:
    backend: BackendId
    entry: str
    bundle: CompiledKernelBundle


@dataclass
class LaunchToken:
    id: str
    status: str


class BaseKernelCompiler:
    backend: BackendId = "cpu"

    def compile(self, module: KIRModule, target: str, options: dict[str, Any] | None = None) -> CompiledKernelBundle:
        raise NotImplementedError


class CudaKernelCompiler(BaseKernelCompiler):
    backend: BackendId = "cuda"

    def compile(self, module: KIRModule, target: str, options: dict[str, Any] | None = None) -> CompiledKernelBundle:
        options = dict(options or {})
        arch = str(options.get("arch", "sm_80"))
        debug = bool(options.get("debug", False))
        source = emit_cuda_cpp(module, arch=arch, debug=debug)
        cache = CudaCacheStore(Path(".manv") / "target" / "cuda_cache")
        cache_key = build_cuda_cache_key(
            kir_hash=module.canonical_hash(),
            arch=arch,
            driver_version="available" if cuda_is_available() else "unavailable",
            nvrtc_version="nvrtc" if cuda_is_available() else "unavailable",
            compile_flags=[f"--gpu-architecture={arch}"],
            cuda_source=source,
        )
        cached = cache.load(cache_key)
        compile_log: list[str] = [f"cuda source emission complete for arch={arch}"]
        if cached is not None:
            source, ptx = cached
            compile_log.append("cuda cache hit")
        else:
            result = compile_cuda_source(source, arch=arch)
            if result.success and result.ptx:
                ptx = result.ptx
                compile_log.append(result.log or "nvrtc compilation completed")
                cache.store(
                    cache_key,
                    source=source,
                    ptx=ptx,
                    metadata={
                        "arch": arch,
                        "backend": "cuda",
                        "compile_log": compile_log,
                        "kir_hash": module.canonical_hash(),
                    },
                )
            else:
                compile_log.append(result.log or "nvrtc compilation unavailable")
                ptx = "\n".join(
                    [
                        "// PTX unavailable on this machine; emitting generated CUDA source instead.",
                        f"// nvrtc_log: {result.log}",
                        source,
                    ]
                )
        return CompiledKernelBundle(
            backend=self.backend,
            target=target,
            binaries={"ptx": ptx, "cuda_cpp": source},
            reflection={"kernels": [kernel.name for kernel in module.kernels], "arch": arch},
            entrypoints=[kernel.name for kernel in module.kernels],
            compile_log=compile_log,
            cache_key=cache_key,
        )


class TextBackendCompiler(BaseKernelCompiler):
    def __init__(self, backend: BackendId, dialect: str):
        self.backend = backend
        self.dialect = dialect

    def compile(self, module: KIRModule, target: str, options: dict[str, Any] | None = None) -> CompiledKernelBundle:
        del options
        code = _emit_text_backend(module, backend=self.backend, dialect=self.dialect)
        return CompiledKernelBundle(
            backend=self.backend,
            target=target,
            binaries={self.dialect: code},
            reflection={"kernels": [kernel.name for kernel in module.kernels], "dialect": self.dialect},
            entrypoints=[kernel.name for kernel in module.kernels],
            compile_log=[f"{self.backend} textual emission complete"],
            cache_key=f"{module.canonical_hash()}:{self.backend}:{target}",
        )


class BaseGpuRuntime:
    backend: BackendId = "cpu"

    def __init__(self, trace: TraceRecorder | None = None):
        self.trace = trace
        self._buffers: dict[str, DeviceBuffer] = {}
        self._launch_counter = 0

    def initialize(self, device_selector: str = "default") -> RuntimeHandle:
        if self.trace:
            self.trace.add("runtime", "initialize", 0.01, {"backend": self.backend, "device": device_selector})
        return RuntimeHandle(backend=self.backend, device=device_selector)

    def allocate(self, size: int, memory_kind: str = "global") -> DeviceBuffer:
        bid = f"buf_{len(self._buffers) + 1}"
        buffer = DeviceBuffer(id=bid, size=size, memory_kind=memory_kind, data=[0] * max(1, size))
        self._buffers[bid] = buffer
        if self.trace:
            self.trace.add("memory", "allocate", 0.01, {"backend": self.backend, "size": size, "kind": memory_kind})
        return buffer

    def free(self, buffer: DeviceBuffer) -> None:
        self._buffers.pop(buffer.id, None)
        if self.trace:
            self.trace.add("memory", "free", 0.01, {"backend": self.backend, "buffer": buffer.id})

    def copy_h2d(self, dst: DeviceBuffer, src: list[Any], bytes_count: int, stream: str | None = None) -> None:
        del stream
        count = min(bytes_count, len(dst.data), len(src))
        dst.data[:count] = list(src[:count])
        if self.trace:
            self.trace.add("memory", "copy_h2d", 0.01, {"backend": self.backend, "bytes": bytes_count})

    def copy_d2h(self, dst: list[Any], src: DeviceBuffer, bytes_count: int, stream: str | None = None) -> None:
        del stream
        count = min(bytes_count, len(dst), len(src.data))
        dst[:count] = list(src.data[:count])
        if self.trace:
            self.trace.add("memory", "copy_d2h", 0.01, {"backend": self.backend, "bytes": bytes_count})

    def copy_d2d(self, dst: DeviceBuffer, src: DeviceBuffer, bytes_count: int, stream: str | None = None) -> None:
        del stream
        count = min(bytes_count, len(dst.data), len(src.data))
        dst.data[:count] = list(src.data[:count])
        if self.trace:
            self.trace.add("memory", "copy_d2d", 0.01, {"backend": self.backend, "bytes": bytes_count})

    def load_kernel(self, bundle: CompiledKernelBundle, entry: str) -> KernelHandle:
        if self.trace:
            self.trace.add("runtime", "load_kernel", 0.01, {"backend": self.backend, "entry": entry})
        return KernelHandle(backend=self.backend, entry=entry, bundle=bundle)

    def launch(
        self,
        handle: KernelHandle,
        bindings: dict[str, list[Any]],
        launch_config: dict[str, int],
        stream: str | None = None,
    ) -> LaunchToken:
        del bindings, stream
        self._launch_counter += 1
        if self.trace:
            self.trace.add(
                "runtime",
                "launch",
                0.02,
                {"backend": self.backend, "entry": handle.entry, "launch": launch_config},
            )
        token = LaunchToken(id=f"launch_{self._launch_counter}", status="submitted")
        token.status = "complete"
        return token

    def synchronize(self, stream_or_token: str | LaunchToken | None = None) -> None:
        del stream_or_token
        if self.trace:
            self.trace.add("runtime", "synchronize", 0.01, {"backend": self.backend})

    def query_events(self) -> dict[str, Any]:
        return {"backend": self.backend, "events": []}

    def shutdown(self) -> None:
        if self.trace:
            self.trace.add("runtime", "shutdown", 0.01, {"backend": self.backend})


class MockGpuRuntime(BaseGpuRuntime):
    def __init__(self, backend: BackendId, trace: TraceRecorder | None = None):
        super().__init__(trace=trace)
        self.backend = backend

    def execute_kir(
        self,
        module: KIRModule,
        *,
        inputs: dict[str, Any] | None = None,
        launch_override: dict[str, int] | None = None,
        capture_debug: bool = True,
        compiled_bundle: CompiledKernelBundle | None = None,
        allow_cpu_fallback: bool = True,
    ) -> dict[str, Any]:
        del compiled_bundle, allow_cpu_fallback
        if self.trace:
            self.trace.add("runtime", "execute_kir", 0.05, {"backend": self.backend})
        result = execute_kernel_ir_reference(
            module,
            inputs=inputs,
            launch_override=launch_override,
            include_trace=True,
            capture_debug=capture_debug,
            check_oob=True,
        )
        # Non-CUDA backends are still CPU-reference runtimes today. Recording
        # that fact explicitly keeps `executed_backend` honest for reports.
        result["_runtime"] = {
            "backend": "cpu",
            "executed_on_gpu": False,
            "fallback_reason": f"{self.backend} runtime is not implemented yet",
        }
        return result


class CudaDispatchRuntime(BaseGpuRuntime):
    def __init__(self, trace: TraceRecorder | None = None):
        super().__init__(trace=trace)
        self.backend = "cuda"
        from .backends.cuda.runtime import CudaRuntime

        self._runtime = CudaRuntime()

    def execute_kir(
        self,
        module: KIRModule,
        *,
        inputs: dict[str, Any] | None = None,
        launch_override: dict[str, int] | None = None,
        capture_debug: bool = True,
        compiled_bundle: CompiledKernelBundle | None = None,
        allow_cpu_fallback: bool = True,
    ) -> dict[str, Any]:
        del capture_debug
        if self.trace:
            self.trace.add("runtime", "execute_kir", 0.05, {"backend": self.backend, "available": self._runtime.available})
        return self._runtime.execute_kir(
            module,
            inputs=inputs,
            launch_override=launch_override,
            compiled_bundle=compiled_bundle,
            allow_cpu_fallback=allow_cpu_fallback,
        )

    def shutdown(self) -> None:
        self._runtime.shutdown()
        super().shutdown()


BACKEND_CAPABILITIES: dict[BackendId, BackendCapabilities] = {
    "cuda": BackendCapabilities(True, True, True, True, True, True, True, True, 1024, 99_000, True, True),
    "rocm": BackendCapabilities(True, True, True, True, True, True, True, True, 1024, 64_000, True, True),
    "level0": BackendCapabilities(True, True, True, True, True, True, True, True, 1024, 64_000, True, False),
    "vulkan-spv": BackendCapabilities(True, True, True, True, True, True, True, True, 1024, 64_000, True, False),
    "directx": BackendCapabilities(True, True, True, True, True, True, True, True, 1024, 64_000, True, False),
    "webgpu": BackendCapabilities(True, False, True, False, True, False, True, True, 256, 32_000, False, False),
    "cpu": BackendCapabilities(True, True, True, True, True, True, True, False, 8192, 0, False, True),
}


BACKEND_COMPILERS: dict[BackendId, BaseKernelCompiler] = {
    "cuda": CudaKernelCompiler(),
    "rocm": TextBackendCompiler("rocm", "hip"),
    "level0": TextBackendCompiler("level0", "spirv_text"),
    "vulkan-spv": TextBackendCompiler("vulkan-spv", "spirv_text"),
    "directx": TextBackendCompiler("directx", "hlsl"),
    "webgpu": TextBackendCompiler("webgpu", "wgsl"),
    "cpu": TextBackendCompiler("cpu", "kir_json"),
}


def compile_kir_backend(
    kernel_ir: KIRModule | dict[str, Any],
    backend: BackendId,
    *,
    target: str = "generic",
    options: dict[str, Any] | None = None,
    trace: TraceRecorder | None = None,
) -> CompiledKernelBundle:
    module = kernel_ir if isinstance(kernel_ir, KIRModule) else parse_kir_module(kernel_ir)
    compiler = BACKEND_COMPILERS[backend]

    if trace:
        with trace.scoped("compile", "kir_backend_compile", {"backend": backend, "target": target}):
            return compiler.compile(module, target=target, options=options)
    return compiler.compile(module, target=target, options=options)


def create_runtime(backend: BackendId, *, trace: TraceRecorder | None = None) -> BaseGpuRuntime:
    if backend == "cuda":
        return CudaDispatchRuntime(trace=trace)
    return MockGpuRuntime(backend=backend, trace=trace)


def list_backends() -> list[BackendId]:
    return ["cuda", "rocm", "level0", "vulkan-spv", "directx", "webgpu", "cpu"]


def get_backend_capabilities(backend: BackendId) -> BackendCapabilities:
    return BACKEND_CAPABILITIES[backend]


def _emit_text_backend(module: KIRModule, *, backend: BackendId, dialect: str) -> str:
    lines = [
        f"// backend: {backend}",
        f"// dialect: {dialect}",
        f"// source: {module.source}",
        f"// hash: {module.canonical_hash()}",
        "",
    ]

    for kernel in module.kernels:
        lines.append(f"kernel {kernel.name} {{")
        lines.append(
            f"  launch grid=({kernel.launch_model.grid_x},{kernel.launch_model.grid_y},{kernel.launch_model.grid_z})"
        )
        lines.append(
            f"  launch block=({kernel.launch_model.block_x},{kernel.launch_model.block_y},{kernel.launch_model.block_z})"
        )
        for op in kernel.all_ops():
            lines.append(f"  {op.id}: {op.opcode}({', '.join(op.inputs)}) -> {', '.join(op.outputs)}")
        lines.append("}")
        lines.append("")

    if dialect == "kir_json":
        lines.append(json.dumps(module.to_dict(), indent=2, sort_keys=True))

    return "\n".join(lines)
