from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal

from .backends.cuda_ptx import emit_cuda_ptx
from .gpu_trace import TraceRecorder
from .kernel_ir import KIRModule, parse_kir_module
from .kernel_mock import execute_kernel_ir_reference


BackendId = Literal["cuda", "rocm", "metal", "vulkan_spirv", "webgpu", "opencl", "directx", "cpu_ref"]


@dataclass
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


@dataclass
class RuntimeHandle:
    backend: BackendId
    device: str


@dataclass
class DeviceBuffer:
    id: str
    size: int
    memory_kind: str
    data: list[Any] = field(default_factory=list)


@dataclass
class KernelHandle:
    backend: BackendId
    entry: str
    bundle: CompiledKernelBundle


@dataclass
class LaunchToken:
    id: str
    status: str


class BaseKernelCompiler:
    backend: BackendId = "cpu_ref"

    def compile(self, module: KIRModule, target: str, options: dict[str, Any] | None = None) -> CompiledKernelBundle:
        raise NotImplementedError


class CudaKernelCompiler(BaseKernelCompiler):
    backend: BackendId = "cuda"

    def compile(self, module: KIRModule, target: str, options: dict[str, Any] | None = None) -> CompiledKernelBundle:
        payload = module.to_dict()
        ptx = emit_cuda_ptx(payload)
        return CompiledKernelBundle(
            backend=self.backend,
            target=target,
            binaries={"ptx": ptx},
            reflection={"kernels": [k.name for k in module.kernels]},
            entrypoints=[k.name for k in module.kernels],
            compile_log=["cuda ptx emission complete"],
            cache_key=module.canonical_hash() + ":cuda:" + target,
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
            reflection={"kernels": [k.name for k in module.kernels], "dialect": self.dialect},
            entrypoints=[k.name for k in module.kernels],
            compile_log=[f"{self.backend} textual emission complete"],
            cache_key=module.canonical_hash() + f":{self.backend}:{target}",
        )


class BaseGpuRuntime:
    backend: BackendId = "cpu_ref"

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
        buf = DeviceBuffer(id=bid, size=size, memory_kind=memory_kind, data=[0] * max(1, size))
        self._buffers[bid] = buf
        if self.trace:
            self.trace.add("memory", "allocate", 0.01, {"backend": self.backend, "size": size, "kind": memory_kind})
        return buf

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

    def launch(self, handle: KernelHandle, bindings: dict[str, list[Any]], launch_config: dict[str, int], stream: str | None = None) -> LaunchToken:
        del stream
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
        inputs: dict[str, list[Any]] | None = None,
        launch_override: dict[str, int] | None = None,
        capture_debug: bool = True,
    ) -> dict[str, Any]:
        if self.trace:
            self.trace.add("runtime", "execute_kir", 0.05, {"backend": self.backend})
        return execute_kernel_ir_reference(
            module,
            inputs=inputs,
            launch_override=launch_override,
            include_trace=True,
            capture_debug=capture_debug,
            check_oob=True,
        )


BACKEND_CAPABILITIES: dict[BackendId, BackendCapabilities] = {
    "cuda": BackendCapabilities(True, True, True, True, True, True, True, True, 1024, 99_000, True, True),
    "rocm": BackendCapabilities(True, True, True, True, True, True, True, True, 1024, 64_000, True, True),
    "metal": BackendCapabilities(True, False, True, True, True, False, True, True, 1024, 32_000, False, False),
    "vulkan_spirv": BackendCapabilities(True, True, True, True, True, True, True, True, 1024, 64_000, True, False),
    "webgpu": BackendCapabilities(True, False, True, False, True, False, True, True, 256, 32_000, False, False),
    "opencl": BackendCapabilities(True, False, True, True, True, False, True, True, 1024, 32_000, True, False),
    "directx": BackendCapabilities(True, True, True, True, True, True, True, True, 1024, 64_000, True, False),
    "cpu_ref": BackendCapabilities(True, True, True, True, True, True, True, False, 8192, 0, False, True),
}


BACKEND_COMPILERS: dict[BackendId, BaseKernelCompiler] = {
    "cuda": CudaKernelCompiler(),
    "rocm": TextBackendCompiler("rocm", "hip"),
    "metal": TextBackendCompiler("metal", "msl"),
    "vulkan_spirv": TextBackendCompiler("vulkan_spirv", "spirv_text"),
    "webgpu": TextBackendCompiler("webgpu", "wgsl"),
    "opencl": TextBackendCompiler("opencl", "opencl_c"),
    "directx": TextBackendCompiler("directx", "hlsl"),
    "cpu_ref": TextBackendCompiler("cpu_ref", "kir_json"),
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


def create_runtime(backend: BackendId, *, trace: TraceRecorder | None = None) -> MockGpuRuntime:
    # v0.1 multi-backend runtime is unified through deterministic CPU reference execution.
    return MockGpuRuntime(backend=backend, trace=trace)


def list_backends() -> list[BackendId]:
    return ["cuda", "rocm", "metal", "vulkan_spirv", "webgpu", "opencl", "directx", "cpu_ref"]


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
        lines.append(f"  launch grid=({kernel.launch_model.grid_x},{kernel.launch_model.grid_y},{kernel.launch_model.grid_z})")
        lines.append(f"  launch block=({kernel.launch_model.block_x},{kernel.launch_model.block_y},{kernel.launch_model.block_z})")
        for op in kernel.all_ops():
            lines.append(f"  {op.id}: {op.opcode}({', '.join(op.inputs)}) -> {', '.join(op.outputs)}")
        lines.append("}")
        lines.append("")

    if dialect == "kir_json":
        lines.append(json.dumps(module.to_dict(), indent=2, sort_keys=True))

    return "\n".join(lines)
