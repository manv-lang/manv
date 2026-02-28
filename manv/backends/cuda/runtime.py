"""CUDA Driver API runtime wrappers.

Why this module exists:
- The CUDA backend needs a real runtime surface that can move bytes, load PTX,
  and launch kernels without depending on a Python CUDA package.
- The language-level `@gpu` feature still needs deterministic fallback
  behavior, so the runtime must make "executed on GPU" vs "fell back" explicit.

Important constraints:
- HLIR remains the semantic authority. This runtime only executes Kernel IR
  that has already been validated and lowered upstream.
- Best-effort users must never get a fake "GPU success" result. If compilation
  or launch fails, the runtime either raises or performs an explicit CPU
  reference fallback with a recorded reason.
- The v1 model keeps an implicit synchronization boundary at each GPU call.
  That is intentionally conservative so CUDA execution cannot reorder visible
  language effects.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from ...kernel_ir import KIRKernel, KIRModule, parse_kir_module
from ...kernel_mock import execute_kernel_ir_reference
from .cache import CudaCacheStore, build_cuda_cache_key
from .codegen import emit_cuda_cpp
from .memory import DeviceBuffer, ResidencyTracker
from .nvrtc_compiler import compile_cuda_source


CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64


def cuda_is_available() -> bool:
    runtime = CudaRuntime()
    return runtime.available


class CudaRuntimeError(RuntimeError):
    """Raised when the CUDA runtime cannot complete a requested operation."""


@dataclass
class DeviceContext:
    device_ordinal: int
    handle: int | None = None
    device_handle: int | None = None


@dataclass
class KernelModule:
    name: str
    ptx: str
    handle: int | None = None
    cache_key: str | None = None


@dataclass
class KernelFunction:
    name: str
    module_name: str
    handle: int | None = None


class CudaRuntime:
    """Small CUDA runtime wrapper with explicit GPU-vs-fallback outcomes.

    The runtime is deliberately conservative:
    - availability means the driver was loaded and at least one device is
      visible;
    - context creation is lazy so simple availability checks do not allocate;
    - launch failures can either raise or fall back, depending on the caller's
      policy.
    """

    def __init__(self, *, arch: str = "sm_80") -> None:
        self._cuda = _load_cuda_driver()
        self._arch = arch
        self._last_error = ""
        self._residency = ResidencyTracker()
        self._context: DeviceContext | None = None
        self._modules: dict[str, KernelModule] = {}
        self._functions: dict[tuple[str, str], KernelFunction] = {}
        self._buffers: dict[int, DeviceBuffer] = {}
        self._available_after_init = self._cuda is not None and self._driver_init()
        device_count = self._query_device_count() if self._available_after_init else 0
        self.available = self._available_after_init and device_count > 0
        if self._available_after_init and device_count <= 0 and not self._last_error:
            self._last_error = "CUDA driver initialized successfully, but no CUDA devices are available"

    def last_error(self) -> str:
        return self._last_error

    def device_count(self) -> int:
        if not self._available_after_init:
            return 0
        return self._query_device_count()

    def execute_kir(
        self,
        module: KIRModule | dict[str, Any],
        *,
        inputs: dict[str, Any] | None = None,
        launch_override: dict[str, int] | None = None,
        allow_cpu_fallback: bool = True,
        compiled_bundle: Any | None = None,
    ) -> dict[str, Any]:
        parsed = module if isinstance(module, KIRModule) else parse_kir_module(module)
        if not self.available:
            return self._fallback_reference(
                parsed,
                inputs=inputs,
                launch_override=launch_override,
                reason=self._last_error or "CUDA driver is unavailable",
                allow_cpu_fallback=allow_cpu_fallback,
            )

        try:
            self._ensure_context()
            source, ptx, cache_key = self._resolve_compiled_artifacts(parsed, compiled_bundle=compiled_bundle)
            loaded_module = self._load_module(source=source, ptx=ptx, cache_key=cache_key)
            host_values = _normalize_host_inputs(inputs)

            kernel_outputs: dict[str, Any] = {}
            trace: dict[str, Any] = {}
            for kernel in parsed.kernels:
                function = self._load_function(loaded_module, kernel.name)
                self._execute_kernel(
                    kernel,
                    function,
                    host_values=host_values,
                    launch_override=launch_override,
                )
                trace[kernel.name] = {
                    "backend": "cuda",
                    "cache_key": cache_key,
                    "grid_x": _resolve_launch(kernel, launch_override)["grid_x"],
                    "block_x": _resolve_launch(kernel, launch_override)["block_x"],
                }
                kernel_outputs[kernel.name] = _resolve_kernel_output(kernel, host_values)

            self.sync()
            self._last_error = ""
            return {
                "kernels": kernel_outputs,
                "buffers": {name: value for name, value in host_values.items() if isinstance(value, list)},
                "scalars": {name: value for name, value in host_values.items() if not isinstance(value, list)},
                "trace": trace,
                "debug_records": [],
                "_runtime": {
                    "backend": "cuda",
                    "executed_on_gpu": True,
                    "cache_key": cache_key,
                },
            }
        except Exception as err:
            reason = str(err)
            self._last_error = reason
            return self._fallback_reference(
                parsed,
                inputs=inputs,
                launch_override=launch_override,
                reason=reason,
                allow_cpu_fallback=allow_cpu_fallback,
            )

    def alloc(self, name: str, nbytes: int, dtype: str) -> DeviceBuffer:
        if not self.available:
            self._last_error = "CUDA alloc requested, but CUDA is unavailable"
            raise CudaRuntimeError(self._last_error)
        self._ensure_context()
        pointer = CUdeviceptr()
        fn = self._driver_function("cuMemAlloc_v2", "cuMemAlloc")
        alloc_nbytes = max(1, int(nbytes))
        self._check(fn(ctypes.byref(pointer), alloc_nbytes), "cuMemAlloc")
        buffer = DeviceBuffer(name=name, device_ptr=int(pointer.value), nbytes=alloc_nbytes, dtype=dtype)
        self._buffers[buffer.device_ptr] = buffer
        return buffer

    def free(self, buffer: DeviceBuffer) -> None:
        if buffer.device_ptr == 0:
            return
        fn = self._driver_function("cuMemFree_v2", "cuMemFree")
        self._check(fn(CUdeviceptr(buffer.device_ptr)), "cuMemFree")
        self._buffers.pop(buffer.device_ptr, None)
        buffer.device_ptr = 0

    def copy_h2d(self, buffer: DeviceBuffer, values: list[Any]) -> None:
        packed, nbytes = _pack_host_values(values, buffer.dtype)
        if packed is None or nbytes == 0:
            buffer.host_shadow = list(values)
            buffer.dirty_host = False
            buffer.dirty_device = False
            return
        fn = self._driver_function("cuMemcpyHtoD_v2", "cuMemcpyHtoD")
        self._check(fn(CUdeviceptr(buffer.device_ptr), ctypes.addressof(packed), nbytes), "cuMemcpyHtoD")
        buffer.host_shadow = list(values)
        buffer.dirty_host = False
        buffer.dirty_device = True

    def copy_d2h(self, buffer: DeviceBuffer, length: int | None = None) -> list[Any]:
        logical_length = len(buffer.host_shadow) if length is None else int(length)
        unpacked = _unpack_host_values(
            self._device_to_host_bytes(buffer.device_ptr, logical_length, buffer.dtype),
            buffer.dtype,
        )
        buffer.host_shadow = unpacked[:]
        buffer.dirty_host = True
        buffer.dirty_device = False
        return unpacked

    def copy_d2d(self, dst: DeviceBuffer, src: DeviceBuffer) -> None:
        fn = self._driver_function("cuMemcpyDtoD_v2", "cuMemcpyDtoD")
        nbytes = min(dst.nbytes, src.nbytes)
        if nbytes <= 0:
            return
        self._check(fn(CUdeviceptr(dst.device_ptr), CUdeviceptr(src.device_ptr), nbytes), "cuMemcpyDtoD")
        dst.host_shadow = list(src.host_shadow)
        dst.dirty_device = True

    def lookup_buffer(self, device_ptr: int) -> DeviceBuffer | None:
        return self._buffers.get(int(device_ptr))

    def sync(self) -> None:
        if not self.available or self._cuda is None:
            return None
        fn = getattr(self._cuda, "cuCtxSynchronize", None)
        if fn is None:
            return None
        self._check(fn(), "cuCtxSynchronize")
        return None

    def dump_ptx(self, path: Path, ptx: str) -> None:
        path.write_text(ptx, encoding="utf-8")

    def shutdown(self) -> None:
        # Resource teardown is deterministic and best-effort. The runtime is
        # used by command-line tools and tests, so teardown should not hide the
        # original failure that triggered shutdown.
        for buffer in list(self._buffers.values()):
            try:
                self.free(buffer)
            except Exception:
                continue

        if self._cuda is not None:
            unload = getattr(self._cuda, "cuModuleUnload", None)
            if unload is not None:
                for module in list(self._modules.values()):
                    if module.handle is None:
                        continue
                    try:
                        unload(CUmodule(module.handle))
                    except Exception:
                        continue

            destroy = getattr(self._cuda, "cuCtxDestroy_v2", None) or getattr(self._cuda, "cuCtxDestroy", None)
            if destroy is not None and self._context is not None and self._context.handle is not None:
                try:
                    destroy(CUcontext(self._context.handle))
                except Exception:
                    pass

        self._modules.clear()
        self._functions.clear()
        self._context = None

    def _driver_init(self) -> bool:
        if self._cuda is None:
            return False
        fn = getattr(self._cuda, "cuInit", None)
        if fn is None:
            self._last_error = "CUDA driver missing cuInit"
            return False
        code = fn(0)
        if code != 0:
            self._last_error = self._format_driver_error(code, "cuInit")
            return False
        return True

    def _query_device_count(self) -> int:
        if self._cuda is None:
            return 0
        fn = getattr(self._cuda, "cuDeviceGetCount", None)
        if fn is None:
            self._last_error = "CUDA driver missing cuDeviceGetCount"
            return 0
        count = ctypes.c_int()
        code = fn(ctypes.byref(count))
        if code != 0:
            self._last_error = self._format_driver_error(code, "cuDeviceGetCount")
            return 0
        return int(count.value)

    def _ensure_context(self) -> DeviceContext:
        if self._context is not None:
            return self._context
        if self._cuda is None:
            raise CudaRuntimeError("CUDA driver is unavailable")

        device_fn = self._driver_function("cuDeviceGet")
        create_fn = self._driver_function("cuCtxCreate_v2", "cuCtxCreate")

        device = CUdevice()
        self._check(device_fn(ctypes.byref(device), 0), "cuDeviceGet")

        context = CUcontext()
        self._check(create_fn(ctypes.byref(context), 0, device), "cuCtxCreate")

        self._context = DeviceContext(
            device_ordinal=0,
            handle=int(context.value) if context.value is not None else None,
            device_handle=int(device.value),
        )
        return self._context

    def _resolve_compiled_artifacts(self, module: KIRModule, *, compiled_bundle: Any | None) -> tuple[str, str, str]:
        source = ""
        ptx = ""
        cache_key = ""
        arch = self._arch

        if compiled_bundle is not None:
            binaries = getattr(compiled_bundle, "binaries", {})
            reflection = getattr(compiled_bundle, "reflection", {})
            cache_key = str(getattr(compiled_bundle, "cache_key", ""))
            source = str(binaries.get("cuda_cpp", ""))
            ptx = str(binaries.get("ptx", ""))
            if isinstance(reflection, dict) and reflection.get("arch"):
                arch = str(reflection["arch"])

        if source and ptx:
            return source, ptx, cache_key or module.canonical_hash()

        if not source:
            source = emit_cuda_cpp(module, arch=arch, debug=True)

        cache_key = cache_key or build_cuda_cache_key(
            kir_hash=module.canonical_hash(),
            arch=arch,
            driver_version=self._driver_version(),
            nvrtc_version="unknown",
            compile_flags=[f"--gpu-architecture={arch}"],
            cuda_source=source,
        )

        cache = CudaCacheStore(Path(".manv") / "target" / "cuda_cache")
        cached = cache.load(cache_key)
        if cached is not None:
            return cached[0], cached[1], cache_key

        compiled = compile_cuda_source(source, arch=arch)
        if not compiled.success or not compiled.ptx:
            raise CudaRuntimeError(compiled.log or "NVRTC compilation failed")

        cache.store(
            cache_key,
            source=source,
            ptx=compiled.ptx,
            metadata={
                "arch": arch,
                "backend": "cuda",
                "driver_version": self._driver_version(),
                "kir_hash": module.canonical_hash(),
                "nvrtc_version": compiled.version,
            },
        )
        return source, compiled.ptx, cache_key

    def _load_module(self, *, source: str, ptx: str, cache_key: str) -> KernelModule:
        del source
        module_key = cache_key or hashlib.sha256(ptx.encode("utf-8")).hexdigest()
        cached = self._modules.get(module_key)
        if cached is not None:
            return cached

        load_fn = self._driver_function("cuModuleLoadDataEx")
        module = CUmodule()
        ptx_bytes = ptx.encode("utf-8")
        self._check(load_fn(ctypes.byref(module), ctypes.c_char_p(ptx_bytes), 0, None, None), "cuModuleLoadDataEx")

        loaded = KernelModule(name=module_key, ptx=ptx, handle=int(module.value) if module.value is not None else None, cache_key=cache_key)
        self._modules[module_key] = loaded
        return loaded

    def _load_function(self, module: KernelModule, kernel_name: str) -> KernelFunction:
        key = (module.name, kernel_name)
        cached = self._functions.get(key)
        if cached is not None:
            return cached

        fn = self._driver_function("cuModuleGetFunction")
        handle = CUfunction()
        self._check(
            fn(
                ctypes.byref(handle),
                CUmodule(module.handle or 0),
                kernel_name.encode("utf-8"),
            ),
            "cuModuleGetFunction",
        )
        loaded = KernelFunction(name=kernel_name, module_name=module.name, handle=int(handle.value) if handle.value is not None else None)
        self._functions[key] = loaded
        return loaded

    def _execute_kernel(
        self,
        kernel: KIRKernel,
        function: KernelFunction,
        *,
        host_values: dict[str, Any],
        launch_override: dict[str, int] | None,
    ) -> None:
        launch = _resolve_launch(kernel, launch_override)
        problem_size = _infer_problem_size(kernel, host_values)
        _ensure_host_buffers(kernel, host_values, problem_size)

        # Every KIR buffer becomes a short-lived device allocation for the
        # kernel launch. This keeps v1 semantics simple and makes copy-back
        # boundaries explicit. Residency tracking is still updated so future
        # async/resident paths can reuse the same bookkeeping surface.
        buffers: dict[str, DeviceBuffer] = {}
        for name, dtype in _kernel_buffers(kernel):
            values = list(host_values.get(name, []))
            buffer = self.alloc(name, _dtype_nbytes(dtype) * max(len(values), 1), dtype)
            self.copy_h2d(buffer, values)
            buffers[name] = buffer
            if name in host_values and isinstance(host_values[name], list):
                self._residency.remember(host_values[name], buffer)

        try:
            kernel_params, keepalive = self._marshal_kernel_params(kernel, buffers, host_values, problem_size)
            del keepalive
            launch_fn = self._driver_function("cuLaunchKernel")
            self._check(
                launch_fn(
                    CUfunction(function.handle or 0),
                    launch["grid_x"],
                    launch["grid_y"],
                    launch["grid_z"],
                    launch["block_x"],
                    launch["block_y"],
                    launch["block_z"],
                    getattr(kernel.launch_model, "shared_bytes", 0),
                    CUstream(),
                    kernel_params,
                    None,
                ),
                f"cuLaunchKernel({kernel.name})",
            )
            self.sync()
            for name, buffer in buffers.items():
                original = host_values.get(name, [])
                copied = self.copy_d2h(buffer, length=len(original) if isinstance(original, list) else None)
                host_values[name] = copied
        finally:
            for buffer in buffers.values():
                try:
                    self.free(buffer)
                except Exception:
                    continue

    def _marshal_kernel_params(
        self,
        kernel: KIRKernel,
        buffers: dict[str, DeviceBuffer],
        host_values: dict[str, Any],
        problem_size: int,
    ) -> tuple[Any, list[Any]]:
        values: list[Any] = []
        params: list[ctypes.c_void_p] = []

        # The generated CUDA ABI is deterministic: declared KIR params first,
        # then the synthetic `manv_n` loop bound used by the grid-stride loop.
        for param in kernel.signature.params:
            if param.kind == "buffer" or param.by_ref:
                buffer = buffers.get(param.name)
                if buffer is None:
                    raise CudaRuntimeError(f"kernel parameter '{param.name}' has no device buffer")
                scalar = CUdeviceptr(buffer.device_ptr)
                values.append(scalar)
                params.append(ctypes.cast(ctypes.pointer(scalar), ctypes.c_void_p))
            else:
                scalar_value = _marshal_scalar_value(host_values.get(param.name), param.dtype, param.name)
                values.append(scalar_value)
                params.append(ctypes.cast(ctypes.pointer(scalar_value), ctypes.c_void_p))

        extent = ctypes.c_int(problem_size)
        values.append(extent)
        params.append(ctypes.cast(ctypes.pointer(extent), ctypes.c_void_p))
        return (ctypes.c_void_p * len(params))(*params), values

    def _device_to_host_bytes(self, device_ptr: int, length: int, dtype: str) -> bytes:
        nbytes = _dtype_nbytes(dtype) * max(length, 0)
        if nbytes == 0:
            return b""
        host_buffer = ctypes.create_string_buffer(nbytes)
        fn = self._driver_function("cuMemcpyDtoH_v2", "cuMemcpyDtoH")
        self._check(fn(ctypes.addressof(host_buffer), CUdeviceptr(device_ptr), nbytes), "cuMemcpyDtoH")
        return bytes(host_buffer.raw[:nbytes])

    def _fallback_reference(
        self,
        module: KIRModule,
        *,
        inputs: dict[str, Any] | None,
        launch_override: dict[str, int] | None,
        reason: str,
        allow_cpu_fallback: bool,
    ) -> dict[str, Any]:
        if not allow_cpu_fallback:
            raise CudaRuntimeError(reason)
        result = execute_kernel_ir_reference(
            module,
            inputs={name: list(value) for name, value in _normalize_host_inputs(inputs).items() if isinstance(value, list)},
            launch_override=launch_override,
            include_trace=True,
            capture_debug=True,
        )
        result["_runtime"] = {
            "backend": "cpu",
            "executed_on_gpu": False,
            "fallback_reason": reason,
        }
        return result

    def _driver_function(self, *names: str) -> Any:
        if self._cuda is None:
            raise CudaRuntimeError("CUDA driver is unavailable")
        for name in names:
            fn = getattr(self._cuda, name, None)
            if fn is not None:
                return fn
        joined = ", ".join(names)
        raise CudaRuntimeError(f"CUDA driver missing required symbol(s): {joined}")

    def _driver_version(self) -> str:
        if self._cuda is None:
            return "unavailable"
        fn = getattr(self._cuda, "cuDriverGetVersion", None)
        if fn is None:
            return "unknown"
        version = ctypes.c_int()
        code = fn(ctypes.byref(version))
        if code != 0:
            return "unknown"
        return str(version.value)

    def _check(self, code: int, op: str) -> None:
        if code != 0:
            raise CudaRuntimeError(self._format_driver_error(code, op))

    def _format_driver_error(self, code: int, op: str) -> str:
        if self._cuda is None:
            return f"{op} failed with code {code}"

        # Error mapping is deliberately centralized here so every driver failure
        # surfaces deterministic, readable diagnostics instead of raw integers.
        name_fn = getattr(self._cuda, "cuGetErrorName", None)
        msg_fn = getattr(self._cuda, "cuGetErrorString", None)
        err_name = ""
        err_msg = ""

        if name_fn is not None:
            raw_name = ctypes.c_char_p()
            try:
                if name_fn(code, ctypes.byref(raw_name)) == 0 and raw_name.value:
                    err_name = raw_name.value.decode("utf-8", errors="replace")
            except Exception:
                err_name = ""

        if msg_fn is not None:
            raw_msg = ctypes.c_char_p()
            try:
                if msg_fn(code, ctypes.byref(raw_msg)) == 0 and raw_msg.value:
                    err_msg = raw_msg.value.decode("utf-8", errors="replace")
            except Exception:
                err_msg = ""

        detail = ": ".join(part for part in [err_name, err_msg] if part)
        if detail:
            return f"{op} failed with code {code}: {detail}"
        return f"{op} failed with code {code}"


def _resolve_launch(kernel: KIRKernel, override: dict[str, int] | None) -> dict[str, int]:
    launch = {
        "grid_x": kernel.launch_model.grid_x,
        "grid_y": kernel.launch_model.grid_y,
        "grid_z": kernel.launch_model.grid_z,
        "block_x": kernel.launch_model.block_x,
        "block_y": kernel.launch_model.block_y,
        "block_z": kernel.launch_model.block_z,
    }
    if override:
        for key, value in override.items():
            if key in launch:
                launch[key] = max(1, int(value))
    return launch


def _infer_problem_size(kernel: KIRKernel, host_values: dict[str, Any]) -> int:
    # Launch extent must describe the logical iteration domain, not just the
    # first declared buffer parameter. Reduction kernels often place a compact
    # temporary/output buffer first, so "first param wins" would silently shrink
    # the launch and produce partial results. We therefore scan the kernel's
    # visible buffer footprint and use the largest participating host buffer as
    # the conservative v1 domain size.
    sizes: list[int] = []
    for param in kernel.signature.params:
        if param.name in host_values and isinstance(host_values[param.name], list):
            sizes.append(len(host_values[param.name]))
    for region in kernel.memory_regions:
        name = str(region.get("name", ""))
        if name in host_values and isinstance(host_values[name], list):
            sizes.append(len(host_values[name]))
    if sizes:
        return max(sizes)
    return 0


def _ensure_host_buffers(kernel: KIRKernel, host_values: dict[str, Any], problem_size: int) -> None:
    for name, _dtype in _kernel_buffers(kernel):
        if name not in host_values:
            host_values[name] = [0] * max(problem_size, 0)


def _kernel_buffers(kernel: KIRKernel) -> list[tuple[str, str]]:
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for param in kernel.signature.params:
        if param.kind != "buffer" and not param.by_ref:
            continue
        if param.name in seen:
            continue
        ordered.append((param.name, param.dtype))
        seen.add(param.name)
    for region in kernel.memory_regions:
        name = str(region.get("name", ""))
        if not name or name in seen:
            continue
        ordered.append((name, str(region.get("dtype", "i32"))))
        seen.add(name)
    return ordered


def _resolve_kernel_output(kernel: KIRKernel, values: dict[str, Any]) -> Any:
    if kernel.memory_regions:
        first_name = str(kernel.memory_regions[0].get("name", ""))
        if first_name and first_name in values and isinstance(values[first_name], list):
            return list(values[first_name])
    return None


def _normalize_host_inputs(inputs: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, value in (inputs or {}).items():
        if isinstance(value, list):
            normalized[str(name)] = list(value)
        else:
            normalized[str(name)] = value
    return normalized


def _dtype_ctype(dtype: str) -> type[Any]:
    normalized = str(dtype).lower()
    if normalized in {"f32", "float"}:
        return ctypes.c_float
    if normalized in {"i64", "int64"}:
        return ctypes.c_longlong
    if normalized in {"bool"}:
        return ctypes.c_bool
    return ctypes.c_int


def _dtype_nbytes(dtype: str) -> int:
    return int(ctypes.sizeof(_dtype_ctype(dtype)))


def _marshal_scalar_value(value: Any, dtype: str, name: str) -> Any:
    ctype = _dtype_ctype(dtype)
    if value is None:
        raise CudaRuntimeError(f"scalar kernel parameter '{name}' is missing")
    return ctype(_coerce_scalar(value, dtype))


def _pack_host_values(values: list[Any], dtype: str) -> tuple[Any | None, int]:
    if not values:
        return None, 0
    ctype = _dtype_ctype(dtype)
    converted = [_coerce_scalar(value, dtype) for value in values]
    array = (ctype * len(converted))(*converted)
    return array, int(ctypes.sizeof(array))


def _unpack_host_values(payload: bytes, dtype: str) -> list[Any]:
    if not payload:
        return []
    ctype = _dtype_ctype(dtype)
    count = len(payload) // ctypes.sizeof(ctype)
    array_type = ctype * count
    values = array_type.from_buffer_copy(payload)
    return [value for value in values]


def _coerce_scalar(value: Any, dtype: str) -> Any:
    normalized = str(dtype).lower()
    if normalized in {"f32", "float"}:
        return float(value)
    if normalized in {"bool"}:
        return bool(value)
    return int(value)


def _load_cuda_driver() -> ctypes.CDLL | None:
    candidates = [
        ctypes.util.find_library("cuda"),
        "nvcuda.dll",
        "libcuda.so",
        "libcuda.so.1",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None
