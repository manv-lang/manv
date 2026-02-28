"""Deterministic device backend probing and selection.

Why this module exists:
- Runtime backend selection is now a first-class subsystem rather than an
  optimistic hard-coded choice.
- The same selection rules must be shared by the CLI, runtime, tests, and
  future single-executable bootstrap path.

Important invariants:
- Auto selection always walks the exact priority order requested by ManV:
  CUDA -> ROCm -> Level Zero -> Vulkan -> DirectX -> WebGPU -> CPU.
- Explicit backend requests are normalized deterministically and preserved in
  the report even if the backend is unavailable. This makes override behavior
  auditable rather than silently rewritten.
- Probe-only backends are allowed to report availability before they support
  real kernel execution. Selection and execution remain separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, Literal, cast

from ..backends.cuda import CudaRuntime, nvrtc_is_available
from .dynamic_loader import try_load_any
from .interfaces import (
    BackendId,
    CompileOptions,
    CompiledKernel,
    DeviceBackend,
    LaunchConfig,
    LaunchResult,
    ProbeCapabilities,
    ProbeDevice,
    ProbeResult,
    SelectionReport,
    SelectionRequest,
)


NormalizedBackendName = BackendId | Literal["auto"]

BACKEND_PRIORITY: tuple[BackendId, ...] = (
    "cuda",
    "rocm",
    "level0",
    "vulkan-spv",
    "directx",
    "webgpu",
    "cpu",
)

_BACKEND_ALIASES: dict[str, NormalizedBackendName] = {
    "auto": "auto",
    "cuda": "cuda",
    "cuda_ptx": "cuda",
    "cuda-ptx": "cuda",
    "rocm": "rocm",
    "hip": "rocm",
    "level0": "level0",
    "level-zero": "level0",
    "level_zero": "level0",
    "oneapi": "level0",
    "vulkan": "vulkan-spv",
    "vulkan-spv": "vulkan-spv",
    "vulkan_spv": "vulkan-spv",
    "vulkan_spirv": "vulkan-spv",
    "spirv": "vulkan-spv",
    "directx": "directx",
    "dx": "directx",
    "webgpu": "webgpu",
    "cpu": "cpu",
    "cpu_ref": "cpu",
    "cpu-reference": "cpu",
    "cpu_reference": "cpu",
}


def normalize_backend_name(name: str) -> NormalizedBackendName:
    """Normalize CLI/runtime backend names onto the canonical backend ids."""

    raw = str(name).strip().lower()
    normalized = raw.replace("_", "-")
    # Some historical callers use `cpu_ref` and `cuda_ptx`, which we preserve
    # only at the input edge. Downstream code should never reintroduce them.
    resolved = _BACKEND_ALIASES.get(raw) or _BACKEND_ALIASES.get(normalized)
    if resolved is None:
        raise RuntimeError(f"unsupported backend '{name}'")
    return resolved


class _ProbeOnlyBackend(DeviceBackend):
    """Base class for probe-first backends.

    Early phases only need deterministic probing. Compile/launch methods are
    intentionally stable `NotImplementedError`s so explicit selection failures
    stay predictable until those backends gain real execution support.
    """

    _backend: BackendId = "cpu"
    _vendor: str = "Unknown"

    def backend_id(self) -> BackendId:
        return self._backend

    def init(self, device_id: str | None) -> object:
        raise NotImplementedError(f"{self._backend} runtime initialization is not implemented yet")

    def compile_kernel(self, kernel_ir, options: CompileOptions) -> CompiledKernel:
        del kernel_ir, options
        raise NotImplementedError(f"{self._backend} kernel compilation is not implemented yet")

    def launch(self, compiled_kernel: CompiledKernel, packed_args, launch_cfg: LaunchConfig, stream=None) -> LaunchResult:
        del compiled_kernel, packed_args, launch_cfg, stream
        raise NotImplementedError(f"{self._backend} kernel launch is not implemented yet")

    def alloc(self, nbytes: int, dtype: str) -> object:
        del nbytes, dtype
        raise NotImplementedError(f"{self._backend} memory allocation is not implemented yet")

    def free(self, buffer: object) -> None:
        del buffer
        raise NotImplementedError(f"{self._backend} memory free is not implemented yet")

    def copy_h2d(self, buffer: object, values: list[object]) -> None:
        del buffer, values
        raise NotImplementedError(f"{self._backend} host-to-device copy is not implemented yet")

    def copy_d2h(self, buffer: object, length: int | None = None) -> list[object]:
        del buffer, length
        raise NotImplementedError(f"{self._backend} device-to-host copy is not implemented yet")

    def copy_d2d(self, dst: object, src: object) -> None:
        del dst, src
        raise NotImplementedError(f"{self._backend} device-to-device copy is not implemented yet")

    def sync(self, stream: object | None = None) -> None:
        del stream
        raise NotImplementedError(f"{self._backend} synchronization is not implemented yet")

    def last_error(self) -> str:
        return ""


class CpuDeviceBackend(_ProbeOnlyBackend):
    _backend: BackendId = "cpu"
    _vendor = "CPU"

    def probe(self) -> ProbeResult:
        return ProbeResult(
            backend="cpu",
            available=True,
            devices=(
                ProbeDevice(
                    id="host",
                    name="Host CPU",
                    vendor="CPU",
                    capability=ProbeCapabilities(
                        supported_dtypes=("bool", "i32", "i64", "f32"),
                        max_threads_per_workgroup=8192,
                    ),
                ),
            ),
            skip_reason=None,
            versions={},
        )


class CudaDeviceBackend(_ProbeOnlyBackend):
    _backend: BackendId = "cuda"
    _vendor = "NVIDIA"

    def probe(self) -> ProbeResult:
        runtime = CudaRuntime()
        driver_version = getattr(runtime, "_driver_version", lambda: "unknown")()
        versions = {
            "driver": str(driver_version),
            "nvrtc": "available" if nvrtc_is_available() else "missing",
        }
        if not runtime.available:
            return ProbeResult(
                backend="cuda",
                available=False,
                devices=(),
                skip_reason="CUDA_DRIVER_UNAVAILABLE",
                versions=versions,
            )
        if not nvrtc_is_available():
            return ProbeResult(
                backend="cuda",
                available=False,
                devices=(),
                skip_reason="CUDA_NVRTC_UNAVAILABLE",
                versions=versions,
            )
        device_count = runtime.device_count()
        if device_count <= 0:
            return ProbeResult(
                backend="cuda",
                available=False,
                devices=(),
                skip_reason="CUDA_NO_DEVICES",
                versions=versions,
            )
        devices = tuple(
            ProbeDevice(
                id=str(index),
                name=f"CUDA Device {index}",
                vendor="NVIDIA",
                capability=ProbeCapabilities(
                    supported_dtypes=("bool", "i32", "i64", "f32"),
                    max_threads_per_workgroup=1024,
                ),
            )
            for index in range(device_count)
        )
        return ProbeResult(
            backend="cuda",
            available=True,
            devices=devices,
            skip_reason=None,
            versions=versions,
        )


class _DynamicLibraryBackend(_ProbeOnlyBackend):
    """Backend whose early-phase probe is based on shared-library discovery."""

    def __init__(
        self,
        *,
        _backend: BackendId,
        _vendor: str,
        _candidates: tuple[str, ...],
        _missing_reason: str,
        _platform_reason: str | None = None,
        _supported_dtypes: tuple[str, ...] = ("bool", "i32", "i64", "f32"),
        _max_threads: int = 1024,
    ) -> None:
        self._backend = _backend
        self._vendor = _vendor
        self._candidates = _candidates
        self._missing_reason = _missing_reason
        self._platform_reason = _platform_reason
        self._supported_dtypes = _supported_dtypes
        self._max_threads = _max_threads

    def probe(self) -> ProbeResult:
        if self._platform_reason is not None:
            return ProbeResult(
                backend=self._backend,
                available=False,
                devices=(),
                skip_reason=self._platform_reason,
                versions={},
            )
        loaded = try_load_any(self._candidates, missing_reason=self._missing_reason)
        if not loaded.available:
            return ProbeResult(
                backend=self._backend,
                available=False,
                devices=(),
                skip_reason=loaded.skip_reason,
                versions={},
            )
        return ProbeResult(
            backend=self._backend,
            available=True,
            devices=(
                ProbeDevice(
                    id="0",
                    name=f"{self._vendor} Device 0",
                    vendor=self._vendor,
                    capability=ProbeCapabilities(
                        supported_dtypes=self._supported_dtypes,
                        max_threads_per_workgroup=self._max_threads,
                    ),
                ),
            ),
            skip_reason=None,
            versions={"loader": str(loaded.loaded_name or "")},
        )


def default_device_backends() -> tuple[DeviceBackend, ...]:
    """Return backends in the exact priority order used by auto selection."""

    directx_platform_reason = None if os.name == "nt" else "DIRECTX_PLATFORM_UNSUPPORTED"
    return (
        CudaDeviceBackend(),
        _DynamicLibraryBackend(
            _backend="rocm",
            _vendor="AMD",
            _candidates=("amdhip64", "amdhip64.dll", "libamdhip64.so"),
            _missing_reason="ROCM_RUNTIME_UNAVAILABLE",
        ),
        _DynamicLibraryBackend(
            _backend="level0",
            _vendor="Intel",
            _candidates=("ze_loader", "ze_loader.dll", "libze_loader.so"),
            _missing_reason="LEVEL0_RUNTIME_UNAVAILABLE",
        ),
        _DynamicLibraryBackend(
            _backend="vulkan-spv",
            _vendor="Khronos",
            _candidates=("vulkan-1", "vulkan-1.dll", "libvulkan.so.1", "libvulkan.so"),
            _missing_reason="VULKAN_RUNTIME_UNAVAILABLE",
        ),
        _DynamicLibraryBackend(
            _backend="directx",
            _vendor="Microsoft",
            _candidates=("d3d12", "d3d12.dll"),
            _missing_reason="DIRECTX_RUNTIME_UNAVAILABLE",
            _platform_reason=directx_platform_reason,
        ),
        _DynamicLibraryBackend(
            _backend="webgpu",
            _vendor="WebGPU",
            _candidates=("wgpu_native", "wgpu_native.dll", "libwgpu_native.so", "libwgpu_native.dylib"),
            _missing_reason="WEBGPU_RUNTIME_UNAVAILABLE",
        ),
        CpuDeviceBackend(),
    )


def resolve_device_selection(
    request: SelectionRequest,
    *,
    backends: Iterable[DeviceBackend] | None = None,
) -> SelectionReport:
    """Resolve the effective backend/device selection for runtime use."""

    backend_name = normalize_backend_name(request.requested_backend)
    ordered_backends = tuple(backends or default_device_backends())
    backend_map = {backend.backend_id(): backend for backend in ordered_backends}

    if backend_name != "auto":
        explicit = cast(BackendId, backend_name)
        backend = backend_map[explicit]
        probe = backend.probe()
        selected_device = request.requested_device
        if selected_device is None and probe.devices:
            selected_device = probe.devices[0].id
        versions = _flatten_versions([probe])
        skip_reasons = {probe.backend: probe.skip_reason} if probe.skip_reason else {}
        return SelectionReport(
            policy=request.policy,
            requested_backend=request.requested_backend,
            requested_device=request.requested_device,
            selected_backend=explicit,
            selected_device=selected_device,
            candidates=(probe,),
            skip_reasons=skip_reasons,
            versions=versions,
        )

    probes: list[ProbeResult] = []
    selected_backend: BackendId = "cpu"
    selected_device: str | None = "host"

    # Auto-selection ordering is an invariant because future packaged-runtime
    # diagnostics and test snapshots depend on this exact, stable priority list.
    for backend_id in BACKEND_PRIORITY:
        probe = backend_map[backend_id].probe()
        probes.append(probe)
        if probe.available:
            selected_backend = probe.backend
            selected_device = request.requested_device or (probe.devices[0].id if probe.devices else None)
            break

    skip_reasons = {
        probe.backend: probe.skip_reason
        for probe in probes
        if probe.skip_reason is not None
    }
    return SelectionReport(
        policy=request.policy,
        requested_backend=request.requested_backend,
        requested_device=request.requested_device,
        selected_backend=selected_backend,
        selected_device=selected_device,
        candidates=tuple(probes),
        skip_reasons=skip_reasons,
        versions=_flatten_versions(probes),
    )


def _flatten_versions(probes: Iterable[ProbeResult]) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for probe in probes:
        for key, value in sorted(probe.versions.items()):
            flattened[f"{probe.backend}.{key}"] = str(value)
    return flattened
