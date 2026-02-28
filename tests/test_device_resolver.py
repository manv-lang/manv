from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.device.interfaces import (
    CompileOptions,
    CompiledKernel,
    DeviceBackend,
    LaunchConfig,
    LaunchResult,
    ProbeCapabilities,
    ProbeDevice,
    ProbeResult,
    SelectionRequest,
)
from manv.device.reporting import render_selection_report
from manv.device.resolver import resolve_device_selection
from manv.gpu_dispatch import backend_selection_report


class _FakeBackend(DeviceBackend):
    def __init__(self, backend: str, *, available: bool, skip_reason: str | None = None) -> None:
        self._backend = backend
        self._available = available
        self._skip_reason = skip_reason

    def backend_id(self) -> str:
        return self._backend

    def probe(self) -> ProbeResult:
        devices = ()
        if self._available:
            devices = (
                ProbeDevice(
                    id="0",
                    name=f"{self._backend} device",
                    vendor=self._backend,
                    capability=ProbeCapabilities(
                        supported_dtypes=("i32", "f32"),
                        max_threads_per_workgroup=256,
                    ),
                ),
            )
        return ProbeResult(
            backend=self._backend,  # type: ignore[arg-type]
            available=self._available,
            devices=devices,
            skip_reason=self._skip_reason,
            versions={"runtime": "test"},
        )

    def init(self, device_id: str | None):
        raise NotImplementedError

    def compile_kernel(self, kernel_ir, options: CompileOptions) -> CompiledKernel:
        raise NotImplementedError

    def launch(self, compiled_kernel: CompiledKernel, packed_args, launch_cfg: LaunchConfig, stream=None) -> LaunchResult:
        raise NotImplementedError

    def alloc(self, nbytes: int, dtype: str):
        raise NotImplementedError

    def free(self, buffer) -> None:
        raise NotImplementedError

    def copy_h2d(self, buffer, values: list[object]) -> None:
        raise NotImplementedError

    def copy_d2h(self, buffer, length: int | None = None) -> list[object]:
        raise NotImplementedError

    def copy_d2d(self, dst, src) -> None:
        raise NotImplementedError

    def sync(self, stream=None) -> None:
        raise NotImplementedError

    def last_error(self) -> str:
        return ""


def _fake_backends() -> list[DeviceBackend]:
    return [
        _FakeBackend("cuda", available=False, skip_reason="CUDA_DRIVER_UNAVAILABLE"),
        _FakeBackend("rocm", available=True),
        _FakeBackend("level0", available=False, skip_reason="LEVEL0_RUNTIME_UNAVAILABLE"),
        _FakeBackend("vulkan-spv", available=False, skip_reason="VULKAN_RUNTIME_UNAVAILABLE"),
        _FakeBackend("directx", available=False, skip_reason="DIRECTX_RUNTIME_UNAVAILABLE"),
        _FakeBackend("webgpu", available=False, skip_reason="WEBGPU_RUNTIME_UNAVAILABLE"),
        _FakeBackend("cpu", available=True),
    ]


def test_auto_selection_uses_priority_order_with_mocked_probes() -> None:
    report = resolve_device_selection(
        SelectionRequest(requested_backend="auto", requested_device=None, policy="run"),
        backends=_fake_backends(),
    )

    assert report.selected_backend == "rocm"
    assert report.selected_device == "0"
    assert [candidate.backend for candidate in report.candidates] == ["cuda", "rocm"]
    assert report.skip_reasons == {"cuda": "CUDA_DRIVER_UNAVAILABLE"}


def test_explicit_backend_override_is_preserved_in_report() -> None:
    report = resolve_device_selection(
        SelectionRequest(requested_backend="directx", requested_device=None, policy="run"),
        backends=_fake_backends(),
    )

    assert report.selected_backend == "directx"
    assert report.selected_device is None
    assert [candidate.backend for candidate in report.candidates] == ["directx"]
    assert report.skip_reasons == {"directx": "DIRECTX_RUNTIME_UNAVAILABLE"}


def test_selection_report_rendering_is_deterministic() -> None:
    report1 = resolve_device_selection(
        SelectionRequest(requested_backend="auto", requested_device=None, policy="compile"),
        backends=_fake_backends(),
    )
    report2 = resolve_device_selection(
        SelectionRequest(requested_backend="auto", requested_device=None, policy="compile"),
        backends=_fake_backends(),
    )

    rendered1 = render_selection_report(report1)
    rendered2 = render_selection_report(report2)

    assert rendered1 == rendered2
    assert '"selected_backend": "rocm"' in rendered1
    assert "timestamp" not in rendered1


def test_dispatch_report_honors_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("MANV_BACKEND", "cpu")
    monkeypatch.setenv("MANV_DEVICE", "env-device")

    report = backend_selection_report("auto", policy="run")

    assert report.requested_backend == "cpu"
    assert report.requested_device == "env-device"
    assert report.selected_backend == "cpu"
    assert report.selected_device == "env-device"
