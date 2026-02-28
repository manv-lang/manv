"""Device backend interfaces shared by host/runtime/backend selection code.

Why this file exists:
- ManV previously mixed backend selection, textual backend emission, and
  runtime execution inside a single compatibility layer.
- The single-executable and auto-selection roadmap needs a stable contract that
  can be consumed by the CLI, the packaged runtime, and backend-specific
  modules without re-deciding what a "device backend" means each time.

Important invariants:
- Host execution remains the semantic authority for full-program behavior.
- Device backends accelerate already-lowered kernel payloads only; they do not
  own control flow, exceptions, imports, or other language semantics.
- Probe and selection reporting must be deterministic so tests can snapshot the
  exact backend decision process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from ..kernel_ir import KIRModule


BackendId = Literal["cpu", "cuda", "rocm", "level0", "vulkan-spv", "directx", "webgpu"]


@dataclass(frozen=True)
class ProbeCapabilities:
    """Minimal capability summary exposed by backend probing.

    The probe stage intentionally reports only low-risk, device-level facts.
    Higher-level execution eligibility still belongs to HLIR/kernelization.
    """

    supported_dtypes: tuple[str, ...]
    max_threads_per_workgroup: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_dtypes": list(self.supported_dtypes),
            "max_threads_per_workgroup": self.max_threads_per_workgroup,
        }


@dataclass(frozen=True)
class ProbeDevice:
    """One device candidate discovered during probing."""

    id: str
    name: str
    vendor: str
    capability: ProbeCapabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "vendor": self.vendor,
            "capability": self.capability.to_dict(),
        }


@dataclass(frozen=True)
class ProbeResult:
    """Structured result from probing a backend.

    `skip_reason` is always stable and single-valued. Keeping one deterministic
    reason per backend makes CLI and test output easier to reason about than a
    variable-length list of nested failures.
    """

    backend: BackendId
    available: bool
    devices: tuple[ProbeDevice, ...] = ()
    skip_reason: str | None = None
    versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "devices": [device.to_dict() for device in self.devices],
            "skip_reason": self.skip_reason,
            "versions": dict(sorted(self.versions.items())),
        }


@dataclass(frozen=True)
class CompileOptions:
    arch: str = "generic"
    debug: bool = False
    extra_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchConfig:
    grid_x: int
    block_x: int
    grid_y: int = 1
    grid_z: int = 1
    block_y: int = 1
    block_z: int = 1
    shared_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "grid_z": self.grid_z,
            "block_x": self.block_x,
            "block_y": self.block_y,
            "block_z": self.block_z,
            "shared_bytes": self.shared_bytes,
        }


PackedLaunchArgs = dict[str, Any]


@dataclass(frozen=True)
class CompiledKernel:
    backend: BackendId
    target: str
    entrypoints: tuple[str, ...]
    binaries: dict[str, str]
    reflection: dict[str, Any]
    compile_log: tuple[str, ...]
    cache_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "target": self.target,
            "entrypoints": list(self.entrypoints),
            "binaries": dict(self.binaries),
            "reflection": dict(self.reflection),
            "compile_log": list(self.compile_log),
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True)
class LaunchResult:
    executed_on_device: bool
    outputs: dict[str, Any]
    runtime_meta: dict[str, Any]


@dataclass(frozen=True)
class SelectionRequest:
    """Backend selection inputs shared by CLI, runtime, and tests."""

    requested_backend: str = "auto"
    requested_device: str | None = None
    policy: str = "auto"


@dataclass(frozen=True)
class SelectionReport:
    """Deterministic backend selection output.

    The report is the single source of truth for:
    - CLI `--report backend`
    - built executable startup diagnostics
    - tests asserting ordering/skip reasons
    """

    policy: str
    requested_backend: str
    requested_device: str | None
    selected_backend: BackendId
    selected_device: str | None
    candidates: tuple[ProbeResult, ...]
    skip_reasons: dict[str, str]
    versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "requested_backend": self.requested_backend,
            "requested_device": self.requested_device,
            "selected_backend": self.selected_backend,
            "selected_device": self.selected_device,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "skip_reasons": dict(sorted(self.skip_reasons.items())),
            "versions": dict(sorted(self.versions.items())),
        }


class HostBackend(ABC):
    """Host runtime contract.

    This is intentionally small in the first phase because the current runtime
    already exists. The goal is to establish the boundary, not to rewrite the
    host runtime all at once.
    """

    @abstractmethod
    def backend_id(self) -> BackendId:
        raise NotImplementedError


class DeviceBackend(ABC):
    """Optional accelerator contract used by runtime selection.

    Compile/launch methods are part of the interface now even though some
    backends are probe-only in early phases. This keeps later phases additive
    instead of forcing a second contract rewrite.
    """

    @abstractmethod
    def backend_id(self) -> BackendId:
        raise NotImplementedError

    @abstractmethod
    def probe(self) -> ProbeResult:
        raise NotImplementedError

    @abstractmethod
    def init(self, device_id: str | None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def compile_kernel(self, kernel_ir: KIRModule | dict[str, Any], options: CompileOptions) -> CompiledKernel:
        raise NotImplementedError

    @abstractmethod
    def launch(
        self,
        compiled_kernel: CompiledKernel,
        packed_args: PackedLaunchArgs,
        launch_cfg: LaunchConfig,
        stream: Any | None = None,
    ) -> LaunchResult:
        raise NotImplementedError

    @abstractmethod
    def alloc(self, nbytes: int, dtype: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def free(self, buffer: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def copy_h2d(self, buffer: Any, values: list[Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def copy_d2h(self, buffer: Any, length: int | None = None) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def copy_d2d(self, dst: Any, src: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def sync(self, stream: Any | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def last_error(self) -> str:
        raise NotImplementedError
