"""Deterministic host backend resolution.

Why this module exists:
- `manv build` and `manv compile` now need a stable host-backend policy that is
  independent from GPU device resolution.
- The selection rule is intentionally simple: `auto` means LLVM unless the user
  explicitly requests interpreter mode.

Important invariants:
- `auto` always resolves to `llvm`.
- Alias handling is performed only at the edge so the rest of the compiler can
  reason about canonical ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..llvm_toolchain import detect_llvm_toolchain
from .interfaces import HostBackend, HostBackendId, HostSelectionReport, HostSelectionRequest


NormalizedHostBackendName = HostBackendId | Literal["auto"]

_HOST_ALIASES: dict[str, NormalizedHostBackendName] = {
    "auto": "auto",
    "llvm": "llvm",
    "native": "llvm",
    "interp": "interp",
    "interpreter": "interp",
}


def normalize_host_backend_name(name: str) -> NormalizedHostBackendName:
    raw = str(name).strip().lower()
    normalized = raw.replace("_", "-")
    resolved = _HOST_ALIASES.get(raw) or _HOST_ALIASES.get(normalized)
    if resolved is None:
        raise RuntimeError(f"unsupported host backend '{name}'")
    return resolved


@dataclass(frozen=True)
class InterpreterHostBackend(HostBackend):
    def backend_id(self) -> HostBackendId:
        return "interp"


@dataclass(frozen=True)
class LlvmHostBackend(HostBackend):
    def backend_id(self) -> HostBackendId:
        return "llvm"


def resolve_host_selection(request: HostSelectionRequest) -> HostSelectionReport:
    normalized = normalize_host_backend_name(request.requested_host_backend)
    resolved: HostBackendId = "llvm" if normalized == "auto" else normalized
    versions: dict[str, str] = {}

    if resolved == "llvm":
        toolchain = detect_llvm_toolchain()
        if toolchain is not None:
            versions = {
                "llvm.clang": toolchain.clang,
                "llvm.version": toolchain.version_text,
            }
        else:
            versions = {"llvm.version": "unavailable"}

    return HostSelectionReport(
        requested_host_backend=request.requested_host_backend,
        resolved_host_backend=resolved,
        policy=request.policy,
        versions=versions,
    )
