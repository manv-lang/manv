"""Interfaces and report types for ManV host backend selection.

Why this module exists:
- The device resolver already has structured reporting. The host side needs the
  same discipline so compile/build decisions can be tested and documented.
- Keeping host selection separate from LLVM codegen keeps the policy layer
  small and auditable.

Important invariants:
- The host report records what the user asked for and what ManV resolved to.
- Reporting omits timestamps and other non-deterministic data so tests can
  snapshot the exact backend decision output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


HostBackendId = Literal["llvm", "interp"]


@dataclass(frozen=True)
class HostSelectionRequest:
    requested_host_backend: str = "auto"
    policy: str = "auto"


@dataclass(frozen=True)
class HostSelectionReport:
    requested_host_backend: str
    resolved_host_backend: HostBackendId
    policy: str
    versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_host_backend": self.requested_host_backend,
            "resolved_host_backend": self.resolved_host_backend,
            "policy": self.policy,
            "versions": dict(sorted(self.versions.items())),
        }


class HostBackend(ABC):
    """Host runtime/compiler contract shell."""

    @abstractmethod
    def backend_id(self) -> HostBackendId:
        raise NotImplementedError
