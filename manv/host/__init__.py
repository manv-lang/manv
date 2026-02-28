"""Host backend selection and reporting.

Why this package exists:
- ManV now distinguishes the CPU-side host backend from optional GPU device
  backends. The host backend decides how normal program code runs; the device
  backend only accelerates `@gpu` regions.
- `manv build` and `manv compile` need a stable place to resolve "auto" host
  behavior without coupling CLI code to LLVM/interpreter implementation
  details.

Important invariants:
- Host-backend selection is deterministic. There is no probing priority list:
  `auto` always resolves to `llvm`.
- The interpreter backend remains available for debugging and parity checks,
  but it is never the resolved default once host backend selection is active.
"""

from .interfaces import HostBackend, HostBackendId, HostSelectionReport, HostSelectionRequest
from .reporting import render_host_selection_report, render_joint_backend_report
from .resolver import (
    InterpreterHostBackend,
    LlvmHostBackend,
    normalize_host_backend_name,
    resolve_host_selection,
)

__all__ = [
    "HostBackend",
    "HostBackendId",
    "HostSelectionReport",
    "HostSelectionRequest",
    "InterpreterHostBackend",
    "LlvmHostBackend",
    "normalize_host_backend_name",
    "resolve_host_selection",
    "render_host_selection_report",
    "render_joint_backend_report",
]
