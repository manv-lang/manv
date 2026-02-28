"""Device backend selection surface for ManV.

Why this package exists:
- The compiler/runtime previously mixed backend naming, probing, dispatch, and
  textual backend emission inside one compatibility module.
- The single-executable roadmap needs a small, explicit subsystem that can be
  reused by the CLI, packaged runtime, debugger, and tests.

Important invariants:
- Host execution remains the semantic authority for whole-program behavior.
- Device probing and selection must be deterministic so `--report backend`
  output is snapshot-friendly and easy to reason about.
- Backend-name normalization lives here so the rest of the codebase stops
  carrying divergent alias tables.
"""

from .interfaces import (
    BackendId,
    CompileOptions,
    CompiledKernel,
    DeviceBackend,
    HostBackend,
    LaunchConfig,
    LaunchResult,
    ProbeCapabilities,
    ProbeDevice,
    ProbeResult,
    SelectionReport,
    SelectionRequest,
)
from .reporting import render_selection_report
from .resolver import (
    BACKEND_PRIORITY,
    default_device_backends,
    normalize_backend_name,
    resolve_device_selection,
)

__all__ = [
    "BACKEND_PRIORITY",
    "BackendId",
    "CompileOptions",
    "CompiledKernel",
    "DeviceBackend",
    "HostBackend",
    "LaunchConfig",
    "LaunchResult",
    "ProbeCapabilities",
    "ProbeDevice",
    "ProbeResult",
    "SelectionReport",
    "SelectionRequest",
    "default_device_backends",
    "normalize_backend_name",
    "render_selection_report",
    "resolve_device_selection",
]
