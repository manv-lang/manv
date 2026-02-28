from __future__ import annotations

from manv.host import HostSelectionRequest, normalize_host_backend_name, resolve_host_selection


def test_host_backend_auto_resolves_to_llvm() -> None:
    report = resolve_host_selection(HostSelectionRequest(requested_host_backend="auto", policy="compile"))
    assert report.resolved_host_backend == "llvm"
    assert report.requested_host_backend == "auto"


def test_host_backend_override_to_interp() -> None:
    report = resolve_host_selection(HostSelectionRequest(requested_host_backend="interp", policy="build"))
    assert report.resolved_host_backend == "interp"


def test_host_backend_aliases_are_stable() -> None:
    assert normalize_host_backend_name("interpreter") == "interp"
    assert normalize_host_backend_name("native") == "llvm"
