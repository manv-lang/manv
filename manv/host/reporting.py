"""Rendering helpers for host/device backend reports."""

from __future__ import annotations

import json

from ..device.interfaces import SelectionReport as DeviceSelectionReport
from .interfaces import HostSelectionReport


def render_host_selection_report(report: HostSelectionReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_joint_backend_report(host_report: HostSelectionReport, device_report: DeviceSelectionReport) -> str:
    payload = {
        **host_report.to_dict(),
        "requested_device_backend": device_report.requested_backend,
        "resolved_device_backend": device_report.selected_backend,
        # Keep the legacy top-level key during the transition so existing
        # reports and tests that still look for `selected_backend` do not need
        # a flag day.
        "selected_backend": device_report.selected_backend,
        "requested_device": device_report.requested_device,
        "selected_device": device_report.selected_device,
        "candidates": [candidate.to_dict() for candidate in device_report.candidates],
        "skip_reasons": dict(sorted(device_report.skip_reasons.items())),
        "versions": {
            **dict(sorted(host_report.versions.items())),
            **dict(sorted(device_report.versions.items())),
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
