"""Deterministic rendering helpers for backend selection reports.

Why this module exists:
- The resolver returns structured data, but several frontends need a stable
  textual representation: CLI, tests, and eventually packaged runtime logs.
- Keeping rendering separate from selection logic prevents the resolver from
  accreting UI concerns.
"""

from __future__ import annotations

import json

from .interfaces import SelectionReport


def render_selection_report(report: SelectionReport) -> str:
    """Render a backend selection report as deterministic JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
