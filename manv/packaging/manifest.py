"""Deterministic metadata structures for embedded ManV build bundles.

Why this module exists:
- The single-file bundle needs a stable manifest that both build-time and
  runtime code can understand without guessing at bundle layout.
- Tests and future debugging/DAP tooling need reproducible metadata, so this
  module centralizes field ordering and serialization rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class EmbeddedBuildMetadata:
    """Versioned metadata embedded inside the single-file bundle."""

    format_version: int
    manv_version: str
    project_name: str
    entry_module: str
    host_target: str
    build_mode: str
    selection_policy: str
    default_backend: str
    portable_cache_mode: bool
    embedded_artifacts: tuple[str, ...]
    source_hash: str
    hlir_hash: str
    kernel_hash: str
    build_flags: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "manv_version": self.manv_version,
            "project_name": self.project_name,
            "entry_module": self.entry_module,
            "host_target": self.host_target,
            "build_mode": self.build_mode,
            "selection_policy": self.selection_policy,
            "default_backend": self.default_backend,
            "portable_cache_mode": self.portable_cache_mode,
            "embedded_artifacts": list(self.embedded_artifacts),
            "source_hash": self.source_hash,
            "hlir_hash": self.hlir_hash,
            "kernel_hash": self.kernel_hash,
            "build_flags": dict(sorted(self.build_flags.items())),
        }


def build_metadata_json(metadata: EmbeddedBuildMetadata) -> str:
    return json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n"
