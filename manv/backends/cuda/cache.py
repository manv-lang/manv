"""Deterministic content-addressed cache for generated CUDA artifacts.

Why this exists:
- NVRTC compilation is expensive enough that repeated compilation should be
  avoided when the IR and toolchain inputs are unchanged.
- Cache behavior must be deterministic so tests can assert stable keys.

The cache intentionally stores both source and PTX:
- `.cu` aids debugging and DAP/source inspection.
- `.ptx` is the executable compilation result when NVRTC is available.
- `.json` records metadata used to compute the cache key.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CudaCacheEntry:
    cache_key: str
    source_path: Path
    ptx_path: Path
    meta_path: Path


def build_cuda_cache_key(
    *,
    kir_hash: str,
    arch: str,
    driver_version: str,
    nvrtc_version: str,
    compile_flags: list[str],
    cuda_source: str,
) -> str:
    payload = {
        "arch": arch,
        "compile_flags": list(compile_flags),
        "cuda_source_sha256": hashlib.sha256(cuda_source.encode("utf-8")).hexdigest(),
        "driver_version": driver_version,
        "kir_hash": kir_hash,
        "nvrtc_version": nvrtc_version,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


class CudaCacheStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def locate(self, cache_key: str) -> CudaCacheEntry:
        return CudaCacheEntry(
            cache_key=cache_key,
            source_path=self.root / f"{cache_key}.cu",
            ptx_path=self.root / f"{cache_key}.ptx",
            meta_path=self.root / f"{cache_key}.json",
        )

    def load(self, cache_key: str) -> tuple[str, str] | None:
        entry = self.locate(cache_key)
        if not (entry.source_path.exists() and entry.ptx_path.exists()):
            return None
        return (
            entry.source_path.read_text(encoding="utf-8"),
            entry.ptx_path.read_text(encoding="utf-8"),
        )

    def store(self, cache_key: str, *, source: str, ptx: str, metadata: dict[str, Any]) -> CudaCacheEntry:
        entry = self.locate(cache_key)
        entry.source_path.write_text(source, encoding="utf-8")
        entry.ptx_path.write_text(ptx, encoding="utf-8")
        entry.meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return entry
