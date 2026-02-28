"""Runtime bootstrap for embedded ManV single-file bundles.

Why this module exists:
- Built bundles need a small runtime entrypoint that can discover embedded
  payloads, materialize program files into a deterministic cache location, and
  invoke the existing host runtime.

Important invariants:
- Bundle extraction is keyed by the bundle hash so repeated runs are stable and
  do not accumulate duplicate directories.
- The extracted payload is treated as build output, not user source of truth.
  The original project checkout is not required after build time.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

from ..runner import run_file


def run_embedded_bundle() -> int:
    bundle_path = Path(sys.argv[0]).resolve()
    metadata = _load_metadata(bundle_path)
    extract_root = _prepare_extract_root(bundle_path, metadata)
    entry_path = _extract_program_payload(bundle_path, extract_root)
    return run_file(
        entry_path,
        stdout=sys.stdout,
        mode="compiled",
        target_name=str(metadata.get("host_target", "x86_64-sysv")),
    )


def _load_metadata(bundle_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(bundle_path, "r") as bundle:
        with bundle.open("manv_embedded/metadata.json") as handle:
            return json.loads(handle.read().decode("utf-8"))


def _prepare_extract_root(bundle_path: Path, metadata: dict[str, object]) -> Path:
    portable = bool(metadata.get("portable_cache_mode", False))
    bundle_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()[:16]
    if portable:
        root = bundle_path.parent / ".manv-runtime-cache"
    else:
        root = _default_runtime_cache_root()
    extract_root = root / bundle_hash
    extract_root.mkdir(parents=True, exist_ok=True)
    return extract_root


def _extract_program_payload(bundle_path: Path, extract_root: Path) -> Path:
    with zipfile.ZipFile(bundle_path, "r") as bundle:
        for member in bundle.namelist():
            if not member.startswith("manv_embedded/program/"):
                continue
            target = extract_root / member.removeprefix("manv_embedded/program/")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as src:
                target.write_bytes(src.read())
    return extract_root / "src" / "main.mv"


def _default_runtime_cache_root() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "ManV" / "runtime_cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "manv" / "runtime_cache"
    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "manv" / "runtime_cache"
    return Path.home() / ".cache" / "manv" / "runtime_cache"
