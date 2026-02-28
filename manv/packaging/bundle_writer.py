"""Writer for custom one-file ManV Python bundles.

Why this module exists:
- The build pipeline needs a custom artifact format rather than a copied source
  directory.
- Keeping zip assembly here isolates packaging mechanics from compiler logic.

Current scope:
- Builds a self-contained `.mvz` bundle containing the ManV runtime package,
  build metadata, entry source/config, and compiled artifacts.
- Uses the standard-library zip format so the runtime can inspect its own
  payload without platform-specific tooling.
"""

from __future__ import annotations

from pathlib import Path
import zipfile

from .manifest import EmbeddedBuildMetadata, build_metadata_json


def write_python_bundle(
    output_path: Path,
    *,
    package_root: Path,
    metadata: EmbeddedBuildMetadata,
    embedded_files: dict[str, bytes],
) -> Path:
    """Write a deterministic `.mvz` ManV bundle."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_main = (
        "from __future__ import annotations\n"
        "from manv.packaging.bootstrap import run_embedded_bundle\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(run_embedded_bundle())\n"
    ).encode("utf-8")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("__main__.py", bootstrap_main)
        bundle.writestr("manv_embedded/metadata.json", build_metadata_json(metadata).encode("utf-8"))

        # Runtime package contents are copied verbatim so the build artifact no
        # longer depends on the project checkout at execution time.
        for path in sorted(package_root.rglob("*")):
            if path.is_dir():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(package_root.parent).as_posix()
            bundle.writestr(relative, path.read_bytes())

        for relative, payload in sorted(embedded_files.items()):
            bundle.writestr(relative, payload)

    return output_path
