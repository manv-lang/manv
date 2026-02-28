"""Custom single-file packaging support for ManV builds.

Why this package exists:
- `manv build` is moving away from a copied source tree toward a one-file
  runnable bundle.
- The packaging layer needs explicit metadata and bundle-writing helpers so the
  build pipeline stays deterministic and inspectable.
"""

from .bootstrap import run_embedded_bundle
from .bundle_writer import write_python_bundle
from .manifest import EmbeddedBuildMetadata, build_metadata_json

__all__ = [
    "EmbeddedBuildMetadata",
    "build_metadata_json",
    "run_embedded_bundle",
    "write_python_bundle",
]
