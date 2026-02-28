"""Backward-compatible CUDA PTX emission wrapper.

Why this file still exists:
- Older compiler/tests import `manv.backends.cuda_ptx.emit_cuda_ptx`.
- The real implementation now lives under `manv.backends.cuda`.

Behavior:
- If NVRTC is available, this wrapper returns real PTX compiled from generated
  CUDA C++.
- If NVRTC is unavailable, it returns the generated CUDA source prefixed with a
  comment explaining that PTX generation could not complete on this machine.
  That keeps compile artifacts inspectable without pretending a kernel launch
  is possible.
"""

from __future__ import annotations

from typing import Any

from .cuda import compile_cuda_source, emit_cuda_cpp


def emit_cuda_ptx(kernel_ir: dict[str, Any]) -> str:
    source = emit_cuda_cpp(kernel_ir)
    result = compile_cuda_source(source, arch="sm_80")
    if result.success:
        return result.ptx
    return "\n".join(
        [
            "// PTX unavailable on this machine; emitting generated CUDA source instead.",
            f"// nvrtc_log: {result.log}",
            source,
        ]
    )


def emit_cuda_ptx_skeleton(kernel_ir: dict[str, Any]) -> str:
    return emit_cuda_ptx(kernel_ir)
