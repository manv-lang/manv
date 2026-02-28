"""CUDA backend entrypoints for ManV.

Why this package exists:
- It isolates the real CUDA-facing implementation from the older mock backend
  surface so the rest of the compiler can migrate incrementally.
- It keeps the NVRTC/compiler, cache, runtime, and eligibility logic in one
  place with explicit contracts between them.
- It centralizes the "best effort vs required" behavior used by `@gpu` calls.

Key invariants:
- HLIR remains the semantic authority. This package may compile or execute only
  what the IR already decided, and must not invent new language meaning.
- Cache keys must be deterministic for the same kernel/module and toolchain.
- Runtime availability checks must fail closed: when CUDA cannot be proven
  available, best-effort callers fall back and required callers error.
"""

from __future__ import annotations

from .cache import CudaCacheEntry, CudaCacheStore, build_cuda_cache_key
from .codegen import emit_cuda_cpp
from .eligibility import GpuEligibilityReport, analyze_hlir_gpu_function
from .nvrtc_compiler import NvrtcCompileResult, compile_cuda_source, nvrtc_is_available
from .runtime import CudaRuntime, cuda_is_available

__all__ = [
    "CudaCacheEntry",
    "CudaCacheStore",
    "CudaRuntime",
    "GpuEligibilityReport",
    "NvrtcCompileResult",
    "analyze_hlir_gpu_function",
    "build_cuda_cache_key",
    "compile_cuda_source",
    "cuda_is_available",
    "emit_cuda_cpp",
    "nvrtc_is_available",
]
