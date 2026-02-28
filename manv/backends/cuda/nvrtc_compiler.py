"""NVRTC compilation helpers.

Why this module exists:
- It keeps all dynamic-library discovery and NVRTC error mapping in one place.
- The rest of the backend should be able to ask a simple question: "can this
  CUDA source be turned into PTX right now?"

Failure model:
- Missing libraries are reported as availability=false rather than raising.
- Compile failures return structured logs and optional source snippets so the
  caller can surface deterministic diagnostics.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass, field
from typing import Any


def nvrtc_is_available() -> bool:
    return _load_nvrtc_library() is not None


@dataclass
class NvrtcCompileResult:
    available: bool
    success: bool
    ptx: str = ""
    log: str = ""
    version: str = "unavailable"
    flags: list[str] = field(default_factory=list)


def compile_cuda_source(source: str, *, arch: str, extra_flags: list[str] | None = None) -> NvrtcCompileResult:
    flags = [f"--gpu-architecture={arch}", *(extra_flags or [])]
    lib = _load_nvrtc_library()
    if lib is None:
        return NvrtcCompileResult(
            available=False,
            success=False,
            log="NVRTC library not found; CUDA source was generated but cannot be JIT-compiled on this machine.",
            flags=flags,
        )

    version = _nvrtc_version(lib)
    try:
        return _compile_with_nvrtc(lib, source, flags=flags, version=version)
    except Exception as err:  # pragma: no cover - exercised only on real NVRTC installations
        return NvrtcCompileResult(
            available=True,
            success=False,
            log=f"NVRTC invocation failed: {err}",
            version=version,
            flags=flags,
        )


def _compile_with_nvrtc(lib: ctypes.CDLL, source: str, *, flags: list[str], version: str) -> NvrtcCompileResult:
    nvrtcProgram = ctypes.c_void_p
    program = nvrtcProgram()
    src_bytes = source.encode("utf-8")
    name_bytes = b"manv_kernel.cu"

    lib.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(nvrtcProgram),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.nvrtcCompileProgram.argtypes = [nvrtcProgram, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.nvrtcGetProgramLogSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetProgramLog.argtypes = [nvrtcProgram, ctypes.c_char_p]
    lib.nvrtcGetPTXSize.argtypes = [nvrtcProgram, ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetPTX.argtypes = [nvrtcProgram, ctypes.c_char_p]
    lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(nvrtcProgram)]

    result = lib.nvrtcCreateProgram(ctypes.byref(program), src_bytes, name_bytes, 0, None, None)
    if result != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed with code {result}")

    flag_values = (ctypes.c_char_p * len(flags))(*(flag.encode("utf-8") for flag in flags)) if flags else None
    compile_code = lib.nvrtcCompileProgram(program, len(flags), flag_values)
    log = _program_log(lib, program)
    if compile_code != 0:
        lib.nvrtcDestroyProgram(ctypes.byref(program))
        return NvrtcCompileResult(available=True, success=False, log=log, version=version, flags=flags)

    ptx_size = ctypes.c_size_t()
    if lib.nvrtcGetPTXSize(program, ctypes.byref(ptx_size)) != 0:
        lib.nvrtcDestroyProgram(ctypes.byref(program))
        raise RuntimeError("nvrtcGetPTXSize failed")
    buffer = ctypes.create_string_buffer(ptx_size.value)
    if lib.nvrtcGetPTX(program, buffer) != 0:
        lib.nvrtcDestroyProgram(ctypes.byref(program))
        raise RuntimeError("nvrtcGetPTX failed")

    lib.nvrtcDestroyProgram(ctypes.byref(program))
    return NvrtcCompileResult(
        available=True,
        success=True,
        ptx=buffer.value.decode("utf-8", errors="replace"),
        log=log,
        version=version,
        flags=flags,
    )


def _program_log(lib: ctypes.CDLL, program: ctypes.c_void_p) -> str:
    size = ctypes.c_size_t()
    if lib.nvrtcGetProgramLogSize(program, ctypes.byref(size)) != 0:
        return ""
    if size.value == 0:
        return ""
    buffer = ctypes.create_string_buffer(size.value)
    if lib.nvrtcGetProgramLog(program, buffer) != 0:
        return ""
    return buffer.value.decode("utf-8", errors="replace")


def _nvrtc_version(lib: ctypes.CDLL) -> str:
    major = ctypes.c_int()
    minor = ctypes.c_int()
    fn = getattr(lib, "nvrtcVersion", None)
    if fn is None:
        return "unknown"
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    if fn(ctypes.byref(major), ctypes.byref(minor)) != 0:
        return "unknown"
    return f"{major.value}.{minor.value}"


def _load_nvrtc_library() -> ctypes.CDLL | None:
    candidates = [
        ctypes.util.find_library("nvrtc"),
        "nvrtc64_120_0.dll",
        "nvrtc64_122_0.dll",
        "nvrtc64_124_0.dll",
        "libnvrtc.so",
        "libnvrtc.so.12",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None
