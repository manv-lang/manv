"""ManV C FFI engine — ctypes-based library loading and type marshaling.

Why this module exists:
- Centralizes library resolution, type mapping, and argument marshaling so
  the interpreter and any future direct-call paths share a single source of
  truth for C ABI semantics.
- Keeps the interpreter.py free of platform-detection and ctypes boilerplate.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import Any

from .memory import ArrayRef, Pointer


# ---------------------------------------------------------------------------
# Library resolution and caching
# ---------------------------------------------------------------------------

_LIB_CACHE: dict[str, ctypes.CDLL] = {}


def resolve_lib(name: str) -> str:
    """Map a short name like ``'c'`` to a platform-specific shared library path."""
    if os.path.isabs(name) or os.sep in name or ("/" in name):
        return name  # absolute or relative path — use directly

    found = ctypes.util.find_library(name)
    if found:
        return found

    # Platform fallback guesses when find_library returns None.
    if sys.platform == "darwin":
        return f"lib{name}.dylib"
    if sys.platform == "win32":
        return f"{name}.dll"
    return f"lib{name}.so"


_WINDOWS_LIB_FALLBACKS: dict[str, list[str]] = {
    "c": ["msvcrt", "ucrtbase"],
    "m": ["msvcrt"],
}


def load_library(name: str) -> ctypes.CDLL:
    """Load (or return the cached) CDLL for the given library identifier."""
    if name not in _LIB_CACHE:
        path = resolve_lib(name)
        try:
            _LIB_CACHE[name] = ctypes.CDLL(path)
        except OSError:
            if sys.platform == "win32":
                for candidate in _WINDOWS_LIB_FALLBACKS.get(name, []):
                    try:
                        _LIB_CACHE[name] = ctypes.CDLL(candidate)
                        break
                    except OSError:
                        continue
            if name not in _LIB_CACHE:
                raise
    return _LIB_CACHE[name]


# ---------------------------------------------------------------------------
# ManV → ctypes type mapping
# ---------------------------------------------------------------------------

VOID_RETURN: frozenset[str | None] = frozenset({None, "void"})

MANV_TO_CTYPE: dict[str, Any] = {
    "int":   ctypes.c_long,
    "i32":   ctypes.c_int,
    "i64":   ctypes.c_int64,
    "float": ctypes.c_double,
    "f32":   ctypes.c_float,
    "f64":   ctypes.c_double,
    "bool":  ctypes.c_bool,
    "ptr":   ctypes.c_void_p,
    "str":   ctypes.c_char_p,
    "u8":    ctypes.c_uint8,
    "usize": ctypes.c_size_t,
}


def _is_pointer_type(t: str | None) -> bool:
    return t == "ptr" or str(t or "").rstrip().endswith("*")


def manv_to_ctype(t: str | None) -> Any:
    """Return the ctypes type for a ManV type name, or ``None`` for void."""
    if t in VOID_RETURN:
        return None
    if _is_pointer_type(t):
        return ctypes.c_void_p
    return MANV_TO_CTYPE.get(str(t), ctypes.c_long)


# ---------------------------------------------------------------------------
# Value marshaling (ManV runtime values → C-compatible ctypes values)
# ---------------------------------------------------------------------------

def marshal_arg(value: Any, type_name: str | None) -> Any:
    """Convert a ManV runtime value to a C-compatible argument."""
    t = str(type_name or "")
    if isinstance(value, (Pointer, ArrayRef)):
        return value.address
    if _is_pointer_type(t) and value is None:
        return 0
    if t == "str":
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return value or b""
    if t == "ptr" or _is_pointer_type(t):
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return int(value) if value is not None else 0
    if t in {"float", "f32", "f64"}:
        return float(value)
    if t == "bool":
        return bool(value)
    if t in {"int", "i32", "i64", "u8", "usize"}:
        return int(value)
    return value


def unmarshal_return(raw: Any, return_type: str | None) -> Any:
    """Convert a ctypes return value back to a ManV runtime value."""
    if return_type in VOID_RETURN:
        return None
    if _is_pointer_type(return_type):
        text = str(return_type).replace("const ", "").strip()
        pointee = "void" if text == "ptr" else text[:-1].strip()
        return Pointer(int(raw) if raw is not None else 0, pointee or "void", foreign=True)
    if return_type == "str":
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return raw or ""
    if return_type == "bool":
        return bool(raw)
    return raw


# ---------------------------------------------------------------------------
# Vararg extra-argument type inference (interpreter only)
# ---------------------------------------------------------------------------

def marshal_vararg(value: Any) -> Any:
    """Best-effort type promotion for positional varargs beyond fixed params."""
    if isinstance(value, (Pointer, ArrayRef)):
        return ctypes.c_void_p(value.address)
    if isinstance(value, bool):
        return ctypes.c_int(int(value))
    if isinstance(value, int):
        return ctypes.c_longlong(value)
    if isinstance(value, float):
        return ctypes.c_double(value)
    if isinstance(value, bytes):
        return ctypes.c_char_p(value)
    if isinstance(value, str):
        return ctypes.c_char_p(value.encode("utf-8"))
    return value
