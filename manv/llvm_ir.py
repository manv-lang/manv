"""Helpers for emitting textual LLVM IR.

Why this module exists:
- The LLVM lowering path emits textual IR in v1 because it is easy to inspect,
  diff, and snapshot in tests.
- Keeping type/symbol/string helpers in one file makes the lowering module much
  easier to read; otherwise the important lowering logic gets buried under
  formatting noise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlvmValue:
    type_name: str
    ref: str
    runtime_class: str | None = None
    container_kind: str | None = None
    element_type: str | None = None
    key_type: str | None = None
    value_type: str | None = None
    literal_value: object | None = None


def sanitize_symbol(name: str) -> str:
    out: list[str] = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "anon"


def llvm_type(type_name: str | None) -> str:
    normalized = (type_name or "none").strip()
    if normalized in {"none", "void"}:
        return "void"
    if normalized in {"bool"}:
        return "i1"
    if normalized in {"i32"}:
        return "i32"
    if normalized in {"i64", "int", "usize"}:
        return "i64"
    if normalized in {"f32", "float"}:
        return "float"
    if normalized in {"f64"}:
        return "double"
    if normalized in {"str", "string"}:
        return "ptr"
    if normalized in {"array", "map", "module", "type", "dynamic"}:
        return "ptr"
    if normalized.startswith("array[") or normalized.startswith("ptr[") or normalized.startswith("ptr<"):
        return "ptr"
    if normalized.startswith("ptr"):
        return "ptr"
    # Unknown nominal types are treated as opaque pointers so user-defined
    # classes and future runtime objects lower through the native path instead
    # of being miscompiled as integers.
    return "ptr"


def llvm_zero(type_name: str) -> str:
    if type_name == "void":
        return ""
    if type_name in {"float", "double"}:
        return "0.0"
    if type_name == "ptr":
        return "null"
    return "0"


def escape_c_string(value: str) -> str:
    out: list[str] = []
    for ch in value:
        code = ord(ch)
        if ch == "\\":
            out.append("\\5C")
        elif ch == "\"":
            out.append("\\22")
        elif ch == "\n":
            out.append("\\0A")
        elif 32 <= code <= 126:
            out.append(ch)
        else:
            out.append(f"\\{code:02X}")
    out.append("\\00")
    return "".join(out)
