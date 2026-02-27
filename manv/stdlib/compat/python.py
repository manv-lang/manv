from __future__ import annotations

COMPAT_NOTES: dict[str, str] = {
    "dict_order": "ManV map preserves insertion order for deterministic iteration.",
    "exceptions": "ManV exception hierarchy is Python-like with runtime-specific subtypes.",
    "kernelization": "Pure-region lowering can defer effectful operations to interpreter fallback.",
}
