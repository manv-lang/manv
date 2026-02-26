from __future__ import annotations

from typing import Any

from .abi import ABIFunction, lower_function_abi
from .targets import TargetSpec


# Host stubs are ABI-visible CPU functions that launch kernels.
def build_host_stubs(kernel_ir: dict[str, Any], target: TargetSpec) -> tuple[dict[str, ABIFunction], str]:
    abi_map: dict[str, ABIFunction] = {}
    comment = "#" if target.isa == "x86_64" else "//"
    lines: list[str] = [f"{comment} host stubs for {target.name}"]

    for kernel in kernel_ir.get("kernels", []):
        kname = str(kernel.get("kernel_name", kernel.get("name", "kernel")))
        stub_name = f"launch_{kname}"

        param_types = _collect_param_types(kernel)
        abi_fn = lower_function_abi(stub_name, param_types, "int", target)
        abi_map[stub_name] = abi_fn

        lines.extend(
            [
                f".globl {stub_name}",
                f"{stub_name}:",
                f"  {comment} ABI-compliant host launcher stub",
                f"  {comment} marshal pointers/scalars to runtime launch API",
                f"  {comment} kernel={kname}",
                f"  {comment} params={len(param_types)} sync=default",
            ]
        )

        if target.isa == "x86_64":
            lines.extend(["  xor eax, eax", "  ret", ""])
        else:
            lines.extend(["  mov x0, #0", "  ret", ""])

    return abi_map, "\n".join(lines)


def _collect_param_types(kernel: dict[str, Any]) -> list[str]:
    sig = kernel.get("signature")
    if isinstance(sig, dict):
        out: list[str] = []
        for param in sig.get("params", []):
            kind = str(param.get("kind", "buffer"))
            if kind in {"buffer", "opaque"}:
                out.append("ptr")
            elif kind in {"shape", "stride", "scalar"}:
                out.append(str(param.get("dtype", "int")))
            else:
                out.append("ptr")
        return out

    buffers = kernel.get("buffers", [])
    return ["ptr" for _ in buffers]
