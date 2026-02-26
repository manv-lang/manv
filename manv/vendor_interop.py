from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class VendorSubstitution:
    library: str
    symbol: str
    reason: str


@dataclass
class SubstitutionRule:
    name: str
    match: Callable[[dict[str, Any]], bool]
    substitute: Callable[[dict[str, Any]], VendorSubstitution]


_RULES: list[SubstitutionRule] = []


def register_rule(rule: SubstitutionRule) -> None:
    _RULES.append(rule)


def try_substitute_kernel(kernel: dict[str, Any]) -> VendorSubstitution | None:
    for rule in _RULES:
        if rule.match(kernel):
            return rule.substitute(kernel)
    return None


def register_default_rules() -> None:
    if _RULES:
        return

    def _is_matmul(kernel: dict[str, Any]) -> bool:
        ops = kernel.get("ops", [])
        op_names = {str(op.get("op", op.get("opcode", ""))) for op in ops}
        return "matmul" in op_names or "gemm" in op_names

    def _sub(kernel: dict[str, Any]) -> VendorSubstitution:
        name = str(kernel.get("kernel_name", "kernel"))
        return VendorSubstitution(library="cublas_or_rocblas", symbol="gemm", reason=f"matched matmul pattern in {name}")

    register_rule(SubstitutionRule(name="gemm", match=_is_matmul, substitute=_sub))
