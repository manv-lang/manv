from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .kernel_ir import ALLOWED_MEMORY_SPACES, KIRModule, parse_kir_module


@dataclass
class KIRVerifyIssue:
    code: str
    message: str
    kernel: str | None = None
    op_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "kernel": self.kernel,
            "op_id": self.op_id,
        }


class KIRVerifyError(RuntimeError):
    def __init__(self, issues: list[KIRVerifyIssue]):
        self.issues = issues
        text = "\n".join(f"{x.code}: {x.message}" for x in issues)
        super().__init__(text)


def verify_kir_module(module_or_payload: KIRModule | dict[str, Any], *, strict: bool = True) -> list[KIRVerifyIssue]:
    module = module_or_payload if isinstance(module_or_payload, KIRModule) else parse_kir_module(module_or_payload)
    issues: list[KIRVerifyIssue] = []

    if not module.version:
        issues.append(KIRVerifyIssue(code="KIR001", message="module version is required"))

    seen_kernel_names: set[str] = set()
    for kernel in module.kernels:
        if kernel.name in seen_kernel_names:
            issues.append(KIRVerifyIssue(code="KIR002", message=f"duplicate kernel name '{kernel.name}'", kernel=kernel.name))
        seen_kernel_names.add(kernel.name)

        if not kernel.blocks:
            issues.append(KIRVerifyIssue(code="KIR003", message="kernel must contain at least one block", kernel=kernel.name))
            continue

        seen_ops: set[str] = set()
        for block in kernel.blocks:
            for op in block.ops:
                if op.id in seen_ops:
                    issues.append(KIRVerifyIssue(code="KIR004", message="duplicate op id", kernel=kernel.name, op_id=op.id))
                seen_ops.add(op.id)

                if not op.dtype:
                    issues.append(KIRVerifyIssue(code="KIR005", message="op dtype must be set", kernel=kernel.name, op_id=op.id))
                if strict and op.dtype in {"dynamic", ""}:
                    issues.append(KIRVerifyIssue(code="KIR006", message="op dtype must be fully resolved", kernel=kernel.name, op_id=op.id))

                if op.memory_space not in ALLOWED_MEMORY_SPACES:
                    issues.append(
                        KIRVerifyIssue(
                            code="KIR007",
                            message=f"invalid memory space '{op.memory_space}'",
                            kernel=kernel.name,
                            op_id=op.id,
                        )
                    )

                if op.opcode in {"alloc", "malloc", "new", "dynamic_alloc"}:
                    issues.append(
                        KIRVerifyIssue(
                            code="KIR008",
                            message="dynamic allocation is not allowed in KIR",
                            kernel=kernel.name,
                            op_id=op.id,
                        )
                    )

                if op.provenance is None or op.provenance.source_span is None:
                    issues.append(
                        KIRVerifyIssue(
                            code="KIR009",
                            message="op provenance must include source span",
                            kernel=kernel.name,
                            op_id=op.id,
                        )
                    )

                if op.opcode == "barrier" and op.memory_space not in {"shared", "local", "private"}:
                    issues.append(
                        KIRVerifyIssue(
                            code="KIR010",
                            message="barrier op uses incompatible memory scope",
                            kernel=kernel.name,
                            op_id=op.id,
                        )
                    )

    return issues


def assert_valid_kir_module(module_or_payload: KIRModule | dict[str, Any], *, strict: bool = True) -> None:
    issues = verify_kir_module(module_or_payload, strict=strict)
    if issues:
        raise KIRVerifyError(issues)
