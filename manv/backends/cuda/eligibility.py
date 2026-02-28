"""Static CUDA eligibility analysis for HLIR-authored `@gpu` functions.

Why this module exists:
- `@gpu` marks GPU intent, not guaranteed GPU execution.
- The runtime needs a deterministic, precomputed explanation for why a function
  can or cannot cross the CUDA boundary.
- HLIR is the authority here: analysis is performed on lowered HLIR structure,
  not on backend guesses or source-pattern string matching.

What this module does not do:
- It does not rewrite the program.
- It does not infer fallback behavior; it only reports eligibility.
- It does not promise full CUDA feature coverage. Unsupported constructs are
  rejected explicitly with stable reason codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...hlir import HFunction, HInstruction


ALLOWED_PARAM_TYPES = {"i32", "f32", "bool"}
ALLOWED_BUFFER_PREFIXES = ("array[",)
ALLOWED_OPS = {
    "declare_var",
    "load_arg",
    "store_var",
    "load_var",
    "const",
    "binop",
    "unary",
    "index",
    "set_index",
    "intrinsic_call",
}
ALLOWED_TERMINATORS = {"br", "cbr", "ret"}


@dataclass(frozen=True)
class GpuEligibilityIssue:
    code: str
    message: str
    hlir_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "hlir_id": self.hlir_id,
        }


@dataclass
class GpuEligibilityReport:
    function: str
    eligible: bool
    mode: str
    policy: str
    issues: list[GpuEligibilityIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "eligible": self.eligible,
            "mode": self.mode,
            "policy": self.policy,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def analyze_hlir_gpu_function(function: HFunction) -> GpuEligibilityReport:
    """Return the deterministic CUDA eligibility report for one HLIR function."""

    gpu_meta = function.attrs.get("gpu", {}) if isinstance(function.attrs, dict) else {}
    mode = str(gpu_meta.get("mode", "kernel"))
    policy = "required" if bool(gpu_meta.get("required", False)) else "best_effort"
    issues: list[GpuEligibilityIssue] = []

    for param in function.params:
        param_type = str(param.get("type") or "")
        if _is_supported_param_type(param_type):
            continue
        issues.append(
            GpuEligibilityIssue(
                code="GPU_INELIGIBLE_UNSUPPORTED_TYPE",
                message=f"unsupported parameter type '{param_type}'",
            )
        )

    for block in function.blocks:
        if block.terminator is not None and block.terminator.op not in ALLOWED_TERMINATORS:
            issues.append(
                GpuEligibilityIssue(
                    code="GPU_INELIGIBLE_UNSUPPORTED_CONTROL_FLOW",
                    message=f"terminator '{block.terminator.op}' is not CUDA-eligible in v1",
                    hlir_id=block.terminator.term_id,
                )
            )

        for instr in block.instructions:
            issues.extend(_classify_instruction(instr))

    return GpuEligibilityReport(
        function=function.name,
        eligible=not issues,
        mode=mode,
        policy=policy,
        issues=issues,
    )


def _is_supported_param_type(param_type: str) -> bool:
    if param_type in ALLOWED_PARAM_TYPES:
        return True
    return any(param_type.startswith(prefix) for prefix in ALLOWED_BUFFER_PREFIXES)


def _classify_instruction(instr: HInstruction) -> list[GpuEligibilityIssue]:
    """Return all deterministic rejection reasons for one HLIR instruction.

    The analyzer intentionally emits the first-order structural reason instead of
    trying to be clever. That keeps diagnostics stable across refactors.
    """

    issues: list[GpuEligibilityIssue] = []
    if instr.op not in ALLOWED_OPS:
        code = {
            "call": "GPU_INELIGIBLE_DYNAMIC_DISPATCH",
            "gpu_call": "GPU_INELIGIBLE_DYNAMIC_DISPATCH",
            "import": "GPU_INELIGIBLE_IO",
            "from_import": "GPU_INELIGIBLE_IO",
            "alloc_array": "GPU_INELIGIBLE_DYNAMIC_ALLOCATION",
            "array": "GPU_INELIGIBLE_DYNAMIC_ALLOCATION",
            "map": "GPU_INELIGIBLE_UNSUPPORTED_TYPE",
            "set_attr": "GPU_INELIGIBLE_DYNAMIC_DISPATCH",
            "attr": "GPU_INELIGIBLE_DYNAMIC_DISPATCH",
            "raise": "GPU_INELIGIBLE_EXCEPTION_BOUNDARY",
            "load_exception": "GPU_INELIGIBLE_EXCEPTION_BOUNDARY",
            "set_exception": "GPU_INELIGIBLE_EXCEPTION_BOUNDARY",
        }.get(instr.op, "GPU_INELIGIBLE_UNSUPPORTED_CONTROL_FLOW")
        issues.append(
            GpuEligibilityIssue(
                code=code,
                message=f"instruction '{instr.op}' is not CUDA-eligible in v1",
                hlir_id=instr.instr_id,
            )
        )
        return issues

    if instr.op == "intrinsic_call":
        name = str(instr.attrs.get("name", ""))
        if name == "syscall_invoke":
            issues.append(
                GpuEligibilityIssue(
                    code="GPU_INELIGIBLE_SYSCALL",
                    message="syscalls cannot cross the CUDA boundary",
                    hlir_id=instr.instr_id,
                )
            )
        elif name not in {"core_len"}:
            issues.append(
                GpuEligibilityIssue(
                    code="GPU_INELIGIBLE_UNSUPPORTED_CONTROL_FLOW",
                    message=f"intrinsic '{name}' is not CUDA-eligible in v1",
                    hlir_id=instr.instr_id,
                )
            )
    return issues
