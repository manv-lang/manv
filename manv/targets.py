from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetSpec:
    """Target+ABI description used by ABI lowering and final code emission."""

    name: str
    isa: str
    abi: str
    pointer_size: int
    endian: str
    gp_arg_regs: tuple[str, ...]
    fp_arg_regs: tuple[str, ...]
    gp_ret_regs: tuple[str, ...]
    fp_ret_regs: tuple[str, ...]
    caller_saved_gp: tuple[str, ...]
    callee_saved_gp: tuple[str, ...]
    caller_saved_fp: tuple[str, ...] = ()
    callee_saved_fp: tuple[str, ...] = ()
    stack_align: int = 16
    red_zone: int = 0
    shadow_space: int = 0
    red_zone_enabled: bool = False
    supports_varargs: bool = False
    aggregate_policy: str = "byref_sret"
    unwind_style: str = "none"


TARGETS: dict[str, TargetSpec] = {
    "x86_64-sysv": TargetSpec(
        name="x86_64-sysv",
        isa="x86_64",
        abi="sysv",
        pointer_size=8,
        endian="little",
        gp_arg_regs=("rdi", "rsi", "rdx", "rcx", "r8", "r9"),
        fp_arg_regs=("xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"),
        gp_ret_regs=("rax", "rdx"),
        fp_ret_regs=("xmm0", "xmm1"),
        caller_saved_gp=("rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"),
        callee_saved_gp=("rbx", "rbp", "r12", "r13", "r14", "r15"),
        stack_align=16,
        red_zone=128,
        red_zone_enabled=False,
        shadow_space=0,
        supports_varargs=True,
        aggregate_policy="sysv_eightbyte",
        unwind_style="dwarf_cfi",
    ),
    "x86_64-win64": TargetSpec(
        name="x86_64-win64",
        isa="x86_64",
        abi="win64",
        pointer_size=8,
        endian="little",
        gp_arg_regs=("rcx", "rdx", "r8", "r9"),
        fp_arg_regs=("xmm0", "xmm1", "xmm2", "xmm3"),
        gp_ret_regs=("rax",),
        fp_ret_regs=("xmm0",),
        caller_saved_gp=("rax", "rcx", "rdx", "r8", "r9", "r10", "r11"),
        callee_saved_gp=("rbx", "rbp", "rdi", "rsi", "r12", "r13", "r14", "r15"),
        caller_saved_fp=("xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5"),
        callee_saved_fp=("xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"),
        stack_align=16,
        shadow_space=32,
        supports_varargs=True,
        aggregate_policy="byref_sret",
        unwind_style="win_seh",
    ),
    "aarch64-aapcs64": TargetSpec(
        name="aarch64-aapcs64",
        isa="aarch64",
        abi="aapcs64",
        pointer_size=8,
        endian="little",
        gp_arg_regs=("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"),
        fp_arg_regs=("v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"),
        gp_ret_regs=("x0", "x1"),
        fp_ret_regs=("v0", "v1"),
        caller_saved_gp=(
            "x0",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
            "x17",
        ),
        callee_saved_gp=("x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30"),
        stack_align=16,
        supports_varargs=True,
        aggregate_policy="byref_sret",
        unwind_style="dwarf_cfi",
    ),
}


def get_target(name: str) -> TargetSpec:
    if name not in TARGETS:
        supported = ", ".join(sorted(TARGETS))
        raise ValueError(f"unsupported target '{name}'. supported: {supported}")
    return TARGETS[name]
