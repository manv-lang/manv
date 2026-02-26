from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .abi import ABIFunction
from .hlir import HFunction, HInstruction, HModule
from .targets import TargetSpec


@dataclass
class RegisterAssignment:
    gp: dict[str, str]
    fp: dict[str, str]
    spills: dict[str, int]


@dataclass
class FunctionCodegenContext:
    var_slots: dict[str, int]


# Emit backend assembly in a form accepted by GNU as.
def emit_target_assembly(module: HModule, target: TargetSpec, abi_functions: dict[str, ABIFunction]) -> str:
    lines: list[str] = [
        _comment(f"manv backend assembly target={target.name}", target),
        ".text",
        "",
    ]

    if target.isa == "x86_64":
        lines.append(".intel_syntax noprefix")
        lines.append("")

    runtime_helpers: set[str] = set()

    for fn in module.functions:
        abi_fn = abi_functions[fn.name]
        reg_map = _allocate_registers(fn, target)
        ctx = _build_function_context(fn)
        lines.extend(_emit_function(fn, target, abi_fn, reg_map, ctx, runtime_helpers))
        lines.append("")

    if "print_i64" in runtime_helpers:
        lines.extend(_emit_print_i64_stub(target))

    return "\n".join(lines)


def _emit_function(
    fn: HFunction,
    target: TargetSpec,
    abi_fn: ABIFunction,
    reg_map: RegisterAssignment,
    ctx: FunctionCodegenContext,
    runtime_helpers: set[str],
) -> list[str]:
    out: list[str] = []

    out.append(f".globl {fn.name}")
    if target.isa == "x86_64" and target.abi != "win64":
        out.append(f".type {fn.name}, @function")
    out.append(f"{fn.name}:")

    stack_alloc = abi_fn.frame.stack_size + abi_fn.frame.shadow_space

    if target.unwind_style == "dwarf_cfi":
        out.append("  .cfi_startproc")

    if target.abi == "win64":
        out.append(f"  .seh_proc {fn.name}")

    if target.isa == "x86_64":
        out.append("  push rbp")
        if target.unwind_style == "dwarf_cfi":
            out.extend(["  .cfi_def_cfa_offset 16", "  .cfi_offset rbp, -16"])
        if target.abi == "win64":
            out.append("  .seh_pushreg rbp")

        out.append("  mov rbp, rsp")
        if target.unwind_style == "dwarf_cfi":
            out.append("  .cfi_def_cfa_register rbp")
        if target.abi == "win64":
            out.append("  .seh_setframe rbp, 0")

        if stack_alloc:
            out.append(f"  sub rsp, {stack_alloc}")
            if target.unwind_style == "dwarf_cfi":
                out.append(f"  .cfi_adjust_cfa_offset {stack_alloc}")
            if target.abi == "win64":
                out.append(f"  .seh_stackalloc {stack_alloc}")

        if target.abi == "win64":
            out.append("  .seh_endprologue")

    else:
        out.extend(["  stp x29, x30, [sp, #-16]!", "  mov x29, sp"])
        if target.unwind_style == "dwarf_cfi":
            out.extend([
                "  .cfi_def_cfa_offset 16",
                "  .cfi_offset x29, -16",
                "  .cfi_offset x30, -8",
                "  .cfi_def_cfa_register x29",
            ])
        if stack_alloc:
            out.append(f"  sub sp, sp, #{stack_alloc}")

    if abi_fn.frame.shadow_space:
        out.append(f"  {_comment('win64 shadow space reserved', target)}")
    if abi_fn.is_varargs:
        out.append(
            f"  {_comment(f'varargs gp={abi_fn.varargs_gp_reg_count} fp={abi_fn.varargs_fp_reg_count}', target)}"
        )

    for block in fn.blocks:
        out.append(f"{fn.name}_{block.label}:")
        for instr in block.instructions:
            out.extend(_emit_instruction(instr, reg_map, target, abi_fn, ctx, runtime_helpers))

        term = block.terminator
        if term is None:
            continue

        if term.op == "br":
            if target.isa == "x86_64":
                out.append(f"  jmp {fn.name}_{term.args[0]}")
            else:
                out.append(f"  b {fn.name}_{term.args[0]}")
        elif term.op == "cbr":
            cond = _operand(term.args[0], reg_map, target)
            if target.isa == "x86_64":
                out.append(f"  cmp {cond}, 0")
                out.append(f"  jne {fn.name}_{term.args[1]}")
                out.append(f"  jmp {fn.name}_{term.args[2]}")
            else:
                out.append(f"  cmp {cond}, #0")
                out.append(f"  b.ne {fn.name}_{term.args[1]}")
                out.append(f"  b {fn.name}_{term.args[2]}")
        elif term.op == "ret":
            if term.args and abi_fn.return_location is not None:
                ret = _operand(term.args[0], reg_map, target)
                out.extend(_emit_return_moves(ret, abi_fn, target))
            out.extend(_emit_epilogue(target, stack_alloc))
        elif term.op == "unreachable":
            out.append("  ud2" if target.isa == "x86_64" else "  brk #0")

    if target.abi == "win64":
        out.append("  .seh_endproc")
    if target.unwind_style == "dwarf_cfi":
        out.append("  .cfi_endproc")

    return out


def _emit_return_moves(ret: str, abi_fn: ABIFunction, target: TargetSpec) -> list[str]:
    out: list[str] = []
    if abi_fn.return_location is None:
        return out

    if abi_fn.return_location.reg:
        out.append(_mov(abi_fn.return_location.reg, ret, target))
        return out

    gp_idx = 0
    fp_idx = 0
    for reg, klass in zip(abi_fn.return_location.regs, abi_fn.return_location.classes or ()):
        src = ret if (gp_idx == 0 and fp_idx == 0) else "0"
        out.append(_mov(reg, src, target))
        if klass == "INTEGER":
            gp_idx += 1
        else:
            fp_idx += 1

    # If classes were omitted, still materialize the first register.
    if not abi_fn.return_location.classes and abi_fn.return_location.regs:
        out.append(_mov(abi_fn.return_location.regs[0], ret, target))

    return out


def _emit_epilogue(target: TargetSpec, stack_alloc: int) -> list[str]:
    if target.isa == "x86_64":
        out = []
        if stack_alloc:
            out.append("  mov rsp, rbp")
        out.extend(["  pop rbp", "  ret"])
        return out

    lines: list[str] = []
    if stack_alloc:
        lines.append(f"  add sp, sp, #{stack_alloc}")
    lines.extend(["  ldp x29, x30, [sp], #16", "  ret"])
    return lines


def _emit_instruction(
    instr: HInstruction,
    reg_map: RegisterAssignment,
    target: TargetSpec,
    abi_fn: ABIFunction,
    ctx: FunctionCodegenContext,
    runtime_helpers: set[str],
) -> list[str]:
    op = instr.op
    lines: list[str] = []

    if op == "const":
        if instr.dest:
            imm, note = _imm(instr.attrs.get("value"))
            if note:
                lines.append(f"  {_comment(note, target)}")
            lines.append(_mov(_operand(instr.dest, reg_map, target), imm, target))
        return lines

    if op == "load_arg":
        idx = int(instr.attrs.get("index", 0))
        src = _arg_source(idx, abi_fn, target)
        lines.append(_mov(_operand(instr.dest, reg_map, target), src, target))
        return lines

    if op == "declare_var":
        _slot_for_var(ctx, str(instr.attrs.get("name", "_tmp")))
        return lines

    if op == "store_var":
        name = str(instr.args[0])
        src = _operand(instr.args[1], reg_map, target)
        lines.append(_mov(_var_addr(name, ctx, target), src, target))
        return lines

    if op == "load_var":
        name = str(instr.args[0])
        dst = _operand(instr.dest, reg_map, target)
        lines.append(_mov(dst, _var_addr(name, ctx, target), target))
        return lines

    if op in {"alloc_array", "array_init_sized", "array", "map", "index", "set_index", "attr"}:
        lines.append(f"  {_comment(op, target)}")
        if instr.dest:
            lines.append(_mov(_operand(instr.dest, reg_map, target), "0", target))
        return lines

    if op == "unary":
        dst = _operand(instr.dest, reg_map, target)
        src = _operand(instr.args[0], reg_map, target)
        unary_op = str(instr.attrs.get("op", ""))
        lines.append(_mov(dst, src, target))
        if unary_op == "-":
            lines.append("  neg " + dst if target.isa == "x86_64" else f"  neg {dst}, {dst}")
        elif unary_op == "!":
            if target.isa == "x86_64":
                lines.extend([f"  cmp {dst}, 0", "  sete al", f"  movzx {dst}, al"])
            else:
                lines.extend([f"  cmp {dst}, #0", f"  cset {dst}, eq"])
        return lines

    if op == "binop":
        dst = _operand(instr.dest, reg_map, target)
        lhs = _operand(instr.args[0], reg_map, target)
        rhs = _operand(instr.args[1], reg_map, target)
        bop = str(instr.attrs.get("op"))
        lines.append(_mov(dst, lhs, target))
        lines.extend(_emit_binop(dst, rhs, bop, target))
        return lines

    if op == "call":
        callee = str(instr.attrs.get("callee", "<unknown>"))
        lowered = _map_runtime_callee(callee, runtime_helpers)

        # Minimal ABI call placement: pack positional args into the integer argument bank.
        for idx, arg in enumerate(instr.args):
            if idx >= len(target.gp_arg_regs):
                lines.append(f"  {_comment('extra call args on stack not yet lowered', target)}")
                break
            dst = target.gp_arg_regs[idx]
            src = _operand(arg, reg_map, target)
            lines.append(_mov(dst, src, target))

        if target.isa == "x86_64":
            if abi_fn.is_varargs and target.abi == "sysv":
                lines.append("  mov al, 0")
            lines.append(f"  call {lowered}")
        else:
            lines.append(f"  bl {lowered}")

        if instr.dest:
            lines.append(_mov(_operand(instr.dest, reg_map, target), target.gp_ret_regs[0], target))
        return lines

    lines.append(f"  {_comment(f'unsupported-instr {op}', target)}")
    if instr.dest:
        lines.append(_mov(_operand(instr.dest, reg_map, target), "0", target))
    return lines


def _emit_binop(dst: str, rhs: str, op: str, target: TargetSpec) -> list[str]:
    if target.isa != "x86_64":
        # Non-x86 backends are shape/ABI-checked in v0.1.0; codegen remains conservative.
        return [_comment(f"aarch64 binop {op} lowered as no-op", target)]

    if op == "+":
        return [f"  add {dst}, {rhs}"]
    if op == "-":
        return [f"  sub {dst}, {rhs}"]
    if op == "*":
        return [f"  imul {dst}, {rhs}"]
    if op == "/":
        return [f"  {_comment('division not lowered in v0.1.0', target)}"]

    cmp_map = {
        "==": "sete",
        "!=": "setne",
        "<": "setl",
        "<=": "setle",
        ">": "setg",
        ">=": "setge",
    }
    if op in cmp_map:
        return [f"  cmp {dst}, {rhs}", f"  {cmp_map[op]} al", f"  movzx {dst}, al"]

    if op == "&&":
        return [f"  and {dst}, {rhs}", f"  cmp {dst}, 0", "  setne al", f"  movzx {dst}, al"]
    if op == "||":
        return [f"  or {dst}, {rhs}", f"  cmp {dst}, 0", "  setne al", f"  movzx {dst}, al"]

    return [f"  {_comment(f'unsupported binop {op}', target)}"]


def _allocate_registers(fn: HFunction, target: TargetSpec) -> RegisterAssignment:
    if target.isa == "x86_64":
        gp_pool = ["r10", "r11", "r12", "r13", "r14", "r15", "rbx"]
        fp_pool = ["xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"]
    else:
        gp_pool = ["x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17"]
        fp_pool = ["v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"]

    gp: dict[str, str] = {}
    fp: dict[str, str] = {}
    spills: dict[str, int] = {}
    spill_slot = 0

    for block in fn.blocks:
        for instr in block.instructions:
            if not instr.dest:
                continue
            t = (instr.type_name or "int").strip()
            if t in {"float", "f32", "f64"}:
                if fp_pool:
                    fp[instr.dest] = fp_pool.pop(0)
                else:
                    spills[instr.dest] = spill_slot
                    spill_slot += 8
                continue
            if gp_pool:
                gp[instr.dest] = gp_pool.pop(0)
            else:
                spills[instr.dest] = spill_slot
                spill_slot += 8

    return RegisterAssignment(gp=gp, fp=fp, spills=spills)


def _build_function_context(fn: HFunction) -> FunctionCodegenContext:
    slots: dict[str, int] = {}
    next_slot = 8
    for block in fn.blocks:
        for instr in block.instructions:
            if instr.op == "declare_var":
                name = str(instr.attrs.get("name", "_tmp"))
                if name not in slots:
                    slots[name] = next_slot
                    next_slot += 8
    return FunctionCodegenContext(var_slots=slots)


def _slot_for_var(ctx: FunctionCodegenContext, name: str) -> int:
    if name not in ctx.var_slots:
        ctx.var_slots[name] = (max(ctx.var_slots.values()) + 8) if ctx.var_slots else 8
    return ctx.var_slots[name]


def _var_addr(name: str, ctx: FunctionCodegenContext, target: TargetSpec) -> str:
    offset = _slot_for_var(ctx, name)
    if target.isa == "x86_64":
        return f"QWORD PTR [rbp-{offset}]"
    return f"[x29, #-{offset}]"


def _arg_source(index: int, abi_fn: ABIFunction, target: TargetSpec) -> str:
    arg_index = index + (1 if abi_fn.sret else 0)
    if arg_index >= len(abi_fn.arg_locations):
        return "0"

    loc = abi_fn.arg_locations[arg_index]
    if loc.kind == "reg" and loc.reg:
        return loc.reg
    if loc.kind == "regs" and loc.regs:
        return loc.regs[0]

    stack_off = int(loc.stack_offset or 0)
    base = 16 + stack_off
    if target.isa == "x86_64":
        return f"QWORD PTR [rbp+{base}]"
    return f"[x29, #{base}]"


def _operand(temp: str | None, reg_map: RegisterAssignment, target: TargetSpec) -> str:
    if temp is None:
        return "0"
    if not temp.startswith("%"):
        return temp
    if temp in reg_map.gp:
        return reg_map.gp[temp]
    if temp in reg_map.fp:
        return reg_map.fp[temp]
    if temp in reg_map.spills:
        offset = reg_map.spills[temp] + 8
        if target.isa == "x86_64":
            return f"QWORD PTR [rbp-{offset}]"
        return f"[x29, #-{offset}]"
    return "rax" if target.isa == "x86_64" else "x0"


def _mov(dst: str, src: str, target: TargetSpec) -> str:
    return f"  mov {dst}, {src}"



def _comment(text: str, target: TargetSpec) -> str:
    return f"# {text}" if target.isa == "x86_64" else f"// {text}"


def _map_runtime_callee(name: str, runtime_helpers: set[str]) -> str:
    if name == "print":
        runtime_helpers.add("print_i64")
        return "manv_rt_print_i64"
    return name


def _emit_print_i64_stub(target: TargetSpec) -> list[str]:
    lines = ["", _comment("runtime helper stubs", target), ".globl manv_rt_print_i64", "manv_rt_print_i64:"]
    if target.isa == "x86_64":
        lines.extend(["  xor eax, eax", "  ret"])
    else:
        lines.extend(["  mov x0, #0", "  ret"])
    return lines


def _imm(value: Any) -> tuple[str, str | None]:
    if value is None:
        return "0", None
    if isinstance(value, bool):
        return ("1" if value else "0"), None
    if isinstance(value, int):
        return str(value), None
    if isinstance(value, float):
        return "0", f"float constant {value} lowered as integer lane 0 in v0.1.0"
    if isinstance(value, str):
        return "0", f"string literal lowered via runtime path in v0.1.0: {value!r}"
    return "0", f"unsupported constant {value!r} lowered as 0"
