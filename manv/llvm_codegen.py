"""HLIR -> textual LLVM IR lowering."""

from __future__ import annotations

from dataclasses import dataclass, field

from .hlir import HBasicBlock, HFunction, HInstruction, HModule, HTerminator
from .llvm_ir import LlvmValue, escape_c_string, llvm_type, llvm_zero, sanitize_symbol
from .targets import TargetSpec


class LlvmLoweringError(RuntimeError):
    pass


@dataclass
class _FunctionState:
    function: HFunction
    target: TargetSpec
    symbol: str
    all_functions: dict[str, HFunction]
    function_symbols: dict[str, str]
    values: dict[str, LlvmValue] = field(default_factory=dict)
    vars: dict[str, str] = field(default_factory=dict)
    var_types: dict[str, str] = field(default_factory=dict)
    next_temp_id: int = 0

    def new_temp(self) -> str:
        self.next_temp_id += 1
        return f"%v{self.next_temp_id}"


def emit_llvm_module(module: HModule, target: TargetSpec, *, source_name: str) -> str:
    strings: dict[str, str] = {}
    all_functions = {fn.name: fn for fn in module.functions}
    function_symbols = {fn.name: _function_symbol(fn.name) for fn in module.functions}
    body_lines: list[str] = []

    for fn in module.functions:
        body_lines.extend(_emit_function(fn, target, all_functions, function_symbols, strings))
        body_lines.append("")

    if "main" in all_functions:
        entry_symbol = function_symbols["main"]
        body_lines.extend(
            [
                "define i32 @main() {",
                "entry:",
                f"  %ret = call i64 @{entry_symbol}()",
                "  %rc = trunc i64 %ret to i32",
                "  ret i32 %rc",
                "}",
                "",
            ]
        )

    lines = [
        f"; ModuleID = '{source_name}'",
        f'source_filename = "{source_name}"',
        f'target triple = "{_target_triple(target)}"',
        "",
        "declare void @manv_rt_print_i64(i64)",
        "declare void @manv_rt_print_f64(double)",
        "declare void @manv_rt_print_bool(i1)",
        "declare void @manv_rt_print_cstr(ptr)",
        "",
    ]
    for text, symbol in sorted(strings.items(), key=lambda item: item[1]):
        escaped = escape_c_string(text)
        length = len(text.encode("utf-8")) + 1
        lines.append(f'@{symbol} = private unnamed_addr constant [{length} x i8] c"{escaped}", align 1')
    if strings:
        lines.append("")

    lines.extend(body_lines)
    return "\n".join(lines).rstrip() + "\n"


def _emit_function(
    function: HFunction,
    target: TargetSpec,
    all_functions: dict[str, HFunction],
    function_symbols: dict[str, str],
    strings: dict[str, str],
) -> list[str]:
    symbol = function_symbols[function.name]
    state = _FunctionState(
        function=function,
        target=target,
        symbol=symbol,
        all_functions=all_functions,
        function_symbols=function_symbols,
    )
    return_type = llvm_type(function.return_type)
    params_text: list[str] = []
    for param in function.params:
        param_name = str(param.get("name"))
        param_type = llvm_type(str(param.get("type")))
        params_text.append(f"{param_type} %{sanitize_symbol(param_name)}")
        state.values[f"@arg:{param_name}"] = LlvmValue(param_type, f"%{sanitize_symbol(param_name)}")

    lines = [f"define {return_type} @{symbol}({', '.join(params_text)}) {{"]
    entry_allocas = _entry_allocas(function, state)

    for block in function.blocks:
        lines.append(f"{_block_label(block)}:")
        if block.label == function.entry:
            lines.extend(entry_allocas)
        for instr in block.instructions:
            lines.extend(_emit_instruction(instr, state, strings))
        lines.extend(_emit_terminator(block.terminator, state))

    lines.append("}")
    return lines


def _entry_allocas(function: HFunction, state: _FunctionState) -> list[str]:
    lines: list[str] = []
    for block in function.blocks:
        for instr in block.instructions:
            if instr.op != "declare_var":
                continue
            var_name = str(instr.attrs.get("name", "_tmp"))
            if var_name in state.vars:
                continue
            var_type = llvm_type(str(instr.attrs.get("type") or "int"))
            slot = f"%{sanitize_symbol(var_name)}.slot"
            state.vars[var_name] = slot
            state.var_types[var_name] = var_type
            lines.append(f"  {slot} = alloca {var_type}, align 8")
    return lines


def _emit_instruction(instr: HInstruction, state: _FunctionState, strings: dict[str, str]) -> list[str]:
    op = instr.op
    out: list[str] = []

    if op == "declare_var":
        return out
    if op == "const":
        if instr.dest is None:
            return out
        ty = llvm_type(instr.type_name)
        value = instr.attrs.get("value")
        if isinstance(value, str):
            symbol = strings.setdefault(value, f".str.{len(strings)}")
            length = len(value.encode("utf-8")) + 1
            temp = state.new_temp()
            out.append(f"  {temp} = getelementptr inbounds [{length} x i8], ptr @{symbol}, i64 0, i64 0")
            state.values[instr.dest] = LlvmValue("ptr", temp)
        elif value is None:
            state.values[instr.dest] = LlvmValue(ty, llvm_zero(ty))
        elif isinstance(value, bool):
            state.values[instr.dest] = LlvmValue("i1", "1" if value else "0")
        elif ty in {"float", "double"}:
            state.values[instr.dest] = LlvmValue(ty, str(value))
        else:
            state.values[instr.dest] = LlvmValue(ty, str(int(value)))
        return out
    if op == "load_arg":
        if instr.dest is not None:
            arg_name = str(instr.attrs.get("name"))
            state.values[instr.dest] = state.values[f"@arg:{arg_name}"]
        return out
    if op == "store_var":
        var_name = str(instr.args[0])
        value = _coerce_value(_operand(str(instr.args[1]), state), state.var_types[var_name], state, out)
        out.append(f"  store {value.type_name} {value.ref}, ptr {state.vars[var_name]}, align 8")
        return out
    if op == "load_var":
        var_name = str(instr.args[0])
        ty = state.var_types[var_name]
        temp = state.new_temp()
        out.append(f"  {temp} = load {ty}, ptr {state.vars[var_name]}, align 8")
        state.values[instr.dest or temp] = LlvmValue(ty, temp)
        return out
    if op == "unary":
        operand = _operand(str(instr.args[0]), state)
        operator = str(instr.attrs.get("op"))
        temp = state.new_temp()
        if operator == "-":
            if operand.type_name in {"float", "double"}:
                out.append(f"  {temp} = fneg {operand.type_name} {operand.ref}")
            else:
                out.append(f"  {temp} = sub {operand.type_name} 0, {operand.ref}")
            state.values[instr.dest or temp] = LlvmValue(operand.type_name, temp)
            return out
        if operator == "!":
            bool_value = _coerce_value(operand, "i1", state, out)
            out.append(f"  {temp} = xor i1 {bool_value.ref}, true")
            state.values[instr.dest or temp] = LlvmValue("i1", temp)
            return out
        raise LlvmLoweringError(f"unsupported unary operator: {operator}")
    if op == "binop":
        lowered, lines = _emit_binop(
            _operand(str(instr.args[0]), state),
            _operand(str(instr.args[1]), state),
            str(instr.attrs.get("op")),
            state,
        )
        out.extend(lines)
        state.values[instr.dest or lowered.ref] = lowered
        return out
    if op == "call":
        return _emit_call(instr, state, out)
    if op == "intrinsic_call":
        return _emit_intrinsic_call(instr, state, out)
    if op == "gpu_call":
        raise LlvmLoweringError("native @gpu lowering is not implemented yet")
    raise LlvmLoweringError(f"unsupported HLIR instruction for LLVM lowering: {op}")


def _emit_call(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    callee = str(instr.attrs.get("callee", ""))
    if callee == "print":
        value = _operand(str(instr.args[0]), state)
        out.extend(_emit_print_call(value, state))
        if instr.dest is not None:
            state.values[instr.dest] = LlvmValue("i64", "0")
        return out

    callee_fn = state.all_functions.get(callee)
    if callee_fn is None:
        raise LlvmLoweringError(f"unknown callee '{callee}'")

    arg_values = []
    for param, arg_name in zip(callee_fn.params, instr.args, strict=True):
        arg_values.append(_coerce_value(_operand(str(arg_name), state), llvm_type(str(param.get("type"))), state, out))

    return_type = llvm_type(callee_fn.return_type)
    arg_text = ", ".join(f"{value.type_name} {value.ref}" for value in arg_values)
    if return_type == "void":
        out.append(f"  call void @{state.function_symbols[callee]}({arg_text})")
        if instr.dest is not None:
            state.values[instr.dest] = LlvmValue("i64", "0")
        return out

    temp = state.new_temp()
    out.append(f"  {temp} = call {return_type} @{state.function_symbols[callee]}({arg_text})")
    if instr.dest is not None:
        state.values[instr.dest] = LlvmValue(return_type, temp)
    return out


def _emit_intrinsic_call(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    name = str(instr.attrs.get("name", ""))
    if name == "io_print":
        if len(instr.args) != 1:
            raise LlvmLoweringError("io_print expects one argument")
        out.extend(_emit_print_call(_operand(str(instr.args[0]), state), state))
        if instr.dest is not None:
            state.values[instr.dest] = LlvmValue("i64", "0")
        return out
    raise LlvmLoweringError(f"unsupported intrinsic call for LLVM lowering: {name}")


def _emit_print_call(value: LlvmValue, state: _FunctionState) -> list[str]:
    out: list[str] = []
    if value.type_name == "ptr":
        out.append(f"  call void @manv_rt_print_cstr(ptr {value.ref})")
    elif value.type_name == "i1":
        out.append(f"  call void @manv_rt_print_bool(i1 {value.ref})")
    elif value.type_name in {"float", "double"}:
        coerced = _coerce_value(value, "double", state, out)
        out.append(f"  call void @manv_rt_print_f64(double {coerced.ref})")
    else:
        coerced = _coerce_value(value, "i64", state, out)
        out.append(f"  call void @manv_rt_print_i64(i64 {coerced.ref})")
    return out


def _emit_terminator(term: HTerminator | None, state: _FunctionState) -> list[str]:
    if term is None:
        return []
    if term.op == "br":
        return [f"  br label %{sanitize_symbol(str(term.args[0]))}"]
    if term.op == "cbr":
        out: list[str] = []
        cond = _coerce_value(_operand(str(term.args[0]), state), "i1", state, out)
        out.append(
            f"  br i1 {cond.ref}, label %{sanitize_symbol(str(term.args[1]))}, label %{sanitize_symbol(str(term.args[2]))}"
        )
        return out
    if term.op == "ret":
        return_type = llvm_type(state.function.return_type)
        if return_type == "void":
            return ["  ret void"]
        if not term.args:
            return [f"  ret {return_type} {llvm_zero(return_type)}"]
        out: list[str] = []
        value = _coerce_value(_operand(str(term.args[0]), state), return_type, state, out)
        out.append(f"  ret {return_type} {value.ref}")
        return out
    raise LlvmLoweringError(f"unsupported HLIR terminator for LLVM lowering: {term.op}")


def _emit_binop(lhs: LlvmValue, rhs: LlvmValue, operator: str, state: _FunctionState) -> tuple[LlvmValue, list[str]]:
    out: list[str] = []
    target_type = _unify_numeric_type(lhs.type_name, rhs.type_name)
    left = _coerce_value(lhs, target_type, state, out)
    right = _coerce_value(rhs, target_type, state, out)
    temp = state.new_temp()

    if target_type in {"float", "double"}:
        arithmetic = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv"}
        comparisons = {"==": "oeq", "!=": "one", "<": "olt", "<=": "ole", ">": "ogt", ">=": "oge"}
        if operator in arithmetic:
            out.append(f"  {temp} = {arithmetic[operator]} {target_type} {left.ref}, {right.ref}")
            return LlvmValue(target_type, temp), out
        if operator in comparisons:
            out.append(f"  {temp} = fcmp {comparisons[operator]} {target_type} {left.ref}, {right.ref}")
            return LlvmValue("i1", temp), out
        raise LlvmLoweringError(f"unsupported floating-point binary operator: {operator}")

    if operator in {"+", "-", "*"}:
        opcode = {"+": "add", "-": "sub", "*": "mul"}[operator]
        out.append(f"  {temp} = {opcode} {target_type} {left.ref}, {right.ref}")
        return LlvmValue(target_type, temp), out
    if operator == "/":
        out.append(f"  {temp} = sdiv {target_type} {left.ref}, {right.ref}")
        return LlvmValue(target_type, temp), out
    if operator in {"==", "!=", "<", "<=", ">", ">="}:
        pred = {"==": "eq", "!=": "ne", "<": "slt", "<=": "sle", ">": "sgt", ">=": "sge"}[operator]
        out.append(f"  {temp} = icmp {pred} {target_type} {left.ref}, {right.ref}")
        return LlvmValue("i1", temp), out
    if operator in {"&&", "||"}:
        left_bool = _coerce_value(left, "i1", state, out)
        right_bool = _coerce_value(right, "i1", state, out)
        opcode = "and" if operator == "&&" else "or"
        out.append(f"  {temp} = {opcode} i1 {left_bool.ref}, {right_bool.ref}")
        return LlvmValue("i1", temp), out
    raise LlvmLoweringError(f"unsupported integer binary operator: {operator}")


def _operand(name: str, state: _FunctionState) -> LlvmValue:
    value = state.values.get(name)
    if value is None:
        raise LlvmLoweringError(f"unknown operand '{name}' in function '{state.function.name}'")
    return value


def _coerce_value(value: LlvmValue, target_type: str, state: _FunctionState, out: list[str]) -> LlvmValue:
    if value.type_name == target_type:
        return value
    temp = state.new_temp()
    if value.type_name == "i1" and target_type in {"i32", "i64"}:
        out.append(f"  {temp} = zext i1 {value.ref} to {target_type}")
        return LlvmValue(target_type, temp)
    if value.type_name in {"i32", "i64"} and target_type == "i1":
        out.append(f"  {temp} = icmp ne {value.type_name} {value.ref}, 0")
        return LlvmValue("i1", temp)
    if value.type_name == "i32" and target_type == "i64":
        out.append(f"  {temp} = sext i32 {value.ref} to i64")
        return LlvmValue("i64", temp)
    if value.type_name == "i64" and target_type == "i32":
        out.append(f"  {temp} = trunc i64 {value.ref} to i32")
        return LlvmValue("i32", temp)
    if value.type_name == "float" and target_type == "double":
        out.append(f"  {temp} = fpext float {value.ref} to double")
        return LlvmValue("double", temp)
    if value.type_name == "double" and target_type == "float":
        out.append(f"  {temp} = fptrunc double {value.ref} to float")
        return LlvmValue("float", temp)
    if value.type_name in {"i32", "i64"} and target_type in {"float", "double"}:
        out.append(f"  {temp} = sitofp {value.type_name} {value.ref} to {target_type}")
        return LlvmValue(target_type, temp)
    if value.type_name in {"float", "double"} and target_type in {"i32", "i64"}:
        out.append(f"  {temp} = fptosi {value.type_name} {value.ref} to {target_type}")
        return LlvmValue(target_type, temp)
    raise LlvmLoweringError(f"cannot coerce {value.type_name} to {target_type}")


def _unify_numeric_type(lhs: str, rhs: str) -> str:
    if "double" in {lhs, rhs}:
        return "double"
    if "float" in {lhs, rhs}:
        return "float"
    if "i64" in {lhs, rhs}:
        return "i64"
    if "i32" in {lhs, rhs}:
        return "i32"
    if "i1" in {lhs, rhs}:
        return "i1"
    return lhs


def _function_symbol(name: str) -> str:
    return f"manv_{sanitize_symbol(name)}"


def _block_label(block: HBasicBlock) -> str:
    return sanitize_symbol(block.label)


def _target_triple(target: TargetSpec) -> str:
    if target.name == "x86_64-sysv":
        return "x86_64-unknown-linux-gnu"
    if target.name == "x86_64-win64":
        return "x86_64-pc-windows-msvc"
    if target.name == "aarch64-aapcs64":
        return "aarch64-unknown-linux-gnu"
    return "unknown-unknown-unknown"
