"""HLIR -> textual LLVM IR lowering.

Why this file exists:
- The LLVM host backend is now the only native code path. It must therefore
  model the real language surface instead of delegating to the removed
  assembly-native fallback.
- The lowering is intentionally explicit and heavily commented because object
  layout, constructor calls, and the current GPU/device boundary are subtle
  semantic seams. Future backend work needs those invariants to stay readable.

Important invariants:
- HLIR remains authoritative. LLVM lowering consumes explicit HLIR operations
  such as `gpu_call`, `method_call`, `attr`, and `set_attr`; it does not
  reconstruct language meaning from source syntax.
- Unsupported LLVM features are reported deterministically before emission so
  users get a stable migration target instead of whichever instruction happens
  to lower first.
- The native path only models the subset we can implement coherently today.
  For example, syscall lowering is intentionally narrow and only covers the
  host-visible result keys that the current source/runtime surface relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hlir import HBasicBlock, HFunction, HInstruction, HModule, HTerminator
from .llvm_ir import LlvmValue, escape_c_string, llvm_type, llvm_zero, sanitize_symbol
from .targets import TargetSpec


class LlvmLoweringError(RuntimeError):
    pass


@dataclass
class _ValueInfo:
    type_name: str
    runtime_class: str | None = None
    container_kind: str | None = None
    element_type: str | None = None
    key_type: str | None = None
    value_type: str | None = None
    literal_value: object | None = None


@dataclass(frozen=True)
class _ClassField:
    name: str
    type_name: str
    index: int


@dataclass(frozen=True)
class _ClassLayout:
    name: str
    struct_name: str
    fields: tuple[_ClassField, ...]

    def field(self, name: str) -> _ClassField | None:
        for field in self.fields:
            if field.name == name:
                return field
        return None

    @property
    def init_name(self) -> str:
        return f"{self.name}.__init__"


@dataclass(frozen=True)
class _FunctionAnalysis:
    slot_infos: dict[str, _ValueInfo]
    value_infos: dict[str, _ValueInfo]
    unsupported: list[str]


@dataclass
class _FunctionState:
    function: HFunction
    target: TargetSpec
    symbol: str
    all_functions: dict[str, HFunction]
    function_symbols: dict[str, str]
    class_layouts: dict[str, _ClassLayout]
    analysis: _FunctionAnalysis
    values: dict[str, LlvmValue] = field(default_factory=dict)
    vars: dict[str, str] = field(default_factory=dict)
    var_types: dict[str, str] = field(default_factory=dict)
    var_runtime_classes: dict[str, str | None] = field(default_factory=dict)
    var_container_kinds: dict[str, str | None] = field(default_factory=dict)
    var_element_types: dict[str, str | None] = field(default_factory=dict)
    var_key_types: dict[str, str | None] = field(default_factory=dict)
    var_value_types: dict[str, str | None] = field(default_factory=dict)
    next_temp_id: int = 0

    def new_temp(self) -> str:
        self.next_temp_id += 1
        return f"%v{self.next_temp_id}"


def emit_llvm_module(module: HModule, target: TargetSpec, *, source_name: str) -> str:
    strings: dict[str, str] = {}
    all_functions = {fn.name: fn for fn in module.functions}
    function_symbols = {fn.name: _function_symbol(fn.name) for fn in module.functions}
    class_layouts = _infer_class_layouts(module)
    analyses = {fn.name: _analyze_function(fn, all_functions, class_layouts) for fn in module.functions}

    unsupported = {name: analysis.unsupported for name, analysis in analyses.items() if analysis.unsupported}
    if unsupported:
        raise LlvmLoweringError(_format_unsupported_lowering(unsupported))

    body_lines: list[str] = []
    for fn in module.functions:
        body_lines.extend(_emit_function(fn, target, all_functions, function_symbols, class_layouts, analyses[fn.name], strings))
        body_lines.append("")

    if "main" in all_functions:
        entry_symbol = function_symbols["main"]
        entry_ret = llvm_type(all_functions["main"].return_type)
        body_lines.extend(_emit_process_entry(entry_symbol, entry_ret))
        body_lines.append("")

    lines = [
        f"; ModuleID = '{source_name}'",
        f'source_filename = "{source_name}"',
        f'target triple = "{_target_triple(target)}"',
        "",
    ]

    for layout in sorted(class_layouts.values(), key=lambda item: item.name):
        field_types = ", ".join(field.type_name for field in layout.fields) or "i8"
        lines.append(f"%{layout.struct_name} = type {{ {field_types} }}")
    if class_layouts:
        lines.append("")

    lines.extend(
        [
            "declare ptr @calloc(i64, i64)",
            "declare void @manv_rt_print_i64(i64)",
            "declare void @manv_rt_print_f64(double)",
            "declare void @manv_rt_print_bool(i1)",
            "declare void @manv_rt_print_cstr(ptr)",
            "declare ptr @manv_rt_array_new(i64, i32, i32)",
            "declare i64 @manv_rt_array_len(ptr)",
            "declare ptr @manv_rt_array_clone_sized(ptr, i64)",
            "declare void @manv_rt_array_set_i64(ptr, i64, i64)",
            "declare void @manv_rt_array_set_f64(ptr, i64, double)",
            "declare void @manv_rt_array_set_ptr(ptr, i64, ptr)",
            "declare i64 @manv_rt_array_get_i64(ptr, i64)",
            "declare double @manv_rt_array_get_f64(ptr, i64)",
            "declare ptr @manv_rt_array_get_ptr(ptr, i64)",
            "declare ptr @manv_rt_map_new(i32, i32)",
            "declare void @manv_rt_map_set_i64_i64(ptr, i64, i64)",
            "declare void @manv_rt_map_set_i64_f64(ptr, i64, double)",
            "declare void @manv_rt_map_set_i64_ptr(ptr, i64, ptr)",
            "declare void @manv_rt_map_set_ptr_i64(ptr, ptr, i64)",
            "declare void @manv_rt_map_set_ptr_f64(ptr, ptr, double)",
            "declare void @manv_rt_map_set_ptr_ptr(ptr, ptr, ptr)",
            "declare i64 @manv_rt_map_get_i64_i64(ptr, i64)",
            "declare double @manv_rt_map_get_i64_f64(ptr, i64)",
            "declare ptr @manv_rt_map_get_i64_ptr(ptr, i64)",
            "declare i64 @manv_rt_map_get_ptr_i64(ptr, ptr)",
            "declare double @manv_rt_map_get_ptr_f64(ptr, ptr)",
            "declare ptr @manv_rt_map_get_ptr_ptr(ptr, ptr)",
            "declare i64 @manv_rt_map_len(ptr)",
            "declare i64 @manv_rt_cstr_len(ptr)",
            "declare ptr @manv_rt_syscall_invoke_cstr(ptr)",
            "declare ptr @manv_rt_syscall_invoke_i64(i64)",
            "declare i1 @manv_rt_syscall_result_ok(ptr)",
            "declare i64 @manv_rt_syscall_result_i64(ptr)",
            "declare ptr @manv_rt_syscall_result_platform(ptr)",
            "declare void @manv_rt_gpu_required_void(ptr)",
            "declare i64 @manv_rt_gpu_required_i64(ptr)",
            "declare double @manv_rt_gpu_required_f64(ptr)",
            "declare float @manv_rt_gpu_required_f32(ptr)",
            "declare ptr @manv_rt_gpu_required_ptr(ptr)",
            "",
        ]
    )

    for text, symbol in sorted(strings.items(), key=lambda item: item[1]):
        escaped = escape_c_string(text)
        length = len(text.encode("utf-8")) + 1
        lines.append(f'@{symbol} = private unnamed_addr constant [{length} x i8] c"{escaped}", align 1')
    if strings:
        lines.append("")

    lines.extend(body_lines)
    return "\n".join(lines).rstrip() + "\n"


def _emit_process_entry(entry_symbol: str, entry_ret: str) -> list[str]:
    """Build the C-process entry shim around ManV's `main`.

    Why this exists:
    - User-facing `main` follows language semantics and may return `none`,
      `bool`, or an integer type.
    - The OS process entry must always be `i32 @main()`.
    """

    lines = ["define i32 @main() {", "entry:"]
    if entry_ret == "void":
        lines.append(f"  call void @{entry_symbol}()")
        lines.append("  ret i32 0")
    elif entry_ret == "i32":
        lines.append(f"  %ret = call i32 @{entry_symbol}()")
        lines.append("  ret i32 %ret")
    elif entry_ret == "i1":
        lines.append(f"  %ret = call i1 @{entry_symbol}()")
        lines.append("  %rc = zext i1 %ret to i32")
        lines.append("  ret i32 %rc")
    else:
        lines.append(f"  %ret = call {entry_ret} @{entry_symbol}()")
        if entry_ret == "i64":
            lines.append("  %rc = trunc i64 %ret to i32")
        else:
            lines.append(f"  %rc = fptosi {entry_ret} %ret to i32")
        lines.append("  ret i32 %rc")
    lines.append("}")
    return lines


def _emit_function(
    function: HFunction,
    target: TargetSpec,
    all_functions: dict[str, HFunction],
    function_symbols: dict[str, str],
    class_layouts: dict[str, _ClassLayout],
    analysis: _FunctionAnalysis,
    strings: dict[str, str],
) -> list[str]:
    symbol = function_symbols[function.name]
    state = _FunctionState(
        function=function,
        target=target,
        symbol=symbol,
        all_functions=all_functions,
        function_symbols=function_symbols,
        class_layouts=class_layouts,
        analysis=analysis,
    )
    return_type = llvm_type(function.return_type)
    params_text: list[str] = []
    for index, param in enumerate(function.params):
        param_name = str(param.get("name"))
        info = _parameter_info(function, index, class_layouts)
        params_text.append(f"{info.type_name} %{sanitize_symbol(param_name)}")
        state.values[f"@arg:{param_name}"] = _materialize(info, f"%{sanitize_symbol(param_name)}")

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
            info = state.analysis.slot_infos.get(var_name, _ValueInfo("i64"))
            slot_type = _storage_type(info.type_name)
            slot = f"%{sanitize_symbol(var_name)}.slot"
            state.vars[var_name] = slot
            state.var_types[var_name] = slot_type
            state.var_runtime_classes[var_name] = info.runtime_class
            state.var_container_kinds[var_name] = info.container_kind
            state.var_element_types[var_name] = info.element_type
            state.var_key_types[var_name] = info.key_type
            state.var_value_types[var_name] = info.value_type
            lines.append(f"  {slot} = alloca {slot_type}, align 8")
        if lines:
            break
    return lines


def _emit_instruction(instr: HInstruction, state: _FunctionState, strings: dict[str, str]) -> list[str]:
    op = instr.op
    out: list[str] = []

    if op == "declare_var":
        return out
    if op == "const":
        if instr.dest is None:
            return out
        literal = instr.attrs.get("value")
        ty = llvm_type(instr.type_name)
        if isinstance(literal, str):
            value = _string_constant_value(literal, state, strings, out)
        elif ty == "void":
            value = LlvmValue("i64", "0", literal_value=literal)
        elif literal is None:
            value = LlvmValue(ty, llvm_zero(ty), literal_value=None)
        elif isinstance(literal, bool):
            value = LlvmValue("i1", "1" if literal else "0", literal_value=literal)
        elif ty in {"float", "double"}:
            value = LlvmValue(ty, str(literal), literal_value=literal)
        else:
            value = LlvmValue(ty, str(int(literal)), literal_value=literal)
        state.values[instr.dest] = value
        return out
    if op == "load_arg":
        if instr.dest is not None:
            arg_name = str(instr.attrs.get("name"))
            state.values[instr.dest] = state.values[f"@arg:{arg_name}"]
        return out
    if op == "store_var":
        var_name = str(instr.args[0])
        source = _operand(str(instr.args[1]), state)
        slot_type = state.var_types[var_name]
        stored = _coerce_value(source, slot_type, state, out)
        out.append(f"  store {stored.type_name} {stored.ref}, ptr {state.vars[var_name]}, align 8")
        state.var_runtime_classes[var_name] = source.runtime_class
        state.var_container_kinds[var_name] = source.container_kind
        state.var_element_types[var_name] = source.element_type
        state.var_key_types[var_name] = source.key_type
        state.var_value_types[var_name] = source.value_type
        return out
    if op == "load_var":
        var_name = str(instr.args[0])
        slot_type = state.var_types[var_name]
        temp = state.new_temp()
        out.append(f"  {temp} = load {slot_type}, ptr {state.vars[var_name]}, align 8")
        state.values[instr.dest or temp] = LlvmValue(
            slot_type,
            temp,
            runtime_class=state.var_runtime_classes.get(var_name),
            container_kind=state.var_container_kinds.get(var_name),
            element_type=state.var_element_types.get(var_name),
            key_type=state.var_key_types.get(var_name),
            value_type=state.var_value_types.get(var_name),
        )
        return out
    if op == "unary":
        operand = _operand(str(instr.args[0]), state)
        operator = str(instr.attrs.get("op"))
        temp = state.new_temp()
        if operator in {"-", "neg"}:
            if operand.type_name in {"float", "double"}:
                out.append(f"  {temp} = fneg {operand.type_name} {operand.ref}")
            else:
                out.append(f"  {temp} = sub {operand.type_name} 0, {operand.ref}")
            state.values[instr.dest or temp] = LlvmValue(operand.type_name, temp)
            return out
        if operator in {"!", "not"}:
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
    if op == "method_call":
        return _emit_method_call(instr, state, out)
    if op == "intrinsic_call":
        return _emit_intrinsic_call(instr, state, out, strings)
    if op == "gpu_call":
        return _emit_gpu_call(instr, state, out, strings)
    if op == "set_attr":
        return _emit_set_attr(instr, state, out)
    if op == "attr":
        return _emit_attr(instr, state, out)
    if op == "array":
        return _emit_array_literal(instr, state, out)
    if op == "alloc_array":
        return _emit_alloc_array(instr, state, out)
    if op == "array_init_sized":
        return _emit_array_init_sized(instr, state, out)
    if op == "map":
        return _emit_map_literal(instr, state, out)
    if op == "index":
        return _emit_index(instr, state, out)
    if op == "set_index":
        return _emit_set_index(instr, state, out)
    raise LlvmLoweringError(f"unsupported HLIR instruction for LLVM lowering: {op}")


def _infer_class_layouts(module: HModule) -> dict[str, _ClassLayout]:
    """Infer native instance layout from HLIR `__init__` stores.

    Why this exists:
    - The source language uses open-ended attribute syntax (`self.name = ...`),
      but the LLVM backend needs a concrete field order and field types.
    - The current object model is intentionally conservative: we only materialize
      fields observed in `__init__`, in source order, which mirrors how the
      interpreter initializes instances today.
    """

    layouts: dict[str, _ClassLayout] = {}
    for function in module.functions:
        owner = _owner_type(function)
        if owner is None or function.name != f"{owner}.__init__":
            continue

        values: dict[str, _ValueInfo] = {}
        slots: dict[str, _ValueInfo | None] = {}
        fixed_slots: set[str] = set()
        field_infos: list[_ClassField] = []

        for index, param in enumerate(function.params):
            info = _parameter_info(function, index, {})
            values[f"@arg:{param.get('name')}"] = info

        for block in function.blocks:
            for instr in block.instructions:
                if instr.op == "declare_var":
                    name = str(instr.attrs.get("name"))
                    explicit = instr.attrs.get("type")
                    if explicit:
                        slots[name] = _value_info_from_type_name(str(explicit), {})
                        fixed_slots.add(name)
                    else:
                        slots[name] = None
                    continue
                if instr.op == "load_arg" and instr.dest is not None:
                    name = str(instr.attrs.get("name"))
                    values[instr.dest] = values[f"@arg:{name}"]
                    continue
                if instr.op == "store_var":
                    var_name = str(instr.args[0])
                    source = values.get(str(instr.args[1]), _ValueInfo("i64"))
                    if var_name not in fixed_slots:
                        slots[var_name] = source
                    continue
                if instr.op == "load_var" and instr.dest is not None:
                    values[instr.dest] = slots.get(str(instr.args[0])) or _ValueInfo("i64")
                    continue
                if instr.op == "const" and instr.dest is not None:
                    literal = instr.attrs.get("value")
                    values[instr.dest] = _const_info(instr.type_name, literal, {})
                    continue
                if instr.op == "set_attr":
                    receiver = values.get(str(instr.args[0]))
                    source = values.get(str(instr.args[1]), _ValueInfo("i64"))
                    attr = str(instr.attrs.get("attr"))
                    if receiver is None or receiver.runtime_class != owner:
                        continue
                    if any(field.name == attr for field in field_infos):
                        continue
                    field_infos.append(_ClassField(name=attr, type_name=_storage_type(source.type_name), index=len(field_infos)))

        if field_infos:
            layouts[owner] = _ClassLayout(name=owner, struct_name=f"manv_obj_{sanitize_symbol(owner)}", fields=tuple(field_infos))

    return layouts


def _analyze_function(
    function: HFunction,
    all_functions: dict[str, HFunction],
    class_layouts: dict[str, _ClassLayout],
) -> _FunctionAnalysis:
    values: dict[str, _ValueInfo] = {}
    slots: dict[str, _ValueInfo | None] = {}
    fixed_slots: set[str] = set()
    aliases: dict[str, str] = {}
    slot_sources: dict[str, str] = {}
    unsupported: list[str] = []

    for index, param in enumerate(function.params):
        values[f"@arg:{param.get('name')}"] = _parameter_info(function, index, class_layouts)

    for block in function.blocks:
        for instr in block.instructions:
            if instr.op == "declare_var":
                name = str(instr.attrs.get("name"))
                explicit = instr.attrs.get("type")
                if explicit:
                    slots[name] = _value_info_from_type_name(str(explicit), class_layouts)
                    fixed_slots.add(name)
                else:
                    slots[name] = None
                continue

            if instr.op == "load_arg" and instr.dest is not None:
                name = str(instr.attrs.get("name"))
                values[instr.dest] = values[f"@arg:{name}"]
                continue

            if instr.op == "store_var":
                var_name = str(instr.args[0])
                source_name = str(instr.args[1])
                source = values.get(source_name, _ValueInfo("i64"))
                if var_name not in fixed_slots:
                    slots[var_name] = source
                elif slots.get(var_name) is not None and source.container_kind is not None:
                    slots[var_name].container_kind = source.container_kind
                    slots[var_name].element_type = source.element_type
                    slots[var_name].key_type = source.key_type
                    slots[var_name].value_type = source.value_type
                if source.container_kind is not None:
                    slot_sources[var_name] = source_name
                continue

            if instr.op == "load_var" and instr.dest is not None:
                var_name = str(instr.args[0])
                values[instr.dest] = slots.get(var_name) or _ValueInfo("i64")
                aliases[instr.dest] = var_name
                continue

            info, reason = _infer_instruction_info(instr, values, function, all_functions, class_layouts)
            if reason is not None and reason not in unsupported:
                unsupported.append(reason)
            if instr.op == "set_index":
                base_name = str(instr.args[0])
                base = values.get(base_name)
                value = values.get(str(instr.args[2]), _ValueInfo("i64"))
                alias_name = aliases.get(base_name)
                if base is not None and base.container_kind in {"array", "empty_array"}:
                    base.container_kind = "array"
                    base.element_type = base.element_type or value.type_name
                    if alias_name is not None and slots.get(alias_name) is not None:
                        slots[alias_name].container_kind = "array"
                        slots[alias_name].element_type = slots[alias_name].element_type or value.type_name
                        source_name = slot_sources.get(alias_name)
                        if source_name is not None and source_name in values:
                            values[source_name].container_kind = "array"
                            values[source_name].element_type = values[source_name].element_type or value.type_name
                if base is not None and base.container_kind == "map":
                    key = values.get(str(instr.args[1]), _ValueInfo("ptr"))
                    base.key_type = base.key_type or key.type_name
                    base.value_type = base.value_type or value.type_name
                    if alias_name is not None and slots.get(alias_name) is not None:
                        slots[alias_name].key_type = slots[alias_name].key_type or key.type_name
                        slots[alias_name].value_type = slots[alias_name].value_type or value.type_name
                        source_name = slot_sources.get(alias_name)
                        if source_name is not None and source_name in values:
                            values[source_name].key_type = values[source_name].key_type or key.type_name
                            values[source_name].value_type = values[source_name].value_type or value.type_name
            if instr.dest is not None and info is not None:
                values[instr.dest] = info

        if block.terminator is not None:
            reason = _unsupported_terminator_reason(block.terminator)
            if reason is not None and reason not in unsupported:
                unsupported.append(reason)

    normalized_slots: dict[str, _ValueInfo] = {}
    for name, info in slots.items():
        normalized_slots[name] = info or _ValueInfo("i64")
    return _FunctionAnalysis(slot_infos=normalized_slots, value_infos=values.copy(), unsupported=unsupported)


def _infer_instruction_info(
    instr: HInstruction,
    values: dict[str, _ValueInfo],
    function: HFunction,
    all_functions: dict[str, HFunction],
    class_layouts: dict[str, _ClassLayout],
) -> tuple[_ValueInfo | None, str | None]:
    if instr.op == "const":
        return _const_info(instr.type_name, instr.attrs.get("value"), class_layouts), None
    if instr.op == "unary":
        operand = values.get(str(instr.args[0]), _ValueInfo("i64"))
        operator = str(instr.attrs.get("op"))
        if operator in {"!", "not"}:
            return _ValueInfo("i1"), None
        return _ValueInfo(operand.type_name), None
    if instr.op == "binop":
        lhs = values.get(str(instr.args[0]), _ValueInfo("i64"))
        rhs = values.get(str(instr.args[1]), _ValueInfo("i64"))
        operator = str(instr.attrs.get("op"))
        if operator in {"==", "!=", "<", "<=", ">", ">=", "&&", "||", "and", "or"}:
            return _ValueInfo("i1"), None
        return _ValueInfo(_unify_numeric_type(lhs.type_name, rhs.type_name)), None
    if instr.op == "call":
        callee = str(instr.attrs.get("callee", ""))
        callee_fn = all_functions.get(callee)
        if callee_fn is not None:
            return _return_info(callee_fn.return_type, class_layouts), None
        if callee in class_layouts:
            return _ValueInfo("ptr", runtime_class=callee), None
        if callee == "<dynamic>":
            return None, "dynamic call target"
        return None, f"call target '{callee}'"
    if instr.op == "method_call":
        receiver = values.get(str(instr.args[0]))
        if receiver is None or receiver.runtime_class is None:
            return None, "method call on non-class value"
        callee_name = f"{receiver.runtime_class}.{instr.attrs.get('method')}"
        callee_fn = all_functions.get(callee_name)
        if callee_fn is None:
            return None, f"method target '{callee_name}'"
        return _return_info(callee_fn.return_type, class_layouts), None
    if instr.op == "attr":
        receiver = values.get(str(instr.args[0]))
        attr = str(instr.attrs.get("attr"))
        if receiver is None or receiver.runtime_class is None:
            return None, "attribute access on non-class value"
        layout = class_layouts.get(receiver.runtime_class)
        field = None if layout is None else layout.field(attr)
        if field is None:
            return None, f"attribute '{attr}' on '{receiver.runtime_class}'"
        return _ValueInfo(field.type_name), None
    if instr.op == "set_attr":
        receiver = values.get(str(instr.args[0]))
        attr = str(instr.attrs.get("attr"))
        if receiver is None or receiver.runtime_class is None:
            return None, "attribute assignment on non-class value"
        layout = class_layouts.get(receiver.runtime_class)
        if layout is None or layout.field(attr) is None:
            return None, f"attribute '{attr}' on '{receiver.runtime_class}'"
        return None, None
    if instr.op == "intrinsic_call":
        name = str(instr.attrs.get("name", ""))
        if name == "io_print":
            return _ValueInfo("i64"), None
        if name == "core_len":
            operand = values.get(str(instr.args[0]))
            if operand is None:
                return None, "intrinsic 'core_len'"
            if operand.container_kind not in {"array", "map", "empty_array"} and operand.type_name != "ptr":
                return None, "intrinsic 'core_len'"
            return _ValueInfo("i64"), None
        if name == "syscall_invoke":
            args_array = values.get(str(instr.args[1]))
            if args_array is None or args_array.container_kind != "empty_array":
                return None, "non-empty syscall argument array"
            target = values.get(str(instr.args[0]), _ValueInfo("ptr"))
            if target.type_name not in {"ptr", "i32", "i64"}:
                return None, "syscall target must be int or string"
            return _ValueInfo("ptr", container_kind="syscall_result"), None
        return None, f"intrinsic '{name}'"
    if instr.op == "gpu_call":
        callee = str(instr.attrs.get("callee", ""))
        callee_fn = all_functions.get(callee)
        if callee_fn is None:
            return None, f"gpu target '{callee}'"
        return _return_info(callee_fn.return_type, class_layouts), None
    if instr.op == "array":
        if not instr.args:
            return _ValueInfo("ptr", container_kind="empty_array", element_type=None), None
        first = values.get(str(instr.args[0]), _ValueInfo("i64"))
        elem_type = first.type_name
        for arg in instr.args[1:]:
            info = values.get(str(arg), _ValueInfo("i64"))
            if info.type_name != elem_type:
                return None, "heterogeneous array literal construction"
        return _ValueInfo("ptr", container_kind="array", element_type=elem_type), None
    if instr.op == "alloc_array":
        return _ValueInfo("ptr", container_kind="array", element_type=None), None
    if instr.op == "array_init_sized":
        seed = values.get(str(instr.args[0]))
        if seed is None or seed.container_kind not in {"array", "empty_array"}:
            return None, "sized array initialization"
        return _ValueInfo("ptr", container_kind="array", element_type=seed.element_type), None
    if instr.op == "map":
        if len(instr.args) % 2 != 0:
            return None, "map literal construction"
        if not instr.args:
            return _ValueInfo("ptr", container_kind="map", key_type=None, value_type=None), None
        key_info = values.get(str(instr.args[0]), _ValueInfo("ptr"))
        value_info = values.get(str(instr.args[1]), _ValueInfo("i64"))
        for index in range(0, len(instr.args), 2):
            key = values.get(str(instr.args[index]), _ValueInfo("ptr"))
            value = values.get(str(instr.args[index + 1]), _ValueInfo("i64"))
            if key.type_name != key_info.type_name or value.type_name != value_info.type_name:
                return None, "heterogeneous map literal construction"
        return _ValueInfo("ptr", container_kind="map", key_type=key_info.type_name, value_type=value_info.type_name), None
    if instr.op == "index":
        base = values.get(str(instr.args[0]))
        key = values.get(str(instr.args[1]))
        if base is None or base.container_kind != "syscall_result":
            if base is None:
                return None, "index access"
            if base.container_kind in {"array", "empty_array"}:
                return _ValueInfo(base.element_type or "i64"), None
            if base.container_kind == "map":
                return _ValueInfo(base.value_type or "i64"), None
            return None, "index access"
        if key is None or not isinstance(key.literal_value, str):
            return None, "dynamic syscall result key"
        if key.literal_value == "ok":
            return _ValueInfo("i1"), None
        if key.literal_value == "result":
            return _ValueInfo("i64"), None
        if key.literal_value == "platform":
            return _ValueInfo("ptr"), None
        return None, f"unsupported syscall result key '{key.literal_value}'"
    if instr.op == "set_index":
        base = values.get(str(instr.args[0]))
        key = values.get(str(instr.args[1]))
        value = values.get(str(instr.args[2]), _ValueInfo("i64"))
        if base is None:
            return None, "index assignment"
        if base.container_kind in {"array", "empty_array"}:
            if key is None or key.type_name not in {"i32", "i64"}:
                return None, "array index must be int"
            base.element_type = base.element_type or value.type_name
            return None, None
        if base.container_kind == "map":
            if key is None:
                return None, "map key type"
            base.key_type = base.key_type or key.type_name
            base.value_type = base.value_type or value.type_name
            return None, None
        return None, "index assignment"
    if instr.op in {"declare_var", "load_arg", "store_var", "load_var"}:
        return None, None
    return None, f"HLIR instruction '{instr.op}'"


def _format_unsupported_lowering(unsupported: dict[str, list[str]]) -> str:
    parts: list[str] = []
    for fn_name, reasons in unsupported.items():
        rendered = ", ".join(sorted(reasons))
        parts.append(f"{fn_name}: {rendered}")
    return "unsupported HLIR features for LLVM lowering: " + "; ".join(parts)


def _unsupported_terminator_reason(term: HTerminator) -> str | None:
    if term.op in {"br", "cbr", "ret"}:
        return None
    return f"terminator '{term.op}'"


def _emit_call(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    callee = str(instr.attrs.get("callee", ""))
    if callee == "print":
        value = _operand(str(instr.args[0]), state)
        out.extend(_emit_print_call(value, state))
        if instr.dest is not None:
            state.values[instr.dest] = LlvmValue("i64", "0", literal_value=0)
        return out

    callee_fn = state.all_functions.get(callee)
    if callee_fn is not None:
        result = _emit_function_call(callee_fn, [_operand(str(arg), state) for arg in instr.args], state, out)
        if instr.dest is not None:
            state.values[instr.dest] = result
        return out

    layout = state.class_layouts.get(callee)
    if layout is None:
        raise LlvmLoweringError(f"unknown callee '{callee}'")

    size_expr = f"ptrtoint (ptr getelementptr ({_layout_ref(layout)}, ptr null, i32 1) to i64)"
    temp = state.new_temp()
    out.append(f"  {temp} = call ptr @calloc(i64 1, i64 {size_expr})")
    instance = LlvmValue("ptr", temp, runtime_class=callee)

    init_fn = state.all_functions.get(layout.init_name)
    if init_fn is not None:
        _emit_function_call(init_fn, [instance, *[_operand(str(arg), state) for arg in instr.args]], state, out)

    if instr.dest is not None:
        state.values[instr.dest] = instance
    return out


def _emit_method_call(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    receiver = _operand(str(instr.args[0]), state)
    if receiver.runtime_class is None:
        raise LlvmLoweringError("method call requires a class instance receiver")
    callee_name = f"{receiver.runtime_class}.{instr.attrs.get('method')}"
    callee_fn = state.all_functions.get(callee_name)
    if callee_fn is None:
        raise LlvmLoweringError(f"unknown method target '{callee_name}'")
    result = _emit_function_call(callee_fn, [receiver, *[_operand(str(arg), state) for arg in instr.args[1:]]], state, out)
    if instr.dest is not None:
        state.values[instr.dest] = result
    return out


def _emit_function_call(callee_fn: HFunction, args: list[LlvmValue], state: _FunctionState, out: list[str]) -> LlvmValue:
    arg_values: list[LlvmValue] = []
    for index, param in enumerate(callee_fn.params):
        expected = _parameter_info(callee_fn, index, state.class_layouts)
        arg_values.append(_coerce_value(args[index], expected.type_name, state, out))

    return_type = llvm_type(callee_fn.return_type)
    arg_text = ", ".join(f"{value.type_name} {value.ref}" for value in arg_values)
    if return_type == "void":
        out.append(f"  call void @{state.function_symbols[callee_fn.name]}({arg_text})")
        return LlvmValue("i64", "0", literal_value=0)

    temp = state.new_temp()
    out.append(f"  {temp} = call {return_type} @{state.function_symbols[callee_fn.name]}({arg_text})")
    runtime_class = callee_fn.return_type if callee_fn.return_type in state.class_layouts else None
    container_kind = _container_kind_for_type(callee_fn.return_type)
    return LlvmValue(return_type, temp, runtime_class=runtime_class, container_kind=container_kind)


def _emit_intrinsic_call(instr: HInstruction, state: _FunctionState, out: list[str], strings: dict[str, str]) -> list[str]:
    name = str(instr.attrs.get("name", ""))
    if name == "io_print":
        if len(instr.args) != 1:
            raise LlvmLoweringError("io_print expects one argument")
        out.extend(_emit_print_call(_operand(str(instr.args[0]), state), state))
        if instr.dest is not None:
            state.values[instr.dest] = LlvmValue("i64", "0", literal_value=0)
        return out
    if name == "core_len":
        if len(instr.args) != 1:
            raise LlvmLoweringError("core_len expects one argument")
        target = _operand(str(instr.args[0]), state)
        temp = state.new_temp()
        if target.container_kind in {"array", "empty_array"}:
            out.append(f"  {temp} = call i64 @manv_rt_array_len(ptr {target.ref})")
        elif target.container_kind == "map":
            out.append(f"  {temp} = call i64 @manv_rt_map_len(ptr {target.ref})")
        elif target.type_name == "ptr":
            out.append(f"  {temp} = call i64 @manv_rt_cstr_len(ptr {target.ref})")
        else:
            raise LlvmLoweringError("core_len requires an array, map, or string-like pointer")
        if instr.dest is not None:
            state.values[instr.dest] = LlvmValue("i64", temp)
        return out
    if name == "syscall_invoke":
        if len(instr.args) != 2:
            raise LlvmLoweringError("syscall_invoke expects target and args array")
        target = _operand(str(instr.args[0]), state)
        args_array = _operand(str(instr.args[1]), state)
        if args_array.container_kind != "empty_array":
            raise LlvmLoweringError("LLVM lowering only supports empty syscall argument arrays today")
        temp = state.new_temp()
        if target.type_name == "ptr":
            out.append(f"  {temp} = call ptr @manv_rt_syscall_invoke_cstr(ptr {target.ref})")
        else:
            numeric = _coerce_value(target, "i64", state, out)
            out.append(f"  {temp} = call ptr @manv_rt_syscall_invoke_i64(i64 {numeric.ref})")
        if instr.dest is not None:
            state.values[instr.dest] = LlvmValue("ptr", temp, container_kind="syscall_result")
        return out
    raise LlvmLoweringError(f"unsupported intrinsic call for LLVM lowering: {name}")


def _emit_gpu_call(instr: HInstruction, state: _FunctionState, out: list[str], strings: dict[str, str]) -> list[str]:
    """Lower `gpu_call` through a native host boundary.

    Why this is shaped this way:
    - Best-effort GPU calls must remain semantically meaningful even before the
      native host runtime has full device dispatch. We therefore lower them to
      the CPU function body directly.
    - Required GPU calls must not silently change meaning. Until the native host
      runtime can call the real device resolver, we lower them to a deterministic
      runtime failure helper instead of pretending offload succeeded.
    """

    callee = str(instr.attrs.get("callee", ""))
    callee_fn = state.all_functions.get(callee)
    if callee_fn is None:
        raise LlvmLoweringError(f"unknown gpu callee '{callee}'")

    policy = str(instr.attrs.get("policy", "best_effort"))
    if policy == "best_effort":
        result = _emit_function_call(callee_fn, [_operand(str(arg), state) for arg in instr.args], state, out)
        if instr.dest is not None:
            state.values[instr.dest] = result
        return out

    helper_name, helper_type = _gpu_required_helper(llvm_type(callee_fn.return_type))
    message = _string_constant_value(callee, state, strings, out)
    if helper_type == "void":
        out.append(f"  call void @{helper_name}(ptr {message.ref})")
        if instr.dest is not None:
            state.values[instr.dest] = LlvmValue("i64", "0", literal_value=0)
        return out

    temp = state.new_temp()
    out.append(f"  {temp} = call {helper_type} @{helper_name}(ptr {message.ref})")
    if instr.dest is not None:
        state.values[instr.dest] = LlvmValue(helper_type, temp)
    return out


def _emit_set_attr(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    receiver = _operand(str(instr.args[0]), state)
    if receiver.runtime_class is None:
        raise LlvmLoweringError("set_attr requires a class receiver")
    layout = state.class_layouts.get(receiver.runtime_class)
    if layout is None:
        raise LlvmLoweringError(f"missing native layout for class '{receiver.runtime_class}'")
    field = layout.field(str(instr.attrs.get("attr")))
    if field is None:
        raise LlvmLoweringError(f"unknown field '{instr.attrs.get('attr')}' on class '{receiver.runtime_class}'")
    value = _coerce_value(_operand(str(instr.args[1]), state), field.type_name, state, out)
    field_ptr = state.new_temp()
    out.append(
        f"  {field_ptr} = getelementptr inbounds {_layout_ref(layout)}, ptr {receiver.ref}, i32 0, i32 {field.index}"
    )
    out.append(f"  store {value.type_name} {value.ref}, ptr {field_ptr}, align 8")
    return out


def _emit_attr(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    receiver = _operand(str(instr.args[0]), state)
    if receiver.runtime_class is None:
        raise LlvmLoweringError("attr requires a class receiver")
    layout = state.class_layouts.get(receiver.runtime_class)
    if layout is None:
        raise LlvmLoweringError(f"missing native layout for class '{receiver.runtime_class}'")
    field = layout.field(str(instr.attrs.get("attr")))
    if field is None:
        raise LlvmLoweringError(f"unknown field '{instr.attrs.get('attr')}' on class '{receiver.runtime_class}'")
    field_ptr = state.new_temp()
    out.append(
        f"  {field_ptr} = getelementptr inbounds {_layout_ref(layout)}, ptr {receiver.ref}, i32 0, i32 {field.index}"
    )
    temp = state.new_temp()
    out.append(f"  {temp} = load {field.type_name}, ptr {field_ptr}, align 8")
    if instr.dest is not None:
        state.values[instr.dest] = LlvmValue(field.type_name, temp)
    return out


def _emit_array_literal(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    """Materialize a runtime array object.

    Why this is eager instead of clever:
    - HLIR array literals are authoritative concrete allocations today. LLVM
      lowering mirrors that model instead of trying to re-interpret them as
      compile-time constants with special semantics.
    - The runtime helper keeps the memory layout deterministic, while the
      element setter selection here preserves the element type inferred during
      whole-function analysis.
    """

    info = state.analysis.value_infos.get(instr.dest or "", _ValueInfo("ptr", container_kind="empty_array"))
    element_type = info.element_type or "i64"
    array_value = _emit_array_allocation(len(instr.args), element_type, state, out)
    for index, arg in enumerate(instr.args):
        _emit_array_store(array_value, LlvmValue("i64", str(index), literal_value=index), _operand(str(arg), state), state, out)
    if instr.dest is not None:
        state.values[instr.dest] = LlvmValue(
            "ptr",
            array_value.ref,
            container_kind="empty_array" if not instr.args else "array",
            element_type=element_type if instr.args else None,
        )
    return out


def _emit_alloc_array(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    info = state.analysis.value_infos.get(instr.dest or "", _ValueInfo("ptr", container_kind="array"))
    element_type = info.element_type or "i64"
    size = _coerce_value(_operand(str(instr.args[0]), state), "i64", state, out)
    array_value = _emit_array_allocation_ref(size.ref, element_type, state, out)
    if instr.dest is not None:
        state.values[instr.dest] = LlvmValue("ptr", array_value.ref, container_kind="array", element_type=element_type)
    return out


def _emit_array_init_sized(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    seed = _operand(str(instr.args[0]), state)
    if seed.container_kind not in {"array", "empty_array"}:
        raise LlvmLoweringError("array_init_sized requires an array seed")
    size = _coerce_value(_operand(str(instr.args[1]), state), "i64", state, out)
    temp = state.new_temp()
    out.append(f"  {temp} = call ptr @manv_rt_array_clone_sized(ptr {seed.ref}, i64 {size.ref})")
    if instr.dest is not None:
        element_type = seed.element_type or state.analysis.value_infos.get(instr.dest, _ValueInfo("ptr")).element_type or "i64"
        state.values[instr.dest] = LlvmValue("ptr", temp, container_kind="array", element_type=element_type)
    return out


def _emit_map_literal(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    info = state.analysis.value_infos.get(instr.dest or "", _ValueInfo("ptr", container_kind="map"))
    key_type = info.key_type or "ptr"
    value_type = info.value_type or "i64"
    temp = state.new_temp()
    out.append(f"  {temp} = call ptr @manv_rt_map_new(i32 {_box_tag(key_type)}, i32 {_box_tag(value_type)})")
    map_value = LlvmValue("ptr", temp, container_kind="map", key_type=key_type, value_type=value_type)
    for index in range(0, len(instr.args), 2):
        key = _operand(str(instr.args[index]), state)
        value = _operand(str(instr.args[index + 1]), state)
        _emit_map_store(map_value, key, value, state, out)
    if instr.dest is not None:
        state.values[instr.dest] = map_value
    return out


def _emit_index(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    base = _operand(str(instr.args[0]), state)
    key = _operand(str(instr.args[1]), state)
    if base.container_kind in {"array", "empty_array"}:
        value = _emit_array_load(base, key, state, out)
    elif base.container_kind == "map":
        value = _emit_map_load(base, key, state, out)
    elif base.container_kind == "syscall_result":
        if not isinstance(key.literal_value, str):
            raise LlvmLoweringError("syscall result indexing requires a constant string key")
        temp = state.new_temp()
        if key.literal_value == "ok":
            out.append(f"  {temp} = call i1 @manv_rt_syscall_result_ok(ptr {base.ref})")
            value = LlvmValue("i1", temp)
        elif key.literal_value == "result":
            out.append(f"  {temp} = call i64 @manv_rt_syscall_result_i64(ptr {base.ref})")
            value = LlvmValue("i64", temp)
        elif key.literal_value == "platform":
            out.append(f"  {temp} = call ptr @manv_rt_syscall_result_platform(ptr {base.ref})")
            value = LlvmValue("ptr", temp)
        else:
            raise LlvmLoweringError(f"unsupported syscall result key '{key.literal_value}'")
    else:
        raise LlvmLoweringError("index requires an array, map, or syscall result receiver")

    if instr.dest is not None:
        state.values[instr.dest] = value
    return out


def _emit_set_index(instr: HInstruction, state: _FunctionState, out: list[str]) -> list[str]:
    base = _operand(str(instr.args[0]), state)
    key = _operand(str(instr.args[1]), state)
    value = _operand(str(instr.args[2]), state)
    if base.container_kind in {"array", "empty_array"}:
        _emit_array_store(base, key, value, state, out)
        return out
    if base.container_kind == "map":
        _emit_map_store(base, key, value, state, out)
        return out
    raise LlvmLoweringError("index assignment requires an array or map receiver")


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


def _emit_array_allocation(length: int, element_type: str, state: _FunctionState, out: list[str]) -> LlvmValue:
    return _emit_array_allocation_ref(str(length), element_type, state, out)


def _emit_array_allocation_ref(length_ref: str, element_type: str, state: _FunctionState, out: list[str]) -> LlvmValue:
    temp = state.new_temp()
    out.append(f"  {temp} = call ptr @manv_rt_array_new(i64 {length_ref}, i32 {_box_tag(element_type)}, i32 0)")
    return LlvmValue("ptr", temp, container_kind="array", element_type=element_type)


def _emit_array_store(base: LlvmValue, key: LlvmValue, value: LlvmValue, state: _FunctionState, out: list[str]) -> None:
    index = _coerce_value(key, "i64", state, out)
    element_type = base.element_type or value.type_name or "i64"
    if element_type == "i1":
        coerced = _coerce_value(value, "i64", state, out)
        out.append(f"  call void @manv_rt_array_set_i64(ptr {base.ref}, i64 {index.ref}, i64 {coerced.ref})")
        return
    if element_type in {"i32", "i64"}:
        coerced = _coerce_value(value, "i64", state, out)
        out.append(f"  call void @manv_rt_array_set_i64(ptr {base.ref}, i64 {index.ref}, i64 {coerced.ref})")
        return
    if element_type in {"float", "double"}:
        coerced = _coerce_value(value, "double", state, out)
        out.append(f"  call void @manv_rt_array_set_f64(ptr {base.ref}, i64 {index.ref}, double {coerced.ref})")
        return
    if element_type == "ptr":
        coerced = _coerce_value(value, "ptr", state, out)
        out.append(f"  call void @manv_rt_array_set_ptr(ptr {base.ref}, i64 {index.ref}, ptr {coerced.ref})")
        return
    raise LlvmLoweringError(f"unsupported native array element type '{element_type}'")


def _emit_array_load(base: LlvmValue, key: LlvmValue, state: _FunctionState, out: list[str]) -> LlvmValue:
    index = _coerce_value(key, "i64", state, out)
    element_type = base.element_type or "i64"
    temp = state.new_temp()
    if element_type == "i1":
        out.append(f"  {temp} = call i64 @manv_rt_array_get_i64(ptr {base.ref}, i64 {index.ref})")
        bool_temp = state.new_temp()
        out.append(f"  {bool_temp} = trunc i64 {temp} to i1")
        return LlvmValue("i1", bool_temp)
    if element_type == "i32":
        out.append(f"  {temp} = call i64 @manv_rt_array_get_i64(ptr {base.ref}, i64 {index.ref})")
        narrow = state.new_temp()
        out.append(f"  {narrow} = trunc i64 {temp} to i32")
        return LlvmValue("i32", narrow)
    if element_type == "i64":
        out.append(f"  {temp} = call i64 @manv_rt_array_get_i64(ptr {base.ref}, i64 {index.ref})")
        return LlvmValue("i64", temp)
    if element_type == "float":
        out.append(f"  {temp} = call double @manv_rt_array_get_f64(ptr {base.ref}, i64 {index.ref})")
        narrow = state.new_temp()
        out.append(f"  {narrow} = fptrunc double {temp} to float")
        return LlvmValue("float", narrow)
    if element_type == "double":
        out.append(f"  {temp} = call double @manv_rt_array_get_f64(ptr {base.ref}, i64 {index.ref})")
        return LlvmValue("double", temp)
    if element_type == "ptr":
        out.append(f"  {temp} = call ptr @manv_rt_array_get_ptr(ptr {base.ref}, i64 {index.ref})")
        return LlvmValue("ptr", temp)
    raise LlvmLoweringError(f"unsupported native array element type '{element_type}'")


def _emit_map_store(base: LlvmValue, key: LlvmValue, value: LlvmValue, state: _FunctionState, out: list[str]) -> None:
    key_type = base.key_type or key.type_name or "ptr"
    value_type = base.value_type or value.type_name or "i64"
    key_value = _coerce_map_key(key, key_type, state, out)
    if key_type == "i64" and value_type in {"i1", "i32", "i64"}:
        stored = _coerce_value(value, "i64", state, out)
        out.append(f"  call void @manv_rt_map_set_i64_i64(ptr {base.ref}, i64 {key_value.ref}, i64 {stored.ref})")
        return
    if key_type == "i64" and value_type in {"float", "double"}:
        stored = _coerce_value(value, "double", state, out)
        out.append(f"  call void @manv_rt_map_set_i64_f64(ptr {base.ref}, i64 {key_value.ref}, double {stored.ref})")
        return
    if key_type == "i64" and value_type == "ptr":
        stored = _coerce_value(value, "ptr", state, out)
        out.append(f"  call void @manv_rt_map_set_i64_ptr(ptr {base.ref}, i64 {key_value.ref}, ptr {stored.ref})")
        return
    if key_type == "ptr" and value_type in {"i1", "i32", "i64"}:
        stored = _coerce_value(value, "i64", state, out)
        out.append(f"  call void @manv_rt_map_set_ptr_i64(ptr {base.ref}, ptr {key_value.ref}, i64 {stored.ref})")
        return
    if key_type == "ptr" and value_type in {"float", "double"}:
        stored = _coerce_value(value, "double", state, out)
        out.append(f"  call void @manv_rt_map_set_ptr_f64(ptr {base.ref}, ptr {key_value.ref}, double {stored.ref})")
        return
    if key_type == "ptr" and value_type == "ptr":
        stored = _coerce_value(value, "ptr", state, out)
        out.append(f"  call void @manv_rt_map_set_ptr_ptr(ptr {base.ref}, ptr {key_value.ref}, ptr {stored.ref})")
        return
    raise LlvmLoweringError(f"unsupported native map key/value types '{key_type}' -> '{value_type}'")


def _emit_map_load(base: LlvmValue, key: LlvmValue, state: _FunctionState, out: list[str]) -> LlvmValue:
    key_type = base.key_type or key.type_name or "ptr"
    value_type = base.value_type or "i64"
    key_value = _coerce_map_key(key, key_type, state, out)
    temp = state.new_temp()
    if key_type == "i64" and value_type == "i1":
        out.append(f"  {temp} = call i64 @manv_rt_map_get_i64_i64(ptr {base.ref}, i64 {key_value.ref})")
        narrow = state.new_temp()
        out.append(f"  {narrow} = trunc i64 {temp} to i1")
        return LlvmValue("i1", narrow)
    if key_type == "i64" and value_type == "i32":
        out.append(f"  {temp} = call i64 @manv_rt_map_get_i64_i64(ptr {base.ref}, i64 {key_value.ref})")
        narrow = state.new_temp()
        out.append(f"  {narrow} = trunc i64 {temp} to i32")
        return LlvmValue("i32", narrow)
    if key_type == "i64" and value_type == "i64":
        out.append(f"  {temp} = call i64 @manv_rt_map_get_i64_i64(ptr {base.ref}, i64 {key_value.ref})")
        return LlvmValue("i64", temp)
    if key_type == "i64" and value_type == "float":
        out.append(f"  {temp} = call double @manv_rt_map_get_i64_f64(ptr {base.ref}, i64 {key_value.ref})")
        narrow = state.new_temp()
        out.append(f"  {narrow} = fptrunc double {temp} to float")
        return LlvmValue("float", narrow)
    if key_type == "i64" and value_type == "double":
        out.append(f"  {temp} = call double @manv_rt_map_get_i64_f64(ptr {base.ref}, i64 {key_value.ref})")
        return LlvmValue("double", temp)
    if key_type == "i64" and value_type == "ptr":
        out.append(f"  {temp} = call ptr @manv_rt_map_get_i64_ptr(ptr {base.ref}, i64 {key_value.ref})")
        return LlvmValue("ptr", temp)
    if key_type == "ptr" and value_type == "i1":
        out.append(f"  {temp} = call i64 @manv_rt_map_get_ptr_i64(ptr {base.ref}, ptr {key_value.ref})")
        narrow = state.new_temp()
        out.append(f"  {narrow} = trunc i64 {temp} to i1")
        return LlvmValue("i1", narrow)
    if key_type == "ptr" and value_type == "i32":
        out.append(f"  {temp} = call i64 @manv_rt_map_get_ptr_i64(ptr {base.ref}, ptr {key_value.ref})")
        narrow = state.new_temp()
        out.append(f"  {narrow} = trunc i64 {temp} to i32")
        return LlvmValue("i32", narrow)
    if key_type == "ptr" and value_type == "i64":
        out.append(f"  {temp} = call i64 @manv_rt_map_get_ptr_i64(ptr {base.ref}, ptr {key_value.ref})")
        return LlvmValue("i64", temp)
    if key_type == "ptr" and value_type == "float":
        out.append(f"  {temp} = call double @manv_rt_map_get_ptr_f64(ptr {base.ref}, ptr {key_value.ref})")
        narrow = state.new_temp()
        out.append(f"  {narrow} = fptrunc double {temp} to float")
        return LlvmValue("float", narrow)
    if key_type == "ptr" and value_type == "double":
        out.append(f"  {temp} = call double @manv_rt_map_get_ptr_f64(ptr {base.ref}, ptr {key_value.ref})")
        return LlvmValue("double", temp)
    if key_type == "ptr" and value_type == "ptr":
        out.append(f"  {temp} = call ptr @manv_rt_map_get_ptr_ptr(ptr {base.ref}, ptr {key_value.ref})")
        return LlvmValue("ptr", temp)
    raise LlvmLoweringError(f"unsupported native map key/value types '{key_type}' -> '{value_type}'")


def _coerce_map_key(key: LlvmValue, key_type: str, state: _FunctionState, out: list[str]) -> LlvmValue:
    if key_type == "ptr":
        return _coerce_value(key, "ptr", state, out)
    if key_type in {"i1", "i32", "i64"}:
        return _coerce_value(key, "i64", state, out)
    raise LlvmLoweringError(f"unsupported native map key type '{key_type}'")


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
    if operator in {"&&", "||", "and", "or"}:
        left_bool = _coerce_value(left, "i1", state, out)
        right_bool = _coerce_value(right, "i1", state, out)
        opcode = "and" if operator in {"&&", "and"} else "or"
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
        return LlvmValue(
            target_type,
            temp,
            runtime_class=value.runtime_class,
            container_kind=value.container_kind,
            element_type=value.element_type,
            key_type=value.key_type,
            value_type=value.value_type,
        )
    if value.type_name in {"i32", "i64"} and target_type == "i1":
        out.append(f"  {temp} = icmp ne {value.type_name} {value.ref}, 0")
        return LlvmValue(
            "i1",
            temp,
            runtime_class=value.runtime_class,
            container_kind=value.container_kind,
            element_type=value.element_type,
            key_type=value.key_type,
            value_type=value.value_type,
        )
    if value.type_name == "i32" and target_type == "i64":
        out.append(f"  {temp} = sext i32 {value.ref} to i64")
        return LlvmValue(
            "i64",
            temp,
            runtime_class=value.runtime_class,
            container_kind=value.container_kind,
            element_type=value.element_type,
            key_type=value.key_type,
            value_type=value.value_type,
        )
    if value.type_name == "i64" and target_type == "i32":
        out.append(f"  {temp} = trunc i64 {value.ref} to i32")
        return LlvmValue(
            "i32",
            temp,
            runtime_class=value.runtime_class,
            container_kind=value.container_kind,
            element_type=value.element_type,
            key_type=value.key_type,
            value_type=value.value_type,
        )
    if value.type_name == "float" and target_type == "double":
        out.append(f"  {temp} = fpext float {value.ref} to double")
        return LlvmValue(
            "double",
            temp,
            runtime_class=value.runtime_class,
            container_kind=value.container_kind,
            element_type=value.element_type,
            key_type=value.key_type,
            value_type=value.value_type,
        )
    if value.type_name == "double" and target_type == "float":
        out.append(f"  {temp} = fptrunc double {value.ref} to float")
        return LlvmValue(
            "float",
            temp,
            runtime_class=value.runtime_class,
            container_kind=value.container_kind,
            element_type=value.element_type,
            key_type=value.key_type,
            value_type=value.value_type,
        )
    if value.type_name in {"i32", "i64"} and target_type in {"float", "double"}:
        out.append(f"  {temp} = sitofp {value.type_name} {value.ref} to {target_type}")
        return LlvmValue(
            target_type,
            temp,
            runtime_class=value.runtime_class,
            container_kind=value.container_kind,
            element_type=value.element_type,
            key_type=value.key_type,
            value_type=value.value_type,
        )
    if value.type_name in {"float", "double"} and target_type in {"i32", "i64"}:
        out.append(f"  {temp} = fptosi {value.type_name} {value.ref} to {target_type}")
        return LlvmValue(
            target_type,
            temp,
            runtime_class=value.runtime_class,
            container_kind=value.container_kind,
            element_type=value.element_type,
            key_type=value.key_type,
            value_type=value.value_type,
        )
    raise LlvmLoweringError(f"cannot coerce {value.type_name} to {target_type}")


def _parameter_info(function: HFunction, index: int, class_layouts: dict[str, _ClassLayout]) -> _ValueInfo:
    param = function.params[index]
    name = str(param.get("name"))
    owner = _owner_type(function)
    if owner is not None and index == 0 and name == "self":
        return _ValueInfo("ptr", runtime_class=owner)
    return _value_info_from_type_name(param.get("type"), class_layouts)


def _return_info(type_name: str | None, class_layouts: dict[str, _ClassLayout]) -> _ValueInfo:
    return _value_info_from_type_name(type_name, class_layouts)


def _value_info_from_type_name(type_name: str | None, class_layouts: dict[str, _ClassLayout]) -> _ValueInfo:
    llvm_ty = llvm_type(type_name)
    runtime_class = type_name if type_name in class_layouts else None
    normalized = (type_name or "").strip()
    container_kind = _container_kind_for_type(type_name)
    return _ValueInfo(
        type_name=_storage_type(llvm_ty),
        runtime_class=runtime_class,
        container_kind=container_kind,
        element_type=None if container_kind != "array" else None,
        key_type=None if container_kind != "map" else None,
        value_type=None if container_kind != "map" else None,
    )


def _const_info(type_name: str | None, literal: object, class_layouts: dict[str, _ClassLayout]) -> _ValueInfo:
    if isinstance(literal, str):
        return _ValueInfo("ptr", literal_value=literal)
    if isinstance(literal, bool):
        return _ValueInfo("i1", literal_value=literal)
    info = _value_info_from_type_name(type_name, class_layouts)
    if info.type_name == "void":
        return _ValueInfo("i64", literal_value=literal)
    return _ValueInfo(info.type_name, literal_value=literal)


def _container_kind_for_type(type_name: str | None) -> str | None:
    normalized = (type_name or "").strip()
    if normalized == "array" or normalized.startswith("array["):
        return "array"
    if normalized == "map":
        return "map"
    return None


def _storage_type(type_name: str) -> str:
    return "i64" if type_name == "void" else type_name


def _materialize(info: _ValueInfo, ref: str) -> LlvmValue:
    return LlvmValue(
        _storage_type(info.type_name),
        ref,
        runtime_class=info.runtime_class,
        container_kind=info.container_kind,
        element_type=info.element_type,
        key_type=info.key_type,
        value_type=info.value_type,
        literal_value=info.literal_value,
    )


def _string_constant_value(text: str, state: _FunctionState, strings: dict[str, str], out: list[str]) -> LlvmValue:
    symbol = strings.setdefault(text, f".str.{len(strings)}")
    length = len(text.encode("utf-8")) + 1
    temp = state.new_temp()
    out.append(f"  {temp} = getelementptr inbounds [{length} x i8], ptr @{symbol}, i64 0, i64 0")
    return LlvmValue("ptr", temp, literal_value=text)


def _gpu_required_helper(return_type: str) -> tuple[str, str]:
    if return_type == "void":
        return "manv_rt_gpu_required_void", "void"
    if return_type == "float":
        return "manv_rt_gpu_required_f32", "float"
    if return_type == "double":
        return "manv_rt_gpu_required_f64", "double"
    if return_type == "ptr":
        return "manv_rt_gpu_required_ptr", "ptr"
    return "manv_rt_gpu_required_i64", "i64"


def _boxed_storage_type(value_type: str) -> str:
    if value_type in {"i1", "i32", "i64"}:
        return "i64"
    if value_type in {"float", "double"}:
        return "double"
    if value_type == "ptr":
        return "ptr"
    raise LlvmLoweringError(f"unsupported boxed storage type '{value_type}'")


def _box_tag(value_type: str) -> int:
    if value_type in {"i1"}:
        return 1
    if value_type in {"i32", "i64"}:
        return 2
    if value_type in {"float", "double"}:
        return 3
    if value_type == "ptr":
        return 4
    raise LlvmLoweringError(f"unsupported boxed value type '{value_type}'")


def _owner_type(function: HFunction) -> str | None:
    if "." not in function.name:
        return None
    return function.name.split(".", 1)[0]


def _layout_ref(layout: _ClassLayout) -> str:
    return f"%{layout.struct_name}"


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
