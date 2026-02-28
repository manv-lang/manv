"""AST -> HLIR lowering with explicit control flow and EH edges.

Why this file exists:
- Encodes execution semantics in a CFG-friendly form that can be interpreted
  directly and later compiled.
- Makes potentially throwing behavior explicit (`invoke`/`raise`) so exception
  handling is testable and backend-independent.
- Preserves stable callsite/attrsite identifiers for future inline caches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import ast
from .diagnostics import Span
from .hlir import HBasicBlock, HFunction, HInstruction, HModule, HTerminator, Provenance, SourceSpan
from .intrinsics import intrinsic_effect_names, resolve_call_alias_name, resolve_intrinsic, resolve_intrinsic_name_from_callee
from .semantics import GpuDecoratorConfig, has_static_method_decorator, normalize_gpu_decorator, normalize_type_name


@dataclass
class _LowerState:
    fn_name: str
    source_name: str
    return_type: str | None
    params: list[dict[str, Any]]
    blocks: dict[str, HBasicBlock] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    current_label: str = "entry"
    temp_counter: int = 0
    label_counter: int = 0
    instr_counter: int = 0
    term_counter: int = 0
    ast_counter: int = 0
    callsite_counter: int = 0
    attrsite_counter: int = 0
    ast_ids: dict[int, str] = field(default_factory=dict)
    loop_targets: list[tuple[str, str]] = field(default_factory=list)
    unwind_targets: list[str] = field(default_factory=list)
    declared_vars: set[str] = field(default_factory=set)
    gpu_functions: dict[str, GpuDecoratorConfig] = field(default_factory=dict)
    static_methods: set[str] = field(default_factory=set)
    known_callables: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._ensure_block("entry")

    def _ensure_block(self, label: str) -> HBasicBlock:
        if label not in self.blocks:
            self.blocks[label] = HBasicBlock(label=label)
            self.order.append(label)
        return self.blocks[label]

    def block(self) -> HBasicBlock:
        return self.blocks[self.current_label]

    def set_block(self, label: str) -> None:
        self._ensure_block(label)
        self.current_label = label

    def new_temp(self) -> str:
        self.temp_counter += 1
        return f"%t{self.temp_counter}"

    def new_label(self, prefix: str) -> str:
        self.label_counter += 1
        return f"{prefix}_{self.label_counter}"

    def _next_instr_id(self) -> str:
        self.instr_counter += 1
        return f"{self.fn_name}.i{self.instr_counter}"

    def _next_term_id(self) -> str:
        self.term_counter += 1
        return f"{self.fn_name}.t{self.term_counter}"

    def next_callsite_id(self) -> str:
        # Stable within one lowering run; intended IC hook.
        self.callsite_counter += 1
        return f"{self.fn_name}.cs{self.callsite_counter}"

    def next_attrsite_id(self) -> str:
        # Stable within one lowering run; intended IC hook.
        self.attrsite_counter += 1
        return f"{self.fn_name}.as{self.attrsite_counter}"

    def _span_to_source(self, span: Span | None) -> SourceSpan | None:
        if span is None:
            return None
        end_line = span.end_line if span.end_line is not None else span.line
        end_col = span.end_column if span.end_column is not None else span.column
        return SourceSpan(
            uri=span.file or self.source_name,
            start_line=span.line,
            start_col=span.column,
            end_line=end_line,
            end_col=end_col,
        )

    def _ast_id(self, node: Any | None) -> str | None:
        if node is None:
            return None
        key = id(node)
        if key not in self.ast_ids:
            self.ast_counter += 1
            self.ast_ids[key] = f"{self.fn_name}.a{self.ast_counter}"
        return self.ast_ids[key]

    def _span_of(self, node: Any | None, span: Span | None) -> Span | None:
        if span is not None:
            return span
        node_span = getattr(node, "span", None)
        if isinstance(node_span, Span):
            return node_span
        return None

    def _provenance(self, instr_id: str, node: Any | None, span: Span | None) -> Provenance:
        source_span = self._span_to_source(self._span_of(node, span))
        return Provenance(primary_span=source_span, ast_id=self._ast_id(node), hlir_id=instr_id)

    def emit(self, instr: HInstruction, node: Any | None = None, span: Span | None = None) -> str | None:
        instr.instr_id = instr.instr_id or self._next_instr_id()
        instr.provenance = instr.provenance or self._provenance(instr.instr_id, node=node, span=span)
        instr.effectful = instr.effectful or bool(instr.effects)
        self.block().instructions.append(instr)
        return instr.dest

    def terminate(
        self,
        op: str,
        args: list[str] | None = None,
        *,
        attrs: dict[str, Any] | None = None,
        node: Any | None = None,
        span: Span | None = None,
    ) -> None:
        term_id = self._next_term_id()
        self.block().terminator = HTerminator(
            op=op,
            args=args or [],
            attrs=attrs or {},
            term_id=term_id,
            provenance=self._provenance(term_id, node=node, span=span),
        )

    def has_terminator(self) -> bool:
        return self.block().terminator is not None


class HLIRLowerer:
    def lower_program(self, program: ast.Program, source_name: str) -> HModule:
        functions: list[HFunction] = []
        gpu_functions = self._collect_gpu_functions(program)
        static_methods = self._collect_static_methods(program)
        known_callables = self._collect_known_callables(program)

        if program.statements:
            top_fn = self._lower_function_decl(
                fn_name="__top_level",
                source_name=source_name,
                params=[],
                return_type=None,
                body=program.statements,
                gpu_functions=gpu_functions,
                static_methods=static_methods,
            )
            functions.append(top_fn)

        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                functions.append(
                    self._lower_function_decl(
                        fn_name=decl.name,
                        source_name=source_name,
                        params=[{"name": p.name, "type": normalize_type_name(p.type_name), "span": p.span} for p in decl.params],
                        return_type=normalize_type_name(decl.return_type),
                        body=decl.body,
                        attrs=_function_attrs(decl),
                        gpu_functions=gpu_functions,
                        static_methods=static_methods,
                        known_callables=known_callables,
                    )
                )
                continue

            if isinstance(decl, ast.TypeDecl):
                for method in decl.methods:
                    functions.append(
                        self._lower_function_decl(
                            fn_name=f"{decl.name}.{method.name}",
                            source_name=source_name,
                            params=[{"name": p.name, "type": normalize_type_name(p.type_name), "span": p.span} for p in method.params],
                            return_type=normalize_type_name(method.return_type),
                            body=method.body,
                            attrs=_function_attrs(method),
                            gpu_functions=gpu_functions,
                            static_methods=static_methods,
                            known_callables=known_callables,
                        )
                    )
                continue

            if isinstance(decl, ast.ImplDecl):
                for method in decl.methods:
                    functions.append(
                        self._lower_function_decl(
                            fn_name=f"{decl.target}.{method.name}",
                            source_name=source_name,
                            params=[{"name": p.name, "type": normalize_type_name(p.type_name), "span": p.span} for p in method.params],
                            return_type=normalize_type_name(method.return_type),
                            body=method.body,
                            attrs=_function_attrs(method),
                            gpu_functions=gpu_functions,
                            static_methods=static_methods,
                            known_callables=known_callables,
                        )
                    )

        return HModule(
            version="0.1",
            source=source_name,
            functions=functions,
            attrs=self._module_attrs(program),
        )

    def _collect_gpu_functions(self, program: ast.Program) -> dict[str, GpuDecoratorConfig]:
        gpu_functions: dict[str, GpuDecoratorConfig] = {}
        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                config = normalize_gpu_decorator(decl)
                if config is not None:
                    gpu_functions[decl.name] = config
            elif isinstance(decl, ast.TypeDecl):
                for method in decl.methods:
                    config = normalize_gpu_decorator(method)
                    if config is not None:
                        gpu_functions[f"{decl.name}.{method.name}"] = config
            elif isinstance(decl, ast.ImplDecl):
                for method in decl.methods:
                    config = normalize_gpu_decorator(method)
                    if config is not None:
                        gpu_functions[f"{decl.target}.{method.name}"] = config
        return gpu_functions

    def _collect_static_methods(self, program: ast.Program) -> set[str]:
        """Collect explicit type-callable methods.

        This set is what lets lowering turn `Math.abs(x)` into a direct call to
        `Math.abs` instead of an instance-style method call that expects a
        receiver. The source language keeps methods instance-oriented by
        default, so only explicit `@static_method` opt-ins land here.
        """

        static_methods: set[str] = set()
        for decl in program.declarations:
            if isinstance(decl, ast.TypeDecl):
                for method in decl.methods:
                    if has_static_method_decorator(method):
                        static_methods.add(f"{decl.name}.{method.name}")
            elif isinstance(decl, ast.ImplDecl):
                for method in decl.methods:
                    if has_static_method_decorator(method):
                        static_methods.add(f"{decl.target}.{method.name}")
        return static_methods

    def _collect_known_callables(self, program: ast.Program) -> set[str]:
        """Collect user-defined callable/type names that outrank builtin aliases.

        Lowering only uses builtin alias lowering as a last resort. This set is
        the static half of that rule: top-level functions and type constructors
        should lower as ordinary named calls even when they reuse builtin names
        like `min`, `max`, or `sum`.
        """

        names: set[str] = set()
        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                names.add(decl.name)
            elif isinstance(decl, ast.TypeDecl):
                names.add(decl.name)
                for method in decl.methods:
                    names.add(f"{decl.name}.{method.name}")
            elif isinstance(decl, ast.ImplDecl):
                for method in decl.methods:
                    names.add(f"{decl.target}.{method.name}")
        return names

    def _module_attrs(self, program: ast.Program) -> dict[str, Any]:
        """Preserve non-executable symbol metadata needed by runtime tooling.

        Why this exists:
        - Docstrings are metadata and must not appear as executable HLIR.
        - Some runtime tooling, such as `help(...)`, still needs access to that
          metadata even when execution happens through HLIR.
        - Keeping this on the module object avoids smuggling documentation
          through fake instructions that would pollute semantic IR.
        """

        function_docs: dict[str, dict[str, Any]] = {}
        type_docs: dict[str, dict[str, Any]] = {}

        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                function_docs[decl.name] = _function_doc_attrs(decl.name, decl)
            elif isinstance(decl, ast.TypeDecl):
                type_docs[decl.name] = {
                    "docstring": decl.docstring,
                    "kind": "class",
                }
                for method in decl.methods:
                    function_docs[f"{decl.name}.{method.name}"] = _function_doc_attrs(f"{decl.name}.{method.name}", method)
            elif isinstance(decl, ast.ImplDecl):
                for method in decl.methods:
                    function_docs[f"{decl.target}.{method.name}"] = _function_doc_attrs(f"{decl.target}.{method.name}", method)

        return {
            "docstring": program.docstring,
            "functions": function_docs,
            "types": type_docs,
        }

    def _lower_function_decl(
        self,
        fn_name: str,
        source_name: str,
        params: list[dict[str, Any]],
        return_type: str | None,
        body: list[Any],
        *,
        attrs: dict[str, Any] | None = None,
        gpu_functions: dict[str, GpuDecoratorConfig] | None = None,
        static_methods: set[str] | None = None,
        known_callables: set[str] | None = None,
    ) -> HFunction:
        state = _LowerState(
            fn_name=fn_name,
            source_name=source_name,
            return_type=return_type,
            params=params,
            gpu_functions=dict(gpu_functions or {}),
            static_methods=set(static_methods or set()),
            known_callables=set(known_callables or set()),
        )

        for index, param in enumerate(params):
            var_name = str(param["name"])
            p_span = param.get("span") if isinstance(param.get("span"), Span) else None
            state.emit(
                HInstruction(op="declare_var", attrs={"name": var_name, "type": param.get("type")}, effects=["writes_memory"]),
                span=p_span,
            )
            temp = state.new_temp()
            state.emit(
                HInstruction(
                    op="load_arg",
                    dest=temp,
                    type_name=param.get("type"),
                    attrs={"index": index, "name": var_name},
                    effects=["reads_memory"],
                ),
                span=p_span,
            )
            state.emit(HInstruction(op="store_var", args=[var_name, temp], effects=["writes_memory"]), span=p_span)
            state.declared_vars.add(var_name)

        self._lower_statements(state, body)

        if not state.has_terminator():
            if return_type is None:
                state.terminate("ret", [])
            else:
                zero = state.new_temp()
                state.emit(HInstruction(op="const", dest=zero, type_name=return_type, attrs={"value": 0}))
                state.terminate("ret", [zero])

        blocks = [state.blocks[label] for label in state.order]
        public_params = [{"name": p.get("name"), "type": p.get("type")} for p in params]
        return HFunction(
            name=fn_name,
            params=public_params,
            return_type=return_type,
            entry="entry",
            blocks=blocks,
            attrs=dict(attrs or {}),
        )

    def _lower_statements(self, state: _LowerState, statements: list[Any]) -> None:
        for stmt in statements:
            if state.has_terminator():
                break
            self._lower_stmt(state, stmt)

    def _lower_stmt(self, state: _LowerState, stmt: Any) -> None:
        if isinstance(stmt, ast.LetStmt):
            self._lower_let(state, stmt)
            return

        if isinstance(stmt, ast.ImportStmt):
            out = state.new_temp()
            state.emit(
                HInstruction(
                    op="import",
                    dest=out,
                    type_name="module",
                    # Preserve level metadata so runtime can resolve
                    # absolute vs package-relative imports.
                    attrs={"module": stmt.module, "alias": stmt.alias, "level": stmt.level},
                    effects=["reads_memory", "writes_memory", "may_throw"],
                ),
                node=stmt,
            )
            bind = stmt.alias or stmt.module.split(".")[-1]
            if bind not in state.declared_vars:
                state.emit(
                    HInstruction(op="declare_var", attrs={"name": bind, "type": "module"}, effects=["writes_memory"]),
                    node=stmt,
                )
                state.declared_vars.add(bind)
            state.emit(HInstruction(op="store_var", args=[bind, out], effects=["writes_memory"]), node=stmt)
            return

        if isinstance(stmt, ast.FromImportStmt):
            out = state.new_temp()
            state.emit(
                HInstruction(
                    op="from_import",
                    dest=out,
                    # `level` is the relative import depth (`.`, `..`, ...).
                    attrs={"module": stmt.module, "name": stmt.name, "alias": stmt.alias, "level": stmt.level},
                    effects=["reads_memory", "writes_memory", "may_throw"],
                ),
                node=stmt,
            )
            bind = stmt.alias or stmt.name
            if bind not in state.declared_vars:
                state.emit(
                    HInstruction(op="declare_var", attrs={"name": bind, "type": None}, effects=["writes_memory"]),
                    node=stmt,
                )
                state.declared_vars.add(bind)
            state.emit(HInstruction(op="store_var", args=[bind, out], effects=["writes_memory"]), node=stmt)
            return

        if isinstance(stmt, ast.AssignStmt):
            value = self._lower_expr(state, stmt.value)
            state.emit(HInstruction(op="store_var", args=[stmt.name, value], effects=["writes_memory"]), node=stmt)
            return

        if isinstance(stmt, ast.SetAttrStmt):
            target = self._lower_expr(state, stmt.target)
            value = self._lower_expr(state, stmt.value)
            state.emit(
                HInstruction(
                    op="set_attr",
                    args=[target, value],
                    attrs={"attr": stmt.attr},
                    effects=["writes_memory", "dynamic_dispatch", "may_throw"],
                ),
                node=stmt,
            )
            return

        if isinstance(stmt, ast.SetIndexStmt):
            target = self._lower_expr(state, stmt.target)
            index = self._lower_expr(state, stmt.index)
            value = self._lower_expr(state, stmt.value)
            state.emit(HInstruction(op="set_index", args=[target, index, value], effects=["writes_memory", "may_throw"]), node=stmt)
            return

        if isinstance(stmt, ast.ExprStmt):
            self._lower_expr(state, stmt.expr)
            return

        if isinstance(stmt, ast.ReturnStmt):
            if stmt.value is None:
                state.terminate("ret", [], node=stmt)
            else:
                value = self._lower_expr(state, stmt.value)
                state.terminate("ret", [value], node=stmt)
            return

        if isinstance(stmt, ast.RaiseStmt):
            if state.unwind_targets:
                if stmt.value is None:
                    current_exc = state.new_temp()
                    state.emit(HInstruction(op="load_exception", dest=current_exc, effects=["reads_memory"]), node=stmt)
                    state.emit(HInstruction(op="set_exception", args=[current_exc], effects=["writes_memory"]), node=stmt)
                else:
                    value = self._lower_expr(state, stmt.value)
                    state.emit(HInstruction(op="set_exception", args=[value], effects=["writes_memory"]), node=stmt)
                state.terminate("br", [state.unwind_targets[-1]], node=stmt)
            else:
                if stmt.value is None:
                    state.terminate("raise", ["__reraise__"], node=stmt)
                else:
                    value = self._lower_expr(state, stmt.value)
                    state.terminate("raise", [value], node=stmt)
            return

        if isinstance(stmt, ast.SyscallStmt):
            target = self._lower_expr(state, stmt.target)
            arg_values = [self._lower_expr(state, arg) for arg in stmt.args]
            arg_array = state.new_temp()
            state.emit(HInstruction(op="array", dest=arg_array, args=arg_values, effects=["allocates"]), node=stmt)
            state.emit(
                HInstruction(
                    op="intrinsic_call",
                    args=[target, arg_array],
                    attrs={"name": "syscall_invoke", "pure_for_kernel": False},
                    effects=["reads_memory", "writes_memory", "dynamic_dispatch", "may_throw"],
                ),
                node=stmt,
            )
            return

        if isinstance(stmt, ast.TryStmt):
            self._lower_try(state, stmt)
            return

        if isinstance(stmt, ast.IfStmt):
            self._lower_if(state, stmt)
            return

        if isinstance(stmt, ast.WhileStmt):
            self._lower_while(state, stmt)
            return

        if isinstance(stmt, ast.ForStmt):
            self._lower_for(state, stmt)
            return

        if isinstance(stmt, ast.BreakStmt):
            if state.loop_targets:
                break_label, _ = state.loop_targets[-1]
                state.terminate("br", [break_label], node=stmt)
            else:
                state.terminate("unreachable", [], node=stmt)
            return

        if isinstance(stmt, ast.ContinueStmt):
            if state.loop_targets:
                _, continue_label = state.loop_targets[-1]
                state.terminate("br", [continue_label], node=stmt)
            else:
                state.terminate("unreachable", [], node=stmt)
            return

        state.emit(HInstruction(op="unsupported_stmt", attrs={"kind": type(stmt).__name__}, effects=["may_throw"]), node=stmt)

    def _lower_let(self, state: _LowerState, stmt: ast.LetStmt) -> None:
        if stmt.name not in state.declared_vars:
            state.emit(
                HInstruction(op="declare_var", attrs={"name": stmt.name, "type": stmt.type_name}, effects=["writes_memory"]),
                node=stmt,
            )
            state.declared_vars.add(stmt.name)

        if stmt.array_size is not None:
            size = self._lower_expr(state, stmt.array_size)
            if stmt.value is None:
                out = state.new_temp()
                state.emit(HInstruction(op="alloc_array", dest=out, type_name="array", args=[size], effects=["allocates"]), node=stmt)
                state.emit(HInstruction(op="store_var", args=[stmt.name, out], effects=["writes_memory"]), node=stmt)
                return
            seed = self._lower_expr(state, stmt.value)
            out = state.new_temp()
            state.emit(HInstruction(op="array_init_sized", dest=out, type_name="array", args=[seed, size], effects=["allocates"]), node=stmt)
            state.emit(HInstruction(op="store_var", args=[stmt.name, out], effects=["writes_memory"]), node=stmt)
            return

        if stmt.value is None:
            value_temp = state.new_temp()
            state.emit(HInstruction(op="const", dest=value_temp, type_name=stmt.type_name, attrs={"value": None}), node=stmt)
        else:
            value_temp = self._lower_expr(state, stmt.value)
        state.emit(HInstruction(op="store_var", args=[stmt.name, value_temp], effects=["writes_memory"]), node=stmt)

    def _lower_if(self, state: _LowerState, stmt: ast.IfStmt) -> None:
        cond = self._lower_expr(state, stmt.condition)
        then_label = state.new_label("if_then")
        else_label = state.new_label("if_else")
        merge_label = state.new_label("if_merge")

        state.terminate("cbr", [cond, then_label, else_label], node=stmt)

        state.set_block(then_label)
        self._lower_statements(state, stmt.then_body)
        if not state.has_terminator():
            state.terminate("br", [merge_label], node=stmt)

        state.set_block(else_label)
        self._lower_statements(state, stmt.else_body)
        if not state.has_terminator():
            state.terminate("br", [merge_label], node=stmt)

        state.set_block(merge_label)

    def _lower_while(self, state: _LowerState, stmt: ast.WhileStmt) -> None:
        cond_label = state.new_label("while_cond")
        body_label = state.new_label("while_body")
        exit_label = state.new_label("while_exit")

        state.terminate("br", [cond_label], node=stmt)

        state.set_block(cond_label)
        cond = self._lower_expr(state, stmt.condition)
        state.terminate("cbr", [cond, body_label, exit_label], node=stmt)

        state.set_block(body_label)
        state.loop_targets.append((exit_label, cond_label))
        self._lower_statements(state, stmt.body)
        state.loop_targets.pop()
        if not state.has_terminator():
            state.terminate("br", [cond_label], node=stmt)

        state.set_block(exit_label)

    def _lower_for(self, state: _LowerState, stmt: ast.ForStmt) -> None:
        # GPU eligibility depends on seeing a canonical counted-loop shape.
        # Lowering therefore normalizes source `for i in a..b:` into explicit
        # init/compare/increment blocks while preserving the original source
        # spans on the generated operations.
        if not isinstance(stmt.iterable, ast.RangeExpr):
            state.emit(HInstruction(op="unsupported_stmt", attrs={"kind": "ForStmt"}, effects=["may_throw"]), node=stmt)
            return

        if stmt.var_name not in state.declared_vars:
            state.emit(
                HInstruction(op="declare_var", attrs={"name": stmt.var_name, "type": "i32"}, effects=["writes_memory"]),
                node=stmt,
            )
            state.declared_vars.add(stmt.var_name)

        start_value = self._lower_expr(state, stmt.iterable.start)
        state.emit(HInstruction(op="store_var", args=[stmt.var_name, start_value], effects=["writes_memory"]), node=stmt)

        cond_label = state.new_label("for_cond")
        body_label = state.new_label("for_body")
        incr_label = state.new_label("for_incr")
        exit_label = state.new_label("for_exit")

        state.terminate("br", [cond_label], node=stmt)

        state.set_block(cond_label)
        loop_index = state.new_temp()
        state.emit(HInstruction(op="load_var", dest=loop_index, args=[stmt.var_name], effects=["reads_memory"]), node=stmt)
        stop_value = self._lower_expr(state, stmt.iterable.stop)
        compare = state.new_temp()
        state.emit(HInstruction(op="binop", dest=compare, args=[loop_index, stop_value], attrs={"op": "<"}), node=stmt)
        state.terminate("cbr", [compare, body_label, exit_label], node=stmt)

        state.set_block(body_label)
        state.loop_targets.append((exit_label, incr_label))
        self._lower_statements(state, stmt.body)
        state.loop_targets.pop()
        if not state.has_terminator():
            state.terminate("br", [incr_label], node=stmt)

        state.set_block(incr_label)
        current_value = state.new_temp()
        state.emit(HInstruction(op="load_var", dest=current_value, args=[stmt.var_name], effects=["reads_memory"]), node=stmt)
        one_value = state.new_temp()
        state.emit(HInstruction(op="const", dest=one_value, type_name="i32", attrs={"value": 1}), node=stmt)
        next_value = state.new_temp()
        state.emit(HInstruction(op="binop", dest=next_value, args=[current_value, one_value], attrs={"op": "+"}), node=stmt)
        state.emit(HInstruction(op="store_var", args=[stmt.var_name, next_value], effects=["writes_memory"]), node=stmt)
        state.terminate("br", [cond_label], node=stmt)

        state.set_block(exit_label)

    def _lower_try(self, state: _LowerState, stmt: ast.TryStmt) -> None:
        # Canonical EH lowering strategy:
        # - Execute try body with explicit unwind target.
        # - Dispatch handlers in source order.
        # - Funnel all exits through finally when present.
        # - Preserve pending exception across finally for rethrow semantics.
        try_body_label = state.new_label("try_body")
        except_dispatch_label = state.new_label("except_dispatch")
        else_label = state.new_label("try_else") if stmt.else_body else ""
        finally_label = state.new_label("try_finally") if stmt.finally_body else ""
        exit_label = state.new_label("try_exit")

        pending_var = ""
        if stmt.finally_body:
            pending_var = f"__eh_pending_{state.new_label('slot')}"
            if pending_var not in state.declared_vars:
                state.emit(
                    HInstruction(op="declare_var", attrs={"name": pending_var, "type": "dynamic"}, effects=["writes_memory"]),
                    node=stmt,
                )
                state.declared_vars.add(pending_var)
            none_temp = state.new_temp()
            state.emit(HInstruction(op="const", dest=none_temp, type_name="none", attrs={"value": None}), node=stmt)
            state.emit(HInstruction(op="store_var", args=[pending_var, none_temp], effects=["writes_memory"]), node=stmt)

        state.terminate("br", [try_body_label], node=stmt)

        state.set_block(try_body_label)
        state.unwind_targets.append(except_dispatch_label)
        self._lower_statements(state, stmt.try_body)
        state.unwind_targets.pop()
        if not state.has_terminator():
            if stmt.else_body:
                state.terminate("br", [else_label], node=stmt)
            elif stmt.finally_body:
                state.terminate("br", [finally_label], node=stmt)
            else:
                state.terminate("br", [exit_label], node=stmt)

        state.set_block(except_dispatch_label)
        exc_temp = state.new_temp()
        state.emit(HInstruction(op="load_exception", dest=exc_temp, effects=["reads_memory"]), node=stmt)

        if not stmt.except_clauses:
            if stmt.finally_body:
                state.emit(HInstruction(op="store_var", args=[pending_var, exc_temp], effects=["writes_memory"]), node=stmt)
                state.terminate("br", [finally_label], node=stmt)
            else:
                state.terminate("raise", [exc_temp], node=stmt)
        else:
            no_match_label = state.new_label("except_no_match")
            check_label = state.current_label
            for index, clause in enumerate(stmt.except_clauses):
                clause_label = state.new_label(f"except_{index}")
                next_label = no_match_label if index == len(stmt.except_clauses) - 1 else state.new_label(f"except_next_{index}")

                state.set_block(check_label)
                match_temp = state.new_temp()
                state.emit(
                    HInstruction(op="exc_match", dest=match_temp, args=[exc_temp], attrs={"type": clause.type_name}, effects=["pure"]),
                    node=clause,
                )
                state.terminate("cbr", [match_temp, clause_label, next_label], node=clause)

                state.set_block(clause_label)
                if clause.bind_name:
                    if clause.bind_name not in state.declared_vars:
                        state.emit(
                            HInstruction(op="declare_var", attrs={"name": clause.bind_name, "type": clause.type_name}, effects=["writes_memory"]),
                            node=clause,
                        )
                        state.declared_vars.add(clause.bind_name)
                    state.emit(HInstruction(op="store_var", args=[clause.bind_name, exc_temp], effects=["writes_memory"]), node=clause)
                self._lower_statements(state, clause.body)
                if not state.has_terminator():
                    if stmt.finally_body:
                        none_temp = state.new_temp()
                        state.emit(HInstruction(op="const", dest=none_temp, type_name="none", attrs={"value": None}), node=clause)
                        state.emit(HInstruction(op="store_var", args=[pending_var, none_temp], effects=["writes_memory"]), node=clause)
                        state.terminate("br", [finally_label], node=clause)
                    else:
                        state.terminate("br", [exit_label], node=clause)
                check_label = next_label

            state.set_block(no_match_label)
            if stmt.finally_body:
                state.emit(HInstruction(op="store_var", args=[pending_var, exc_temp], effects=["writes_memory"]), node=stmt)
                state.terminate("br", [finally_label], node=stmt)
            else:
                state.terminate("raise", [exc_temp], node=stmt)

        if stmt.else_body:
            state.set_block(else_label)
            self._lower_statements(state, stmt.else_body)
            if not state.has_terminator():
                if stmt.finally_body:
                    state.terminate("br", [finally_label], node=stmt)
                else:
                    state.terminate("br", [exit_label], node=stmt)

        if stmt.finally_body:
            state.set_block(finally_label)
            state.emit(HInstruction(op="finally_enter", effects=["writes_memory"]), node=stmt)
            self._lower_statements(state, stmt.finally_body)
            if not state.has_terminator():
                state.emit(HInstruction(op="finally_exit", effects=["reads_memory"]), node=stmt)
                pending_temp = state.new_temp()
                state.emit(HInstruction(op="load_var", dest=pending_temp, args=[pending_var], effects=["reads_memory"]), node=stmt)
                none_temp = state.new_temp()
                state.emit(HInstruction(op="const", dest=none_temp, type_name="none", attrs={"value": None}), node=stmt)
                has_pending = state.new_temp()
                state.emit(HInstruction(op="binop", dest=has_pending, args=[pending_temp, none_temp], attrs={"op": "!="}), node=stmt)
                finally_raise_label = state.new_label("finally_raise")
                finally_continue_label = state.new_label("finally_continue")
                state.terminate("cbr", [has_pending, finally_raise_label, finally_continue_label], node=stmt)

                state.set_block(finally_raise_label)
                state.terminate("raise", [pending_temp], node=stmt)

                state.set_block(finally_continue_label)
                state.terminate("br", [exit_label], node=stmt)

        state.set_block(exit_label)

    def _lower_expr(self, state: _LowerState, expr: Any) -> str:
        if isinstance(expr, ast.LiteralExpr):
            out = state.new_temp()
            state.emit(HInstruction(op="const", dest=out, type_name=expr.literal_type, attrs={"value": expr.value}), node=expr)
            return out

        if isinstance(expr, ast.IdentifierExpr):
            out = state.new_temp()
            state.emit(HInstruction(op="load_var", dest=out, args=[expr.name], effects=["reads_memory"]), node=expr)
            return out

        if isinstance(expr, ast.UnaryExpr):
            inner = self._lower_expr(state, expr.expr)
            out = state.new_temp()
            state.emit(HInstruction(op="unary", dest=out, args=[inner], attrs={"op": expr.op}), node=expr)
            return out

        if isinstance(expr, ast.BinaryExpr):
            left = self._lower_expr(state, expr.left)
            right = self._lower_expr(state, expr.right)
            out = state.new_temp()
            state.emit(HInstruction(op="binop", dest=out, args=[left, right], attrs={"op": expr.op}), node=expr)
            return out

        if isinstance(expr, ast.RangeExpr):
            out = state.new_temp()
            state.emit(HInstruction(op="const", dest=out, type_name="range", attrs={"value": None}), node=expr)
            return out

        if isinstance(expr, ast.CallExpr):
            arg_values = [self._lower_expr(state, arg) for arg in expr.args]
            callsite_id = state.next_callsite_id()
            intrinsic_name = resolve_intrinsic_name_from_callee(expr.callee) or self._resolve_unshadowed_call_alias(
                state,
                expr.callee,
            )
            if intrinsic_name is not None:
                spec = resolve_intrinsic(intrinsic_name)
                effects = ["may_throw"] if spec is None else intrinsic_effect_names(spec)
                ret_type = "dynamic"
                if spec is not None and isinstance(spec.return_type, str):
                    ret_type = spec.return_type
                out = state.new_temp()
                intrinsic_may_throw = True if spec is None else bool(spec.may_throw)
                if state.unwind_targets and intrinsic_may_throw:
                    normal_label = state.new_label("invoke_ok")
                    state.terminate(
                        "invoke",
                        [*arg_values, normal_label, state.unwind_targets[-1]],
                        attrs={
                            "kind": "intrinsic",
                            "name": intrinsic_name,
                            "signature_id": intrinsic_name,
                            "callsite_id": callsite_id,
                            "dest": out,
                            "type": ret_type,
                        },
                        node=expr,
                    )
                    state.set_block(normal_label)
                    return out
                state.emit(
                    HInstruction(
                        op="intrinsic_call",
                        dest=out,
                        type_name=ret_type,
                        args=arg_values,
                        attrs={
                            "name": intrinsic_name,
                            "signature_id": intrinsic_name,
                            "callsite_id": callsite_id,
                            "pure_for_kernel": bool(spec.pure_for_kernel) if spec is not None else False,
                        },
                        effects=effects,
                    ),
                    node=expr,
                )
                return out

            if isinstance(expr.callee, ast.AttributeExpr):
                method_name = expr.callee.attr
                static_callee = self._qualified_static_callee(expr.callee, state.static_methods)
                if static_callee is not None:
                    return self._lower_named_call(state, expr, static_callee, arg_values, callsite_id)

                receiver = self._lower_expr(state, expr.callee.value)
                out = state.new_temp()
                if state.unwind_targets:
                    normal_label = state.new_label("invoke_ok")
                    state.terminate(
                        "invoke",
                        [receiver, *arg_values, normal_label, state.unwind_targets[-1]],
                        attrs={"kind": "method_call", "method": method_name, "callsite_id": callsite_id, "dest": out},
                        node=expr,
                    )
                    state.set_block(normal_label)
                    return out
                state.emit(
                    HInstruction(
                        op="method_call",
                        dest=out,
                        args=[receiver, *arg_values],
                        attrs={"method": method_name, "callsite_id": callsite_id},
                        effects=["dynamic_dispatch", "may_throw"],
                    ),
                    node=expr,
                )
                return out

            callee_name = "<dynamic>"
            if isinstance(expr.callee, ast.IdentifierExpr):
                callee_name = expr.callee.name
            return self._lower_named_call(state, expr, callee_name, arg_values, callsite_id)

        if isinstance(expr, ast.SyscallExpr):
            target = self._lower_expr(state, expr.target)
            arg_values = [self._lower_expr(state, arg) for arg in expr.args]
            args_array = state.new_temp()
            state.emit(HInstruction(op="array", dest=args_array, args=arg_values, effects=["allocates"]), node=expr)
            out = state.new_temp()
            if state.unwind_targets:
                normal_label = state.new_label("invoke_ok")
                state.terminate(
                    "invoke",
                    [target, args_array, normal_label, state.unwind_targets[-1]],
                    attrs={"kind": "intrinsic", "name": "syscall_invoke", "signature_id": "syscall_invoke", "dest": out, "type": "map"},
                    node=expr,
                )
                state.set_block(normal_label)
                return out
            state.emit(
                HInstruction(
                    op="intrinsic_call",
                    dest=out,
                    type_name="map",
                    args=[target, args_array],
                    attrs={"name": "syscall_invoke", "signature_id": "syscall_invoke", "pure_for_kernel": False},
                    effects=["reads_memory", "writes_memory", "dynamic_dispatch", "may_throw"],
                ),
                node=expr,
            )
            return out

        if isinstance(expr, ast.ArrayExpr):
            elems = [self._lower_expr(state, e) for e in expr.elements]
            out = state.new_temp()
            state.emit(HInstruction(op="array", dest=out, args=elems, effects=["allocates"]), node=expr)
            return out

        if isinstance(expr, ast.MapExpr):
            flat: list[str] = []
            for k, v in expr.entries:
                flat.append(self._lower_expr(state, k))
                flat.append(self._lower_expr(state, v))
            out = state.new_temp()
            state.emit(HInstruction(op="map", dest=out, args=flat, effects=["allocates"]), node=expr)
            return out

        if isinstance(expr, ast.IndexExpr):
            base = self._lower_expr(state, expr.value)
            index = self._lower_expr(state, expr.index)
            out = state.new_temp()
            state.emit(HInstruction(op="index", dest=out, args=[base, index], effects=["reads_memory", "may_throw"]), node=expr)
            return out

        if isinstance(expr, ast.AttributeExpr):
            base = self._lower_expr(state, expr.value)
            out = state.new_temp()
            state.emit(
                HInstruction(
                    op="attr",
                    dest=out,
                    args=[base],
                    attrs={"attr": expr.attr, "attrsite_id": state.next_attrsite_id()},
                    effects=["dynamic_dispatch", "may_throw"],
                ),
                node=expr,
            )
            return out

        out = state.new_temp()
        state.emit(HInstruction(op="const", dest=out, attrs={"value": None}), node=expr)
        return out

    def _qualified_static_callee(self, attr_expr: ast.AttributeExpr, static_methods: set[str]) -> str | None:
        owner_name: str | None = None
        if isinstance(attr_expr.value, ast.IdentifierExpr):
            owner_name = attr_expr.value.name
        elif isinstance(attr_expr.value, ast.AttributeExpr):
            owner_name = attr_expr.value.attr

        if owner_name is None:
            return None

        candidate = f"{owner_name}.{attr_expr.attr}"
        if candidate in static_methods:
            return candidate
        return None

    def _lower_named_call(
        self,
        state: _LowerState,
        expr: ast.CallExpr,
        callee_name: str,
        arg_values: list[str],
        callsite_id: str,
    ) -> str:
        gpu_config = state.gpu_functions.get(callee_name)
        out = state.new_temp()
        if gpu_config is not None:
            # HLIR is the semantic authority for GPU dispatch decisions.
            # Downstream passes must consult this explicit op instead of
            # inferring GPU intent from backend or naming conventions.
            if state.unwind_targets:
                normal_label = state.new_label("invoke_ok")
                state.terminate(
                    "invoke",
                    [*arg_values, normal_label, state.unwind_targets[-1]],
                    attrs={
                        "kind": "gpu_call",
                        "callee": callee_name,
                        "callsite_id": callsite_id,
                        "dest": out,
                        "policy": "required" if gpu_config.required else "best_effort",
                        "mode": gpu_config.mode,
                    },
                    node=expr,
                )
                state.set_block(normal_label)
                return out
            state.emit(
                HInstruction(
                    op="gpu_call",
                    dest=out,
                    args=arg_values,
                    attrs={
                        "callee": callee_name,
                        "callsite_id": callsite_id,
                        "policy": "required" if gpu_config.required else "best_effort",
                        "mode": gpu_config.mode,
                    },
                    effects=["reads_memory", "writes_memory", "may_throw"],
                ),
                node=expr,
            )
            return out
        if state.unwind_targets:
            normal_label = state.new_label("invoke_ok")
            state.terminate(
                "invoke",
                [*arg_values, normal_label, state.unwind_targets[-1]],
                attrs={"kind": "call", "callee": callee_name, "callsite_id": callsite_id, "dest": out},
                node=expr,
            )
            state.set_block(normal_label)
            return out
        state.emit(
            HInstruction(
                op="call",
                dest=out,
                args=arg_values,
                attrs={"callee": callee_name, "callsite_id": callsite_id},
                effects=["dynamic_dispatch", "may_throw"],
            ),
            node=expr,
        )
        return out

    def _resolve_unshadowed_call_alias(self, state: _LowerState, callee: Any) -> str | None:
        """Return a builtin alias only when lowering cannot see a source binding.

        This mirrors semantic and interpreter precedence rules so HLIR remains
        authoritative: user-defined functions, imports, locals, and known type
        constructors win before builtin call aliases are considered.
        """

        if not isinstance(callee, ast.IdentifierExpr):
            return None
        if callee.name in state.declared_vars:
            return None
        if callee.name in state.known_callables:
            return None
        return resolve_call_alias_name(callee)


def lower_ast_to_hlir(program: ast.Program, source_name: str) -> HModule:
    return HLIRLowerer().lower_program(program, source_name)


def _function_attrs(decl: ast.FnDecl) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    gpu = normalize_gpu_decorator(decl)
    if gpu is not None:
        attrs["gpu"] = {
            "required": gpu.required,
            "mode": gpu.mode,
        }
    if has_static_method_decorator(decl):
        attrs["static_method"] = True
    return attrs


def _function_doc_attrs(display_name: str, decl: ast.FnDecl) -> dict[str, Any]:
    return {
        "docstring": decl.docstring,
        "signature": _function_signature_text(display_name, decl),
    }


def _function_signature_text(display_name: str, decl: ast.FnDecl) -> str:
    args = ", ".join(f"{p.name}: {normalize_type_name(p.type_name) or 'any'}" for p in decl.params)
    ret = normalize_type_name(decl.return_type) or "none"
    prefixes: list[str] = []
    if has_static_method_decorator(decl):
        prefixes.append("@static_method")
    gpu = normalize_gpu_decorator(decl)
    if gpu is not None:
        required = "true" if gpu.required else "false"
        prefixes.append(f'@gpu(required={required}, mode="{gpu.mode}")')
    prefix = ""
    if prefixes:
        prefix = " ".join(prefixes) + " "
    return f"{prefix}fn {display_name}({args}) -> {ret}"
