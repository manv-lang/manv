from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import ast
from .diagnostics import Span
from .hlir import HBasicBlock, HFunction, HInstruction, HModule, HTerminator, Provenance, SourceSpan


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
    ast_ids: dict[int, str] = field(default_factory=dict)
    loop_targets: list[tuple[str, str]] = field(default_factory=list)
    declared_vars: set[str] = field(default_factory=set)

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

    def terminate(self, op: str, args: list[str] | None = None, node: Any | None = None, span: Span | None = None) -> None:
        term_id = self._next_term_id()
        self.block().terminator = HTerminator(
            op=op,
            args=args or [],
            term_id=term_id,
            provenance=self._provenance(term_id, node=node, span=span),
        )

    def has_terminator(self) -> bool:
        return self.block().terminator is not None


class HLIRLowerer:
    def lower_program(self, program: ast.Program, source_name: str) -> HModule:
        functions: list[HFunction] = []

        if program.statements:
            top_fn = self._lower_function_decl(
                fn_name="__top_level",
                source_name=source_name,
                params=[],
                return_type=None,
                body=program.statements,
            )
            functions.append(top_fn)

        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                functions.append(
                    self._lower_function_decl(
                        fn_name=decl.name,
                        source_name=source_name,
                        params=[{"name": p.name, "type": p.type_name, "span": p.span} for p in decl.params],
                        return_type=decl.return_type,
                        body=decl.body,
                    )
                )
                continue

            if isinstance(decl, ast.TypeDecl):
                for method in decl.methods:
                    functions.append(
                        self._lower_function_decl(
                            fn_name=f"{decl.name}.{method.name}",
                            source_name=source_name,
                            params=[{"name": p.name, "type": p.type_name, "span": p.span} for p in method.params],
                            return_type=method.return_type,
                            body=method.body,
                        )
                    )
                continue

            if isinstance(decl, ast.ImplDecl):
                for method in decl.methods:
                    functions.append(
                        self._lower_function_decl(
                            fn_name=f"{decl.target}.{method.name}",
                            source_name=source_name,
                            params=[{"name": p.name, "type": p.type_name, "span": p.span} for p in method.params],
                            return_type=method.return_type,
                            body=method.body,
                        )
                    )

        return HModule(version="0.1", source=source_name, functions=functions)

    def _lower_function_decl(
        self,
        fn_name: str,
        source_name: str,
        params: list[dict[str, Any]],
        return_type: str | None,
        body: list[Any],
    ) -> HFunction:
        state = _LowerState(fn_name=fn_name, source_name=source_name, return_type=return_type, params=params)

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
        return HFunction(name=fn_name, params=public_params, return_type=return_type, entry="entry", blocks=blocks)

    def _lower_statements(self, state: _LowerState, statements: list[Any]) -> None:
        for stmt in statements:
            if state.has_terminator():
                break
            self._lower_stmt(state, stmt)

    def _lower_stmt(self, state: _LowerState, stmt: Any) -> None:
        if isinstance(stmt, ast.LetStmt):
            self._lower_let(state, stmt)
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
            if stmt.value is None:
                state.emit(HInstruction(op="raise", effects=["may_throw"], attrs={"reraise": True}), node=stmt)
            else:
                value = self._lower_expr(state, stmt.value)
                state.emit(HInstruction(op="raise", args=[value], effects=["may_throw"]), node=stmt)
            state.terminate("unreachable", [], node=stmt)
            return

        if isinstance(stmt, ast.TryStmt):
            # v1 lowering keeps explicit marker + flattened bodies for artifact visibility.
            state.emit(
                HInstruction(
                    op="try_region",
                    attrs={
                        "except": [{"type": c.type_name, "bind": c.bind_name} for c in stmt.except_clauses],
                        "else_count": len(stmt.else_body),
                        "finally_count": len(stmt.finally_body),
                    },
                    effects=["may_throw", "writes_memory"],
                ),
                node=stmt,
            )
            for inner in stmt.try_body:
                if state.has_terminator():
                    break
                self._lower_stmt(state, inner)
            for clause in stmt.except_clauses:
                for inner in clause.body:
                    if state.has_terminator():
                        break
                    self._lower_stmt(state, inner)
            for inner in stmt.else_body:
                if state.has_terminator():
                    break
                self._lower_stmt(state, inner)
            for inner in stmt.finally_body:
                if state.has_terminator():
                    break
                self._lower_stmt(state, inner)
            return

        if isinstance(stmt, ast.IfStmt):
            self._lower_if(state, stmt)
            return

        if isinstance(stmt, ast.WhileStmt):
            self._lower_while(state, stmt)
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

        if isinstance(expr, ast.CallExpr):
            arg_values = [self._lower_expr(state, arg) for arg in expr.args]
            callee_name = "<dynamic>"
            if isinstance(expr.callee, ast.IdentifierExpr):
                callee_name = expr.callee.name
            out = state.new_temp()
            state.emit(
                HInstruction(
                    op="call",
                    dest=out,
                    args=arg_values,
                    attrs={"callee": callee_name},
                    effects=["dynamic_dispatch", "may_throw"],
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
                    attrs={"attr": expr.attr},
                    effects=["dynamic_dispatch", "may_throw"],
                ),
                node=expr,
            )
            return out

        out = state.new_temp()
        state.emit(HInstruction(op="const", dest=out, attrs={"value": None}), node=expr)
        return out


def lower_ast_to_hlir(program: ast.Program, source_name: str) -> HModule:
    return HLIRLowerer().lower_program(program, source_name)

