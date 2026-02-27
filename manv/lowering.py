from __future__ import annotations

from typing import Any

from . import ast
from .hir import HIRFunction, HIRModule, HIRStatement
from .intrinsics import resolve_call_alias_name, resolve_intrinsic_name_from_callee


def lower_ast_to_hir(program: ast.Program, source_name: str) -> HIRModule:
    functions: list[HIRFunction] = []
    top_level: list[HIRStatement] = []
    stubs: list[dict[str, Any]] = []

    for decl in program.declarations:
        if isinstance(decl, ast.FnDecl):
            functions.append(_lower_fn_decl(decl, name_override=None))
            continue

        if isinstance(decl, ast.TypeDecl):
            stubs.append(
                {
                    "kind": "type",
                    "name": decl.name,
                    "base": decl.base_name,
                    "methods": [m.name for m in decl.methods],
                }
            )
            for method in decl.methods:
                functions.append(_lower_fn_decl(method, name_override=f"{decl.name}.{method.name}"))
            continue

        if isinstance(decl, ast.ImplDecl):
            stubs.append(
                {
                    "kind": "impl",
                    "target": decl.target,
                    "methods": [m.name for m in decl.methods],
                }
            )
            for method in decl.methods:
                functions.append(_lower_fn_decl(method, name_override=f"{decl.target}.{method.name}"))
            continue

        if isinstance(decl, ast.MacroDeclStub):
            stubs.append({"kind": "macro", "name": decl.name, "params": decl.params, "body": decl.body})

    for stmt in program.statements:
        top_level.append(_stmt_to_hir(stmt))

    return HIRModule(
        version="0.1",
        source=source_name,
        functions=functions,
        top_level=top_level,
        stubs=stubs,
    )


def _lower_fn_decl(decl: ast.FnDecl, name_override: str | None) -> HIRFunction:
    return HIRFunction(
        name=name_override or decl.name,
        params=[{"name": p.name, "type": p.type_name} for p in decl.params],
        return_type=decl.return_type,
        body=[_stmt_to_hir(s) for s in decl.body],
    )


def _stmt_to_hir(stmt: object) -> HIRStatement:
    if isinstance(stmt, ast.LetStmt):
        return HIRStatement(
            kind="let",
            attrs={
                "name": stmt.name,
                "type": stmt.type_name,
                "value": _expr_to_hir(stmt.value),
                "array_size": _expr_to_hir(stmt.array_size),
            },
        )
    if isinstance(stmt, ast.AssignStmt):
        return HIRStatement(kind="assign", attrs={"name": stmt.name, "value": _expr_to_hir(stmt.value)})
    if isinstance(stmt, ast.SetAttrStmt):
        return HIRStatement(
            kind="set_attr",
            attrs={"target": _expr_to_hir(stmt.target), "attr": stmt.attr, "value": _expr_to_hir(stmt.value)},
        )
    if isinstance(stmt, ast.SetIndexStmt):
        return HIRStatement(
            kind="set_index",
            attrs={
                "target": _expr_to_hir(stmt.target),
                "index": _expr_to_hir(stmt.index),
                "value": _expr_to_hir(stmt.value),
            },
        )
    if isinstance(stmt, ast.ReturnStmt):
        return HIRStatement(kind="return", attrs={"value": _expr_to_hir(stmt.value)})
    if isinstance(stmt, ast.RaiseStmt):
        return HIRStatement(kind="raise", attrs={"value": _expr_to_hir(stmt.value)})
    if isinstance(stmt, ast.SyscallStmt):
        return HIRStatement(
            kind="syscall",
            attrs={"target": _expr_to_hir(stmt.target), "args": [_expr_to_hir(a) for a in stmt.args]},
        )
    if isinstance(stmt, ast.TryStmt):
        return HIRStatement(
            kind="try",
            attrs={
                "try": [_stmt_to_hir(s).__dict__ for s in stmt.try_body],
                "except": [
                    {
                        "type": clause.type_name,
                        "bind": clause.bind_name,
                        "body": [_stmt_to_hir(s).__dict__ for s in clause.body],
                    }
                    for clause in stmt.except_clauses
                ],
                "else": [_stmt_to_hir(s).__dict__ for s in stmt.else_body],
                "finally": [_stmt_to_hir(s).__dict__ for s in stmt.finally_body],
            },
        )
    if isinstance(stmt, ast.IfStmt):
        return HIRStatement(
            kind="if",
            attrs={
                "condition": _expr_to_hir(stmt.condition),
                "then": [_stmt_to_hir(s).__dict__ for s in stmt.then_body],
                "else": [_stmt_to_hir(s).__dict__ for s in stmt.else_body],
            },
        )
    if isinstance(stmt, ast.WhileStmt):
        return HIRStatement(
            kind="while",
            attrs={
                "condition": _expr_to_hir(stmt.condition),
                "body": [_stmt_to_hir(s).__dict__ for s in stmt.body],
            },
        )
    if isinstance(stmt, ast.ExprStmt):
        return HIRStatement(kind="expr", attrs={"expr": _expr_to_hir(stmt.expr)})
    if isinstance(stmt, ast.BreakStmt):
        return HIRStatement(kind="break", attrs={})
    if isinstance(stmt, ast.ContinueStmt):
        return HIRStatement(kind="continue", attrs={})
    if isinstance(stmt, ast.UnsupportedStmt):
        return HIRStatement(kind="stub_stmt", attrs={"feature": stmt.feature, "detail": stmt.detail})
    return HIRStatement(kind="unknown", attrs={"node": type(stmt).__name__})


def _expr_to_hir(expr: object | None) -> Any:
    if expr is None:
        return None
    if isinstance(expr, ast.LiteralExpr):
        return {"kind": "literal", "type": expr.literal_type, "value": expr.value}
    if isinstance(expr, ast.IdentifierExpr):
        return {"kind": "ident", "name": expr.name}
    if isinstance(expr, ast.UnaryExpr):
        return {"kind": "unary", "op": expr.op, "expr": _expr_to_hir(expr.expr)}
    if isinstance(expr, ast.BinaryExpr):
        return {
            "kind": "binary",
            "op": expr.op,
            "left": _expr_to_hir(expr.left),
            "right": _expr_to_hir(expr.right),
        }
    if isinstance(expr, ast.CallExpr):
        intrinsic_name = resolve_intrinsic_name_from_callee(expr.callee) or resolve_call_alias_name(expr.callee)
        if intrinsic_name is not None:
            return {
                "kind": "intrinsic_call",
                "name": intrinsic_name,
                "args": [_expr_to_hir(a) for a in expr.args],
            }
        return {"kind": "call", "callee": _expr_to_hir(expr.callee), "args": [_expr_to_hir(a) for a in expr.args]}
    if isinstance(expr, ast.AttributeExpr):
        return {"kind": "attr", "value": _expr_to_hir(expr.value), "attr": expr.attr}
    if isinstance(expr, ast.IndexExpr):
        return {"kind": "index", "value": _expr_to_hir(expr.value), "index": _expr_to_hir(expr.index)}
    if isinstance(expr, ast.ArrayExpr):
        return {"kind": "array", "elements": [_expr_to_hir(e) for e in expr.elements]}
    if isinstance(expr, ast.SyscallExpr):
        return {"kind": "syscall", "target": _expr_to_hir(expr.target), "args": [_expr_to_hir(a) for a in expr.args]}
    if isinstance(expr, ast.MapExpr):
        return {"kind": "map", "entries": [[_expr_to_hir(k), _expr_to_hir(v)] for k, v in expr.entries]}
    return {"kind": "unknown", "node": type(expr).__name__}
