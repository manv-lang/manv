from __future__ import annotations

from dataclasses import dataclass

from . import ast
from .diagnostics import Diagnostic, ManvError, diag
from .intrinsics import (
    BUILTIN_ALIASES,
    IntrinsicTypeVar,
    intrinsic_type_matches,
    is_std_source_path,
    resolve_call_alias_name,
    resolve_intrinsic,
    resolve_intrinsic_name_from_callee,
)


BUILTIN_FUNCTIONS = set(BUILTIN_ALIASES.keys())
PRIMITIVE_TYPES = {"int", "float", "bool", "str", "u8", "usize", "array", "map", "none"}
BUILTIN_TYPES = {
    "object",
    "type",
    "BaseException",
    "Exception",
    "TypeError",
    "AttributeError",
    "KeyError",
    "IndexError",
    "ValueError",
    "RuntimeError",
    "OutOfMemoryError",
}
STUB_FEATURE_DECLS = (ast.MacroDeclStub,)
STUB_FEATURE_STMTS = (ast.UnsupportedStmt,)


@dataclass
class SemanticResult:
    diagnostics: list[Diagnostic]


class Scope:
    def __init__(self, parent: "Scope | None" = None):
        self.parent = parent
        self.symbols: dict[str, str | None] = {}

    def define(self, name: str, type_name: str | None) -> None:
        self.symbols[name] = type_name

    def has_local(self, name: str) -> bool:
        return name in self.symbols

    def lookup(self, name: str) -> str | None:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def contains(self, name: str) -> bool:
        if name in self.symbols:
            return True
        if self.parent is not None:
            return self.parent.contains(name)
        return False


class SemanticAnalyzer:
    def __init__(self, file: str):
        self.file = file
        self.is_std_source = is_std_source_path(file)
        self.diagnostics: list[Diagnostic] = []
        self.functions: dict[str, ast.FnDecl] = {}
        self.types: set[str] = set(BUILTIN_TYPES)

    def analyze(self, program: ast.Program) -> SemanticResult:
        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                self._register_function(decl.name, decl)
            elif isinstance(decl, ast.TypeDecl):
                if decl.name in self.types:
                    self._add_error("E2001", f"duplicate type '{decl.name}'", decl.span.line, decl.span.column)
                self.types.add(decl.name)
                for method in decl.methods:
                    self._register_function(f"{decl.name}.{method.name}", method)
            elif isinstance(decl, ast.ImplDecl):
                if decl.target not in self.types:
                    self._add_error("E2014", f"impl target '{decl.target}' is undefined", decl.span.line, decl.span.column)
                for method in decl.methods:
                    self._register_function(f"{decl.target}.{method.name}", method)

        global_scope = Scope()
        for stmt in program.statements:
            self._analyze_stmt(stmt, global_scope, None, loop_depth=0, except_depth=0)

        for decl in program.declarations:
            if isinstance(decl, STUB_FEATURE_DECLS):
                continue
            if isinstance(decl, ast.FnDecl):
                self._analyze_function_decl(decl, global_scope)
            if isinstance(decl, ast.TypeDecl):
                for method in decl.methods:
                    self._analyze_function_decl(method, global_scope)
            if isinstance(decl, ast.ImplDecl):
                for method in decl.methods:
                    self._analyze_function_decl(method, global_scope)

        return SemanticResult(diagnostics=self.diagnostics)

    def _register_function(self, name: str, decl: ast.FnDecl) -> None:
        if name in self.functions:
            self._add_error("E2001", f"duplicate function '{name}'", decl.span.line, decl.span.column)
        self.functions[name] = decl

    def _analyze_function_decl(self, decl: ast.FnDecl, parent: Scope) -> None:
        fn_scope = Scope(parent=parent)
        for param in decl.params:
            fn_scope.define(param.name, param.type_name)
        for stmt in decl.body:
            self._analyze_stmt(stmt, fn_scope, decl, loop_depth=0, except_depth=0)

    def assert_valid(self, result: SemanticResult) -> None:
        errors = [d for d in result.diagnostics if d.severity == "error"]
        if errors:
            raise ManvError(errors[0], errors[1:])

    def _analyze_stmt(
        self,
        stmt: object,
        scope: Scope,
        fn_decl: ast.FnDecl | None,
        loop_depth: int,
        except_depth: int,
    ) -> None:
        if isinstance(stmt, ast.LetStmt):
            if scope.has_local(stmt.name):
                self._add_error("E2007", f"duplicate variable declaration '{stmt.name}' in same scope", stmt.span.line, stmt.span.column)
            if stmt.array_size is not None:
                size_type = self._analyze_expr(stmt.array_size, scope, as_callee=False)
                if size_type and not self._type_compatible("int", size_type):
                    self._add_error("E2006", f"array size for '{stmt.name}' must be int-compatible, got {size_type}", stmt.span.line, stmt.span.column)
            value_type = self._analyze_expr(stmt.value, scope, as_callee=False) if stmt.value is not None else None
            if stmt.array_size is not None and value_type is None:
                value_type = "array"
            scope.define(stmt.name, stmt.type_name or value_type)
            if stmt.type_name and value_type and not self._type_compatible(stmt.type_name, value_type):
                self._add_error("E2002", f"type mismatch for '{stmt.name}': expected {stmt.type_name}, got {value_type}", stmt.span.line, stmt.span.column)
            return

        if isinstance(stmt, ast.AssignStmt):
            current = scope.lookup(stmt.name)
            if not scope.contains(stmt.name):
                self._add_error("E2003", f"undefined variable '{stmt.name}'", stmt.span.line, stmt.span.column)
            value_type = self._analyze_expr(stmt.value, scope, as_callee=False)
            if current and value_type and not self._type_compatible(current, value_type):
                self._add_error("E2004", f"type mismatch in assignment to '{stmt.name}': expected {current}, got {value_type}", stmt.span.line, stmt.span.column)
            return

        if isinstance(stmt, ast.SetAttrStmt):
            self._analyze_expr(stmt.target, scope, as_callee=False)
            self._analyze_expr(stmt.value, scope, as_callee=False)
            return

        if isinstance(stmt, ast.SetIndexStmt):
            target_type = self._analyze_expr(stmt.target, scope, as_callee=False)
            index_type = self._analyze_expr(stmt.index, scope, as_callee=False)
            self._analyze_expr(stmt.value, scope, as_callee=False)
            if target_type and not (
                target_type == "array"
                or target_type == "map"
                or target_type.startswith("array[")
                or target_type.startswith("[")
                or target_type.startswith("map[")
            ):
                self._add_error("E2012", f"index assignment target must be array/map, got {target_type}", stmt.span.line, stmt.span.column)
            if target_type and (target_type == "array" or target_type.startswith("array") or target_type.startswith("[")):
                if index_type and not self._type_compatible("int", index_type):
                    self._add_error("E2013", f"array index must be int-compatible, got {index_type}", stmt.span.line, stmt.span.column)
            return

        if isinstance(stmt, ast.ReturnStmt):
            if stmt.value is not None:
                ret_type = self._analyze_expr(stmt.value, scope, as_callee=False)
                if fn_decl and fn_decl.return_type and ret_type and not self._type_compatible(fn_decl.return_type, ret_type):
                    self._add_error("E2005", f"return type mismatch in '{fn_decl.name}': expected {fn_decl.return_type}, got {ret_type}", stmt.span.line, stmt.span.column)
            return

        if isinstance(stmt, ast.RaiseStmt):
            if stmt.value is None and except_depth <= 0:
                self._add_error("E2015", "bare 'raise' is only valid inside except blocks", stmt.span.line, stmt.span.column)
            if stmt.value is not None:
                self._analyze_expr(stmt.value, scope, as_callee=False)
            return

        if isinstance(stmt, ast.SyscallStmt):
            target_type = self._analyze_expr(stmt.target, scope, as_callee=False)
            if target_type and target_type not in {"str", "int"}:
                self._add_error("E2024", f"syscall target must be str or int, got {target_type}", stmt.span.line, stmt.span.column)
            for arg in stmt.args:
                self._analyze_expr(arg, scope, as_callee=False)
            return

        if isinstance(stmt, ast.TryStmt):
            try_scope = Scope(parent=scope)
            for inner in stmt.try_body:
                self._analyze_stmt(inner, try_scope, fn_decl, loop_depth=loop_depth, except_depth=except_depth)
            for clause in stmt.except_clauses:
                clause_scope = Scope(parent=scope)
                if clause.bind_name:
                    clause_scope.define(clause.bind_name, clause.type_name)
                for inner in clause.body:
                    self._analyze_stmt(inner, clause_scope, fn_decl, loop_depth=loop_depth, except_depth=except_depth + 1)
            else_scope = Scope(parent=scope)
            for inner in stmt.else_body:
                self._analyze_stmt(inner, else_scope, fn_decl, loop_depth=loop_depth, except_depth=except_depth)
            finally_scope = Scope(parent=scope)
            for inner in stmt.finally_body:
                self._analyze_stmt(inner, finally_scope, fn_decl, loop_depth=loop_depth, except_depth=except_depth)
            return

        if isinstance(stmt, ast.IfStmt):
            self._analyze_expr(stmt.condition, scope, as_callee=False)
            then_scope = Scope(parent=scope)
            for inner in stmt.then_body:
                self._analyze_stmt(inner, then_scope, fn_decl, loop_depth=loop_depth, except_depth=except_depth)
            else_scope = Scope(parent=scope)
            for inner in stmt.else_body:
                self._analyze_stmt(inner, else_scope, fn_decl, loop_depth=loop_depth, except_depth=except_depth)
            return

        if isinstance(stmt, ast.WhileStmt):
            self._analyze_expr(stmt.condition, scope, as_callee=False)
            body_scope = Scope(parent=scope)
            for inner in stmt.body:
                self._analyze_stmt(inner, body_scope, fn_decl, loop_depth=loop_depth + 1, except_depth=except_depth)
            return

        if isinstance(stmt, ast.BreakStmt):
            if loop_depth <= 0:
                self._add_error("E2008", "'break' is only valid inside loops", stmt.span.line, stmt.span.column)
            return

        if isinstance(stmt, ast.ContinueStmt):
            if loop_depth <= 0:
                self._add_error("E2009", "'continue' is only valid inside loops", stmt.span.line, stmt.span.column)
            return

        if isinstance(stmt, ast.ExprStmt):
            self._analyze_expr(stmt.expr, scope, as_callee=False)
            return

        if isinstance(stmt, STUB_FEATURE_STMTS):
            return

        self._add_error("E2099", f"unsupported statement '{type(stmt).__name__}'", 1, 1)

    def _analyze_intrinsic_call(
        self,
        intrinsic_name: str,
        args: list[object],
        scope: Scope,
        *,
        line: int,
        col: int,
        enforce_std_only: bool,
    ) -> str | None:
        spec = resolve_intrinsic(intrinsic_name)
        if spec is None:
            self._add_error("E2020", f"unknown intrinsic '{intrinsic_name}'", line, col)
            for arg in args:
                self._analyze_expr(arg, scope, as_callee=False)
            return None

        if enforce_std_only and spec.std_only and not self.is_std_source:
            self._add_error("E2022", f"intrinsic '{intrinsic_name}' is restricted to std sources", line, col)

        got_types: list[str | None] = [self._analyze_expr(arg, scope, as_callee=False) for arg in args]
        expected = spec.arg_types
        if len(args) != len(expected):
            self._add_error(
                "E2021",
                f"intrinsic '{intrinsic_name}' expects {len(expected)} args, got {len(args)}",
                line,
                col,
            )
        else:
            for idx, (exp, got) in enumerate(zip(expected, got_types, strict=True)):
                if not intrinsic_type_matches(exp, got):
                    label = exp.name if isinstance(exp, IntrinsicTypeVar) else exp
                    self._add_error(
                        "E2021",
                        f"intrinsic '{intrinsic_name}' arg {idx + 1} type mismatch: expected {label}, got {got}",
                        line,
                        col,
                    )

        if isinstance(spec.return_type, IntrinsicTypeVar):
            return None
        ret = str(spec.return_type)
        if ret == "none":
            return None
        return ret

    def _analyze_expr(self, expr: object, scope: Scope, as_callee: bool) -> str | None:
        if isinstance(expr, ast.LiteralExpr):
            return expr.literal_type

        if isinstance(expr, ast.IdentifierExpr):
            if expr.name == "std":
                return "namespace"
            if expr.name == "__intrin":
                self._add_error("E2023", "invalid '__intrin' usage; only '__intrin.<name>(...)' is allowed", expr.span.line, expr.span.column)
                return "namespace"
            if as_callee:
                if expr.name in self.functions or expr.name in BUILTIN_FUNCTIONS or expr.name in self.types:
                    return "fn"
                self._add_error("E2011", f"undefined function or type '{expr.name}'", expr.span.line, expr.span.column)
                return None
            symbol_type = scope.lookup(expr.name)
            if not scope.contains(expr.name):
                self._add_error("E2010", f"undefined variable '{expr.name}'", expr.span.line, expr.span.column)
                return None
            return symbol_type

        if isinstance(expr, ast.UnaryExpr):
            inner = self._analyze_expr(expr.expr, scope, as_callee=False)
            if expr.op in {"!", "not"}:
                return "bool"
            return inner

        if isinstance(expr, ast.BinaryExpr):
            left = self._analyze_expr(expr.left, scope, as_callee=False)
            right = self._analyze_expr(expr.right, scope, as_callee=False)
            if expr.op in {"and", "or", "&&", "||"}:
                return "bool"
            if expr.op in {"==", "!=", "<", "<=", ">", ">="}:
                return "bool"
            if left and right and left != right and left in {"int", "float"} and right in {"int", "float"}:
                return "float"
            return left or right

        if isinstance(expr, ast.CallExpr):
            intrinsic_name = resolve_intrinsic_name_from_callee(expr.callee)
            if intrinsic_name is not None:
                return self._analyze_intrinsic_call(
                    intrinsic_name,
                    expr.args,
                    scope,
                    line=expr.span.line,
                    col=expr.span.column,
                    enforce_std_only=True,
                )

            alias = resolve_call_alias_name(expr.callee)
            if alias is not None:
                return self._analyze_intrinsic_call(
                    alias,
                    expr.args,
                    scope,
                    line=expr.span.line,
                    col=expr.span.column,
                    enforce_std_only=False,
                )

            self._analyze_expr(expr.callee, scope, as_callee=True)
            for arg in expr.args:
                self._analyze_expr(arg, scope, as_callee=False)
            if isinstance(expr.callee, ast.IdentifierExpr):
                fn = self.functions.get(expr.callee.name)
                if fn:
                    return fn.return_type
                if expr.callee.name in self.types:
                    return expr.callee.name
            return None

        if isinstance(expr, ast.AttributeExpr):
            if isinstance(expr.value, ast.IdentifierExpr) and expr.value.name == "__intrin":
                if as_callee:
                    return "fn"
                self._add_error("E2023", "invalid '__intrin' usage; only '__intrin.<name>(...)' is allowed", expr.span.line, expr.span.column)
                return None
            base_type = self._analyze_expr(expr.value, scope, as_callee=False)
            if base_type == "namespace":
                return "namespace_attr"
            return None

        if isinstance(expr, ast.IndexExpr):
            self._analyze_expr(expr.value, scope, as_callee=False)
            self._analyze_expr(expr.index, scope, as_callee=False)
            return None

        if isinstance(expr, ast.ArrayExpr):
            for e in expr.elements:
                self._analyze_expr(e, scope, as_callee=False)
            return "array"

        if isinstance(expr, ast.SyscallExpr):
            target_type = self._analyze_expr(expr.target, scope, as_callee=False)
            if target_type and target_type not in {"str", "int"}:
                self._add_error("E2024", f"syscall target must be str or int, got {target_type}", expr.span.line, expr.span.column)
            for arg in expr.args:
                self._analyze_expr(arg, scope, as_callee=False)
            return "map"

        if isinstance(expr, ast.MapExpr):
            for k, v in expr.entries:
                self._analyze_expr(k, scope, as_callee=False)
                self._analyze_expr(v, scope, as_callee=False)
            return "map"

        return None

    def _type_compatible(self, expected: str, actual: str) -> bool:
        if expected == actual:
            return True
        if expected in {"int", "usize", "u8"} and actual == "int":
            return True
        if expected == "float" and actual in {"float", "int"}:
            return True
        if expected.startswith("[") and actual == "array":
            return True
        if expected.startswith("array[") and actual == "array":
            return True
        if expected == "array" and actual == "array":
            return True
        if expected.startswith("map[") and actual == "map":
            return True
        if expected == "map" and actual == "map":
            return True
        if expected in self.types and actual in self.types:
            return expected == actual
        return expected not in PRIMITIVE_TYPES

    def _add_error(self, code: str, message: str, line: int, column: int) -> None:
        self.diagnostics.append(diag(code, message, self.file, line, column))
