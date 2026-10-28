"""Flow checks that need the shape of the code, not types: use of a variable that may not be assigned
yet (definite assignment) and statements that can never run (unreachable code).

Why this exists:
- `int x` declares a variable without a value. Reading it before every path has assigned it is a
  bug that is cheap to catch here and a confusing `none` at run time.
- A statement after `return`, `raise`, `break` or `continue` never runs. That is almost always a
  mistake, so it is reported (as a warning: the program still runs).

Only variables declared without a value are tracked, so ordinary code never sees a false alarm.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Callable

from . import ast
from .memory import SCALARS, is_pointer_type

_TERMINATORS = (ast.ReturnStmt, ast.RaiseStmt, ast.BreakStmt, ast.ContinueStmt)
# Nodes whose bodies run later or elsewhere: a read inside them says nothing about this point in the flow.
_DEFERRED = (ast.FnDecl, ast.LambdaExpr)


def check_program(
    program: ast.Program,
    struct_names: set[str],
    error: Callable[[str, str, int, int], None],
    warning: Callable[[str, str, int, int], None],
) -> None:
    """Run both checks over the top-level statements and every function body."""
    _Flow(struct_names, error).block(program.statements, set())
    for decl in program.declarations:
        for body in _bodies(decl):
            _Flow(struct_names, error).block(body, set())
    _unreachable(program.statements, warning)
    for decl in program.declarations:
        for body in _bodies(decl):
            _unreachable(body, warning)


def _bodies(decl: Any) -> list[list[Any]]:
    if isinstance(decl, ast.FnDecl):
        return [decl.body]
    if isinstance(decl, (ast.TypeDecl, ast.ImplDecl)):
        return [method.body for method in decl.methods]
    return []


# ------------------------------------------------------------------------------------------------
# Unreachable code
# ------------------------------------------------------------------------------------------------
def _unreachable(body: list[Any], warning: Callable[[str, str, int, int], None]) -> None:
    for index, stmt in enumerate(body):
        if isinstance(stmt, _TERMINATORS) and index + 1 < len(body):
            nxt = body[index + 1]
            span = getattr(nxt, "span", None)
            if span is not None:
                warning("W2001", "unreachable code: this statement can never run", span.line, span.column)
            break
        for inner in _child_blocks(stmt):
            _unreachable(inner, warning)


def _child_blocks(stmt: Any) -> list[list[Any]]:
    if isinstance(stmt, ast.IfStmt):
        return [stmt.then_body, stmt.else_body]
    if isinstance(stmt, (ast.WhileStmt, ast.ForStmt)):
        return [stmt.body, stmt.else_body]
    if isinstance(stmt, ast.WithStmt):
        return [stmt.body]
    if isinstance(stmt, ast.TryStmt):
        blocks = [stmt.try_body, stmt.else_body or [], stmt.finally_body or []]
        blocks.extend(clause.body for clause in stmt.except_clauses)
        return blocks
    if isinstance(stmt, ast.MatchStmt):
        return [case.body for case in stmt.cases]
    if isinstance(stmt, ast.FnDecl):
        return [stmt.body]
    return []


# ------------------------------------------------------------------------------------------------
# Definite assignment
# ------------------------------------------------------------------------------------------------
class _Flow:
    """Walks a block keeping the set of tracked variables that may still be unassigned."""

    def __init__(self, struct_names: set[str], error: Callable[[str, str, int, int], None]):
        self.struct_names = struct_names
        self.error = error

    # returns (state after the block, whether the block always leaves through return/raise/break/continue)
    def block(self, body: list[Any], state: set[str]) -> tuple[set[str], bool]:
        state = set(state)
        for stmt in body:
            state, terminated = self.stmt(stmt, state)
            if terminated:
                return state, True
        return state, False

    def stmt(self, stmt: Any, state: set[str]) -> tuple[set[str], bool]:
        if isinstance(stmt, ast.LetStmt):
            if stmt.value is not None:
                self.reads(stmt.value, state)
            if stmt.array_size is not None:
                self.reads(stmt.array_size, state)
            state.discard(stmt.name)
            if stmt.value is None and not stmt.c_array and stmt.array_size is None and self._starts_empty(stmt.type_name):
                state.add(stmt.name)
            return state, False
        if isinstance(stmt, ast.AssignStmt):
            self.reads(stmt.value, state)
            state.discard(stmt.name)
            return state, False
        if isinstance(stmt, ast.DestructureStmt):
            self.reads(stmt.value, state)
            for target in stmt.targets:
                state.discard(target.name)
            return state, False
        if isinstance(stmt, ast.AugAssignStmt):
            self._read_name(stmt.name, state, stmt)
            self.reads(stmt.value, state)
            return state, False
        if isinstance(stmt, ast.IfStmt):
            self.reads(stmt.condition, state)
            then_state, then_done = self.block(stmt.then_body, state)
            else_state, else_done = self.block(stmt.else_body, state)
            return self._join([(then_state, then_done), (else_state, else_done)])
        if isinstance(stmt, ast.WhileStmt):
            self.reads(stmt.condition, state)
            self.block(stmt.body, state)
            self.block(stmt.else_body, state)
            return state, False  # the body may run zero times: its assignments do not count
        if isinstance(stmt, ast.ForStmt):
            self.reads(stmt.iterable, state)
            self.block(stmt.body, state - {stmt.var_name})
            self.block(stmt.else_body, state)
            return state, False
        if isinstance(stmt, ast.WithStmt):
            self.reads(stmt.context, state)
            return self.block(stmt.body, state)
        if isinstance(stmt, ast.TryStmt):
            body_state, body_done = self.block(stmt.try_body, state)
            if stmt.else_body and not body_done:
                body_state, body_done = self.block(stmt.else_body, body_state)
            paths = [(body_state, body_done)]
            for clause in stmt.except_clauses:
                paths.append(self.block(clause.body, state))  # an exception can come from anywhere in the body
            joined, done = self._join(paths)
            if stmt.finally_body:
                joined, finally_done = self.block(stmt.finally_body, joined if not done else state)
                done = done or finally_done
            return joined, done
        if isinstance(stmt, ast.MatchStmt):
            self.reads(stmt.subject, state)
            paths = []
            has_default = False
            for case in stmt.cases:
                if case.guard is not None:
                    self.reads(case.guard, state)
                if isinstance(case.pattern, ast.IdentifierExpr) and case.pattern.name == "_" and case.guard is None:
                    has_default = True
                paths.append(self.block(case.body, state))
            if not has_default:
                paths.append((set(state), False))
            return self._join(paths)
        if isinstance(stmt, _DEFERRED):
            return state, False
        if isinstance(stmt, (ast.ReturnStmt, ast.RaiseStmt)):
            if getattr(stmt, "value", None) is not None:
                self.reads(stmt.value, state)
            return state, True
        if isinstance(stmt, (ast.BreakStmt, ast.ContinueStmt)):
            return state, True
        self.reads(stmt, state)  # expression statements, stores through pointers, attribute and index assignments
        return state, False

    def _starts_empty(self, type_name: str | None) -> bool:
        """A declaration without a value leaves a plain variable unassigned; pointers, structs and arrays get a value."""
        if type_name is None:
            return False
        base = type_name.replace("const ", "").strip()
        return not (is_pointer_type(base) or base in self.struct_names or base.startswith("array") or base.endswith("]"))

    @staticmethod
    def _join(paths: list[tuple[set[str], bool]]) -> tuple[set[str], bool]:
        live = [state for state, done in paths if not done]
        if not live:
            return set(), True
        merged: set[str] = set()
        for state in live:
            merged |= state
        return merged, False

    # -- reading expressions -------------------------------------------------------------------
    def reads(self, node: Any, state: set[str]) -> None:
        if not state:
            return
        if isinstance(node, ast.IdentifierExpr):
            self._read_name(node.name, state, node)
        elif isinstance(node, _DEFERRED):
            return
        elif isinstance(node, (list, tuple)):
            for item in node:
                self.reads(item, state)
        elif is_dataclass(node) and not isinstance(node, type):
            for f in fields(node):
                if f.name != "span":
                    self.reads(getattr(node, f.name), state)

    def _read_name(self, name: str, state: set[str], node: Any) -> None:
        if name in state:
            span = getattr(node, "span", None)
            line, column = (span.line, span.column) if span is not None else (1, 1)
            self.error("E2079", f"'{name}' may be used before it is assigned: assign it on every path first", line, column)
            state.discard(name)  # one report per variable is enough
