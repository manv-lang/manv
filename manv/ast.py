"""ManV AST node model.

Why this file exists:
- It defines the canonical in-memory syntax tree shared by parser, semantic
  analysis, interpreter, and all lowering stages.
- It is intentionally "dumb data" (dataclasses only) so every execution mode
  can consume the same structure without hidden behavior.
- New language features (for example relative imports and class syntax aliases)
  are represented here first so downstream passes stay synchronized.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from .diagnostics import Span


# Function/type declaration surface.
@dataclass
class Param:
    name: str
    type_name: str | None
    span: Span


@dataclass
class Decorator:
    """Structured decorator payload preserved all the way into lowering.

    Why this exists:
    - The parser should not leave decorator arguments as raw token strings.
    - Semantics and HLIR lowering need a stable, typed representation so
      deterministic validation and GPU policy normalization happen once.
    """

    name: str
    args: list[Any]
    kwargs: dict[str, Any]
    span: Span


@dataclass
class Program:
    declarations: list[Any]
    statements: list[Any]
    span: Span


# Import surface.
#
# `level` follows Python-style semantics:
# - 0 => absolute import
# - 1 => current package (`from .x import y`)
# - 2+ => parent package traversal (`from ..x import y`)
@dataclass
class ImportStmt:
    module: str
    alias: str | None
    span: Span
    level: int = 0


@dataclass
class FromImportStmt:
    module: str
    name: str
    alias: str | None
    span: Span
    level: int = 0


@dataclass
class FnDecl:
    name: str
    params: list[Param]
    return_type: str | None
    body: list[Any]
    span: Span
    decorators: list[Decorator] = field(default_factory=list)


@dataclass
class TypeDecl:
    name: str
    base_name: str | None
    methods: list[FnDecl]
    span: Span


@dataclass
class ImplDecl:
    target: str
    methods: list[FnDecl]
    span: Span


@dataclass
class MacroDeclStub:
    name: str
    params: list[str]
    body: list[str]
    span: Span


@dataclass
class LetStmt:
    name: str
    type_name: str | None
    value: Any | None
    span: Span
    array_size: Any | None = None


@dataclass
class AssignStmt:
    name: str
    value: Any
    span: Span


@dataclass
class SetAttrStmt:
    target: Any
    attr: str
    value: Any
    span: Span


@dataclass
class SetIndexStmt:
    target: Any
    index: Any
    value: Any
    span: Span


@dataclass
class ReturnStmt:
    value: Any | None
    span: Span


@dataclass
class ExceptClause:
    type_name: str
    bind_name: str | None
    body: list[Any]
    span: Span


@dataclass
class TryStmt:
    try_body: list[Any]
    except_clauses: list[ExceptClause]
    else_body: list[Any]
    finally_body: list[Any]
    span: Span


@dataclass
class RaiseStmt:
    value: Any | None
    span: Span


@dataclass
class IfStmt:
    condition: Any
    then_body: list[Any]
    else_body: list[Any]
    span: Span


@dataclass
class WhileStmt:
    condition: Any
    body: list[Any]
    span: Span


@dataclass
class ForStmt:
    """Canonical source-level for-range loop.

    The frontend only accepts `for <name> in <start>..<stop>:` in v1.
    Downstream passes lower this into explicit control-flow while preserving
    the original range structure for GPU eligibility checks.
    """

    var_name: str
    iterable: Any
    body: list[Any]
    span: Span


@dataclass
class ExprStmt:
    expr: Any
    span: Span


@dataclass
class BreakStmt:
    span: Span


@dataclass
class ContinueStmt:
    span: Span


@dataclass
class SyscallStmt:
    target: Any
    args: list[Any]
    span: Span


@dataclass
class SyscallExpr:
    target: Any
    args: list[Any]
    span: Span


@dataclass
class UnsupportedStmt:
    feature: str
    detail: str
    span: Span


@dataclass
class IdentifierExpr:
    name: str
    span: Span


@dataclass
class LiteralExpr:
    value: Any
    literal_type: str
    span: Span


@dataclass
class UnaryExpr:
    op: str
    expr: Any
    span: Span


@dataclass
class BinaryExpr:
    left: Any
    op: str
    right: Any
    span: Span


@dataclass
class RangeExpr:
    """Half-open source range used by `for ... in a..b` syntax."""

    start: Any
    stop: Any
    span: Span


@dataclass
class CallExpr:
    callee: Any
    args: list[Any]
    span: Span


@dataclass
class AttributeExpr:
    value: Any
    attr: str
    span: Span


@dataclass
class IndexExpr:
    value: Any
    index: Any
    span: Span


@dataclass
class ArrayExpr:
    elements: list[Any]
    span: Span


@dataclass
class MapExpr:
    entries: list[tuple[Any, Any]]
    span: Span


# Backward compatible aliases retained for consumers/tests that still import the old names.
TypeDeclStub = TypeDecl
ImplDeclStub = ImplDecl


def _convert(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_convert(v) for v in value]
    if isinstance(value, list):
        return [_convert(v) for v in value]
    if is_dataclass(value):
        data: dict[str, Any] = {"node": value.__class__.__name__}
        for f in fields(value):
            data[f.name] = _convert(getattr(value, f.name))
        return data
    return value


def to_dict(node: Any) -> dict[str, Any]:
    return _convert(node)
