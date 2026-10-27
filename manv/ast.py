"""ManV AST node model.

Why this file exists:
- It defines the canonical in-memory syntax tree shared by parser, semantic
  analysis, interpreter, and all lowering stages.
- It is intentionally "dumb data" (dataclasses only) so every execution mode
  can consume the same structure without hidden behavior.
- New language features are represented here first so downstream passes stay
  synchronized.

Module system surface:
- `module <name>` — declares the canonical identity of this file's module.
- `include` — the single import keyword; three syntactic forms:
    include <module>                    whole-module, bound as leaf or alias
    include <module> as <alias>         whole-module with alias
    include <name> from <module>        specific symbol, bound as name or alias
    include <name> from <module> as <alias>
    include <a>, <b> from <module>      multiple symbols (expands to list)
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
    docstring: str | None = None


# Module system surface.
#
# `level` on IncludeStmt follows the same relative-depth convention used by
# the resolver:
# - 0  => absolute
# - 1  => current package  (`.sibling`)
# - 2+ => parent traversal (`..parent.mod`)

@dataclass
class ModuleDecl:
    """Declares the canonical identity of this source file's module.

    `module mypackage` (or `module pkg.sub`) at the top of a .mv file
    registers a human-readable name that the runtime and tooling surface.
    Module resolution still uses file paths; this is metadata, not a
    replacement for path-based resolution.
    """
    name: str
    span: Span


@dataclass
class IncludeStmt:
    """New-style import statement.  All three syntactic forms desugar here.

    Whole-module forms (name is None):
        include mymod              → module="mymod", name=None, alias=None
        include mymod as m         → module="mymod", name=None, alias="m"
        include pkg.sub            → module="pkg.sub", name=None, alias=None

    Single-name forms (name is not None):
        include Foo from mymod         → module="mymod", name="Foo",  alias=None
        include Foo from mymod as F    → module="mymod", name="Foo",  alias="F"

    Multi-name form: the parser expands `include A, B from mymod` into a
    list[IncludeStmt], one per name — the same pattern the old
    from-import list used.  Each element has a distinct name and optional alias.
    """
    module: str
    name: str | None       # None → whole-module include
    alias: str | None      # binding alias (None → use name or leaf of module)
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
    docstring: str | None = None


@dataclass
class TypeAttrDecl:
    """Immutable attribute declaration inside a type/class body.

    Why this exists:
    - `Math.pi`-style constants need a source-level declaration form that lives
      on the type object rather than executing as an ordinary local statement.
    - The interpreter already supports type attrs at runtime; this node makes
      that surface explicit in the AST so semantics, LSP, and lowering can
      reason about them deterministically.
    """

    name: str
    type_name: str | None
    value: Any
    span: Span


@dataclass
class TypeDecl:
    name: str
    attrs: list[TypeAttrDecl]
    methods: list[FnDecl]
    span: Span
    base_names: list[str] = field(default_factory=list)
    docstring: str | None = None


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
class MacroCallExpr:
    """Call-site node for macro invocations: ``name!(arg1, arg2)``."""

    name: str
    args: list[Any]
    span: Span


@dataclass
class LetStmt:
    """A declaration: `T name = v`, or `const [T] name = v` when is_const is set."""

    name: str
    type_name: str | None
    value: Any | None
    span: Span
    array_size: Any | None = None
    is_const: bool = False


@dataclass
class AssignStmt:
    """`name = value`. When `name` was not bound yet, semantic analysis sets `declares`
    and the statement introduces the variable (inferred type) instead of assigning."""

    name: str
    value: Any
    span: Span
    declares: bool = False


@dataclass
class AugAssignStmt:
    name: str
    op: str
    value: Any
    span: Span


@dataclass
class AugAssignAttrStmt:
    target: Any
    attr: str
    op: str
    value: Any
    span: Span


@dataclass
class AugAssignIndexStmt:
    target: Any
    index: Any
    op: str
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
    else_body: list[Any] = field(default_factory=list)


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
    else_body: list[Any] = field(default_factory=list)


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
class WithStmt:
    context: Any
    bind_name: str | None
    body: list[Any]
    span: Span


@dataclass
class CaseClause:
    pattern: Any
    guard: Any | None
    body: list[Any]
    span: Span


@dataclass
class MatchStmt:
    subject: Any
    cases: list[CaseClause]
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
class ExternFnDecl:
    """Single extern C function declaration.

    `extern "c" fn malloc(size: int) -> ptr`
    `extern "c" fn c_len(s: ptr) -> int as "strlen"`  # alias form
    """
    name: str            # ManV binding name
    c_name: str          # actual C symbol name (equals name unless `as` alias given)
    lib: str             # library identifier ("c", "m", "ssl", absolute path)
    params: list[Any]
    return_type: str | None   # None / "void" → C void
    varargs: bool             # True when trailing `...` present (printf-style)
    span: Span


@dataclass
class ExternBlock:
    """Block of extern declarations sharing a library name.

    extern "c":
        fn malloc(size: int) -> ptr
        fn free(p: ptr)
    """
    lib: str
    fns: list[Any]
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
    raw_value: str | None = None


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
class CondExpr:
    """`then_expr if condition else else_expr`: only the chosen branch is evaluated."""

    condition: Any
    then_expr: Any
    else_expr: Any
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


@dataclass
class LambdaExpr:
    """Anonymous function expression: ``fn(params) -> type: body``.

    Why this exists:
    - First-class functions and closures require the ability to create a
      function value anywhere an expression is expected.
    - The body shares syntax with FnDecl; the only difference is there is
      no name token and the node lives in expression position.
    """

    params: list[Param]
    return_type: str | None
    body: list[Any]
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
        data: dict[str, Any] = {"node": type(value).__name__}
        for f in fields(value):
            data[f.name] = _convert(getattr(value, f.name))
        return data
    return value


def to_dict(node: Any) -> dict[str, Any]:
    return _convert(node)
