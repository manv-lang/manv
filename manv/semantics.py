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
BUILTIN_FUNCTIONS.update({"type", "isinstance", "issubclass", "id", "help"})
PRIMITIVE_TYPES = {"int", "i32", "float", "f32", "bool", "str", "u8", "usize", "array", "map", "none", "range"}
BUILTIN_TYPES = {
    "object",
    "type",
    "BaseException",
    "Exception",
    "StopIteration",
    "ImportError",
    "TypeError",
    "AttributeError",
    "KeyError",
    "IndexError",
    "ValueError",
    "OverflowError",
    "RuntimeError",
    "OSError",
    "OutOfMemoryError",
}
STUB_FEATURE_DECLS = (ast.MacroDeclStub,)
STUB_FEATURE_STMTS = (ast.UnsupportedStmt,)


@dataclass
class SemanticResult:
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class GpuDecoratorConfig:
    """Normalized `@gpu` policy captured at the semantic boundary.

    This normalization step is the single place that turns loose decorator
    syntax into stable compiler meaning. Every downstream phase should consume
    the normalized values rather than re-parsing raw decorator arguments.
    """

    required: bool = False
    mode: str = "kernel"


def has_static_method_decorator(decl: ast.FnDecl) -> bool:
    """Return whether a function is explicitly marked as a type-callable method.

    Why this exists:
    - ManV methods are instance-oriented by default.
    - The language still wants an explicit way to say "this method lives on the
      class/type object and does not consume `self`".
    - Interpreter, HLIR lowering, and editor tooling all need the same
      normalized answer instead of ad-hoc decorator string checks.
    """

    return any(decorator.name == "static_method" for decorator in decl.decorators)


def accessor_kind(decl: ast.FnDecl) -> str | None:
    """Return the accessor role for a method, if any.

    Accessors are semantically different from ordinary methods:
    - They participate in attribute reads/writes instead of call syntax.
    - A getter/setter pair should be able to share the same property name.
    - Downstream passes should not need to re-scan raw decorator lists.
    """

    has_getter = any(decorator.name == "getter" for decorator in decl.decorators)
    has_setter = any(decorator.name == "setter" for decorator in decl.decorators)
    if has_getter and has_setter:
        return "both"
    if has_getter:
        return "getter"
    if has_setter:
        return "setter"
    return None


def accessor_property_name(decl: ast.FnDecl) -> str | None:
    """Return the property name targeted by a getter/setter decorator."""

    kind = accessor_kind(decl)
    if kind is None or kind == "both":
        return None
    decorator = next(decorator for decorator in decl.decorators if decorator.name == kind)
    property_name = decl.name
    if decorator.args:
        raise ValueError(f"@{kind} does not accept positional arguments")
    for key, value in decorator.kwargs.items():
        if key != "name":
            raise KeyError(key)
        if not isinstance(value, ast.LiteralExpr) or value.literal_type != "str":
            raise TypeError(f"@{kind}(name=...) expects a string literal")
        property_name = str(value.value)
    return property_name


def normalize_static_method_decorator(decl: ast.FnDecl) -> bool:
    """Validate `@static_method` usage and return whether it is present."""

    static_decorators = [decorator for decorator in decl.decorators if decorator.name == "static_method"]
    if not static_decorators:
        return False

    decorator = static_decorators[-1]
    if decorator.args:
        raise ValueError("@static_method does not accept positional arguments")
    if decorator.kwargs:
        raise KeyError(sorted(decorator.kwargs)[0])
    return True


def normalize_getter_decorator(decl: ast.FnDecl) -> str | None:
    """Validate `@getter` usage and return the property name when present."""

    kind = accessor_kind(decl)
    if kind is None:
        return None
    if kind == "both":
        raise ValueError("a method cannot be both '@getter' and '@setter'")
    if kind != "getter":
        return None
    return accessor_property_name(decl)


def normalize_setter_decorator(decl: ast.FnDecl) -> str | None:
    """Validate `@setter` usage and return the property name when present."""

    kind = accessor_kind(decl)
    if kind is None:
        return None
    if kind == "both":
        raise ValueError("a method cannot be both '@getter' and '@setter'")
    if kind != "setter":
        return None
    return accessor_property_name(decl)


def normalize_type_name(type_name: str | None) -> str | None:
    """Normalize user-facing type aliases into the internal semantic surface.

    Why this exists:
    - The current language still accepts legacy `int`/`float` names.
    - GPU eligibility wants deterministic scalar names (`i32`, `f32`).
    - The new `T[]` syntax should lower to the same internal array type form
      as existing `array[T]` annotations.
    """

    if type_name is None:
        return None

    text = type_name.strip()
    array_depth = 0
    while text.endswith("[]"):
        array_depth += 1
        text = text[:-2].strip()

    base = {
        "int": "i32",
        "float": "f32",
        "void": "none",
    }.get(text, text)

    for _ in range(array_depth):
        base = f"array[{base}]"
    return base


def normalize_gpu_decorator(decl: ast.FnDecl) -> GpuDecoratorConfig | None:
    """Return the normalized GPU policy for a function or `None` if undecorated."""

    gpu_decorators = [decorator for decorator in decl.decorators if decorator.name == "gpu"]
    if not gpu_decorators:
        return None

    decorator = gpu_decorators[-1]
    required = False
    mode = "kernel"

    if decorator.args:
        raise ValueError("@gpu does not accept positional arguments in v1")

    for key, value in decorator.kwargs.items():
        if key == "required":
            if not isinstance(value, ast.LiteralExpr) or value.literal_type != "bool":
                raise TypeError("@gpu(required=...) expects a bool literal")
            required = bool(value.value)
            continue
        if key == "mode":
            if not isinstance(value, ast.LiteralExpr) or value.literal_type != "str":
                raise TypeError("@gpu(mode=...) expects a string literal")
            mode = str(value.value)
            if mode not in {"kernel", "graph"}:
                raise ValueError("@gpu(mode=...) must be 'kernel' or 'graph'")
            continue
        raise KeyError(key)

    return GpuDecoratorConfig(required=required, mode=mode)


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
        self.type_attrs: dict[str, dict[str, str | None]] = {}
        self.type_accessors: dict[str, dict[str, dict[str, ast.FnDecl]]] = {}

    def analyze(self, program: ast.Program) -> SemanticResult:
        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                self._register_function(decl.name, decl)
            elif isinstance(decl, ast.TypeDecl):
                if decl.name in self.types:
                    self._add_error("E2001", f"duplicate type '{decl.name}'", decl.span.line, decl.span.column)
                self.types.add(decl.name)
                attr_table: dict[str, str | None] = {}
                for attr in decl.attrs:
                    if attr.name in attr_table:
                        self._add_error("E2001", f"duplicate type attribute '{decl.name}.{attr.name}'", attr.span.line, attr.span.column)
                    attr_table[attr.name] = normalize_type_name(attr.type_name)
                self.type_attrs[decl.name] = attr_table
                self.type_accessors[decl.name] = self._collect_type_accessors(decl.name, decl.methods)
                for method in decl.methods:
                    if accessor_kind(method) in {"getter", "setter", "both"}:
                        continue
                    self._register_function(f"{decl.name}.{method.name}", method)
            elif isinstance(decl, ast.ImplDecl):
                if decl.target not in self.types:
                    self._add_error("E2014", f"impl target '{decl.target}' is undefined", decl.span.line, decl.span.column)
                accessor_table = self.type_accessors.setdefault(decl.target, {"getters": {}, "setters": {}})
                self._merge_impl_accessors(decl.target, decl.methods, accessor_table)
                for method in decl.methods:
                    if accessor_kind(method) in {"getter", "setter", "both"}:
                        continue
                    self._register_function(f"{decl.target}.{method.name}", method)

        global_scope = Scope()
        for stmt in program.statements:
            self._analyze_stmt(stmt, global_scope, None, loop_depth=0, except_depth=0)

        for decl in program.declarations:
            if isinstance(decl, STUB_FEATURE_DECLS):
                continue
            if isinstance(decl, ast.FnDecl):
                self._analyze_function_decl(decl, global_scope, is_method=False)
            if isinstance(decl, ast.TypeDecl):
                type_scope = Scope(parent=global_scope)
                for attr in decl.attrs:
                    if attr.value is not None:
                        value_type = self._analyze_expr(attr.value, type_scope, as_callee=False)
                        if attr.type_name and value_type and not self._type_compatible(attr.type_name, value_type):
                            self._add_error("E2002", f"type mismatch for '{decl.name}.{attr.name}': expected {attr.type_name}, got {value_type}", attr.span.line, attr.span.column)
                    type_scope.define(attr.name, normalize_type_name(attr.type_name))
                for method in decl.methods:
                    self._analyze_function_decl(method, type_scope, is_method=True)
            if isinstance(decl, ast.ImplDecl):
                for method in decl.methods:
                    self._analyze_function_decl(method, global_scope, is_method=True)

        return SemanticResult(diagnostics=self.diagnostics)

    def _register_function(self, name: str, decl: ast.FnDecl) -> None:
        if name in self.functions:
            self._add_error("E2001", f"duplicate function '{name}'", decl.span.line, decl.span.column)
        self.functions[name] = decl

    def _collect_type_accessors(self, owner: str, methods: list[ast.FnDecl]) -> dict[str, dict[str, ast.FnDecl]]:
        """Build deterministic accessor tables for one type body.

        Accessors are stored separately from ordinary methods because property
        syntax (`obj.name`) should not collide with method call syntax and
        getter/setter pairs need to share one public property name cleanly.
        """

        table: dict[str, dict[str, ast.FnDecl]] = {"getters": {}, "setters": {}}
        self._merge_impl_accessors(owner, methods, table)
        return table

    def _merge_impl_accessors(
        self,
        owner: str,
        methods: list[ast.FnDecl],
        table: dict[str, dict[str, ast.FnDecl]],
    ) -> None:
        for method in methods:
            kind = accessor_kind(method)
            if kind not in {"getter", "setter"}:
                continue
            try:
                property_name = accessor_property_name(method)
            except ValueError as err:
                self._add_error("E2043", str(err), method.span.line, method.span.column)
                continue
            except TypeError as err:
                self._add_error("E2044", str(err), method.span.line, method.span.column)
                continue
            except KeyError as err:
                self._add_error("E2045", f"unknown @{kind} option '{err.args[0]}'", method.span.line, method.span.column)
                continue
            if property_name is None:
                continue
            bucket = "getters" if kind == "getter" else "setters"
            if property_name in table[bucket]:
                self._add_error(
                    "E2046" if kind == "getter" else "E2047",
                    f"duplicate {kind} for property '{owner}.{property_name}'",
                    method.span.line,
                    method.span.column,
                )
                continue
            table[bucket][property_name] = method

    def _analyze_function_decl(self, decl: ast.FnDecl, parent: Scope, *, is_method: bool) -> None:
        self._validate_function_decorators(decl, is_method=is_method)
        fn_scope = Scope(parent=parent)
        for param in decl.params:
            fn_scope.define(param.name, normalize_type_name(param.type_name))
        for stmt in decl.body:
            self._analyze_stmt(stmt, fn_scope, decl, loop_depth=0, except_depth=0)

    def _validate_function_decorators(self, decl: ast.FnDecl, *, is_method: bool) -> None:
        gpu_count = 0
        static_count = 0
        getter_count = 0
        setter_count = 0
        for decorator in decl.decorators:
            if decorator.name == "gpu":
                gpu_count += 1
                if gpu_count > 1:
                    self._add_error("E2031", "duplicate '@gpu' decorator", decorator.span.line, decorator.span.column)
                    continue

                try:
                    normalize_gpu_decorator(ast.FnDecl(name=decl.name, params=decl.params, return_type=decl.return_type, body=decl.body, span=decl.span, decorators=[decorator]))
                except ValueError as err:
                    self._add_error("E2032", str(err), decorator.span.line, decorator.span.column)
                except TypeError as err:
                    self._add_error("E2033", str(err), decorator.span.line, decorator.span.column)
                except KeyError as err:
                    self._add_error("E2034", f"unknown @gpu option '{err.args[0]}'", decorator.span.line, decorator.span.column)
                continue

            if decorator.name == "static_method":
                static_count += 1
                if static_count > 1:
                    self._add_error("E2039", "duplicate '@static_method' decorator", decorator.span.line, decorator.span.column)
                    continue
                if not is_method:
                    self._add_error("E2040", "'@static_method' is only valid on type or impl methods", decorator.span.line, decorator.span.column)
                    continue

                try:
                    normalize_static_method_decorator(ast.FnDecl(name=decl.name, params=decl.params, return_type=decl.return_type, body=decl.body, span=decl.span, decorators=[decorator]))
                except ValueError as err:
                    self._add_error("E2041", str(err), decorator.span.line, decorator.span.column)
                except KeyError as err:
                    self._add_error("E2042", f"unknown @static_method option '{err.args[0]}'", decorator.span.line, decorator.span.column)
                continue

            if decorator.name == "getter":
                getter_count += 1
                if getter_count > 1:
                    self._add_error("E2048", "duplicate '@getter' decorator", decorator.span.line, decorator.span.column)
                    continue
                if not is_method:
                    self._add_error("E2049", "'@getter' is only valid on type or impl methods", decorator.span.line, decorator.span.column)
                    continue
                try:
                    normalize_getter_decorator(ast.FnDecl(name=decl.name, params=decl.params, return_type=decl.return_type, body=decl.body, span=decl.span, decorators=[decorator]))
                except ValueError as err:
                    self._add_error("E2050", str(err), decorator.span.line, decorator.span.column)
                except TypeError as err:
                    self._add_error("E2051", str(err), decorator.span.line, decorator.span.column)
                except KeyError as err:
                    self._add_error("E2052", f"unknown @getter option '{err.args[0]}'", decorator.span.line, decorator.span.column)
                continue

            if decorator.name == "setter":
                setter_count += 1
                if setter_count > 1:
                    self._add_error("E2053", "duplicate '@setter' decorator", decorator.span.line, decorator.span.column)
                    continue
                if not is_method:
                    self._add_error("E2054", "'@setter' is only valid on type or impl methods", decorator.span.line, decorator.span.column)
                    continue
                try:
                    normalize_setter_decorator(ast.FnDecl(name=decl.name, params=decl.params, return_type=decl.return_type, body=decl.body, span=decl.span, decorators=[decorator]))
                except ValueError as err:
                    self._add_error("E2055", str(err), decorator.span.line, decorator.span.column)
                except TypeError as err:
                    self._add_error("E2056", str(err), decorator.span.line, decorator.span.column)
                except KeyError as err:
                    self._add_error("E2057", f"unknown @setter option '{err.args[0]}'", decorator.span.line, decorator.span.column)
                continue

            if decorator.name != "gpu":
                self._add_error("E2030", f"unknown decorator '@{decorator.name}'", decorator.span.line, decorator.span.column)

        if has_static_method_decorator(decl) and accessor_kind(decl) in {"getter", "setter", "both"}:
            self._add_error("E2058", "accessors cannot also be marked '@static_method'", decl.span.line, decl.span.column)

        kind = accessor_kind(decl)
        if kind == "both":
            self._add_error("E2059", "a method cannot be both '@getter' and '@setter'", decl.span.line, decl.span.column)
        if kind == "getter":
            if len(decl.params) != 1 or decl.params[0].name != "self":
                self._add_error("E2060", "@getter methods must have signature 'fn name(self) -> ...'", decl.span.line, decl.span.column)
            if normalize_type_name(decl.return_type) == "none":
                self._add_error("E2061", "@getter methods must return a value", decl.span.line, decl.span.column)
        if kind == "setter":
            if len(decl.params) != 2 or decl.params[0].name != "self":
                self._add_error("E2062", "@setter methods must have signature 'fn name(self, value) -> none'", decl.span.line, decl.span.column)
            if decl.return_type is not None and normalize_type_name(decl.return_type) != "none":
                self._add_error("E2063", "@setter methods must return none", decl.span.line, decl.span.column)

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
            declared_type = normalize_type_name(stmt.type_name) or normalize_type_name(value_type)
            scope.define(stmt.name, declared_type)
            if stmt.type_name and value_type and not self._type_compatible(stmt.type_name, value_type):
                self._add_error("E2002", f"type mismatch for '{stmt.name}': expected {stmt.type_name}, got {value_type}", stmt.span.line, stmt.span.column)
            return

        if isinstance(stmt, ast.ImportStmt):
            bind = stmt.alias if stmt.alias else stmt.module.split(".")[-1]
            scope.define(bind, "module")
            return

        if isinstance(stmt, ast.FromImportStmt):
            bind = stmt.alias if stmt.alias else stmt.name
            scope.define(bind, None)
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
            if isinstance(stmt.target, ast.IdentifierExpr) and stmt.target.name in self.type_attrs:
                if stmt.attr in self.type_attrs.get(stmt.target.name, {}):
                    self._add_error("E2038", f"type attribute '{stmt.target.name}.{stmt.attr}' is immutable", stmt.span.line, stmt.span.column)
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

        if isinstance(stmt, ast.ForStmt):
            iterable_type = self._analyze_expr(stmt.iterable, scope, as_callee=False)
            if iterable_type != "range":
                self._add_error("E2035", "for-loops currently require a range expression like '0..n'", stmt.span.line, stmt.span.column)
            body_scope = Scope(parent=scope)
            body_scope.define(stmt.var_name, "i32")
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

    def _resolve_unshadowed_call_alias(self, callee: object, scope: Scope) -> str | None:
        """Return a builtin call alias only when user code does not shadow it.

        Why this exists:
        - Builtin aliases like `min`/`max` are convenience spellings over
          narrow intrinsics.
        - They should behave like normal builtins and lose to user/module
          bindings with the same name.
        - This is what lets stdlib modules define public functions such as
          `math.min` without semantic analysis silently re-routing calls to the
          builtin intrinsic surface.
        """

        if not isinstance(callee, ast.IdentifierExpr):
            return None
        if scope.contains(callee.name):
            return None
        if callee.name in self.functions or callee.name in self.types:
            return None
        return BUILTIN_ALIASES.get(callee.name)

    def _analyze_expr(self, expr: object, scope: Scope, as_callee: bool) -> str | None:
        if isinstance(expr, ast.LiteralExpr):
            return {"int": "i32", "float": "f32"}.get(expr.literal_type, expr.literal_type)

        if isinstance(expr, ast.IdentifierExpr):
            if expr.name == "std":
                return "namespace"
            if expr.name == "__intrin":
                self._add_error("E2023", "invalid '__intrin' usage; only '__intrin.<name>(...)' is allowed", expr.span.line, expr.span.column)
                return "namespace"
            if as_callee:
                if (
                    expr.name in self.functions
                    or expr.name in BUILTIN_FUNCTIONS
                    or expr.name in self.types
                    or scope.contains(expr.name)
                ):
                    return "fn"
                self._add_error("E2011", f"undefined function or type '{expr.name}'", expr.span.line, expr.span.column)
                return None
            if expr.name in self.functions:
                return "fn"
            if expr.name in self.types:
                return normalize_type_name(expr.name)
            symbol_type = scope.lookup(expr.name)
            if not scope.contains(expr.name):
                self._add_error("E2010", f"undefined variable '{expr.name}'", expr.span.line, expr.span.column)
                return None
            return normalize_type_name(symbol_type)

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
                return "f32"
            return left or right

        if isinstance(expr, ast.RangeExpr):
            start_type = self._analyze_expr(expr.start, scope, as_callee=False)
            stop_type = self._analyze_expr(expr.stop, scope, as_callee=False)
            if start_type and not self._type_compatible("int", start_type):
                self._add_error("E2036", f"range start must be int-compatible, got {start_type}", expr.span.line, expr.span.column)
            if stop_type and not self._type_compatible("int", stop_type):
                self._add_error("E2037", f"range stop must be int-compatible, got {stop_type}", expr.span.line, expr.span.column)
            return "range"

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

            alias = self._resolve_unshadowed_call_alias(expr.callee, scope)
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
                    return normalize_type_name(fn.return_type)
                if expr.callee.name in self.types:
                    return normalize_type_name(expr.callee.name)
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
        expected = normalize_type_name(expected) or expected
        actual = normalize_type_name(actual) or actual
        if expected == actual:
            return True
        if expected in {"int", "i32", "usize", "u8"} and actual in {"int", "i32"}:
            return True
        if expected in {"float", "f32"} and actual in {"float", "f32", "int", "i32"}:
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
