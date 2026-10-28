from __future__ import annotations

from dataclasses import dataclass

from . import ast
from .memory import SCALARS, canonical_pointer_type, is_pointer_type, pointee_of
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
# Builtins that work on raw memory: name -> (fewest args, most args, result type).
POINTER_BUILTINS: dict[str, tuple[int, int, str | None]] = {
    "malloc": (1, 1, "void*"),
    "calloc": (2, 2, "void*"),
    "realloc": (2, 2, "void*"),
    "free": (1, 1, None),
    "memcpy": (3, 3, "void*"),
    "memmove": (3, 3, "void*"),
    "memset": (3, 3, "void*"),
    "memcmp": (3, 3, "int"),
    "cstr": (1, 1, "u8*"),
    "str_from": (1, 2, "str"),
}
BUILTIN_FUNCTIONS.update(POINTER_BUILTINS)
_POINTEE_ALIASES = {"int": "i64", "float": "f64", "usize": "u64", "ptr": "void*"}
_INTEGER_FAMILY = {"int", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "usize", "bool"}
_FLOAT_FAMILY = {"float", "f32", "f64"}
PRIMITIVE_TYPES = {"int", "i32", "float", "f32", "bool", "str", "u8", "usize", "array", "map", "tuple", "none", "range"}
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
    "PointerError",
    "NullPointerError",
}
STUB_FEATURE_DECLS = ()
STUB_FEATURE_STMTS = (ast.UnsupportedStmt, ast.ExternBlock)


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


def has_classmethod_decorator(decl: ast.FnDecl) -> bool:
    """Return whether a function is explicitly marked as a classmethod."""
    return any(decorator.name == "classmethod" for decorator in decl.decorators)


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
    if text == "ptr" or text.endswith("*"):
        return canonical_pointer_type(text)  # pointer types keep their pointee spelling (`int*` is not `i32*`)
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
        self.consts: set[str] = set()
        self.annotations: dict[str, str | None] = {}

    def define(self, name: str, type_name: str | None, *, const: bool = False, annotation: str | None = None) -> None:
        self.symbols[name] = type_name
        self.annotations[name] = annotation
        if const:
            self.consts.add(name)
        else:
            self.consts.discard(name)

    def is_const(self, name: str) -> bool:
        """True when the nearest binding of `name` is a constant."""
        if name in self.symbols:
            return name in self.consts
        return self.parent.is_const(name) if self.parent is not None else False

    def annotation_of(self, name: str) -> str | None:
        """The type as written at the declaration (`i32`, `Vec2`, `u8*`), or None when it was inferred."""
        if name in self.symbols:
            return self.annotations.get(name)
        return self.parent.annotation_of(name) if self.parent is not None else None

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
        self.extern_fns: set[str] = set()
        self.types: set[str] = set(BUILTIN_TYPES)
        self.type_attrs: dict[str, dict[str, str | None]] = {}
        self.type_accessors: dict[str, dict[str, dict[str, ast.FnDecl]]] = {}
        self.structs: dict[str, ast.StructDecl] = {}

    def analyze(self, program: ast.Program) -> SemanticResult:
        for decl in program.declarations:
            if isinstance(decl, ast.StructDecl):
                if decl.name in self.types:
                    self._add_error("E2001", f"duplicate type '{decl.name}'", decl.span.line, decl.span.column)
                self.types.add(decl.name)
                self.structs[decl.name] = decl
        for decl in program.declarations:
            if isinstance(decl, ast.StructDecl):
                self._check_struct(decl)

        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                self._register_function(decl.name, decl)
            elif isinstance(decl, ast.ExternBlock):
                for fn in decl.fns:
                    self.extern_fns.add(fn.name)
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

    def _check_struct(self, decl: ast.StructDecl) -> None:
        seen: set[str] = set()
        for f in decl.fields:
            if f.name in seen:
                self._add_error("E2001", f"duplicate field '{decl.name}.{f.name}'", f.span.line, f.span.column)
            seen.add(f.name)
            base = f.type_name.replace("const ", "").strip()
            if is_pointer_type(f.type_name):
                continue
            if base == decl.name:
                self._add_error("E2076", f"struct '{decl.name}' cannot contain itself; use '{decl.name}*'", f.span.line, f.span.column)
            elif base not in SCALARS and base not in self.structs:
                self._add_error("E2075", f"unknown field type '{f.type_name}' in struct '{decl.name}'", f.span.line, f.span.column)
            if f.count is not None and not (isinstance(f.count, ast.LiteralExpr) and isinstance(f.count.value, int) and f.count.value > 0):
                self._add_error("E2077", f"array size of '{decl.name}.{f.name}' must be a positive integer literal", f.span.line, f.span.column)

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
            if param.default is not None:
                self._analyze_expr(param.default, parent, as_callee=False)
            fn_scope.define(param.name, normalize_type_name(param.type_name), annotation=param.type_name)
        for stmt in decl.body:
            self._analyze_stmt(stmt, fn_scope, decl, loop_depth=0, except_depth=0)

    def _check_call_arguments(self, call: ast.CallExpr, fn: ast.FnDecl) -> None:
        """Arity and keyword checks for a call to a known top-level function."""
        line, col = call.span.line, call.span.column
        names = [p.name for p in fn.params]
        if len(call.args) > len(names):
            self._add_error("E2066", f"'{fn.name}' takes at most {len(names)} argument(s), got {len(call.args)}", line, col)
            return
        bound = set(names[: len(call.args)])
        for name, _ in call.kwargs:
            if name not in names:
                self._add_error("E2067", f"'{fn.name}' has no parameter named '{name}'", line, col)
            elif name in bound:
                self._add_error("E2068", f"'{fn.name}' got more than one value for parameter '{name}'", line, col)
            bound.add(name)
        missing = [p.name for p in fn.params if p.default is None and p.name not in bound]
        if missing:
            self._add_error("E2069", f"'{fn.name}' is missing argument(s): {', '.join(missing)}", line, col)

    def _check_store_through(self, ptr_type: str | None, value_type: str | None, stmt: Any) -> None:
        """`*p = v` / `p[i] = v`: p must be a pointer to something writable that accepts v."""
        line, col = stmt.span.line, stmt.span.column
        if ptr_type is None:
            return
        if not is_pointer_type(ptr_type):
            if not ptr_type.startswith("array["):
                self._add_error("E2070", f"cannot store through a value of type {ptr_type}; it is not a pointer", line, col)
            return
        if ptr_type.startswith("const "):
            self._add_error("E2072", f"cannot assign through '{ptr_type}': the pointee is const", line, col)
            return
        pointee = pointee_of(ptr_type)
        if pointee == "void":
            self._add_error("E2070", "cannot store through a void*; cast it to a typed pointer first", line, col)
        elif value_type and not self._type_compatible(pointee, value_type):
            self._add_error("E2002", f"type mismatch storing through '{ptr_type}': expected {pointee}, got {value_type}", line, col)

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
            annotation = None if stmt.c_array else stmt.type_name
            scope.define(stmt.name, declared_type, const=stmt.is_const, annotation=annotation)
            if stmt.type_name and value_type and not self._type_compatible(stmt.type_name, value_type):
                self._add_error("E2002", f"type mismatch for '{stmt.name}': expected {stmt.type_name}, got {value_type}", stmt.span.line, stmt.span.column)
            return

        if isinstance(stmt, ast.ModuleDecl):
            # Module declarations are pure metadata; no binding is introduced.
            return

        if isinstance(stmt, ast.FnDecl):
            # A function declared inside another body: visible from its declaration on,
            # and able to read the enclosing scopes (closure).
            if scope.has_local(stmt.name):
                self._add_error("E2007", f"duplicate variable declaration '{stmt.name}' in same scope", stmt.span.line, stmt.span.column)
            scope.define(stmt.name, "fn")
            self._analyze_function_decl(stmt, scope, is_method=False)
            return

        if isinstance(stmt, ast.AugAssignStmt):
            if not scope.contains(stmt.name):
                self._add_error("E2003", f"undefined variable '{stmt.name}'", stmt.span.line, stmt.span.column)
            elif scope.is_const(stmt.name):
                self._add_error("E2064", f"cannot assign to constant '{stmt.name}'", stmt.span.line, stmt.span.column)
            self._analyze_expr(stmt.value, scope, as_callee=False)
            return

        if isinstance(stmt, ast.AugAssignAttrStmt):
            self._check_not_const_target(stmt.target, scope, stmt)
            self._analyze_expr(stmt.target, scope, as_callee=False)
            if isinstance(stmt.target, ast.IdentifierExpr) and stmt.target.name in self.type_attrs:
                if stmt.attr in self.type_attrs.get(stmt.target.name, {}):
                    self._add_error("E2038", f"type attribute '{stmt.target.name}.{stmt.attr}' is immutable", stmt.span.line, stmt.span.column)
            self._analyze_expr(stmt.value, scope, as_callee=False)
            return

        if isinstance(stmt, ast.AugAssignIndexStmt):
            target_type = self._analyze_expr(stmt.target, scope, as_callee=False)
            self._analyze_expr(stmt.index, scope, as_callee=False)
            value_type = self._analyze_expr(stmt.value, scope, as_callee=False)
            if is_pointer_type(target_type):
                self._check_store_through(target_type, value_type, stmt)
                return
            self._check_not_const_target(stmt.target, scope, stmt)
            if target_type and not (
                target_type == "array"
                or target_type == "map"
                or target_type.startswith("array[")
                or target_type.startswith("[")
                or target_type.startswith("map[")
            ):
                self._add_error("E2012", f"index assignment target must be array/map, got {target_type}", stmt.span.line, stmt.span.column)
            return

        if isinstance(stmt, ast.MatchStmt):
            self._analyze_expr(stmt.subject, scope, as_callee=False)
            for case in stmt.cases:
                case_scope = Scope(parent=scope)
                pattern = case.pattern
                if isinstance(pattern, ast.IdentifierExpr) and pattern.name == "_":
                    pass  # wildcard
                elif isinstance(pattern, ast.IdentifierExpr) and not scope.contains(pattern.name):
                    case_scope.define(pattern.name, None)  # an unbound name captures the subject
                else:
                    self._analyze_expr(pattern, scope, as_callee=False)
                if case.guard is not None:
                    self._analyze_expr(case.guard, case_scope, as_callee=False)
                for inner in case.body:
                    self._analyze_stmt(inner, case_scope, fn_decl, loop_depth=loop_depth, except_depth=except_depth)
            return

        if isinstance(stmt, ast.IncludeStmt):
            if stmt.name is None:
                # Whole-module include: bind as alias or leaf of dotted path.
                bind = stmt.alias if stmt.alias else stmt.module.split(".")[-1]
                scope.define(bind, "module")
            else:
                # Specific-name include: bind as alias or the symbol name itself.
                bind = stmt.alias if stmt.alias else stmt.name
                scope.define(bind, None)
            return

        if isinstance(stmt, ast.AssignStmt):
            if not scope.contains(stmt.name):
                # First assignment to a new name declares it (type inferred from the value).
                value_type = self._analyze_expr(stmt.value, scope, as_callee=False)
                stmt.declares = True
                scope.define(stmt.name, normalize_type_name(value_type))
                return
            current = scope.lookup(stmt.name)
            declared = scope.annotation_of(stmt.name)
            if declared and is_pointer_type(normalize_type_name(declared)):
                stmt.pointer_type = declared  # the runtime converts void*/arrays to this pointer type
            if scope.is_const(stmt.name):
                self._add_error("E2064", f"cannot assign to constant '{stmt.name}'", stmt.span.line, stmt.span.column)
            value_type = self._analyze_expr(stmt.value, scope, as_callee=False)
            if current and value_type and not self._type_compatible(current, value_type):
                self._add_error("E2004", f"type mismatch in assignment to '{stmt.name}': expected {current}, got {value_type}", stmt.span.line, stmt.span.column)
            return

        if isinstance(stmt, ast.DestructureStmt):
            self._analyze_expr(stmt.value, scope, as_callee=False)
            seen: set[str] = set()
            for target in stmt.targets:
                if target.name in seen:
                    self._add_error("E2007", f"'{target.name}' appears twice in the same unpacking", target.span.line, target.span.column)
                seen.add(target.name)
                if stmt.always_declare or not scope.contains(target.name):
                    target.declares = True
                    scope.define(target.name, None)
                elif scope.is_const(target.name):
                    self._add_error("E2064", f"cannot assign to constant '{target.name}'", target.span.line, target.span.column)
            return

        if isinstance(stmt, ast.SetDerefStmt):
            ptr_type = self._analyze_expr(stmt.target, scope, as_callee=False)
            value_type = self._analyze_expr(stmt.value, scope, as_callee=False)
            self._check_store_through(ptr_type, value_type, stmt)
            return

        if isinstance(stmt, ast.SetAttrStmt):
            self._check_not_const_target(stmt.target, scope, stmt)
            self._analyze_expr(stmt.target, scope, as_callee=False)
            if isinstance(stmt.target, ast.IdentifierExpr) and stmt.target.name in self.type_attrs:
                if stmt.attr in self.type_attrs.get(stmt.target.name, {}):
                    self._add_error("E2038", f"type attribute '{stmt.target.name}.{stmt.attr}' is immutable", stmt.span.line, stmt.span.column)
            self._analyze_expr(stmt.value, scope, as_callee=False)
            return

        if isinstance(stmt, ast.SetIndexStmt):
            target_type = self._analyze_expr(stmt.target, scope, as_callee=False)
            index_type = self._analyze_expr(stmt.index, scope, as_callee=False)
            value_type = self._analyze_expr(stmt.value, scope, as_callee=False)
            if is_pointer_type(target_type):
                self._check_store_through(target_type, value_type, stmt)  # writes the pointee, not the name
                return
            self._check_not_const_target(stmt.target, scope, stmt)
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
            else_scope = Scope(parent=scope)
            for inner in stmt.else_body:
                self._analyze_stmt(inner, else_scope, fn_decl, loop_depth=loop_depth, except_depth=except_depth)
            return

        if isinstance(stmt, ast.ForStmt):
            self._analyze_expr(stmt.iterable, scope, as_callee=False)
            body_scope = Scope(parent=scope)
            # Range loops (a..b) produce integer indices; array/string
            # iteration produces opaque elements typed as None (any).
            loop_var_type = "i32" if isinstance(stmt.iterable, ast.RangeExpr) else None
            body_scope.define(stmt.var_name, loop_var_type)
            for inner in stmt.body:
                self._analyze_stmt(inner, body_scope, fn_decl, loop_depth=loop_depth + 1, except_depth=except_depth)
            for inner in stmt.else_body if hasattr(stmt, "else_body") else []:
                self._analyze_stmt(inner, body_scope, fn_decl, loop_depth=loop_depth, except_depth=except_depth)
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

    def _check_not_const_target(self, target: object, scope: Scope, stmt: object) -> None:
        """`C[i] = v` and `C.f = v` change the value behind the name, so a constant `C` forbids them."""
        if isinstance(target, ast.IdentifierExpr) and scope.is_const(target.name):
            self._add_error(
                "E2065",
                f"cannot modify constant '{target.name}'",
                stmt.span.line,
                stmt.span.column,
            )

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

        if intrinsic_name == "io_print":
            # print takes any number of values and writes them separated by spaces.
            for arg in args:
                self._analyze_expr(arg, scope, as_callee=False)
            return None

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
        if callee.name in self.functions or callee.name in self.types or callee.name in self.extern_fns:
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
                    or expr.name in self.extern_fns
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

        if isinstance(expr, ast.AddrOfExpr):
            return self._analyze_address_of(expr, scope)

        if isinstance(expr, ast.DerefExpr):
            ptr_type = self._analyze_expr(expr.operand, scope, as_callee=False)
            return self._deref_type(ptr_type, expr)

        if isinstance(expr, ast.CastExpr):
            source = self._analyze_expr(expr.operand, scope, as_callee=False)
            return self._check_cast(source, expr)

        if isinstance(expr, ast.SizeofExpr):
            if expr.operand is not None and not (expr.type_name is not None and self._names_a_type(expr.type_name, scope)):
                self._analyze_expr(expr.operand, scope, as_callee=False)
                if isinstance(expr.operand, ast.IdentifierExpr):
                    declared = scope.annotation_of(expr.operand.name)
                    if declared and not declared.endswith("]"):
                        expr.type_name, expr.operand = declared, None  # sizeof(x) is the size of x's declared type
            return "int"

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
            if (is_pointer_type(left) or is_pointer_type(right) or left == "null" or right == "null") and expr.op != "in":
                return self._pointer_binary(expr, left, right)
            if expr.op in {"==", "!=", "<", "<=", ">", ">=", "in"}:
                return "bool"
            if left and right and left != right and left in {"int", "float"} and right in {"int", "float"}:
                return "f32"
            return left or right

        if isinstance(expr, ast.CondExpr):
            self._analyze_expr(expr.condition, scope, as_callee=False)
            then_type = self._analyze_expr(expr.then_expr, scope, as_callee=False)
            else_type = self._analyze_expr(expr.else_expr, scope, as_callee=False)
            return then_type if then_type == else_type else None

        if isinstance(expr, ast.RangeExpr):
            start_type = None if expr.start is None else self._analyze_expr(expr.start, scope, as_callee=False)
            stop_type = None if expr.stop is None else self._analyze_expr(expr.stop, scope, as_callee=False)
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
            if (
                isinstance(expr.callee, ast.IdentifierExpr)
                and expr.callee.name in POINTER_BUILTINS
                and expr.callee.name not in self.functions
                and expr.callee.name not in self.extern_fns
                and not scope.contains(expr.callee.name)
            ):
                low, high, result = POINTER_BUILTINS[expr.callee.name]
                if not low <= len(expr.args) <= high:
                    wanted = str(low) if low == high else f"{low} to {high}"
                    self._add_error("E2066", f"'{expr.callee.name}' takes {wanted} argument(s), got {len(expr.args)}", expr.span.line, expr.span.column)
                return result
            for _, value in expr.kwargs:
                self._analyze_expr(value, scope, as_callee=False)
            if isinstance(expr.callee, ast.IdentifierExpr):
                fn = self.functions.get(expr.callee.name)
                if fn and not scope.contains(expr.callee.name):
                    self._check_call_arguments(expr, fn)
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

        if isinstance(expr, ast.TupleExpr):
            for e in expr.elements:
                self._analyze_expr(e, scope, as_callee=False)
            return "tuple"

        if isinstance(expr, ast.FStringExpr):
            for part in expr.parts:
                if not isinstance(part, str):
                    self._analyze_expr(part[0], scope, as_callee=False)
            return "str"

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

    # ------------------------------------------------------------------ pointers
    def _names_a_type(self, name: str, scope: Scope) -> bool:
        """A lone name in `sizeof(name)` is a type when it is a builtin or struct and no variable hides it."""
        if scope.contains(name):
            return False
        return name in SCALARS or name in self.structs or name == "void"

    def _same_pointee(self, a: str, b: str) -> bool:
        a, b = _POINTEE_ALIASES.get(a, a), _POINTEE_ALIASES.get(b, b)
        if is_pointer_type(a) and is_pointer_type(b):
            return self._same_pointee(pointee_of(a), pointee_of(b)) or pointee_of(a) == "void" or pointee_of(b) == "void"
        return a == b

    def _pointer_compatible(self, expected: str, actual: str) -> bool:
        if actual == "null":
            return is_pointer_type(expected)
        if is_pointer_type(expected) and actual.startswith("array[") and actual.endswith("]"):
            # a C array decays to a pointer to its element type (a managed array is rejected when it runs)
            ep = pointee_of(expected)
            return ep == "void" or self._same_pointee(ep, actual[len("array["):-1])
        if expected == "null" or not is_pointer_type(expected) or not is_pointer_type(actual):
            return False
        if actual.startswith("const ") and not expected.startswith("const "):
            return False  # would drop const
        ep, ap = pointee_of(expected), pointee_of(actual)
        return ep == "void" or ap == "void" or self._same_pointee(ep, ap)

    def _deref_type(self, ptr_type: str | None, expr: Any) -> str | None:
        if ptr_type is None or ptr_type == "null":
            if ptr_type == "null":
                self._add_error("E2070", "cannot dereference null", expr.span.line, expr.span.column)
            return None
        if is_pointer_type(ptr_type):
            pointee = pointee_of(ptr_type)
            if pointee == "void":
                self._add_error("E2070", "cannot dereference a void*; cast it to a typed pointer first", expr.span.line, expr.span.column)
                return None
            return pointee
        if ptr_type.startswith("array["):
            return None  # a C array decays to a pointer
        if ptr_type in PRIMITIVE_TYPES or ptr_type in {"i32", "f32"}:
            self._add_error("E2070", f"cannot dereference a value of type {ptr_type}; it is not a pointer", expr.span.line, expr.span.column)
        return None

    def _analyze_address_of(self, expr: ast.AddrOfExpr, scope: Scope) -> str | None:
        operand = expr.operand
        line, col = expr.span.line, expr.span.column
        if isinstance(operand, ast.IdentifierExpr):
            inner = self._analyze_expr(operand, scope, as_callee=False)
            if inner in {"str", "map", "tuple", "fn"} or inner == "array":
                self._add_error("E2074", f"cannot take the address of '{operand.name}': it is a managed {inner} value, not raw memory", line, col)
                return None
            declared = scope.annotation_of(operand.name)
            expr.pointee = declared
            if declared is None or declared.endswith("]"):
                return None
            const = "const " if scope.is_const(operand.name) else ""
            return canonical_pointer_type(f"{const}{declared}*")
        if isinstance(operand, ast.IndexExpr):
            base = self._analyze_expr(operand.value, scope, as_callee=False)
            self._analyze_expr(operand.index, scope, as_callee=False)
            if base in {"str", "map", "tuple", "array"}:
                self._add_error("E2074", f"cannot take the address of an element of a managed {base}", line, col)
                return None
            return base if is_pointer_type(base) else None
        if isinstance(operand, ast.AttributeExpr):
            self._analyze_expr(operand, scope, as_callee=False)
            return None
        if isinstance(operand, ast.DerefExpr):
            return self._analyze_expr(operand.operand, scope, as_callee=False)
        self._analyze_expr(operand, scope, as_callee=False)
        self._add_error("E2071", "cannot take the address of this expression: it is not a variable, element or field", line, col)
        return None

    def _check_cast(self, source: str | None, expr: ast.CastExpr) -> str | None:
        target = normalize_type_name(expr.type_name) or expr.type_name
        base = target.replace("const ", "").strip()
        if not (is_pointer_type(target) or base in SCALARS or base in {"i32", "f32", "void"}):
            self._add_error("E2073", f"cannot cast to '{expr.type_name}'", expr.span.line, expr.span.column)
            return None
        if source is None:
            return target
        if is_pointer_type(target) and (source in _FLOAT_FAMILY or source in {"f32", "str", "map", "tuple"}):
            self._add_error("E2073", f"cannot cast {source} to {expr.type_name}", expr.span.line, expr.span.column)
        elif is_pointer_type(source) and (base in _FLOAT_FAMILY or base == "f32"):
            self._add_error("E2073", f"cannot cast {source} to {expr.type_name}", expr.span.line, expr.span.column)
        return target

    def _pointer_binary(self, expr: ast.BinaryExpr, left: str | None, right: str | None) -> str | None:
        op = expr.op
        line, col = expr.span.line, expr.span.column
        lp, rp = is_pointer_type(left) or left == "null", is_pointer_type(right) or right == "null"
        if op in {"==", "!=", "<", "<=", ">", ">="}:
            if lp and rp:
                return "bool"
            if (lp and right is not None) or (rp and left is not None):
                self._add_error("E2073", f"cannot compare a pointer with {right if lp else left}; compare with another pointer or null", line, col)
            return "bool"
        if op == "+":
            if lp and rp:
                self._add_error("E2073", "cannot add two pointers", line, col)
                return None
            return left if lp else right
        if op == "-":
            if lp and rp:
                if is_pointer_type(left) and is_pointer_type(right) and not self._same_pointee(pointee_of(left), pointee_of(right)):
                    self._add_error("E2073", f"cannot subtract pointers to different types ({left} and {right})", line, col)
                return "int"
            if rp:
                self._add_error("E2073", "cannot subtract a pointer from a number", line, col)
                return None
            return left
        self._add_error("E2073", f"operator '{op}' is not defined for pointers", line, col)
        return None

    def _type_compatible(self, expected: str, actual: str) -> bool:
        expected = normalize_type_name(expected) or expected
        actual = normalize_type_name(actual) or actual
        if is_pointer_type(expected) or is_pointer_type(actual) or expected == "null" or actual == "null":
            return self._pointer_compatible(expected, actual)
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
