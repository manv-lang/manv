"""Reference AST interpreter for ManV semantics.

Why this file exists:
- It is the semantic authority for language behavior (classes, exceptions,
  imports, intrinsics, and control flow).
- Compiled mode is required to match this observable behavior.
- It also serves as the reference implementation for new runtime features
  before lower-level execution paths are optimized.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, TextIO

from . import ast
from .diagnostics import ManvError, diag
from .gpu_dispatch import backend_selection_report
from .lexer import Lexer
from .object_runtime import (
    BoundMethodObject,
    ExceptionObject,
    Heap,
    InstanceObject,
    ModuleObject,
    OutOfMemoryError,
    SuperProxy,
    TypeObject,
)
from .semantics import (
    accessor_kind,
    accessor_property_name,
    has_classmethod_decorator,
    has_static_method_decorator,
    normalize_gpu_decorator,
    normalize_type_name,
)
from .parser import Parser
from .intrinsics import (
    IntrinsicCallable,
    IntrinsicNamespace,
    StdCallable,
    StdNamespace,
    invoke_intrinsic,
    resolve_call_alias_name,
    std_namespace_attr,
)
from .runtime import unsupported_feature
from .semantics import SemanticAnalyzer
from .semantics_core import eval_binary, eval_unary
from .builtin_methods import BuiltinMethods


class ReturnSignal(Exception):
    def __init__(self, value: object):
        self.value = value


class BreakSignal(Exception):
    pass


class ContinueSignal(Exception):
    pass


class RaiseSignal(Exception):
    def __init__(self, error: ExceptionObject):
        self.error = error


@dataclass
class FunctionValue:
    decl: ast.FnDecl
    owner_type: TypeObject | None = None
    globals_env: "Environment | None" = None
    is_static_method: bool = False
    is_class_method: bool = False


class Environment:
    def __init__(self, parent: "Environment | None" = None):
        self.parent = parent
        self.values: dict[str, object] = {}

    def define(self, name: str, value: object) -> None:
        self.values[name] = value

    def lookup(self, name: str) -> object:
        if name in self.values:
            return self.values[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise KeyError(name)

    def assign(self, name: str, value: object) -> None:
        if name in self.values:
            self.values[name] = value
            return
        if self.parent is not None:
            self.parent.assign(name, value)
            return
        raise KeyError(name)


class Interpreter:
    """AST interpreter used as the baseline semantics engine."""

    def __init__(
        self,
        file: str,
        stdout: TextIO | None = None,
        *,
        deterministic_gc: bool = False,
        gc_stress: bool = False,
        stable_debug_format: bool = False,
        heap_max_objects: int = 0,
    ):
        self.file = file
        self.stdout = stdout or io.StringIO()
        self.global_env = Environment()
        self.functions: dict[str, FunctionValue] = {}
        self._macros: dict[str, ast.MacroDeclStub] = {}
        self.types: dict[str, TypeObject] = {}
        self.type_constants: dict[str, set[str]] = {}
        self.call_stack: list[dict[str, Any]] = []
        self.active_exceptions: list[ExceptionObject] = []
        self.stable_debug_format = stable_debug_format
        self.module_cache: dict[str, ModuleObject] = {}
        self._loading_modules: set[str] = set()
        # Tracks active module execution chain for relative import resolution.
        self._module_exec_stack: list[str] = []
        # Tracks canonical modules loaded from package `__init__.mv`.
        self._package_modules: set[str] = set()
        self.source_root = Path(file).resolve().parent
        # Deterministic module search order:
        # 1) current project source root
        # 2) MANV_PATH roots
        # 3) bundled compiler-shipped ManV std source
        self.module_search_roots: list[Path] = self._build_module_search_roots()

        self.heap = Heap(deterministic_gc=deterministic_gc, gc_stress=gc_stress, max_objects=heap_max_objects)
        self.heap.register_root_provider(self._gc_roots)
        self._bootstrap_runtime_types()
        self._builtin_methods = BuiltinMethods(self)

    def _bootstrap_runtime_types(self) -> None:
        object_t = self.heap.allocate("Type", TypeObject(name="object"))
        type_t = self.heap.allocate("Type", TypeObject(name="type", base=object_t))
        object_t.base = None
        object_t.mro = ["object"]
        type_t.mro = ["type", "object"]

        self.types["object"] = object_t
        self.types["type"] = type_t

        for primitive in ("int", "float", "bool", "str", "array", "map", "none"):
            self._define_type(primitive, "object")

        exc_base = self._define_type("BaseException", "object")
        exc = self._define_type("Exception", "BaseException")
        self._define_type("StopIteration", "Exception")
        self._define_type("ImportError", "Exception")
        self._define_type("TypeError", "Exception")
        self._define_type("AttributeError", "Exception")
        self._define_type("KeyError", "Exception")
        self._define_type("IndexError", "Exception")
        self._define_type("ValueError", "Exception")
        self._define_type("OverflowError", "Exception")
        self._define_type("RuntimeError", "Exception")
        self._define_type("OSError", "Exception")
        self._define_type("OutOfMemoryError", "RuntimeError")

        # expose runtime types as globals so programs can raise/catch them by name
        for name, t in self.types.items():
            self.global_env.define(name, t)

        del exc_base, exc

    def _define_type(self, name: str, base_name: str) -> TypeObject:
        base = self.types.get(base_name)
        if base is None:
            base = self.types["object"]
        obj = self.heap.allocate("Type", TypeObject(name=name, base=base, mro=[name, *base.mro]))
        self.types[name] = obj
        return obj

    def _c3_mro(self, type_obj: TypeObject, base_objects: list[TypeObject]) -> list[str]:
        """C3 linearization algorithm for MRO computation."""
        if not base_objects:
            return [type_obj.name, "object"]
        # C3: merge([type] + [mro(B) for B in bases] + [bases])
        sequences = [[type_obj.name]] + [list(b.mro) for b in base_objects] + [[b.name for b in base_objects]]
        result: list[str] = []
        while sequences:
            # Find a good head: not in tail of any other sequence
            head = None
            for seq in sequences:
                if not seq:
                    continue
                candidate = seq[0]
                if all(candidate not in s[1:] for s in sequences):
                    head = candidate
                    break
            if head is None:
                # Inconsistent MRO — fall back to simple order
                break
            result.append(head)
            sequences = [s[1:] if s and s[0] == head else s for s in sequences]
            sequences = [s for s in sequences if s]
        return result

    def _runtime_function_value(self, method: ast.FnDecl, owner_type: TypeObject) -> FunctionValue:
        return FunctionValue(
            decl=method,
            owner_type=owner_type,
            globals_env=self.global_env,
            is_static_method=has_static_method_decorator(method),
            is_class_method=has_classmethod_decorator(method),
        )

    def _register_type_method(self, type_obj: TypeObject, method: ast.FnDecl) -> None:
        """Attach one source method to the runtime type object.

        Accessors are stored in dedicated getter/setter tables instead of the
        ordinary method table. That separation is what lets a getter and setter
        share the same public property name while still leaving ordinary method
        dispatch rules unchanged.
        """

        fn_val = self._runtime_function_value(method, type_obj)
        accessor = accessor_kind(method)
        if accessor in {"getter", "setter"}:
            property_name = accessor_property_name(method) or method.name
            if accessor == "getter":
                type_obj.getters[property_name] = fn_val
            else:
                type_obj.setters[property_name] = fn_val
            return

        key = "__init__" if method.name == "init" else method.name
        type_obj.methods[key] = fn_val
        if method.name == "init":
            type_obj.methods["init"] = fn_val

    def _gc_roots(self) -> list[Any]:
        roots: list[Any] = [self.global_env.values, self.types, self.active_exceptions, self.module_cache]
        for frame in self.call_stack:
            roots.append(frame.get("env"))
        return roots

    def load_program(self, program: ast.Program) -> None:
        # First register all type names so inheritance can resolve in second pass.
        for decl in program.declarations:
            if isinstance(decl, ast.TypeDecl):
                primary_base = decl.base_names[0] if decl.base_names else "object"
                if decl.name not in self.types:
                    self._define_type(decl.name, primary_base)

        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                fn_val = FunctionValue(decl=decl, globals_env=self.global_env)
                self.functions[decl.name] = fn_val
                self.global_env.define(decl.name, fn_val)
            elif isinstance(decl, ast.TypeDecl):
                type_obj = self.types[decl.name]
                if decl.base_names:
                    primary_base = decl.base_names[0]
                    if primary_base in self.types:
                        type_obj.base = self.types[primary_base]
                    # Build MRO with all bases
                    base_objects = [self.types[b] for b in decl.base_names if b in self.types]
                    type_obj.mro = self._c3_mro(type_obj, base_objects)
                type_obj.docstring = decl.docstring
                if decl.attrs:
                    self.type_constants.setdefault(type_obj.name, set()).update(attr.name for attr in decl.attrs)
                    for attr in decl.attrs:
                        type_obj.attrs[attr.name] = self.eval_expr(attr.value, self.global_env)
                for method in decl.methods:
                    self._register_type_method(type_obj, method)
            elif isinstance(decl, ast.ImplDecl):
                type_obj = self.types.get(decl.target)
                if type_obj is None:
                    self._raise_runtime("TypeError", f"impl target '{decl.target}' is undefined", decl.span)
                for method in decl.methods:
                    self._register_type_method(type_obj, method)
            elif isinstance(decl, ast.MacroDeclStub):
                self._macros[decl.name] = decl
            elif isinstance(decl, ast.ModuleDecl):
                # Module declarations are metadata; record on the module object if available.
                if self._module_exec_stack:
                    canonical = self._module_exec_stack[-1]
                    mod_obj = self.module_cache.get(canonical)
                    if mod_obj is not None:
                        mod_obj.declared_name = decl.name
            elif isinstance(decl, ast.ExternBlock):
                for fn_decl in decl.fns:
                    cfn = self._make_extern_callable(fn_decl)
                    self.global_env.define(fn_decl.name, cfn)
            else:
                raise unsupported_feature(type(decl).__name__, self.file, 1, 1)

        for name, type_obj in self.types.items():
            self.global_env.define(name, type_obj)

        for stmt in program.statements:
            self.execute_stmt(stmt, self.global_env)

    def run_main(self, program: ast.Program) -> int:
        self.load_program(program)
        if "main" not in self.functions:
            raise ManvError(diag("E3001", "missing 'main' function", self.file, 1, 1))
        result = self._call_function(self.functions["main"], [])
        if result is None:
            return 0
        if isinstance(result, bool):
            return int(result)
        if isinstance(result, int):
            return result
        raise ManvError(diag("E3002", "main must return int-compatible value", self.file, 1, 1))

    def execute_stmt(self, stmt: object, env: Environment, in_loop: bool = False) -> object | None:
        if isinstance(stmt, ast.LetStmt):
            if stmt.array_size is not None:
                raw_size = self.eval_expr(stmt.array_size, env)
                if not isinstance(raw_size, int) or raw_size < 0:
                    self._raise_runtime("ValueError", f"array size for '{stmt.name}' must be a non-negative integer", stmt.span)
                if stmt.value is None:
                    value = [None] * raw_size
                else:
                    seed = self.eval_expr(stmt.value, env)
                    if not isinstance(seed, list):
                        self._raise_runtime("TypeError", f"array initializer for '{stmt.name}' must be an array literal", stmt.span)
                    if len(seed) > raw_size:
                        self._raise_runtime("ValueError", f"initializer for '{stmt.name}' has {len(seed)} items, exceeds size {raw_size}", stmt.span)
                    value = seed + [None] * (raw_size - len(seed))
            else:
                value = self.eval_expr(stmt.value, env) if stmt.value is not None else None
            env.define(stmt.name, value)
            return None

        if isinstance(stmt, ast.FnDecl):
            # Local function declaration — creates a closure capturing the current env.
            fn_val = FunctionValue(decl=stmt, globals_env=env)
            env.define(stmt.name, fn_val)
            return None

        if isinstance(stmt, ast.ExternBlock):
            for fn_decl in stmt.fns:
                cfn = self._make_extern_callable(fn_decl)
                env.define(fn_decl.name, cfn)
            return None

        if isinstance(stmt, ast.ModuleDecl):
            # Store the canonical module name on the currently-loading ModuleObject
            # so the runtime and tooling can surface it without reparsing the file.
            if self._module_exec_stack:
                canonical = self._module_exec_stack[-1]
                mod_obj = self.module_cache.get(canonical)
                if mod_obj is not None:
                    mod_obj.declared_name = stmt.name
            return None

        if isinstance(stmt, ast.IncludeStmt):
            module = self._import_module(stmt.module, stmt.span, level=stmt.level)
            if stmt.name is None:
                # Whole-module include: bind under alias or leaf of dotted path.
                bind = stmt.alias if stmt.alias else stmt.module.split(".")[-1]
                env.define(bind, module)
            else:
                # Specific-name include: resolve symbol then bind.
                if stmt.name not in module.exports:
                    self._raise_runtime(
                        "ImportError",
                        f"cannot include '{stmt.name}' from '{stmt.module}'",
                        stmt.span,
                    )
                bind = stmt.alias if stmt.alias else stmt.name
                env.define(bind, module.exports[stmt.name])
            return None

        if isinstance(stmt, ast.AssignStmt):
            value = self.eval_expr(stmt.value, env)
            if stmt.declares:
                env.define(stmt.name, value)
                return None
            try:
                env.assign(stmt.name, value)
            except KeyError:
                self._raise_runtime("RuntimeError", f"undefined variable '{stmt.name}'", stmt.span)
            return None

        if isinstance(stmt, ast.AugAssignStmt):
            try:
                current = env.lookup(stmt.name)
            except KeyError:
                self._raise_runtime("RuntimeError", f"undefined variable '{stmt.name}'", stmt.span)
            rhs = self.eval_expr(stmt.value, env)
            op = stmt.op[:-1]  # strip '=' to get '+', '-', '*', '/', '%'
            if isinstance(current, InstanceObject):
                DUNDER_MAP = {"+": ("__add__", "add"), "-": ("__sub__", "sub"), "*": ("__mul__", "mul"),
                              "/": ("__truediv__", "truediv"), "%": ("__mod__", "mod"), "**": ("__pow__", "pow")}
                if op in DUNDER_MAP:
                    dunder, alias = DUNDER_MAP[op]
                    result, found = self._try_dunder(current, dunder, [rhs])
                    if not found:
                        result, found = self._try_dunder(current, alias, [rhs])
                    if found:
                        try:
                            env.assign(stmt.name, result)
                        except KeyError:
                            self._raise_runtime("RuntimeError", f"undefined variable '{stmt.name}'", stmt.span)
                        return None
            result = eval_binary(op, current, rhs)
            try:
                env.assign(stmt.name, result)
            except KeyError:
                self._raise_runtime("RuntimeError", f"undefined variable '{stmt.name}'", stmt.span)
            return None

        if isinstance(stmt, ast.AugAssignAttrStmt):
            target = self.eval_expr(stmt.target, env)
            current = self._lookup_attr(target, stmt.attr, stmt.span)
            rhs = self.eval_expr(stmt.value, env)
            op = stmt.op[:-1]
            result = eval_binary(op, current, rhs)
            self._store_attr(target, stmt.attr, result, stmt.span)
            return None

        if isinstance(stmt, ast.AugAssignIndexStmt):
            target = self.eval_expr(stmt.target, env)
            index = self.eval_expr(stmt.index, env)
            if isinstance(target, list):
                idx = self._normalize_list_index(target, index, stmt.span)
                current = target[idx]
                rhs = self.eval_expr(stmt.value, env)
                op = stmt.op[:-1]
                result = eval_binary(op, current, rhs)
                target[idx] = result
            elif isinstance(target, dict):
                current = target.get(index)
                rhs = self.eval_expr(stmt.value, env)
                op = stmt.op[:-1]
                result = eval_binary(op, current, rhs)
                target[index] = result
            elif isinstance(target, InstanceObject):
                # Get current value via __getitem__, apply op, set via __setitem__
                current, found = self._try_dunder(target, "__getitem__", [index])
                if not found:
                    current, found = self._try_dunder(target, "getitem", [index])
                if not found:
                    self._raise_runtime("TypeError", "augmented index assignment target must be array or map", stmt.span)
                rhs = self.eval_expr(stmt.value, env)
                op = stmt.op[:-1]
                DUNDER_MAP = {"+": ("__add__", "add"), "-": ("__sub__", "sub"), "*": ("__mul__", "mul"),
                              "/": ("__truediv__", "truediv"), "%": ("__mod__", "mod"), "**": ("__pow__", "pow")}
                if isinstance(current, InstanceObject) and op in DUNDER_MAP:
                    dunder, alias = DUNDER_MAP[op]
                    result, dfound = self._try_dunder(current, dunder, [rhs])
                    if not dfound:
                        result, dfound = self._try_dunder(current, alias, [rhs])
                    if not dfound:
                        result = eval_binary(op, current, rhs)
                else:
                    result = eval_binary(op, current, rhs)
                _, sfound = self._try_dunder(target, "__setitem__", [index, result])
                if not sfound:
                    _, sfound = self._try_dunder(target, "setitem", [index, result])
                if not sfound:
                    self._raise_runtime("TypeError", "augmented index assignment target must be array or map", stmt.span)
            else:
                self._raise_runtime("TypeError", "augmented index assignment target must be array or map", stmt.span)
            return None

        if isinstance(stmt, ast.SetAttrStmt):
            target = self.eval_expr(stmt.target, env)
            value = self.eval_expr(stmt.value, env)
            self._store_attr(target, stmt.attr, value, stmt.span)
            return None

        if isinstance(stmt, ast.SetIndexStmt):
            target = self.eval_expr(stmt.target, env)
            index = self.eval_expr(stmt.index, env)
            value = self.eval_expr(stmt.value, env)
            if isinstance(target, list):
                idx = self._normalize_list_index(target, index, stmt.span)
                target[idx] = value
                return None
            if isinstance(target, dict):
                self._assert_hashable(index, stmt.span)
                target[index] = value
                return None
            if isinstance(target, InstanceObject):
                _, found = self._try_dunder(target, "__setitem__", [index, value])
                if not found:
                    _, found = self._try_dunder(target, "setitem", [index, value])
                if found:
                    return None
            self._raise_runtime("TypeError", "index assignment target must be array or map", stmt.span)

        if isinstance(stmt, ast.ReturnStmt):
            value = self.eval_expr(stmt.value, env) if stmt.value is not None else None
            raise ReturnSignal(value)

        if isinstance(stmt, ast.RaiseStmt):
            if stmt.value is None:
                if not self.active_exceptions:
                    self._raise_runtime("RuntimeError", "bare raise outside except", stmt.span)
                raise RaiseSignal(self.active_exceptions[-1])
            value = self.eval_expr(stmt.value, env)
            self._raise_value(value, stmt.span)

        if isinstance(stmt, ast.SyscallStmt):
            self.eval_expr(ast.SyscallExpr(target=stmt.target, args=stmt.args, span=stmt.span), env)
            return None

        if isinstance(stmt, ast.TryStmt):
            self._execute_try(stmt, env, in_loop=in_loop)
            return None

        if isinstance(stmt, ast.IfStmt):
            branch = stmt.then_body if self.eval_expr(stmt.condition, env) else stmt.else_body
            branch_env = Environment(parent=env)
            for inner in branch:
                self.execute_stmt(inner, branch_env, in_loop=in_loop)
            return None

        if isinstance(stmt, ast.WhileStmt):
            broke = False
            while self.eval_expr(stmt.condition, env):
                loop_env = Environment(parent=env)
                try:
                    for inner in stmt.body:
                        self.execute_stmt(inner, loop_env, in_loop=True)
                except ContinueSignal:
                    continue
                except BreakSignal:
                    broke = True
                    break
            if not broke and stmt.else_body:
                else_env = Environment(parent=env)
                for inner in stmt.else_body:
                    self.execute_stmt(inner, else_env, in_loop=in_loop)
            return None

        if isinstance(stmt, ast.ForStmt):
            if isinstance(stmt.iterable, ast.RangeExpr):
                raw_start = self.eval_expr(stmt.iterable.start, env)
                raw_stop = self.eval_expr(stmt.iterable.stop, env)
                if not isinstance(raw_start, int) or not isinstance(raw_stop, int):
                    self._raise_runtime("TypeError", "range bounds must be integers", stmt.span)
                items: list[object] = list(range(raw_start, raw_stop))
                broke = False
                for item in items:
                    loop_env = Environment(parent=env)
                    loop_env.define(stmt.var_name, item)
                    try:
                        for inner in stmt.body:
                            self.execute_stmt(inner, loop_env, in_loop=True)
                    except ContinueSignal:
                        continue
                    except BreakSignal:
                        broke = True
                        break
                if not broke and stmt.else_body:
                    else_env = Environment(parent=env)
                    for inner in stmt.else_body:
                        self.execute_stmt(inner, else_env, in_loop=in_loop)
                return None
            else:
                iterable = self.eval_expr(stmt.iterable, env)
                if isinstance(iterable, str):
                    iterable = list(iterable)
                elif isinstance(iterable, dict):
                    iterable = list(iterable.keys())
                if isinstance(iterable, list):
                    items = list(iterable)
                    broke = False
                    for item in items:
                        loop_env = Environment(parent=env)
                        loop_env.define(stmt.var_name, item)
                        try:
                            for inner in stmt.body:
                                self.execute_stmt(inner, loop_env, in_loop=True)
                        except ContinueSignal:
                            continue
                        except BreakSignal:
                            broke = True
                            break
                    if not broke and stmt.else_body:
                        else_env = Environment(parent=env)
                        for inner in stmt.else_body:
                            self.execute_stmt(inner, else_env, in_loop=in_loop)
                    return None
                elif isinstance(iterable, InstanceObject):
                    # Use __iter__ / __next__ protocol
                    iter_method = self._lookup_method(iterable.type_obj, "__iter__") or self._lookup_method(iterable.type_obj, "iter_")
                    if iter_method is not None:
                        iterator = self._call_function(iter_method, [iterable])
                    else:
                        iterator = iterable  # object IS the iterator
                    if not isinstance(iterator, InstanceObject):
                        self._raise_runtime("TypeError", "__iter__ must return an object", stmt.span)
                    next_method = self._lookup_method(iterator.type_obj, "__next__") or self._lookup_method(iterator.type_obj, "next_")
                    if next_method is None:
                        self._raise_runtime("TypeError", f"'{iterator.type_obj.name}' has no __next__ method", stmt.span)
                    stop_iter_type = self.types.get("StopIteration")
                    broke = False
                    guard = 0
                    while True:
                        guard += 1
                        if guard > 1_000_000:
                            self._raise_runtime("RuntimeError", "iterator guard exceeded", stmt.span)
                        try:
                            item = self._call_function(next_method, [iterator])
                        except RaiseSignal as rs:
                            if stop_iter_type is not None and self._is_type_or_subtype(rs.error.type_obj, stop_iter_type):
                                break
                            raise
                        loop_env = Environment(parent=env)
                        loop_env.define(stmt.var_name, item)
                        try:
                            for inner in stmt.body:
                                self.execute_stmt(inner, loop_env, in_loop=True)
                        except ContinueSignal:
                            continue
                        except BreakSignal:
                            broke = True
                            break
                    if not broke and stmt.else_body:
                        else_env = Environment(parent=env)
                        for inner in stmt.else_body:
                            self.execute_stmt(inner, else_env, in_loop=in_loop)
                    return None
                else:
                    self._raise_runtime("TypeError", "for loop iterable must be an array, range, or iterable object", stmt.span)

        if isinstance(stmt, ast.ExprStmt):
            return self.eval_expr(stmt.expr, env)

        if isinstance(stmt, ast.BreakStmt):
            if not in_loop:
                self._raise_runtime("RuntimeError", "'break' used outside loop", stmt.span)
            raise BreakSignal()

        if isinstance(stmt, ast.ContinueStmt):
            if not in_loop:
                self._raise_runtime("RuntimeError", "'continue' used outside loop", stmt.span)
            raise ContinueSignal()

        if isinstance(stmt, ast.WithStmt):
            ctx_value = self.eval_expr(stmt.context, env)
            enter_val = None
            if isinstance(ctx_value, InstanceObject):
                enter_method = self._lookup_method(ctx_value.type_obj, "__enter__")
                if enter_method is not None:
                    enter_val = self._call_function(enter_method, [ctx_value])
            with_env = Environment(parent=env)
            if stmt.bind_name is not None:
                with_env.define(stmt.bind_name, enter_val if enter_val is not None else ctx_value)
            exc_to_raise: RaiseSignal | None = None
            try:
                for inner in stmt.body:
                    self.execute_stmt(inner, with_env, in_loop=in_loop)
            except RaiseSignal as rs:
                exc_to_raise = rs
            finally:
                if isinstance(ctx_value, InstanceObject):
                    exit_method = self._lookup_method(ctx_value.type_obj, "__exit__")
                    if exit_method is not None:
                        if exc_to_raise is not None:
                            err = exc_to_raise.error
                            exc_type = err.type_obj
                            exc_val = err
                            exc_tb = str(err.stacktrace) if err.stacktrace else None
                            suppress = self._call_function(exit_method, [ctx_value, exc_type, exc_val, exc_tb])
                            if suppress:
                                exc_to_raise = None
                        else:
                            self._call_function(exit_method, [ctx_value, None, None, None])
            if exc_to_raise is not None:
                raise exc_to_raise
            return None

        if isinstance(stmt, ast.MatchStmt):
            subject = self.eval_expr(stmt.subject, env)
            for case in stmt.cases:
                pattern = case.pattern
                matched = False
                if isinstance(pattern, ast.IdentifierExpr) and pattern.name == "_":
                    matched = True
                elif isinstance(pattern, ast.LiteralExpr):
                    matched = (subject == pattern.value)
                elif isinstance(pattern, ast.IdentifierExpr):
                    try:
                        pat_val = env.lookup(pattern.name)
                        matched = (subject == pat_val)
                    except KeyError:
                        matched = True
                        case_env = Environment(parent=env)
                        case_env.define(pattern.name, subject)
                        if case.guard is not None:
                            if not self.eval_expr(case.guard, case_env):
                                matched = False
                        if matched:
                            for inner in case.body:
                                self.execute_stmt(inner, case_env, in_loop=in_loop)
                            return None
                        continue
                else:
                    try:
                        pat_val = self.eval_expr(pattern, env)
                        matched = (subject == pat_val)
                    except Exception:
                        matched = False
                if matched:
                    if case.guard is not None:
                        case_env = Environment(parent=env)
                        if not self.eval_expr(case.guard, case_env):
                            continue
                    case_env = Environment(parent=env)
                    for inner in case.body:
                        self.execute_stmt(inner, case_env, in_loop=in_loop)
                    return None
            return None

        if isinstance(stmt, ast.UnsupportedStmt):
            raise unsupported_feature(stmt.feature, self.file, stmt.span.line, stmt.span.column, stmt.detail)

        raise unsupported_feature(type(stmt).__name__, self.file, 1, 1)

    def _execute_try(self, stmt: ast.TryStmt, env: Environment, *, in_loop: bool) -> None:
        had_try_exception = False
        pending_raise: RaiseSignal | None = None
        pending_control: Exception | None = None

        try:
            try_env = Environment(parent=env)
            for inner in stmt.try_body:
                self.execute_stmt(inner, try_env, in_loop=in_loop)
        except RaiseSignal as rs:
            had_try_exception = True
            handled = False
            for clause in stmt.except_clauses:
                if self._matches_exception(rs.error, clause.type_name):
                    handled = True
                    clause_env = Environment(parent=env)
                    if clause.bind_name:
                        clause_env.define(clause.bind_name, rs.error)
                    self.active_exceptions.append(rs.error)
                    try:
                        for inner in clause.body:
                            self.execute_stmt(inner, clause_env, in_loop=in_loop)
                    except (ReturnSignal, BreakSignal, ContinueSignal) as ctl:
                        pending_control = ctl
                    except RaiseSignal as inner_raise:
                        pending_raise = inner_raise
                    finally:
                        self.active_exceptions.pop()
                    break
            if not handled:
                pending_raise = rs
        except (ReturnSignal, BreakSignal, ContinueSignal) as ctl:
            pending_control = ctl
        else:
            else_env = Environment(parent=env)
            for inner in stmt.else_body:
                self.execute_stmt(inner, else_env, in_loop=in_loop)

        try:
            finally_env = Environment(parent=env)
            for inner in stmt.finally_body:
                self.execute_stmt(inner, finally_env, in_loop=in_loop)
        except RaiseSignal as final_raise:
            pending_raise = final_raise
            pending_control = None
        except (ReturnSignal, BreakSignal, ContinueSignal) as final_ctl:
            pending_control = final_ctl
            pending_raise = None

        if pending_raise is not None:
            raise pending_raise
        if pending_control is not None:
            raise pending_control

        del had_try_exception

    def eval_expr(self, expr: object | None, env: Environment) -> object:
        if expr is None:
            return None

        if isinstance(expr, ast.LiteralExpr):
            return expr.value

        if isinstance(expr, ast.LambdaExpr):
            # Create an anonymous FnDecl stub so we can reuse FunctionValue.
            stub = ast.FnDecl(
                name="<lambda>",
                params=expr.params,
                return_type=expr.return_type,
                body=expr.body,
                span=expr.span,
            )
            return FunctionValue(decl=stub, globals_env=env)

        if isinstance(expr, ast.IdentifierExpr):
            if expr.name == "std":
                return StdNamespace()
            if expr.name == "__intrin":
                return IntrinsicNamespace()
            try:
                return env.lookup(expr.name)
            except KeyError:
                if expr.name in self.functions:
                    return self.functions[expr.name]
                if expr.name in self.types:
                    return self.types[expr.name]
                self._raise_runtime("RuntimeError", f"undefined variable '{expr.name}'", expr.span)

        if isinstance(expr, ast.UnaryExpr):
            value = self.eval_expr(expr.expr, env)
            if isinstance(value, InstanceObject):
                if expr.op == "-":
                    result, found = self._try_dunder(value, "__neg__", [])
                    if not found:
                        result, found = self._try_dunder(value, "neg", [])
                    if found:
                        return result
                elif expr.op == "not":
                    result, found = self._try_dunder(value, "__bool__", [])
                    if not found:
                        result, found = self._try_dunder(value, "bool_", [])
                    if found:
                        return not bool(result)
            try:
                return eval_unary(expr.op, value)
            except Exception:
                self._raise_runtime("TypeError", f"unsupported unary operator '{expr.op}'", expr.span)

        if isinstance(expr, ast.CondExpr):
            chosen = expr.then_expr if self.eval_expr(expr.condition, env) else expr.else_expr
            return self.eval_expr(chosen, env)

        if isinstance(expr, ast.BinaryExpr):
            left = self.eval_expr(expr.left, env)
            if expr.op in {"and", "&&"}:
                # The right operand is only evaluated when it can change the result.
                return bool(left) and bool(self.eval_expr(expr.right, env))
            if expr.op in {"or", "||"}:
                return bool(left) or bool(self.eval_expr(expr.right, env))
            right = self.eval_expr(expr.right, env)
            if expr.op in {"==", "!="}:
                eq = self._equals(left, right, expr.span)
                return eq if expr.op == "==" else (not eq)
            # 'in' operator dispatches to right.__contains__(left)
            if expr.op == "in":
                if isinstance(right, InstanceObject):
                    result, found = self._try_dunder(right, "__contains__", [left])
                    if not found:
                        result, found = self._try_dunder(right, "contains", [left])
                    if found:
                        return bool(result)
                try:
                    return eval_binary(expr.op, left, right)
                except Exception:
                    self._raise_runtime("TypeError", f"unsupported binary operator '{expr.op}'", expr.span)
            # All other binary operators: try dunder on left operand
            DUNDER_MAP = {
                "+": ("__add__", "add"), "-": ("__sub__", "sub"), "*": ("__mul__", "mul"),
                "/": ("__truediv__", "truediv"), "%": ("__mod__", "mod"), "**": ("__pow__", "pow"),
                "<": ("__lt__", "lt"), "<=": ("__le__", "le"), ">": ("__gt__", "gt"), ">=": ("__ge__", "ge"),
            }
            if isinstance(left, InstanceObject) and expr.op in DUNDER_MAP:
                dunder, alias = DUNDER_MAP[expr.op]
                result, found = self._try_dunder(left, dunder, [right])
                if not found:
                    result, found = self._try_dunder(left, alias, [right])
                if found:
                    return result
            try:
                return eval_binary(expr.op, left, right)
            except Exception:
                self._raise_runtime("TypeError", f"unsupported binary operator '{expr.op}'", expr.span)

        if isinstance(expr, ast.ArrayExpr):
            return [self.eval_expr(e, env) for e in expr.elements]

        if isinstance(expr, ast.MapExpr):
            out: dict[object, object] = {}
            for key, value in expr.entries:
                k = self.eval_expr(key, env)
                self._assert_hashable(k, expr.span)
                out[k] = self.eval_expr(value, env)
            return out

        if isinstance(expr, ast.SyscallExpr):
            target = self.eval_expr(expr.target, env)
            call_args = [self.eval_expr(arg, env) for arg in expr.args]
            return self._invoke_intrinsic("syscall_invoke", [target, call_args], expr.span)

        if isinstance(expr, ast.IndexExpr):
            target = self.eval_expr(expr.value, env)
            if isinstance(expr.index, ast.RangeExpr):
                return self._slice(target, expr.index, env, expr.span)
            index = self.eval_expr(expr.index, env)
            if isinstance(target, list):
                idx = self._normalize_list_index(target, index, expr.span)
                return target[idx]
            if isinstance(target, str):
                if not isinstance(index, int) or isinstance(index, bool):
                    self._raise_runtime("TypeError", "string index must be int", expr.span)
                idx = index + len(target) if index < 0 else index
                if idx < 0 or idx >= len(target):
                    self._raise_runtime("IndexError", f"string index out of range: {index}", expr.span)
                return target[idx]
            if isinstance(target, dict):
                self._assert_hashable(index, expr.span)
                if index not in target:
                    self._raise_runtime("KeyError", f"missing key: {index}", expr.span, payload=index)
                return target[index]
            if isinstance(target, InstanceObject):
                result, found = self._try_dunder(target, "__getitem__", [index])
                if not found:
                    result, found = self._try_dunder(target, "getitem", [index])
                if found:
                    return result
            self._raise_runtime("TypeError", "index target must be list or map", expr.span)

        if isinstance(expr, ast.AttributeExpr):
            base = self.eval_expr(expr.value, env)
            namespaced = std_namespace_attr(base, expr.attr)
            if namespaced is not None:
                return namespaced
            return self._lookup_attr(base, expr.attr, expr.span)

        if isinstance(expr, ast.CallExpr):
            if isinstance(expr.callee, ast.IdentifierExpr):
                name = expr.callee.name
                if name == "type":
                    if len(expr.args) != 1:
                        self._raise_runtime("TypeError", "type() expects exactly 1 argument", expr.span)
                    value = self.eval_expr(expr.args[0], env)
                    return self._runtime_type(value)
                if name == "isinstance":
                    if len(expr.args) != 2:
                        self._raise_runtime("TypeError", "isinstance() expects exactly 2 arguments", expr.span)
                    value = self.eval_expr(expr.args[0], env)
                    maybe_type = self.eval_expr(expr.args[1], env)
                    if not isinstance(maybe_type, TypeObject):
                        self._raise_runtime("TypeError", "isinstance() second arg must be a type", expr.span)
                    if isinstance(value, InstanceObject):
                        return self._is_type_or_subtype(value.type_obj, maybe_type)
                    return self._is_type_or_subtype(self._runtime_type(value), maybe_type)
                if name == "issubclass":
                    if len(expr.args) != 2:
                        self._raise_runtime("TypeError", "issubclass() expects exactly 2 arguments", expr.span)
                    child = self.eval_expr(expr.args[0], env)
                    parent = self.eval_expr(expr.args[1], env)
                    if not isinstance(child, TypeObject) or not isinstance(parent, TypeObject):
                        self._raise_runtime("TypeError", "issubclass() args must be types", expr.span)
                    return self._is_type_or_subtype(child, parent)
                if name == "id":
                    if len(expr.args) != 1:
                        self._raise_runtime("TypeError", "id() expects exactly 1 argument", expr.span)
                    value = self.eval_expr(expr.args[0], env)
                    heap_id = getattr(value, "_heap_id", None)
                    return int(heap_id) if isinstance(heap_id, int) else int(id(value))
                if name == "help":
                    if len(expr.args) != 1:
                        self._raise_runtime("TypeError", "help() expects exactly 1 argument", expr.span)
                    value = self.eval_expr(expr.args[0], env)
                    self.stdout.write(self._help_text(value) + "\n")
                    return None
                if name == "super":
                    # Find the enclosing instance and owner type from call stack
                    self_obj = None
                    owner_type_obj = None
                    for frame in reversed(self.call_stack):
                        frame_env_vals = frame.get("env", {})
                        if "self" in frame_env_vals and self_obj is None:
                            self_obj = frame_env_vals["self"]
                        if frame.get("owner_type") is not None and owner_type_obj is None:
                            owner_type_obj = frame.get("owner_type")
                        if self_obj is not None and owner_type_obj is not None:
                            break
                    if self_obj is None or not isinstance(self_obj, InstanceObject):
                        self._raise_runtime("RuntimeError", "super() called outside instance method", expr.span)
                    # Find next type in MRO after owner_type_obj
                    mro_names = self_obj.type_obj.mro
                    start_type = None
                    if owner_type_obj is not None:
                        found_owner = False
                        for mro_name in mro_names:
                            if found_owner:
                                start_type = self.types.get(mro_name)
                                break
                            if mro_name == owner_type_obj.name:
                                found_owner = True
                    try:
                        proxy = self.heap.allocate("SuperProxy", SuperProxy(receiver=self_obj, start_type=start_type))
                    except OutOfMemoryError as exc:
                        self._raise_runtime("OutOfMemoryError", str(exc), expr.span)
                    return proxy

            args = [self.eval_expr(arg, env) for arg in expr.args]

            if isinstance(expr.callee, ast.IdentifierExpr):
                alias = self._resolve_unshadowed_call_alias(expr.callee, env)
                if alias is not None:
                    if alias == "io_print":
                        return self._invoke_intrinsic(alias, [[self._to_text(a) for a in args]], expr.span)
                    if alias == "core_str" and len(args) == 1:
                        return self._to_text(args[0])
                    return self._invoke_intrinsic(alias, args, expr.span)

            callee = self.eval_expr(expr.callee, env)
            return self._call_value(callee, args, expr.span)

        if isinstance(expr, ast.MacroCallExpr):
            macro = self._macros.get(expr.name)
            if macro is None:
                self._raise_runtime("RuntimeError", f"undefined macro '{expr.name}'", expr.span)
            if len(expr.args) != len(macro.params):
                self._raise_runtime(
                    "TypeError",
                    f"macro '{expr.name}' expects {len(macro.params)} argument(s), got {len(expr.args)}",
                    expr.span,
                )
            arg_values = [self.eval_expr(arg, env) for arg in expr.args]
            # Re-parse the macro body as a function so the body executes in a
            # fresh local scope with params bound to the evaluated arguments.
            param_str = ", ".join(macro.params)
            indented_body = "\n    ".join(macro.body) if macro.body else "return none"
            fn_src = f"fn __macro_{macro.name}__({param_str}):\n    {indented_body}\n"
            try:
                toks = Lexer(source=fn_src, file=f"<macro:{macro.name}>").tokenize()
                macro_prog = Parser(tokens=toks, file=f"<macro:{macro.name}>", source_lines=fn_src.splitlines()).parse()
                fn_decl = macro_prog.declarations[0]
            except Exception as exc:
                self._raise_runtime("RuntimeError", f"macro '{macro.name}' body parse error: {exc}", expr.span)
            macro_fn = FunctionValue(decl=fn_decl, globals_env=self.global_env)
            return self._call_function(macro_fn, arg_values)

        self._raise_runtime("RuntimeError", f"unsupported expression '{type(expr).__name__}'", self._span_of(expr))

    def _resolve_unshadowed_call_alias(self, callee: ast.IdentifierExpr, env: Environment) -> str | None:
        """Resolve builtin aliases only when normal runtime bindings do not win.

        This keeps interpreter execution aligned with source-level expectations:
        local variables, imported functions, module globals, and declared types
        should all beat fallback builtin aliases like `min` and `max`.
        """

        try:
            env.lookup(callee.name)
        except KeyError:
            pass
        else:
            return None

        if callee.name in self.functions or callee.name in self.types:
            return None
        return resolve_call_alias_name(callee)

    def _call_value(self, callee: Any, args: list[Any], span: Any) -> Any:
        if isinstance(callee, FunctionValue):
            return self._call_function(callee, args)
        if isinstance(callee, BoundMethodObject):
            fn = callee.function
            if isinstance(fn, FunctionValue) and fn.is_class_method:
                cls_arg = callee.receiver.type_obj if isinstance(callee.receiver, InstanceObject) else callee.receiver
                return self._call_function(fn, [cls_arg, *args])
            return self._call_function(fn, [callee.receiver, *args])
        if isinstance(callee, SuperProxy):
            # super() itself is not callable; attribute access via super().method() is
            self._raise_runtime("TypeError", "super() object is not directly callable", span)
        if isinstance(callee, TypeObject):
            cast = self._try_primitive_cast(callee, args, span)
            if cast is not None:
                return cast
            return self._construct_instance(callee, args, span)
        if isinstance(callee, InstanceObject):
            call_method = self._lookup_method(callee.type_obj, "__call__") or self._lookup_method(callee.type_obj, "call_")
            if call_method is not None:
                return self._call_function(call_method, [callee, *args])
            self._raise_runtime("TypeError", f"'{callee.type_obj.name}' object is not callable", span)
        if isinstance(callee, IntrinsicCallable):
            return self._invoke_intrinsic(callee.name, args, span)
        if isinstance(callee, StdCallable):
            return self._invoke_intrinsic(callee.intrinsic, args, span)
        if callable(callee):
            # Python callables registered by extern declarations or other host bindings.
            try:
                return callee(*args)
            except RaiseSignal:
                raise
            except Exception as exc:
                self._raise_runtime("RuntimeError", str(exc), span)
        self._raise_runtime("TypeError", "call target is not callable", span)

    def _try_primitive_cast(self, type_obj: TypeObject, args: list[Any], span: Any) -> Any | None:
        """Handle int(), float(), str(), bool() type-cast calls on primitive types."""
        if len(args) != 1:
            return None
        name = type_obj.name
        val = args[0]
        if name == "float":
            if isinstance(val, float):
                return val
            if isinstance(val, int):
                return float(val)
            if isinstance(val, str):
                try:
                    return float(val)
                except ValueError:
                    self._raise_runtime("ValueError", f"cannot convert {val!r} to float", span)
            return None
        if name == "int":
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            if isinstance(val, str):
                try:
                    return int(val)
                except ValueError:
                    self._raise_runtime("ValueError", f"cannot convert {val!r} to int", span)
            return None
        if name == "str":
            return self._to_text(val)
        if name == "bool":
            if isinstance(val, bool):
                return val
            if isinstance(val, InstanceObject):
                result, found = self._try_dunder(val, "__bool__", [])
                if not found:
                    result, found = self._try_dunder(val, "bool_", [])
                if found:
                    return bool(result)
                # Fall back: True if object exists (like Python)
                return True
            return bool(val)
        return None

    def _construct_instance(self, type_obj: TypeObject, args: list[Any], span: Any) -> InstanceObject:
        instance: InstanceObject
        try:
            instance = self.heap.allocate(type_obj.name, InstanceObject(type_obj=type_obj))
        except OutOfMemoryError as exc:
            self._raise_runtime("OutOfMemoryError", str(exc), span)
        init_fn = self._lookup_method(type_obj, "__init__")
        if init_fn is None:
            init_fn = self._lookup_method(type_obj, "init")
        
        if init_fn is not None:
            self._call_function(init_fn, [instance, *args])
        elif args:
            base_exc = self.types.get("BaseException")
            if base_exc is not None and self._is_type_or_subtype(type_obj, base_exc) and len(args) <= 1:
                instance.attrs["message"] = "" if not args else str(args[0])
            else:
                self._raise_runtime("TypeError", f"{type_obj.name}() takes no arguments", span)
        return instance

    def _lookup_method(self, type_obj: TypeObject, name: str) -> FunctionValue | None:
        cur: TypeObject | None = type_obj
        while cur is not None:
            if name in cur.methods:
                return cur.methods[name]
            cur = cur.base
        return None

    def _lookup_getter(self, type_obj: TypeObject, name: str) -> FunctionValue | None:
        cur: TypeObject | None = type_obj
        while cur is not None:
            if name in cur.getters:
                return cur.getters[name]
            cur = cur.base
        return None

    def _lookup_setter(self, type_obj: TypeObject, name: str) -> FunctionValue | None:
        cur: TypeObject | None = type_obj
        while cur is not None:
            if name in cur.setters:
                return cur.setters[name]
            cur = cur.base
        return None

    def _lookup_attr(self, base: Any, attr: str, span: Any) -> Any:
        if isinstance(base, (str, list, dict)):
            bound = self._builtin_methods.lookup(base, attr, span)
            if bound is not None:
                return bound
        if isinstance(base, SuperProxy):
            # Start lookup from start_type in MRO
            start = base.start_type
            if start is None:
                start = self.types.get("object")
            if start is None:
                self._raise_runtime("AttributeError", "super() has no valid base", span)
            method = None
            cur: TypeObject | None = start
            while cur is not None:
                if attr in cur.methods:
                    method = cur.methods[attr]
                    break
                cur = cur.base
            if method is None:
                self._raise_runtime("AttributeError", f"super() has no attribute '{attr}'", span)
            if method.is_static_method:
                return method
            try:
                return self.heap.allocate("BoundMethod", BoundMethodObject(receiver=base.receiver, function=method))
            except OutOfMemoryError as exc:
                self._raise_runtime("OutOfMemoryError", str(exc), span)
        if isinstance(base, InstanceObject):
            getter = self._lookup_getter(base.type_obj, attr)
            if getter is not None:
                return self._call_function(getter, [base])
            if attr in base.attrs:
                value = base.attrs[attr]
                if isinstance(value, FunctionValue):
                    if value.is_static_method:
                        return value
                    try:
                        return self.heap.allocate("BoundMethod", BoundMethodObject(receiver=base, function=value))
                    except OutOfMemoryError as exc:
                        self._raise_runtime("OutOfMemoryError", str(exc), span)
                return value
            class_attr = self._lookup_class_attr(base.type_obj, attr)
            if class_attr is not None:
                return class_attr
            method = self._lookup_method(base.type_obj, attr)
            if method is not None:
                if method.is_static_method:
                    return method
                try:
                    return self.heap.allocate("BoundMethod", BoundMethodObject(receiver=base, function=method))
                except OutOfMemoryError as exc:
                    self._raise_runtime("OutOfMemoryError", str(exc), span)
            self._raise_runtime("AttributeError", f"'{base.type_obj.name}' has no attribute '{attr}'", span)
        if isinstance(base, TypeObject):
            if hasattr(base, attr) and not attr.startswith("_"):
                return getattr(base, attr)
            class_attr = self._lookup_class_attr(base, attr)
            if class_attr is not None:
                return class_attr
            method = self._lookup_method(base, attr)
            if method is not None:
                if method.is_static_method:
                    return method
                self._raise_runtime(
                    "AttributeError",
                    f"instance method '{base.name}.{attr}' requires an object; mark it @static_method to call it on the type",
                    span,
                )
            self._raise_runtime("AttributeError", f"type '{base.name}' has no attribute '{attr}'", span)
        if isinstance(base, ModuleObject):
            if attr in base.exports:
                return base.exports[attr]
            self._raise_runtime("AttributeError", f"module '{base.name}' has no attribute '{attr}'", span)
        if isinstance(base, dict):
            if attr in base:
                return base[attr]
            self._raise_runtime("AttributeError", f"map has no attribute '{attr}'", span)
        if isinstance(base, ExceptionObject):
            if attr == "message":
                return base.message
            if attr == "type":
                return base.type_obj.name
        self._raise_runtime("AttributeError", f"attribute access not supported for '{attr}'", span)

    def _lookup_class_attr(self, type_obj: TypeObject, attr: str) -> Any | None:
        cur: TypeObject | None = type_obj
        while cur is not None:
            if attr in cur.attrs:
                return cur.attrs[attr]
            cur = cur.base
        return None

    def _store_attr(self, target: Any, attr: str, value: Any, span: Any) -> None:
        if isinstance(target, InstanceObject):
            setter = self._lookup_setter(target.type_obj, attr)
            if setter is not None:
                self._call_function(setter, [target, value])
                return
            target.attrs[attr] = value
            return
        if isinstance(target, TypeObject):
            if attr in self.type_constants.get(target.name, set()):
                self._raise_runtime("RuntimeError", f"type attribute '{target.name}.{attr}' is immutable", span)
            target.attrs[attr] = value
            return
        if isinstance(target, ModuleObject):
            target.exports[attr] = value
            return
        self._raise_runtime("TypeError", "attribute assignment target must be an object instance", span)

    def _slice(self, target: Any, rng: "ast.RangeExpr", env: Environment, span: Any) -> Any:
        """`xs[a..b]` / `s[a..b]`: a new array or string with the items from a up to (not including) b.
        Negative bounds count from the end, and bounds outside the sequence are clamped."""
        if not isinstance(target, (list, str)):
            self._raise_runtime("TypeError", "only arrays and strings can be sliced", span)
        start = self.eval_expr(rng.start, env)
        stop = self.eval_expr(rng.stop, env)
        for bound in (start, stop):
            if not isinstance(bound, int) or isinstance(bound, bool):
                self._raise_runtime("TypeError", "slice bounds must be integers", span)
        return target[start:stop]

    def _normalize_list_index(self, target: list[Any], index: Any, span: Any) -> int:
        if not isinstance(index, int):
            self._raise_runtime("TypeError", "list index must be int", span)
        idx = index
        if idx < 0:
            idx += len(target)
        if idx < 0 or idx >= len(target):
            self._raise_runtime("IndexError", f"list index out of range: {index}", span)
        return idx

    def _assert_hashable(self, key: Any, span: Any) -> None:
        try:
            hash(key)
        except Exception:
            self._raise_runtime("TypeError", "unhashable map key", span)

    def _call_function(self, fn: FunctionValue, args: list[object]) -> object:
        if len(args) != len(fn.decl.params):
            self._raise_runtime("TypeError", f"function '{fn.decl.name}' expects {len(fn.decl.params)} args, got {len(args)}", fn.decl.span)

        gpu_config = normalize_gpu_decorator(fn.decl)
        if gpu_config is not None and gpu_config.required:
            selection = backend_selection_report("auto", policy="required")
            if selection.selected_backend == "cpu":
                self._raise_runtime(
                    "RuntimeError",
                    "GPU backend unavailable for required @gpu call",
                    fn.decl.span,
                )
            if selection.selected_backend != "cuda":
                self._raise_runtime(
                    "RuntimeError",
                    f"required @gpu backend '{selection.selected_backend}' is not implemented in interpreter mode",
                    fn.decl.span,
                )

        env = Environment(parent=fn.globals_env or self.global_env)
        for param, value in zip(fn.decl.params, args, strict=True):
            env.define(param.name, value)

        self.call_stack.append({"function": fn.decl.name, "span": fn.decl.span, "env": env.values, "owner_type": fn.owner_type})
        try:
            for stmt in fn.decl.body:
                self.execute_stmt(stmt, env, in_loop=False)
        except ReturnSignal as signal:
            return signal.value
        finally:
            self.call_stack.pop()
        return None

    def _help_text(self, value: Any) -> str:
        """Render deterministic help text from runtime doc metadata."""

        if isinstance(value, BoundMethodObject):
            return self._help_text(value.function)

        if isinstance(value, FunctionValue):
            return self._format_function_help(value)

        if isinstance(value, TypeObject):
            lines = [f"class {value.name}"]
            lines.append(value.docstring or "No docstring available.")
            return "\n".join(lines)

        if isinstance(value, ModuleObject):
            lines = [f"module {value.name}"]
            lines.append(value.docstring or "No docstring available.")
            return "\n".join(lines)

        return "\n".join(
            [
                f"value of type {self._runtime_type_name(value)}",
                "No docstring available.",
            ]
        )

    def _format_function_help(self, fn: FunctionValue) -> str:
        prefixes: list[str] = []
        if fn.is_static_method:
            prefixes.append("@static_method")
        gpu = normalize_gpu_decorator(fn.decl)
        if gpu is not None:
            required = "true" if gpu.required else "false"
            prefixes.append(f'@gpu(required={required}, mode="{gpu.mode}")')
        prefix = ""
        if prefixes:
            prefix = " ".join(prefixes) + " "
        args = ", ".join(f"{param.name}: {normalize_type_name(param.type_name) or 'any'}" for param in fn.decl.params)
        ret = normalize_type_name(fn.decl.return_type) or "none"
        qualified_name = fn.decl.name
        if fn.owner_type is not None:
            qualified_name = f"{fn.owner_type.name}.{fn.decl.name}"
        signature = f"{prefix}fn {qualified_name}({args}) -> {ret}"
        return "\n".join([signature, fn.decl.docstring or "No docstring available."])

    def _runtime_type_name(self, value: Any) -> str:
        runtime_type = self._runtime_type(value)
        if isinstance(runtime_type, TypeObject):
            return runtime_type.name
        return type(value).__name__

    def _gc_intrinsic_hooks(self) -> dict[str, Any]:
        return {
            "collect": self.heap.collect,
            "stats": lambda: {
                "objects": len(self.heap._records),
                "roots": len(self.heap.roots_snapshot()),
                "deterministic_gc": bool(self.heap.deterministic_gc),
                "gc_stress": bool(self.heap.gc_stress),
            },
            "set_deterministic_gc": lambda flag: setattr(self.heap, "deterministic_gc", bool(flag)),
            "set_gc_stress": lambda flag: setattr(self.heap, "gc_stress", bool(flag)),
        }

    def _invoke_intrinsic(self, name: str, args: list[Any], span: Any) -> Any:
        try:
            return invoke_intrinsic(
                name,
                args,
                stdout_write=lambda s: (self.stdout.write(s), None)[1],
                stdin_readline=lambda: "",
                gc_hooks=self._gc_intrinsic_hooks(),
            )
        except OverflowError as exc:
            self._raise_runtime("OverflowError", f"intrinsic {name}: {exc}", span)
        except TypeError as exc:
            self._raise_runtime("TypeError", f"intrinsic {name}: {exc}", span)
        except ValueError as exc:
            self._raise_runtime("ValueError", f"intrinsic {name}: {exc}", span)
        except FileNotFoundError as exc:
            self._raise_runtime("OSError", f"intrinsic {name}: {exc}", span, payload={"op": name}, code=getattr(exc, "errno", None))
        except OSError as exc:
            self._raise_runtime("OSError", f"intrinsic {name}: {exc}", span, payload={"op": name}, code=getattr(exc, "errno", None))
        except Exception as exc:
            self._raise_runtime("RuntimeError", f"intrinsic {name}: {exc}", span)

    def _raise_value(self, value: Any, span: Any) -> None:
        if isinstance(value, ExceptionObject):
            raise RaiseSignal(value)
        if isinstance(value, InstanceObject) and self._is_type_or_subtype(value.type_obj, self.types["BaseException"]):
            exc = self._ensure_exception_instance(value)
            raise RaiseSignal(exc)
        self._raise_runtime("TypeError", "raised value is not an exception instance", span)

    def _ensure_exception_instance(self, value: InstanceObject) -> ExceptionObject:
        message = value.attrs.get("message", "")
        if not isinstance(message, str):
            message = str(message)
        code = value.attrs.get("code")
        if code is not None and not isinstance(code, int):
            try:
                code = int(code)
            except Exception:
                code = None
        stack = self._capture_stacktrace()
        try:
            return self.heap.allocate(
                "Exception",
                ExceptionObject(type_obj=value.type_obj, message=message, payload=value, code=code, stacktrace=stack),
            )
        except OutOfMemoryError as exc:
            self._raise_runtime("OutOfMemoryError", str(exc), self._span_of(value))

    def _raise_runtime(self, type_name: str, message: str, span: Any, payload: Any = None, code: int | None = None) -> NoReturn:
        t = self.types.get(type_name) or self.types["RuntimeError"]
        stack = self._capture_stacktrace()
        try:
            exc = self.heap.allocate("Exception", ExceptionObject(type_obj=t, message=message, payload=payload, code=code, stacktrace=stack))
        except OutOfMemoryError as alloc_err:
            t2 = self.types["RuntimeError"]
            exc = ExceptionObject(type_obj=t2, message=str(alloc_err), payload=None, stacktrace=stack)
        raise RaiseSignal(exc)

    def _capture_stacktrace(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for frame in self.call_stack:
            span = frame.get("span")
            out.append(
                {
                    "function": frame.get("function"),
                    "file": getattr(span, "file", self.file),
                    "line": getattr(span, "line", 1),
                    "column": getattr(span, "column", 1),
                    "callsite_id": frame.get("callsite_id"),
                    "hlir_id": frame.get("hlir_id"),
                }
            )
        return out

    def _matches_exception(self, error: ExceptionObject, type_name: str) -> bool:
        target = self.types.get(type_name)
        if target is None:
            return False
        return self._is_type_or_subtype(error.type_obj, target)

    def _is_type_or_subtype(self, child: TypeObject, parent: TypeObject) -> bool:
        cur: TypeObject | None = child
        while cur is not None:
            if cur.name == parent.name:
                return True
            cur = cur.base
        return False

    def _runtime_type(self, value: Any) -> Any:
        if isinstance(value, InstanceObject):
            return value.type_obj
        if isinstance(value, TypeObject):
            return self.types.get("type", value)
        if isinstance(value, ExceptionObject):
            return value.type_obj
        if value is None:
            return self.types["none"]
        if isinstance(value, bool):
            return self.types["bool"]
        if isinstance(value, int):
            return self.types["int"]
        if isinstance(value, float):
            return self.types["float"]
        if isinstance(value, str):
            return self.types["str"]
        if isinstance(value, list):
            return self.types["array"]
        if isinstance(value, dict):
            return self.types["map"]
        return self.types["object"]

    def _equals(self, left: Any, right: Any, span: Any) -> bool:
        if isinstance(left, InstanceObject):
            eq_method = self._lookup_method(left.type_obj, "__eq__") or self._lookup_method(left.type_obj, "eq")
            if eq_method is None:
                return left is right
            result = self._call_function(eq_method, [left, right])
            return bool(result)
        if isinstance(right, InstanceObject):
            return right is left
        try:
            return bool(left == right)
        except Exception:
            self._raise_runtime("TypeError", "equality comparison failed", span)

    def _try_dunder(self, obj: Any, dunder_name: str, args: list[Any]) -> tuple[Any, bool]:
        """Try calling a dunder method on an instance. Returns (result, found)."""
        if not isinstance(obj, InstanceObject):
            return None, False
        method = self._lookup_method(obj.type_obj, dunder_name)
        if method is None:
            return None, False
        result = self._call_function(method, [obj, *args])
        return result, True

    def _import_module(self, module_name: str, span: Any, *, level: int = 0) -> ModuleObject:
        # Convert relative forms into a stable canonical module key.
        canonical_name = self._canonicalize_module_name(module_name, level, span)

        existing = self.module_cache.get(canonical_name)
        if existing is not None:
            # Import caching guarantees single initialization semantics.
            return existing

        if canonical_name in self._loading_modules:
            cached = self.module_cache.get(canonical_name)
            if cached is not None:
                # Cycle handling: expose partially initialized module object.
                return cached
            self._raise_runtime("ImportError", f"cyclic import failed for '{canonical_name}'", span)

        try:
            module_obj = self.heap.allocate("Module", ModuleObject(name=canonical_name, exports={}))
        except OutOfMemoryError as exc:
            self._raise_runtime("OutOfMemoryError", str(exc), span)
        self.module_cache[canonical_name] = module_obj
        self._loading_modules.add(canonical_name)
        try:
            module_path = self._resolve_module_path(canonical_name)
            if module_path is None:
                self._raise_runtime("ImportError", f"module not found: '{canonical_name}'", span)
            # Package marker is needed so `from . import x` inside __init__.mv
            # uses the package itself as the import base.
            if module_path.name == "__init__.mv":
                self._package_modules.add(canonical_name)
            source = module_path.read_text(encoding="utf-8")
            lexer = Lexer(source=source, file=str(module_path))
            tokens = lexer.tokenize()
            parser = Parser(tokens=tokens, file=str(module_path), source_lines=source.splitlines())
            program = parser.parse()

            analyzer = SemanticAnalyzer(file=str(module_path))
            result = analyzer.analyze(program)
            analyzer.assert_valid(result)
            module_obj.docstring = program.docstring

            module_env = Environment(parent=self.global_env)
            module_env.define(canonical_name.split(".")[-1], module_obj)

            for decl in program.declarations:
                if isinstance(decl, ast.FnDecl):
                    # Functions capture module environment for global lookup parity.
                    fn_val = FunctionValue(decl=decl, globals_env=module_env)
                    module_env.define(decl.name, fn_val)
                    self.functions[f"{canonical_name}.{decl.name}"] = fn_val
                elif isinstance(decl, ast.TypeDecl):
                    # Type declarations are attached to runtime type registry.
                    primary_base = decl.base_names[0] if decl.base_names else "object"
                    if decl.name not in self.types:
                        self._define_type(decl.name, primary_base)
                    type_obj = self.types[decl.name]
                    if decl.base_names:
                        pbase = decl.base_names[0]
                        if pbase in self.types:
                            type_obj.base = self.types[pbase]
                        base_objects = [self.types[b] for b in decl.base_names if b in self.types]
                        type_obj.mro = self._c3_mro(type_obj, base_objects)
                    type_obj.docstring = decl.docstring
                    module_env.define(decl.name, type_obj)
                    if decl.attrs:
                        self.type_constants.setdefault(type_obj.name, set()).update(attr.name for attr in decl.attrs)
                        for attr in decl.attrs:
                            type_obj.attrs[attr.name] = self.eval_expr(attr.value, module_env)
                    for method in decl.methods:
                        fn_val = FunctionValue(
                            decl=method,
                            owner_type=type_obj,
                            globals_env=module_env,
                            is_static_method=has_static_method_decorator(method),
                            is_class_method=has_classmethod_decorator(method),
                        )
                        accessor = accessor_kind(method)
                        if accessor in {"getter", "setter"}:
                            property_name = accessor_property_name(method) or method.name
                            if accessor == "getter":
                                type_obj.getters[property_name] = fn_val
                            else:
                                type_obj.setters[property_name] = fn_val
                            continue
                        key = "__init__" if method.name == "init" else method.name
                        type_obj.methods[key] = fn_val
                        if method.name == "init":
                            type_obj.methods["init"] = fn_val
                elif isinstance(decl, ast.ImplDecl):
                    # Impl declarations patch method table on existing type.
                    type_obj = self.types.get(decl.target)
                    if type_obj is None:
                        self._raise_runtime("TypeError", f"impl target '{decl.target}' is undefined", decl.span)
                    for method in decl.methods:
                        fn_val = FunctionValue(
                            decl=method,
                            owner_type=type_obj,
                            globals_env=module_env,
                            is_static_method=has_static_method_decorator(method),
                            is_class_method=has_classmethod_decorator(method),
                        )
                        accessor = accessor_kind(method)
                        if accessor in {"getter", "setter"}:
                            property_name = accessor_property_name(method) or method.name
                            if accessor == "getter":
                                type_obj.getters[property_name] = fn_val
                            else:
                                type_obj.setters[property_name] = fn_val
                            continue
                        key = "__init__" if method.name == "init" else method.name
                        type_obj.methods[key] = fn_val
                        if method.name == "init":
                            type_obj.methods["init"] = fn_val
                elif isinstance(decl, ast.ExternBlock):
                    for fn_decl in decl.fns:
                        cfn = self._make_extern_callable(fn_decl)
                        module_env.define(fn_decl.name, cfn)

            self._module_exec_stack.append(canonical_name)
            try:
                for stmt in program.statements:
                    self.execute_stmt(stmt, module_env)
            finally:
                self._module_exec_stack.pop()

            for name, value in module_env.values.items():
                if not name.startswith("_"):
                    # Export policy: underscore-prefixed names remain private.
                    module_obj.exports[name] = value
            return module_obj
        finally:
            self._loading_modules.discard(canonical_name)

    def _canonicalize_module_name(self, module_name: str, level: int, span: Any) -> str:
        if level <= 0:
            return module_name

        # Relative imports are only valid while executing a module body.
        if not self._module_exec_stack:
            self._raise_runtime("ImportError", "relative import requires package context", span)

        current = self._module_exec_stack[-1]
        # Package module (`pkg.__init__`) anchors at `pkg`, regular module
        # (`pkg.mod`) anchors at package path `pkg`.
        if current in self._package_modules:
            package_parts = current.split(".")
        else:
            package_parts = current.split(".")[:-1]
        up = level - 1
        if up > len(package_parts):
            self._raise_runtime("ImportError", f"relative import beyond top-level package: '{'.' * level}{module_name}'", span)

        base_parts = package_parts[: len(package_parts) - up]
        if module_name:
            base_parts.extend(module_name.split("."))
        if not base_parts:
            self._raise_runtime("ImportError", "relative import target is empty", span)
        return ".".join(base_parts)

    def _resolve_module_path(self, module_name: str) -> Path | None:
        rel = Path(*module_name.split("."))
        for root in self.module_search_roots:
            candidates = [
                root / f"{rel}.mv",
                root / rel / "__init__.mv",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()
        return None

    def _build_module_search_roots(self) -> list[Path]:
        roots: list[Path] = [self.source_root]

        # Additional roots may be supplied by the runtime environment.
        raw_path = os.getenv("MANV_PATH", "")
        if raw_path.strip():
            for item in raw_path.split(os.pathsep):
                entry = item.strip()
                if not entry:
                    continue
                try:
                    candidate = Path(entry).resolve()
                except Exception:
                    continue
                if candidate.exists() and candidate.is_dir():
                    roots.append(candidate)

        # Global package cache — installed ManV packages (dependencies from `manv add`).
        # The standard library is parked (see TODO.md): the installed `std` package is
        # deliberately kept off the search path so a run never depends on what happens
        # to be in this machine's cache.
        try:
            from . import packages as _pkg_mod
            for _pkg_root in _pkg_mod.installed_package_roots(exclude=frozenset({_pkg_mod.STDLIB_PACKAGE_NAME})):
                roots.append(_pkg_root)
        except Exception:
            pass  # package cache unavailable; proceed without it

        # Keep first-seen order while de-duplicating resolved directories.
        out: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            out.append(root)
        return out

    def _to_text(self, value: Any, *, nested: bool = False) -> str:
        """How a value is written by print() and str(): `true`/`false`/`none`, a type's
        __repr__ when it defines one, and collections of the same. Strings are quoted only
        when they sit inside a collection."""
        if value is True:
            return "true"
        if value is False:
            return "false"
        if value is None:
            return "none"
        if isinstance(value, str):
            return repr(value) if nested else value
        if isinstance(value, InstanceObject):
            # print(obj) and str(obj) prefer __str__; objects inside a collection use __repr__.
            hooks = ("__repr__", "repr_") if nested else ("__str__", "str_", "__repr__", "repr_")
            for hook in hooks:
                result, found = self._try_dunder(value, hook, [])
                if found:
                    return str(result)
            return self._stringify(value)
        if isinstance(value, list):
            return "[" + ", ".join(self._to_text(v, nested=True) for v in value) + "]"
        if isinstance(value, dict):
            items = list(value.items())
            if self.stable_debug_format:
                items.sort(key=lambda kv: repr(kv[0]))
            return "{" + ", ".join(f"{self._to_text(k, nested=True)}: {self._to_text(v, nested=True)}" for k, v in items) + "}"
        if isinstance(value, (ExceptionObject, ModuleObject)):
            return self._stringify(value)
        return str(value)

    def _stringify(self, value: Any) -> str:
        if isinstance(value, InstanceObject):
            attrs = value.attrs
            if self.stable_debug_format:
                keys = sorted(attrs.keys())
            else:
                keys = list(attrs.keys())
            preview = ", ".join(f"{k}={self._stringify(attrs[k])}" for k in keys[:4])
            suffix = "" if len(keys) <= 4 else ", ..."
            return f"<{value.type_obj.name} {preview}{suffix}>"
        if isinstance(value, ExceptionObject):
            return f"{value.type_obj.name}({value.message})"
        if isinstance(value, ModuleObject):
            return f"<module {value.name}>"
        if isinstance(value, dict) and self.stable_debug_format:
            items = sorted(value.items(), key=lambda kv: repr(kv[0]))
            return "{" + ", ".join(f"{self._stringify(k)}: {self._stringify(v)}" for k, v in items) + "}"
        return str(value)

    def _make_extern_callable(self, fn_decl: "ast.ExternFnDecl") -> Any:
        """Build a Python callable that invokes a C function via ctypes.

        The returned object is stored in the environment under `fn_decl.name`
        so ManV programs can call it like a regular function.
        """
        from .ffi import load_library, manv_to_ctype, marshal_arg, marshal_vararg, unmarshal_return

        lib_id = fn_decl.lib

        def _caller(*args: Any) -> Any:
            try:
                clib = load_library(lib_id)
                cfunc = getattr(clib, fn_decl.c_name)
                cfunc.restype = manv_to_ctype(fn_decl.return_type)
                if not fn_decl.varargs:
                    cfunc.argtypes = [manv_to_ctype(p.type_name) for p in fn_decl.params]

                cargs = []
                for i, arg in enumerate(args):
                    param_type = fn_decl.params[i].type_name if i < len(fn_decl.params) else None
                    if fn_decl.varargs and i >= len(fn_decl.params):
                        cargs.append(marshal_vararg(arg))
                    else:
                        cargs.append(marshal_arg(arg, param_type))

                result = cfunc(*cargs)
                return unmarshal_return(result, fn_decl.return_type)
            except OSError as exc:
                self._raise_runtime("RuntimeError", f"extern call failed: {exc}", fn_decl.span)

        _caller.__name__ = fn_decl.name
        return _caller

    def _span_of(self, node: Any) -> Any:
        return getattr(node, "span", type("_S", (), {"line": 1, "column": 1})())
