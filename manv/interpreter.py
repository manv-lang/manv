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
from typing import Any, TextIO

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
    TypeObject,
)
from .semantics import normalize_gpu_decorator
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
        self.types: dict[str, TypeObject] = {}
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

    def _gc_roots(self) -> list[Any]:
        roots: list[Any] = [self.global_env.values, self.types, self.active_exceptions, self.module_cache]
        for frame in self.call_stack:
            roots.append(frame.get("env"))
        return roots

    def load_program(self, program: ast.Program) -> None:
        # First register all type names so inheritance can resolve in second pass.
        for decl in program.declarations:
            if isinstance(decl, ast.TypeDecl):
                base_name = decl.base_name or "object"
                if decl.name not in self.types:
                    self._define_type(decl.name, base_name)

        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                fn_val = FunctionValue(decl=decl, globals_env=self.global_env)
                self.functions[decl.name] = fn_val
                self.global_env.define(decl.name, fn_val)
            elif isinstance(decl, ast.TypeDecl):
                type_obj = self.types[decl.name]
                if decl.base_name and decl.base_name in self.types:
                    type_obj.base = self.types[decl.base_name]
                    type_obj.mro = [type_obj.name, *type_obj.base.mro]
                for method in decl.methods:
                    key = "__init__" if method.name == "init" else method.name
                    fn_val = FunctionValue(decl=method, owner_type=type_obj, globals_env=self.global_env)
                    type_obj.methods[key] = fn_val
                    if method.name == "init":
                        type_obj.methods["init"] = fn_val
            elif isinstance(decl, ast.ImplDecl):
                type_obj = self.types.get(decl.target)
                if type_obj is None:
                    self._raise_runtime("TypeError", f"impl target '{decl.target}' is undefined", decl.span)
                for method in decl.methods:
                    key = "__init__" if method.name == "init" else method.name
                    fn_val = FunctionValue(decl=method, owner_type=type_obj, globals_env=self.global_env)
                    type_obj.methods[key] = fn_val
                    if method.name == "init":
                        type_obj.methods["init"] = fn_val
            elif isinstance(decl, ast.MacroDeclStub):
                continue
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

        if isinstance(stmt, ast.ImportStmt):
            # Import statement binds module object under alias or leaf name.
            module = self._import_module(stmt.module, stmt.span, level=stmt.level)
            bind = stmt.alias if stmt.alias else stmt.module.split(".")[-1]
            env.define(bind, module)
            return None

        if isinstance(stmt, ast.FromImportStmt):
            # From-import binds the requested symbol directly.
            module = self._import_module(stmt.module, stmt.span, level=stmt.level)
            if stmt.name not in module.exports:
                self._raise_runtime("ImportError", f"cannot import '{stmt.name}' from '{stmt.module}'", stmt.span)
            bind = stmt.alias if stmt.alias else stmt.name
            env.define(bind, module.exports[stmt.name])
            return None

        if isinstance(stmt, ast.AssignStmt):
            value = self.eval_expr(stmt.value, env)
            try:
                env.assign(stmt.name, value)
            except KeyError:
                self._raise_runtime("RuntimeError", f"undefined variable '{stmt.name}'", stmt.span)
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
            guard = 0
            while self.eval_expr(stmt.condition, env):
                guard += 1
                if guard > 1_000_000:
                    self._raise_runtime("RuntimeError", "loop iteration guard exceeded", stmt.span)
                loop_env = Environment(parent=env)
                try:
                    for inner in stmt.body:
                        self.execute_stmt(inner, loop_env, in_loop=True)
                except ContinueSignal:
                    continue
                except BreakSignal:
                    break
            return None

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
            try:
                return eval_unary(expr.op, value)
            except Exception:
                self._raise_runtime("TypeError", f"unsupported unary operator '{expr.op}'", expr.span)

        if isinstance(expr, ast.BinaryExpr):
            left = self.eval_expr(expr.left, env)
            right = self.eval_expr(expr.right, env)
            if expr.op in {"==", "!="}:
                eq = self._equals(left, right, expr.span)
                return eq if expr.op == "==" else (not eq)
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
            index = self.eval_expr(expr.index, env)
            if isinstance(target, list):
                idx = self._normalize_list_index(target, index, expr.span)
                return target[idx]
            if isinstance(target, dict):
                self._assert_hashable(index, expr.span)
                if index not in target:
                    self._raise_runtime("KeyError", f"missing key: {index}", expr.span, payload=index)
                return target[index]
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

            args = [self.eval_expr(arg, env) for arg in expr.args]

            if isinstance(expr.callee, ast.IdentifierExpr):
                alias = resolve_call_alias_name(expr.callee)
                if alias is not None:
                    if alias == "io_print":
                        return self._invoke_intrinsic(alias, [args], expr.span)
                    return self._invoke_intrinsic(alias, args, expr.span)

            callee = self.eval_expr(expr.callee, env)
            return self._call_value(callee, args, expr.span)

        self._raise_runtime("RuntimeError", f"unsupported expression '{type(expr).__name__}'", self._span_of(expr))

    def _call_value(self, callee: Any, args: list[Any], span: Any) -> Any:
        if isinstance(callee, FunctionValue):
            return self._call_function(callee, args)
        if isinstance(callee, BoundMethodObject):
            return self._call_function(callee.function, [callee.receiver, *args])
        if isinstance(callee, TypeObject):
            return self._construct_instance(callee, args, span)
        if isinstance(callee, IntrinsicCallable):
            return self._invoke_intrinsic(callee.name, args, span)
        if isinstance(callee, StdCallable):
            return self._invoke_intrinsic(callee.intrinsic, args, span)
        self._raise_runtime("TypeError", "call target is not callable", span)

    def _construct_instance(self, type_obj: TypeObject, args: list[Any], span: Any) -> InstanceObject:
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

    def _lookup_attr(self, base: Any, attr: str, span: Any) -> Any:
        if isinstance(base, InstanceObject):
            if attr in base.attrs:
                value = base.attrs[attr]
                if isinstance(value, FunctionValue):
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
                return method
            self._raise_runtime("AttributeError", f"type '{base.name}' has no attribute '{attr}'", span)
        if isinstance(base, ModuleObject):
            if attr in base.exports:
                return base.exports[attr]
            self._raise_runtime("AttributeError", f"module '{base.name}' has no attribute '{attr}'", span)
        if isinstance(base, dict):
            if attr in base:
                return base[attr]
            self._raise_runtime("AttributeError", f"map has no attribute '{attr}'", span)
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
            target.attrs[attr] = value
            return
        if isinstance(target, TypeObject):
            target.attrs[attr] = value
            return
        if isinstance(target, ModuleObject):
            target.exports[attr] = value
            return
        self._raise_runtime("TypeError", "attribute assignment target must be an object instance", span)

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

        self.call_stack.append({"function": fn.decl.name, "span": fn.decl.span, "env": env.values})
        try:
            for stmt in fn.decl.body:
                self.execute_stmt(stmt, env, in_loop=False)
        except ReturnSignal as signal:
            return signal.value
        finally:
            self.call_stack.pop()
        return None

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
                stdout_write=self.stdout.write,
                stdin_readline=lambda: "",
                gc_hooks=self._gc_intrinsic_hooks(),
            )
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

    def _raise_runtime(self, type_name: str, message: str, span: Any, payload: Any = None, code: int | None = None) -> None:
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
                    base_name = decl.base_name or "object"
                    if decl.name not in self.types:
                        self._define_type(decl.name, base_name)
                    type_obj = self.types[decl.name]
                    if decl.base_name and decl.base_name in self.types:
                        type_obj.base = self.types[decl.base_name]
                        type_obj.mro = [type_obj.name, *type_obj.base.mro]
                    module_env.define(decl.name, type_obj)
                    for method in decl.methods:
                        key = "__init__" if method.name == "init" else method.name
                        fn_val = FunctionValue(decl=method, owner_type=type_obj, globals_env=module_env)
                        type_obj.methods[key] = fn_val
                        if method.name == "init":
                            type_obj.methods["init"] = fn_val
                elif isinstance(decl, ast.ImplDecl):
                    # Impl declarations patch method table on existing type.
                    type_obj = self.types.get(decl.target)
                    if type_obj is None:
                        self._raise_runtime("TypeError", f"impl target '{decl.target}' is undefined", decl.span)
                    for method in decl.methods:
                        key = "__init__" if method.name == "init" else method.name
                        fn_val = FunctionValue(decl=method, owner_type=type_obj, globals_env=module_env)
                        type_obj.methods[key] = fn_val
                        if method.name == "init":
                            type_obj.methods["init"] = fn_val

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

        # Bundled std fallback enables bootstrapping without per-project copies.
        bundled_std = (Path(__file__).resolve().parents[1] / "std" / "src").resolve()
        if bundled_std.exists() and bundled_std.is_dir():
            roots.append(bundled_std)

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

    def _span_of(self, node: Any) -> Any:
        return getattr(node, "span", type("_S", (), {"line": 1, "column": 1})())
