from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, TextIO

from . import ast
from .diagnostics import ManvError, diag
from .object_runtime import (
    BoundMethodObject,
    ExceptionObject,
    Heap,
    InstanceObject,
    OutOfMemoryError,
    TypeObject,
)
from .runtime import unsupported_feature
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

        exc_base = self._define_type("BaseException", "object")
        exc = self._define_type("Exception", "BaseException")
        self._define_type("TypeError", "Exception")
        self._define_type("AttributeError", "Exception")
        self._define_type("KeyError", "Exception")
        self._define_type("IndexError", "Exception")
        self._define_type("ValueError", "Exception")
        self._define_type("RuntimeError", "Exception")
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
        roots: list[Any] = [self.global_env.values, self.types, self.active_exceptions]
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
                self.functions[decl.name] = FunctionValue(decl=decl)
            elif isinstance(decl, ast.TypeDecl):
                type_obj = self.types[decl.name]
                if decl.base_name and decl.base_name in self.types:
                    type_obj.base = self.types[decl.base_name]
                    type_obj.mro = [type_obj.name, *type_obj.base.mro]
                for method in decl.methods:
                    type_obj.methods[method.name] = FunctionValue(decl=method, owner_type=type_obj)
            elif isinstance(decl, ast.ImplDecl):
                type_obj = self.types.get(decl.target)
                if type_obj is None:
                    self._raise_runtime("TypeError", f"impl target '{decl.target}' is undefined", decl.span)
                for method in decl.methods:
                    type_obj.methods[method.name] = FunctionValue(decl=method, owner_type=type_obj)
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
                return {"__std__": True}
            if expr.name in self.functions:
                return self.functions[expr.name]
            if expr.name in self.types:
                return self.types[expr.name]
            try:
                return env.lookup(expr.name)
            except KeyError:
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
            if isinstance(base, dict) and base.get("__std__") is True and expr.attr in {"gpu", "memory"}:
                return {"__stub_module__": expr.attr}
            return self._lookup_attr(base, expr.attr, expr.span)

        if isinstance(expr, ast.CallExpr):
            args = [self.eval_expr(arg, env) for arg in expr.args]

            if isinstance(expr.callee, ast.IdentifierExpr):
                name = expr.callee.name
                if name == "print":
                    self.stdout.write(" ".join(self._stringify(a) for a in args) + "\n")
                    return None
                if name == "len":
                    if len(args) != 1:
                        self._raise_runtime("TypeError", "len expects one argument", expr.span)
                    return len(args[0])

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
        if isinstance(callee, dict) and "__stub_module__" in callee:
            raise unsupported_feature(f"std.{callee['__stub_module__']}", self.file, span.line, span.column)
        self._raise_runtime("TypeError", "call target is not callable", span)

    def _construct_instance(self, type_obj: TypeObject, args: list[Any], span: Any) -> InstanceObject:
        try:
            instance = self.heap.allocate(type_obj.name, InstanceObject(type_obj=type_obj))
        except OutOfMemoryError as exc:
            self._raise_runtime("OutOfMemoryError", str(exc), span)
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
                return base.attrs[attr]
            method = self._lookup_method(base.type_obj, attr)
            if method is not None:
                try:
                    return self.heap.allocate("BoundMethod", BoundMethodObject(receiver=base, function=method))
                except OutOfMemoryError as exc:
                    self._raise_runtime("OutOfMemoryError", str(exc), span)
            self._raise_runtime("AttributeError", f"'{base.type_obj.name}' has no attribute '{attr}'", span)
        if isinstance(base, TypeObject):
            if attr in base.attrs:
                return base.attrs[attr]
            method = self._lookup_method(base, attr)
            if method is not None:
                return method
            self._raise_runtime("AttributeError", f"type '{base.name}' has no attribute '{attr}'", span)
        self._raise_runtime("AttributeError", f"attribute access not supported for '{attr}'", span)

    def _store_attr(self, target: Any, attr: str, value: Any, span: Any) -> None:
        if isinstance(target, InstanceObject):
            target.attrs[attr] = value
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

        env = Environment(parent=self.global_env)
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
        stack = self._capture_stacktrace()
        try:
            return self.heap.allocate(
                "Exception",
                ExceptionObject(type_obj=value.type_obj, message=message, payload=value, stacktrace=stack),
            )
        except OutOfMemoryError as exc:
            self._raise_runtime("OutOfMemoryError", str(exc), self._span_of(value))

    def _raise_runtime(self, type_name: str, message: str, span: Any, payload: Any = None) -> None:
        t = self.types.get(type_name) or self.types["RuntimeError"]
        stack = self._capture_stacktrace()
        try:
            exc = self.heap.allocate("Exception", ExceptionObject(type_obj=t, message=message, payload=payload, stacktrace=stack))
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
        if isinstance(value, dict) and self.stable_debug_format:
            items = sorted(value.items(), key=lambda kv: repr(kv[0]))
            return "{" + ", ".join(f"{self._stringify(k)}: {self._stringify(v)}" for k, v in items) + "}"
        return str(value)

    def _span_of(self, node: Any) -> Any:
        return getattr(node, "span", type("_S", (), {"line": 1, "column": 1})())



