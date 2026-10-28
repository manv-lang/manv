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
import sys
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
    format_field,
    format_memory_value,
    invoke_intrinsic,
    resolve_call_alias_name,
    std_namespace_attr,
)
from .runtime import unsupported_feature
from .semantics import SemanticAnalyzer
from .semantics_core import eval_binary, eval_unary
from .builtin_methods import BuiltinMethods
from .memory import (
    ArrayRef,
    FLOAT_TYPES,
    INT_TYPES,
    NULL,
    SCALARS,
    Memory,
    MemoryFault,
    Pointer,
    StructRef,
    canonical_pointer_type,
    is_pointer_type,
    pointee_of,
)


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


class ExitSignal(BaseException):
    """`exit(code)`: leaves the program immediately. A BaseException so no ManV handler or host
    wrapper can swallow it."""

    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


MAX_CALL_DEPTH = 1000  # ManV calls; deeper raises a catchable RecursionError
_NO_DECLARED = object()  # `_declared_value` found nothing special about this declaration


class MemCell:
    """A variable whose address was taken (`&x`): its value lives in raw memory, so a pointer and
    the variable always agree. Lookups and assignments go through the cell."""

    __slots__ = ("memory", "ptr", "type_name")

    def __init__(self, memory: Memory, ptr: Pointer, type_name: str):
        self.memory = memory
        self.ptr = ptr
        self.type_name = type_name

    def load(self) -> object:
        return self.memory.load(self.ptr, self.type_name)

    def store(self, value: object) -> None:
        self.memory.store(self.ptr, value, self.type_name)


class StructType:
    """The name of a `struct`, callable as a constructor: `Vec2(1.0, 2.0)`."""

    def __init__(self, name: str):
        self.name = name


class Environment:
    # Set by the interpreter: turns a MemoryFault raised by a variable's cell into a ManV exception.
    on_fault: Any = None

    def __init__(self, parent: "Environment | None" = None):
        self.parent = parent
        self.values: dict[str, object] = {}

    def define(self, name: str, value: object) -> None:
        self.values[name] = value

    def lookup(self, name: str) -> object:
        if name in self.values:
            value = self.values[name]
            return value.load() if type(value) is MemCell else value
        if self.parent is not None:
            return self.parent.lookup(name)
        raise KeyError(name)

    def assign(self, name: str, value: object) -> None:
        if name in self.values:
            current = self.values[name]
            if type(current) is MemCell:
                try:
                    current.store(value)
                except MemoryFault as fault:
                    Environment.on_fault(fault)
                return
            self.values[name] = value
            return
        if self.parent is not None:
            self.parent.assign(name, value)
            return
        raise KeyError(name)

    def owner_of(self, name: str) -> "Environment | None":
        env: Environment | None = self
        while env is not None:
            if name in env.values:
                return env
            env = env.parent
        return None


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
        self.memory = Memory()
        self._frame_cells: list[list[tuple[Environment, str, MemCell]]] = [[]]
        Environment.on_fault = lambda fault: self._raise_fault(fault, None)
        self._install_memory_builtins()
        self.argv: list[str] = []
        self.stdin: TextIO | None = None  # defaults to sys.stdin when input() is called
        self._call_depth = 0
        self._install_host_builtins()

    def _bootstrap_runtime_types(self) -> None:
        object_t = self.heap.allocate("Type", TypeObject(name="object"))
        type_t = self.heap.allocate("Type", TypeObject(name="type", base=object_t))
        object_t.base = None
        object_t.mro = ["object"]
        type_t.mro = ["type", "object"]

        self.types["object"] = object_t
        self.types["type"] = type_t

        for primitive in ("int", "float", "bool", "str", "array", "map", "tuple", "none"):
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
        self._define_type("ZeroDivisionError", "Exception")
        self._define_type("RecursionError", "RuntimeError")
        self._define_type("EOFError", "Exception")
        self._define_type("PointerError", "Exception")
        self._define_type("NullPointerError", "PointerError")

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
            elif isinstance(decl, ast.StructDecl):
                self._define_struct(decl)
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
        # Each ManV call uses several Python frames; leave room for MAX_CALL_DEPTH of them.
        previous_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(previous_limit, MAX_CALL_DEPTH * 25))
        try:
            return self._run_main(program)
        except ExitSignal as signal:
            return signal.code
        finally:
            sys.setrecursionlimit(previous_limit)

    def _run_main(self, program: ast.Program) -> int:
        self.load_program(program)
        if "main" not in self.functions:
            if program.statements:
                return 0  # script mode: the top-level statements were the program
            raise ManvError(diag("E3001", "missing 'main' function (and no top-level statements to run)", self.file, 1, 1))
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
            if stmt.c_array:
                env.define(stmt.name, self._make_c_array(stmt, env))
                return None
            if stmt.type_name and stmt.array_size is None:
                declared = self._declared_value(stmt, env)
                if declared is not _NO_DECLARED:
                    env.define(stmt.name, declared)
                    return None
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
            value = self._own(self.eval_expr(stmt.value, env))
            if stmt.pointer_type is not None:
                value = self._coerce_pointer(value, stmt.pointer_type)
            if stmt.declares:
                env.define(stmt.name, value)
                return None
            try:
                env.assign(stmt.name, value)
            except KeyError:
                self._raise_runtime("RuntimeError", f"undefined variable '{stmt.name}'", stmt.span)
            return None

        if isinstance(stmt, ast.DestructureStmt):
            value = self.eval_expr(stmt.value, env)
            if not isinstance(value, (list, tuple)):
                self._raise_runtime("TypeError", "only an array or tuple can be unpacked", stmt.span)
            if len(value) != len(stmt.targets):
                self._raise_runtime(
                    "ValueError", f"cannot unpack {len(value)} value(s) into {len(stmt.targets)} name(s)", stmt.span
                )
            for target, item in zip(stmt.targets, value):
                if stmt.always_declare or target.declares:
                    env.define(target.name, item)
                    continue
                try:
                    env.assign(target.name, item)
                except KeyError:
                    self._raise_runtime("RuntimeError", f"undefined variable '{target.name}'", stmt.span)
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
            result = self._augmented(op, current, rhs, stmt.span)
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
            result = self._augmented(op, current, rhs, stmt.span)
            self._store_attr(target, stmt.attr, result, stmt.span)
            return None

        if isinstance(stmt, ast.SetDerefStmt):
            self._exec_set_deref(stmt, env)
            return None

        if isinstance(stmt, ast.AugAssignIndexStmt):
            target = self.eval_expr(stmt.target, env)
            index = self.eval_expr(stmt.index, env)
            if isinstance(target, (Pointer, ArrayRef)):
                slot, slot_type = self._element_slot(target, index, stmt.span)
                rhs = self.eval_expr(stmt.value, env)
                self._store_result(slot, slot_type, stmt.op[:-1], rhs, stmt.span)
                return None
            if isinstance(target, list):
                idx = self._normalize_list_index(target, index, stmt.span)
                current = target[idx]
                rhs = self.eval_expr(stmt.value, env)
                op = stmt.op[:-1]
                result = self._augmented(op, current, rhs, stmt.span)
                target[idx] = result
            elif isinstance(target, dict):
                current = target.get(index)
                rhs = self.eval_expr(stmt.value, env)
                op = stmt.op[:-1]
                result = self._augmented(op, current, rhs, stmt.span)
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
                        result = self._augmented(op, current, rhs, stmt.span)
                else:
                    result = self._augmented(op, current, rhs, stmt.span)
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
            if isinstance(target, (Pointer, ArrayRef)):
                slot, slot_type = self._element_slot(target, index, stmt.span)
                self._guard(lambda: self.memory.store(slot, value, slot_type), stmt.span)
                return None
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
                if isinstance(iterable, tuple):
                    iterable = list(iterable)
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
            if isinstance(left, (Pointer, ArrayRef)) or isinstance(right, (Pointer, ArrayRef)):
                return self._pointer_binary(expr.op, left, right, expr.span)
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
                if isinstance(right, (list, tuple)):
                    return any(self._equals(item, left, expr.span) for item in right)
                if isinstance(right, dict):
                    self._assert_hashable(left, expr.span)
                    return left in right
                if isinstance(right, str):
                    if not isinstance(left, str):
                        self._raise_runtime("TypeError", "'in <string>' needs a string on the left", expr.span)
                    return left in right
                self._raise_runtime("TypeError", "'in' needs an array, map or string on the right", expr.span)
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
            return self._binary_value(expr.op, left, right, expr.span)

        if isinstance(expr, ast.ArrayExpr):
            return [self.eval_expr(e, env) for e in expr.elements]

        if isinstance(expr, ast.TupleExpr):
            return tuple(self.eval_expr(e, env) for e in expr.elements)

        if isinstance(expr, ast.AddrOfExpr):
            return self._eval_address_of(expr, env)

        if isinstance(expr, ast.DerefExpr):
            ptr = self._as_pointer(self.eval_expr(expr.operand, env), expr.span, "dereference")
            return self._guard(lambda: self.memory.load(ptr), expr.span)

        if isinstance(expr, ast.CastExpr):
            return self._cast(self.eval_expr(expr.operand, env), expr.type_name, expr.span)

        if isinstance(expr, ast.SizeofExpr):
            return self._eval_sizeof(expr, env)

        if isinstance(expr, ast.FStringExpr):
            pieces: list[str] = []
            for part in expr.parts:
                if isinstance(part, str):
                    pieces.append(part)
                    continue
                field, spec = part
                value = self.eval_expr(field, env)
                text = self._to_text(value)
                if spec:
                    try:
                        text = format_field(value, text, spec)
                    except ValueError as exc:
                        self._raise_runtime("ValueError", str(exc), expr.span)
                pieces.append(text)
            return "".join(pieces)

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
            if isinstance(target, (Pointer, ArrayRef)):
                slot, slot_type = self._element_slot(target, index, expr.span)
                return self._guard(lambda: self.memory.load(slot, slot_type), expr.span)
            if isinstance(target, (list, tuple)):
                idx = self._normalize_list_index(target, index, expr.span)
                return target[idx]
            if isinstance(target, str):
                if not isinstance(index, int) or isinstance(index, bool):
                    self._raise_runtime("TypeError", "string index must be int", expr.span)
                idx = index + len(target) if index < 0 else index
                if idx < 0 or idx >= len(target):
                    self._raise_runtime("IndexError", f"index {index} out of range for a string of length {len(target)}", expr.span)
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
            kwargs = {name: self.eval_expr(value, env) for name, value in expr.kwargs}

            if isinstance(expr.callee, ast.IdentifierExpr):
                alias = self._resolve_unshadowed_call_alias(expr.callee, env)
                if alias is not None:
                    if kwargs:
                        self._raise_runtime("TypeError", f"builtin '{expr.callee.name}' does not take keyword arguments", expr.span)
                    if alias == "io_print":
                        return self._invoke_intrinsic(alias, [[self._to_text(a) for a in args]], expr.span)
                    if alias == "core_str" and len(args) == 1:
                        return self._to_text(args[0])
                    return self._invoke_intrinsic(alias, args, expr.span)

            callee = self.eval_expr(expr.callee, env)
            return self._call_value(callee, args, expr.span, kwargs)

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

    def _call_value(self, callee: Any, args: list[Any], span: Any, kwargs: dict[str, Any] | None = None) -> Any:
        if isinstance(callee, FunctionValue):
            return self._call_function(callee, args, kwargs)
        if isinstance(callee, BoundMethodObject):
            fn = callee.function
            if isinstance(fn, FunctionValue) and fn.is_class_method:
                cls_arg = callee.receiver.type_obj if isinstance(callee.receiver, InstanceObject) else callee.receiver
                return self._call_function(fn, [cls_arg, *args], kwargs)
            return self._call_function(fn, [callee.receiver, *args], kwargs)
        if isinstance(callee, StructType):
            return self._construct_struct(callee, args, kwargs or {}, span)
        if kwargs and isinstance(callee, TypeObject):
            return self._construct_instance(callee, args, span, kwargs)
        if kwargs and isinstance(callee, InstanceObject):
            call_method = self._lookup_method(callee.type_obj, "__call__") or self._lookup_method(callee.type_obj, "call_")
            if call_method is not None:
                return self._call_function(call_method, [callee, *args], kwargs)
        if kwargs:
            self._raise_runtime("TypeError", "this callable does not take keyword arguments", span)
        if isinstance(callee, SuperProxy):
            # super() itself is not callable; attribute access via super().method() is
            self._raise_runtime("TypeError", "super() object is not directly callable", span)
        if isinstance(callee, StructType):
            return self._construct_struct(callee, args, kwargs or {}, span)
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
            except MemoryFault as fault:
                self._raise_fault(fault, span)
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

    def _construct_instance(self, type_obj: TypeObject, args: list[Any], span: Any, kwargs: dict[str, Any] | None = None) -> InstanceObject:
        instance: InstanceObject
        try:
            instance = self.heap.allocate(type_obj.name, InstanceObject(type_obj=type_obj))
        except OutOfMemoryError as exc:
            self._raise_runtime("OutOfMemoryError", str(exc), span)
        init_fn = self._lookup_method(type_obj, "__init__")
        if init_fn is None:
            init_fn = self._lookup_method(type_obj, "init")
        
        if init_fn is not None:
            self._call_function(init_fn, [instance, *args], kwargs)
        elif kwargs:
            self._raise_runtime("TypeError", f"{type_obj.name}() takes no keyword arguments", span)
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
        if isinstance(base, StructRef):
            return self._guard(lambda: self._struct_get(base, attr), span)
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
        if isinstance(target, StructRef):
            self._guard(lambda: self._struct_set(target, attr, value), span)
            return
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
        if not isinstance(target, (list, tuple, str)):
            self._raise_runtime("TypeError", "only arrays, tuples and strings can be sliced", span)
        start = None if rng.start is None else self.eval_expr(rng.start, env)
        stop = None if rng.stop is None else self.eval_expr(rng.stop, env)
        for bound in (start, stop):
            if bound is not None and (not isinstance(bound, int) or isinstance(bound, bool)):
                self._raise_runtime("TypeError", "slice bounds must be integers", span)
        return target[start:stop]

    def _normalize_list_index(self, target: list[Any], index: Any, span: Any) -> int:
        if not isinstance(index, int):
            self._raise_runtime("TypeError", "list index must be int", span)
        idx = index
        if idx < 0:
            idx += len(target)
        if idx < 0 or idx >= len(target):
            self._raise_runtime("IndexError", f"index {index} out of range for an array of length {len(target)}", span)
        return idx

    def _assert_hashable(self, key: Any, span: Any) -> None:
        try:
            hash(key)
        except Exception:
            self._raise_runtime("TypeError", "unhashable map key", span)

    def _bind_arguments(self, fn: FunctionValue, args: list[object], kwargs: dict[str, Any] | None) -> list[object]:
        """Match positional and keyword arguments to the parameters, filling in defaults.
        A default is evaluated at every call that needs it, in the scope the function was defined in."""
        params = fn.decl.params
        span = fn.decl.span
        if len(args) > len(params):
            expects = f"expects {len(params)} args" if not any(p.default is not None for p in params) else f"takes at most {len(params)} args"
            self._raise_runtime("TypeError", f"function '{fn.decl.name}' {expects}, got {len(args)}", span)
        bound: list[object] = list(args)
        for index in range(len(args), len(params)):
            param = params[index]
            if kwargs and param.name in kwargs:
                bound.append(kwargs[param.name])
            elif param.default is not None:
                bound.append(self.eval_expr(param.default, fn.globals_env or self.global_env))
            else:
                if any(p.default is not None for p in params) or kwargs:
                    self._raise_runtime("TypeError", f"function '{fn.decl.name}' is missing argument '{param.name}'", span)
                self._raise_runtime("TypeError", f"function '{fn.decl.name}' expects {len(params)} args, got {len(args)}", span)
        if kwargs:
            names = [p.name for p in params]
            for name in kwargs:
                if name not in names:
                    self._raise_runtime("TypeError", f"function '{fn.decl.name}' has no parameter named '{name}'", span)
                if names.index(name) < len(args):
                    self._raise_runtime("TypeError", f"function '{fn.decl.name}' got more than one value for '{name}'", span)
        return bound

    def _call_function(self, fn: FunctionValue, args: list[object], kwargs: dict[str, Any] | None = None) -> object:
        if kwargs or len(args) != len(fn.decl.params):
            args = self._bind_arguments(fn, args, kwargs)

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
            env.define(param.name, self._bind_value(value, param.type_name))

        if self._call_depth >= MAX_CALL_DEPTH:
            self._raise_runtime("RecursionError", f"maximum recursion depth exceeded ({MAX_CALL_DEPTH} nested calls)", fn.decl.span)
        self.call_stack.append({"function": fn.decl.name, "span": fn.decl.span, "env": env.values, "owner_type": fn.owner_type})
        self._frame_cells.append([])
        self._call_depth += 1
        try:
            for stmt in fn.decl.body:
                self.execute_stmt(stmt, env, in_loop=False)
        except ReturnSignal as signal:
            return self._bind_value(signal.value, fn.decl.return_type)
        finally:
            self._call_depth -= 1
            self.call_stack.pop()
            self._end_frame()
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
            exc = self._ensure_exception_instance(value, span)
            raise RaiseSignal(exc)
        self._raise_runtime("TypeError", "raised value is not an exception instance", span)

    def _ensure_exception_instance(self, value: InstanceObject, span: Any = None) -> ExceptionObject:
        message = value.attrs.get("message", "")
        if not isinstance(message, str):
            message = str(message)
        code = value.attrs.get("code")
        if code is not None and not isinstance(code, int):
            try:
                code = int(code)
            except Exception:
                code = None
        stack = self._capture_stacktrace(span)
        try:
            return self.heap.allocate(
                "Exception",
                ExceptionObject(type_obj=value.type_obj, message=message, payload=value, code=code, stacktrace=stack),
            )
        except OutOfMemoryError as exc:
            self._raise_runtime("OutOfMemoryError", str(exc), self._span_of(value))

    def _raise_runtime(self, type_name: str, message: str, span: Any, payload: Any = None, code: int | None = None) -> NoReturn:
        t = self.types.get(type_name) or self.types["RuntimeError"]
        stack = self._capture_stacktrace(span)
        try:
            exc = self.heap.allocate("Exception", ExceptionObject(type_obj=t, message=message, payload=payload, code=code, stacktrace=stack))
        except OutOfMemoryError as alloc_err:
            t2 = self.types["RuntimeError"]
            exc = ExceptionObject(type_obj=t2, message=str(alloc_err), payload=None, stacktrace=stack)
        raise RaiseSignal(exc)

    def _capture_stacktrace(self, at: Any = None) -> list[dict[str, Any]]:
        """The call stack, innermost last. `at` is where the error happened: it replaces the
        innermost frame's position (a frame otherwise only knows where its function is declared)."""
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
        line = getattr(at, "line", None)
        if isinstance(line, int) and line > 0:
            here = {"file": getattr(at, "file", self.file), "line": line, "column": getattr(at, "column", 1) or 1}
            if out:
                out[-1].update(here)
            else:
                out.append({"function": "<module>", "callsite_id": None, "hlir_id": None, **here})
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
        if isinstance(value, tuple):
            return self.types["tuple"]
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
        if isinstance(value, (Pointer, StructRef, ArrayRef)):
            return format_memory_value(value, lambda v: self._to_text(v, nested=True))
        if isinstance(value, list):
            return "[" + ", ".join(self._to_text(v, nested=True) for v in value) + "]"
        if isinstance(value, tuple):
            body = ", ".join(self._to_text(v, nested=True) for v in value)
            return "(" + body + ("," if len(value) == 1 else "") + ")"
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

    # ======================================================================= pointers
    def _raise_fault(self, fault: MemoryFault, span: Any) -> NoReturn:
        error = {"null": "NullPointerError", "pointer": "PointerError", "type": "TypeError"}.get(fault.kind, "PointerError")
        self._raise_runtime(error, fault.message, span)

    def _guard(self, action: Any, span: Any) -> Any:
        """Run a memory operation and report a MemoryFault as the matching ManV exception."""
        try:
            return action()
        except MemoryFault as fault:
            self._raise_fault(fault, span)

    def _as_pointer(self, value: Any, span: Any, what: str) -> Pointer:
        if isinstance(value, Pointer):
            return value
        if isinstance(value, ArrayRef):
            return Pointer(value.address, value.elem_type)
        self._raise_runtime("TypeError", f"cannot {what} a {self._runtime_type_name(value)}: it is not a pointer", span)

    def _element_slot(self, target: Any, index: Any, span: Any) -> tuple[Pointer, str]:
        """The address and type of `target[index]` for a pointer or a C array."""
        if isinstance(index, bool) or not isinstance(index, int):
            self._raise_runtime("TypeError", "pointer index must be an int", span)
        if isinstance(target, ArrayRef):
            if not 0 <= index < target.count:
                self._raise_runtime("IndexError", f"array index {index} out of range for an array of {target.count}", span)
            elem = target.elem_type
            return Pointer(target.address + index * self._guard(lambda: self.memory.sizeof(elem), span), elem), elem
        pointee = target.pointee
        size = self._guard(lambda: self.memory.sizeof(pointee), span)
        return Pointer(target.address + index * size, pointee, target.foreign), pointee

    def _own(self, value: Any) -> Any:
        """Struct values are copied when bound to a new name (value semantics, as in C)."""
        if isinstance(value, StructRef):
            return self.memory.copy_struct(value)
        return value

    def _coerce_pointer(self, value: Any, type_name: str | None) -> Any:
        """Implicit conversion at a declared pointer type: `void*` and arrays become `T*`."""
        if type_name and is_pointer_type(type_name):
            target = pointee_of(canonical_pointer_type(type_name))
            if isinstance(value, ArrayRef):
                return Pointer(value.address, target if target != "void" else value.elem_type)
            if isinstance(value, Pointer) and value.pointee != target:
                return value.retyped(target)
        return value

    def _bind_value(self, value: Any, type_name: str | None) -> Any:
        return self._own(self._coerce_pointer(value, type_name))

    def _declared_value(self, stmt: "ast.LetStmt", env: Environment) -> Any:
        """The initial value for `T name [= v]` when T is a pointer or struct type; else a marker."""
        type_name = stmt.type_name or ""
        if is_pointer_type(type_name):
            if stmt.value is None:
                return Pointer(0, pointee_of(canonical_pointer_type(type_name)))
            return self._coerce_pointer(self.eval_expr(stmt.value, env), type_name)
        if type_name in self.memory.structs:
            if stmt.value is None:
                return self.memory.new_struct(type_name)
            return self._own(self.eval_expr(stmt.value, env))
        return _NO_DECLARED

    def _make_c_array(self, stmt: "ast.LetStmt", env: Environment) -> ArrayRef:
        elem = stmt.type_name[len("array["):-1]
        count = self.eval_expr(stmt.array_size, env)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            self._raise_runtime("ValueError", f"array size for '{stmt.name}' must be a non-negative integer", stmt.span)
        ref = self._guard(lambda: self.memory.new_array(elem, count), stmt.span)
        if stmt.value is not None:
            seed = self.eval_expr(stmt.value, env)
            if not isinstance(seed, list):
                self._raise_runtime("TypeError", f"initializer for '{stmt.name}' must be an array literal", stmt.span)
            if len(seed) > count:
                self._raise_runtime("ValueError", f"initializer for '{stmt.name}' has {len(seed)} items, exceeds size {count}", stmt.span)
            size = self._guard(lambda: self.memory.sizeof(elem), stmt.span)
            for i, item in enumerate(seed):
                slot = Pointer(ref.address + i * size, elem)
                self._guard(lambda slot=slot, item=item: self.memory.store(slot, item, elem), stmt.span)
        return ref

    def _end_frame(self) -> None:
        """Leaving a function: its address-taken locals die. A pointer that outlives them now faults
        ("use after free"), and a closure that kept the variable sees its last value."""
        for owner, name, cell in self._frame_cells.pop():
            if owner.values.get(name) is cell:
                try:
                    owner.values[name] = cell.load()
                except MemoryFault:
                    owner.values.pop(name, None)
            self.memory.release(cell.ptr.address)

    def _eval_address_of(self, expr: "ast.AddrOfExpr", env: Environment) -> Pointer:
        operand = expr.operand
        if isinstance(operand, ast.IdentifierExpr):
            owner = env.owner_of(operand.name)
            if owner is None:
                self._raise_runtime("TypeError", f"cannot take the address of '{operand.name}'", expr.span)
            raw = owner.values[operand.name]
            if isinstance(raw, MemCell):
                return Pointer(raw.ptr.address, raw.type_name)
            if isinstance(raw, StructRef):
                return Pointer(raw.address, raw.sdef.name)
            if isinstance(raw, ArrayRef):
                return Pointer(raw.address, raw.elem_type)
            type_name = self._cell_type(raw, expr.pointee, expr.span)
            size = self._guard(lambda: self.memory.sizeof(type_name), expr.span)
            slot = self.memory.alloc(size)
            cell = MemCell(self.memory, slot, type_name)
            self._guard(lambda: cell.store(raw), expr.span)
            owner.values[operand.name] = cell
            self._frame_cells[-1].append((owner, operand.name, cell))
            return Pointer(slot.address, type_name)
        if isinstance(operand, ast.IndexExpr):
            base = self.eval_expr(operand.value, env)
            if not isinstance(base, (Pointer, ArrayRef)):
                self._raise_runtime("TypeError", f"cannot take the address of an element of a {self._runtime_type_name(base)}", expr.span)
            index = self.eval_expr(operand.index, env)
            if isinstance(base, ArrayRef):
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= base.count:
                    self._raise_runtime("IndexError", f"array index {index} out of range for an array of {base.count}", expr.span)
                size = self._guard(lambda: self.memory.sizeof(base.elem_type), expr.span)
                return Pointer(base.address + index * size, base.elem_type)
            slot, slot_type = self._element_slot(base, index, expr.span)
            return Pointer(slot.address, slot_type, base.foreign)
        if isinstance(operand, ast.AttributeExpr):
            base = self.eval_expr(operand.value, env)
            if not isinstance(base, StructRef):
                self._raise_runtime("TypeError", f"cannot take the address of a field of a {self._runtime_type_name(base)}", expr.span)
            ptr, field = self._guard(lambda: self.memory.field_pointer(base, operand.attr), expr.span)
            return Pointer(ptr.address, field.type_name)
        if isinstance(operand, ast.DerefExpr):
            return self._as_pointer(self.eval_expr(operand.operand, env), expr.span, "take the address through")
        self._raise_runtime("TypeError", "cannot take the address of this expression", expr.span)

    def _cell_type(self, value: Any, declared: str | None, span: Any) -> str:
        """The memory type for a variable that is about to be given an address."""
        if declared:
            canonical = canonical_pointer_type(declared) if (declared == "ptr" or declared.endswith("*")) else declared
            if canonical in SCALARS or is_pointer_type(canonical):
                return canonical
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, Pointer):
            return value.pointee + "*"
        self._raise_runtime("TypeError", f"cannot take the address of a {self._runtime_type_name(value)}: only numbers, bools, pointers, structs and C arrays live in raw memory", span)

    def _exec_set_deref(self, stmt: "ast.SetDerefStmt", env: Environment) -> None:
        ptr = self._as_pointer(self.eval_expr(stmt.target, env), stmt.span, "store through")
        value = self.eval_expr(stmt.value, env)
        self._store_result(ptr, ptr.pointee, stmt.op, value, stmt.span)

    def _store_result(self, slot: Pointer, slot_type: str, op: str | None, rhs: Any, span: Any) -> None:
        """`*slot = rhs`, or `*slot op= rhs` when `op` is set."""
        if op is not None:
            current = self._guard(lambda: self.memory.load(slot, slot_type), span)
            if isinstance(current, Pointer):
                rhs = self._pointer_binary(op, current, rhs, span)
            else:
                rhs = self._binary_value(op, current, rhs, span)
        self._guard(lambda: self.memory.store(slot, rhs, slot_type), span)

    def _augmented(self, op: str, current: Any, rhs: Any, span: Any) -> Any:
        if isinstance(current, (Pointer, ArrayRef)) or isinstance(rhs, (Pointer, ArrayRef)):
            return self._pointer_binary(op, current, rhs, span)
        return self._binary_value(op, current, rhs, span)

    def _binary_value(self, op: str, left: Any, right: Any, span: Any) -> Any:
        """`left op right` on plain values, reporting failures as ManV exceptions."""
        try:
            return eval_binary(op, left, right)
        except ZeroDivisionError:
            self._raise_runtime("ZeroDivisionError", "modulo by zero" if op == "%" else "division by zero", span)
        except RaiseSignal:
            raise
        except Exception:
            self._raise_runtime(
                "TypeError",
                f"unsupported operand types for '{op}': {self._runtime_type_name(left)} and {self._runtime_type_name(right)}",
                span,
            )

    def _pointer_binary(self, op: str, left: Any, right: Any, span: Any) -> Any:
        lp = isinstance(left, (Pointer, ArrayRef))
        rp = isinstance(right, (Pointer, ArrayRef))
        if lp:
            left = self._as_pointer(left, span, "use")
        if rp:
            right = self._as_pointer(right, span, "use")
        if op in {"==", "!=", "<", "<=", ">", ">="}:
            if not (lp and rp):
                self._raise_runtime("TypeError", "a pointer can only be compared with another pointer (use null for no pointer)", span)
            a, b = left.address, right.address
            return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]

        def scale(ptr: Pointer) -> int:
            if ptr.pointee == "void":
                self._raise_runtime("TypeError", "arithmetic on a void* is not allowed; cast it to u8* first", span)
            return self._guard(lambda: self.memory.sizeof(ptr.pointee), span)

        def whole(n: Any) -> int:
            if isinstance(n, bool) or not isinstance(n, int):
                self._raise_runtime("TypeError", "pointer arithmetic needs an int", span)
            return n

        if op == "+" and lp != rp:
            ptr, n = (left, right) if lp else (right, left)
            return Pointer(ptr.address + whole(n) * scale(ptr), ptr.pointee, ptr.foreign)
        if op == "-" and lp and not rp:
            return Pointer(left.address - whole(right) * scale(left), left.pointee, left.foreign)
        if op == "-" and lp and rp:
            size = scale(left)
            if size != scale(right):
                self._raise_runtime("TypeError", "cannot subtract pointers to types of different sizes", span)
            return (left.address - right.address) // size
        self._raise_runtime("TypeError", f"operator '{op}' is not defined for pointers", span)

    def _cast(self, value: Any, type_name: str, span: Any) -> Any:
        target = canonical_pointer_type(type_name) if (type_name == "ptr" or type_name.endswith("*")) else type_name.replace("const ", "").strip()
        if is_pointer_type(target):
            pointee = pointee_of(target)
            if isinstance(value, ArrayRef):
                return Pointer(value.address, pointee)
            if isinstance(value, Pointer):
                return value.retyped(pointee)
            if isinstance(value, int) and not isinstance(value, bool):
                return Pointer(value, pointee) if value else Pointer(0, pointee)
            self._raise_runtime("TypeError", f"cannot cast a {self._runtime_type_name(value)} to {type_name}", span)
        if isinstance(value, (Pointer, ArrayRef)):
            address = value.address
            if target in INT_TYPES:
                return self._wrap_int(address, target)
            self._raise_runtime("TypeError", f"cannot cast a pointer to {type_name}; use usize or a pointer type", span)
        if target in INT_TYPES:
            if isinstance(value, (int, float, bool)):
                return self._wrap_int(int(value), target)
        elif target in FLOAT_TYPES:
            if isinstance(value, (int, float, bool)):
                return float(value)
        elif target == "bool":
            return bool(value)
        elif target == "void":
            return None
        self._raise_runtime("TypeError", f"cannot cast a {self._runtime_type_name(value)} to {type_name}", span)

    @staticmethod
    def _wrap_int(value: int, type_name: str) -> int:
        bits = SCALARS[type_name][0] * 8
        value &= (1 << bits) - 1
        if SCALARS[type_name][1].islower() and value >= 1 << (bits - 1):
            value -= 1 << bits
        return value

    def _eval_sizeof(self, expr: "ast.SizeofExpr", env: Environment) -> int:
        if expr.type_name is not None:
            owner = env.owner_of(expr.type_name) if expr.operand is not None else None
            is_variable = owner is not None and not isinstance(owner.values[expr.type_name], (StructType, TypeObject))
            if not is_variable:
                return self._guard(lambda: self.memory.sizeof(expr.type_name), expr.span)
        value = self.eval_expr(expr.operand, env)
        if isinstance(value, StructRef):
            return value.sdef.size
        if isinstance(value, ArrayRef):
            return value.count * self._guard(lambda: self.memory.sizeof(value.elem_type), expr.span)
        if isinstance(value, Pointer):
            return 8
        if isinstance(value, bool):
            return 1
        if isinstance(value, (int, float)):
            return 8
        self._raise_runtime("TypeError", f"sizeof does not apply to a {self._runtime_type_name(value)}; it is not raw memory", expr.span)

    # ----------------------------------------------------------------------- structs
    def _define_struct(self, decl: "ast.StructDecl") -> None:
        fields: list[tuple[str, str, int | None]] = []
        for f in decl.fields:
            count = None
            if f.count is not None:
                count = self.eval_expr(f.count, self.global_env)
            type_name = f.type_name.replace("const ", "").strip()
            fields.append((f.name, canonical_pointer_type(type_name) if type_name.endswith("*") else type_name, count))
        self._guard(lambda: self.memory.define_struct(decl.name, fields), decl.span)
        self.global_env.define(decl.name, StructType(decl.name))

    def _struct_get(self, ref: StructRef, attr: str) -> Any:
        ptr, field = self.memory.field_pointer(ref, attr)
        if field.count is not None:
            return ArrayRef(self.memory, ptr.address, field.type_name, field.count, owned=False)
        return self.memory.load(ptr, field.type_name)

    def _struct_set(self, ref: StructRef, attr: str, value: Any) -> None:
        ptr, field = self.memory.field_pointer(ref, attr)
        if field.count is not None:
            raise MemoryFault("type", f"assign the elements of '{attr}' one at a time")
        if isinstance(value, Pointer) and is_pointer_type(field.type_name):
            value = value.retyped(pointee_of(field.type_name))
        self.memory.store(ptr, value, field.type_name)

    def _construct_struct(self, struct_type: StructType, args: list[Any], kwargs: dict[str, Any], span: Any) -> StructRef:
        sdef = self.memory.structs[struct_type.name]
        if len(args) > len(sdef.fields):
            self._raise_runtime("TypeError", f"{sdef.name}() takes at most {len(sdef.fields)} argument(s), got {len(args)}", span)
        ref = self.memory.new_struct(sdef.name)
        names = [f.name for f in sdef.fields]
        for name in kwargs:
            if name not in names:
                self._raise_runtime("TypeError", f"struct {sdef.name} has no field '{name}'", span)
            if names.index(name) < len(args):
                self._raise_runtime("TypeError", f"{sdef.name}() got more than one value for '{name}'", span)
        for name, value in [*zip(names, args), *kwargs.items()]:
            self._guard(lambda name=name, value=value: self._struct_set(ref, name, value), span)
        return ref

    # ----------------------------------------------------------------------- builtins
    def _install_memory_builtins(self) -> None:
        mem = self.memory

        def size_arg(value: Any, what: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise MemoryFault("type", f"{what} must be an int")
            return value

        def ptr_arg(value: Any, what: str) -> Pointer:
            if isinstance(value, Pointer):
                return value
            if isinstance(value, ArrayRef):
                return Pointer(value.address, value.elem_type)
            raise MemoryFault("type", f"{what} must be a pointer")

        def malloc(n: Any) -> Pointer:
            return mem.alloc(size_arg(n, "size"), heap=True)

        def calloc(count: Any, size: Any) -> Pointer:
            return mem.alloc(size_arg(count, "count") * size_arg(size, "size"), heap=True)

        def realloc(p: Any, n: Any) -> Pointer:
            return mem.realloc(ptr_arg(p, "pointer"), size_arg(n, "size"))

        def free(p: Any) -> None:
            mem.free(ptr_arg(p, "pointer"))

        def memcpy(dst: Any, src: Any, n: Any) -> Pointer:
            d, s, count = ptr_arg(dst, "destination"), ptr_arg(src, "source"), size_arg(n, "size")
            mem.move(d, s, count)
            return d

        def memset(dst: Any, byte: Any, n: Any) -> Pointer:
            d = ptr_arg(dst, "destination")
            mem.fill(d, size_arg(byte, "byte"), size_arg(n, "size"))
            return d

        def memcmp(a: Any, b: Any, n: Any) -> int:
            return mem.compare(ptr_arg(a, "first"), ptr_arg(b, "second"), size_arg(n, "size"))

        def cstr(s: Any) -> Pointer:
            if not isinstance(s, str):
                raise MemoryFault("type", "cstr() needs a string")
            data = s.encode("utf-8")
            ptr = mem.alloc(len(data) + 1, heap=True)
            mem.write_bytes(ptr.address, data + b"\0")
            return ptr.retyped("u8")

        def str_from(p: Any, n: Any = None) -> str:
            ptr = ptr_arg(p, "pointer")
            raw = mem.read_cstring(ptr, None if n is None else size_arg(n, "length"))
            return raw.decode("utf-8", errors="replace")

        for name, fn in {
            "malloc": malloc, "calloc": calloc, "realloc": realloc, "free": free,
            "memcpy": memcpy, "memmove": memcpy, "memset": memset, "memcmp": memcmp,
            "cstr": cstr, "str_from": str_from,
        }.items():
            fn.__name__ = name
            self.global_env.define(name, fn)

    def _install_host_builtins(self) -> None:
        """`input`, `exit`, `args` and `env`: the program's connection to its surroundings."""

        def input_(prompt: Any = "") -> str:
            if not isinstance(prompt, str):
                raise TypeError("input() prompt must be a string")
            if prompt:
                self.stdout.write(prompt)
                getattr(self.stdout, "flush", lambda: None)()
            source = self.stdin or sys.stdin
            line = source.readline()
            if line == "":
                self._raise_runtime("EOFError", "input() reached the end of the input", None)
            return line.rstrip("\r\n")

        def exit_(code: Any = 0) -> None:
            if isinstance(code, bool) or not isinstance(code, int):
                raise TypeError("exit() needs an int exit code")
            raise ExitSignal(code)

        def args_() -> list[str]:
            return list(self.argv)

        def env_(name: Any, default: Any = None) -> Any:
            if not isinstance(name, str):
                raise TypeError("env() needs a variable name")
            return os.environ.get(name, default)

        for name, fn in {"input": input_, "exit": exit_, "args": args_, "env": env_}.items():
            fn.__name__ = name
            self.global_env.define(name, fn)

    def _span_of(self, node: Any) -> Any:
        return getattr(node, "span", type("_S", (), {"line": 1, "column": 1})())
