"""Reference executor for HLIR modules.

Why this file exists:
- Provides an executable semantic oracle for lowered HLIR functions.
- Enables parity checks between AST semantics, HLIR lowering, and compiled
  artifact generation.
- Supplies tracing hooks used by graph capture and debugging workflows.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, TextIO

from .gpu_execution import GpuExecutionEngine
from .hlir import HFunction, HInstruction, HModule
from .intrinsics import BUILTIN_ALIASES, IntrinsicNamespace, StdNamespace, invoke_intrinsic, std_namespace_attr
from .semantics_core import eval_binary, eval_unary


@dataclass
class HLIRExecutionResult:
    value: Any
    stdout: str


@dataclass
class HLIRFrame:
    function: str
    block: str
    args: list[Any]
    values: dict[str, Any] = field(default_factory=dict)
    vars_mem: dict[str, Any] = field(default_factory=dict)
    current_instr_id: str | None = None
    force_cpu: bool = False


class HLIRTracer:
    """Observer API for runtime instrumentation/debugging over HLIR execution."""

    def on_call(self, fn_name: str, args: list[Any], depth: int) -> None:
        return

    def on_return(self, fn_name: str, value: Any, depth: int) -> None:
        return

    def before_instruction(
        self,
        frame: HLIRFrame,
        block_label: str,
        instr: HInstruction,
        resolved_args: list[Any],
    ) -> None:
        return

    def on_instruction(
        self,
        fn_name: str,
        block_label: str,
        instr: HInstruction,
        resolved_args: list[Any],
        result: Any,
    ) -> None:
        return

    def on_exception(self, frame: HLIRFrame, block_label: str, instr: HInstruction, error: Exception) -> None:
        return

    def on_gpu_execution(self, fn_name: str, details: dict[str, Any]) -> None:
        return


class HLIRInterpreter:
    def __init__(
        self,
        stdout: TextIO | None = None,
        tracer: HLIRTracer | None = None,
        *,
        preferred_backend: str = "auto",
        preferred_device: str | None = None,
    ):
        self.stdout = stdout or io.StringIO()
        self.tracer = tracer
        self.preferred_backend = preferred_backend
        self.preferred_device = preferred_device
        self.functions: dict[str, HFunction] = {}
        self.call_stack: list[HLIRFrame] = []
        self.gpu_engine: GpuExecutionEngine | None = None

    def run_module(self, module: HModule, entry: str = "main") -> HLIRExecutionResult:
        self.functions = {fn.name: fn for fn in module.functions}
        self.call_stack = []
        self.gpu_engine = GpuExecutionEngine(
            module,
            cpu_fallback=lambda name, args: self._call(name, args, force_cpu=True),
            on_gpu_execution=None,
            preferred_backend=self.preferred_backend,
            preferred_device=self.preferred_device,
        )
        if self.tracer is not None:
            self.gpu_engine.on_gpu_execution = lambda details: self.tracer.on_gpu_execution(str(details.get("function", "")), details)
        if "__top_level" in self.functions:
            self._call("__top_level", [])
        value = self._call(entry, [])
        return HLIRExecutionResult(value=value, stdout=self.stdout.getvalue() if isinstance(self.stdout, io.StringIO) else "")

    def _call(self, name: str, args: list[Any], *, force_cpu: bool = False) -> Any:
        depth = len(self.call_stack)
        if self.tracer is not None:
            self.tracer.on_call(name, args, depth)

        if name == "type":
            if len(args) != 1:
                raise RuntimeError("type() expects exactly 1 argument")
            out = type(args[0]).__name__
            if self.tracer is not None:
                self.tracer.on_return(name, out, depth)
            return out
        if name == "isinstance":
            if len(args) != 2:
                raise RuntimeError("isinstance() expects exactly 2 arguments")
            rhs = args[1]
            if isinstance(rhs, str):
                out = type(args[0]).__name__ == rhs
            else:
                out = isinstance(args[0], rhs)
            if self.tracer is not None:
                self.tracer.on_return(name, out, depth)
            return out
        if name == "issubclass":
            if len(args) != 2:
                raise RuntimeError("issubclass() expects exactly 2 arguments")
            lhs = args[0]
            rhs = args[1]
            if not isinstance(lhs, type) or not isinstance(rhs, type):
                raise RuntimeError("issubclass() args must be class objects in HLIR")
            out = issubclass(lhs, rhs)
            if self.tracer is not None:
                self.tracer.on_return(name, out, depth)
            return out
        if name == "id":
            if len(args) != 1:
                raise RuntimeError("id() expects exactly 1 argument")
            out = int(id(args[0]))
            if self.tracer is not None:
                self.tracer.on_return(name, out, depth)
            return out

        alias = BUILTIN_ALIASES.get(name)
        if alias is not None:
            payload = [args] if alias == "io_print" else list(args)
            out = self._invoke_intrinsic(alias, payload)
            if self.tracer is not None:
                self.tracer.on_return(name, out, depth)
            return out

        if name not in self.functions:
            raise RuntimeError(f"undefined function: {name}")

        fn = self.functions[name]
        out = self._execute_function(fn, args, force_cpu=force_cpu)
        if self.tracer is not None:
            self.tracer.on_return(name, out, depth)
        return out

    def _execute_function(self, fn: HFunction, args: list[Any], *, force_cpu: bool = False) -> Any:
        blocks = {b.label: b for b in fn.blocks}
        frame = HLIRFrame(function=fn.name, block=fn.entry, args=list(args), force_cpu=force_cpu)
        self.call_stack.append(frame)

        try:
            current = fn.entry
            while True:
                frame.block = current
                block = blocks[current]
                for instr in block.instructions:
                    frame.current_instr_id = instr.instr_id
                    arg_vals = [
                        frame.values[a] if a.startswith("%") and a in frame.values else frame.vars_mem.get(a, a)
                        for a in instr.args
                    ]

                    if self.tracer is not None:
                        self.tracer.before_instruction(frame, block.label, instr, arg_vals)

                    try:
                        result = self._eval_instruction(instr, arg_vals, frame.values, frame.vars_mem, args, frame)
                    except Exception as exc:  # pragma: no cover - delegated to debugger path
                        if self.tracer is not None:
                            self.tracer.on_exception(frame, block.label, instr, exc)
                        raise

                    if instr.dest is not None:
                        frame.values[instr.dest] = result
                    if self.tracer is not None:
                        self.tracer.on_instruction(fn.name, block.label, instr, arg_vals, result)

                if block.terminator is None:
                    return None

                term = block.terminator
                if term.op == "br":
                    current = term.args[0]
                    continue
                if term.op == "cbr":
                    cond_value = self._resolve_value(term.args[0], frame.values, frame.vars_mem)
                    current = term.args[1] if bool(cond_value) else term.args[2]
                    continue
                if term.op == "invoke":
                    if len(term.args) < 2:
                        raise RuntimeError("invalid invoke terminator")
                    normal_label = term.args[-2]
                    unwind_label = term.args[-1]
                    call_arg_tokens = term.args[:-2]
                    call_arg_values = [self._resolve_value(tok, frame.values, frame.vars_mem) for tok in call_arg_tokens]
                    attrs = term.attrs or {}
                    kind = str(attrs.get("kind", "call"))
                    try:
                        if kind == "intrinsic":
                            name = str(attrs.get("name", ""))
                            result = self._invoke_intrinsic(name, call_arg_values)
                        elif kind == "gpu_call":
                            if self.gpu_engine is None:
                                raise RuntimeError("gpu engine is not initialized")
                            result, _ = self.gpu_engine.execute(
                                callee=str(attrs.get("callee", "")),
                                args=call_arg_values,
                                policy=str(attrs.get("policy", "best_effort")),
                                mode=str(attrs.get("mode", "kernel")),
                            )
                        else:
                            callee = str(attrs.get("callee", ""))
                            result = self._call(callee, call_arg_values)
                        dest = attrs.get("dest")
                        if isinstance(dest, str) and dest:
                            frame.values[dest] = result
                        current = normal_label
                    except Exception as exc:
                        frame.vars_mem["__eh_exception__"] = exc
                        current = unwind_label
                    continue
                if term.op == "raise":
                    if not term.args or term.args[0] == "__reraise__":
                        exc = frame.vars_mem.get("__eh_exception__")
                        if exc is None:
                            raise RuntimeError("bare raise outside exception handler")
                    else:
                        value = self._resolve_value(term.args[0], frame.values, frame.vars_mem)
                        exc = value if isinstance(value, Exception) else RuntimeError(str(value))
                    raise exc
                if term.op == "ret":
                    if not term.args:
                        return None
                    return self._resolve_value(term.args[0], frame.values, frame.vars_mem)
                if term.op == "unreachable":
                    raise RuntimeError("entered unreachable terminator")
                raise RuntimeError(f"unsupported terminator: {term.op}")
        finally:
            self.call_stack.pop()

    def _invoke_intrinsic(self, name: str, args: list[Any]) -> Any:
        return invoke_intrinsic(
            name,
            args,
            stdout_write=self.stdout.write,
            stdin_readline=lambda: "",
            gc_hooks={},
        )

    def _resolve_value(self, token: str, values: dict[str, Any], vars_mem: dict[str, Any]) -> Any:
        if token.startswith("%"):
            return values[token]
        if token in vars_mem:
            return vars_mem[token]
        return token

    def _eval_instruction(
        self,
        instr: HInstruction,
        resolved_args: list[Any],
        values: dict[str, Any],
        vars_mem: dict[str, Any],
        call_args: list[Any],
        frame: HLIRFrame,
    ) -> Any:
        op = instr.op
        if op == "const":
            return instr.attrs.get("value")
        if op == "load_arg":
            idx = int(instr.attrs.get("index", 0))
            if idx >= len(call_args):
                return None
            return call_args[idx]
        if op == "declare_var":
            vars_mem[str(instr.attrs.get("name"))] = None
            return None
        if op == "store_var":
            name = instr.args[0]
            value = resolved_args[1]
            vars_mem[name] = value
            return value
        if op == "load_var":
            name = instr.args[0]
            if name == "std":
                return StdNamespace()
            if name == "__intrin":
                return IntrinsicNamespace()
            if name not in vars_mem:
                raise RuntimeError(f"undefined variable: {name}")
            return vars_mem[name]
        if op == "set_exception":
            vars_mem["__eh_exception__"] = resolved_args[0] if resolved_args else None
            return None
        if op == "load_exception":
            return vars_mem.get("__eh_exception__")
        if op == "exc_match":
            err = resolved_args[0] if resolved_args else None
            type_name = str(instr.attrs.get("type", "Exception"))
            return self._exc_match(err, type_name)
        if op in {"finally_enter", "finally_exit"}:
            return None
        if op == "alloc_array":
            n = int(resolved_args[0])
            return [None] * n
        if op == "array_init_sized":
            seed, size = resolved_args
            n = int(size)
            if not isinstance(seed, list):
                raise RuntimeError("array initializer must be list")
            if len(seed) > n:
                raise RuntimeError("initializer exceeds static size")
            return seed + [None] * (n - len(seed))
        if op == "unary":
            return eval_unary(str(instr.attrs.get("op")), resolved_args[0])
        if op == "binop":
            return eval_binary(str(instr.attrs.get("op")), resolved_args[0], resolved_args[1])
        if op == "array":
            return list(resolved_args)
        if op == "map":
            out: dict[Any, Any] = {}
            for i in range(0, len(resolved_args), 2):
                out[resolved_args[i]] = resolved_args[i + 1]
            return out
        if op == "index":
            target, index = resolved_args
            return target[index]
        if op == "set_index":
            target, index, value = resolved_args
            target[index] = value
            return None
        if op == "attr":
            base = resolved_args[0]
            attr = str(instr.attrs.get("attr"))
            namespaced = std_namespace_attr(base, attr)
            if namespaced is not None:
                return namespaced
            if isinstance(base, dict):
                if attr in base:
                    return base[attr]
                raise RuntimeError(f"missing attribute '{attr}'")
            raise RuntimeError(f"unsupported attribute: {attr}")
        if op == "import":
            # HLIR execution currently models import values symbolically.
            # Full module loading remains AST-interpreter-authoritative in v1.
            return {"__module__": str(instr.attrs.get("module", "")), "__level__": int(instr.attrs.get("level", 0) or 0)}
        if op == "from_import":
            module = str(instr.attrs.get("module", ""))
            name = str(instr.attrs.get("name", ""))
            # Preserve imported symbol metadata for traceability/testing.
            return {"__from__": module, "__name__": name, "__level__": int(instr.attrs.get("level", 0) or 0)}
        if op == "intrinsic_call":
            name = str(instr.attrs.get("name", ""))
            return self._invoke_intrinsic(name, list(resolved_args))
        if op == "gpu_call":
            callee = str(instr.attrs.get("callee", ""))
            if frame.force_cpu:
                return self._call(callee, list(resolved_args), force_cpu=True)
            if self.gpu_engine is None:
                raise RuntimeError("gpu engine is not initialized")
            value, _ = self.gpu_engine.execute(
                callee=callee,
                args=list(resolved_args),
                policy=str(instr.attrs.get("policy", "best_effort")),
                mode=str(instr.attrs.get("mode", "kernel")),
            )
            return value
        if op == "call":
            callee = str(instr.attrs.get("callee", ""))
            if callee == "<dynamic>":
                raise RuntimeError("dynamic calls are not supported in HLIR")
            return self._call(callee, list(resolved_args), force_cpu=frame.force_cpu)
        if op == "unsupported_stmt":
            raise RuntimeError(f"unsupported statement in HLIR: {instr.attrs.get('kind')}")
        raise RuntimeError(f"unsupported instruction: {op}")

    def _exc_match(self, err: Any, type_name: str) -> bool:
        if err is None:
            return False
        if type_name in {"BaseException", "Exception"}:
            return isinstance(err, Exception)
        mapping: dict[str, type[Exception]] = {
            "TypeError": TypeError,
            "ValueError": ValueError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "RuntimeError": RuntimeError,
            "OSError": OSError,
        }
        exc_cls = mapping.get(type_name)
        if exc_cls is None:
            # Unknown language-level type in HLIR path: match by class name conservatively.
            return err.__class__.__name__ == type_name
        return isinstance(err, exc_cls)
