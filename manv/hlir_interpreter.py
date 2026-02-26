from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, TextIO

from .hlir import HFunction, HInstruction, HModule
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


class HLIRInterpreter:
    def __init__(self, stdout: TextIO | None = None, tracer: HLIRTracer | None = None):
        self.stdout = stdout or io.StringIO()
        self.tracer = tracer
        self.functions: dict[str, HFunction] = {}
        self.call_stack: list[HLIRFrame] = []

    def run_module(self, module: HModule, entry: str = "main") -> HLIRExecutionResult:
        self.functions = {fn.name: fn for fn in module.functions}
        self.call_stack = []
        if "__top_level" in self.functions:
            self._call("__top_level", [])
        value = self._call(entry, [])
        return HLIRExecutionResult(value=value, stdout=self.stdout.getvalue() if isinstance(self.stdout, io.StringIO) else "")

    def _call(self, name: str, args: list[Any]) -> Any:
        depth = len(self.call_stack)
        if self.tracer is not None:
            self.tracer.on_call(name, args, depth)

        if name == "print":
            self.stdout.write(" ".join(str(a) for a in args) + "\n")
            if self.tracer is not None:
                self.tracer.on_return(name, None, depth)
            return None
        if name == "len":
            if len(args) != 1:
                raise RuntimeError("len expects one argument")
            out = len(args[0])
            if self.tracer is not None:
                self.tracer.on_return(name, out, depth)
            return out

        if name not in self.functions:
            raise RuntimeError(f"undefined function: {name}")

        fn = self.functions[name]
        out = self._execute_function(fn, args)
        if self.tracer is not None:
            self.tracer.on_return(name, out, depth)
        return out

    def _execute_function(self, fn: HFunction, args: list[Any]) -> Any:
        blocks = {b.label: b for b in fn.blocks}
        frame = HLIRFrame(function=fn.name, block=fn.entry, args=list(args))
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
                        result = self._eval_instruction(instr, arg_vals, frame.values, frame.vars_mem, args)
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
                if term.op == "ret":
                    if not term.args:
                        return None
                    return self._resolve_value(term.args[0], frame.values, frame.vars_mem)
                if term.op == "unreachable":
                    raise RuntimeError("entered unreachable terminator")
                raise RuntimeError(f"unsupported terminator: {term.op}")
        finally:
            self.call_stack.pop()

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
                return {"__std__": True}
            if name not in vars_mem:
                raise RuntimeError(f"undefined variable: {name}")
            return vars_mem[name]
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
            if isinstance(base, dict) and base.get("__std__") is True and attr in {"gpu", "memory"}:
                return {"__stub_module__": attr}
            raise RuntimeError(f"unsupported attribute: {attr}")
        if op == "call":
            callee = str(instr.attrs.get("callee", ""))
            if callee == "<dynamic>":
                raise RuntimeError("dynamic calls are not supported in HLIR")
            return self._call(callee, list(resolved_args))
        if op == "unsupported_stmt":
            raise RuntimeError(f"unsupported statement in HLIR: {instr.attrs.get('kind')}")
        raise RuntimeError(f"unsupported instruction: {op}")
