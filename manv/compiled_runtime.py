from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, TextIO

from . import ast
from .abi import ABIFunction, lower_function_abi
from .cpu_codegen import emit_target_assembly
from .diagnostics import ManvError, diag
from .gpu_dispatch import dispatch_kernel_ir
from .graph_capture import GraphCaptureTracer
from .graph_ir import lower_hir_to_graph
from .graph_opt import optimize_graph_ir
from .hlir import HModule
from .hlir_interpreter import HLIRInterpreter
from .hlir_lowering import lower_ast_to_hlir
from .host_stub import build_host_stubs
from .interpreter import Interpreter, RaiseSignal
from .kernel_ir import lower_graph_to_kernel
from .kernel_mock import execute_kernel_ir
from .lowering import lower_ast_to_hir
from .targets import TargetSpec, get_target


@dataclass
class CompiledArtifacts:
    hlir: dict[str, Any]
    abi: dict[str, Any]
    asm: str
    graph: dict[str, Any]
    kernel: dict[str, Any]
    host_stub_abi: dict[str, Any]
    host_stub_asm: str
    capture_graph: dict[str, Any] | None
    kernel_exec: dict[str, Any]
    gpu_dispatch: dict[str, Any] | None


@dataclass
class CompiledRunResult:
    exit_code: int
    artifacts: CompiledArtifacts
    stdout: str


def compile_and_run_program(
    program: ast.Program,
    source_name: str,
    stdout: TextIO | None = None,
    target_name: str = "x86_64-sysv",
    optimize: bool = True,
    capture: bool = False,
    deterministic_gc: bool = False,
    gc_stress: bool = False,
    stable_debug_format: bool = False,
) -> CompiledRunResult:
    target = get_target(target_name)

    # In v1 compiled mode, artifacts are emitted but execution remains interpreter-authoritative.
    out_stream = stdout or io.StringIO()
    ast_interp = Interpreter(
        file=source_name,
        stdout=out_stream,
        deterministic_gc=deterministic_gc,
        gc_stress=gc_stress,
        stable_debug_format=stable_debug_format,
    )
    try:
        exit_code = ast_interp.run_main(program)
    except RaiseSignal as rs:
        frame = rs.error.stacktrace[-1] if rs.error.stacktrace else {"line": 1, "column": 1}
        raise ManvError(
            diag(
                "E3900",
                f"{rs.error.type_obj.name}: {rs.error.message}",
                source_name,
                int(frame.get("line", 1)),
                int(frame.get("column", 1)),
            )
        ) from None

    hlir_module = lower_ast_to_hlir(program, source_name)
    tracer = GraphCaptureTracer() if capture else None
    if capture:
        sink = io.StringIO()
        HLIRInterpreter(stdout=sink, tracer=tracer).run_module(hlir_module, entry="main")

    hir_module = lower_ast_to_hir(program, source_name)
    graph = tracer.to_graph_ir() if tracer else lower_hir_to_graph(hir_module)
    if optimize:
        graph = optimize_graph_ir(graph)

    kernel = lower_graph_to_kernel(graph)
    kernel_exec = execute_kernel_ir(kernel)

    gpu_dispatch = None
    try:
        gpu_dispatch = dispatch_kernel_ir(kernel, backend="auto", target=target_name, strict_verify=False).to_dict()
    except Exception:
        gpu_dispatch = None

    abi_map = _lower_module_abi(hlir_module, target)
    asm_text = emit_target_assembly(hlir_module, target, abi_map)

    host_abi, host_stub_asm = build_host_stubs(kernel, target)

    artifacts = CompiledArtifacts(
        hlir=hlir_module.to_dict(),
        abi={name: spec.to_dict() for name, spec in abi_map.items()},
        asm=asm_text,
        graph=graph,
        kernel=kernel,
        host_stub_abi={name: spec.to_dict() for name, spec in host_abi.items()},
        host_stub_asm=host_stub_asm,
        capture_graph=tracer.to_graph_ir() if tracer else None,
        kernel_exec=kernel_exec,
        gpu_dispatch=gpu_dispatch,
    )

    out_value = out_stream.getvalue() if isinstance(out_stream, io.StringIO) else ""
    return CompiledRunResult(exit_code=exit_code, artifacts=artifacts, stdout=out_value)


def _lower_module_abi(module: HModule, target: TargetSpec) -> dict[str, ABIFunction]:
    abi_map: dict[str, ABIFunction] = {}
    for fn in module.functions:
        param_types = [str(p.get("type")) if p.get("type") is not None else None for p in fn.params]
        abi_map[fn.name] = lower_function_abi(fn.name, param_types, fn.return_type, target)
    return abi_map
