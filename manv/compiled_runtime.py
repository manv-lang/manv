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
from .graph_ir import lower_hlir_to_graph
from .graph_opt import optimize_graph_ir
from .hlir import HModule
from .hlir_interpreter import HLIRInterpreter
from .hlir_lowering import lower_ast_to_hlir
from .host_stub import build_host_stubs
from .interpreter import Interpreter, RaiseSignal
from .kernel_ir import lower_graph_to_kernel
from .kernel_mock import execute_kernel_ir
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
    backend: str = "auto",
    device: str | None = None,
    optimize: bool = True,
    capture: bool = False,
    deterministic_gc: bool = False,
    gc_stress: bool = False,
    stable_debug_format: bool = False,
) -> CompiledRunResult:
    target = get_target(target_name)
    uses_gpu_decorators = _program_uses_gpu_decorators(program)

    out_stream = stdout or io.StringIO()
    hlir_module = lower_ast_to_hlir(program, source_name)
    if uses_gpu_decorators:
        try:
            result = HLIRInterpreter(stdout=out_stream, preferred_backend=backend, preferred_device=device).run_module(
                hlir_module, entry="main"
            )
            exit_code = int(result.value) if isinstance(result.value, int) else 0
        except RuntimeError as err:
            raise ManvError(diag("E3900", str(err), source_name, 1, 1)) from None
    else:
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

    tracer = GraphCaptureTracer() if capture else None
    if capture:
        sink = io.StringIO()
        HLIRInterpreter(stdout=sink, tracer=tracer, preferred_backend=backend, preferred_device=device).run_module(
            hlir_module, entry="main"
        )

    graph = tracer.to_graph_ir() if tracer else lower_hlir_to_graph(hlir_module)
    if optimize:
        graph = optimize_graph_ir(graph)

    kernel = lower_graph_to_kernel(graph)
    kernel_exec = execute_kernel_ir(kernel)

    gpu_dispatch = None
    try:
        gpu_dispatch = dispatch_kernel_ir(
            kernel,
            backend=backend,
            target=target_name,
            strict_verify=False,
            device=device,
        ).to_dict()
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


def _program_uses_gpu_decorators(program: ast.Program) -> bool:
    for decl in program.declarations:
        if isinstance(decl, ast.FnDecl) and decl.decorators:
            return True
        if isinstance(decl, ast.TypeDecl) and any(method.decorators for method in decl.methods):
            return True
        if isinstance(decl, ast.ImplDecl) and any(method.decorators for method in decl.methods):
            return True
    return False
