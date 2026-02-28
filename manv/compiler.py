from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterable

from . import ast
from .abi import lower_function_abi
from .cpu_codegen import emit_target_assembly
from .debug_mapping import build_source_map_from_hlir
from .diagnostics import ManvError, diag
from .backends.cuda import analyze_hlir_gpu_function
from .device import normalize_backend_name
from .gpu_backends import compile_kir_backend
from .graph_capture import GraphCaptureTracer
from .graph_ir import lower_hlir_to_graph
from .graph_opt import optimize_graph_ir
from .host import HostSelectionRequest, resolve_host_selection
from .hlir_lowering import lower_ast_to_hlir
from .hlir_interpreter import HLIRInterpreter
from .host_stub import build_host_stubs
from .kernel_ir import lower_graph_to_kernel
from .kernel_mock import execute_kernel_ir
from .lexer import Lexer
from .lowering import lower_ast_to_hir
from .llvm_codegen import LlvmLoweringError, emit_llvm_module
from .llvm_toolchain import build_llvm_artifacts
from .native_toolchain import build_native_artifacts
from .parser import Parser
from .semantics import SemanticAnalyzer
from .targets import get_target


def parse_program(source: str, file: str) -> ast.Program:
    lexer = Lexer(source=source, file=file)
    tokens = lexer.tokenize()
    parser = Parser(tokens=tokens, file=file, source_lines=source.splitlines())
    return parser.parse()


def analyze_program(program: ast.Program, file: str) -> None:
    analyzer = SemanticAnalyzer(file=file)
    result = analyzer.analyze(program)
    analyzer.assert_valid(result)


def compile_pipeline(source: str, file: str, optimize: bool = True) -> tuple[dict, dict, dict, dict]:
    artifacts = compile_pipeline_full(source, file, optimize=optimize)
    return (
        artifacts["ast"],
        artifacts["hir"],
        artifacts["graph"],
        artifacts["kernel"],
    )


def compile_pipeline_full(
    source: str,
    file: str,
    *,
    optimize: bool = True,
    target_name: str = "x86_64-sysv",
    capture_graph: bool = False,
) -> dict[str, Any]:
    program = parse_program(source, file)
    analyze_program(program, file)

    ast_ir = ast.to_dict(program)
    hir_module = lower_ast_to_hir(program, source_name=file)
    hir_ir = hir_module.to_dict()

    hlir_module = lower_ast_to_hlir(program, source_name=file)
    hlir_ir = hlir_module.to_dict()
    gpu_report = {
        fn.name: analyze_hlir_gpu_function(fn).to_dict()
        for fn in hlir_module.functions
        if isinstance(fn.attrs, dict) and "gpu" in fn.attrs
    }
    source_map = build_source_map_from_hlir(hlir_module)

    capture_ir: dict[str, Any] | None = None
    if capture_graph:
        tracer = GraphCaptureTracer()
        sink = io.StringIO()
        HLIRInterpreter(stdout=sink, tracer=tracer).run_module(hlir_module, entry="main")
        graph_ir = tracer.to_graph_ir()
        capture_ir = graph_ir
    else:
        graph_ir = lower_hlir_to_graph(hlir_module)

    if optimize:
        graph_ir = optimize_graph_ir(graph_ir)

    kernel_ir = lower_graph_to_kernel(graph_ir)
    kernel_exec = execute_kernel_ir(kernel_ir)

    target = get_target(target_name)
    abi_map = {}
    for fn in hlir_module.functions:
        param_types = [str(p.get("type")) if p.get("type") is not None else None for p in fn.params]
        abi_map[fn.name] = lower_function_abi(fn.name, param_types, fn.return_type, target)
    asm_text = emit_target_assembly(hlir_module, target, abi_map)

    host_stub_abi, host_stub_asm = build_host_stubs(kernel_ir, target)

    return {
        "ast": ast_ir,
        "hir": hir_ir,
        "hlir": hlir_ir,
        "_hlir_module": hlir_module,
        "gpu_report": gpu_report,
        "source_map": source_map.to_dict(),
        "graph": graph_ir,
        "capture": capture_ir,
        "kernel": kernel_ir,
        "kernel_exec": kernel_exec,
        "abi": {name: spec.to_dict() for name, spec in abi_map.items()},
        "host_stub_abi": {name: spec.to_dict() for name, spec in host_stub_abi.items()},
        "asm": asm_text,
        "host_stub": host_stub_asm,
    }


def compile_target(
    source_path: Path,
    out_dir: Path,
    emit: Iterable[str],
    backend: str = "none",
    optimize: bool = True,
    target_name: str = "x86_64-sysv",
    capture_graph: bool = False,
    *,
    host_backend: str = "auto",
    cuda_arch: str = "sm_80",
    cuda_dump_kernels: bool = False,
    link_libs: Iterable[str] = (),
    link_paths: Iterable[str] = (),
    link_args: Iterable[str] = (),
    stem_override: str | None = None,
) -> dict[str, Path]:
    source = source_path.read_text(encoding="utf-8")
    artifacts = compile_pipeline_full(
        source,
        str(source_path),
        optimize=optimize,
        target_name=target_name,
        capture_graph=capture_graph,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_override or source_path.stem

    emit_set = {kind for kind in emit}
    emit_llvm_ir = "llvm_ir" in emit_set or "ll" in emit_set

    backend_payload: str | None = None
    backend_kind: str | None = None
    if backend != "none":
        normalized = normalize_backend_name(backend)
        if normalized == "auto":
            raise ManvError(diag("E7101", "compile backend cannot be 'auto'; choose an explicit backend", str(source_path), 1, 1))
        resolved = normalized
        bundle = compile_kir_backend(
            artifacts["kernel"],
            resolved,
            target=target_name,
            options={"arch": cuda_arch, "debug": cuda_dump_kernels},
        )
        artifacts["backend_bundle"] = bundle.to_dict()
        if resolved == "cuda":
            backend_kind = "ptx"
            backend_payload = bundle.binaries.get("ptx", "")
            artifacts["cuda_cpp"] = bundle.binaries.get("cuda_cpp", "")
        elif resolved == "rocm":
            backend_kind = "hip"
            backend_payload = bundle.binaries.get("hip", "")
        elif resolved == "level0":
            backend_kind = "spirv"
            backend_payload = bundle.binaries.get("spirv_text", "")
        elif resolved == "vulkan-spv":
            backend_kind = "spirv"
            backend_payload = bundle.binaries.get("spirv_text", "")
        elif resolved == "webgpu":
            backend_kind = "wgsl"
            backend_payload = bundle.binaries.get("wgsl", "")
        elif resolved == "directx":
            backend_kind = "hlsl"
            backend_payload = bundle.binaries.get("hlsl", "")
        else:
            backend_kind = "kir"
            backend_payload = bundle.binaries.get("kir_json", json.dumps(artifacts["kernel"], indent=2, sort_keys=True))

        emit_set.add("backend_bundle")
        emit_set.add(backend_kind)
        if resolved == "cuda" and cuda_dump_kernels:
            emit_set.add("cuda_cpp")

    target = get_target(target_name)
    host_selection = resolve_host_selection(
        HostSelectionRequest(
            requested_host_backend=host_backend,
            policy="compile",
        )
    )

    emit_native_obj = "native_obj" in emit_set or "obj" in emit_set
    emit_native_exe = "native_exe" in emit_set or "exe" in emit_set
    native_paths: dict[str, Path] = {}
    if host_selection.resolved_host_backend == "interp" and (emit_native_obj or emit_native_exe or emit_llvm_ir):
        raise ManvError(
            diag(
                "E5206",
                "host backend 'interp' does not emit native LLVM artifacts; use --host llvm or auto",
                str(source_path),
                1,
                1,
            )
        )

    if host_selection.resolved_host_backend == "llvm" and (emit_native_obj or emit_native_exe or emit_llvm_ir or "asm" in emit_set):
        llvm_text: str | None = None
        lowering_error: Exception | None = None
        try:
            llvm_text = emit_llvm_module(
                artifacts["_hlir_module"],
                target,
                source_name=str(source_path),
            )
        except LlvmLoweringError as err:
            lowering_error = err

        if llvm_text is None and emit_llvm_ir:
            raise ManvError(diag("E5207", f"LLVM lowering failed: {lowering_error}", str(source_path), 1, 1))

        try:
            if llvm_text is not None:
                native_paths.update(
                    build_llvm_artifacts(
                        llvm_ir=llvm_text,
                        out_dir=out_dir,
                        stem=stem,
                        target=target,
                        emit_ir=emit_llvm_ir,
                        emit_object=emit_native_obj,
                        emit_executable=emit_native_exe,
                        emit_asm="asm" in emit_set,
                        link_libs=tuple(link_libs),
                        link_paths=tuple(link_paths),
                        link_args=tuple(link_args),
                        allow_asm_fallback=True,
                        fallback_asm_text=str(artifacts["asm"]),
                    )
                )
            elif emit_native_obj or emit_native_exe:
                native = build_native_artifacts(
                    asm_text=str(artifacts["asm"]),
                    out_dir=out_dir,
                    stem=stem,
                    target=target,
                    emit_object=emit_native_obj,
                    emit_executable=emit_native_exe,
                )
                if native.object_path is not None:
                    native_paths["native_obj"] = native.object_path
                if native.executable_path is not None:
                    native_paths["native_exe"] = native.executable_path
        except ManvError as err:
            if host_backend != "auto" or err.diagnostic.code not in {"E5101", "E5102", "E5201", "E5203", "E5204", "E5205"}:
                raise

    written: dict[str, Path] = {}
    for kind in sorted(emit_set):
        if kind in {"native_obj", "native_exe", "obj", "exe", "llvm_ir", "ll"}:
            continue

        if kind in {"ptx", "hip", "spirv", "wgsl", "hlsl", "kir", "cuda_cpp"}:
            ext = {
                "ptx": "cuda.ptx",
                "hip": "rocm.hip",
                "spirv": "spirv.txt",
                "wgsl": "webgpu.wgsl",
                "hlsl": "directx.hlsl",
                "kir": "cpu.kir",
                "cuda_cpp": "cuda.cu",
            }[kind]
            path = out_dir / f"{stem}.{ext}"
            payload_text = str(artifacts.get("cuda_cpp", "")) if kind == "cuda_cpp" else backend_payload or ""
            path.write_text(payload_text, encoding="utf-8")
            written[kind] = path
            continue

        if kind == "asm":
            path = out_dir / f"{stem}.{target_name}.s"
            path.write_text(str(artifacts["asm"]), encoding="utf-8")
            written[kind] = path
            continue

        if kind == "host_stub":
            path = out_dir / f"{stem}.{target_name}.host_stub.s"
            path.write_text(str(artifacts["host_stub"]), encoding="utf-8")
            written[kind] = path
            continue

        if kind not in artifacts or artifacts[kind] is None:
            continue

        path = out_dir / f"{stem}.{kind}.json"
        path.write_text(_to_json(artifacts[kind]), encoding="utf-8")
        written[kind] = path

    written.update(native_paths)
    return written


def _to_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


