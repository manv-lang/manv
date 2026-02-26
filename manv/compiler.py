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
from .gpu_backends import compile_kir_backend
from .gpu_dispatch import select_backend
from .graph_capture import GraphCaptureTracer
from .graph_ir import lower_hir_to_graph
from .graph_opt import optimize_graph_ir
from .hlir_lowering import lower_ast_to_hlir
from .hlir_interpreter import HLIRInterpreter
from .host_stub import build_host_stubs
from .kernel_ir import lower_graph_to_kernel
from .kernel_mock import execute_kernel_ir
from .lexer import Lexer
from .lowering import lower_ast_to_hir
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
    source_map = build_source_map_from_hlir(hlir_module)

    capture_ir: dict[str, Any] | None = None
    if capture_graph:
        tracer = GraphCaptureTracer()
        sink = io.StringIO()
        HLIRInterpreter(stdout=sink, tracer=tracer).run_module(hlir_module, entry="main")
        graph_ir = tracer.to_graph_ir()
        capture_ir = graph_ir
    else:
        graph_ir = lower_hir_to_graph(hir_module)

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
    stem = source_path.stem

    emit_set = {kind for kind in emit}

    backend_payload: str | None = None
    backend_kind: str | None = None
    if backend != "none":
        resolved = select_backend(backend)
        bundle = compile_kir_backend(artifacts["kernel"], resolved, target=target_name)
        artifacts["backend_bundle"] = bundle.to_dict()
        if resolved == "cuda":
            backend_kind = "ptx"
            backend_payload = bundle.binaries.get("ptx", "")
        elif resolved == "rocm":
            backend_kind = "hip"
            backend_payload = bundle.binaries.get("hip", "")
        elif resolved == "metal":
            backend_kind = "msl"
            backend_payload = bundle.binaries.get("msl", "")
        elif resolved == "vulkan_spirv":
            backend_kind = "spirv"
            backend_payload = bundle.binaries.get("spirv_text", "")
        elif resolved == "webgpu":
            backend_kind = "wgsl"
            backend_payload = bundle.binaries.get("wgsl", "")
        elif resolved == "opencl":
            backend_kind = "opencl"
            backend_payload = bundle.binaries.get("opencl_c", "")
        elif resolved == "directx":
            backend_kind = "hlsl"
            backend_payload = bundle.binaries.get("hlsl", "")
        else:
            backend_kind = "kir"
            backend_payload = bundle.binaries.get("kir_json", json.dumps(artifacts["kernel"], indent=2, sort_keys=True))

        emit_set.add("backend_bundle")
        emit_set.add(backend_kind)

    target = get_target(target_name)

    emit_native_obj = "native_obj" in emit_set or "obj" in emit_set
    emit_native_exe = "native_exe" in emit_set or "exe" in emit_set
    native_paths: dict[str, Path] = {}
    if emit_native_obj or emit_native_exe:
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

    written: dict[str, Path] = {}
    for kind in sorted(emit_set):
        if kind in {"native_obj", "native_exe", "obj", "exe"}:
            continue

        if kind in {"ptx", "hip", "msl", "spirv", "wgsl", "opencl", "hlsl", "kir"}:
            ext = {
                "ptx": "cuda.ptx",
                "hip": "rocm.hip",
                "msl": "metal.msl",
                "spirv": "vulkan.spv.txt",
                "wgsl": "webgpu.wgsl",
                "opencl": "opencl.cl",
                "hlsl": "directx.hlsl",
                "kir": "cpu_ref.kir",
            }[kind]
            path = out_dir / f"{stem}.{ext}"
            path.write_text(backend_payload or "", encoding="utf-8")
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


