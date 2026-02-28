from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.compiler import analyze_program, parse_program
from manv.gpu_execution import lower_hlir_function_to_backend_ir
from manv.graph_ir import extract_hlir_gpu_regions, lower_hlir_to_graph
from manv.hlir_interpreter import HLIRInterpreter
from manv.hlir_lowering import lower_ast_to_hlir
from manv.backends.cuda import analyze_hlir_gpu_function


def _source(header: str = "@gpu") -> str:
    return (
        f"{header}\n"
        "fn add(a: f32[], b: f32[], out: f32[]) -> void:\n"
        "    for i in 0..len(out):\n"
        "        out[i] = a[i] + b[i]\n"
        "\n"
        "fn main() -> int:\n"
        "    let a: f32[] = [1.0, 2.0]\n"
        "    let b: f32[] = [3.0, 4.0]\n"
        "    let out: f32[] = [0.0, 0.0]\n"
        "    add(a, b, out)\n"
        "    print(out[0])\n"
        "    return 0\n"
    )


def test_gpu_decorator_lowers_to_structured_hlir_gpu_call() -> None:
    program = parse_program(_source(), "gpu_test.mv")
    analyze_program(program, "gpu_test.mv")
    hlir = lower_ast_to_hlir(program, "gpu_test.mv")

    add_fn = next(fn for fn in hlir.functions if fn.name == "add")
    main_fn = next(fn for fn in hlir.functions if fn.name == "main")

    # The function itself carries normalized GPU policy metadata.
    assert add_fn.attrs["gpu"] == {"required": False, "mode": "kernel"}

    # The callsite in `main` is an explicit `gpu_call`, not a normal `call`.
    ops = [instr.op for block in main_fn.blocks for instr in block.instructions]
    assert "gpu_call" in ops
    assert "call" not in ops


def test_gpu_eligibility_accepts_simple_elementwise_loop() -> None:
    program = parse_program(_source(), "gpu_test.mv")
    analyze_program(program, "gpu_test.mv")
    hlir = lower_ast_to_hlir(program, "gpu_test.mv")

    add_fn = next(fn for fn in hlir.functions if fn.name == "add")
    report = analyze_hlir_gpu_function(add_fn)

    assert report.eligible is True
    assert report.issues == []


def test_gpu_best_effort_falls_back_to_cpu_hlir(monkeypatch: pytest.MonkeyPatch) -> None:
    # The runtime path must preserve semantics even when CUDA is unavailable.
    monkeypatch.setattr(
        "manv.gpu_execution.backend_selection_report",
        lambda *args, **kwargs: type(
            "Selection",
            (),
            {"selected_backend": "cpu", "selected_device": None, "to_dict": lambda self: {"selected_backend": "cpu"}},
        )(),
    )

    program = parse_program(_source(), "gpu_test.mv")
    analyze_program(program, "gpu_test.mv")
    hlir = lower_ast_to_hlir(program, "gpu_test.mv")

    out = StringIO()
    result = HLIRInterpreter(stdout=out).run_module(hlir, entry="main")

    assert result.value == 0
    assert out.getvalue().strip() in {"4.0", "4"}


def test_gpu_required_errors_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "manv.gpu_execution.backend_selection_report",
        lambda *args, **kwargs: type(
            "Selection",
            (),
            {"selected_backend": "cpu", "selected_device": None, "to_dict": lambda self: {"selected_backend": "cpu"}},
        )(),
    )

    program = parse_program(_source(header='@gpu(required=True, mode="kernel")'), "gpu_required.mv")
    analyze_program(program, "gpu_required.mv")
    hlir = lower_ast_to_hlir(program, "gpu_required.mv")

    with pytest.raises(RuntimeError) as err:
        HLIRInterpreter(stdout=StringIO()).run_module(hlir, entry="main")

    assert "GPU backend unavailable for required @gpu call" in str(err.value)


def test_nested_cuda_intrinsic_is_lowered_as_intrinsic_call() -> None:
    source = (
        "fn main() -> int:\n"
        "    __intrin.cuda.is_available()\n"
        "    return 0\n"
    )
    program = parse_program(source, "intrin_cuda.mv")
    analyze_program(program, "intrin_cuda.mv")
    hlir = lower_ast_to_hlir(program, "intrin_cuda.mv")

    main_fn = next(fn for fn in hlir.functions if fn.name == "main")
    intrinsic_instrs = [instr for block in main_fn.blocks for instr in block.instructions if instr.op == "intrinsic_call"]
    assert intrinsic_instrs
    assert intrinsic_instrs[0].attrs["name"] == "cuda_is_available"


def test_hlir_graph_lowering_is_authoritative_for_default_pipeline_shapes() -> None:
    program = parse_program(_source(), "gpu_graph.mv")
    analyze_program(program, "gpu_graph.mv")
    hlir = lower_ast_to_hlir(program, "gpu_graph.mv")

    graph = lower_hlir_to_graph(hlir)
    add_fn = next(fn for fn in graph["functions"] if fn["name"] == "add")

    assert graph["origin"] == "hlir"
    assert graph["kind"] == "tensor_dag"
    assert add_fn["regions"]
    assert add_fn["regions"][0]["kind"] == "elementwise"
    assert add_fn["regions"][0]["output_buffer"] == "out"


def test_kernel_mode_lowering_keeps_scalar_params_in_signature() -> None:
    source = (
        "@gpu\n"
        "fn saxpy(alpha: f32, a: f32[], out: f32[]) -> void:\n"
        "    for i in 0..len(out):\n"
        "        out[i] = alpha * a[i]\n"
    )
    program = parse_program(source, "saxpy.mv")
    analyze_program(program, "saxpy.mv")
    hlir = lower_ast_to_hlir(program, "saxpy.mv")

    fn = next(fn for fn in hlir.functions if fn.name == "saxpy")
    kernel_ir, _launch, dispatch_inputs, output_plan = lower_hlir_function_to_backend_ir(
        fn,
        args=[2.0, [1.0, 2.0], [0.0, 0.0]],
        mode="kernel",
    )

    params = kernel_ir["kernels"][0]["signature"]["params"]
    assert any(param["name"] == "alpha" and param["kind"] == "scalar" and param["by_ref"] is False for param in params)
    assert dispatch_inputs["alpha"] == 2.0
    assert output_plan == {"kind": "buffer", "name": "out"}


def test_graph_mode_lowering_builds_multi_kernel_reduction_module() -> None:
    source = (
        '@gpu(mode="graph")\n'
        "fn reduce(a: i32[]) -> i32:\n"
        "    let acc: i32 = 0\n"
        "    for i in 0..len(a):\n"
        "        acc = acc + a[i]\n"
        "    return acc\n"
    )
    program = parse_program(source, "reduce.mv")
    analyze_program(program, "reduce.mv")
    hlir = lower_ast_to_hlir(program, "reduce.mv")

    fn = next(fn for fn in hlir.functions if fn.name == "reduce")
    regions = extract_hlir_gpu_regions(fn)
    kernel_ir, launch_override, dispatch_inputs, output_plan = lower_hlir_function_to_backend_ir(
        fn,
        args=[[1, 2, 3, 4]],
        mode="graph",
    )

    assert regions and regions[0]["kind"] == "reduction"
    assert launch_override is None
    assert len(kernel_ir["kernels"]) == 2
    assert kernel_ir["kernels"][0]["debug_meta"]["kernel_kind"] == "reduction_partial"
    assert kernel_ir["kernels"][1]["debug_meta"]["kernel_kind"] == "reduction_finalize"
    assert dispatch_inputs["__manv_partial_0"] == [0]
    assert dispatch_inputs["__manv_return_0"] == [0]
    assert output_plan == {"kind": "return_scalar", "name": "__manv_return_0"}
