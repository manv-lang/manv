from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.gpu_backends import compile_kir_backend, list_backends
from manv.gpu_dispatch import dispatch_kernel_ir, backend_capability_table
from manv.kernel_mock import KIRRuntimeError, execute_kernel_ir_reference
from manv.kir_verify import verify_kir_module


def _prov() -> dict[str, object]:
    return {
        "graph_node_id": "g1",
        "hlir_op_id": "main.i1",
        "source_span": {
            "uri": "test.mv",
            "start_line": 1,
            "start_col": 1,
            "end_line": 1,
            "end_col": 10,
        },
        "inline_chain": [],
    }


def _vector_add_kernel_ir() -> dict[str, object]:
    return {
        "version": "0.1",
        "source": "test.mv",
        "kernels": [
            {
                "kernel_name": "vec_add",
                "signature": {
                    "params": [
                        {"index": 0, "name": "out", "kind": "buffer", "dtype": "i64", "by_ref": True, "alignment": 8, "address_space": "global"},
                        {"index": 1, "name": "a", "kind": "buffer", "dtype": "i64", "by_ref": True, "alignment": 8, "address_space": "global"},
                        {"index": 2, "name": "b", "kind": "buffer", "dtype": "i64", "by_ref": True, "alignment": 8, "address_space": "global"},
                    ],
                    "return_policy": "void",
                },
                "launch_model": {"grid_x": 1, "grid_y": 1, "grid_z": 1, "block_x": 4, "block_y": 1, "block_z": 1, "shared_bytes": 0},
                "memory_regions": [
                    {"name": "out", "dtype": "i64", "space": "global"},
                    {"name": "a", "dtype": "i64", "space": "global"},
                    {"name": "b", "dtype": "i64", "space": "global"},
                ],
                "blocks": [
                    {
                        "id": "entry",
                        "ops": [
                            {"id": "t0", "opcode": "thread_id_x", "inputs": [], "outputs": ["t0"], "attrs": {}, "dtype": "i64", "memory_space": "private", "provenance": _prov()},
                            {"id": "a0", "opcode": "buffer_load", "inputs": ["t0"], "outputs": ["a0"], "attrs": {"buffer": "a"}, "dtype": "i64", "memory_space": "global", "provenance": _prov()},
                            {"id": "b0", "opcode": "buffer_load", "inputs": ["t0"], "outputs": ["b0"], "attrs": {"buffer": "b"}, "dtype": "i64", "memory_space": "global", "provenance": _prov()},
                            {"id": "s0", "opcode": "binary::+", "inputs": ["a0", "b0"], "outputs": ["s0"], "attrs": {}, "dtype": "i64", "memory_space": "private", "provenance": _prov()},
                            {"id": "w0", "opcode": "buffer_store", "inputs": ["t0", "s0"], "outputs": [], "attrs": {"buffer": "out"}, "dtype": "void", "memory_space": "global", "provenance": _prov()},
                        ],
                        "terminator": "ret",
                    }
                ],
            }
        ],
    }


def test_kir_verifier_rejects_missing_provenance() -> None:
    payload = _vector_add_kernel_ir()
    payload["kernels"][0]["blocks"][0]["ops"][0]["provenance"] = None
    issues = verify_kir_module(payload, strict=True)
    assert any(i.code == "KIR009" for i in issues)


def test_kir_verifier_rejects_invalid_memory_space() -> None:
    payload = _vector_add_kernel_ir()
    payload["kernels"][0]["blocks"][0]["ops"][0]["memory_space"] = "warp"
    issues = verify_kir_module(payload, strict=True)
    assert any(i.code == "KIR007" for i in issues)


def test_backend_compile_smoke_all_backends() -> None:
    payload = _vector_add_kernel_ir()
    for backend in [b for b in list_backends() if b != "cpu_ref"]:
        bundle = compile_kir_backend(payload, backend, target="generic")
        assert bundle.backend == backend
        assert bundle.entrypoints
        assert bundle.binaries


def test_cross_backend_equivalence_against_cpu_reference() -> None:
    payload = _vector_add_kernel_ir()
    inputs = {
        "out": [0, 0, 0, 0, 0, 0, 0, 0],
        "a": [1, 2, 3, 4, 0, 0, 0, 0],
        "b": [5, 6, 7, 8, 0, 0, 0, 0],
    }

    cpu = execute_kernel_ir_reference(payload, inputs=inputs, include_trace=True)
    expected = cpu["buffers"]["out"][:4]

    for backend in ["cuda", "rocm", "metal", "vulkan-spv", "webgpu", "opencl", "directx"]:
        result = dispatch_kernel_ir(payload, backend=backend, inputs=inputs)
        got = result.outputs["buffers"]["out"][:4]
        assert got == expected


def test_cpu_reference_oob_detection() -> None:
    payload = _vector_add_kernel_ir()
    payload["kernels"][0]["blocks"][0]["ops"] = [
        {"id": "k0", "opcode": "const", "inputs": [], "outputs": ["k0"], "attrs": {"value": 10}, "dtype": "i64", "memory_space": "private", "provenance": _prov()},
        {"id": "v0", "opcode": "const", "inputs": [], "outputs": ["v0"], "attrs": {"value": 7}, "dtype": "i64", "memory_space": "private", "provenance": _prov()},
        {"id": "w0", "opcode": "buffer_store", "inputs": ["k0", "v0"], "outputs": [], "attrs": {"buffer": "out"}, "dtype": "void", "memory_space": "global", "provenance": _prov()},
    ]

    with pytest.raises(KIRRuntimeError):
        execute_kernel_ir_reference(payload, inputs={"out": [0, 0]})


def test_debug_assert_propagation() -> None:
    payload = _vector_add_kernel_ir()
    payload["kernels"][0]["blocks"][0]["ops"] = [
        {"id": "c0", "opcode": "const", "inputs": [], "outputs": ["c0"], "attrs": {"value": 0}, "dtype": "i64", "memory_space": "private", "provenance": _prov()},
        {"id": "a0", "opcode": "kernel_assert", "inputs": ["c0"], "outputs": [], "attrs": {"message": "assert failed"}, "dtype": "void", "memory_space": "private", "provenance": _prov()},
    ]

    result = execute_kernel_ir_reference(payload, inputs={"out": [0]})
    assert result["debug_records"]
    assert result["debug_records"][0]["kind"] == "assert"


def test_dispatch_trace_export_and_capability_table() -> None:
    payload = _vector_add_kernel_ir()
    result = dispatch_kernel_ir(payload, backend="cuda", inputs={"out": [0] * 8, "a": [1] * 8, "b": [2] * 8})
    assert "traceEvents" in result.trace
    assert len(result.trace["traceEvents"]) >= 2

    table = backend_capability_table()
    assert "cuda" in table
    assert "webgpu" in table
    assert table["cuda"]["barriers"] is True
