from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.kernel_ir import lower_graph_to_kir_module


def _graph(nodes: list[dict[str, object]]) -> dict[str, object]:
    return {
        "version": "0.1",
        "source": "test.mv",
        "kind": "tensor_dag",
        "functions": [
            {
                "name": "main",
                "params": [],
                "nodes": nodes,
                "edges": [],
                "regions": [],
            }
        ],
        "stubs": [],
    }


def test_kernelization_allows_pure_nodes() -> None:
    payload = _graph(
        [
            {"id": "n1", "op": "const", "inputs": [], "outputs": ["v1"], "effects": ["pure"], "dtype": "i64", "attrs": {"value": 1}},
            {"id": "n2", "op": "const", "inputs": [], "outputs": ["v2"], "effects": ["pure"], "dtype": "i64", "attrs": {"value": 2}},
            {"id": "n3", "op": "binary::+", "inputs": ["v1", "v2"], "outputs": ["v3"], "effects": ["pure"], "dtype": "i64", "attrs": {}},
        ]
    )

    module = lower_graph_to_kir_module(payload).to_dict()
    meta = module["metadata"]["kernelization"]["main"]

    assert meta["kernelized_nodes"] == 3
    assert meta["fallback_required"] is False
    assert not meta["skipped"]
    assert len(module["kernels"]) == 1


def test_kernelization_blocks_dynamic_dispatch_and_may_throw() -> None:
    payload = _graph(
        [
            {"id": "n1", "op": "const", "inputs": [], "outputs": ["v1"], "effects": ["pure"], "dtype": "i64", "attrs": {"value": 1}},
            {"id": "n2", "op": "call", "inputs": ["v1"], "outputs": ["v2"], "effects": ["dynamic_dispatch", "may_throw"], "dtype": "dynamic", "attrs": {"arity": 0}},
            {"id": "n3", "op": "binary::+", "inputs": ["v1", "v1"], "outputs": ["v3"], "effects": ["pure"], "dtype": "i64", "attrs": {}},
        ]
    )

    module = lower_graph_to_kir_module(payload).to_dict()
    meta = module["metadata"]["kernelization"]["main"]

    assert meta["kernelized_nodes"] == 2
    assert meta["fallback_required"] is True
    assert meta["skipped"][0]["id"] == "n2"
    assert meta["skipped"][0]["reason"] == "effect_blocked"

    op_ids = [op["id"] for op in module["kernels"][0]["ops"]]
    assert "n2" not in op_ids


def test_kernelization_fallback_when_no_nodes_are_eligible() -> None:
    payload = _graph(
        [
            {"id": "n1", "op": "raise", "inputs": [], "outputs": [], "effects": ["may_throw"], "dtype": "void", "attrs": {}},
            {"id": "n2", "op": "try_region", "inputs": [], "outputs": [], "effects": ["writes_memory", "may_throw"], "dtype": "control", "attrs": {"non_graphable": True}},
        ]
    )

    module = lower_graph_to_kir_module(payload).to_dict()
    meta = module["metadata"]["kernelization"]["main"]

    assert meta["kernelized_nodes"] == 0
    assert meta["fallback_required"] is True
    assert len(module["kernels"]) == 0


def test_kernelization_blocks_non_kernel_safe_intrinsics() -> None:
    payload = _graph(
        [
            {
                "id": "n1",
                "op": "intrinsic_call",
                "inputs": [],
                "outputs": ["v1"],
                "effects": ["pure"],
                "dtype": "i64",
                "attrs": {"name": "core_len", "pure_for_kernel": False},
            },
            {
                "id": "n2",
                "op": "binary::+",
                "inputs": ["v1", "v1"],
                "outputs": ["v2"],
                "effects": ["pure"],
                "dtype": "i64",
                "attrs": {},
            },
        ]
    )

    module = lower_graph_to_kir_module(payload).to_dict()
    meta = module["metadata"]["kernelization"]["main"]

    assert meta["kernelized_nodes"] == 1
    assert meta["fallback_required"] is True
    assert meta["skipped"][0]["id"] == "n1"
    assert meta["skipped"][0]["reason"] == "intrinsic_not_kernel_safe"
    assert meta["skipped"][0]["intrinsic"] == "core_len"


def test_kernelization_allows_kernel_safe_intrinsics() -> None:
    payload = _graph(
        [
            {
                "id": "n1",
                "op": "intrinsic_call",
                "inputs": [],
                "outputs": ["v1"],
                "effects": ["pure"],
                "dtype": "i64",
                "attrs": {"name": "math_abs", "pure_for_kernel": True},
            },
            {
                "id": "n2",
                "op": "binary::+",
                "inputs": ["v1", "v1"],
                "outputs": ["v2"],
                "effects": ["pure"],
                "dtype": "i64",
                "attrs": {},
            },
        ]
    )

    module = lower_graph_to_kir_module(payload).to_dict()
    meta = module["metadata"]["kernelization"]["main"]

    assert meta["kernelized_nodes"] == 2
    assert meta["fallback_required"] is False
    assert not meta["skipped"]
    assert [op["id"] for op in module["kernels"][0]["ops"]] == ["n1", "n2"]
