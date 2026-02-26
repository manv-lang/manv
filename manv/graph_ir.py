from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hir import HIRModule


@dataclass
class NodeBuilder:
    counter: int = 0

    def next_id(self) -> str:
        self.counter += 1
        return f"n{self.counter}"


def lower_hir_to_graph(hir: HIRModule) -> dict[str, Any]:
    out_functions: list[dict[str, Any]] = []
    for fn in hir.functions:
        builder = NodeBuilder()
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        env_outputs: dict[str, str] = {}
        regions: list[dict[str, Any]] = []

        for stmt in fn.body:
            _lower_stmt(stmt.__dict__, builder, nodes, edges, env_outputs, regions)

        out_functions.append(
            {
                "name": fn.name,
                "params": fn.params,
                "nodes": nodes,
                "edges": edges,
                "regions": regions,
            }
        )

    return {
        "version": hir.version,
        "source": hir.source,
        "kind": "tensor_dag",
        "functions": out_functions,
        "stubs": hir.stubs,
    }


def _lower_stmt(
    stmt: dict[str, Any],
    builder: NodeBuilder,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    env_outputs: dict[str, str],
    regions: list[dict[str, Any]],
) -> None:
    kind = stmt["kind"]
    attrs = stmt["attrs"]
    if kind == "let":
        value_ref = _lower_expr(attrs.get("value"), builder, nodes, edges, env_outputs)
        node_id = builder.next_id()
        nodes.append(
            {
                "id": node_id,
                "op": "bind",
                "inputs": [value_ref] if value_ref else [],
                "outputs": [attrs["name"]],
                "dtype": attrs.get("type") or "dynamic",
                "shape": None,
                "attrs": {"array_size": attrs.get("array_size")},
                "effects": ["writes_memory"],
            }
        )
        if value_ref:
            edges.append({"from": value_ref, "to": node_id})
        env_outputs[attrs["name"]] = node_id
        return

    if kind == "assign":
        value_ref = _lower_expr(attrs.get("value"), builder, nodes, edges, env_outputs)
        node_id = builder.next_id()
        nodes.append(
            {
                "id": node_id,
                "op": "assign",
                "inputs": [value_ref] if value_ref else [],
                "outputs": [attrs["name"]],
                "dtype": "dynamic",
                "shape": None,
                "attrs": {},
                "effects": ["writes_memory"],
            }
        )
        if value_ref:
            edges.append({"from": value_ref, "to": node_id})
        env_outputs[attrs["name"]] = node_id
        return

    if kind == "set_attr":
        target_ref = _lower_expr(attrs.get("target"), builder, nodes, edges, env_outputs)
        value_ref = _lower_expr(attrs.get("value"), builder, nodes, edges, env_outputs)
        inputs = [i for i in [target_ref, value_ref] if i]
        node_id = builder.next_id()
        nodes.append(
            {
                "id": node_id,
                "op": "set_attr",
                "inputs": inputs,
                "outputs": [],
                "dtype": "void",
                "shape": None,
                "attrs": {"attr": attrs.get("attr")},
                "effects": ["writes_memory", "dynamic_dispatch", "may_throw"],
            }
        )
        for src in inputs:
            edges.append({"from": src, "to": node_id})
        return

    if kind == "set_index":
        target_ref = _lower_expr(attrs.get("target"), builder, nodes, edges, env_outputs)
        index_ref = _lower_expr(attrs.get("index"), builder, nodes, edges, env_outputs)
        value_ref = _lower_expr(attrs.get("value"), builder, nodes, edges, env_outputs)
        inputs = [i for i in [target_ref, index_ref, value_ref] if i]
        node_id = builder.next_id()
        nodes.append(
            {
                "id": node_id,
                "op": "set_index",
                "inputs": inputs,
                "outputs": [],
                "dtype": "void",
                "shape": None,
                "attrs": {},
                "effects": ["writes_memory", "may_throw"],
            }
        )
        for src in inputs:
            edges.append({"from": src, "to": node_id})
        return

    if kind == "expr":
        _lower_expr(attrs.get("expr"), builder, nodes, edges, env_outputs)
        return

    if kind == "return":
        value_ref = _lower_expr(attrs.get("value"), builder, nodes, edges, env_outputs)
        node_id = builder.next_id()
        nodes.append(
            {
                "id": node_id,
                "op": "return",
                "inputs": [value_ref] if value_ref else [],
                "outputs": [],
                "dtype": "void",
                "shape": None,
                "attrs": {},
                "effects": ["writes_memory"],
            }
        )
        if value_ref:
            edges.append({"from": value_ref, "to": node_id})
        return

    if kind == "raise":
        value_ref = _lower_expr(attrs.get("value"), builder, nodes, edges, env_outputs)
        node_id = builder.next_id()
        nodes.append(
            {
                "id": node_id,
                "op": "raise",
                "inputs": [value_ref] if value_ref else [],
                "outputs": [],
                "dtype": "void",
                "shape": None,
                "attrs": {},
                "effects": ["may_throw"],
            }
        )
        if value_ref:
            edges.append({"from": value_ref, "to": node_id})
        return

    if kind in {"break", "continue"}:
        node_id = builder.next_id()
        nodes.append(
            {
                "id": node_id,
                "op": kind,
                "inputs": [],
                "outputs": [],
                "dtype": "control",
                "shape": None,
                "attrs": {},
                "effects": ["writes_memory"],
            }
        )
        return

    if kind in {"if", "while", "try"}:
        region_id = builder.next_id()
        regions.append({"id": region_id, "kind": kind, "attrs": attrs})
        nodes.append(
            {
                "id": region_id,
                "op": f"{kind}_region",
                "inputs": [],
                "outputs": [],
                "dtype": "control",
                "shape": None,
                "attrs": {"non_graphable": True},
                "effects": ["writes_memory", "may_throw"],
            }
        )
        return

    node_id = builder.next_id()
    nodes.append(
        {
            "id": node_id,
            "op": f"stub::{kind}",
            "inputs": [],
            "outputs": [],
            "dtype": "dynamic",
            "shape": None,
            "attrs": attrs,
            "effects": ["may_throw"],
        }
    )


def _lower_expr(
    expr: dict[str, Any] | None,
    builder: NodeBuilder,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    env_outputs: dict[str, str],
) -> str | None:
    if expr is None:
        return None

    kind = expr.get("kind")

    if kind == "literal":
        node_id = builder.next_id()
        nodes.append(
            {
                "id": node_id,
                "op": "const",
                "inputs": [],
                "outputs": [node_id],
                "dtype": expr.get("type", "dynamic"),
                "shape": None,
                "attrs": {"value": expr.get("value")},
                "effects": ["pure"],
            }
        )
        return node_id

    if kind == "ident":
        name = expr.get("name", "")
        return env_outputs.get(name, f"arg::{name}")

    if kind == "binary":
        left = _lower_expr(expr.get("left"), builder, nodes, edges, env_outputs)
        right = _lower_expr(expr.get("right"), builder, nodes, edges, env_outputs)
        node_id = builder.next_id()
        inputs = [i for i in [left, right] if i]
        nodes.append(
            {
                "id": node_id,
                "op": f"binary::{expr.get('op')}",
                "inputs": inputs,
                "outputs": [node_id],
                "dtype": "dynamic",
                "shape": None,
                "attrs": {},
                "effects": ["pure"],
            }
        )
        for src in inputs:
            edges.append({"from": src, "to": node_id})
        return node_id

    if kind == "unary":
        inner = _lower_expr(expr.get("expr"), builder, nodes, edges, env_outputs)
        node_id = builder.next_id()
        inputs = [inner] if inner else []
        nodes.append(
            {
                "id": node_id,
                "op": f"unary::{expr.get('op')}",
                "inputs": inputs,
                "outputs": [node_id],
                "dtype": "dynamic",
                "shape": None,
                "attrs": {},
                "effects": ["pure"],
            }
        )
        if inner:
            edges.append({"from": inner, "to": node_id})
        return node_id

    if kind == "call":
        callee = _lower_expr(expr.get("callee"), builder, nodes, edges, env_outputs)
        arg_refs = [_lower_expr(a, builder, nodes, edges, env_outputs) for a in expr.get("args", [])]
        inputs = [i for i in [callee, *arg_refs] if i]
        node_id = builder.next_id()
        nodes.append(
            {
                "id": node_id,
                "op": "call",
                "inputs": inputs,
                "outputs": [node_id],
                "dtype": "dynamic",
                "shape": None,
                "attrs": {"arity": len(expr.get("args", []))},
                "effects": ["dynamic_dispatch", "may_throw"],
            }
        )
        for src in inputs:
            edges.append({"from": src, "to": node_id})
        return node_id

    if kind in {"array", "map", "index", "attr"}:
        node_id = builder.next_id()
        effect = ["pure"]
        if kind in {"index", "attr"}:
            effect = ["may_throw", "dynamic_dispatch"]
        nodes.append(
            {
                "id": node_id,
                "op": kind,
                "inputs": [],
                "outputs": [node_id],
                "dtype": "dynamic",
                "shape": None,
                "attrs": expr,
                "effects": effect,
            }
        )
        return node_id

    node_id = builder.next_id()
    nodes.append(
        {
            "id": node_id,
            "op": "unknown_expr",
            "inputs": [],
            "outputs": [node_id],
            "dtype": "dynamic",
            "shape": None,
            "attrs": expr,
            "effects": ["may_throw"],
        }
    )
    return node_id
