from __future__ import annotations

from typing import Any


SIDE_EFFECT_OPS = {
    "return",
    "call",
    "assign",
    "set_attr",
    "set_index",
    "raise",
    "break",
    "continue",
    "if_region",
    "while_region",
    "try_region",
    "import",
    "from_import",
}


def optimize_graph_ir(graph_ir: dict[str, Any]) -> dict[str, Any]:
    constant_folding = 0
    dead_nodes_removed = 0
    cse_total = 0
    fusion_total = 0

    for fn in graph_ir.get("functions", []):
        nodes = fn.get("nodes", [])
        fold_count = _constant_fold(nodes)
        cse_count = _common_subexpression_elimination(fn)
        fusion_count = _effect_safe_fusion(fn)
        prune_count = _dead_node_elimination(fn)
        fn["optimization"] = {
            "constant_folding": fold_count,
            "cse": cse_count,
            "fusion": fusion_count,
            "layout_normalization": 0,
            "memory_reuse": 0,
            "dead_nodes_removed": prune_count,
        }
        constant_folding += fold_count
        cse_total += cse_count
        fusion_total += fusion_count
        dead_nodes_removed += prune_count

    graph_ir["optimization"] = {
        "constant_folding": constant_folding,
        "cse": cse_total,
        "fusion": fusion_total,
        "layout_normalization": 0,
        "memory_reuse": 0,
        "dead_nodes_removed": dead_nodes_removed,
    }
    return graph_ir


def _constant_fold(nodes: list[dict[str, Any]]) -> int:
    folded = 0
    node_by_id = {node.get("id"): node for node in nodes}

    for node in nodes:
        op = str(node.get("op", ""))
        if op.startswith("binary::"):
            inputs = node.get("inputs", [])
            if len(inputs) != 2:
                continue
            left = _const_value(node_by_id.get(inputs[0]))
            right = _const_value(node_by_id.get(inputs[1]))
            if left is None or right is None:
                continue
            value = _eval_binary(op.split("::", 1)[1], left, right)
            if value is None:
                continue
            node["op"] = "const"
            node["inputs"] = []
            node["dtype"] = _dtype_of(value)
            node["attrs"] = {"value": value, "folded_from": op}
            folded += 1
            continue

        if op.startswith("unary::"):
            inputs = node.get("inputs", [])
            if len(inputs) != 1:
                continue
            value = _const_value(node_by_id.get(inputs[0]))
            if value is None:
                continue
            unary_op = op.split("::", 1)[1]
            if unary_op == "-":
                value = -value
            else:
                continue
            node["op"] = "const"
            node["inputs"] = []
            node["dtype"] = _dtype_of(value)
            node["attrs"] = {"value": value, "folded_from": op}
            folded += 1

    return folded


def _common_subexpression_elimination(function_graph: dict[str, Any]) -> int:
    nodes: list[dict[str, Any]] = function_graph.get("nodes", [])
    edges: list[dict[str, Any]] = function_graph.get("edges", [])

    seen: dict[tuple[str, tuple[str, ...], tuple[tuple[str, Any], ...]], str] = {}
    replace: dict[str, str] = {}
    removed = 0

    for node in nodes:
        op = str(node.get("op", ""))
        if op in SIDE_EFFECT_OPS or op.startswith("stub::"):
            continue
        key = (
            op,
            tuple(str(x) for x in node.get("inputs", [])),
            _cse_attrs_key(dict(node.get("attrs", {}))),
        )
        node_id = str(node.get("id"))
        if key in seen:
            replace[node_id] = seen[key]
            removed += 1
        else:
            seen[key] = node_id

    if not replace:
        return 0

    for node in nodes:
        node["inputs"] = [replace.get(str(inp), str(inp)) for inp in node.get("inputs", [])]

    function_graph["edges"] = [
        {
            "from": replace.get(str(edge.get("from")), str(edge.get("from"))),
            "to": replace.get(str(edge.get("to")), str(edge.get("to"))),
        }
        for edge in edges
        if str(edge.get("from")) not in replace
    ]

    function_graph["nodes"] = [node for node in nodes if str(node.get("id")) not in replace]
    return removed


def _effect_safe_fusion(function_graph: dict[str, Any]) -> int:
    nodes: list[dict[str, Any]] = function_graph.get("nodes", [])
    fused = 0
    for idx in range(len(nodes) - 1):
        a = nodes[idx]
        b = nodes[idx + 1]
        op_a = str(a.get("op", ""))
        op_b = str(b.get("op", ""))
        if op_a.startswith("binary::") and op_b.startswith("binary::"):
            if not a.get("effectful") and not b.get("effectful"):
                b_attrs = dict(b.get("attrs", {}))
                b_attrs["fused_with"] = str(a.get("id"))
                b["attrs"] = b_attrs
                fused += 1
    return fused


def _dead_node_elimination(function_graph: dict[str, Any]) -> int:
    nodes: list[dict[str, Any]] = function_graph.get("nodes", [])
    edges: list[dict[str, Any]] = function_graph.get("edges", [])
    node_by_id = {node.get("id"): node for node in nodes}

    roots: list[str] = []
    for node in nodes:
        op = str(node.get("op", ""))
        if op in SIDE_EFFECT_OPS or op.startswith("stub::"):
            roots.append(str(node.get("id")))

    reachable: set[str] = set()
    stack = roots[:]
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        node = node_by_id.get(current)
        if not node:
            continue
        for source in node.get("inputs", []):
            if source in node_by_id:
                stack.append(source)

    before = len(nodes)
    function_graph["nodes"] = [node for node in nodes if node.get("id") in reachable]
    keep_ids = {str(node.get("id")) for node in function_graph["nodes"]}
    function_graph["edges"] = [
        edge for edge in edges if str(edge.get("from")) in keep_ids and str(edge.get("to")) in keep_ids
    ]
    after = len(function_graph["nodes"])
    return max(0, before - after)


def _cse_attrs_key(attrs: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    # Ignore non-semantic metadata so equivalent compute ops can fold together.
    ignored = {"result", "provenance", "hlir_id", "fn", "block"}
    items: list[tuple[str, Any]] = []
    for key, value in attrs.items():
        normalized_key = str(key)
        if normalized_key in ignored:
            continue
        items.append((normalized_key, _freeze_attr(value)))
    return tuple(sorted(items))


def _freeze_attr(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_attr(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_attr(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_attr(v) for v in value))
    return value


def _const_value(node: dict[str, Any] | None) -> Any | None:
    if not node:
        return None
    if node.get("op") != "const":
        return None
    attrs = node.get("attrs", {})
    if not isinstance(attrs, dict):
        return None
    return attrs.get("value")


def _dtype_of(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "dynamic"


def _eval_binary(op: str, left: Any, right: Any) -> Any | None:
    try:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "%":
            return left % right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
    except Exception:
        return None
    return None

