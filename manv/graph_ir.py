"""HIR/HLIR -> Graph IR lowering.

Why this file exists:
- Converts structured HIR statements into a dependency graph suitable for
  optimization and kernelization analysis.
- Preserves effect metadata so side-effect ordering and eligibility checks
  remain explicit.
- Keeps import/EH-adjacent side effects visible to later passes.
- Provides an HLIR-authored graph path so CUDA graph-mode planning does not
  depend on the less-authoritative HIR view of the program.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hir import HIRModule
from .hlir import HFunction, HInstruction, HModule
from .intrinsics import intrinsic_effect_names, resolve_intrinsic


@dataclass
class NodeBuilder:
    counter: int = 0

    def next_id(self) -> str:
        self.counter += 1
        return f"n{self.counter}"


def lower_hlir_to_graph(hlir: HModule) -> dict[str, Any]:
    """Lower HLIR into a tensor-DAG-shaped graph with loop region summaries.

    HLIR is the semantic authority for GPU execution. The graph emitted here is
    therefore the one the compiler should prefer when it needs a static graph
    view of the program, especially for `@gpu(mode="graph")` planning.
    """

    return {
        "version": hlir.version,
        "source": hlir.source,
        "kind": "tensor_dag",
        "origin": "hlir",
        "functions": [_lower_hlir_function(fn) for fn in hlir.functions],
        "stubs": [],
    }


def extract_hlir_gpu_regions(function: HFunction) -> list[dict[str, Any]]:
    """Extract canonical counted-loop regions from HLIR.

    The summaries produced here intentionally describe the GPU-relevant
    semantics of a loop:
    - iteration extent
    - elementwise output or reduction accumulator
    - scalar and buffer dependencies
    - expression tree used per iteration

    CUDA graph-mode lowering consumes these summaries directly so it does not
    need to rediscover loop shape from raw CFG edges.
    """

    param_types = {str(param.get("name")): str(param.get("type", "dynamic")) for param in function.params}
    blocks = list(function.blocks)
    regions: list[dict[str, Any]] = []

    for index, block in enumerate(blocks):
        if not block.label.startswith("for_body"):
            continue

        body_summary = _summarize_hlir_loop_body(block.instructions, param_types)
        if body_summary is None:
            continue

        region = {
            "id": f"{function.name}:{block.label}",
            "kind": body_summary["kind"],
            "function": function.name,
            "block": block.label,
            "extent": _infer_hlir_loop_extent(blocks, index, param_types),
            "expr": body_summary["expr"],
            "buffer_inputs": sorted(body_summary["buffer_inputs"]),
            "scalar_inputs": sorted(body_summary["scalar_inputs"]),
            "provenance": body_summary["provenance"],
        }
        if body_summary["kind"] == "elementwise":
            region["output_buffer"] = body_summary["output_buffer"]
        else:
            region["accumulator"] = body_summary["accumulator"]
            region["reduction_op"] = body_summary["reduction_op"]
        regions.append(region)

    return regions


def lower_hir_to_graph(hir: HIRModule) -> dict[str, Any]:
    out_functions: list[dict[str, Any]] = []
    for fn in hir.functions:
        builder = NodeBuilder()
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        env_outputs: dict[str, str] = {}
        regions: list[dict[str, Any]] = []

        for stmt in fn.body:
            # Lower each statement as a graph node sequence while preserving
            # variable bindings and effect ordering edges.
            _lower_stmt(stmt.__dict__, builder, nodes, edges, env_outputs, regions)

        out_functions.append(
            {
                "name": fn.name,
                "params": fn.params,
                "attrs": dict(getattr(fn, "attrs", {})),
                "nodes": nodes,
                "edges": edges,
                "regions": regions,
            }
        )

    return {
        "version": hir.version,
        "source": hir.source,
        "kind": "tensor_dag",
        "origin": "hir",
        "functions": out_functions,
        "stubs": hir.stubs,
    }


def _lower_hlir_function(function: HFunction) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    effect_edges: list[dict[str, Any]] = []
    value_sources: dict[str, str] = {}
    variable_sources: dict[str, str | None] = {str(param.get("name")): f"arg::{param.get('name')}" for param in function.params}
    last_effect_node: str | None = None

    for block in function.blocks:
        for instr in block.instructions:
            last_effect_node = _lower_hlir_instruction(
                function=function,
                block_label=block.label,
                instr=instr,
                nodes=nodes,
                edges=edges,
                effect_edges=effect_edges,
                value_sources=value_sources,
                variable_sources=variable_sources,
                last_effect_node=last_effect_node,
            )

        term = block.terminator
        if term is None:
            continue
        if term.op == "ret":
            inputs = [_resolve_graph_input(token, value_sources, variable_sources) for token in term.args if token]
            node_id = term.term_id or f"{function.name}:{block.label}:ret"
            nodes.append(
                {
                    "id": node_id,
                    "op": "return",
                    "inputs": inputs,
                    "outputs": [],
                    "dtype": "void",
                    "shape": None,
                    "attrs": {
                        "fn": function.name,
                        "block": block.label,
                        "hlir_id": term.term_id,
                        "provenance": term.provenance.to_dict() if term.provenance else None,
                    },
                    "effects": ["writes_memory"],
                    "effectful": True,
                    "provenance": term.provenance.to_dict() if term.provenance else None,
                }
            )
            for source in inputs:
                if not source.startswith("arg::"):
                    edges.append({"from": source, "to": node_id})
            if last_effect_node is not None:
                effect_edges.append({"from": last_effect_node, "to": node_id})
            last_effect_node = node_id
            continue

        if term.op in {"cbr", "br", "raise", "invoke", "unreachable"}:
            node_id = term.term_id or f"{function.name}:{block.label}:{term.op}"
            nodes.append(
                {
                    "id": node_id,
                    "op": f"{term.op}_region" if term.op in {"cbr", "br", "invoke"} else term.op,
                    "inputs": [],
                    "outputs": [],
                    "dtype": "control",
                    "shape": None,
                    "attrs": {
                        "non_graphable": True,
                        "fn": function.name,
                        "block": block.label,
                        "hlir_id": term.term_id,
                        "targets": list(term.args),
                        "provenance": term.provenance.to_dict() if term.provenance else None,
                    },
                    "effects": ["writes_memory", "may_throw"],
                    "effectful": True,
                    "provenance": term.provenance.to_dict() if term.provenance else None,
                }
            )
            if last_effect_node is not None:
                effect_edges.append({"from": last_effect_node, "to": node_id})
            last_effect_node = node_id

    return {
        "name": function.name,
        "params": list(function.params),
        "attrs": dict(function.attrs),
        "nodes": nodes,
        "edges": edges,
        "effect_edges": effect_edges,
        "regions": extract_hlir_gpu_regions(function),
    }


def _lower_hlir_instruction(
    *,
    function: HFunction,
    block_label: str,
    instr: HInstruction,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    effect_edges: list[dict[str, Any]],
    value_sources: dict[str, str],
    variable_sources: dict[str, str | None],
    last_effect_node: str | None,
) -> str | None:
    op = instr.op

    # Storage plumbing is modeled symbolically so the graph sees the current
    # value of locals/temps without turning every `load_var` into a fake tensor
    # node. This keeps the graph readable while preserving data dependencies.
    if op == "declare_var":
        variable_sources[str(instr.attrs.get("name"))] = None
        return last_effect_node

    if op == "load_var":
        name = str(instr.args[0]) if instr.args else ""
        source = variable_sources.get(name)
        if instr.dest is not None and source is not None:
            value_sources[instr.dest] = source
        return last_effect_node

    if op == "store_var":
        name = str(instr.args[0]) if instr.args else ""
        source = _resolve_graph_input(instr.args[1], value_sources, variable_sources) if len(instr.args) > 1 else None
        variable_sources[name] = source
        node_id = instr.instr_id or f"{function.name}:{block_label}:{name}:store"
        nodes.append(
            {
                "id": node_id,
                "op": "assign",
                "inputs": [source] if source is not None else [],
                "outputs": [name],
                "dtype": "dynamic",
                "shape": None,
                "attrs": {
                    "fn": function.name,
                    "block": block_label,
                    "hlir_id": instr.instr_id,
                    "provenance": instr.provenance.to_dict() if instr.provenance else None,
                },
                "effects": ["writes_memory"],
                "effectful": True,
                "provenance": instr.provenance.to_dict() if instr.provenance else None,
            }
        )
        if source is not None and not source.startswith("arg::"):
            edges.append({"from": source, "to": node_id})
        if last_effect_node is not None:
            effect_edges.append({"from": last_effect_node, "to": node_id})
        return node_id

    node_id = instr.instr_id or f"{function.name}:{block_label}:{op}"
    inputs = [_resolve_graph_input(token, value_sources, variable_sources) for token in instr.args]
    inputs = [token for token in inputs if token]
    graph_op = _graph_op_name(instr)
    nodes.append(
        {
            "id": node_id,
            "op": graph_op,
            "inputs": inputs,
            "outputs": [instr.dest] if instr.dest else [],
            "dtype": instr.type_name or "dynamic",
            "shape": None,
            "attrs": {
                "fn": function.name,
                "block": block_label,
                "hlir_id": instr.instr_id,
                "provenance": instr.provenance.to_dict() if instr.provenance else None,
                **instr.attrs,
            },
            "effects": list(instr.effects) or ["pure"],
            "effectful": bool(instr.effects),
            "provenance": instr.provenance.to_dict() if instr.provenance else None,
        }
    )
    for source in inputs:
        if not source.startswith("arg::"):
            edges.append({"from": source, "to": node_id})
    if instr.dest is not None:
        value_sources[instr.dest] = node_id
    if instr.effects:
        if last_effect_node is not None:
            effect_edges.append({"from": last_effect_node, "to": node_id})
        return node_id
    return last_effect_node


def _resolve_graph_input(token: str, value_sources: dict[str, str], variable_sources: dict[str, str | None]) -> str:
    if token.startswith("%"):
        return value_sources.get(token, token)
    source = variable_sources.get(token)
    if source is not None:
        return source
    return token


def _graph_op_name(instr: HInstruction) -> str:
    if instr.op == "binop":
        return f"binary::{instr.attrs.get('op', '+')}"
    if instr.op == "unary":
        return f"unary::{instr.attrs.get('op', '-')}"
    return instr.op


def _summarize_hlir_loop_body(instructions: list[HInstruction], param_types: dict[str, str]) -> dict[str, Any] | None:
    temp_values: dict[str, Any] = {}
    output_buffer: str | None = None
    reduction_accumulator: str | None = None
    reduction_op = "+"
    reduction_expr: Any | None = None
    elementwise_expr: Any | None = None
    provenance: dict[str, Any] | None = None

    for instr in instructions:
        if instr.op == "load_var" and instr.dest is not None:
            temp_values[instr.dest] = {"kind": "var", "name": str(instr.args[0]), "dtype": param_types.get(str(instr.args[0]), "dynamic")}
            continue

        if instr.op == "const" and instr.dest is not None:
            temp_values[instr.dest] = {"kind": "const", "value": instr.attrs.get("value"), "dtype": instr.type_name or "dynamic"}
            continue

        if instr.op == "intrinsic_call" and instr.dest is not None:
            name = str(instr.attrs.get("name", ""))
            args = [temp_values.get(token, {"kind": "var", "name": token, "dtype": param_types.get(token, "dynamic")}) for token in instr.args]
            temp_values[instr.dest] = {"kind": "intrinsic", "name": name, "args": args, "dtype": instr.type_name or "dynamic"}
            continue

        if instr.op == "index" and instr.dest is not None:
            base = temp_values.get(instr.args[0], {"kind": "var", "name": instr.args[0], "dtype": param_types.get(instr.args[0], "dynamic")})
            index = temp_values.get(instr.args[1], {"kind": "var", "name": instr.args[1], "dtype": param_types.get(instr.args[1], "dynamic")})
            temp_values[instr.dest] = {"kind": "index", "base": base, "index": index, "dtype": instr.type_name or "dynamic"}
            continue

        if instr.op == "binop" and instr.dest is not None:
            lhs = temp_values.get(instr.args[0], {"kind": "var", "name": instr.args[0], "dtype": param_types.get(instr.args[0], "dynamic")})
            rhs = temp_values.get(instr.args[1], {"kind": "var", "name": instr.args[1], "dtype": param_types.get(instr.args[1], "dynamic")})
            temp_values[instr.dest] = {
                "kind": "binop",
                "op": str(instr.attrs.get("op", "+")),
                "lhs": lhs,
                "rhs": rhs,
                "dtype": instr.type_name or _infer_expr_dtype(lhs, rhs),
            }
            continue

        if instr.op == "set_index":
            target = temp_values.get(instr.args[0], {"kind": "var", "name": instr.args[0], "dtype": param_types.get(instr.args[0], "dynamic")})
            value = temp_values.get(instr.args[2], {"kind": "var", "name": instr.args[2], "dtype": param_types.get(instr.args[2], "dynamic")})
            if _expr_kind(target) == "var" and _is_array_param(str(target.get("name")), param_types):
                output_buffer = str(target.get("name"))
                elementwise_expr = value
                provenance = instr.provenance.to_dict() if instr.provenance else None
            continue

        if instr.op == "store_var":
            accumulator = str(instr.args[0]) if instr.args else ""
            value = temp_values.get(instr.args[1], {"kind": "var", "name": instr.args[1], "dtype": param_types.get(instr.args[1], "dynamic")}) if len(instr.args) > 1 else None
            if isinstance(value, dict) and value.get("kind") == "binop":
                lhs = value.get("lhs")
                rhs = value.get("rhs")
                lhs_is_acc = _expr_kind(lhs) == "var" and str(lhs.get("name")) == accumulator
                rhs_is_acc = _expr_kind(rhs) == "var" and str(rhs.get("name")) == accumulator
                if lhs_is_acc or rhs_is_acc:
                    reduction_accumulator = accumulator
                    reduction_op = str(value.get("op", "+"))
                    reduction_expr = rhs if lhs_is_acc else lhs
                    provenance = instr.provenance.to_dict() if instr.provenance else None

    if output_buffer is not None and elementwise_expr is not None:
        return {
            "kind": "elementwise",
            "output_buffer": output_buffer,
            "expr": elementwise_expr,
            "buffer_inputs": _collect_expr_inputs(elementwise_expr, param_types, want="buffer"),
            "scalar_inputs": _collect_expr_inputs(elementwise_expr, param_types, want="scalar"),
            "provenance": provenance,
        }

    if reduction_accumulator is not None and reduction_expr is not None:
        return {
            "kind": "reduction",
            "accumulator": reduction_accumulator,
            "reduction_op": reduction_op,
            "expr": reduction_expr,
            "buffer_inputs": _collect_expr_inputs(reduction_expr, param_types, want="buffer"),
            "scalar_inputs": _collect_expr_inputs(reduction_expr, param_types, want="scalar"),
            "provenance": provenance,
        }

    return None


def _infer_hlir_loop_extent(blocks: list[Any], body_index: int, param_types: dict[str, str]) -> dict[str, Any]:
    cond_block = blocks[body_index - 1] if body_index > 0 else None
    if cond_block is None:
        return {"kind": "dynamic"}

    temp_values: dict[str, Any] = {}
    for instr in cond_block.instructions:
        if instr.op == "load_var" and instr.dest is not None:
            temp_values[instr.dest] = {"kind": "var", "name": str(instr.args[0]), "dtype": param_types.get(str(instr.args[0]), "dynamic")}
            continue
        if instr.op == "const" and instr.dest is not None:
            temp_values[instr.dest] = {"kind": "const", "value": instr.attrs.get("value"), "dtype": instr.type_name or "dynamic"}
            continue
        if instr.op == "intrinsic_call" and instr.dest is not None:
            args = [temp_values.get(token, {"kind": "var", "name": token, "dtype": param_types.get(token, "dynamic")}) for token in instr.args]
            temp_values[instr.dest] = {"kind": "intrinsic", "name": str(instr.attrs.get("name", "")), "args": args, "dtype": instr.type_name or "dynamic"}

    for value in temp_values.values():
        if not isinstance(value, dict) or value.get("kind") != "intrinsic":
            continue
        if str(value.get("name")) != "core_len":
            continue
        args = list(value.get("args", []))
        if args and _expr_kind(args[0]) == "var":
            name = str(args[0].get("name"))
            if _is_array_param(name, param_types):
                return {"kind": "len", "buffer": name}

    for value in temp_values.values():
        if isinstance(value, dict) and value.get("kind") == "const":
            return {"kind": "const", "value": value.get("value")}

    return {"kind": "dynamic"}


def _collect_expr_inputs(expr: Any, param_types: dict[str, str], *, want: str) -> set[str]:
    if not isinstance(expr, dict):
        return set()

    kind = _expr_kind(expr)
    if kind == "var":
        name = str(expr.get("name"))
        if want == "buffer" and _is_array_param(name, param_types):
            return {name}
        if want == "scalar" and name in param_types and not _is_array_param(name, param_types):
            return {name}
        return set()
    if kind == "index":
        return _collect_expr_inputs(expr.get("base"), param_types, want=want) | _collect_expr_inputs(expr.get("index"), param_types, want=want)
    if kind == "binop":
        return _collect_expr_inputs(expr.get("lhs"), param_types, want=want) | _collect_expr_inputs(expr.get("rhs"), param_types, want=want)
    if kind == "intrinsic":
        found: set[str] = set()
        for arg in expr.get("args", []):
            found |= _collect_expr_inputs(arg, param_types, want=want)
        return found
    return set()


def _expr_kind(expr: Any) -> str:
    if not isinstance(expr, dict):
        return ""
    return str(expr.get("kind", ""))


def _is_array_param(name: str, param_types: dict[str, str]) -> bool:
    return str(param_types.get(name, "")).startswith("array[")


def _expr_dtype(expr: Any) -> str:
    if not isinstance(expr, dict):
        return "i32"
    dtype = str(expr.get("dtype", ""))
    if dtype in {"f32", "i32"}:
        return dtype
    if expr.get("kind") == "const":
        return "f32" if isinstance(expr.get("value"), float) else "i32"
    return "i32"


def _infer_expr_dtype(lhs: Any, rhs: Any) -> str:
    if _expr_dtype(lhs) == "f32" or _expr_dtype(rhs) == "f32":
        return "f32"
    return "i32"


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

    if kind == "import":
        node_id = builder.next_id()
        bind = str(attrs.get("alias") or str(attrs.get("module", "")).split(".")[-1])
        nodes.append(
            {
                "id": node_id,
                "op": "import",
                "inputs": [],
                "outputs": [bind],
                "dtype": "module",
                "shape": None,
                # `level` carries relative import depth metadata from parser/AST.
                "attrs": {"module": attrs.get("module"), "alias": attrs.get("alias"), "level": attrs.get("level", 0)},
                "effects": ["reads_memory", "writes_memory", "may_throw"],
            }
        )
        env_outputs[bind] = node_id
        return

    if kind == "from_import":
        node_id = builder.next_id()
        bind = str(attrs.get("alias") or attrs.get("name"))
        nodes.append(
            {
                "id": node_id,
                "op": "from_import",
                "inputs": [],
                "outputs": [bind],
                "dtype": "dynamic",
                "shape": None,
                "attrs": {
                    "module": attrs.get("module"),
                    "name": attrs.get("name"),
                    "alias": attrs.get("alias"),
                    # Preserve relative depth semantics for downstream tools.
                    "level": attrs.get("level", 0),
                },
                "effects": ["reads_memory", "writes_memory", "may_throw"],
            }
        )
        env_outputs[bind] = node_id
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

    if kind == "syscall":
        target_ref = _lower_expr(attrs.get("target"), builder, nodes, edges, env_outputs)
        args_ref = _lower_expr({"kind": "array", "elements": attrs.get("args", [])}, builder, nodes, edges, env_outputs)
        inputs = [i for i in [target_ref, args_ref] if i]
        node_id = builder.next_id()
        nodes.append(
            {
                "id": node_id,
                "op": "syscall",
                "inputs": inputs,
                "outputs": [],
                "dtype": "void",
                "shape": None,
                "attrs": {"name": "syscall_invoke"},
                "effects": ["reads_memory", "writes_memory", "dynamic_dispatch", "may_throw"],
            }
        )
        for src in inputs:
            edges.append({"from": src, "to": node_id})
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

    if kind in {"if", "while", "try", "for"}:
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

    if kind == "range":
        start = _lower_expr(expr.get("start"), builder, nodes, edges, env_outputs)
        stop = _lower_expr(expr.get("stop"), builder, nodes, edges, env_outputs)
        node_id = builder.next_id()
        inputs = [i for i in [start, stop] if i]
        nodes.append(
            {
                "id": node_id,
                "op": "range",
                "inputs": inputs,
                "outputs": [node_id],
                "dtype": "range",
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

    if kind == "intrinsic_call":
        name = str(expr.get("name", ""))
        arg_refs = [_lower_expr(a, builder, nodes, edges, env_outputs) for a in expr.get("args", [])]
        inputs = [i for i in arg_refs if i]
        spec = resolve_intrinsic(name)
        effects = ["may_throw"] if spec is None else intrinsic_effect_names(spec)
        node_id = builder.next_id()
        nodes.append(
            {
                "id": node_id,
                "op": f"intrinsic::{name}",
                "inputs": inputs,
                "outputs": [node_id],
                "dtype": "dynamic" if spec is None else str(spec.return_type),
                "shape": None,
                "attrs": {
                    "name": name,
                    "signature_id": name,
                    "pure_for_kernel": bool(spec.pure_for_kernel) if spec is not None else False,
                },
                "effects": effects,
            }
        )
        for src in inputs:
            edges.append({"from": src, "to": node_id})
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

    if kind == "syscall":
        target_ref = _lower_expr(expr.get("target"), builder, nodes, edges, env_outputs)
        arg_refs = [_lower_expr(a, builder, nodes, edges, env_outputs) for a in expr.get("args", [])]
        array_id = builder.next_id()
        arr_inputs = [i for i in arg_refs if i]
        nodes.append(
            {
                "id": array_id,
                "op": "array",
                "inputs": arr_inputs,
                "outputs": [array_id],
                "dtype": "array",
                "shape": None,
                "attrs": {"from": "syscall_args"},
                "effects": ["allocates"],
            }
        )
        for src in arr_inputs:
            edges.append({"from": src, "to": array_id})

        node_id = builder.next_id()
        inputs = [i for i in [target_ref, array_id] if i]
        nodes.append(
            {
                "id": node_id,
                "op": "intrinsic::syscall_invoke",
                "inputs": inputs,
                "outputs": [node_id],
                "dtype": "map",
                "shape": None,
                "attrs": {"name": "syscall_invoke", "signature_id": "syscall_invoke", "pure_for_kernel": False},
                "effects": ["reads_memory", "writes_memory", "dynamic_dispatch", "may_throw"],
            }
        )
        for src in inputs:
            edges.append({"from": src, "to": node_id})
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
