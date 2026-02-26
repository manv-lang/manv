from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .kernel_ir import KIRKernel, KIRModule, KIROp, parse_kir_module
from .semantics_core import eval_binary, eval_unary


@dataclass
class CPUThreadContext:
    global_id: tuple[int, int, int]
    local_id: tuple[int, int, int]
    group_id: tuple[int, int, int]
    local_size: tuple[int, int, int]
    global_size: tuple[int, int, int]


@dataclass
class CPUExecutionState:
    buffers: dict[str, list[Any]] = field(default_factory=dict)
    trace: dict[str, dict[str, Any]] = field(default_factory=dict)
    debug_records: list[dict[str, Any]] = field(default_factory=list)


class KIRRuntimeError(RuntimeError):
    pass


# Backwards-compatible entrypoint used by existing compiler/debug code.
def execute_kernel_ir(kernel_ir: dict[str, Any], *, include_trace: bool = False) -> dict[str, Any]:
    result = execute_kernel_ir_reference(kernel_ir, include_trace=include_trace)
    out: dict[str, Any] = {"kernels": result["kernels"]}
    if include_trace:
        out["trace"] = result["trace"]
    if result.get("debug_records"):
        out["debug_records"] = result["debug_records"]
    return out


def execute_kernel_ir_reference(
    kernel_ir: KIRModule | dict[str, Any],
    *,
    inputs: dict[str, list[Any]] | None = None,
    launch_override: dict[str, int] | None = None,
    include_trace: bool = False,
    capture_debug: bool = True,
    check_oob: bool = True,
) -> dict[str, Any]:
    module = kernel_ir if isinstance(kernel_ir, KIRModule) else parse_kir_module(kernel_ir)
    state = CPUExecutionState()

    if inputs:
        for name, buf in inputs.items():
            state.buffers[name] = list(buf)

    kernel_outputs: dict[str, Any] = {}

    for kernel in module.kernels:
        _ensure_buffers(state, kernel)
        launch = _resolve_launch(kernel, launch_override)
        trace_values: dict[str, Any] = {}

        for gz in range(launch["grid_z"]):
            for gy in range(launch["grid_y"]):
                for gx in range(launch["grid_x"]):
                    for bz in range(launch["block_z"]):
                        for by in range(launch["block_y"]):
                            for bx in range(launch["block_x"]):
                                ctx = CPUThreadContext(
                                    global_id=(
                                        gx * launch["block_x"] + bx,
                                        gy * launch["block_y"] + by,
                                        gz * launch["block_z"] + bz,
                                    ),
                                    local_id=(bx, by, bz),
                                    group_id=(gx, gy, gz),
                                    local_size=(launch["block_x"], launch["block_y"], launch["block_z"]),
                                    global_size=(
                                        launch["grid_x"] * launch["block_x"],
                                        launch["grid_y"] * launch["block_y"],
                                        launch["grid_z"] * launch["block_z"],
                                    ),
                                )
                                _execute_kernel_thread(
                                    kernel,
                                    ctx,
                                    state,
                                    trace_values,
                                    capture_debug=capture_debug,
                                    check_oob=check_oob,
                                )

        state.trace[kernel.name] = trace_values
        kernel_outputs[kernel.name] = _resolve_kernel_output(kernel, state)

    result: dict[str, Any] = {"kernels": kernel_outputs}
    if include_trace:
        result["trace"] = state.trace
    if capture_debug:
        result["debug_records"] = state.debug_records
    result["buffers"] = state.buffers
    return result


def _resolve_launch(kernel: KIRKernel, override: dict[str, int] | None) -> dict[str, int]:
    launch = {
        "grid_x": kernel.launch_model.grid_x,
        "grid_y": kernel.launch_model.grid_y,
        "grid_z": kernel.launch_model.grid_z,
        "block_x": kernel.launch_model.block_x,
        "block_y": kernel.launch_model.block_y,
        "block_z": kernel.launch_model.block_z,
    }
    if override:
        for key, value in override.items():
            if key in launch:
                launch[key] = int(value)
    for key, value in launch.items():
        if value <= 0:
            raise KIRRuntimeError(f"invalid launch dimension {key}={value}")
    return launch


def _ensure_buffers(state: CPUExecutionState, kernel: KIRKernel) -> None:
    for region in kernel.memory_regions:
        name = str(region.get("name", ""))
        if not name:
            continue
        if name not in state.buffers:
            state.buffers[name] = [0] * 64


def _execute_kernel_thread(
    kernel: KIRKernel,
    ctx: CPUThreadContext,
    state: CPUExecutionState,
    trace_values: dict[str, Any],
    *,
    capture_debug: bool,
    check_oob: bool,
) -> None:
    values: dict[str, Any] = {
        "%thread.x": ctx.global_id[0],
        "%thread.y": ctx.global_id[1],
        "%thread.z": ctx.global_id[2],
        "%local.x": ctx.local_id[0],
        "%local.y": ctx.local_id[1],
        "%local.z": ctx.local_id[2],
        "%group.x": ctx.group_id[0],
        "%group.y": ctx.group_id[1],
        "%group.z": ctx.group_id[2],
    }

    for block in kernel.blocks:
        for op in block.ops:
            result = _eval_op(op, values, state, ctx, capture_debug=capture_debug, check_oob=check_oob)
            values[op.id] = result
            trace_values[op.id] = result


def _eval_op(
    op: KIROp,
    values: dict[str, Any],
    state: CPUExecutionState,
    ctx: CPUThreadContext,
    *,
    capture_debug: bool,
    check_oob: bool,
) -> Any:
    name = op.opcode

    if name in {"thread_id_x", "builtin.thread.x"}:
        return ctx.global_id[0]
    if name in {"thread_id_y", "builtin.thread.y"}:
        return ctx.global_id[1]
    if name in {"thread_id_z", "builtin.thread.z"}:
        return ctx.global_id[2]

    if name == "const":
        return _resolve_default_value(op.attrs)

    if name in {"binop", "binary"}:
        a = _resolve_arg(op.inputs, 0, values)
        b = _resolve_arg(op.inputs, 1, values)
        return _safe_eval_binary(str(op.attrs.get("op", "+")), a, b)

    if isinstance(name, str) and name.startswith("binary::"):
        a = _resolve_arg(op.inputs, 0, values)
        b = _resolve_arg(op.inputs, 1, values)
        return _safe_eval_binary(name.split("::", 1)[1], a, b)

    if name in {"unary"}:
        a = _resolve_arg(op.inputs, 0, values)
        return _safe_eval_unary(str(op.attrs.get("op", "-")), a)

    if isinstance(name, str) and name.startswith("unary::"):
        a = _resolve_arg(op.inputs, 0, values)
        return _safe_eval_unary(name.split("::", 1)[1], a)

    if name == "buffer_load":
        buffer_name = str(op.attrs.get("buffer"))
        index = int(_resolve_arg(op.inputs, 0, values))
        return _load_buffer(state, buffer_name, index, check_oob)

    if name == "buffer_store":
        buffer_name = str(op.attrs.get("buffer"))
        index = int(_resolve_arg(op.inputs, 0, values))
        value = _resolve_arg(op.inputs, 1, values)
        _store_buffer(state, buffer_name, index, value, check_oob)
        return value

    if name in {"barrier", "sync"}:
        return None

    if name in {"assert", "kernel_assert"}:
        cond = bool(_resolve_arg(op.inputs, 0, values)) if op.inputs else bool(op.attrs.get("condition", False))
        if not cond and capture_debug:
            state.debug_records.append(
                {
                    "kind": "assert",
                    "kernel": op.provenance.graph_node_id if op.provenance else None,
                    "op_id": op.id,
                    "thread": {"x": ctx.global_id[0], "y": ctx.global_id[1], "z": ctx.global_id[2]},
                    "message": str(op.attrs.get("message", "kernel assertion failed")),
                    "provenance": op.provenance.to_dict() if op.provenance else None,
                }
            )
        return cond

    if name == "return":
        return _resolve_arg(op.inputs, 0, values) if op.inputs else None

    if name in {
        "bind",
        "assign",
        "call",
        "index",
        "array",
        "map",
        "attr",
        "set_index",
        "store_var",
        "load_var",
        "load_arg",
    }:
        return _resolve_arg(op.inputs, 0, values) if op.inputs else _resolve_default_value(op.attrs)

    return _resolve_default_value(op.attrs)


def _safe_eval_binary(op: str, left: Any, right: Any) -> Any:
    try:
        return eval_binary(op, left, right)
    except Exception:
        return None


def _safe_eval_unary(op: str, value: Any) -> Any:
    try:
        return eval_unary(op, value)
    except Exception:
        return None


def _resolve_default_value(attrs: dict[str, Any]) -> Any:
    if "value" in attrs:
        return attrs.get("value")
    return attrs.get("result")


def _resolve_arg(args: list[str], index: int, values: dict[str, Any]) -> Any:
    if index >= len(args):
        return None
    token = args[index]
    if token in values:
        return values[token]
    if token.startswith("%") and token not in values:
        return None
    # literals in legacy ops flow through as strings; parse int when possible.
    try:
        return int(token)
    except Exception:
        return token


def _load_buffer(state: CPUExecutionState, name: str, index: int, check_oob: bool) -> Any:
    if name not in state.buffers:
        state.buffers[name] = [0] * 64
    buf = state.buffers[name]
    if check_oob and (index < 0 or index >= len(buf)):
        raise KIRRuntimeError(f"buffer_load out-of-bounds: {name}[{index}] len={len(buf)}")
    if index < 0 or index >= len(buf):
        return 0
    return buf[index]


def _store_buffer(state: CPUExecutionState, name: str, index: int, value: Any, check_oob: bool) -> None:
    if name not in state.buffers:
        state.buffers[name] = [0] * 64
    buf = state.buffers[name]
    if check_oob and (index < 0 or index >= len(buf)):
        raise KIRRuntimeError(f"buffer_store out-of-bounds: {name}[{index}] len={len(buf)}")
    if 0 <= index < len(buf):
        buf[index] = value


def _resolve_kernel_output(kernel: KIRKernel, state: CPUExecutionState) -> Any:
    if kernel.memory_regions:
        first_name = str(kernel.memory_regions[0].get("name", ""))
        if first_name and first_name in state.buffers:
            return list(state.buffers[first_name])
    # Fallback for expression-like kernels that only compute a scalar return op.
    if kernel.blocks and kernel.blocks[0].ops:
        last = kernel.blocks[0].ops[-1]
        if last.opcode == "return" and last.inputs:
            return None
    return None
