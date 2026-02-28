"""HLIR-authored GPU execution boundary for `@gpu` functions.

Why this module exists:
- `@gpu` changes call dispatch policy, not source-language meaning.
- The interpreter/runtime needs one place that decides whether a decorated call
  should execute via CUDA or fall back to CPU execution of the same HLIR body.

Important invariants:
- CPU fallback always reuses the same HLIR function body. The fallback is not a
  separate source-language implementation path, which keeps HLIR authoritative.
- Graph mode and kernel mode both start from HLIR facts. Neither path is
  allowed to infer new language semantics from backend-specific heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Callable

from .backends.cuda import analyze_hlir_gpu_function
from .gpu_dispatch import backend_selection_report, dispatch_kernel_ir
from .graph_ir import extract_hlir_gpu_regions
from .hlir import HFunction, HModule


@dataclass
class GpuExecutionDecision:
    executed_on_gpu: bool
    fallback_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class GpuExecutionEngine:
    def __init__(
        self,
        module: HModule,
        cpu_fallback: Callable[[str, list[Any]], Any],
        on_gpu_execution: Callable[[dict[str, Any]], None] | None = None,
        *,
        preferred_backend: str = "auto",
        preferred_device: str | None = None,
    ):
        self.module = module
        self.cpu_fallback = cpu_fallback
        self.on_gpu_execution = on_gpu_execution
        self.preferred_backend = preferred_backend
        self.preferred_device = preferred_device
        self._functions = {fn.name: fn for fn in module.functions}

    def execute(self, *, callee: str, args: list[Any], policy: str, mode: str) -> tuple[Any, GpuExecutionDecision]:
        function = self._functions.get(callee)
        if function is None:
            raise RuntimeError(f"undefined function: {callee}")

        report = analyze_hlir_gpu_function(function)
        if not report.eligible:
            if policy == "required":
                reasons = ", ".join(issue.code for issue in report.issues)
                raise RuntimeError(f"@gpu(required=True) function '{callee}' is not CUDA-eligible: {reasons}")
            return self.cpu_fallback(callee, args), GpuExecutionDecision(False, "ineligible")

        selection = backend_selection_report(
            self.preferred_backend,
            device=self.preferred_device,
            policy="required" if policy == "required" else "best_effort",
        )
        if selection.selected_backend == "cpu":
            if policy == "required":
                raise RuntimeError("GPU backend unavailable for required @gpu call")
            return self.cpu_fallback(callee, args), GpuExecutionDecision(
                False,
                "GPU_BACKEND_UNAVAILABLE",
                details={"selection_report": selection.to_dict()},
            )

        if selection.selected_backend != "cuda":
            if policy == "required":
                raise RuntimeError(
                    f"@gpu(required=True) function '{callee}' selected unsupported backend '{selection.selected_backend}'"
                )
            return self.cpu_fallback(callee, args), GpuExecutionDecision(
                False,
                f"{selection.selected_backend.upper()}_BACKEND_NOT_IMPLEMENTED",
                details={"selection_report": selection.to_dict()},
            )

        kernel_ir, launch_override, dispatch_inputs, output_plan = lower_hlir_function_to_backend_ir(function, args=args, mode=mode)
        try:
            result = dispatch_kernel_ir(
                kernel_ir,
                backend=selection.selected_backend,
                target="generic",
                inputs=dispatch_inputs,
                launch_override=launch_override,
                strict_verify=False,
                allow_cpu_fallback=False,
                device=selection.selected_device,
            )
        except Exception as err:
            if policy == "required":
                raise RuntimeError(f"@gpu(required=True) function '{callee}' failed during CUDA compile/launch: {err}") from None
            return self.cpu_fallback(callee, args), GpuExecutionDecision(
                False,
                "GPU_LAUNCH_FAILED",
                details={"error": str(err), "selection_report": selection.to_dict()},
            )

        if result.executed_backend != "cuda":
            if policy == "required":
                raise RuntimeError(f"@gpu(required=True) function '{callee}' did not execute on CUDA")
            return self.cpu_fallback(callee, args), GpuExecutionDecision(
                False,
                "GPU_RUNTIME_FELL_BACK",
                details={"selection_report": selection.to_dict(), "dispatch": result.to_dict()},
            )

        value = _apply_output_plan(function, args, result.outputs, output_plan)
        details = {
            "backend": "cuda",
            "function": callee,
            "mode": mode,
            "kernel_names": list(result.bundle.entrypoints),
            "cache_key": result.bundle.cache_key,
            "cuda_source": result.bundle.binaries.get("cuda_cpp", ""),
            "trace": result.outputs.get("trace", {}),
            "selection_report": selection.to_dict(),
        }
        if self.on_gpu_execution is not None:
            self.on_gpu_execution(details)
        return value, GpuExecutionDecision(True, details=details)


def lower_hlir_function_to_backend_ir(
    function: HFunction,
    *,
    args: list[Any],
    mode: str,
) -> tuple[dict[str, Any], dict[str, int] | None, dict[str, Any], dict[str, Any]]:
    if mode == "graph":
        return _lower_hlir_function_graph_mode(function, args=args)
    return _lower_hlir_function_kernel_mode(function, args=args)


def _lower_hlir_function_kernel_mode(
    function: HFunction,
    *,
    args: list[Any],
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any], dict[str, Any]]:
    """Lower a single canonical elementwise loop into one CUDA kernel."""

    regions = [region for region in extract_hlir_gpu_regions(function) if region.get("kind") == "elementwise"]
    if not regions:
        raise RuntimeError(f"function '{function.name}' is CUDA-eligible but no elementwise loop region was found")

    region = regions[0]
    param_types = {str(param.get("name")): str(param.get("type", "dynamic")) for param in function.params}
    output_name = str(region.get("output_buffer"))
    output_length = _resolve_region_extent_length(region, function, args)
    block_x = min(256, max(1, output_length))
    grid_x = max(1, math.ceil(max(output_length, 1) / block_x))

    ops = _build_elementwise_ops(region, param_types)
    params = _signature_params(function, extra_scalars=[])
    kernel_hash = hashlib.sha256(function.name.encode("utf-8")).hexdigest()[:12]
    kernel_name = f"manv_{function.name.replace('.', '_')}_{kernel_hash}"

    kernel_ir = {
        "version": "0.1",
        "source": function.name,
        "kernels": [
            {
                "kernel_name": kernel_name,
                "function": function.name,
                "signature": {"params": params, "return_policy": "void"},
                "launch_model": {"grid_x": grid_x, "grid_y": 1, "grid_z": 1, "block_x": block_x, "block_y": 1, "block_z": 1, "shared_bytes": 0},
                "memory_regions": _memory_regions_from_params(params),
                "debug_meta": {
                    "kernel_kind": "elementwise",
                    "hlir_function": function.name,
                    "lowering_mode": "kernel",
                    "region_id": region.get("id"),
                },
                "blocks": [{"id": "entry", "ops": ops, "terminator": "ret"}],
            }
        ],
    }
    return (
        kernel_ir,
        {"grid_x": grid_x, "block_x": block_x},
        _dispatch_inputs_for_function(function, args),
        {"kind": "buffer", "name": output_name},
    )


def _lower_hlir_function_graph_mode(
    function: HFunction,
    *,
    args: list[Any],
) -> tuple[dict[str, Any], None, dict[str, Any], dict[str, Any]]:
    """Lower HLIR loop regions into a possibly multi-kernel graph-mode module.

    The current graph-mode implementation focuses on the patterns ManV can
    explain clearly today:
    - fused elementwise loops
    - reductions lowered into partial + finalize kernels
    - sequential multi-kernel execution for multiple regions in one function
    """

    regions = extract_hlir_gpu_regions(function)
    if not regions:
        raise RuntimeError(f"function '{function.name}' has no graph-mode GPU regions")

    param_types = {str(param.get("name")): str(param.get("type", "dynamic")) for param in function.params}
    dispatch_inputs = _dispatch_inputs_for_function(function, args)
    kernels: list[dict[str, Any]] = []
    output_plan: dict[str, Any] = {"kind": "void"}

    for index, region in enumerate(regions):
        kind = str(region.get("kind"))
        if kind == "elementwise":
            output_name = str(region.get("output_buffer"))
            length = _resolve_region_extent_length(region, function, args)
            block_x = min(256, max(1, length))
            grid_x = max(1, math.ceil(max(length, 1) / block_x))
            params = _signature_params(function, extra_scalars=[])
            kernels.append(
                {
                    "kernel_name": f"{_stable_kernel_prefix(function.name)}_graph_{index}",
                    "function": function.name,
                    "signature": {"params": params, "return_policy": "void"},
                    "launch_model": {"grid_x": grid_x, "grid_y": 1, "grid_z": 1, "block_x": block_x, "block_y": 1, "block_z": 1, "shared_bytes": 0},
                    "memory_regions": _memory_regions_from_params(params),
                    "debug_meta": {
                        "kernel_kind": "elementwise",
                        "hlir_function": function.name,
                        "lowering_mode": "graph",
                        "region_id": region.get("id"),
                    },
                    "blocks": [{"id": "entry", "ops": _build_elementwise_ops(region, param_types), "terminator": "ret"}],
                }
            )
            output_plan = {"kind": "buffer", "name": output_name}
            continue

        if kind == "reduction":
            reduction_kernels, output_plan, temp_inputs = _build_reduction_kernels(function, region, args, kernel_index=index)
            kernels.extend(reduction_kernels)
            dispatch_inputs.update(temp_inputs)
            continue

        raise RuntimeError(f"unsupported graph-mode region kind: {kind}")

    return (
        {
            "version": "0.1",
            "source": function.name,
            "kernels": kernels,
            "metadata": {"lowering_mode": "graph", "function": function.name},
        },
        None,
        dispatch_inputs,
        output_plan,
    )


def _build_reduction_kernels(
    function: HFunction,
    region: dict[str, Any],
    args: list[Any],
    *,
    kernel_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    reduction_op = str(region.get("reduction_op", "+"))
    if reduction_op != "+":
        raise RuntimeError(f"only '+' reductions are currently supported, got {reduction_op!r}")

    length = _resolve_region_extent_length(region, function, args)
    block_x = 256
    grid_x = max(1, min(math.ceil(max(length, 1) / 512), 4096))
    partial_name = f"__manv_partial_{kernel_index}"
    return_name = f"__manv_return_{kernel_index}"
    partial_count = grid_x
    param_types = {str(param.get("name")): str(param.get("type", "dynamic")) for param in function.params}
    value_dtype = _expr_dtype(region.get("expr"), param_types)

    partial_params = [
        {
            "index": 0,
            "name": partial_name,
            "kind": "buffer",
            "dtype": value_dtype,
            "by_ref": True,
            "alignment": 8,
            "address_space": "global",
        },
        *_signature_params(function, extra_scalars=[]),
    ]
    finalize_params = [
        {
            "index": 0,
            "name": return_name,
            "kind": "buffer",
            "dtype": value_dtype,
            "by_ref": True,
            "alignment": 8,
            "address_space": "global",
        },
        {
            "index": 1,
            "name": partial_name,
            "kind": "buffer",
            "dtype": value_dtype,
            "by_ref": True,
            "alignment": 8,
            "address_space": "global",
        },
    ]

    reduction_value_ops, reduction_value_name = _build_reduction_value_ops(region, param_types)
    partial_kernel = {
        "kernel_name": f"{_stable_kernel_prefix(function.name)}_reduce_partial_{kernel_index}",
        "function": function.name,
        "signature": {"params": partial_params, "return_policy": "void"},
        "launch_model": {"grid_x": grid_x, "grid_y": 1, "grid_z": 1, "block_x": block_x, "block_y": 1, "block_z": 1, "shared_bytes": 0},
        "memory_regions": _memory_regions_from_params(partial_params),
        "debug_meta": {
            "kernel_kind": "reduction_partial",
            "hlir_function": function.name,
            "lowering_mode": "graph",
            "region_id": region.get("id"),
            "reduction_op": reduction_op,
            "reduction_value": reduction_value_name,
            "output_buffer": partial_name,
            "value_dtype": value_dtype,
        },
        "blocks": [{"id": "entry", "ops": reduction_value_ops, "terminator": "ret"}],
    }

    finalize_ops = [
        {
            "id": "tid0",
            "opcode": "thread_id_x",
            "inputs": [],
            "outputs": ["tid0"],
            "attrs": {},
            "dtype": "i32",
            "memory_space": "private",
            "provenance": region.get("provenance"),
        },
        {
            "id": "partial_load",
            "opcode": "buffer_load",
            "inputs": ["tid0"],
            "outputs": ["partial_load"],
            "attrs": {"buffer": partial_name},
            "dtype": value_dtype,
            "memory_space": "global",
            "provenance": region.get("provenance"),
        },
    ]
    finalize_kernel = {
        "kernel_name": f"{_stable_kernel_prefix(function.name)}_reduce_finalize_{kernel_index}",
        "function": function.name,
        "signature": {"params": finalize_params, "return_policy": "void"},
        "launch_model": {"grid_x": 1, "grid_y": 1, "grid_z": 1, "block_x": min(256, max(1, partial_count)), "block_y": 1, "block_z": 1, "shared_bytes": 0},
        "memory_regions": _memory_regions_from_params(finalize_params),
        "debug_meta": {
            "kernel_kind": "reduction_finalize",
            "hlir_function": function.name,
            "lowering_mode": "graph",
            "region_id": region.get("id"),
            "reduction_op": reduction_op,
            "reduction_value": "partial_load",
            "output_buffer": return_name,
            "value_dtype": value_dtype,
        },
        "blocks": [{"id": "entry", "ops": finalize_ops, "terminator": "ret"}],
    }

    temp_inputs = {
        partial_name: [0] * partial_count,
        return_name: [0],
    }
    return [partial_kernel, finalize_kernel], {"kind": "return_scalar", "name": return_name}, temp_inputs


def _build_elementwise_ops(region: dict[str, Any], param_types: dict[str, str]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = [
        {
            "id": "tid0",
            "opcode": "thread_id_x",
            "inputs": [],
            "outputs": ["tid0"],
            "attrs": {},
            "dtype": "i32",
            "memory_space": "private",
            "provenance": region.get("provenance"),
        }
    ]
    value_name = _emit_expr_ops(region.get("expr"), param_types, ops, loop_value="tid0")
    ops.append(
        {
            "id": f"store_{region.get('output_buffer')}",
            "opcode": "buffer_store",
            "inputs": ["tid0", value_name],
            "outputs": [],
            "attrs": {"buffer": str(region.get("output_buffer"))},
            "dtype": "void",
            "memory_space": "global",
            "provenance": region.get("provenance"),
        }
    )
    return ops


def _build_reduction_value_ops(region: dict[str, Any], param_types: dict[str, str]) -> tuple[list[dict[str, Any]], str]:
    ops: list[dict[str, Any]] = [
        {
            "id": "tid0",
            "opcode": "thread_id_x",
            "inputs": [],
            "outputs": ["tid0"],
            "attrs": {},
            "dtype": "i32",
            "memory_space": "private",
            "provenance": region.get("provenance"),
        }
    ]
    value_name = _emit_expr_ops(region.get("expr"), param_types, ops, loop_value="tid0")
    return ops, value_name


def _emit_expr_ops(expr: Any, param_types: dict[str, str], ops: list[dict[str, Any]], *, loop_value: str) -> str:
    if not isinstance(expr, dict):
        const_id = f"const_{len(ops)}"
        ops.append(
            {
                "id": const_id,
                "opcode": "const",
                "inputs": [],
                "outputs": [const_id],
                "attrs": {"value": expr},
                "dtype": "i32",
                "memory_space": "private",
                "provenance": None,
            }
        )
        return const_id

    kind = str(expr.get("kind", ""))
    if kind == "const":
        const_id = f"const_{len(ops)}"
        ops.append(
            {
                "id": const_id,
                "opcode": "const",
                "inputs": [],
                "outputs": [const_id],
                "attrs": {"value": expr.get("value")},
                "dtype": str(expr.get("dtype", "i32")),
                "memory_space": "private",
                "provenance": None,
            }
        )
        return const_id

    if kind == "var":
        return str(expr.get("name"))

    if kind == "index":
        base = expr.get("base")
        if not isinstance(base, dict) or str(base.get("kind")) != "var":
            raise RuntimeError("graph-mode only supports array indexing from named buffers")
        load_id = f"load_{len(ops)}"
        ops.append(
            {
                "id": load_id,
                "opcode": "buffer_load",
                "inputs": [loop_value],
                "outputs": [load_id],
                "attrs": {"buffer": str(base.get("name"))},
                "dtype": _expr_dtype(expr, param_types),
                "memory_space": "global",
                "provenance": None,
            }
        )
        return load_id

    if kind == "binop":
        lhs = _emit_expr_ops(expr.get("lhs"), param_types, ops, loop_value=loop_value)
        rhs = _emit_expr_ops(expr.get("rhs"), param_types, ops, loop_value=loop_value)
        out_id = f"bin_{len(ops)}"
        ops.append(
            {
                "id": out_id,
                "opcode": f"binary::{expr.get('op', '+')}",
                "inputs": [lhs, rhs],
                "outputs": [out_id],
                "attrs": {},
                "dtype": _expr_dtype(expr, param_types),
                "memory_space": "private",
                "provenance": None,
            }
        )
        return out_id

    raise RuntimeError(f"unsupported graph-mode expression kind: {kind}")


def _signature_params(function: HFunction, *, extra_scalars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for index, param in enumerate(function.params):
        name = str(param.get("name"))
        dtype = _dtype_from_type(str(param.get("type", "dynamic")))
        is_buffer = str(param.get("type", "")).startswith("array[")
        params.append(
            {
                "index": index,
                "name": name,
                "kind": "buffer" if is_buffer else "scalar",
                "dtype": dtype,
                "by_ref": is_buffer,
                "alignment": 8,
                "address_space": "global" if is_buffer else "private",
            }
        )
    for extra_index, param in enumerate(extra_scalars, start=len(params)):
        params.append({**param, "index": extra_index})
    return params


def _memory_regions_from_params(params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": str(param.get("name")), "dtype": str(param.get("dtype", "i32")), "space": "global"}
        for param in params
        if str(param.get("kind")) == "buffer"
    ]


def _dispatch_inputs_for_function(function: HFunction, args: list[Any]) -> dict[str, Any]:
    return {str(param.get("name")): value for param, value in zip(function.params, args, strict=True)}


def _apply_output_plan(function: HFunction, call_args: list[Any], outputs: dict[str, Any], output_plan: dict[str, Any]) -> Any:
    buffers = outputs.get("buffers", {}) if isinstance(outputs, dict) else {}
    kind = str(output_plan.get("kind", "void"))
    if kind == "buffer":
        output_name = str(output_plan.get("name", ""))
        if output_name and output_name in buffers:
            for param, arg in zip(function.params, call_args, strict=True):
                if str(param.get("name")) != output_name or not isinstance(arg, list):
                    continue
                arg[:] = list(buffers[output_name])
                return None
        return None
    if kind == "return_scalar":
        output_name = str(output_plan.get("name", ""))
        values = buffers.get(output_name, [])
        if isinstance(values, list) and values:
            return values[0]
        return None
    return None


def _resolve_region_extent_length(region: dict[str, Any], function: HFunction, args: list[Any]) -> int:
    extent = region.get("extent", {})
    if isinstance(extent, dict):
        if extent.get("kind") == "len":
            buffer_name = str(extent.get("buffer"))
            for param, value in zip(function.params, args, strict=True):
                if str(param.get("name")) == buffer_name and isinstance(value, list):
                    return len(value)
        if extent.get("kind") == "const":
            return int(extent.get("value", 0) or 0)
    # Deterministic fallback: use the first array argument length when the loop
    # bound could not be resolved statically.
    for value in args:
        if isinstance(value, list):
            return len(value)
    return 0


def _expr_dtype(expr: Any, param_types: dict[str, str]) -> str:
    if not isinstance(expr, dict):
        return "i32"
    kind = str(expr.get("kind", ""))
    dtype = str(expr.get("dtype", ""))
    if dtype:
        return _dtype_from_type(dtype)
    if kind == "var":
        return _dtype_from_type(str(param_types.get(str(expr.get("name")), "i32")))
    if kind == "index":
        base = expr.get("base")
        if isinstance(base, dict) and str(base.get("kind")) == "var":
            return _dtype_from_type(str(param_types.get(str(base.get("name")), "i32")))
    if kind == "binop":
        lhs = _expr_dtype(expr.get("lhs"), param_types)
        rhs = _expr_dtype(expr.get("rhs"), param_types)
        return "f32" if "f32" in {lhs, rhs} else "i32"
    return "i32"


def _stable_kernel_prefix(function_name: str) -> str:
    kernel_hash = hashlib.sha256(function_name.encode("utf-8")).hexdigest()[:12]
    return f"manv_{function_name.replace('.', '_')}_{kernel_hash}"


def _dtype_from_type(type_name: str) -> str:
    normalized = str(type_name)
    if "f32" in normalized or normalized == "float":
        return "f32"
    if normalized == "bool":
        return "bool"
    return "i32"
