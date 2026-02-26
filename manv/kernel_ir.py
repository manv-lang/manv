from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any


ALLOWED_MEMORY_SPACES = {"global", "shared", "local", "private"}
KERNEL_BLOCKING_EFFECTS = {"may_throw", "dynamic_dispatch"}


@dataclass
class KIRSourceSpan:
    uri: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "start_line": self.start_line,
            "start_col": self.start_col,
            "end_line": self.end_line,
            "end_col": self.end_col,
        }


@dataclass
class KIRProvenance:
    graph_node_id: str | None = None
    hlir_op_id: str | None = None
    source_span: KIRSourceSpan | None = None
    inline_chain: list[KIRSourceSpan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_node_id": self.graph_node_id,
            "hlir_op_id": self.hlir_op_id,
            "source_span": self.source_span.to_dict() if self.source_span else None,
            "inline_chain": [x.to_dict() for x in self.inline_chain],
        }


@dataclass
class KIRType:
    kind: str
    bits: int | None = None
    lanes: int = 1
    shape: list[int] | None = None
    address_space: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "bits": self.bits,
            "lanes": self.lanes,
            "shape": self.shape,
            "address_space": self.address_space,
        }


@dataclass
class KIRValue:
    id: str
    type: KIRType
    storage_class: str = "private"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.to_dict(),
            "storage_class": self.storage_class,
        }


@dataclass
class KIROp:
    id: str
    opcode: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    effects: list[str] = field(default_factory=list)
    dtype: str = "i64"
    memory_space: str = "private"
    provenance: KIRProvenance | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "opcode": self.opcode,
            "op": self.opcode,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attrs": self.attrs,
            "effects": self.effects,
            "dtype": self.dtype,
            "memory_space": self.memory_space,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }
        return out


@dataclass
class KIRBlock:
    id: str
    ops: list[KIROp] = field(default_factory=list)
    terminator: str = "ret"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ops": [op.to_dict() for op in self.ops],
            "terminator": self.terminator,
        }


@dataclass
class KernelParam:
    index: int
    name: str
    kind: str
    dtype: str
    by_ref: bool = True
    alignment: int = 8
    address_space: str = "global"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
            "by_ref": self.by_ref,
            "alignment": self.alignment,
            "address_space": self.address_space,
        }


@dataclass
class KernelSignature:
    params: list[KernelParam]
    return_policy: str = "void"
    debug_buffer_slot: int | None = None
    assert_buffer_slot: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": [p.to_dict() for p in self.params],
            "return_policy": self.return_policy,
            "debug_buffer_slot": self.debug_buffer_slot,
            "assert_buffer_slot": self.assert_buffer_slot,
        }


@dataclass
class LaunchConfig:
    grid_x: int = 1
    grid_y: int = 1
    grid_z: int = 1
    block_x: int = 1
    block_y: int = 1
    block_z: int = 1
    shared_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "grid_z": self.grid_z,
            "block_x": self.block_x,
            "block_y": self.block_y,
            "block_z": self.block_z,
            "shared_bytes": self.shared_bytes,
        }


@dataclass
class KIRKernel:
    name: str
    signature: KernelSignature
    launch_model: LaunchConfig
    blocks: list[KIRBlock]
    memory_regions: list[dict[str, Any]] = field(default_factory=list)
    debug_meta: dict[str, Any] = field(default_factory=dict)
    function: str | None = None

    def all_ops(self) -> list[KIROp]:
        out: list[KIROp] = []
        for block in self.blocks:
            out.extend(block.ops)
        return out

    def to_dict(self) -> dict[str, Any]:
        ops = [op.to_dict() for op in self.all_ops()]
        return {
            "kernel_name": self.name,
            "name": self.name,
            "function": self.function,
            "signature": self.signature.to_dict(),
            "launch_model": self.launch_model.to_dict(),
            "launch": {
                "grid": [self.launch_model.grid_x, self.launch_model.grid_y, self.launch_model.grid_z],
                "block": [self.launch_model.block_x, self.launch_model.block_y, self.launch_model.block_z],
                "thread_index": ["tx", "ty", "tz"],
                "block_index": ["bx", "by", "bz"],
            },
            "blocks": [b.to_dict() for b in self.blocks],
            "ops": ops,
            "buffers": self.memory_regions,
            "memory_regions": self.memory_regions,
            "debug_meta": self.debug_meta,
            "control": "structured",
        }


@dataclass
class KIRModule:
    version: str
    source: str
    kernels: list[KIRKernel]
    constants: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "constants": self.constants,
            "metadata": self.metadata,
            "kernels": [k.to_dict() for k in self.kernels],
            "backend_boundary": {
                "status": "mock_or_external_backend",
                "target": "multi_backend",
                "message": "Kernel IR is ready for backend lowering; CPU reference backend available.",
            },
        }

    def canonical_hash(self) -> str:
        payload = self.to_dict()
        import json

        packed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(packed).hexdigest()


def lower_graph_to_kir_module(graph_ir: dict[str, Any]) -> KIRModule:
    kernels: list[KIRKernel] = []
    source = str(graph_ir.get("source", "<unknown>"))
    kernelization_meta: dict[str, Any] = {}

    if graph_ir.get("kind") == "tensor_dag_capture":
        fn_name = "<capture>"
        nodes = [n for n in graph_ir.get("nodes", []) if isinstance(n, dict)]
        eligible, skipped = _partition_kernelizable_nodes(nodes)
        kernelization_meta[fn_name] = {
            "input_nodes": len(nodes),
            "kernelized_nodes": len(eligible),
            "skipped": skipped,
            "fallback_required": bool(skipped),
        }
        if eligible:
            kernels.append(_build_kernel_from_nodes("captured_kernel_0", fn_name, eligible, source))
    else:
        for fn in graph_ir.get("functions", []):
            name = str(fn.get("name", "anonymous"))
            nodes = [n for n in fn.get("nodes", []) if isinstance(n, dict)]
            eligible, skipped = _partition_kernelizable_nodes(nodes)
            kernelization_meta[name] = {
                "input_nodes": len(nodes),
                "kernelized_nodes": len(eligible),
                "skipped": skipped,
                "fallback_required": bool(skipped),
            }
            if eligible:
                kernels.append(_build_kernel_from_nodes(f"{name}_kernel_0", name, eligible, source))

    module = KIRModule(version=str(graph_ir.get("version", "0.1")), source=source, kernels=kernels)
    module.metadata["kind"] = "kir"
    module.metadata["kernelization"] = kernelization_meta
    return module


def lower_graph_to_kernel(graph_ir: dict[str, Any]) -> dict[str, Any]:
    """Compatibility entrypoint returning dict artifacts expected by existing compiler flow."""
    return lower_graph_to_kir_module(graph_ir).to_dict()


def parse_kir_module(payload: dict[str, Any]) -> KIRModule:
    kernels: list[KIRKernel] = []
    for raw in payload.get("kernels", []):
        params: list[KernelParam] = []
        signature_raw = raw.get("signature")
        if isinstance(signature_raw, dict):
            for p in signature_raw.get("params", []):
                params.append(
                    KernelParam(
                        index=int(p.get("index", len(params))),
                        name=str(p.get("name", f"arg{len(params)}")),
                        kind=str(p.get("kind", "buffer")),
                        dtype=str(p.get("dtype", "i64")),
                        by_ref=bool(p.get("by_ref", True)),
                        alignment=int(p.get("alignment", 8)),
                        address_space=str(p.get("address_space", "global")),
                    )
                )
        else:
            for idx, b in enumerate(raw.get("buffers", [])):
                params.append(
                    KernelParam(
                        index=idx,
                        name=str(b.get("name", f"arg{idx}")),
                        kind="buffer",
                        dtype=str(b.get("dtype", "i64")),
                        by_ref=True,
                        alignment=8,
                        address_space=str(b.get("space", "global")),
                    )
                )

        signature = KernelSignature(
            params=params,
            return_policy=str((signature_raw or {}).get("return_policy", "void")),
            debug_buffer_slot=(signature_raw or {}).get("debug_buffer_slot") if isinstance(signature_raw, dict) else None,
            assert_buffer_slot=(signature_raw or {}).get("assert_buffer_slot") if isinstance(signature_raw, dict) else None,
        )

        lm_raw = raw.get("launch_model")
        if isinstance(lm_raw, dict):
            launch_model = LaunchConfig(
                grid_x=int(lm_raw.get("grid_x", 1)),
                grid_y=int(lm_raw.get("grid_y", 1)),
                grid_z=int(lm_raw.get("grid_z", 1)),
                block_x=int(lm_raw.get("block_x", 1)),
                block_y=int(lm_raw.get("block_y", 1)),
                block_z=int(lm_raw.get("block_z", 1)),
                shared_bytes=int(lm_raw.get("shared_bytes", 0)),
            )
        else:
            launch = raw.get("launch", {})
            grid = launch.get("grid", [1, 1, 1])
            block = launch.get("block", [1, 1, 1])
            launch_model = LaunchConfig(
                grid_x=int(grid[0] if len(grid) > 0 else 1),
                grid_y=int(grid[1] if len(grid) > 1 else 1),
                grid_z=int(grid[2] if len(grid) > 2 else 1),
                block_x=int(block[0] if len(block) > 0 else 1),
                block_y=int(block[1] if len(block) > 1 else 1),
                block_z=int(block[2] if len(block) > 2 else 1),
                shared_bytes=0,
            )

        blocks: list[KIRBlock] = []
        raw_blocks = raw.get("blocks")
        if isinstance(raw_blocks, list) and raw_blocks:
            for blk in raw_blocks:
                ops = [_parse_kir_op(op) for op in blk.get("ops", [])]
                blocks.append(KIRBlock(id=str(blk.get("id", "entry")), ops=ops, terminator=str(blk.get("terminator", "ret"))))
        else:
            ops = [_parse_kir_op(op) for op in raw.get("ops", [])]
            blocks.append(KIRBlock(id="entry", ops=ops, terminator="ret"))

        kernels.append(
            KIRKernel(
                name=str(raw.get("kernel_name", raw.get("name", "kernel"))),
                function=raw.get("function"),
                signature=signature,
                launch_model=launch_model,
                blocks=blocks,
                memory_regions=list(raw.get("memory_regions", raw.get("buffers", []))),
                debug_meta=dict(raw.get("debug_meta", {})),
            )
        )

    return KIRModule(
        version=str(payload.get("version", "0.1")),
        source=str(payload.get("source", "<unknown>")),
        kernels=kernels,
        constants=dict(payload.get("constants", {})),
        metadata=dict(payload.get("metadata", {})),
    )


def _build_kernel_from_nodes(kernel_name: str, function_name: str, nodes: list[dict[str, Any]], source: str) -> KIRKernel:
    ops = [_node_to_kir_op(node, source=source) for node in nodes]
    block = KIRBlock(id="entry", ops=ops, terminator="ret")
    regions = _collect_memory_regions(nodes)
    params = [
        KernelParam(index=i, name=str(buf.get("name", f"arg{i}")), kind="buffer", dtype=str(buf.get("dtype", "i64")), by_ref=True, alignment=8, address_space=str(buf.get("space", "global")))
        for i, buf in enumerate(regions)
    ]
    signature = KernelSignature(params=params, return_policy="void")
    launch_model = LaunchConfig(grid_x=1, grid_y=1, grid_z=1, block_x=1, block_y=1, block_z=1, shared_bytes=0)
    return KIRKernel(
        name=kernel_name,
        function=function_name,
        signature=signature,
        launch_model=launch_model,
        blocks=[block],
        memory_regions=regions,
        debug_meta={"source": source},
    )


def _node_to_kir_op(node: dict[str, Any], source: str) -> KIROp:
    attrs = dict(node.get("attrs", {}))
    if "result" not in attrs and "result" in node:
        attrs["result"] = node.get("result")

    prov = _to_provenance(node.get("provenance"), attrs, node_id=str(node.get("id")), source=source)
    dtype = str(node.get("dtype", "i64"))
    if dtype in {"dynamic", "void", "control", ""}:
        dtype = "i64" if str(node.get("op")) not in {"return", "break", "continue"} else "void"

    space = str(node.get("memory_space", attrs.get("space", "private")))
    if space not in ALLOWED_MEMORY_SPACES:
        space = "private"

    return KIROp(
        id=str(node.get("id")),
        opcode=str(node.get("op")),
        inputs=[str(x) for x in node.get("inputs", [])],
        outputs=[str(x) for x in node.get("outputs", [])],
        attrs=attrs,
        effects=["memory"] if bool(node.get("effectful", False)) else [],
        dtype=dtype,
        memory_space=space,
        provenance=prov,
    )


def _to_provenance(raw: Any, attrs: dict[str, Any], node_id: str, source: str) -> KIRProvenance:
    if not isinstance(raw, dict):
        raw = attrs.get("provenance", {}) if isinstance(attrs.get("provenance"), dict) else {}

    span_raw = raw.get("primary_span") if isinstance(raw, dict) else None
    if not isinstance(span_raw, dict):
        span_raw = raw.get("source_span") if isinstance(raw, dict) else None

    if isinstance(span_raw, dict):
        source_span = KIRSourceSpan(
            uri=str(span_raw.get("uri", source)),
            start_line=int(span_raw.get("start_line", 1)),
            start_col=int(span_raw.get("start_col", 1)),
            end_line=int(span_raw.get("end_line", span_raw.get("start_line", 1))),
            end_col=int(span_raw.get("end_col", span_raw.get("start_col", 1))),
        )
    else:
        source_span = KIRSourceSpan(uri=source, start_line=1, start_col=1, end_line=1, end_col=1)

    return KIRProvenance(
        graph_node_id=str(raw.get("graph_id", raw.get("graph_node_id", node_id))) if isinstance(raw, dict) else node_id,
        hlir_op_id=(str(raw.get("hlir_id")) if isinstance(raw, dict) and raw.get("hlir_id") else None),
        source_span=source_span,
        inline_chain=[],
    )


def _parse_kir_op(raw: dict[str, Any]) -> KIROp:
    attrs = dict(raw.get("attrs", {}))
    raw_prov = raw.get("provenance")
    if raw_prov is None and not isinstance(attrs.get("provenance"), dict):
        prov = None
    else:
        prov = _to_provenance(raw_prov, attrs, node_id=str(raw.get("id", "op")), source="<unknown>")
    return KIROp(
        id=str(raw.get("id", "op")),
        opcode=str(raw.get("opcode", raw.get("op", "unknown"))),
        inputs=[str(x) for x in raw.get("inputs", [])],
        outputs=[str(x) for x in raw.get("outputs", [])],
        attrs=attrs,
        effects=[str(x) for x in raw.get("effects", [])],
        dtype=str(raw.get("dtype", "i64")),
        memory_space=str(raw.get("memory_space", "private")),
        provenance=prov,
    )


def _collect_memory_regions(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names: set[str] = set()
    for node in nodes:
        for out in node.get("outputs", []):
            names.add(str(out))

    return [
        {
            "name": name,
            "dtype": "i64",
            "shape": None,
            "space": "global",
        }
        for name in sorted(names)
    ]

def _partition_kernelizable_nodes(nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for node in nodes:
        effects = {str(e) for e in node.get("effects", [])}
        op = str(node.get("op", ""))
        non_graphable = bool(node.get("attrs", {}).get("non_graphable"))
        reason = None
        if non_graphable:
            reason = "non_graphable_region"
        elif effects.intersection(KERNEL_BLOCKING_EFFECTS):
            reason = "effect_blocked"
        elif op in {"raise", "try_region", "if_region", "while_region"}:
            reason = "control_or_exception"

        if reason is None:
            eligible.append(node)
        else:
            skipped.append(
                {
                    "id": str(node.get("id", "")),
                    "op": op,
                    "effects": sorted(effects),
                    "reason": reason,
                }
            )
    return eligible, skipped

