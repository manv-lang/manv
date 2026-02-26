from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hlir import HInstruction
from .hlir_interpreter import HLIRTracer


DISALLOWED_EFFECTS = {"may_throw", "dynamic_dispatch", "allocates", "writes_memory", "reads_memory"}


@dataclass
class GraphCaptureTracer(HLIRTracer):
    node_counter: int = 0
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)
    effect_edges: list[dict[str, str]] = field(default_factory=list)
    value_to_node: dict[str, str] = field(default_factory=dict)
    last_effect_node: str | None = None
    skipped_ops: list[dict[str, Any]] = field(default_factory=list)
    kernel_only: bool = False

    def on_instruction(
        self,
        fn_name: str,
        block_label: str,
        instr: HInstruction,
        resolved_args: list[Any],
        result: Any,
    ) -> None:
        del resolved_args

        effects = set(instr.effects or [])
        if self.kernel_only and effects.intersection(DISALLOWED_EFFECTS):
            self.skipped_ops.append(
                {
                    "hlir_id": instr.instr_id,
                    "op": instr.op,
                    "effects": sorted(effects),
                    "reason": "non_kernelizable_effects",
                }
            )
            return

        node_id = self._new_id()
        input_nodes: list[str] = []
        for token in instr.args:
            if token.startswith("%") and token in self.value_to_node:
                input_nodes.append(self.value_to_node[token])
                self.edges.append({"from": self.value_to_node[token], "to": node_id})

        prov = instr.provenance.to_dict() if instr.provenance else None
        effectful = bool(effects)
        node = {
            "id": node_id,
            "op": instr.op,
            "inputs": input_nodes,
            "effectful": effectful,
            "effects": sorted(effects) or ["pure"],
            "dtype": instr.type_name or "dynamic",
            "shape": None,
            "attrs": {
                "fn": fn_name,
                "block": block_label,
                "hlir_id": instr.instr_id,
                "provenance": prov,
                "result": result,
                **instr.attrs,
            },
            "provenance": prov,
            "result": result,
        }

        if effectful:
            if self.last_effect_node is not None:
                self.effect_edges.append({"from": self.last_effect_node, "to": node_id})
            self.last_effect_node = node_id

        self.nodes.append(node)
        if instr.dest is not None:
            self.value_to_node[instr.dest] = node_id

    def to_graph_ir(self) -> dict[str, Any]:
        return {
            "version": "0.1",
            "kind": "tensor_dag_capture",
            "nodes": self.nodes,
            "edges": self.edges,
            "effect_edges": self.effect_edges,
            "skipped": self.skipped_ops,
        }

    def _new_id(self) -> str:
        self.node_counter += 1
        return f"g{self.node_counter}"
