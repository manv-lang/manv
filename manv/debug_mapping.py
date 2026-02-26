from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hlir import HModule, Provenance


@dataclass(frozen=True)
class ExecPoint:
    kind: str
    function: str
    block: str
    op_id: str

    def key(self) -> str:
        return f"{self.kind}:{self.function}:{self.block}:{self.op_id}"


@dataclass
class SourceMap:
    _by_point: dict[str, Provenance] = field(default_factory=dict)
    _by_file_line: dict[tuple[str, int], list[ExecPoint]] = field(default_factory=dict)

    def add(self, point: ExecPoint, provenance: Provenance | None) -> None:
        if provenance is None:
            return
        self._by_point[point.key()] = provenance
        span = provenance.primary_span
        if span is None:
            return
        bucket_key = (span.uri, span.start_line)
        bucket = self._by_file_line.setdefault(bucket_key, [])
        bucket.append(point)

    def bind(self, point: ExecPoint) -> Provenance | None:
        return self._by_point.get(point.key())

    def find_first_point(self, uri: str, line: int, col: int | None = None) -> ExecPoint | None:
        points = list(self._by_file_line.get((uri, line), []))
        if not points:
            return None

        points.sort(key=lambda p: p.op_id)
        if col is None:
            return points[0]

        for point in points:
            prov = self._by_point.get(point.key())
            if prov is None or prov.primary_span is None:
                continue
            if prov.primary_span.start_col >= col:
                return point
        return points[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": {
                key: prov.to_dict()
                for key, prov in sorted(self._by_point.items(), key=lambda item: item[0])
            }
        }


def build_source_map_from_hlir(module: HModule) -> SourceMap:
    source_map = SourceMap()
    for fn in module.functions:
        for block in fn.blocks:
            for instr in block.instructions:
                if instr.instr_id is None:
                    continue
                source_map.add(
                    ExecPoint(kind="HLIR", function=fn.name, block=block.label, op_id=instr.instr_id),
                    instr.provenance,
                )
            term = block.terminator
            if term is not None and term.term_id is not None:
                source_map.add(
                    ExecPoint(kind="HLIR", function=fn.name, block=block.label, op_id=term.term_id),
                    term.provenance,
                )
    return source_map


def provenance_from_kernel_op(op: dict[str, Any]) -> Provenance | None:
    raw = op.get("provenance")
    if not isinstance(raw, dict):
        return None

    span = raw.get("primary_span")
    if not isinstance(span, dict):
        span = raw.get("source_span")

    primary = None
    if isinstance(span, dict):
        from .hlir import SourceSpan

        primary = SourceSpan(
            uri=str(span.get("uri", "")),
            start_line=int(span.get("start_line", 1)),
            start_col=int(span.get("start_col", 1)),
            end_line=int(span.get("end_line", span.get("start_line", 1))),
            end_col=int(span.get("end_col", span.get("start_col", 1))),
        )

    graph_id = raw.get("graph_id", raw.get("graph_node_id"))
    kir_id = raw.get("kir_id", op.get("id"))

    return Provenance(
        primary_span=primary,
        ast_id=raw.get("ast_id"),
        hlir_id=raw.get("hlir_id", raw.get("hlir_op_id")),
        graph_id=str(graph_id) if graph_id is not None else None,
        kir_id=str(kir_id) if kir_id is not None else None,
    )
