from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class OutOfMemoryError(RuntimeError):
    pass


@dataclass
class HeapHeader:
    type_ptr: str
    mark: bool = False
    generation: int = 0
    flags: int = 0


@dataclass
class HeapRecord:
    obj_id: int
    header: HeapHeader
    payload: Any


@dataclass
class TypeObject:
    name: str
    base: "TypeObject | None" = None
    methods: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    mro: list[str] = field(default_factory=list)
    _heap_id: int | None = None


@dataclass
class InstanceObject:
    type_obj: TypeObject
    attrs: dict[str, Any] = field(default_factory=dict)
    _heap_id: int | None = None


@dataclass
class BoundMethodObject:
    receiver: InstanceObject
    function: Any
    _heap_id: int | None = None


@dataclass
class ExceptionObject:
    type_obj: TypeObject
    message: str
    payload: Any = None
    stacktrace: list[dict[str, Any]] = field(default_factory=list)
    _heap_id: int | None = None


class Heap:
    """Simple stop-the-world mark-and-sweep heap used by interpreter semantics."""

    def __init__(
        self,
        *,
        deterministic_gc: bool = False,
        gc_stress: bool = False,
        max_objects: int = 0,
    ):
        self.deterministic_gc = deterministic_gc
        self.gc_stress = gc_stress
        self.max_objects = max_objects
        self._next_id = 1
        self._records: dict[int, HeapRecord] = {}
        self._root_providers: list[Callable[[], list[Any]]] = []
        self._alloc_count = 0

    def register_root_provider(self, provider: Callable[[], list[Any]]) -> None:
        self._root_providers.append(provider)

    def allocate(self, type_ptr: str, payload: Any) -> Any:
        # Run deterministic collection before allocation to keep test behavior stable.
        if self.gc_stress:
            self.collect("stress")
        elif self.deterministic_gc and self._alloc_count % 32 == 0:
            self.collect("deterministic")

        if self.max_objects > 0 and len(self._records) >= self.max_objects:
            self.collect("oom-precheck")
            if len(self._records) >= self.max_objects:
                raise OutOfMemoryError("heap allocation limit reached")

        obj_id = self._next_id
        self._next_id += 1
        header = HeapHeader(type_ptr=type_ptr)
        setattr(payload, "_heap_id", obj_id)
        self._records[obj_id] = HeapRecord(obj_id=obj_id, header=header, payload=payload)
        self._alloc_count += 1
        return payload

    def roots_snapshot(self) -> list[int]:
        out: list[int] = []
        for value in self._all_roots():
            rid = self._root_id(value)
            if rid is not None:
                out.append(rid)
        return sorted(set(out))

    def collect(self, reason: str) -> None:
        del reason
        for rec in self._records.values():
            rec.header.mark = False

        roots = self._all_roots()
        for root in roots:
            self._mark(root)

        dead = [obj_id for obj_id, rec in self._records.items() if not rec.header.mark]
        for obj_id in dead:
            del self._records[obj_id]

    def _all_roots(self) -> list[Any]:
        roots: list[Any] = []
        for provider in self._root_providers:
            try:
                roots.extend(provider())
            except Exception:
                continue
        return roots

    def _root_id(self, value: Any) -> int | None:
        rid = getattr(value, "_heap_id", None)
        if isinstance(rid, int) and rid in self._records:
            return rid
        return None

    def _mark(self, value: Any) -> None:
        rid = self._root_id(value)
        if rid is not None:
            rec = self._records[rid]
            if rec.header.mark:
                return
            rec.header.mark = True
            self._mark_refs(rec.payload)
            return

        # Non-heap containers may still hold heap references.
        self._mark_refs(value)

    def _mark_refs(self, value: Any) -> None:
        if isinstance(value, InstanceObject):
            self._mark(value.type_obj)
            for child in value.attrs.values():
                self._mark(child)
            return
        if isinstance(value, TypeObject):
            if value.base is not None:
                self._mark(value.base)
            for child in value.attrs.values():
                self._mark(child)
            return
        if isinstance(value, BoundMethodObject):
            self._mark(value.receiver)
            self._mark(value.function)
            return
        if isinstance(value, ExceptionObject):
            self._mark(value.type_obj)
            self._mark(value.payload)
            return
        if isinstance(value, list):
            for child in value:
                self._mark(child)
            return
        if isinstance(value, dict):
            for k, v in value.items():
                self._mark(k)
                self._mark(v)
