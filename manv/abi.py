from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .targets import TargetSpec


ABI_INT = "int"
ABI_FLOAT = "float"
ABI_VECTOR = "vector"
ABI_AGGREGATE = "aggregate"

SYSV_NO_CLASS = "NO"
SYSV_INTEGER = "INTEGER"
SYSV_SSE = "SSE"
SYSV_SSEUP = "SSEUP"
SYSV_X87 = "X87"
SYSV_X87UP = "X87UP"
SYSV_COMPLEX_X87 = "COMPLEX_X87"
SYSV_MEMORY = "MEMORY"

SYSV_FP_CLASSES = {SYSV_SSE, SYSV_SSEUP}
SYSV_GP_CLASSES = {SYSV_INTEGER}


@dataclass(frozen=True)
class ABILocation:
    kind: str
    reg: str | None = None
    regs: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    stack_offset: int | None = None
    by_ref: bool = False
    hidden: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reg": self.reg,
            "regs": list(self.regs),
            "classes": list(self.classes),
            "stack_offset": self.stack_offset,
            "by_ref": self.by_ref,
            "hidden": self.hidden,
        }


@dataclass(frozen=True)
class ABIFrame:
    stack_size: int
    stack_align: int
    shadow_space: int
    callee_saved_used: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack_size": self.stack_size,
            "stack_align": self.stack_align,
            "shadow_space": self.shadow_space,
            "callee_saved_used": list(self.callee_saved_used),
        }


@dataclass(frozen=True)
class ABIFunction:
    name: str
    target: str
    arg_locations: tuple[ABILocation, ...]
    return_location: ABILocation | None
    frame: ABIFrame
    sret: bool
    is_varargs: bool = False
    fixed_param_count: int | None = None
    varargs_gp_reg_count: int = 0
    varargs_fp_reg_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "arg_locations": [a.to_dict() for a in self.arg_locations],
            "return_location": self.return_location.to_dict() if self.return_location else None,
            "frame": self.frame.to_dict(),
            "sret": self.sret,
            "is_varargs": self.is_varargs,
            "fixed_param_count": self.fixed_param_count,
            "varargs_gp_reg_count": self.varargs_gp_reg_count,
            "varargs_fp_reg_count": self.varargs_fp_reg_count,
        }


def classify_type(type_name: str | None) -> str:
    if type_name is None:
        return ABI_INT
    t = type_name.strip()
    if t in {"int", "u8", "usize", "bool", "ptr"}:
        return ABI_INT
    if t in {"float", "f32", "f64"}:
        return ABI_FLOAT
    if t.startswith("vec") or t.startswith("simd"):
        return ABI_VECTOR
    if t in {"str", "array", "map"}:
        return ABI_AGGREGATE
    if t.startswith("array[") or t.startswith("map[") or t.startswith("["):
        return ABI_AGGREGATE
    if t.startswith("struct") or t.startswith("tuple"):
        return ABI_AGGREGATE
    return ABI_AGGREGATE


def classify_sysv_aggregate(type_name: str | None) -> dict[str, Any]:
    """
    SysV AMD64 aggregate classification.

    For v0.1 we fully implement the two-eightbyte aggregate path used by our type
    system (ints/floats/vectors/arrays/tuples/structs), while conservatively routing
    unsupported long-double/x87 forms to MEMORY.
    """
    if type_name is None:
        return {"mode": "memory", "classes": [SYSV_MEMORY], "size": 0}

    t = type_name.strip()
    if classify_type(t) != ABI_AGGREGATE:
        cls = SYSV_SSE if classify_type(t) in {ABI_FLOAT, ABI_VECTOR} else SYSV_INTEGER
        return {"mode": "register", "classes": [cls], "size": _primitive_size(t)}

    layout = _type_layout(t)
    if layout["memory"] or layout["size"] == 0 or layout["size"] > 16:
        return {"mode": "memory", "classes": [SYSV_MEMORY], "size": layout["size"]}

    used_eightbytes = (layout["size"] + 7) // 8
    slots = [SYSV_NO_CLASS] * used_eightbytes

    for offset, size, cls in layout["fields"]:
        if cls == SYSV_NO_CLASS:
            continue
        start = offset // 8
        end = (offset + max(size, 1) - 1) // 8
        for i in range(start, min(end + 1, used_eightbytes)):
            slots[i] = _merge_sysv_class(slots[i], cls)

    slots = _normalize_sysv_classes(slots)

    if not slots or any(c == SYSV_MEMORY for c in slots):
        return {"mode": "memory", "classes": [SYSV_MEMORY], "size": layout["size"]}
    if any(c in {SYSV_X87, SYSV_X87UP, SYSV_COMPLEX_X87} for c in slots):
        return {"mode": "memory", "classes": [SYSV_MEMORY], "size": layout["size"]}
    if len(slots) > 2:
        return {"mode": "memory", "classes": [SYSV_MEMORY], "size": layout["size"]}

    return {"mode": "register", "classes": slots, "size": layout["size"]}


def lower_function_abi(
    fn_name: str,
    param_types: list[str | None],
    return_type: str | None,
    target: TargetSpec,
    estimated_stack: int = 64,
    used_callee_saved: tuple[str, ...] = (),
    *,
    is_varargs: bool = False,
    fixed_param_count: int | None = None,
) -> ABIFunction:
    if is_varargs and not target.supports_varargs:
        raise ValueError(f"target {target.name} does not support varargs")

    gp_index = 0
    fp_index = 0
    stack_offset = 0
    arg_locs: list[ABILocation] = []

    fixed_count = len(param_types) if fixed_param_count is None else fixed_param_count

    def alloc_gp(*, by_ref: bool = False, hidden: bool = False) -> ABILocation:
        nonlocal gp_index, stack_offset
        if gp_index < len(target.gp_arg_regs):
            loc = ABILocation(kind="reg", reg=target.gp_arg_regs[gp_index], by_ref=by_ref, hidden=hidden)
            gp_index += 1
            return loc
        loc = ABILocation(kind="stack", stack_offset=stack_offset, by_ref=by_ref, hidden=hidden)
        stack_offset += target.pointer_size
        return loc

    def alloc_fp() -> ABILocation:
        nonlocal fp_index, stack_offset
        if fp_index < len(target.fp_arg_regs):
            loc = ABILocation(kind="reg", reg=target.fp_arg_regs[fp_index])
            fp_index += 1
            return loc
        loc = ABILocation(kind="stack", stack_offset=stack_offset)
        stack_offset += target.pointer_size
        return loc

    sret = _needs_sret(return_type, target)
    if sret:
        arg_locs.append(alloc_gp(by_ref=True, hidden=True))

    varargs_gp = 0
    varargs_fp = 0

    for idx, param_type in enumerate(param_types):
        cls = classify_type(param_type)
        in_varargs_tail = is_varargs and idx >= fixed_count

        if _uses_sysv_eightbyte(target, cls):
            agg = classify_sysv_aggregate(param_type)
            loc = _alloc_sysv_aggregate_arg(agg, target, gp_index, fp_index, stack_offset)

            if loc is None:
                # Memory-class aggregate argument occupies stack storage.
                size = max(8, _align_up(int(agg.get("size", target.pointer_size)), 8))
                arg_locs.append(ABILocation(kind="stack", stack_offset=stack_offset, classes=(SYSV_MEMORY,)))
                stack_offset += size
            else:
                arg_locs.append(loc["location"])
                gp_index = loc["gp_index"]
                fp_index = loc["fp_index"]
                if in_varargs_tail:
                    varargs_gp += loc["gp_used"]
                    varargs_fp += loc["fp_used"]
            continue

        if cls == ABI_FLOAT or cls == ABI_VECTOR:
            loc = alloc_fp()
            arg_locs.append(loc)
            if in_varargs_tail and loc.kind == "reg":
                varargs_fp += 1
            continue

        if cls == ABI_AGGREGATE:
            loc = alloc_gp(by_ref=True)
            arg_locs.append(loc)
            if in_varargs_tail and loc.kind == "reg":
                varargs_gp += 1
            continue

        loc = alloc_gp()
        arg_locs.append(loc)
        if in_varargs_tail and loc.kind == "reg":
            varargs_gp += 1

    ret_loc = _lower_return_abi(return_type, target, sret)
    if ret_loc is None and not sret and classify_type(return_type) == ABI_AGGREGATE:
        # Conservative fallback for unsupported aggregate return combinations.
        sret = True

    stack_size = _align_up(max(estimated_stack, stack_offset), target.stack_align)
    frame = ABIFrame(
        stack_size=stack_size,
        stack_align=target.stack_align,
        shadow_space=target.shadow_space,
        callee_saved_used=tuple(r for r in used_callee_saved if r in target.callee_saved_gp),
    )

    return ABIFunction(
        name=fn_name,
        target=target.name,
        arg_locations=tuple(arg_locs),
        return_location=ret_loc,
        frame=frame,
        sret=sret,
        is_varargs=is_varargs,
        fixed_param_count=fixed_count,
        varargs_gp_reg_count=varargs_gp,
        varargs_fp_reg_count=varargs_fp,
    )


def _needs_sret(return_type: str | None, target: TargetSpec) -> bool:
    if classify_type(return_type) != ABI_AGGREGATE:
        return False
    if target.aggregate_policy == "sysv_eightbyte" and target.abi == "sysv":
        return classify_sysv_aggregate(return_type)["mode"] == "memory"
    return True


def _lower_return_abi(return_type: str | None, target: TargetSpec, sret: bool) -> ABILocation | None:
    if sret:
        return None

    cls = classify_type(return_type)
    if cls == ABI_AGGREGATE and target.aggregate_policy == "sysv_eightbyte" and target.abi == "sysv":
        agg = classify_sysv_aggregate(return_type)
        if agg["mode"] != "register":
            return None

        gp_idx = 0
        fp_idx = 0
        regs: list[str] = []
        for klass in agg["classes"]:
            if klass in SYSV_FP_CLASSES:
                if fp_idx >= len(target.fp_ret_regs):
                    return None
                regs.append(target.fp_ret_regs[fp_idx])
                fp_idx += 1
            else:
                if gp_idx >= len(target.gp_ret_regs):
                    return None
                regs.append(target.gp_ret_regs[gp_idx])
                gp_idx += 1

        return ABILocation(kind="regs", regs=tuple(regs), classes=tuple(agg["classes"]))

    if cls in {ABI_FLOAT, ABI_VECTOR} and target.fp_ret_regs:
        return ABILocation(kind="reg", reg=target.fp_ret_regs[0])

    if target.gp_ret_regs:
        return ABILocation(kind="reg", reg=target.gp_ret_regs[0])

    return ABILocation(kind="stack", stack_offset=0)


def _uses_sysv_eightbyte(target: TargetSpec, cls: str) -> bool:
    return target.abi == "sysv" and target.aggregate_policy == "sysv_eightbyte" and cls == ABI_AGGREGATE


def _alloc_sysv_aggregate_arg(
    agg: dict[str, Any],
    target: TargetSpec,
    gp_index: int,
    fp_index: int,
    stack_offset: int,
) -> dict[str, Any] | None:
    if agg["mode"] != "register":
        return None

    classes: list[str] = list(agg["classes"])
    gp_need = sum(1 for c in classes if c in SYSV_GP_CLASSES)
    fp_need = sum(1 for c in classes if c in SYSV_FP_CLASSES)

    if gp_index + gp_need > len(target.gp_arg_regs) or fp_index + fp_need > len(target.fp_arg_regs):
        return None

    regs: list[str] = []
    gp_used = 0
    fp_used = 0
    for klass in classes:
        if klass in SYSV_FP_CLASSES:
            regs.append(target.fp_arg_regs[fp_index])
            fp_index += 1
            fp_used += 1
        else:
            regs.append(target.gp_arg_regs[gp_index])
            gp_index += 1
            gp_used += 1

    return {
        "location": ABILocation(kind="regs", regs=tuple(regs), classes=tuple(classes), stack_offset=stack_offset),
        "gp_index": gp_index,
        "fp_index": fp_index,
        "gp_used": gp_used,
        "fp_used": fp_used,
    }


def _primitive_size(t: str) -> int:
    if t in {"u8", "bool"}:
        return 1
    if t in {"f32"}:
        return 4
    if t.startswith("vec"):
        return 16
    if t in {"float", "f64", "int", "usize", "ptr"}:
        return 8
    return 8


def _type_layout(type_name: str) -> dict[str, Any]:
    t = type_name.strip()

    if t in {"str", "array", "map"}:
        # Dynamic runtime-managed aggregates are memory-class by policy.
        return {"size": 24, "align": 8, "fields": [], "memory": True}

    if t.startswith("array[") and t.endswith("]") and ";" in t:
        inner = t[len("array[") : -1]
        elem_t, count_s = _split_once_top_level(inner, ";")
        count = max(0, int(count_s.strip()))
        elem = _type_layout(elem_t.strip())
        if elem["memory"]:
            return {"size": elem["size"] * count, "align": max(8, elem["align"]), "fields": [], "memory": True}

        fields: list[tuple[int, int, str]] = []
        offset = 0
        for _ in range(count):
            offset = _align_up(offset, elem["align"])
            for fo, fs, fc in elem["fields"]:
                fields.append((offset + fo, fs, fc))
            offset += elem["size"]

        size = _align_up(offset, elem["align"])
        return {"size": size, "align": elem["align"], "fields": fields, "memory": False}

    if t.startswith("[") and t.endswith("]") and ";" in t:
        inner = t[1:-1]
        elem_t, count_s = _split_once_top_level(inner, ";")
        return _type_layout(f"array[{elem_t.strip()};{count_s.strip()}]")

    if t.startswith("tuple[") and t.endswith("]"):
        fields_t = _split_top_level(t[len("tuple[") : -1], ",")
        return _layout_fields(fields_t)

    if t.startswith("struct{") and t.endswith("}"):
        fields_t = _split_top_level(t[len("struct{") : -1], ",")
        typed_fields = [_extract_field_type(f) for f in fields_t]
        return _layout_fields(typed_fields)

    cls = classify_type(t)
    if cls == ABI_FLOAT:
        size = _primitive_size(t)
        return {"size": size, "align": min(8, size), "fields": [(0, size, SYSV_SSE)], "memory": False}

    if cls == ABI_VECTOR:
        size = _primitive_size(t)
        return {"size": size, "align": min(16, size), "fields": [(0, size, SYSV_SSE)], "memory": False}

    if cls == ABI_INT:
        size = _primitive_size(t)
        return {"size": size, "align": min(8, size), "fields": [(0, size, SYSV_INTEGER)], "memory": False}

    return {"size": 24, "align": 8, "fields": [], "memory": True}


def _layout_fields(field_types: list[str]) -> dict[str, Any]:
    offset = 0
    max_align = 1
    fields: list[tuple[int, int, str]] = []
    memory = False

    for ft in field_types:
        layout = _type_layout(ft.strip())
        max_align = max(max_align, layout["align"])
        offset = _align_up(offset, layout["align"])

        if layout["memory"]:
            memory = True

        if layout["fields"]:
            for fo, fs, fc in layout["fields"]:
                fields.append((offset + fo, fs, fc))
        else:
            fields.append((offset, layout["size"], SYSV_MEMORY if layout["memory"] else SYSV_INTEGER))

        offset += layout["size"]

    size = _align_up(offset, max_align)
    return {"size": size, "align": max_align, "fields": fields, "memory": memory}


def _extract_field_type(field: str) -> str:
    part = field.strip()
    if ":" not in part:
        return part
    left, right = part.split(":", 1)
    if left.strip() and right.strip():
        return right.strip()
    return part


def _split_top_level(s: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    opens = "[{("
    closes = "]})"

    for ch in s:
        if ch in opens:
            depth += 1
        elif ch in closes:
            depth = max(0, depth - 1)

        if ch == sep and depth == 0:
            part = "".join(cur).strip()
            if part:
                parts.append(part)
            cur = []
            continue
        cur.append(ch)

    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _split_once_top_level(s: str, sep: str) -> tuple[str, str]:
    parts = _split_top_level(s, sep)
    if len(parts) != 2:
        raise ValueError(f"cannot split '{s}' by '{sep}'")
    return parts[0], parts[1]


def _merge_sysv_class(a: str, b: str) -> str:
    if a == b:
        return a
    if a == SYSV_NO_CLASS:
        return b
    if b == SYSV_NO_CLASS:
        return a
    if a == SYSV_MEMORY or b == SYSV_MEMORY:
        return SYSV_MEMORY

    if a in {SYSV_X87, SYSV_X87UP, SYSV_COMPLEX_X87} or b in {SYSV_X87, SYSV_X87UP, SYSV_COMPLEX_X87}:
        return SYSV_MEMORY

    if a == SYSV_INTEGER or b == SYSV_INTEGER:
        return SYSV_INTEGER

    if a in SYSV_FP_CLASSES and b in SYSV_FP_CLASSES:
        return SYSV_SSE

    return SYSV_MEMORY


def _normalize_sysv_classes(classes: list[str]) -> list[str]:
    out = [c for c in classes if c != SYSV_NO_CLASS]
    if not out:
        return [SYSV_INTEGER]

    for i, c in enumerate(out):
        if c == SYSV_SSEUP:
            if i == 0 or out[i - 1] not in {SYSV_SSE, SYSV_SSEUP}:
                out[i] = SYSV_SSE

    return out


def _align_up(value: int, align: int) -> int:
    if align <= 0:
        return value
    return ((value + align - 1) // align) * align
