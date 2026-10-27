from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Callable
from urllib import parse as urlparse
from urllib import request as urlrequest

from .backends.cuda.runtime import CudaRuntime, cuda_is_available as backend_cuda_is_available
from .gpu_backends import list_backends
from .gpu_dispatch import backend_capability_table, dispatch_kernel_ir


class Effect(StrEnum):
    PURE = "pure"
    READS_MEMORY = "reads_memory"
    WRITES_MEMORY = "writes_memory"
    ALLOCATES = "allocates"
    MAY_THROW = "may_throw"
    IO = "io"
    SLEEP = "sleep"
    DYNAMIC_DISPATCH = "dynamic_dispatch"


@dataclass(frozen=True)
class IntrinsicTypeVar:
    name: str


ANY_T = IntrinsicTypeVar("any")


@dataclass(frozen=True)
class IntrinsicSpec:
    name: str
    arg_types: list[str | IntrinsicTypeVar]
    return_type: str | IntrinsicTypeVar
    effects: set[Effect]
    may_throw: bool
    std_only: bool = True
    deterministic: bool = True
    pure_for_kernel: bool = False
    version: int = 1
    capability: str | None = None


@dataclass(frozen=True)
class IntrinsicCallable:
    name: str


@dataclass(frozen=True)
class StdCallable:
    intrinsic: str


@dataclass(frozen=True)
class StdModule:
    name: str


class IntrinsicNamespace:
    pass


class StdNamespace:
    pass


BUILTIN_ALIASES: dict[str, str] = {
    "print": "io_print",
    "len": "core_len",
    "repr": "core_repr",
    "hash": "core_hash",
    "min": "core_min",
    "max": "core_max",
    "sum": "core_sum",
    "any": "core_any",
    "all": "core_all",
    "sorted": "core_sorted",
    "range": "core_range",
    "enumerate": "core_enumerate",
    "zip": "core_zip",
    "int": "core_int",
    "float": "core_float",
    "bool": "core_bool",
    "str": "core_str",
    "iter": "core_iter",
    "next": "core_next",
    "map_keys": "core_map_keys",
    "map_values": "core_map_values",
    "map_has_key": "core_map_has_key",
    "array_append": "core_array_append",
    "array_pop": "core_array_pop",
    "ptr_null": "ptr_null",
    "ptr_is_null": "ptr_is_null",
    "c_str": "c_str",
    "ptr_to_str": "ptr_to_str",
    "ptr_add": "ptr_add",
}


_INTRINSICS: dict[str, IntrinsicSpec] = {}
_HANDLERS: dict[str, Callable[..., Any]] = {}
_CUDA_RUNTIME: CudaRuntime | None = None


def register_intrinsic(spec: IntrinsicSpec) -> None:
    _INTRINSICS[spec.name] = spec


def register_intrinsic_handler(name: str, fn: Callable[..., Any]) -> None:
    _HANDLERS[name] = fn


def _split_intrinsic_id(name: str) -> tuple[str, int | None]:
    if "@" not in name:
        return name, None
    base, raw_version = name.rsplit("@", 1)
    try:
        return base, int(raw_version)
    except Exception:
        return name, None


def resolve_intrinsic(name: str) -> IntrinsicSpec | None:
    base, requested_version = _split_intrinsic_id(name)
    spec = _INTRINSICS.get(base)
    if spec is None:
        return None
    if requested_version is not None and requested_version != spec.version:
        return None
    return spec


def all_intrinsics() -> list[IntrinsicSpec]:
    return [spec for _, spec in sorted(_INTRINSICS.items(), key=lambda kv: kv[0])]


def intrinsic_effect_names(spec: IntrinsicSpec) -> list[str]:
    return sorted(effect.value for effect in spec.effects)


def intrinsic_public_id(spec: IntrinsicSpec) -> str:
    return f"{spec.name}@{spec.version}"


def resolve_intrinsic_name_from_callee(expr: Any) -> str | None:
    from . import ast

    parts: list[str] = []
    current = expr
    while isinstance(current, ast.AttributeExpr):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.IdentifierExpr) or current.name != "__intrin":
        return None
    parts.reverse()
    if not parts:
        return None
    if parts[0] == "cuda":
        return "_".join(parts)
    return parts[0] if len(parts) == 1 else "_".join(parts)


def resolve_call_alias_name(callee: Any) -> str | None:
    from . import ast

    if not isinstance(callee, ast.IdentifierExpr):
        return None
    return BUILTIN_ALIASES.get(callee.name)


def is_std_source_path(file: str) -> bool:
    try:
        current = Path(file).resolve()
    except Exception:
        return False

    probe = current if current.is_dir() else current.parent
    while True:
        cfg = probe / "project.toml"
        legacy = probe / "manv.toml"
        selected = cfg if cfg.exists() else legacy if legacy.exists() else None
        if selected is not None:
            try:
                import tomllib

                data = tomllib.loads(selected.read_text(encoding="utf-8"))
                project = data.get("project", {}) if isinstance(data, dict) else {}
                package = data.get("package", {}) if isinstance(data, dict) else {}
                name = str((project.get("name") if isinstance(project, dict) else None) or (package.get("name") if isinstance(package, dict) else "")).strip().lower()
                return name == "std"
            except Exception:
                return False
        if probe.parent == probe:
            return False
        probe = probe.parent


def invoke_intrinsic(
    name: str,
    args: list[Any],
    *,
    stdout_write: Callable[[str], None] | None = None,
    stdin_readline: Callable[[], str] | None = None,
    gc_hooks: dict[str, Callable[..., Any]] | None = None,
) -> Any:
    base, _ = _split_intrinsic_id(name)
    spec = resolve_intrinsic(name)
    if spec is None:
        raise RuntimeError(f"unknown intrinsic: {name}")
    fn = _HANDLERS.get(base)
    if fn is None:
        raise RuntimeError(f"intrinsic handler missing: {base}")
    return fn(
        args,
        stdout_write=stdout_write,
        stdin_readline=stdin_readline,
        gc_hooks=gc_hooks or {},
    )


_INTRIN_NS_SHORT: dict[str, str] = {
    # Allow `__intrin.array_append` in addition to `__intrin.core_array_append`.
    "array_append": "core_array_append",
    "array_pop": "core_array_pop",
    "map_keys": "core_map_keys",
    "map_values": "core_map_values",
    "map_has_key": "core_map_has_key",
}


def std_namespace_attr(base: Any, attr: str) -> Any | None:
    if isinstance(base, IntrinsicNamespace):
        resolved = _INTRIN_NS_SHORT.get(attr, attr)
        return IntrinsicCallable(resolved)
    if isinstance(base, StdNamespace):
        if attr in {"core", "io", "fs", "path", "time", "rand", "json", "memory", "gpu", "sys", "os", "process", "url", "http", "str"}:
            return StdModule(attr)
        return None
    if isinstance(base, StdModule):
        key = f"{base.name}.{attr}"
        mapping = {
            "core.len": "core_len",
            "core.repr": "core_repr",
            "core.hash": "core_hash",
            "core.min": "core_min",
            "core.max": "core_max",
            "core.sum": "core_sum",
            "core.any": "core_any",
            "core.all": "core_all",
            "core.sorted": "core_sorted",
            "core.range": "core_range",
            "core.enumerate": "core_enumerate",
            "core.zip": "core_zip",
            "core.int": "core_int",
            "core.float": "core_float",
            "core.bool": "core_bool",
            "core.str": "core_str",
            "core.iter": "core_iter",
            "core.next": "core_next",
            "io.print": "io_print",
            "io.read_line": "io_read_line",
            "fs.exists": "fs_exists",
            "fs.read_text": "fs_read_text",
            "fs.write_text": "fs_write_text",
            "fs.mkdir": "fs_mkdir",
            "fs.list": "fs_list",
            "fs.remove": "fs_remove",
            "path.join": "path_join",
            "path.basename": "path_basename",
            "path.dirname": "path_dirname",
            "path.normalize": "path_normalize",
            "path.is_abs": "path_is_abs",
            "time.now_ms": "time_now_ms",
            "time.monotonic_ms": "time_monotonic_ms",
            "time.sleep_ms": "time_sleep_ms",
            "rand.seed": "rand_seed",
            "rand.int": "rand_int",
            "rand.float": "rand_float",
            "json.parse": "json_parse",
            "json.stringify": "json_stringify",
            "sys.capabilities": "sys_capabilities",
            "sys.require": "sys_require",
            "os.getenv": "os_getenv",
            "os.setenv": "os_setenv",
            "os.getcwd": "os_getcwd",
            "os.chdir": "os_chdir",
            "process.run": "process_run",
            "url.parse": "url_parse",
            "http.request": "http_request",
            "gpu.backends": "gpu_backends",
            "gpu.capabilities": "gpu_capabilities",
            "gpu.dispatch": "gpu_dispatch",
            "memory.collect": "mem_collect",
            "memory.stats": "mem_stats",
            "memory.set_deterministic_gc": "mem_set_deterministic_gc",
            "memory.set_gc_stress": "mem_set_gc_stress",
            "core.str_upper": "str_upper",
            "core.str_lower": "str_lower",
            "core.str_strip": "str_strip",
            "core.str_lstrip": "str_lstrip",
            "core.str_rstrip": "str_rstrip",
            "core.str_split": "str_split",
            "core.str_join": "str_join",
            "core.str_replace": "str_replace",
            "core.str_startswith": "str_startswith",
            "core.str_endswith": "str_endswith",
            "core.str_contains": "str_contains",
            "core.str_find": "str_find",
            "core.str_char_at": "str_char_at",
            "core.str_slice": "str_slice",
            "core.str_to_chars": "str_to_chars",
            "core.str_concat": "str_concat",
            "core.str_repeat": "str_repeat",
            "core.str_is_empty": "str_is_empty",
            "core.str_trim": "str_trim",
            "core.str_pad_left": "str_pad_left",
            "core.str_pad_right": "str_pad_right",
            "core.str_to_int": "str_to_int",
            "core.str_to_float": "str_to_float",
            "core.str_format": "str_format",
            "core.map_keys": "core_map_keys",
            "core.map_values": "core_map_values",
            "core.map_has_key": "core_map_has_key",
            "core.array_append": "core_array_append",
            "core.array_pop": "core_array_pop",
        }
        intrinsic = mapping.get(key)
        if intrinsic is None:
            return None
        return StdCallable(intrinsic=intrinsic)
    return None


def infer_runtime_type_name(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, bytes):
        return "bytes"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "map"
    return "object"


def intrinsic_type_matches(expected: str | IntrinsicTypeVar, got: str | None) -> bool:
    if got is None:
        return True
    if isinstance(expected, IntrinsicTypeVar):
        return True
    if expected == got:
        return True
    if expected == "any":
        return True
    if expected in {"int", "usize", "u8", "i32"} and got in {"int", "i32"}:
        return True
    if expected in {"float", "f32"} and got in {"float", "f32", "int", "i32"}:
        return True
    if expected == "str_or_none" and got in {"str", "none"}:
        return True
    if expected == "array" and got.startswith("array"):
        return True
    if expected == "map" and got.startswith("map"):
        return True
    return False


def _ensure_arity(name: str, args: list[Any], *, min_n: int, max_n: int | None = None) -> None:
    if len(args) < min_n:
        raise TypeError(f"{name} expects at least {min_n} args, got {len(args)}")
    if max_n is not None and len(args) > max_n:
        raise TypeError(f"{name} expects at most {max_n} args, got {len(args)}")


def _expect_type(name: str, value: Any, kind: str) -> None:
    if kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} expects int")
        return
    if kind == "bool":
        if not isinstance(value, bool):
            raise TypeError(f"{name} expects bool")
        return
    if kind == "str":
        if not isinstance(value, str):
            raise TypeError(f"{name} expects str")
        return
    if kind == "str_or_none":
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} expects str or none")
        return
    if kind == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} expects float")
        return
    if kind == "array":
        if not isinstance(value, list):
            raise TypeError(f"{name} expects array")
        return
    if kind == "map":
        if not isinstance(value, dict):
            raise TypeError(f"{name} expects map")
        return


def _math_number(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} expects int or float")
    return float(value)


def _math_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} expects int")
    return int(value)


def _round_ties_away_from_zero(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return value
    if value >= 0.0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def _float_binary_nan_propagating(lhs: Any, rhs: Any, *, op: str) -> float:
    left = _math_number(op, lhs)
    right = _math_number(op, rhs)
    if math.isnan(left) or math.isnan(right):
        return math.nan
    return min(left, right) if op == "min_float" else max(left, right)


def _float_total_sqrt(value: Any) -> float:
    x = _math_number("sqrt_float", value)
    if math.isnan(x):
        return math.nan
    if x < 0.0:
        return math.nan
    return math.sqrt(x)


def _float_total_log(value: Any) -> float:
    x = _math_number("log_float", value)
    if math.isnan(x):
        return math.nan
    if x == 0.0:
        return -math.inf
    if x < 0.0:
        return math.nan
    return math.log(x)


def _fs_exists(args: list[Any], **_: Any) -> bool:
    _ensure_arity("fs_exists", args, min_n=1, max_n=1)
    _expect_type("fs_exists", args[0], "str")
    return Path(args[0]).exists()


def _fs_read_text(args: list[Any], **_: Any) -> str:
    _ensure_arity("fs_read_text", args, min_n=1, max_n=1)
    _expect_type("fs_read_text", args[0], "str")
    return Path(args[0]).read_text(encoding="utf-8")


def _fs_write_text(args: list[Any], **_: Any) -> None:
    _ensure_arity("fs_write_text", args, min_n=2, max_n=2)
    _expect_type("fs_write_text", args[0], "str")
    _expect_type("fs_write_text", args[1], "str")
    Path(args[0]).write_text(args[1], encoding="utf-8")
    return None


def _fs_mkdir(args: list[Any], **_: Any) -> None:
    _ensure_arity("fs_mkdir", args, min_n=2, max_n=2)
    _expect_type("fs_mkdir", args[0], "str")
    _expect_type("fs_mkdir", args[1], "bool")
    Path(args[0]).mkdir(parents=bool(args[1]), exist_ok=True)
    return None


def _fs_list(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("fs_list", args, min_n=1, max_n=1)
    _expect_type("fs_list", args[0], "str")
    return sorted(x.name for x in Path(args[0]).iterdir())


def _fs_remove(args: list[Any], **_: Any) -> None:
    _ensure_arity("fs_remove", args, min_n=1, max_n=1)
    _expect_type("fs_remove", args[0], "str")
    p = Path(args[0])
    if p.is_dir():
        p.rmdir()
    elif p.exists():
        p.unlink()
    return None


def _path_join(args: list[Any], **_: Any) -> str:
    _ensure_arity("path_join", args, min_n=1, max_n=1)
    _expect_type("path_join", args[0], "array")
    parts = [str(x) for x in args[0]]
    return str(Path(parts[0]).joinpath(*parts[1:])) if parts else ""


def _path_basename(args: list[Any], **_: Any) -> str:
    _ensure_arity("path_basename", args, min_n=1, max_n=1)
    _expect_type("path_basename", args[0], "str")
    return Path(args[0]).name


def _path_dirname(args: list[Any], **_: Any) -> str:
    _ensure_arity("path_dirname", args, min_n=1, max_n=1)
    _expect_type("path_dirname", args[0], "str")
    return str(Path(args[0]).parent)


def _path_normalize(args: list[Any], **_: Any) -> str:
    _ensure_arity("path_normalize", args, min_n=1, max_n=1)
    _expect_type("path_normalize", args[0], "str")
    return str(Path(args[0]))


def _path_is_abs(args: list[Any], **_: Any) -> bool:
    _ensure_arity("path_is_abs", args, min_n=1, max_n=1)
    _expect_type("path_is_abs", args[0], "str")
    return Path(args[0]).is_absolute()


def _core_len(args: list[Any], **_: Any) -> int:
    _ensure_arity("core_len", args, min_n=1, max_n=1)
    return len(args[0])


def _core_repr(args: list[Any], **_: Any) -> str:
    _ensure_arity("core_repr", args, min_n=1, max_n=1)
    return repr(args[0])


def _core_hash(args: list[Any], **_: Any) -> int:
    _ensure_arity("core_hash", args, min_n=1, max_n=1)
    return hash(args[0])


def _core_min(args: list[Any], **_: Any) -> Any:
    _ensure_arity("core_min", args, min_n=1, max_n=1)
    _expect_type("core_min", args[0], "array")
    return min(args[0])


def _core_max(args: list[Any], **_: Any) -> Any:
    _ensure_arity("core_max", args, min_n=1, max_n=1)
    _expect_type("core_max", args[0], "array")
    return max(args[0])


def _core_sum(args: list[Any], **_: Any) -> Any:
    _ensure_arity("core_sum", args, min_n=1, max_n=2)
    _expect_type("core_sum", args[0], "array")
    start = args[1] if len(args) > 1 else 0
    return sum(args[0], start)


def _core_any(args: list[Any], **_: Any) -> bool:
    _ensure_arity("core_any", args, min_n=1, max_n=1)
    _expect_type("core_any", args[0], "array")
    return any(args[0])


def _core_all(args: list[Any], **_: Any) -> bool:
    _ensure_arity("core_all", args, min_n=1, max_n=1)
    _expect_type("core_all", args[0], "array")
    return all(args[0])


def _core_sorted(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("core_sorted", args, min_n=1, max_n=1)
    _expect_type("core_sorted", args[0], "array")
    return sorted(args[0])


def _core_range(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("core_range", args, min_n=1, max_n=3)
    if len(args) == 1:
        _expect_type("core_range", args[0], "int")
        return list(range(int(args[0])))
    if len(args) == 2:
        _expect_type("core_range", args[0], "int")
        _expect_type("core_range", args[1], "int")
        return list(range(int(args[0]), int(args[1])))
    _expect_type("core_range", args[0], "int")
    _expect_type("core_range", args[1], "int")
    _expect_type("core_range", args[2], "int")
    return list(range(int(args[0]), int(args[1]), int(args[2])))


def _core_enumerate(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("core_enumerate", args, min_n=1, max_n=2)
    _expect_type("core_enumerate", args[0], "array")
    start = int(args[1]) if len(args) > 1 else 0
    return [[idx, value] for idx, value in enumerate(args[0], start=start)]


def _core_zip(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("core_zip", args, min_n=1)
    for value in args:
        _expect_type("core_zip", value, "array")
    return [list(row) for row in zip(*args)]


def _core_int(args: list[Any], **_: Any) -> int:
    _ensure_arity("core_int", args, min_n=1, max_n=1)
    return int(args[0])


def _core_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("core_float", args, min_n=1, max_n=1)
    return float(args[0])


def _core_bool(args: list[Any], **_: Any) -> bool:
    _ensure_arity("core_bool", args, min_n=1, max_n=1)
    return bool(args[0])


def format_value(value: Any, *, nested: bool = False) -> str:
    """How print() and str() write a value: `true`/`false`/`none`, collections of the same, and a
    string as itself (quoted when it sits inside a collection)."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "none"
    if isinstance(value, str):
        return repr(value) if nested else value
    if isinstance(value, list):
        return "[" + ", ".join(format_value(v, nested=True) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{format_value(k, nested=True)}: {format_value(v, nested=True)}" for k, v in value.items()) + "}"
    return str(value)


def _core_str(args: list[Any], **_: Any) -> str:
    _ensure_arity("core_str", args, min_n=1, max_n=1)
    return format_value(args[0])


def _core_iter(args: list[Any], **_: Any) -> Any:
    _ensure_arity("core_iter", args, min_n=1, max_n=1)
    return iter(args[0])


def _core_next(args: list[Any], **_: Any) -> Any:
    _ensure_arity("core_next", args, min_n=1, max_n=2)
    it = args[0]
    if len(args) == 1:
        return next(it)
    return next(it, args[1])


def _abs_int(args: list[Any], **_: Any) -> int:
    _ensure_arity("abs_int", args, min_n=1, max_n=1)
    value = _math_int("abs_int", args[0])
    if value == -(2**31):
        raise OverflowError("abs_int overflows for INT_MIN")
    return abs(value)


def _abs_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("abs_float", args, min_n=1, max_n=1)
    return abs(_math_number("abs_float", args[0]))


def _min_int(args: list[Any], **_: Any) -> int:
    _ensure_arity("min_int", args, min_n=2, max_n=2)
    return min(_math_int("min_int", args[0]), _math_int("min_int", args[1]))


def _min_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("min_float", args, min_n=2, max_n=2)
    return _float_binary_nan_propagating(args[0], args[1], op="min_float")


def _max_int(args: list[Any], **_: Any) -> int:
    _ensure_arity("max_int", args, min_n=2, max_n=2)
    return max(_math_int("max_int", args[0]), _math_int("max_int", args[1]))


def _max_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("max_float", args, min_n=2, max_n=2)
    return _float_binary_nan_propagating(args[0], args[1], op="max_float")


def _floor_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("floor_float", args, min_n=1, max_n=1)
    return float(math.floor(_math_number("floor_float", args[0])))


def _ceil_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("ceil_float", args, min_n=1, max_n=1)
    return float(math.ceil(_math_number("ceil_float", args[0])))


def _trunc_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("trunc_float", args, min_n=1, max_n=1)
    return float(math.trunc(_math_number("trunc_float", args[0])))


def _round_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("round_float", args, min_n=1, max_n=1)
    return float(_round_ties_away_from_zero(_math_number("round_float", args[0])))


def _sqrt_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("sqrt_float", args, min_n=1, max_n=1)
    return _float_total_sqrt(args[0])


def _exp_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("exp_float", args, min_n=1, max_n=1)
    value = _math_number("exp_float", args[0])
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _log_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("log_float", args, min_n=1, max_n=1)
    return _float_total_log(args[0])


def _sin_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("sin_float", args, min_n=1, max_n=1)
    return math.sin(_math_number("sin_float", args[0]))


def _cos_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("cos_float", args, min_n=1, max_n=1)
    return math.cos(_math_number("cos_float", args[0]))


def _atan2_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("atan2_float", args, min_n=2, max_n=2)
    return math.atan2(_math_number("atan2_float", args[0]), _math_number("atan2_float", args[1]))


def _is_nan(args: list[Any], **_: Any) -> bool:
    _ensure_arity("is_nan", args, min_n=1, max_n=1)
    return math.isnan(_math_number("is_nan", args[0]))


def _is_inf(args: list[Any], **_: Any) -> bool:
    _ensure_arity("is_inf", args, min_n=1, max_n=1)
    return math.isinf(_math_number("is_inf", args[0]))


def _is_finite(args: list[Any], **_: Any) -> bool:
    _ensure_arity("is_finite", args, min_n=1, max_n=1)
    return math.isfinite(_math_number("is_finite", args[0]))


def _str_upper(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_upper", args, min_n=1, max_n=1)
    _expect_type("str_upper", args[0], "str")
    return str(args[0]).upper()


def _str_lower(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_lower", args, min_n=1, max_n=1)
    _expect_type("str_lower", args[0], "str")
    return str(args[0]).lower()


def _str_strip(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_strip", args, min_n=1, max_n=2)
    _expect_type("str_strip", args[0], "str")
    chars = str(args[1]) if len(args) > 1 and args[1] is not None else None
    return str(args[0]).strip(chars)


def _str_lstrip(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_lstrip", args, min_n=1, max_n=2)
    _expect_type("str_lstrip", args[0], "str")
    chars = str(args[1]) if len(args) > 1 and args[1] is not None else None
    return str(args[0]).lstrip(chars)


def _str_rstrip(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_rstrip", args, min_n=1, max_n=2)
    _expect_type("str_rstrip", args[0], "str")
    chars = str(args[1]) if len(args) > 1 and args[1] is not None else None
    return str(args[0]).rstrip(chars)


def _str_split(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("str_split", args, min_n=1, max_n=3)
    _expect_type("str_split", args[0], "str")
    s = str(args[0])
    sep = str(args[1]) if len(args) > 1 and args[1] is not None else None
    maxsplit = int(args[2]) if len(args) > 2 and args[2] is not None else -1
    return s.split(sep, maxsplit) if maxsplit >= 0 else s.split(sep)


def _str_join(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_join", args, min_n=2, max_n=2)
    _expect_type("str_join", args[0], "str")
    _expect_type("str_join", args[1], "array")
    return str(args[0]).join(str(x) for x in args[1])


def _str_replace(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_replace", args, min_n=3, max_n=4)
    _expect_type("str_replace", args[0], "str")
    _expect_type("str_replace", args[1], "str")
    _expect_type("str_replace", args[2], "str")
    count = int(args[3]) if len(args) > 3 and args[3] is not None else -1
    return str(args[0]).replace(str(args[1]), str(args[2]), count) if count >= 0 else str(args[0]).replace(str(args[1]), str(args[2]))


def _str_startswith(args: list[Any], **_: Any) -> bool:
    _ensure_arity("str_startswith", args, min_n=2, max_n=2)
    _expect_type("str_startswith", args[0], "str")
    _expect_type("str_startswith", args[1], "str")
    return str(args[0]).startswith(str(args[1]))


def _str_endswith(args: list[Any], **_: Any) -> bool:
    _ensure_arity("str_endswith", args, min_n=2, max_n=2)
    _expect_type("str_endswith", args[0], "str")
    _expect_type("str_endswith", args[1], "str")
    return str(args[0]).endswith(str(args[1]))


def _str_contains(args: list[Any], **_: Any) -> bool:
    _ensure_arity("str_contains", args, min_n=2, max_n=2)
    _expect_type("str_contains", args[0], "str")
    _expect_type("str_contains", args[1], "str")
    return str(args[1]) in str(args[0])


def _str_find(args: list[Any], **_: Any) -> int:
    _ensure_arity("str_find", args, min_n=2, max_n=2)
    _expect_type("str_find", args[0], "str")
    _expect_type("str_find", args[1], "str")
    return str(args[0]).find(str(args[1]))


def _str_rfind(args: list[Any], **_: Any) -> int:
    _ensure_arity("str_rfind", args, min_n=2, max_n=2)
    _expect_type("str_rfind", args[0], "str")
    _expect_type("str_rfind", args[1], "str")
    return str(args[0]).rfind(str(args[1]))


def _str_length(args: list[Any], **_: Any) -> int:
    _ensure_arity("str_length", args, min_n=1, max_n=1)
    _expect_type("str_length", args[0], "str")
    return len(str(args[0]))


def _str_char_at(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_char_at", args, min_n=2, max_n=2)
    _expect_type("str_char_at", args[0], "str")
    _expect_type("str_char_at", args[1], "int")
    s = str(args[0])
    idx = int(args[1])
    if idx < 0 or idx >= len(s):
        raise IndexError(f"str_char_at: index {idx} out of range for string of length {len(s)}")
    return s[idx]


def _str_slice(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_slice", args, min_n=3, max_n=3)
    _expect_type("str_slice", args[0], "str")
    _expect_type("str_slice", args[1], "int")
    _expect_type("str_slice", args[2], "int")
    return str(args[0])[int(args[1]):int(args[2])]


def _str_to_chars(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("str_to_chars", args, min_n=1, max_n=1)
    _expect_type("str_to_chars", args[0], "str")
    return list(str(args[0]))


def _str_concat(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_concat", args, min_n=2, max_n=2)
    _expect_type("str_concat", args[0], "str")
    _expect_type("str_concat", args[1], "str")
    return str(args[0]) + str(args[1])


def _str_repeat(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_repeat", args, min_n=2, max_n=2)
    _expect_type("str_repeat", args[0], "str")
    _expect_type("str_repeat", args[1], "int")
    return str(args[0]) * int(args[1])


def _str_is_empty(args: list[Any], **_: Any) -> bool:
    _ensure_arity("str_is_empty", args, min_n=1, max_n=1)
    _expect_type("str_is_empty", args[0], "str")
    return len(str(args[0])) == 0


def _str_trim(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_trim", args, min_n=1, max_n=1)
    _expect_type("str_trim", args[0], "str")
    return str(args[0]).strip()


def _str_pad_left(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_pad_left", args, min_n=2, max_n=3)
    _expect_type("str_pad_left", args[0], "str")
    _expect_type("str_pad_left", args[1], "int")
    fill = str(args[2]) if len(args) > 2 and args[2] is not None else " "
    return str(args[0]).rjust(int(args[1]), fill[0] if fill else " ")


def _str_pad_right(args: list[Any], **_: Any) -> str:
    _ensure_arity("str_pad_right", args, min_n=2, max_n=3)
    _expect_type("str_pad_right", args[0], "str")
    _expect_type("str_pad_right", args[1], "int")
    fill = str(args[2]) if len(args) > 2 and args[2] is not None else " "
    return str(args[0]).ljust(int(args[1]), fill[0] if fill else " ")


def _str_to_int(args: list[Any], **_: Any) -> int:
    _ensure_arity("str_to_int", args, min_n=1, max_n=2)
    _expect_type("str_to_int", args[0], "str")
    base = int(args[1]) if len(args) > 1 and args[1] is not None else 10
    return int(str(args[0]), base)


def _str_to_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("str_to_float", args, min_n=1, max_n=1)
    _expect_type("str_to_float", args[0], "str")
    return float(str(args[0]))


def _str_format(args: list[Any], **_: Any) -> str:
    import re
    _ensure_arity("str_format", args, min_n=1)
    _expect_type("str_format", args[0], "str")
    template = str(args[0])
    # If called as str_format(template, array) from ManV stdlib, unpack the array.
    # If called variadically as str_format(template, v0, v1, ...), use args[1:] directly.
    if len(args) == 2 and isinstance(args[1], list):
        fmt_args: list[Any] = args[1]
    else:
        fmt_args = args[1:]
    result = template
    for i, val in enumerate(fmt_args):
        result = result.replace(f"{{{i}}}", str(val))
    idx = [0]

    def replace_bare(m: re.Match[str]) -> str:
        val = str(fmt_args[idx[0]]) if idx[0] < len(fmt_args) else m.group(0)
        idx[0] += 1
        return val

    result = re.sub(r'\{\}', replace_bare, result)
    return result


def _core_map_keys(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("core_map_keys", args, min_n=1, max_n=1)
    _expect_type("core_map_keys", args[0], "map")
    return list(args[0].keys())


def _core_map_values(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("core_map_values", args, min_n=1, max_n=1)
    _expect_type("core_map_values", args[0], "map")
    return list(args[0].values())


def _core_map_has_key(args: list[Any], **_: Any) -> bool:
    _ensure_arity("core_map_has_key", args, min_n=2, max_n=2)
    _expect_type("core_map_has_key", args[0], "map")
    return args[1] in args[0]


def _core_map_delete(args: list[Any], **_: Any) -> None:
    _ensure_arity("core_map_delete", args, min_n=2, max_n=2)
    _expect_type("core_map_delete", args[0], "map")
    args[0].pop(args[1], None)


def _core_array_push(args: list[Any], **_: Any) -> None:
    """Mutating push: appends element to the array in-place."""
    _ensure_arity("core_array_push", args, min_n=2, max_n=2)
    _expect_type("core_array_push", args[0], "array")
    args[0].append(args[1])


def _core_array_slice(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("core_array_slice", args, min_n=3, max_n=3)
    _expect_type("core_array_slice", args[0], "array")
    start = int(args[1])
    stop = int(args[2])
    return list(args[0])[start:stop]


def _core_array_append(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("core_array_append", args, min_n=2, max_n=2)
    _expect_type("core_array_append", args[0], "array")
    args[0].append(args[1])
    return args[0]


def _core_array_pop(args: list[Any], **_: Any) -> Any:
    _ensure_arity("core_array_pop", args, min_n=1, max_n=2)
    _expect_type("core_array_pop", args[0], "array")
    if not args[0]:
        raise IndexError("pop from empty array")
    if len(args) == 2:
        idx = args[1]
        if not isinstance(idx, int):
            raise TypeError("array_pop index must be int")
        return args[0].pop(idx)
    return args[0].pop()


def _io_print(args: list[Any], *, stdout_write: Callable[[str], None] | None = None, **_: Any) -> None:
    writer = stdout_write or (lambda _: None)
    parts = args
    if len(args) == 1 and isinstance(args[0], list):
        parts = args[0]
    writer(" ".join(format_value(x) for x in parts) + "\n")
    return None


def _io_read_line(args: list[Any], *, stdin_readline: Callable[[], str] | None = None, **_: Any) -> str:
    _ensure_arity("io_read_line", args, min_n=0, max_n=0)
    reader = stdin_readline or (lambda: "")
    raw = reader()
    return raw.rstrip("\r\n")


def _time_now_ms(args: list[Any], **_: Any) -> int:
    _ensure_arity("time_now_ms", args, min_n=0, max_n=0)
    return int(time.time() * 1000)


def _time_monotonic_ms(args: list[Any], **_: Any) -> int:
    _ensure_arity("time_monotonic_ms", args, min_n=0, max_n=0)
    return int(time.monotonic() * 1000)


def _time_sleep_ms(args: list[Any], **_: Any) -> None:
    _ensure_arity("time_sleep_ms", args, min_n=1, max_n=1)
    _expect_type("time_sleep_ms", args[0], "int")
    time.sleep(max(0, int(args[0])) / 1000.0)
    return None


def _rand_seed(args: list[Any], **_: Any) -> None:
    _ensure_arity("rand_seed", args, min_n=1, max_n=1)
    _expect_type("rand_seed", args[0], "int")
    random.seed(int(args[0]))
    return None


def _rand_int(args: list[Any], **_: Any) -> int:
    _ensure_arity("rand_int", args, min_n=2, max_n=2)
    _expect_type("rand_int", args[0], "int")
    _expect_type("rand_int", args[1], "int")
    lo = int(args[0])
    hi = int(args[1])
    if hi < lo:
        raise ValueError("rand_int requires hi >= lo")
    return random.randint(lo, hi)


def _rand_float(args: list[Any], **_: Any) -> float:
    _ensure_arity("rand_float", args, min_n=0, max_n=0)
    return random.random()


def _json_parse(args: list[Any], **_: Any) -> dict[str, Any]:
    _ensure_arity("json_parse", args, min_n=1, max_n=1)
    _expect_type("json_parse", args[0], "str")
    parsed = json.loads(args[0])
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def _json_stringify(args: list[Any], **_: Any) -> str:
    _ensure_arity("json_stringify", args, min_n=1, max_n=1)
    return json.dumps(args[0], sort_keys=True)


def _sys_capabilities(args: list[Any], **_: Any) -> dict[str, Any]:
    _ensure_arity("sys_capabilities", args, min_n=0, max_n=0)
    return {
        "fs": True,
        "path": True,
        "time": True,
        "random": True,
        "json": True,
        "process": True,
        "network": True,
        "threading": True,
        "compression": True,
        "gpu": True,
    }


def _sys_require(args: list[Any], **_: Any) -> None:
    _ensure_arity("sys_require", args, min_n=1, max_n=1)
    _expect_type("sys_require", args[0], "str")
    caps = _sys_capabilities([])
    if not bool(caps.get(str(args[0]), False)):
        raise RuntimeError(f"required capability not available: {args[0]}")
    return None


def _os_getenv(args: list[Any], **_: Any) -> str:
    _ensure_arity("os_getenv", args, min_n=1, max_n=2)
    _expect_type("os_getenv", args[0], "str")
    default = None
    if len(args) > 1:
        _expect_type("os_getenv", args[1], "str_or_none")
        default = args[1]
    value = os.getenv(str(args[0]), default)
    return "" if value is None else str(value)


def _os_setenv(args: list[Any], **_: Any) -> None:
    _ensure_arity("os_setenv", args, min_n=2, max_n=2)
    _expect_type("os_setenv", args[0], "str")
    _expect_type("os_setenv", args[1], "str")
    os.environ[str(args[0])] = str(args[1])
    return None


def _os_getcwd(args: list[Any], **_: Any) -> str:
    _ensure_arity("os_getcwd", args, min_n=0, max_n=0)
    return os.getcwd()


def _os_chdir(args: list[Any], **_: Any) -> None:
    _ensure_arity("os_chdir", args, min_n=1, max_n=1)
    _expect_type("os_chdir", args[0], "str")
    os.chdir(str(args[0]))
    return None


def _process_run(args: list[Any], **_: Any) -> dict[str, Any]:
    _ensure_arity("process_run", args, min_n=1, max_n=1)
    _expect_type("process_run", args[0], "array")
    argv = [str(x) for x in args[0]]
    out = subprocess.run(argv, capture_output=True, text=True, check=False)
    return {
        "exit": int(out.returncode),
        "stdout": str(out.stdout),
        "stderr": str(out.stderr),
    }


def _url_parse(args: list[Any], **_: Any) -> dict[str, Any]:
    _ensure_arity("url_parse", args, min_n=1, max_n=1)
    _expect_type("url_parse", args[0], "str")
    parsed = urlparse.urlparse(str(args[0]))
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }


def _http_request(args: list[Any], **_: Any) -> dict[str, Any]:
    _ensure_arity("http_request", args, min_n=4, max_n=4)
    method, url, body, headers = args
    _expect_type("http_request", method, "str")
    _expect_type("http_request", url, "str")
    _expect_type("http_request", body, "str")
    _expect_type("http_request", headers, "map")
    req = urlrequest.Request(
        url=str(url),
        data=str(body).encode("utf-8"),
        headers={str(k): str(v) for k, v in dict(headers).items()},
        method=str(method).upper(),
    )
    with urlrequest.urlopen(req, timeout=10) as resp:
        payload = resp.read()
        return {
            "status": int(resp.status),
            "body": payload.decode("utf-8", errors="replace"),
            "headers": {str(k): str(v) for k, v in resp.headers.items()},
        }


def _syscall_invoke(args: list[Any], **_: Any) -> dict[str, Any]:
    _ensure_arity("syscall_invoke", args, min_n=2, max_n=2)
    target, call_args = args
    _expect_type("syscall_invoke", call_args, "array")
    platform_name = os.name
    argv = list(call_args)
    try:
        _os_syscall = getattr(os, "syscall", None)
        if isinstance(target, int):
            if _os_syscall is not None:
                result = _os_syscall(int(target), *argv)
                return {"ok": True, "result": result, "platform": platform_name}
            raise OSError("numeric syscall is not available on this platform")

        if not isinstance(target, str):
            raise TypeError("syscall target must be int or str")

        name = target.strip()
        if _os_syscall is not None:
            # Named calls can be passed as numeric strings.
            try:
                num = int(name)
                result = _os_syscall(num, *argv)
                return {"ok": True, "result": result, "platform": platform_name}
            except Exception:
                pass

        if platform_name == "nt":
            if name in {"GetCurrentProcessId", "getpid"}:
                return {"ok": True, "result": os.getpid(), "platform": platform_name}
            if name in {"GetCurrentDirectory", "getcwd"}:
                return {"ok": True, "result": os.getcwd(), "platform": platform_name}
            raise OSError(f"unsupported Windows syscall alias: {name}")

        # POSIX aliases for common operations.
        if name == "getpid":
            return {"ok": True, "result": os.getpid(), "platform": platform_name}
        if name == "getcwd":
            return {"ok": True, "result": os.getcwd(), "platform": platform_name}
        raise OSError(f"unsupported syscall alias: {name}")
    except Exception as exc:
        if isinstance(exc, OSError):
            raise
        raise OSError(str(exc)) from exc


def _mem_collect(args: list[Any], *, gc_hooks: dict[str, Callable[..., Any]] | None = None, **_: Any) -> None:
    _ensure_arity("mem_collect", args, min_n=0, max_n=0)
    hooks = gc_hooks or {}
    fn = hooks.get("collect")
    if fn is not None:
        fn("intrinsic")
    return None


def _mem_stats(args: list[Any], *, gc_hooks: dict[str, Callable[..., Any]] | None = None, **_: Any) -> dict[str, Any]:
    _ensure_arity("mem_stats", args, min_n=0, max_n=0)
    hooks = gc_hooks or {}
    fn = hooks.get("stats")
    if fn is not None:
        return dict(fn())
    return {}


def _mem_set_deterministic_gc(args: list[Any], *, gc_hooks: dict[str, Callable[..., Any]] | None = None, **_: Any) -> None:
    _ensure_arity("mem_set_deterministic_gc", args, min_n=1, max_n=1)
    _expect_type("mem_set_deterministic_gc", args[0], "bool")
    hooks = gc_hooks or {}
    fn = hooks.get("set_deterministic_gc")
    if fn is not None:
        fn(bool(args[0]))
    return None


def _mem_set_gc_stress(args: list[Any], *, gc_hooks: dict[str, Callable[..., Any]] | None = None, **_: Any) -> None:
    _ensure_arity("mem_set_gc_stress", args, min_n=1, max_n=1)
    _expect_type("mem_set_gc_stress", args[0], "bool")
    hooks = gc_hooks or {}
    fn = hooks.get("set_gc_stress")
    if fn is not None:
        fn(bool(args[0]))
    return None


def _cuda_runtime() -> CudaRuntime:
    global _CUDA_RUNTIME
    if _CUDA_RUNTIME is None:
        _CUDA_RUNTIME = CudaRuntime()
    return _CUDA_RUNTIME


def _cuda_buffer_handle(buffer: Any) -> dict[str, Any]:
    return {
        "name": buffer.name,
        "device_ptr": buffer.device_ptr,
        "nbytes": buffer.nbytes,
        "dtype": buffer.dtype,
    }


def _cuda_buffer_from_handle(handle: dict[str, Any]) -> Any:
    runtime = _cuda_runtime()
    ptr = int(handle.get("device_ptr", 0))
    buffer = runtime.lookup_buffer(ptr)
    if buffer is None:
        raise RuntimeError(f"unknown CUDA buffer handle: {ptr}")
    return buffer


def _cuda_is_available(args: list[Any], **_: Any) -> bool:
    _ensure_arity("cuda_is_available", args, min_n=0, max_n=0)
    return backend_cuda_is_available()


def _cuda_device_count(args: list[Any], **_: Any) -> int:
    _ensure_arity("cuda_device_count", args, min_n=0, max_n=0)
    return _cuda_runtime().device_count()


def _cuda_set_device(args: list[Any], **_: Any) -> None:
    _ensure_arity("cuda_set_device", args, min_n=1, max_n=1)
    _expect_type("cuda_set_device", args[0], "int")
    # v1 runtime uses the default device only; device selection is reserved for
    # later runtime expansion while the intrinsic surface is stabilized.
    return None


def _cuda_alloc(args: list[Any], **_: Any) -> dict[str, Any]:
    _ensure_arity("cuda_alloc", args, min_n=2, max_n=2)
    _expect_type("cuda_alloc", args[0], "str")
    _expect_type("cuda_alloc", args[1], "int")
    buffer = _cuda_runtime().alloc(str(args[0]), int(args[1]), "i32")
    return _cuda_buffer_handle(buffer)


def _cuda_free(args: list[Any], **_: Any) -> None:
    _ensure_arity("cuda_free", args, min_n=1, max_n=1)
    _expect_type("cuda_free", args[0], "map")
    _cuda_runtime().free(_cuda_buffer_from_handle(dict(args[0])))
    return None


def _cuda_memcpy_h2d(args: list[Any], **_: Any) -> None:
    _ensure_arity("cuda_memcpy_h2d", args, min_n=2, max_n=2)
    _expect_type("cuda_memcpy_h2d", args[0], "map")
    _expect_type("cuda_memcpy_h2d", args[1], "array")
    _cuda_runtime().copy_h2d(_cuda_buffer_from_handle(dict(args[0])), list(args[1]))
    return None


def _cuda_memcpy_d2h(args: list[Any], **_: Any) -> list[Any]:
    _ensure_arity("cuda_memcpy_d2h", args, min_n=1, max_n=1)
    _expect_type("cuda_memcpy_d2h", args[0], "map")
    return _cuda_runtime().copy_d2h(_cuda_buffer_from_handle(dict(args[0])))


def _cuda_memcpy_d2d(args: list[Any], **_: Any) -> None:
    _ensure_arity("cuda_memcpy_d2d", args, min_n=2, max_n=2)
    _expect_type("cuda_memcpy_d2d", args[0], "map")
    _expect_type("cuda_memcpy_d2d", args[1], "map")
    _cuda_runtime().copy_d2d(_cuda_buffer_from_handle(dict(args[0])), _cuda_buffer_from_handle(dict(args[1])))
    return None


def _cuda_launch(args: list[Any], **_: Any) -> dict[str, Any]:
    _ensure_arity("cuda_launch", args, min_n=3, max_n=3)
    _expect_type("cuda_launch", args[0], "map")
    _expect_type("cuda_launch", args[1], "map")
    _expect_type("cuda_launch", args[2], "map")
    return {"status": "submitted"}


def _cuda_sync(args: list[Any], **_: Any) -> None:
    _ensure_arity("cuda_sync", args, min_n=0, max_n=0)
    _cuda_runtime().sync()
    return None


def _cuda_last_error(args: list[Any], **_: Any) -> str:
    _ensure_arity("cuda_last_error", args, min_n=0, max_n=0)
    return _cuda_runtime().last_error()


def _gpu_backends(args: list[Any], **_: Any) -> list[str]:
    _ensure_arity("gpu_backends", args, min_n=0, max_n=0)
    return [str(b) for b in list_backends()]


def _gpu_capabilities(args: list[Any], **_: Any) -> dict[str, dict[str, Any]]:
    _ensure_arity("gpu_capabilities", args, min_n=0, max_n=0)
    return backend_capability_table()


def _gpu_dispatch(args: list[Any], **_: Any) -> dict[str, Any]:
    _ensure_arity("gpu_dispatch", args, min_n=5, max_n=5)
    kernel, backend, target, inputs, launch = args
    _expect_type("gpu_dispatch", kernel, "map")
    _expect_type("gpu_dispatch", backend, "str")
    _expect_type("gpu_dispatch", target, "str")
    _expect_type("gpu_dispatch", inputs, "map")
    _expect_type("gpu_dispatch", launch, "map")
    result = dispatch_kernel_ir(
        kernel,
        backend=str(backend),
        target=str(target),
        inputs={str(k): list(v) for k, v in dict(inputs).items()},
        launch_override={str(k): int(v) for k, v in dict(launch).items()},
        strict_verify=False,
    )
    return result.to_dict()


def _ptr_null(args: list[Any], **_: Any) -> int:
    """Return a null pointer value (0)."""
    return 0


def _ptr_is_null(args: list[Any], **_: Any) -> bool:
    """Return True if the pointer value is null (0 or None)."""
    _ensure_arity("ptr_is_null", args, min_n=1, max_n=1)
    v = args[0]
    return v == 0 or v is None


def _c_str(args: list[Any], **_: Any) -> bytes:
    """Convert a ManV str to a C-compatible bytes value (UTF-8 encoded, null-terminated)."""
    _ensure_arity("c_str", args, min_n=1, max_n=1)
    v = args[0]
    if isinstance(v, bytes):
        return v
    if isinstance(v, str):
        return v.encode("utf-8")
    return str(v).encode("utf-8")


def _ptr_to_str(args: list[Any], **_: Any) -> str:
    """Convert a C pointer / bytes value to a ManV str."""
    _ensure_arity("ptr_to_str", args, min_n=1, max_n=1)
    v = args[0]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, str):
        return v
    return str(v)


def _ptr_add(args: list[Any], **_: Any) -> int:
    """Add an integer offset to a pointer value and return the result."""
    _ensure_arity("ptr_add", args, min_n=2, max_n=2)
    base = args[0] or 0
    offset = int(args[1])
    return int(base) + offset


def _register_defaults() -> None:
    register_intrinsic(
        IntrinsicSpec("core_len", [ANY_T], "int", {Effect.PURE}, may_throw=True, deterministic=True, pure_for_kernel=False)
    )
    register_intrinsic(IntrinsicSpec("core_repr", [ANY_T], "str", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_hash", [ANY_T], "int", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_min", ["array"], ANY_T, {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_max", ["array"], ANY_T, {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_sum", ["array"], ANY_T, {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_any", ["array"], "bool", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_all", ["array"], "bool", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_sorted", ["array"], "array", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_range", ["int"], "array", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_enumerate", ["array"], "array", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_zip", ["array"], "array", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_int", [ANY_T], "int", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_float", [ANY_T], "float", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_bool", [ANY_T], "bool", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_str", [ANY_T], "str", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_iter", [ANY_T], ANY_T, {Effect.READS_MEMORY}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_next", [ANY_T], ANY_T, {Effect.READS_MEMORY}, may_throw=True))
    mathfx = {Effect.PURE}
    register_intrinsic(IntrinsicSpec("abs_int", ["int"], "int", mathfx, may_throw=True, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("abs_float", ["float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("min_int", ["int", "int"], "int", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("min_float", ["float", "float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("max_int", ["int", "int"], "int", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("max_float", ["float", "float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("floor_float", ["float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("ceil_float", ["float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("trunc_float", ["float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("round_float", ["float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("sqrt_float", ["float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("exp_float", ["float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("log_float", ["float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("sin_float", ["float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("cos_float", ["float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("atan2_float", ["float", "float"], "float", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("is_nan", ["float"], "bool", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("is_inf", ["float"], "bool", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    register_intrinsic(IntrinsicSpec("is_finite", ["float"], "bool", mathfx, may_throw=False, std_only=False, pure_for_kernel=True))
    strfx = {Effect.PURE}
    register_intrinsic(IntrinsicSpec("str_upper", ["str"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_lower", ["str"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_strip", ["str"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_lstrip", ["str"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_rstrip", ["str"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_split", ["str", "str"], "array", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_join", ["str", "array"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_replace", ["str", "str", "str"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_startswith", ["str", "str"], "bool", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_endswith", ["str", "str"], "bool", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_contains", ["str", "str"], "bool", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_find", ["str", "str"], "int", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_char_at", ["str", "int"], "str", strfx, may_throw=True))
    register_intrinsic(IntrinsicSpec("str_slice", ["str", "int", "int"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_to_chars", ["str"], "array", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_concat", ["str", "str"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_repeat", ["str", "int"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_is_empty", ["str"], "bool", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_trim", ["str"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_pad_left", ["str", "int", "str"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_pad_right", ["str", "int", "str"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_to_int", ["str"], "int", strfx, may_throw=True))
    register_intrinsic(IntrinsicSpec("str_to_float", ["str"], "float", strfx, may_throw=True))
    register_intrinsic(IntrinsicSpec("str_format", ["str", "array"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_length", ["str"], "int", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_rfind", ["str", "str"], "int", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_ord", ["str"], "int", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("str_chr", ["int"], "str", strfx, may_throw=False))
    register_intrinsic(IntrinsicSpec("core_map_keys", ["map"], "array", {Effect.PURE}, may_throw=False))
    register_intrinsic(IntrinsicSpec("core_map_values", ["map"], "array", {Effect.PURE}, may_throw=False))
    register_intrinsic(IntrinsicSpec("core_map_has_key", ["map", ANY_T], "bool", {Effect.PURE}, may_throw=False))
    register_intrinsic(IntrinsicSpec("core_map_delete", ["map", ANY_T], "none", {Effect.WRITES_MEMORY}, may_throw=False))
    register_intrinsic(IntrinsicSpec("core_array_append", ["array", ANY_T], "array", {Effect.PURE, Effect.ALLOCATES}, may_throw=False))
    register_intrinsic(IntrinsicSpec("core_array_push", ["array", ANY_T], "none", {Effect.WRITES_MEMORY}, may_throw=False))
    register_intrinsic(IntrinsicSpec("core_array_pop", ["array"], ANY_T, {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("core_array_slice", ["array", "int", "int"], "array", {Effect.PURE, Effect.ALLOCATES}, may_throw=False))
    register_intrinsic(
        IntrinsicSpec("io_print", [ANY_T], "none", {Effect.IO, Effect.WRITES_MEMORY}, may_throw=True, deterministic=True)
    )
    register_intrinsic(
        IntrinsicSpec("io_read_line", [], "str", {Effect.IO, Effect.READS_MEMORY}, may_throw=True, deterministic=False)
    )

    fsfx = {Effect.READS_MEMORY, Effect.WRITES_MEMORY, Effect.MAY_THROW}
    register_intrinsic(IntrinsicSpec("fs_exists", ["str"], "bool", fsfx, may_throw=True))
    register_intrinsic(IntrinsicSpec("fs_read_text", ["str"], "str", fsfx, may_throw=True))
    register_intrinsic(IntrinsicSpec("fs_write_text", ["str", "str"], "none", fsfx, may_throw=True))
    register_intrinsic(IntrinsicSpec("fs_mkdir", ["str", "bool"], "none", fsfx, may_throw=True))
    register_intrinsic(IntrinsicSpec("fs_list", ["str"], "array", fsfx, may_throw=True))
    register_intrinsic(IntrinsicSpec("fs_remove", ["str"], "none", fsfx, may_throw=True))

    register_intrinsic(IntrinsicSpec("path_join", ["array"], "str", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("path_basename", ["str"], "str", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("path_dirname", ["str"], "str", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("path_normalize", ["str"], "str", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("path_is_abs", ["str"], "bool", {Effect.PURE}, may_throw=True))

    register_intrinsic(IntrinsicSpec("time_now_ms", [], "int", {Effect.READS_MEMORY}, may_throw=False, deterministic=False))
    register_intrinsic(
        IntrinsicSpec("time_monotonic_ms", [], "int", {Effect.READS_MEMORY}, may_throw=False, deterministic=False)
    )
    register_intrinsic(IntrinsicSpec("time_sleep_ms", ["int"], "none", {Effect.SLEEP}, may_throw=True, deterministic=False))

    register_intrinsic(IntrinsicSpec("rand_seed", ["int"], "none", {Effect.WRITES_MEMORY}, may_throw=False, deterministic=False))
    register_intrinsic(IntrinsicSpec("rand_int", ["int", "int"], "int", {Effect.READS_MEMORY}, may_throw=True, deterministic=False))
    register_intrinsic(IntrinsicSpec("rand_float", [], "float", {Effect.READS_MEMORY}, may_throw=False, deterministic=False))

    register_intrinsic(IntrinsicSpec("json_parse", ["str"], "map", {Effect.ALLOCATES, Effect.MAY_THROW}, may_throw=True))
    register_intrinsic(IntrinsicSpec("json_stringify", [ANY_T], "str", {Effect.ALLOCATES}, may_throw=True))
    register_intrinsic(IntrinsicSpec("sys_capabilities", [], "map", {Effect.PURE}, may_throw=False))
    register_intrinsic(IntrinsicSpec("sys_require", ["str"], "none", {Effect.MAY_THROW}, may_throw=True))
    register_intrinsic(IntrinsicSpec("os_getenv", ["str"], "str", {Effect.READS_MEMORY}, may_throw=False))
    register_intrinsic(IntrinsicSpec("os_setenv", ["str", "str"], "none", {Effect.WRITES_MEMORY}, may_throw=True))
    register_intrinsic(IntrinsicSpec("os_getcwd", [], "str", {Effect.PURE}, may_throw=False))
    register_intrinsic(IntrinsicSpec("os_chdir", ["str"], "none", {Effect.WRITES_MEMORY}, may_throw=True))
    register_intrinsic(IntrinsicSpec("process_run", ["array"], "map", {Effect.IO, Effect.MAY_THROW}, may_throw=True))
    register_intrinsic(IntrinsicSpec("url_parse", ["str"], "map", {Effect.PURE}, may_throw=True))
    register_intrinsic(IntrinsicSpec("http_request", ["str", "str", "str", "map"], "map", {Effect.IO, Effect.MAY_THROW}, may_throw=True))
    register_intrinsic(
        IntrinsicSpec("syscall_invoke", [ANY_T, "array"], "map", {Effect.IO, Effect.MAY_THROW}, may_throw=False, std_only=False)
    )

    register_intrinsic(IntrinsicSpec("mem_collect", [], "none", {Effect.WRITES_MEMORY}, may_throw=False))
    register_intrinsic(IntrinsicSpec("mem_stats", [], "map", {Effect.READS_MEMORY}, may_throw=False))
    register_intrinsic(IntrinsicSpec("mem_set_deterministic_gc", ["bool"], "none", {Effect.WRITES_MEMORY}, may_throw=False))
    register_intrinsic(IntrinsicSpec("mem_set_gc_stress", ["bool"], "none", {Effect.WRITES_MEMORY}, may_throw=False))

    cudafx = {Effect.READS_MEMORY, Effect.WRITES_MEMORY, Effect.MAY_THROW}
    register_intrinsic(IntrinsicSpec("cuda_is_available", [], "bool", {Effect.READS_MEMORY}, may_throw=False, std_only=False))
    register_intrinsic(IntrinsicSpec("cuda_device_count", [], "int", {Effect.READS_MEMORY}, may_throw=False, std_only=False))
    register_intrinsic(IntrinsicSpec("cuda_set_device", ["int"], "none", {Effect.WRITES_MEMORY}, may_throw=False, std_only=False))
    register_intrinsic(IntrinsicSpec("cuda_alloc", ["str", "int"], "map", cudafx, may_throw=True, std_only=False))
    register_intrinsic(IntrinsicSpec("cuda_free", ["map"], "none", cudafx, may_throw=True, std_only=False))
    register_intrinsic(IntrinsicSpec("cuda_memcpy_h2d", ["map", "array"], "none", cudafx, may_throw=True, std_only=False))
    register_intrinsic(IntrinsicSpec("cuda_memcpy_d2h", ["map"], "array", cudafx, may_throw=True, std_only=False))
    register_intrinsic(IntrinsicSpec("cuda_memcpy_d2d", ["map", "map"], "none", cudafx, may_throw=True, std_only=False))
    register_intrinsic(IntrinsicSpec("cuda_launch", ["map", "map", "map"], "map", cudafx, may_throw=True, std_only=False))
    register_intrinsic(IntrinsicSpec("cuda_sync", [], "none", cudafx, may_throw=False, std_only=False))
    register_intrinsic(IntrinsicSpec("cuda_last_error", [], "str", {Effect.READS_MEMORY}, may_throw=False, std_only=False))

    gpufx = {Effect.READS_MEMORY, Effect.WRITES_MEMORY, Effect.MAY_THROW}
    register_intrinsic(IntrinsicSpec("gpu_backends", [], "array", gpufx, may_throw=False))
    register_intrinsic(IntrinsicSpec("gpu_capabilities", [], "map", gpufx, may_throw=False))
    register_intrinsic(IntrinsicSpec("gpu_dispatch", ["map", "str", "str", "map", "map"], "map", gpufx, may_throw=True))

    register_intrinsic_handler("core_len", _core_len)
    register_intrinsic_handler("core_repr", _core_repr)
    register_intrinsic_handler("core_hash", _core_hash)
    register_intrinsic_handler("core_min", _core_min)
    register_intrinsic_handler("core_max", _core_max)
    register_intrinsic_handler("core_sum", _core_sum)
    register_intrinsic_handler("core_any", _core_any)
    register_intrinsic_handler("core_all", _core_all)
    register_intrinsic_handler("core_sorted", _core_sorted)
    register_intrinsic_handler("core_range", _core_range)
    register_intrinsic_handler("core_enumerate", _core_enumerate)
    register_intrinsic_handler("core_zip", _core_zip)
    register_intrinsic_handler("core_int", _core_int)
    register_intrinsic_handler("core_float", _core_float)
    register_intrinsic_handler("core_bool", _core_bool)
    register_intrinsic_handler("core_str", _core_str)
    register_intrinsic_handler("core_iter", _core_iter)
    register_intrinsic_handler("core_next", _core_next)
    register_intrinsic_handler("abs_int", _abs_int)
    register_intrinsic_handler("abs_float", _abs_float)
    register_intrinsic_handler("min_int", _min_int)
    register_intrinsic_handler("min_float", _min_float)
    register_intrinsic_handler("max_int", _max_int)
    register_intrinsic_handler("max_float", _max_float)
    register_intrinsic_handler("floor_float", _floor_float)
    register_intrinsic_handler("ceil_float", _ceil_float)
    register_intrinsic_handler("trunc_float", _trunc_float)
    register_intrinsic_handler("round_float", _round_float)
    register_intrinsic_handler("sqrt_float", _sqrt_float)
    register_intrinsic_handler("exp_float", _exp_float)
    register_intrinsic_handler("log_float", _log_float)
    register_intrinsic_handler("sin_float", _sin_float)
    register_intrinsic_handler("cos_float", _cos_float)
    register_intrinsic_handler("atan2_float", _atan2_float)
    register_intrinsic_handler("is_nan", _is_nan)
    register_intrinsic_handler("is_inf", _is_inf)
    register_intrinsic_handler("is_finite", _is_finite)
    register_intrinsic_handler("str_upper", _str_upper)
    register_intrinsic_handler("str_lower", _str_lower)
    register_intrinsic_handler("str_strip", _str_strip)
    register_intrinsic_handler("str_lstrip", _str_lstrip)
    register_intrinsic_handler("str_rstrip", _str_rstrip)
    register_intrinsic_handler("str_split", _str_split)
    register_intrinsic_handler("str_join", _str_join)
    register_intrinsic_handler("str_replace", _str_replace)
    register_intrinsic_handler("str_startswith", _str_startswith)
    register_intrinsic_handler("str_endswith", _str_endswith)
    register_intrinsic_handler("str_contains", _str_contains)
    register_intrinsic_handler("str_find", _str_find)
    register_intrinsic_handler("str_rfind", _str_rfind)
    register_intrinsic_handler("str_length", _str_length)
    register_intrinsic_handler("str_ord", lambda args, **_: ord(str(args[0])))
    register_intrinsic_handler("str_chr", lambda args, **_: chr(int(args[0])))
    register_intrinsic_handler("str_char_at", _str_char_at)
    register_intrinsic_handler("str_slice", _str_slice)
    register_intrinsic_handler("str_to_chars", _str_to_chars)
    register_intrinsic_handler("str_concat", _str_concat)
    register_intrinsic_handler("str_repeat", _str_repeat)
    register_intrinsic_handler("str_is_empty", _str_is_empty)
    register_intrinsic_handler("str_trim", _str_trim)
    register_intrinsic_handler("str_pad_left", _str_pad_left)
    register_intrinsic_handler("str_pad_right", _str_pad_right)
    register_intrinsic_handler("str_to_int", _str_to_int)
    register_intrinsic_handler("str_to_float", _str_to_float)
    register_intrinsic_handler("str_format", _str_format)
    register_intrinsic_handler("core_map_keys", _core_map_keys)
    register_intrinsic_handler("core_map_values", _core_map_values)
    register_intrinsic_handler("core_map_has_key", _core_map_has_key)
    register_intrinsic_handler("core_map_delete", _core_map_delete)
    register_intrinsic_handler("core_array_push", _core_array_push)
    register_intrinsic_handler("core_array_slice", _core_array_slice)
    register_intrinsic_handler("core_array_append", _core_array_append)
    register_intrinsic_handler("core_array_pop", _core_array_pop)
    register_intrinsic_handler("io_print", _io_print)
    register_intrinsic_handler("io_read_line", _io_read_line)
    register_intrinsic_handler("fs_exists", _fs_exists)
    register_intrinsic_handler("fs_read_text", _fs_read_text)
    register_intrinsic_handler("fs_write_text", _fs_write_text)
    register_intrinsic_handler("fs_mkdir", _fs_mkdir)
    register_intrinsic_handler("fs_list", _fs_list)
    register_intrinsic_handler("fs_remove", _fs_remove)
    register_intrinsic_handler("path_join", _path_join)
    register_intrinsic_handler("path_basename", _path_basename)
    register_intrinsic_handler("path_dirname", _path_dirname)
    register_intrinsic_handler("path_normalize", _path_normalize)
    register_intrinsic_handler("path_is_abs", _path_is_abs)
    register_intrinsic_handler("time_now_ms", _time_now_ms)
    register_intrinsic_handler("time_monotonic_ms", _time_monotonic_ms)
    register_intrinsic_handler("time_sleep_ms", _time_sleep_ms)
    register_intrinsic_handler("rand_seed", _rand_seed)
    register_intrinsic_handler("rand_int", _rand_int)
    register_intrinsic_handler("rand_float", _rand_float)
    register_intrinsic_handler("json_parse", _json_parse)
    register_intrinsic_handler("json_stringify", _json_stringify)
    register_intrinsic_handler("sys_capabilities", _sys_capabilities)
    register_intrinsic_handler("sys_require", _sys_require)
    register_intrinsic_handler("os_getenv", _os_getenv)
    register_intrinsic_handler("os_setenv", _os_setenv)
    register_intrinsic_handler("os_getcwd", _os_getcwd)
    register_intrinsic_handler("os_chdir", _os_chdir)
    register_intrinsic_handler("process_run", _process_run)
    register_intrinsic_handler("url_parse", _url_parse)
    register_intrinsic_handler("http_request", _http_request)
    register_intrinsic_handler("syscall_invoke", _syscall_invoke)
    register_intrinsic_handler("mem_collect", _mem_collect)
    register_intrinsic_handler("mem_stats", _mem_stats)
    register_intrinsic_handler("mem_set_deterministic_gc", _mem_set_deterministic_gc)
    register_intrinsic_handler("mem_set_gc_stress", _mem_set_gc_stress)
    register_intrinsic_handler("cuda_is_available", _cuda_is_available)
    register_intrinsic_handler("cuda_device_count", _cuda_device_count)
    register_intrinsic_handler("cuda_set_device", _cuda_set_device)
    register_intrinsic_handler("cuda_alloc", _cuda_alloc)
    register_intrinsic_handler("cuda_free", _cuda_free)
    register_intrinsic_handler("cuda_memcpy_h2d", _cuda_memcpy_h2d)
    register_intrinsic_handler("cuda_memcpy_d2h", _cuda_memcpy_d2h)
    register_intrinsic_handler("cuda_memcpy_d2d", _cuda_memcpy_d2d)
    register_intrinsic_handler("cuda_launch", _cuda_launch)
    register_intrinsic_handler("cuda_sync", _cuda_sync)
    register_intrinsic_handler("cuda_last_error", _cuda_last_error)
    register_intrinsic_handler("gpu_backends", _gpu_backends)
    register_intrinsic_handler("gpu_capabilities", _gpu_capabilities)
    register_intrinsic_handler("gpu_dispatch", _gpu_dispatch)
    # ptr / C FFI intrinsics
    register_intrinsic(IntrinsicSpec("ptr_null", [], "int", {Effect.PURE}, may_throw=False, std_only=False))
    register_intrinsic(IntrinsicSpec("ptr_is_null", [ANY_T], "bool", {Effect.PURE}, may_throw=False, std_only=False))
    register_intrinsic(IntrinsicSpec("c_str", ["str"], "str", {Effect.ALLOCATES}, may_throw=False, std_only=False))
    register_intrinsic(IntrinsicSpec("ptr_to_str", [ANY_T], "str", {Effect.READS_MEMORY}, may_throw=False, std_only=False))
    register_intrinsic(IntrinsicSpec("ptr_add", [ANY_T, "int"], "int", {Effect.PURE}, may_throw=False, std_only=False))
    register_intrinsic_handler("ptr_null", _ptr_null)
    register_intrinsic_handler("ptr_is_null", _ptr_is_null)
    register_intrinsic_handler("c_str", _c_str)
    register_intrinsic_handler("ptr_to_str", _ptr_to_str)
    register_intrinsic_handler("ptr_add", _ptr_add)


_register_defaults()
