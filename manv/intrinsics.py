from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Callable
from urllib import parse as urlparse
from urllib import request as urlrequest

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
}


_INTRINSICS: dict[str, IntrinsicSpec] = {}
_HANDLERS: dict[str, Callable[..., Any]] = {}


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

    if not isinstance(expr, ast.AttributeExpr):
        return None
    if not isinstance(expr.value, ast.IdentifierExpr):
        return None
    if expr.value.name != "__intrin":
        return None
    return expr.attr


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


def std_namespace_attr(base: Any, attr: str) -> Any | None:
    if isinstance(base, IntrinsicNamespace):
        return IntrinsicCallable(attr)
    if isinstance(base, StdNamespace):
        if attr in {"core", "io", "fs", "path", "time", "rand", "json", "memory", "gpu", "sys", "os", "process", "url", "http"}:
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
    if expected in {"int", "usize", "u8"} and got == "int":
        return True
    if expected == "float" and got in {"float", "int"}:
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


def _core_str(args: list[Any], **_: Any) -> str:
    _ensure_arity("core_str", args, min_n=1, max_n=1)
    return str(args[0])


def _core_iter(args: list[Any], **_: Any) -> Any:
    _ensure_arity("core_iter", args, min_n=1, max_n=1)
    return iter(args[0])


def _core_next(args: list[Any], **_: Any) -> Any:
    _ensure_arity("core_next", args, min_n=1, max_n=2)
    it = args[0]
    if len(args) == 1:
        return next(it)
    return next(it, args[1])


def _io_print(args: list[Any], *, stdout_write: Callable[[str], None] | None = None, **_: Any) -> None:
    writer = stdout_write or (lambda s: None)
    parts = args
    if len(args) == 1 and isinstance(args[0], list):
        parts = args[0]
    writer(" ".join(str(x) for x in parts) + "\n")
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
        if isinstance(target, int):
            if hasattr(os, "syscall"):
                result = os.syscall(int(target), *argv)
                return {"ok": True, "result": result, "platform": platform_name}
            raise OSError("numeric syscall is not available on this platform")

        if not isinstance(target, str):
            raise TypeError("syscall target must be int or str")

        name = target.strip()
        if hasattr(os, "syscall"):
            # Named calls can be passed as numeric strings.
            try:
                num = int(name)
                result = os.syscall(num, *argv)
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


def _gpu_backends(args: list[Any], **_: Any) -> list[str]:
    _ensure_arity("gpu_backends", args, min_n=0, max_n=0)
    return list_backends()


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
    register_intrinsic_handler("gpu_backends", _gpu_backends)
    register_intrinsic_handler("gpu_capabilities", _gpu_capabilities)
    register_intrinsic_handler("gpu_dispatch", _gpu_dispatch)


_register_defaults()
