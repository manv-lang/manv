from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.diagnostics import ManvError
from manv.intrinsics import all_intrinsics, invoke_intrinsic
from manv.runner import run_file


MIN_EXPECTED_INTRINSICS = {
    "core_len",
    "io_print",
    "io_read_line",
    "fs_exists",
    "fs_read_text",
    "fs_write_text",
    "fs_mkdir",
    "fs_list",
    "fs_remove",
    "path_join",
    "path_basename",
    "path_dirname",
    "path_normalize",
    "path_is_abs",
    "time_now_ms",
    "time_monotonic_ms",
    "time_sleep_ms",
    "rand_seed",
    "rand_int",
    "rand_float",
    "json_parse",
    "json_stringify",
    "mem_collect",
    "mem_stats",
    "mem_set_deterministic_gc",
    "mem_set_gc_stress",
    "gpu_backends",
    "gpu_capabilities",
    "gpu_dispatch",
}


def _write_project(tmp_path: Path, source: str, *, is_std: bool) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if is_std:
        (tmp_path / "project.toml").write_text('[project]\nname = "std"\nversion = "0.1.0"\n', encoding="utf-8")
    src = tmp_path / "main.mv"
    src.write_text(source, encoding="utf-8")
    return src


def test_intrinsic_registry_matches_v1_core_set() -> None:
    names = {spec.name for spec in all_intrinsics()}
    assert MIN_EXPECTED_INTRINSICS.issubset(names)


def test_intrinsic_semantic_validation_and_std_boundary(tmp_path: Path) -> None:
    outside_std = _write_project(
        tmp_path / "user_pkg",
        (
            "fn main() -> int:\n"
            "    __intrin.core_len([1, 2])\n"
            "    return 0\n"
        ),
        is_std=False,
    )
    with pytest.raises(ManvError) as err1:
        run_file(outside_std, mode="interpreter")
    assert "E2022" in err1.value.render()

    std_root = tmp_path / "std_pkg"
    std_root.mkdir(parents=True, exist_ok=True)

    unknown = _write_project(
        std_root / "unknown",
        (
            "fn main() -> int:\n"
            "    __intrin.no_such()\n"
            "    return 0\n"
        ),
        is_std=True,
    )
    with pytest.raises(ManvError) as err2:
        run_file(unknown, mode="interpreter")
    assert "E2020" in err2.value.render()

    bad_sig = _write_project(
        std_root / "bad_sig",
        (
            "fn main() -> int:\n"
            "    __intrin.rand_int(1)\n"
            "    return 0\n"
        ),
        is_std=True,
    )
    with pytest.raises(ManvError) as err3:
        run_file(bad_sig, mode="interpreter")
    assert "E2021" in err3.value.render()

    bad_ns = _write_project(
        std_root / "bad_ns",
        (
            "fn main() -> int:\n"
            "    __intrin\n"
            "    return 0\n"
        ),
        is_std=True,
    )
    with pytest.raises(ManvError) as err4:
        run_file(bad_ns, mode="interpreter")
    assert "E2023" in err4.value.render()


def test_intrinsic_runtime_happy_path_and_mode_equivalence(tmp_path: Path) -> None:
    data_file = (tmp_path / "std_run" / "data.txt")
    data_file.parent.mkdir(parents=True, exist_ok=True)
    p = data_file.as_posix()

    src = _write_project(
        data_file.parent,
        (
            "fn main() -> int:\n"
            f"    str p = \"{p}\"\n"
            "    __intrin.fs_write_text(p, \"hello\")\n"
            "    print(len([1, 2, 3]))\n"
            "    __intrin.io_print([\"intrin\", 7])\n"
            "    print(__intrin.fs_exists(p))\n"
            "    print(__intrin.fs_read_text(p))\n"
            "    print(__intrin.path_basename(p))\n"
            "    return 0\n"
        ),
        is_std=True,
    )

    out_i = StringIO()
    out_c = StringIO()
    code_i = run_file(src, stdout=out_i, mode="interpreter")
    code_c = run_file(src, stdout=out_c, mode="compiled")

    assert code_i == 0
    assert code_c == 0
    assert out_i.getvalue() == out_c.getvalue()

    lines = out_i.getvalue().strip().splitlines()
    assert lines[0] == "3"
    assert lines[1] == "intrin 7"
    assert lines[2] in {"True", "true"}
    assert lines[3] == "hello"
    assert lines[4] == "data.txt"


def test_intrinsic_handler_runtime_matrix(tmp_path: Path) -> None:
    out = StringIO()
    gc_state = {
        "deterministic_gc": False,
        "gc_stress": False,
    }

    def _stats() -> dict[str, object]:
        return {
            "objects": 0,
            "roots": 0,
            "deterministic_gc": gc_state["deterministic_gc"],
            "gc_stress": gc_state["gc_stress"],
        }

    hooks = {
        "collect": lambda _reason: None,
        "stats": _stats,
        "set_deterministic_gc": lambda flag: gc_state.__setitem__("deterministic_gc", bool(flag)),
        "set_gc_stress": lambda flag: gc_state.__setitem__("gc_stress", bool(flag)),
    }

    f = tmp_path / "f.txt"
    d = tmp_path / "d"

    assert invoke_intrinsic("core_len", [[1, 2, 3]]) == 3
    assert invoke_intrinsic("io_read_line", [], stdin_readline=lambda: "abc\n") == "abc"
    invoke_intrinsic("io_print", [["a", 1]], stdout_write=out.write)
    assert out.getvalue().strip() == "a 1"

    invoke_intrinsic("fs_write_text", [str(f), "hello"])
    assert invoke_intrinsic("fs_exists", [str(f)]) is True
    assert invoke_intrinsic("fs_read_text", [str(f)]) == "hello"
    invoke_intrinsic("fs_mkdir", [str(d), True])
    listed = invoke_intrinsic("fs_list", [str(tmp_path)])
    assert "f.txt" in listed and "d" in listed
    invoke_intrinsic("fs_remove", [str(f)])
    assert invoke_intrinsic("fs_exists", [str(f)]) is False

    joined = invoke_intrinsic("path_join", [[str(tmp_path), "d", "x"]])
    assert isinstance(joined, str)
    assert invoke_intrinsic("path_basename", [joined]) == "x"
    assert invoke_intrinsic("path_dirname", [joined]).endswith("d")
    assert isinstance(invoke_intrinsic("path_normalize", [joined]), str)
    assert isinstance(invoke_intrinsic("path_is_abs", [joined]), bool)

    now_ms = invoke_intrinsic("time_now_ms", [])
    mono_ms = invoke_intrinsic("time_monotonic_ms", [])
    assert isinstance(now_ms, int) and now_ms > 0
    assert isinstance(mono_ms, int) and mono_ms > 0
    assert invoke_intrinsic("time_sleep_ms", [0]) is None

    assert invoke_intrinsic("rand_seed", [42]) is None
    ri = invoke_intrinsic("rand_int", [1, 3])
    rf = invoke_intrinsic("rand_float", [])
    assert 1 <= ri <= 3
    assert 0.0 <= rf < 1.0

    parsed = invoke_intrinsic("json_parse", ['{"a":1}'])
    assert parsed["a"] == 1
    rendered = invoke_intrinsic("json_stringify", [{"z": 2, "a": 1}])
    assert rendered == '{"a": 1, "z": 2}'

    assert invoke_intrinsic("mem_collect", [], gc_hooks=hooks) is None
    assert invoke_intrinsic("mem_stats", [], gc_hooks=hooks)["deterministic_gc"] is False
    invoke_intrinsic("mem_set_deterministic_gc", [True], gc_hooks=hooks)
    invoke_intrinsic("mem_set_gc_stress", [True], gc_hooks=hooks)
    stats = invoke_intrinsic("mem_stats", [], gc_hooks=hooks)
    assert stats["deterministic_gc"] is True
    assert stats["gc_stress"] is True

    backends = invoke_intrinsic("gpu_backends", [])
    caps = invoke_intrinsic("gpu_capabilities", [])
    assert isinstance(backends, list)
    assert isinstance(caps, dict)
    disp = invoke_intrinsic(
        "gpu_dispatch",
        [{"version": "0.1", "source": "x", "kernels": []}, "cpu_ref", "generic", {}, {}],
    )
    assert disp["selected_backend"] == "cpu_ref"
    assert isinstance(disp["trace"], dict)
