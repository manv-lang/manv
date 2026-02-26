from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import platform
import subprocess

import pytest

from manv.abi import classify_sysv_aggregate, lower_function_abi
from manv.compiler import compile_pipeline_full, compile_target
from manv.native_toolchain import detect_toolchain, host_default_target
from manv.runner import run_target
from manv.targets import get_target


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "e2e" / "fixtures"


def test_interpreter_vs_compiled_equivalence() -> None:
    project = FIXTURES / "run_core_features"
    out_interp = StringIO()
    out_comp = StringIO()

    interp_code = run_target(project, stdout=out_interp, mode="interpreter", target_name="x86_64-sysv")
    comp_code = run_target(project, stdout=out_comp, mode="compiled", target_name="x86_64-sysv")

    assert interp_code == comp_code
    assert out_interp.getvalue() == out_comp.getvalue()


def test_abi_codegen_patterns_by_target(tmp_path: Path) -> None:
    source_path = FIXTURES / "compile_ok" / "src" / "main.mv"

    out_sysv = tmp_path / "sysv"
    out_win = tmp_path / "win"
    out_arm = tmp_path / "arm"

    compile_target(source_path, out_sysv, emit=["asm", "abi"], target_name="x86_64-sysv")
    compile_target(source_path, out_win, emit=["asm", "abi"], target_name="x86_64-win64")
    compile_target(source_path, out_arm, emit=["asm", "abi"], target_name="aarch64-aapcs64")

    asm_sysv = (out_sysv / "main.x86_64-sysv.s").read_text(encoding="utf-8")
    asm_win = (out_win / "main.x86_64-win64.s").read_text(encoding="utf-8")
    asm_arm = (out_arm / "main.aarch64-aapcs64.s").read_text(encoding="utf-8")

    assert ".cfi_startproc" in asm_sysv
    assert "push rbp" in asm_sysv
    assert ".cfi_endproc" in asm_sysv

    assert ".seh_proc" in asm_win
    assert ".seh_stackalloc" in asm_win
    assert ".seh_endprologue" in asm_win

    assert "stp x29, x30" in asm_arm
    assert ".cfi_startproc" in asm_arm


def test_graph_capture_kernel_mock_pipeline() -> None:
    source_path = FIXTURES / "run_core_features" / "src" / "main.mv"
    source = source_path.read_text(encoding="utf-8")

    artifacts = compile_pipeline_full(
        source,
        str(source_path),
        optimize=True,
        target_name="x86_64-sysv",
        capture_graph=True,
    )

    assert artifacts["capture"] is not None
    assert "nodes" in artifacts["capture"]
    assert "kernels" in artifacts["kernel_exec"]


def test_host_stub_artifacts(tmp_path: Path) -> None:
    source_path = FIXTURES / "compile_cuda" / "src" / "main.mv"
    written = compile_target(
        source_path,
        tmp_path,
        emit=["host_stub", "host_stub_abi", "kernel"],
        backend="cuda-ptx",
        target_name="x86_64-sysv",
    )

    assert "host_stub" in written
    assert "host_stub_abi" in written

    stub_text = written["host_stub"].read_text(encoding="utf-8")
    stub_abi = json.loads(written["host_stub_abi"].read_text(encoding="utf-8"))

    assert "launch_" in stub_text
    assert isinstance(stub_abi, dict)


def test_sysv_aggregate_eightbyte_classification() -> None:
    reg_cls = classify_sysv_aggregate("tuple[int,float]")
    assert reg_cls["mode"] == "register"
    assert reg_cls["classes"] == ["INTEGER", "SSE"]

    mem_cls = classify_sysv_aggregate("array[int;3]")
    assert mem_cls["mode"] == "memory"


def test_sysv_varargs_accounting() -> None:
    target = get_target("x86_64-sysv")
    abi = lower_function_abi(
        "vfn",
        ["int", "float", "int"],
        "int",
        target,
        is_varargs=True,
        fixed_param_count=1,
    )
    assert abi.is_varargs
    assert abi.varargs_fp_reg_count == 1
    assert abi.varargs_gp_reg_count == 1


def test_native_machine_code_artifacts(tmp_path: Path) -> None:
    if detect_toolchain() is None:
        pytest.skip("native toolchain not available")

    host_target = host_default_target()
    source_path = tmp_path / "main.mv"
    source_path.write_text(
        "fn main() -> int:\n"
        "    return 0\n",
        encoding="utf-8",
    )

    written = compile_target(
        source_path,
        tmp_path,
        emit=["asm", "native_obj", "native_exe"],
        target_name=host_target,
    )

    assert "native_obj" in written
    assert "native_exe" in written
    assert written["native_obj"].exists()
    assert written["native_exe"].exists()

    if platform.system().lower() == "windows":
        proc = subprocess.run([str(written["native_exe"])], capture_output=True, text=True, check=False)
    else:
        proc = subprocess.run([str(written["native_exe"])], capture_output=True, text=True, check=False)
    assert proc.returncode == 0
