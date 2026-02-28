from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.compiler import analyze_program, parse_program
from manv.hlir_lowering import lower_ast_to_hlir
from manv.llvm_codegen import LlvmLoweringError, emit_llvm_module
from manv.targets import get_target


def test_llvm_lowering_supports_classes_syscall_and_required_gpu_stub() -> None:
    source = (
        "@gpu(required=true)\n"
        "fn add(a: f32, b: f32) -> f32:\n"
        "    return a + b\n"
        "\n"
        "class User:\n"
        "    fn __init__(self, name: str) -> none:\n"
        "        self.name = name\n"
        "\n"
        "    fn repr(self) -> none:\n"
        "        print(self.name)\n"
        "\n"
        "fn main() -> none:\n"
        "    let user = User(\"x\")\n"
        "    user.repr()\n"
        "    let r = syscall(\"getpid\")\n"
        "    print(r[\"ok\"])\n"
        "    let out = add(1.0, 2.0)\n"
        "    print(out)\n"
    )

    program = parse_program(source, "llvm_supported.mv")
    analyze_program(program, "llvm_supported.mv")
    module = lower_ast_to_hlir(program, "llvm_supported.mv")

    rendered = emit_llvm_module(module, get_target("x86_64-sysv"), source_name="llvm_supported.mv")

    assert "%manv_obj_User = type { ptr }" in rendered
    assert "call void @manv_User_repr(ptr" in rendered
    assert "call ptr @manv_rt_syscall_invoke_cstr(ptr" in rendered
    assert "call i1 @manv_rt_syscall_result_ok(ptr" in rendered
    assert "call float @manv_rt_gpu_required_f32(ptr" in rendered


def test_llvm_lowering_supports_arrays_maps_and_core_len() -> None:
    source = (
        "fn main() -> none:\n"
        "    array xs[3] = [1, 2]\n"
        "    xs[2] = 9\n"
        "    let m = {\"ok\": 1}\n"
        "    m[\"next\"] = xs[2]\n"
        "    print(__intrin.core_len(xs))\n"
        "    print(m[\"ok\"])\n"
        "    print((__intrin.core_len(xs) > 0) and true)\n"
    )

    path = "std/src/llvm_containers.mv"
    program = parse_program(source, path)
    analyze_program(program, path)
    module = lower_ast_to_hlir(program, path)

    rendered = emit_llvm_module(module, get_target("x86_64-sysv"), source_name=path)

    assert "call ptr @manv_rt_array_clone_sized" in rendered
    assert "call void @manv_rt_array_set_i64" in rendered
    assert "call i64 @manv_rt_array_len(ptr" in rendered
    assert "call ptr @manv_rt_map_new" in rendered
    assert "call void @manv_rt_map_set_ptr_i64" in rendered
    assert "call i64 @manv_rt_map_get_ptr_i64" in rendered


def test_llvm_lowering_reports_remaining_unsupported_surface() -> None:
    source = (
        "fn main() -> none:\n"
        "    let xs = [1, 2.0]\n"
        "    print(xs)\n"
    )

    program = parse_program(source, "llvm_unsupported.mv")
    analyze_program(program, "llvm_unsupported.mv")
    module = lower_ast_to_hlir(program, "llvm_unsupported.mv")

    with pytest.raises(LlvmLoweringError) as err:
        emit_llvm_module(module, get_target("x86_64-sysv"), source_name="llvm_unsupported.mv")

    rendered = str(err.value)
    assert "unsupported HLIR features for LLVM lowering" in rendered
    assert "main: heterogeneous array literal construction" in rendered
