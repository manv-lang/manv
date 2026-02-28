from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.compiler import parse_program
from manv.diagnostics import ManvError
from manv.hlir_lowering import lower_ast_to_hlir
from manv.runner import run_file
from manv.semantics import SemanticAnalyzer


def test_static_method_decorator_enables_type_calls_in_interpreter_and_compiled(tmp_path: Path) -> None:
    source = (
        "class Math:\n"
        "    @static_method\n"
        "    fn abs(x: int) -> int:\n"
        "        if x < 0:\n"
        "            return 0 - x\n"
        "        return x\n"
        "\n"
        "fn main() -> int:\n"
        "    print(Math.abs(-7))\n"
        "    return 0\n"
    )
    path = tmp_path / "main.mv"
    path.write_text(source, encoding="utf-8")

    assert run_file(path, mode="interpreter") == 0
    assert run_file(path, mode="compiled") == 0


def test_static_method_decorator_is_rejected_on_top_level_functions() -> None:
    source = (
        "@static_method\n"
        "fn abs(x: int) -> int:\n"
        "    return x\n"
    )
    program = parse_program(source, "static_method_top_level.mv")
    analyzer = SemanticAnalyzer("static_method_top_level.mv")
    result = analyzer.analyze(program)

    assert any(diag.code == "E2040" for diag in result.diagnostics)


def test_static_type_call_lowers_to_direct_qualified_call() -> None:
    source = (
        "class Math:\n"
        "    @static_method\n"
        "    fn abs(x: int) -> int:\n"
        "        return x\n"
        "\n"
        "fn main() -> int:\n"
        "    return Math.abs(3)\n"
    )
    program = parse_program(source, "static_method_lowering.mv")
    module = lower_ast_to_hlir(program, "static_method_lowering.mv")
    main_fn = next(fn for fn in module.functions if fn.name == "main")

    call_instrs = [instr for block in main_fn.blocks for instr in block.instructions if instr.op == "call"]
    method_instrs = [instr for block in main_fn.blocks for instr in block.instructions if instr.op == "method_call"]

    assert any(instr.attrs.get("callee") == "Math.abs" for instr in call_instrs)
    assert not method_instrs


def test_instance_method_call_on_type_requires_static_method(tmp_path: Path) -> None:
    source = (
        "class Math:\n"
        "    fn abs(self, x: int) -> int:\n"
        "        return x\n"
        "\n"
        "fn main() -> int:\n"
        "    return Math.abs(3)\n"
    )
    path = tmp_path / "main.mv"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ManvError) as err:
        run_file(path, mode="interpreter")

    assert "requires an object" in err.value.render()
