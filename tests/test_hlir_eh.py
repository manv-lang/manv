from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.compiler import analyze_program, parse_program
from manv.hlir_interpreter import HLIRInterpreter
from manv.hlir_lowering import lower_ast_to_hlir


def _lower(source: str):
    program = parse_program(source, "<mem>")
    analyze_program(program, "<mem>")
    return lower_ast_to_hlir(program, "<mem>")


def test_hlir_try_lowering_uses_invoke_unwind_blocks() -> None:
    hlir = _lower(
        "fn main() -> int:\n"
        "    try:\n"
        "        int(\"x\")\n"
        "    except ValueError:\n"
        "        print(7)\n"
        "    finally:\n"
        "        print(9)\n"
        "    return 0\n"
    )
    fn = next(f for f in hlir.functions if f.name == "main")
    instr_ops = [instr.op for block in fn.blocks for instr in block.instructions]
    term_ops = [block.terminator.op for block in fn.blocks if block.terminator is not None]

    assert "try_region" not in instr_ops
    assert "invoke" in term_ops
    assert "load_exception" in instr_ops
    assert "exc_match" in instr_ops
    assert "finally_enter" in instr_ops
    assert "finally_exit" in instr_ops


def test_hlir_interpreter_try_except_finally_behavior() -> None:
    hlir = _lower(
        "fn main() -> int:\n"
        "    try:\n"
        "        int(\"x\")\n"
        "    except ValueError:\n"
        "        print(7)\n"
        "    finally:\n"
        "        print(9)\n"
        "    return 0\n"
    )
    out = StringIO()
    result = HLIRInterpreter(stdout=out).run_module(hlir, entry="main")
    assert result.value == 0
    assert out.getvalue().strip().splitlines() == ["7", "9"]

