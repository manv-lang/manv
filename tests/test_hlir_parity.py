"""The HLIR interpreter (the compiler's reference execution) must agree with the AST interpreter
on behaviour that was once different: short-circuit evaluation and how values are printed."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from manv.compiler import analyze_program, parse_program
from manv.hlir_interpreter import HLIRInterpreter
from manv.hlir_lowering import lower_ast_to_hlir

CONFORMANCE = Path(__file__).parent / "conformance"
PROGRAMS = ["basics/short_circuit", "basics/short_circuit_nested", "basics/bool_and_none_printing", "basics/print_multiple_arguments", "control_flow/elif_chain", "control_flow/conditional_expression"]


@pytest.mark.parametrize("name", PROGRAMS)
def test_hlir_interpreter_matches_the_expected_output(name: str) -> None:
    source = (CONFORMANCE / f"{name}.mv").read_text(encoding="utf-8")
    expected = (CONFORMANCE / f"{name}.expected").read_text(encoding="utf-8").replace("\r\n", "\n")

    program = parse_program(source, f"{name}.mv")
    analyze_program(program, f"{name}.mv")
    out = StringIO()
    HLIRInterpreter(stdout=out).run_module(lower_ast_to_hlir(program, f"{name}.mv"), entry="main")

    assert out.getvalue() == expected
