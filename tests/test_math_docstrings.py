from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv import ast
from manv.compiler import parse_program
from manv.diagnostics import ManvError
from manv.hlir_lowering import lower_ast_to_hlir
from manv.runner import run_file


def test_parser_extracts_docstrings_and_type_attributes() -> None:
    source = (
        '"""Module docs\\nline two"""\n'
        "\n"
        "class Math:\n"
        '    "Scalar math utilities."\n'
        "    let pi: f32 = 3.0\n"
        "    fn sqrt(x: f32) -> f32:\n"
        '        "Returns sqrt(x)."\n'
        "        return x\n"
        "\n"
        "fn main() -> int:\n"
        "    let x = 1\n"
        '    "not a docstring"\n'
        "    return x\n"
    )

    program = parse_program(source, "docstrings.mv")

    assert program.docstring == "Module docs\\nline two"

    math_decl = next(decl for decl in program.declarations if isinstance(decl, ast.TypeDecl))
    assert math_decl.name == "Math"
    assert math_decl.docstring == "Scalar math utilities."
    assert len(math_decl.attrs) == 1
    assert math_decl.attrs[0].name == "pi"
    assert math_decl.attrs[0].type_name == "f32"

    sqrt_decl = math_decl.methods[0]
    assert sqrt_decl.docstring == "Returns sqrt(x)."
    assert len(sqrt_decl.body) == 1

    main_decl = next(
        decl for decl in program.declarations if isinstance(decl, ast.FnDecl) and decl.name == "main"
    )
    assert main_decl.docstring is None
    assert len(main_decl.body) == 3
    assert isinstance(main_decl.body[1], ast.ExprStmt)
    assert isinstance(main_decl.body[1].expr, ast.LiteralExpr)
    assert main_decl.body[1].expr.value == "not a docstring"


def test_docstrings_are_metadata_only_and_do_not_reach_hlir() -> None:
    source = (
        "fn main() -> int:\n"
        '    """Function docs should not become executable IR."""\n'
        "    return 0\n"
    )

    program = parse_program(source, "hlir_docstrings.mv")
    module = lower_ast_to_hlir(program, "hlir_docstrings.mv")
    payload = json.dumps(module.to_dict(), sort_keys=True)

    assert module.attrs["functions"]["main"]["docstring"] == "Function docs should not become executable IR."
    main_payload = json.dumps(next(fn.to_dict() for fn in module.functions if fn.name == "main"), sort_keys=True)
    assert "Function docs should not become executable IR." not in main_payload
    main_decl = next(decl for decl in program.declarations if isinstance(decl, ast.FnDecl))
    assert main_decl.docstring == "Function docs should not become executable IR."
    assert len(main_decl.body) == 1


def test_math_stdlib_runtime_matches_in_interpreter_and_compiled(tmp_path: Path) -> None:
    source = (
        "from math import approx_eq, tau, pi, is_nan, sqrt, is_inf, log\n"
        "from math import TensorMath\n"
        "fn main() -> int:\n"
        "    print(approx_eq(tau, pi * 2.0, 0.000001, 0.000001))\n"
        "    print(is_nan(sqrt(-1.0)))\n"
        "    print(is_inf(log(0.0)))\n"
        "    print(is_nan(log(-1.0)))\n"
        "    print(TensorMath([1.0, -2.0, 3.0]).relu().sum())\n"
        "    print(TensorMath([1.0, 2.0, 3.0]).mean())\n"
        "    return 0\n"
    )
    path = tmp_path / "main.mv"
    path.write_text(source, encoding="utf-8")

    out_interpreter = StringIO()
    out_compiled = StringIO()

    exit_interpreter = run_file(path, stdout=out_interpreter, mode="interpreter")
    exit_compiled = run_file(path, stdout=out_compiled, mode="compiled")

    assert exit_interpreter == 0
    assert exit_compiled == 0
    assert out_interpreter.getvalue() == out_compiled.getvalue()

    lines = out_interpreter.getvalue().strip().splitlines()
    assert lines[:4] == ["True", "True", "True", "True"]
    assert lines[4:] == ["4.0", "2.0"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from math import abs\n"
            "fn main() -> int:\n"
            "    abs(-2147483648)\n"
            "    return 0\n",
            "OverflowError",
        ),
        (
            "from math import TensorMath\n"
            "fn main() -> int:\n"
            "    TensorMath([]).sum()\n"
            "    return 0\n",
            "ValueError",
        ),
    ],
)
def test_math_edge_errors_are_deterministic(tmp_path: Path, source: str, expected: str) -> None:
    path = tmp_path / "main.mv"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ManvError) as err_interpreter:
        run_file(path, mode="interpreter")
    with pytest.raises(ManvError) as err_compiled:
        run_file(path, mode="compiled")

    assert expected in err_interpreter.value.render()
    assert expected in err_compiled.value.render()
