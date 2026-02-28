from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.compiler import parse_program
from manv.runner import run_file
from manv.semantics import SemanticAnalyzer


def test_getter_and_setter_route_attribute_reads_and_writes(tmp_path: Path) -> None:
    source = (
        "class User:\n"
        "    fn __init__(self, name: str) -> none:\n"
        "        self._name = name\n"
        "\n"
        "    @getter\n"
        "    fn name(self) -> str:\n"
        '        "Public display name."\n'
        "        return self._name\n"
        "\n"
        "    @setter\n"
        "    fn name(self, value: str) -> none:\n"
        "        self._name = value\n"
        "\n"
        "fn main() -> int:\n"
        "    let user = User(\"Ada\")\n"
        "    print(user.name)\n"
        "    user.name = \"Grace\"\n"
        "    print(user.name)\n"
        "    return 0\n"
    )
    path = tmp_path / "main.mv"
    path.write_text(source, encoding="utf-8")

    out = StringIO()
    assert run_file(path, mode="interpreter", stdout=out) == 0
    assert out.getvalue().strip().splitlines() == ["Ada", "Grace"]

    compiled_out = StringIO()
    assert run_file(path, mode="compiled", stdout=compiled_out) == 0
    assert compiled_out.getvalue().strip().splitlines() == ["Ada", "Grace"]


def test_accessor_signatures_are_validated_deterministically() -> None:
    source = (
        "class User:\n"
        "    @getter\n"
        "    fn name(value: str) -> str:\n"
        "        return value\n"
        "\n"
        "    @setter\n"
        "    fn age(self) -> int:\n"
        "        return 0\n"
    )
    program = parse_program(source, "accessor_signature.mv")
    analyzer = SemanticAnalyzer("accessor_signature.mv")
    result = analyzer.analyze(program)

    codes = {diag.code for diag in result.diagnostics}
    assert "E2060" in codes
    assert "E2062" in codes
    assert "E2063" in codes


def test_accessor_pairs_can_share_one_property_name() -> None:
    source = (
        "class User:\n"
        "    @getter\n"
        "    fn name(self) -> str:\n"
        "        return \"Ada\"\n"
        "\n"
        "    @setter\n"
        "    fn name(self, value: str) -> none:\n"
        "        return\n"
    )
    program = parse_program(source, "accessor_pair.mv")
    analyzer = SemanticAnalyzer("accessor_pair.mv")
    result = analyzer.analyze(program)

    assert not [diag for diag in result.diagnostics if diag.severity == "error"]


def test_duplicate_getter_for_same_property_is_rejected() -> None:
    source = (
        "class User:\n"
        "    @getter\n"
        "    fn name(self) -> str:\n"
        "        return \"Ada\"\n"
        "\n"
        "    @getter(name=\"name\")\n"
        "    fn public_name(self) -> str:\n"
        "        return \"Grace\"\n"
    )
    program = parse_program(source, "duplicate_getter.mv")
    analyzer = SemanticAnalyzer("duplicate_getter.mv")
    result = analyzer.analyze(program)

    assert any(diag.code == "E2046" for diag in result.diagnostics)


def test_accessor_cannot_be_static_method() -> None:
    source = (
        "class User:\n"
        "    @static_method\n"
        "    @getter\n"
        "    fn name(self) -> str:\n"
        "        return \"Ada\"\n"
    )
    program = parse_program(source, "accessor_static.mv")
    analyzer = SemanticAnalyzer("accessor_static.mv")
    result = analyzer.analyze(program)

    assert any(diag.code == "E2058" for diag in result.diagnostics)
