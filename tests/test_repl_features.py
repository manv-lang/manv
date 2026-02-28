from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.repl import ReplSession, _style_line


def test_repl_completion_words_include_dynamic_symbols() -> None:
    out = StringIO()
    session = ReplSession(out)
    session.execute_source(
        "fn add(a: int, b: int) -> int:\n"
        "    return a + b\n"
    )

    words = set(session.completion_words())
    assert "add" in words
    assert "print" in words
    assert "len" in words
    assert "help" in words
    assert "__intrin" in words


def test_repl_registers_type_declarations_and_exposes_them_to_completion() -> None:
    out = StringIO()
    session = ReplSession(out)
    session.execute_source(
        "class Math:\n"
        "    let pi: f32 = 3.14\n"
        "    fn sqrt(self, x: f32) -> f32:\n"
        "        return x\n"
    )

    rendered = out.getvalue()
    assert "registered type Math" in rendered
    assert "Math" in set(session.completion_words())


def test_repl_executes_accessor_backed_properties() -> None:
    out = StringIO()
    session = ReplSession(out)
    session.execute_source(
        "class User:\n"
        "    fn __init__(self, name: str) -> none:\n"
        "        self._name = name\n"
        "    @getter\n"
        "    fn name(self) -> str:\n"
        "        return self._name\n"
        "    @setter\n"
        "    fn name(self, value: str) -> none:\n"
        "        self._name = value\n"
    )
    session.execute_source(
        "let user = User(\"Ada\")\n"
        "print(user.name)\n"
        "user.name = \"Grace\"\n"
        "print(user.name)\n"
    )

    assert out.getvalue().strip().splitlines()[-2:] == ["Ada", "Grace"]


def test_style_line_highlights_core_token_kinds() -> None:
    fragments = _style_line('if true: print(__intrin.core_len([1])) # note', {"print", "core_len", "if"})
    styled = {(style, text) for style, text in fragments if text.strip()}

    assert ("class:keyword", "if") in styled
    assert ("class:builtin", "print") in styled
    assert ("class:intrinsic", "__intrin.core_len") in styled
    assert ("class:number", "1") in styled
    assert any(style == "class:comment" for style, text in styled if text.startswith("#"))


def test_style_line_highlights_nested_cuda_intrinsic_chain() -> None:
    fragments = _style_line("__intrin.cuda.device_count()", {"device_count"})
    styled = {(style, text) for style, text in fragments if text.strip()}

    assert ("class:intrinsic", "__intrin.cuda.device_count") in styled
