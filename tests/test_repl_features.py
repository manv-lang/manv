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
    assert "__intrin" in words


def test_style_line_highlights_core_token_kinds() -> None:
    fragments = _style_line('if true: print(__intrin.core_len([1])) # note', {"print", "core_len", "if"})
    styled = {(style, text) for style, text in fragments if text.strip()}

    assert ("class:keyword", "if") in styled
    assert ("class:builtin", "print") in styled
    assert ("class:intrinsic", "__intrin") in styled
    assert ("class:number", "1") in styled
    assert any(style == "class:comment" for style, text in styled if text.startswith("#"))
