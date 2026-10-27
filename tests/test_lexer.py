"""Lexer behaviour that is easier to check on tokens than through a whole program."""

from __future__ import annotations

from manv.lexer import Lexer

PROGRAM = (
    "fn main() -> int:\n"
    "    array xs = [1,\n"
    "        2]\n"
    "    if true:\n"
    "        print(\"hi\")\n"
    "    return 0\n"
)


def _kinds(source: str) -> list[tuple[str, str]]:
    return [(t.kind, t.lexeme) for t in Lexer(source, "<test>").tokenize()]


def test_crlf_source_lexes_like_lf_source() -> None:
    assert _kinds(PROGRAM.replace("\n", "\r\n")) == _kinds(PROGRAM)


def test_mixed_line_endings_lex_like_lf_source() -> None:
    mixed = PROGRAM.replace("\n", "\r\n", 3)
    assert _kinds(mixed) == _kinds(PROGRAM)


def test_number_tokens_are_normalized_and_keep_the_source_spelling() -> None:
    tokens = [t for t in Lexer("0xFF 1_000 2.5e-3 0b11\n", "<test>").tokenize() if t.kind == "NUMBER"]
    assert [t.lexeme for t in tokens] == ["255", "1000", "0.0025", "3"]
    assert [t.raw_lexeme for t in tokens] == ["0xFF", "1_000", "2.5e-3", "0b11"]


def test_range_after_an_integer_is_not_a_float() -> None:
    kinds = [(t.kind, t.lexeme) for t in Lexer("0..5\n", "<test>").tokenize()][:3]
    assert kinds == [("NUMBER", "0"), ("OP", ".."), ("NUMBER", "5")]
