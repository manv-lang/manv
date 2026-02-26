from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    kind: str
    lexeme: str
    line: int
    column: int

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "lexeme": self.lexeme,
            "line": self.line,
            "column": self.column,
        }


KEYWORDS = {
    "fn",
    "let",
    "int",
    "str",
    "array",
    "map",
    "u8",
    "usize",
    "float",
    "bool",
    "if",
    "else",
    "while",
    "break",
    "continue",
    "return",
    "and",
    "or",
    "not",
    "true",
    "false",
    "none",
    "type",
    "impl",
    "macro",
    "try",
    "except",
    "finally",
    "raise",
    "as",
    "gpu",
    "memory",
}

DOUBLE_CHAR_OPS = {"->", "==", "!=", "<=", ">=", "&&", "||"}
SINGLE_CHAR_OPS = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "=",
    "!",
    "<",
    ">",
    ":",
    ",",
    ";",
    ".",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
}
