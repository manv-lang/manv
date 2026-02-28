from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    kind: str
    lexeme: str
    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None
    raw_lexeme: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "lexeme": self.lexeme,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "raw_lexeme": self.raw_lexeme,
        }


KEYWORDS = {
    "fn",
    "let",
    "int",
    "i32",
    "str",
    "array",
    "map",
    "u8",
    "usize",
    "float",
    "f32",
    "bool",
    "void",
    "if",
    "else",
    "while",
    "for",
    "in",
    "break",
    "continue",
    "return",
    "and",
    "or",
    "not",
    "true",
    "false",
    "True",
    "False",
    "none",
    "type",
    "class",
    "impl",
    "macro",
    "import",
    "from",
    "try",
    "except",
    "finally",
    "raise",
    "as",
    "syscall",
    "gpu",
    "memory",
}

DOUBLE_CHAR_OPS = {"->", "==", "!=", "<=", ">=", "&&", "||", ".."}
SINGLE_CHAR_OPS = {
    "@",
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
