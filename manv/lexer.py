from __future__ import annotations

from .diagnostics import ManvError, diag
from .tokens import DOUBLE_CHAR_OPS, KEYWORDS, SINGLE_CHAR_OPS, Token


class Lexer:
    def __init__(self, source: str, file: str):
        self.source = source
        self.file = file
        self.lines = source.splitlines()

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        indents = [0]

        for line_no, raw in enumerate(self.lines, start=1):
            stripped = raw.lstrip(" ")
            if stripped == "" or stripped.startswith("#"):
                tokens.append(Token("NEWLINE", "\n", line_no, 1))
                continue

            indent = len(raw) - len(stripped)
            if "\t" in raw[:indent]:
                raise ManvError(
                    diag(
                        "E0001",
                        "tabs are not allowed for indentation",
                        self.file,
                        line_no,
                        1,
                        raw,
                    )
                )

            if indent > indents[-1]:
                indents.append(indent)
                tokens.append(Token("INDENT", "<indent>", line_no, 1))
            else:
                while indent < indents[-1]:
                    indents.pop()
                    tokens.append(Token("DEDENT", "<dedent>", line_no, 1))
                if indent != indents[-1]:
                    raise ManvError(
                        diag(
                            "E0002",
                            "inconsistent indentation",
                            self.file,
                            line_no,
                            1,
                            raw,
                        )
                    )

            self._tokenize_line(stripped, line_no, indent, tokens)
            tokens.append(Token("NEWLINE", "\n", line_no, len(raw) + 1))

        last_line = max(len(self.lines), 1)
        while len(indents) > 1:
            indents.pop()
            tokens.append(Token("DEDENT", "<dedent>", last_line, 1))
        tokens.append(Token("EOF", "", last_line + 1, 1))
        return tokens

    def _tokenize_line(self, line: str, line_no: int, indent: int, out: list[Token]) -> None:
        i = 0
        while i < len(line):
            ch = line[i]
            col = indent + i + 1

            if ch in {" ", "\r"}:
                i += 1
                continue
            if ch == "#":
                return

            if i + 1 < len(line):
                pair = line[i : i + 2]
                if pair in DOUBLE_CHAR_OPS:
                    out.append(Token("OP", pair, line_no, col))
                    i += 2
                    continue

            if ch in SINGLE_CHAR_OPS:
                out.append(Token("OP", ch, line_no, col))
                i += 1
                continue

            if ch == '"' or ch == "'":
                quote = ch
                i += 1
                start = i
                escaped = False
                value_chars: list[str] = []
                while i < len(line):
                    current = line[i]
                    if escaped:
                        value_chars.append(current)
                        escaped = False
                        i += 1
                        continue
                    if current == "\\":
                        escaped = True
                        i += 1
                        continue
                    if current == quote:
                        break
                    value_chars.append(current)
                    i += 1
                if i >= len(line) or line[i] != quote:
                    raise ManvError(
                        diag(
                            "E0003",
                            "unterminated string literal",
                            self.file,
                            line_no,
                            col,
                            line,
                        )
                    )
                out.append(Token("STRING", "".join(value_chars), line_no, col))
                i += 1
                continue

            if ch.isdigit():
                start = i
                has_dot = False
                while i < len(line):
                    if line[i].isdigit():
                        i += 1
                        continue
                    if (
                        line[i] == "."
                        and not has_dot
                        and (i + 1 >= len(line) or line[i + 1] != ".")
                    ):
                        has_dot = True
                        i += 1
                        continue
                    break
                out.append(Token("NUMBER", line[start:i], line_no, indent + start + 1))
                continue

            if ch.isalpha() or ch == "_":
                start = i
                while i < len(line) and (line[i].isalnum() or line[i] == "_"):
                    i += 1
                text = line[start:i]
                kind = "KEYWORD" if text in KEYWORDS else "IDENT"
                out.append(Token(kind, text, line_no, indent + start + 1))
                continue

            raise ManvError(
                diag(
                    "E0004",
                    f"unexpected character '{ch}'",
                    self.file,
                    line_no,
                    col,
                    line,
                )
            )
