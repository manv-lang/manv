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
        line_index = 0

        while line_index < len(self.lines):
            line_no = line_index + 1
            raw = self.lines[line_index]
            stripped = raw.lstrip(" ")
            if stripped == "" or stripped.startswith("#"):
                tokens.append(Token("NEWLINE", "\n", line_no, 1, end_line=line_no, end_column=len(raw) + 1))
                line_index += 1
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
                tokens.append(Token("INDENT", "<indent>", line_no, 1, end_line=line_no, end_column=1))
            else:
                while indent < indents[-1]:
                    indents.pop()
                    tokens.append(Token("DEDENT", "<dedent>", line_no, 1, end_line=line_no, end_column=1))
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

            if stripped.startswith('"""') or stripped.startswith("'''"):
                token, closing_line = self._tokenize_triple_docstring(line_index, indent)
                tokens.append(token)
                closing_raw = self.lines[closing_line]
                closing_no = closing_line + 1
                tokens.append(Token("NEWLINE", "\n", closing_no, len(closing_raw) + 1, end_line=closing_no, end_column=len(closing_raw) + 1))
                line_index = closing_line + 1
                continue

            self._tokenize_line(stripped, line_no, indent, tokens)
            tokens.append(Token("NEWLINE", "\n", line_no, len(raw) + 1, end_line=line_no, end_column=len(raw) + 1))
            line_index += 1

        last_line = max(len(self.lines), 1)
        while len(indents) > 1:
            indents.pop()
            tokens.append(Token("DEDENT", "<dedent>", last_line, 1, end_line=last_line, end_column=1))
        tokens.append(Token("EOF", "", last_line + 1, 1, end_line=last_line + 1, end_column=1))
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
                    out.append(Token("OP", pair, line_no, col, end_line=line_no, end_column=col + 1))
                    i += 2
                    continue

            if ch in SINGLE_CHAR_OPS:
                out.append(Token("OP", ch, line_no, col, end_line=line_no, end_column=col))
                i += 1
                continue

            if ch == '"' or ch == "'":
                if line[i : i + 3] == ch * 3:
                    raise ManvError(
                        diag(
                            "E0005",
                            "triple-quoted strings are only supported as standalone docstrings",
                            self.file,
                            line_no,
                            col,
                            line,
                        )
                    )
                quote = ch
                i += 1
                escaped = False
                value_chars: list[str] = []
                raw_chars: list[str] = []
                while i < len(line):
                    current = line[i]
                    if escaped:
                        raw_chars.append(current)
                        value_chars.append(current)
                        escaped = False
                        i += 1
                        continue
                    if current == "\\":
                        raw_chars.append(current)
                        escaped = True
                        i += 1
                        continue
                    if current == quote:
                        break
                    raw_chars.append(current)
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
                out.append(
                    Token(
                        "STRING",
                        "".join(value_chars),
                        line_no,
                        col,
                        end_line=line_no,
                        end_column=indent + i + 1,
                        raw_lexeme="".join(raw_chars),
                    )
                )
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
                out.append(
                    Token(
                        "NUMBER",
                        line[start:i],
                        line_no,
                        indent + start + 1,
                        end_line=line_no,
                        end_column=indent + i,
                    )
                )
                continue

            if ch.isalpha() or ch == "_":
                start = i
                while i < len(line) and (line[i].isalnum() or line[i] == "_"):
                    i += 1
                text = line[start:i]
                kind = "KEYWORD" if text in KEYWORDS else "IDENT"
                out.append(
                    Token(
                        kind,
                        text,
                        line_no,
                        indent + start + 1,
                        end_line=line_no,
                        end_column=indent + i,
                    )
                )
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

    def _tokenize_triple_docstring(self, start_line_index: int, indent: int) -> tuple[Token, int]:
        """Lex a triple-quoted docstring token spanning one or more physical lines.

        Why this exists:
        - Docstrings need multiline text preservation, but the general language
          still treats ordinary runtime strings as the narrower existing surface.
        - Keeping this path explicit lets us preserve exact raw text for docs
          without silently broadening arbitrary multiline string semantics.
        """

        raw = self.lines[start_line_index]
        stripped = raw[indent:]
        quote = stripped[:3]
        remainder = stripped[3:]
        close_index = self._find_triple_quote_end(remainder, quote)
        if close_index >= 0:
            tail = remainder[close_index + 3 :].strip()
            if tail and not tail.startswith("#"):
                raise ManvError(
                    diag(
                        "E0005",
                        "docstring literal must be the only statement on its line",
                        self.file,
                        start_line_index + 1,
                        indent + close_index + 4,
                        raw,
                    )
                )
            raw_text = remainder[:close_index]
            return (
                Token(
                    "STRING",
                    self._decode_string_text(raw_text),
                    start_line_index + 1,
                    indent + 1,
                    end_line=start_line_index + 1,
                    end_column=indent + close_index + 6,
                    raw_lexeme=raw_text,
                ),
                start_line_index,
            )

        pieces: list[str] = [remainder]
        for line_index in range(start_line_index + 1, len(self.lines)):
            line = self.lines[line_index]
            close_index = self._find_triple_quote_end(line, quote)
            if close_index >= 0:
                tail = line[close_index + 3 :].strip()
                if tail and not tail.startswith("#"):
                    raise ManvError(
                        diag(
                            "E0005",
                            "docstring literal must be the only statement on its line",
                            self.file,
                            line_index + 1,
                            close_index + 4,
                            line,
                        )
                    )
                pieces.append(line[:close_index])
                raw_text = "\n".join(pieces)
                return (
                    Token(
                        "STRING",
                        self._decode_string_text(raw_text),
                        start_line_index + 1,
                        indent + 1,
                        end_line=line_index + 1,
                        end_column=close_index + 3,
                        raw_lexeme=raw_text,
                    ),
                    line_index,
                )
            pieces.append(line)

        raise ManvError(
            diag(
                "E0003",
                "unterminated string literal",
                self.file,
                start_line_index + 1,
                indent + 1,
                raw,
            )
        )

    def _find_triple_quote_end(self, text: str, quote: str) -> int:
        i = 0
        while i <= len(text) - 3:
            if text[i : i + 3] == quote and not self._is_escaped(text, i):
                return i
            i += 1
        return -1

    def _is_escaped(self, text: str, index: int) -> bool:
        slash_count = 0
        probe = index - 1
        while probe >= 0 and text[probe] == "\\":
            slash_count += 1
            probe -= 1
        return (slash_count % 2) == 1

    def _decode_string_text(self, raw_text: str) -> str:
        value_chars: list[str] = []
        i = 0
        while i < len(raw_text):
            current = raw_text[i]
            if current == "\\" and i + 1 < len(raw_text):
                value_chars.append(raw_text[i + 1])
                i += 2
                continue
            value_chars.append(current)
            i += 1
        return "".join(value_chars)
