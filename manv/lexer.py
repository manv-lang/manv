from __future__ import annotations

from .diagnostics import ManvError, diag
from .tokens import DOUBLE_CHAR_OPS, KEYWORDS, SINGLE_CHAR_OPS, Token


_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'"}


class Lexer:
    def __init__(self, source: str, file: str):
        self.source = source
        self.file = file
        self.lines = source.splitlines()

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        indents = [0]
        line_index = 0
        # Open brackets as (char, line, column). While one is open, line breaks are not
        # statement ends and indentation means nothing ("continuation lines").
        self._brackets: list[tuple[str, int, int]] = []
        # A block lambda inside brackets (`f(xs, fn(x):` + indented body) is the one place
        # layout still applies. Each entry is [bracket depth to return to, body indent or None].
        blocks: list[list[int | None]] = []

        while line_index < len(self.lines):
            line_no = line_index + 1
            raw = self.lines[line_index]

            if len(self._brackets) > self._block_base(blocks):
                body = raw.lstrip(" \t")
                if body != "" and not body.startswith("#"):
                    start = len(tokens)
                    self._tokenize_line(body, line_no, len(raw) - len(body), tokens)
                    self._end_line(tokens, start, line_no, raw, blocks)
                line_index += 1
                continue

            stripped = raw.lstrip(" ")
            # Leading whitespace of either kind: a line holding only tabs is blank, and a
            # tab anywhere in the indentation is reported as such (not as a stray character).
            body = raw.lstrip(" \t")
            if body == "" or body.startswith("#"):
                tokens.append(Token("NEWLINE", "\n", line_no, 1, end_line=line_no, end_column=len(raw) + 1))
                line_index += 1
                continue

            indent = len(raw) - len(stripped)
            if "\t" in raw[: len(raw) - len(body)]:
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
                if blocks and blocks[-1][1] is None:
                    blocks[-1][1] = indent
                indents.append(indent)
                tokens.append(Token("INDENT", "<indent>", line_no, 1, end_line=line_no, end_column=1))
            else:
                while indent < indents[-1]:
                    indents.pop()
                    tokens.append(Token("DEDENT", "<dedent>", line_no, 1, end_line=line_no, end_column=1))
                while blocks and blocks[-1][1] is not None and indent < blocks[-1][1]:
                    blocks.pop()
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

            start = len(tokens)
            self._tokenize_line(stripped, line_no, indent, tokens)
            self._end_line(tokens, start, line_no, raw, blocks)
            line_index += 1

        if self._brackets:
            char, open_line, open_col = self._brackets[-1]
            raise ManvError(
                diag(
                    "E0007",
                    f"unclosed '{char}'",
                    self.file,
                    open_line,
                    open_col,
                    self.lines[open_line - 1],
                )
            )

        last_line = max(len(self.lines), 1)
        while len(indents) > 1:
            indents.pop()
            tokens.append(Token("DEDENT", "<dedent>", last_line, 1, end_line=last_line, end_column=1))
        tokens.append(Token("EOF", "", last_line + 1, 1, end_line=last_line + 1, end_column=1))
        return tokens

    @staticmethod
    def _block_base(blocks: list[list[int | None]]) -> int:
        """Bracket depth at which a new line starts a statement (0, or inside a block lambda)."""
        return blocks[-1][0] if blocks else 0

    def _end_line(self, tokens: list[Token], start: int, line_no: int, raw: str, blocks: list[list[int | None]]) -> None:
        """Close a physical line: emit NEWLINE when a statement ended, or open a block lambda."""
        newline = Token("NEWLINE", "\n", line_no, len(raw) + 1, end_line=line_no, end_column=len(raw) + 1)
        depth = len(self._brackets)
        if depth <= self._block_base(blocks):
            tokens.append(newline)
            return
        line_tokens = tokens[start:]
        opens_block_lambda = (
            bool(line_tokens)
            and line_tokens[-1].kind == "OP"
            and line_tokens[-1].lexeme == ":"
            and any(tok.kind == "KEYWORD" and tok.lexeme == "fn" for tok in line_tokens)
        )
        if opens_block_lambda:
            tokens.append(newline)
            blocks.append([depth, None])

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
                if ch in "([{":
                    self._brackets.append((ch, line_no, col))
                elif ch in ")]}" and self._brackets:
                    self._brackets.pop()
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
                value_chars: list[str] = []
                raw_chars: list[str] = []
                while i < len(line):
                    current = line[i]
                    if current == "\\" and i + 1 < len(line):
                        decoded, width = self._decode_escape(line, i, line_no, indent)
                        raw_chars.append(line[i : i + width])
                        value_chars.append(decoded)
                        i += width
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

    def _decode_escape(self, line: str, i: int, line_no: int, indent: int) -> tuple[str, int]:
        """Decode the escape sequence starting at line[i] (a backslash). Returns (text, width)."""
        kind = line[i + 1]
        if kind in _SIMPLE_ESCAPES:
            return _SIMPLE_ESCAPES[kind], 2
        digits = {"x": 2, "u": 4}.get(kind)
        if digits is not None:
            hex_text = line[i + 2 : i + 2 + digits]
            if len(hex_text) == digits and all(c in "0123456789abcdefABCDEF" for c in hex_text):
                return chr(int(hex_text, 16)), 2 + digits
            message = f"invalid \\{kind} escape: expected {digits} hex digits"
        else:
            message = f"unknown escape sequence '\\{kind}'"
        raise ManvError(diag("E0006", message, self.file, line_no, indent + i + 1, line))

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
