from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Span:
    file: str
    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: str
    span: Span
    line_text: str = ""

    def render(self) -> str:
        location = f"{self.span.file}:{self.span.line}:{self.span.column}"
        header = f"{self.severity}[{self.code}]: {self.message}"
        arrow = f" --> {location}"
        if not self.line_text:
            return f"{header}\n{arrow}"
        caret_len = max(1, (self.span.end_column or self.span.column) - self.span.column + 1)
        caret = " " * (self.span.column - 1) + "^" * caret_len
        return f"{header}\n{arrow}\n  |\n{self.span.line:>2} | {self.line_text}\n  | {caret}"


class ManvError(Exception):
    def __init__(self, diagnostic: Diagnostic, extra: Iterable[Diagnostic] | None = None):
        self.diagnostic = diagnostic
        self.extra = list(extra or [])
        super().__init__(diagnostic.message)

    def render(self) -> str:
        parts = [self.diagnostic.render()]
        parts.extend(d.render() for d in self.extra)
        return "\n\n".join(parts)


def diag(
    code: str,
    message: str,
    file: str,
    line: int,
    column: int,
    line_text: str = "",
    severity: str = "error",
) -> Diagnostic:
    return Diagnostic(
        code=code,
        message=message,
        severity=severity,
        span=Span(file=file, line=line, column=column),
        line_text=line_text,
    )


def read_line(path: str, line: int) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    lines = file_path.read_text(encoding="utf-8").splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return ""
