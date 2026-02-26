from __future__ import annotations

from typing import TextIO

from . import ast
from .compiler import parse_program
from .diagnostics import ManvError
from .interpreter import FunctionValue, Interpreter


class ReplSession:
    def __init__(self, out: TextIO):
        self.out = out
        self.interpreter = Interpreter(file="<repl>", stdout=out)

    def execute_source(self, source: str) -> None:
        program = parse_program(source, "<repl>")
        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                self.interpreter.functions[decl.name] = FunctionValue(decl=decl)
                self.out.write(f"registered fn {decl.name}\n")
            elif isinstance(decl, (ast.TypeDeclStub, ast.ImplDeclStub, ast.MacroDeclStub)):
                self.out.write(f"stub accepted: {type(decl).__name__}\n")

        for stmt in program.statements:
            value = self.interpreter.execute_stmt(stmt, self.interpreter.global_env)
            if isinstance(stmt, ast.ExprStmt) and value is not None:
                self.out.write(f"{value}\n")


def run_repl(in_stream: TextIO, out_stream: TextIO) -> int:
    session = ReplSession(out=out_stream)
    out_stream.write("manv repl (type :q to quit)\n")
    out_stream.flush()
    if _supports_interactive_editing(in_stream, out_stream):
        return _run_interactive_repl(session, out_stream)
    return _run_stream_repl(session, in_stream, out_stream)


def _run_stream_repl(session: ReplSession, in_stream: TextIO, out_stream: TextIO) -> int:
    buffer: list[str] = []
    prompt = ">>> "

    while True:
        out_stream.write(prompt)
        out_stream.flush()
        line = in_stream.readline()
        if line == "":
            break
        raw = line.rstrip("\n")
        prompt, should_exit = _handle_line(session, raw, buffer, prompt, out_stream)
        if should_exit:
            break
    return 0


def _run_interactive_repl(session: ReplSession, out_stream: TextIO) -> int:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory

    prompt_session = PromptSession(history=InMemoryHistory())
    buffer: list[str] = []
    prompt = ">>> "

    while True:
        try:
            raw = prompt_session.prompt(prompt)
        except EOFError:
            break
        except KeyboardInterrupt:
            buffer = []
            prompt = ">>> "
            continue
        prompt, should_exit = _handle_line(session, raw, buffer, prompt, out_stream)
        if should_exit:
            break
    return 0


def _handle_line(
    session: ReplSession,
    raw: str,
    buffer: list[str],
    prompt: str,
    out_stream: TextIO,
) -> tuple[str, bool]:
    if raw.strip() in {":q", "exit"}:
        return prompt, True

    if raw.strip() == "":
        if buffer:
            source = "\n".join(buffer) + "\n"
            try:
                session.execute_source(source)
            except ManvError as err:
                out_stream.write(err.render() + "\n")
            buffer = []
        return ">>> ", False

    buffer.append(raw)
    if raw.endswith(":"):
        return "... ", False

    if not any(l.endswith(":") for l in buffer):
        source = "\n".join(buffer) + "\n"
        try:
            session.execute_source(source)
        except ManvError as err:
            out_stream.write(err.render() + "\n")
        buffer = []
        return ">>> ", False
    return "... ", False


def _supports_interactive_editing(in_stream: TextIO, out_stream: TextIO) -> bool:
    isatty_in = getattr(in_stream, "isatty", None)
    isatty_out = getattr(out_stream, "isatty", None)
    if not bool(callable(isatty_in) and isatty_in() and callable(isatty_out) and isatty_out()):
        return False
    try:
        import prompt_toolkit  # noqa: F401
    except Exception:
        return False
    return True
