from __future__ import annotations

import re
from typing import TextIO

from . import ast
from .compiler import parse_program
from .diagnostics import ManvError
from .interpreter import Interpreter
from .intrinsics import all_intrinsics
from .tokens import DOUBLE_CHAR_OPS, KEYWORDS, SINGLE_CHAR_OPS


INTRINSIC_NAMES = sorted(spec.name for spec in all_intrinsics())
BUILTIN_WORDS = {"print", "len", "help", "type", "isinstance", "issubclass", "id", "std", "__intrin"}
REPL_TOKEN_RE = re.compile(
    r"(\s+|#.*$|\"(?:\\.|[^\"])*\"?|\'(?:\\.|[^\'])*\'?|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|\d+\.\d+|\d+|==|!=|<=|>=|->|&&|\|\||.)"
)


class ReplSession:
    def __init__(self, out: TextIO):
        self.out = out
        self.interpreter = Interpreter(file="<repl>", stdout=out)

    def execute_source(self, source: str) -> None:
        program = parse_program(source, "<repl>")
        # The REPL should mirror the real language surface, including modern
        # class/type declarations and docstring-carrying definitions. We load
        # declarations through the interpreter's normal registration path so
        # runtime type state, method tables, and globals stay consistent with
        # file execution instead of maintaining a separate REPL-only model.
        if program.declarations:
            declarations_only = ast.Program(
                declarations=program.declarations,
                statements=[],
                span=program.span,
                docstring=program.docstring,
            )
            self.interpreter.load_program(declarations_only)
        for decl in program.declarations:
            if isinstance(decl, ast.FnDecl):
                self.out.write(f"registered fn {decl.name}\n")
            elif isinstance(decl, ast.TypeDecl):
                self.out.write(f"registered type {decl.name}\n")
            elif isinstance(decl, ast.ImplDecl):
                self.out.write(f"registered impl {decl.target}\n")
            elif isinstance(decl, ast.MacroDeclStub):
                self.out.write(f"stub accepted: {type(decl).__name__}\n")

        for stmt in program.statements:
            value = self.interpreter.execute_stmt(stmt, self.interpreter.global_env)
            if isinstance(stmt, ast.ExprStmt) and value is not None:
                self.out.write(f"{value}\n")

    def completion_words(self) -> list[str]:
        words = set(KEYWORDS)
        words.update(BUILTIN_WORDS)
        words.update(self.interpreter.functions.keys())
        words.update(self.interpreter.types.keys())
        words.update(self.interpreter.global_env.values.keys())
        return sorted(words)


def run_repl(in_stream: TextIO, out_stream: TextIO) -> int:
    session = ReplSession(out=out_stream)
    out_stream.write("ManV repl (type :q to quit)\n")
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
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.lexers import Lexer
    from prompt_toolkit.styles import Style

    class ManvReplCompleter(Completer):
        _word_pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

        def get_completions(self, document, complete_event):  # type: ignore[no-untyped-def]
            before = document.current_line_before_cursor
            cuda_ctx = re.search(r"__intrin\.cuda\.([A-Za-z_0-9]*)$", before)
            if cuda_ctx is not None:
                prefix = cuda_ctx.group(1)
                for name in INTRINSIC_NAMES:
                    if not name.startswith("cuda_"):
                        continue
                    public = name[len("cuda_") :]
                    if public.startswith(prefix):
                        yield Completion(public, start_position=-len(prefix), display=f"__intrin.cuda.{public}")
                return

            intrinsic_ctx = re.search(r"__intrin\.([A-Za-z_0-9]*)$", before)
            if intrinsic_ctx is not None:
                prefix = intrinsic_ctx.group(1)
                for name in INTRINSIC_NAMES:
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix), display=f"__intrin.{name}")
                return

            word = document.get_word_before_cursor(pattern=self._word_pattern)
            for entry in session.completion_words():
                if not word or entry.startswith(word):
                    yield Completion(entry, start_position=-len(word))

    class ManvReplLexer(Lexer):
        def lex_document(self, document):  # type: ignore[no-untyped-def]
            known_words = set(session.completion_words())

            def get_line(lineno: int) -> list[tuple[str, str]]:
                if lineno < 0 or lineno >= len(document.lines):
                    return []
                line = document.lines[lineno]
                return _style_line(line, known_words)

            return get_line

    style = Style.from_dict(
        {
            "prompt": "ansicyan bold",
            "keyword": "ansimagenta bold",
            "builtin": "ansiblue",
            "intrinsic": "ansired bold",
            "number": "ansiyellow",
            "string": "ansigreen",
            "comment": "ansibrightblack italic",
            "operator": "ansicyan",
            "name": "ansigray",
        }
    )

    prompt_session = PromptSession(
        history=InMemoryHistory(),
        completer=ManvReplCompleter(),
        complete_while_typing=True,
        lexer=ManvReplLexer(),
        style=style,
        auto_suggest=AutoSuggestFromHistory(),
    )
    buffer: list[str] = []
    prompt = ">>> "

    while True:
        try:
            raw = prompt_session.prompt([("class:prompt", prompt)])
        except EOFError:
            break
        except KeyboardInterrupt:
            buffer.clear()
            prompt = ">>> "
            continue
        prompt, should_exit = _handle_line(session, raw, buffer, prompt, out_stream)
        if should_exit:
            break
    return 0


def _style_line(line: str, known_words: set[str]) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    for part in REPL_TOKEN_RE.findall(line):
        style = ""
        if part.isspace():
            style = ""
        elif part.startswith("#"):
            style = "class:comment"
        elif part.startswith('"') or part.startswith("'"):
            style = "class:string"
        elif part.startswith("__intrin.cuda."):
            style = "class:intrinsic"
        elif part.startswith("__intrin."):
            style = "class:intrinsic"
        elif part.startswith("std."):
            style = "class:builtin"
        elif part in KEYWORDS:
            style = "class:keyword"
        elif part == "__intrin":
            style = "class:intrinsic"
        elif part in {"print", "len", "help", "type", "isinstance", "issubclass", "id", "std"}:
            style = "class:builtin"
        elif part and part[0].isdigit():
            style = "class:number"
        elif part in DOUBLE_CHAR_OPS or part in SINGLE_CHAR_OPS:
            style = "class:operator"
        elif part in known_words:
            style = "class:name"
        fragments.append((style, part))
    return fragments


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
            buffer.clear()
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
        buffer.clear()
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
