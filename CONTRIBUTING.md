# Contributing to ManV

Start with [syntax.md](syntax.md) (what the language is) and [TODO.md](TODO.md) (what is being worked on and in what order).

## Setup

ManV needs Python 3.12 or newer.

```
pip install -e .
pip install pytest
```

This installs the `manv` command and the runtime dependencies. Run a program with `manv run examples/01_hello_world.mv`, or `python -m manv.cli run <file>` without installing.

## Running the tests

```
python -m pytest -q
```

Some tests need something your machine may not have. They are marked and skipped with a stated reason, never failed:

| Marker | Needs | Skipped when |
|--------|-------|--------------|
| `needs_clang` | an LLVM toolchain (`clang` on `PATH`) | `clang` is not found |
| `stdlib` | the standard library | always, unless you pass `--run-stdlib` |

The standard library is parked until the interpreter works, so `stdlib` tests are skipped by default. `python -m pytest --run-stdlib` runs them. They fail today, and that list is the starting point for the library redesign. Do not fix them in passing.

A green run is `0 failed`. Skips and the single `xfail` (a GPU/debugger test that is out of scope for now) are expected.

## Where things live

| Path | What |
|------|------|
| `manv/lexer.py`, `manv/tokens.py` | tokenizer (keywords, operators, indentation) |
| `manv/parser.py`, `manv/ast.py` | the grammar. **The parser is the authority** for what is valid syntax |
| `manv/semantics*.py` | static checks and diagnostics (`E2xxx`) |
| `manv/interpreter.py` | the current tree-walking interpreter |
| `manv/hlir*.py`, `manv/graph_*.py`, `manv/kernel_*.py` | the compiler's intermediate representations |
| `manv/cli.py` | the command-line interface (`argparse`) |
| `tests/` | unit and integration tests |
| `tests/e2e/fixtures/` | small projects used by the CLI tests |
| `tests/conformance/` | language-behaviour tests (`.mv` + `.expected`). See its README |
| `benchmarks/` | performance benchmarks. See its README |
| `examples/` | runnable example programs |

A syntax change touches the lexer or parser, the semantic pass, the interpreter, [syntax.md](syntax.md), the tree-sitter grammar in `tree-sitter-manv/`, and the tests. Update the docs in the same change.

## Conventions

- **Line endings.** The repository stores source files with CRLF line endings. Keep a file's existing endings when you edit it, and don't reformat whole files.
- **Commits.** Use a conventional prefix (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`) with an optional scope, for example `fix(init): ...`. Explain *why* in the body.
- **Don't add dependencies lightly.** The CLI deliberately uses only the standard library's `argparse`.
- **Diagnostics are a feature.** Every error has a stable `E####` code, a location and the offending source line. Add a test for the code and message when you add a new one.
- **Add a test with every fix**, and a conformance case when the change affects what a program means.
