# Changelog

All notable changes to ManV are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). ManV has not had a stable release, so everything is under *Unreleased* for now.

## Unreleased

### Added
- `syntax.md`: the language's philosophy and syntax, with every construct marked as working, parsed-but-unsupported, or planned.
- `TODO.md`: the roadmap to a working interpreter, then the compiler.
- Test markers `stdlib` and `needs_clang`, and a `--run-stdlib` option (see `conftest.py`). Tests that need something missing are skipped with a stated reason instead of failing.
- `CONTRIBUTING.md`.
- `tests/conformance/`: language-behaviour tests (`.mv` programs with hand-written expected output, 55 cases). Known gaps are strict expected-failures, so fixing one is recorded by deleting its `.xfail` file.
- `benchmarks/`: 11 benchmarks in ManV and CPython with a runner, output cross-checking and recorded baselines for the current interpreter.
- A GitHub Actions workflow that runs the tests on Linux, macOS and Windows (not yet run).

### Changed
- String escape sequences are interpreted (`\n \t \r \0 \\ \" \' \xNN \uNNNN`); an unknown escape is `E0006`. Before, a backslash was silently dropped, so `"\n"` printed `n`.
- Array, map and call literals can span lines, with comments and blank lines inside. An unclosed bracket is `E0007`. Block lambdas inside a call keep working.
- Augmented assignment (`+=` and friends), `match`/`case` and nested functions are accepted by the semantic pass. The interpreter already implemented them.
- The CLI uses `argparse` instead of Typer. Options that default to on (`--cuda-jit`, `--optimize`) now also accept `--no-<flag>`, and option abbreviations are rejected. The console script is now `manv.cli:main`, so editable installs must be reinstalled.
- The interpreter no longer puts the installed `std` package on its module search path. Programs resolve modules from the project root, `MANV_PATH`, and dependencies added with `manv add`.
- `examples/08_modules_stdlib.mv` moved to `examples/_parked/` until the standard library is redesigned.

### Removed
- The "stdlib not installed" warning that was printed by most CLI commands.
- The `typer` dependency, and the packages that came only with it (click, rich, shellingham, markdown-it-py, mdurl).
- Compiled Python files and test build artifacts that were committed despite `.gitignore`.

### Fixed
- `case x if cond:` never parsed: the guard `if` is a keyword token but the parser looked for an identifier.
- The `else:` body of a `while` loop was skipped by the semantic pass, so undefined names in it went unreported.
- A `while` loop failed with `loop iteration guard exceeded` after 1,000,000 iterations. The cap is removed.
- A line indented with only tabs was reported as `E0004` (unexpected character); it is now `E0001` (tabs are not allowed for indentation), and a line containing only whitespace is treated as blank.
- `manv init` generated a `module` line that did not parse for project names containing spaces, a leading digit, a reserved word or non-ASCII characters. The name in `project.toml` is unchanged; only the module identifier is normalized.
