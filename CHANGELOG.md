# Changelog

All notable changes to ManV are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). ManV has not had a stable release, so everything is under *Unreleased* for now.

## Unreleased

### Added
- Default argument values and keyword arguments at call sites (`E1019`, `E1020`, `E2066`-`E2069`).
- Single-expression lambdas: `fn(x): x * 2`.
- Script mode: a file of top-level statements runs without a `main` function.
- f-strings with format specs: `f"{name}: {value:.2f}"` (syntax errors are `E1018`).
- Tuples `(a, b)` and unpacking (`a, b = f()`, `for k, v in ...`, `return a, b`).
- The `in` / `not in` operators, open-ended slices (`a[1..]`, `a[..2]`), and the builtins `ord` and `chr`.
- `elif`, and the conditional expression `a if cond else b` (the missing `else` is `E1017`).
- `syntax.md`: the language's philosophy and syntax, with every construct marked as working, parsed-but-unsupported, or planned.
- `TODO.md`: the roadmap to a working interpreter, then the compiler.
- Test markers `stdlib` and `needs_clang`, and a `--run-stdlib` option (see `conftest.py`). Tests that need something missing are skipped with a stated reason instead of failing.
- `CONTRIBUTING.md`.
- `tests/conformance/`: language-behaviour tests (`.mv` programs with hand-written expected output, 55 cases). Known gaps are strict expected-failures, so fixing one is recorded by deleting its `.xfail` file.
- `benchmarks/`: 11 benchmarks in ManV and CPython with a runner, output cross-checking and recorded baselines for the current interpreter.
- A GitHub Actions workflow that runs the tests on Linux, macOS and Windows (not yet run).

### Changed
- `m.items()` and `enumerate()` return tuples instead of two-element arrays.
- Strings: indexing (`s[i]`, `s[-1]`), slicing (`s[a..b]`, also on arrays), iteration over characters, and methods (`upper`, `lower`, `strip`, `split`, `join`, `replace`, `find`, `startswith`, `endswith`, `contains`, ...).
- Arrays gained `append`, `pop`, `insert`, `extend`, `remove`, `index`, `contains`, `count`, `reverse`, `sort`, `copy` and `clear`; maps gained `keys`, `values`, `items`, `get`, `has`, `remove`, `copy` and `clear`, and `for k in m` iterates keys in insertion order. Arrays and maps are reference values; `copy()` makes an independent one.
- **Declarations:** `let` is removed (using it is `E1015`). A typed declaration `T name = v` now works for every type, including `i64`, `ptr`, user-defined types and `f32[]`. `name = v` declares the variable when the name is not bound yet (scope is by block). `const [T] NAME = v` declares a constant: assigning it is `E2064`, changing it through its name (`C[i] = v`, `C.f = v`) is `E2065`, and a missing value is `E1016`. Type bodies declare fields as `int x = 0`.
- `tools/migrate_let.py` rewrites old code (including programs embedded in Python test strings) to the new syntax. It was run over `examples/` and `tests/`.
- The REPL decides declare-or-assign against its own globals and tracks constants itself, since it has no semantic pass.
- String escape sequences are interpreted (`\n \t \r \0 \\ \" \' \xNN \uNNNN`); an unknown escape is `E0006`. Before, a backslash was silently dropped, so `"\n"` printed `n`.
- Array, map and call literals can span lines, with comments and blank lines inside. An unclosed bracket is `E0007`. Block lambdas inside a call keep working.
- Number literals: hex (`0xFF`), octal (`0o17`), binary (`0b101`), `_` digit separators (`1_000`) and float exponents (`1e-3`). A malformed number is `E0008`.
- Augmented assignment (`+=` and friends), `match`/`case` and nested functions are accepted by the semantic pass. The interpreter already implemented them.
- The CLI uses `argparse` instead of Typer. Options that default to on (`--cuda-jit`, `--optimize`) now also accept `--no-<flag>`, and option abbreviations are rejected. The console script is now `manv.cli:main`, so editable installs must be reinstalled.
- The interpreter no longer puts the installed `std` package on its module search path. Programs resolve modules from the project root, `MANV_PATH`, and dependencies added with `manv add`.
- `examples/08_modules_stdlib.mv` moved to `examples/_parked/` until the standard library is redesigned.

### Removed
- The "stdlib not installed" warning that was printed by most CLI commands.
- The `typer` dependency, and the packages that came only with it (click, rich, shellingham, markdown-it-py, mdurl).
- Compiled Python files and test build artifacts that were committed despite `.gitignore`.

### Fixed
- Runtime errors (uncaught exceptions) were always reported at `1:1`; they now point at the line and column where the error happened (`tests/test_runtime_positions.py`).
- `not` bound tighter than comparison, so `not a == b` meant `(not a) == b`; it is now looser than comparison.
- `print` and `str()` wrote Python's spellings (`True`, `None`); they now write `true`, `false` and `none`, also inside arrays and maps. `print(obj)` and `str(obj)` use the type's `__str__` or `__repr__` instead of dumping interpreter internals.
- `and` / `or` evaluated both operands; they now short-circuit, in the interpreter and in HLIR lowering.
- `e.message` and `e.type` can be read on a caught exception. A reserved word is allowed as an attribute name after `.`.
- `print` with more than one argument was rejected by the semantic pass (`E2021`) although the runtime supported it; it now accepts any number of values.
- The HLIR interpreter printed `True`/`None`; it shares the interpreter's formatter now.
- Output printed before an uncaught error was discarded; `manv run` now prints it before the diagnostic.
- `case x if cond:` never parsed: the guard `if` is a keyword token but the parser looked for an identifier.
- The `else:` body of a `while` loop was skipped by the semantic pass, so undefined names in it went unreported.
- A `while` loop failed with `loop iteration guard exceeded` after 1,000,000 iterations. The cap is removed.
- A line indented with only tabs was reported as `E0004` (unexpected character); it is now `E0001` (tabs are not allowed for indentation), and a line containing only whitespace is treated as blank.
- `manv init` generated a `module` line that did not parse for project names containing spaces, a leading digit, a reserved word or non-ASCII characters. The name in `project.toml` is unchanged; only the module identifier is normalized.
