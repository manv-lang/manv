# Conformance tests

Each `.mv` program here is run with `manv run`, and its behaviour is compared with what the language is supposed to do. They are the specification in executable form. The same suite will be run against every engine (the Python interpreter, the C engine, the compiler), and any difference is a bug in the engine, not in the test.

Run them with `python -m pytest tests/test_conformance.py`.

## Layout

One feature per program, grouped by area:

| Directory | What it covers |
|-----------|----------------|
| `basics/` | values, operators, conversions, printing |
| `control_flow/` | `if`, loops, `match`, augmented assignment |
| `functions/` | calls, recursion, higher-order functions, closures |
| `collections/` | arrays and maps |
| `types/` | `type`, `impl`, inheritance, accessors |
| `exceptions/` | `try` / `except` / `else` / `finally`, `raise` |
| `declarations/` | `T x = ...`, `x = ...`, `const` (the redesign in [syntax.md §5.1](../../syntax.md)) |
| `errors/` | programs that must be rejected, with a specific diagnostic code |

Next to `name.mv` there is a sidecar for each thing the run is expected to do:

| File | Meaning |
|------|---------|
| `name.expected` | the exact standard output |
| `name.error` | the program must fail (exit code 1) and print `error[<CODE>]`. The file holds the code, e.g. `E2010` |
| `name.exit` | the exit code `main` returns, when it is not 0 |
| `name.xfail` | a known gap. The text is the reason, and it should point at the TODO item |

A program needs `.expected` or `.error` (or both: an erroring program may still have printed something first). A program that succeeds must also leave stderr empty.

## Rules for writing a case

1. **Write the expected output by hand, from the language rules** in [syntax.md](../../syntax.md). Never copy it from what an implementation printed, or the test only proves the implementation agrees with itself.
2. **Only pin what is decided.** If the behaviour is an open question (see the decisions table in [TODO.md](../../TODO.md)), leave it out of the suite. Examples: integer `/`, how `array_append` treats its argument, the order of `map_keys`, how a collection of strings prints.
3. **Keep each case small and about one thing**, so a failure points at one feature.
4. **Prefer syntax that survives the planned changes.** The typed form `int x = 1` stays valid after `let` is removed. Avoid `let` in new cases. Cases that need the new declaration syntax live in `declarations/` and are marked `.xfail` until that work lands.

## Known gaps (`.xfail`)

An `.xfail` case runs and is *expected to fail*. It is **strict**: when the gap is fixed the case starts passing, the run reports it as an error, and you delete the `.xfail` file. That is how a fix is recorded. `grep -r "" tests/conformance --include=*.xfail` lists every open gap with its reason.

## Running against another engine

`tests/test_conformance.py` calls `manv run <file>`. When the C engine lands, parametrize the invocation over `--engine` there. The programs and expected files do not change.
