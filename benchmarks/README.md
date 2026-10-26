# Benchmarks

The speed of the interpreter is a design goal ([TODO.md, Phase 5](../TODO.md)), so it is measured from the start. This directory holds the programs, a runner, and the recorded baseline for the current Python tree-walking interpreter. The C engine will be judged against these numbers.

```
python benchmarks/run.py                          # small sizes, median of 3 runs
python benchmarks/run.py --size full --runs 5     # the sizes a fast engine is meant for
python benchmarks/run.py --only fib loop
python benchmarks/run.py --save benchmarks/baseline-small.json
python benchmarks/run.py --compare benchmarks/baseline-small.json
```

## How it works

Every benchmark in `src/` exists twice with the same algorithm: `name.mv` for ManV and `name.py` for CPython. The runner fills in the `@N@` / `@M@` size placeholders, runs each version as a separate process, and reports the median wall-clock time. Wall-clock includes process startup on purpose, because startup is part of what a user feels.

Both versions must print the same output. A mismatch is reported as a failure, so the benchmarks also act as a cross-check of ManV's arithmetic and collections against CPython.

| Benchmark | What it stresses |
|-----------|------------------|
| `startup` | process start and hello world |
| `loop` | a tight `while` loop of integer arithmetic |
| `fib` | recursive function calls |
| `nested_loops` | `for` over ranges, integer `*` and `%` |
| `array_fill_sum` | building and traversing an array |
| `map_ops` | map insertion and lookup with string keys |
| `string_build` | repeated string concatenation |
| `method_calls` | method calls and field updates on a user type |
| `closure_calls` | calls through a closure |
| `exceptions` | `raise` and `except` in a loop |
| `mandelbrot` | float arithmetic and branching |

## Sizes

- `small` is tuned so the current Python interpreter spends about a second or two per benchmark. At this size CPython finishes inside its own startup time, so **the `manv/cpython` ratio is not meaningful for small**. Use it as a quick check and to track changes in ManV alone (`--compare`).
- `full` is sized for a fast engine. Compare engines at this size.

## Baselines

`baseline-small.json` and `baseline-full.json` record the Python tree-walking interpreter before any engine work (CPython 3.13, Windows, AMD64). They are single-machine numbers: compare against them on the same machine, and re-record them if the hardware changes.

## Adding a benchmark

1. Add `src/<name>.mv` and `src/<name>.py` with the same algorithm. Print a single checksum.
2. Add the name and its `small` / `full` placeholder values to `SIZES` in `run.py`.
3. Use only constructs the current interpreter supports (see [syntax.md](../syntax.md)). Avoid anything whose semantics are still undecided.

To add an engine, add an entry to `engines()` in `run.py`: the source suffix and the command prefix. The C engine goes there.
