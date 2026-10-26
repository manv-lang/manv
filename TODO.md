# ManV TODO

**Goal of this plan:** make ManV a real language. **The interpreter comes first, and the compiler comes after.**

The compiler is a separate effort ([§9](#9-after-the-interpreter-the-compiler)). It starts only when the interpreter meets the exit criteria in [§0](#0-what-interpreter-works-means), because the interpreter defines the language's meaning and the compiler has to match it.

Companion document: [syntax.md](syntax.md) describes the language as it is and as it is meant to be. Many items below close the gaps it marks ⚠️ and 🧭.

## Ground rules (decided)

1. **The interpreter's hot path is written in C.** "Fast" is a design goal of the interpreter, not a later optimization. See [Phase 5](#phase-5--the-c-engine-fast-interpreter) for what that does and does not mean.
2. **There is no standard library for now.** Plan, write and test everything as if `stdlib/` did not exist. The core language plus a small set of **native builtins** must be enough to write real programs. The stdlib is redesigned after the interpreter works and the syntax is settled, and *only then* ([Parked: the standard library](#parked-the-standard-library)). Nothing in the stdlib gets edited, migrated or fixed in the meantime, so a syntax change never means touching hundreds of library files.

---

## Status snapshot (2026-10-26)

What already exists, checked by running it:

- A lexer, a recursive-descent parser, a semantic pass and a tree-walking interpreter. Running `python -m manv.cli run file.mv` works.
- All 10 programs in `examples/` run to completion with exit code 0, though several print wrong output (see [P1](#phase-1--interpreter-correctness-bugs)).
- `pytest`: **252 passed, 12 skipped (with stated reasons), 15 expected failures, 0 failed**. At the start the suite was 203 passed / 12 failed; Phase 0 below got it to zero failures (see [how the 12 were resolved](#how-the-12-failing-tests-were-resolved)).
- `tests/conformance/` holds language-behaviour cases (`.mv` + hand-written `.expected`) and `benchmarks/` holds the performance harness with a recorded baseline. Both are described in their READMEs.
- A large amount of compiler-side code already exists (HLIR, Graph IR, Kernel IR, LLVM, PTX, SPIR-V, WGSL, DAP, LSP, registry). It is out of scope until the interpreter is done.
- **Performance baseline** (measured on this machine, 2026-10-26; `gcc` 15.2 from w64devkit is installed, `clang` and MSVC are not):

  | Measurement | ManV interpreter (Python tree-walker) | CPython 3.13, for scale |
  |-------------|---------------------------------------|-------------------------|
  | Startup (`hello world`) | 0.46 s to 0.65 s (varies by run) | 0.07 s |
  | `while` loop, 2,000,000 iterations (one compare, two adds, two assignments each) | 5.1 s total, about 4.5 s net of startup (~2.2 µs/iteration) | 0.20 s |
  | `fib(22)` (about 57k calls) | 1.16 s total, about 0.5 s net of startup (~9 µs/call) | 0.002 s |

  These first numbers are single runs, so they are indicative. The reproducible measurements are in [benchmarks/](benchmarks/README.md) (`baseline-small.json`, `baseline-full.json`). At the full benchmark sizes the interpreter is **14× to 150× slower than CPython** depending on the workload (about 14× on map operations and string building, 60× to 130× on calls, arrays, nested loops and exceptions, 93× on a tight `while` loop, 150× on float-heavy code; single run each). They are the numbers the C engine has to beat.

**Not yet true of the interpreter:** it is not a language you could write a real program in. Basic things fail: `x += 1`, `match`, nested functions, `"\n"`, string indexing and methods, multi-line literals, `elif`, and printing an object with a `__repr__`. Details in the phases below.

### How the 12 failing tests were resolved

None of them was fixed by changing what it tests; each is now skipped or expected-to-fail with a stated reason. The skip logic lives in `conftest.py`.

| Cause | Tests | Now |
|-------|-------|-----|
| Need the standard library (`include math`, `std` imports, std-only intrinsics like `core_len`) | `test_math_docstrings` (3), `test_oop_exceptions_runtime` (2), `test_syscall_keyword` (1), `test_llvm_codegen::…core_len` (1) | marker `stdlib`, skipped unless `--run-stdlib`. With the flag all 7 still fail, as expected. They are the starting regression list for the library redesign. |
| Need an LLVM toolchain (`clang`), which this machine does not have | `test_abi_runtime` (2), `test_cli_matrix::test_compile_cuda_ptx_backend`, `test_cli_matrix::test_test_command_runs_fixture_suite` (its `build_ok` case) | marker `needs_clang`, skipped when `clang` is not on `PATH`. |
| GPU / debugger: with CUDA mocked, the program runs but never takes the GPU path | `test_dap_debugger::…cuda_source` | `xfail` (non-strict). Out of scope until the interpreter works; re-verify with the DAP work in Phase 6. |

---

## 0. What "interpreter works" means

The interpreter is done when **all** of these are true. They are the exit criteria for Phases 0–8.

- [ ] `pytest` is green on a machine with no `clang`, no GPU and **no standard library**. Compiler tests that need those, and tests that need the stdlib, are skipped with a stated reason, not failed.
- [ ] Every program in `examples/` runs and its output matches a committed `.expected` file (golden test).
- [ ] Every ✅ construct in [syntax.md](syntax.md) has a test. Every ⚠️ has been fixed or deliberately removed from the grammar. No construct parses and then dies with `E2099 unsupported statement`.
- [ ] A fresh clone can build the engine and run a program with no network access, no hand-set environment variables, and no `stdlib/` directory.
- [ ] A non-trivial program (about 300 lines: a text-processing tool, an expression evaluator, or a data-structure library) runs correctly using only the language and the native builtins.
- [ ] **The C engine runs the whole conformance suite** and agrees with the expected output on every case ([Phase 5](#phase-5--the-c-engine-fast-interpreter)), and meets its performance milestone M1.
- [ ] Uncaught errors print a clear diagnostic with source location and exit non-zero. Nothing prints a Python traceback or an internal object dump.
- [ ] `syntax.md` and the code agree, which is checked by a test that runs every fenced example in it.

---

## Phase 0 — Get a trustworthy baseline

Nothing else is worth building on a red test suite.

- [x] **Cut the interpreter's dependency on the stdlib.** The module search path is the project root, `MANV_PATH`, and packages installed from the global cache. The cache used to include the installed `std` package, so a run depended on what happened to be installed on the machine. `std` is now excluded (`installed_package_roots(exclude=...)`); dependencies added with `manv add` still resolve. Excluding it changed no test result.
- [x] Add a `stdlib` pytest marker and skip every test that needs the stdlib, with the reason "stdlib parked". Do not fix or delete them. They are the starting regression list for the stdlib redesign. (`--run-stdlib` runs them.)
- [x] Add markers for tests that need `clang`. A missing tool is a skip, not a failure. (No CUDA-specific marker was needed: the CUDA-named tests fail for other reasons, see above.)
- [x] Move `examples/08_modules_stdlib.mv` to `examples/_parked/`, so the example set is stdlib-free. Turning the remaining examples into golden tests is still open (Phase 1 and §0).
- [x] Get to 0 failures.
- [ ] Set up CI on Linux, macOS and Windows. `.github/workflows/ci.yml` is written but **has never run**. The first run on Linux and macOS may expose problems the Windows-only local runs hid (the repository stores CRLF line endings, `.exe` fixtures, path handling). Treat its first results as findings.
- [x] Untrack the build output that was committed despite `.gitignore` (19 files: `__pycache__/*.pyc`, `tests/e2e/fixtures/**/.manv/target/*.exe|*.o`).
- [x] Add a `CHANGELOG.md` and a short `CONTRIBUTING.md` (how to run tests, where the grammar lives).

## Phase 1 — Interpreter correctness bugs

Things that parse and run but give a wrong answer. These are the highest priority because they are silently wrong.

> **Where to fix these.** Items about *what the language means* (output text, escapes, exceptions, integer behavior) get a conformance test with a hand-written `.expected` file first (Phase 7). The fix itself belongs in the C engine, not the Python tree-walker, which is going to be replaced (Decision 11). Lexer-level and semantic-pass fixes are shared by both and are done in Python as usual.

- [ ] **`print` of booleans and none.** `print(true)` prints `True` and `print(none)` prints `None`. Define the canonical text (`true`, `false`, `none`) and apply it everywhere a value is rendered (`print`, `str()`, `repr`, collection printing).
- [ ] **`print(obj)` ignores `__repr__`.** It dumps the internal `InstanceObject(type_obj=TypeObject(...))`. Honor `__repr__` (and `__str__` if adopted), with a sane default such as `<Point>`. Never leak interpreter internals.
- [ ] **String escapes.** `"a\nb"` yields `anb`. Implement `\n \t \r \\ \" \' \0 \xNN \uNNNN` in the lexer, and error on unknown escapes (`E0006`).
- [ ] **`zip`.** It takes `zip([a, b])` (one array of arrays) and returns the wrong shape (`[['x','y','z']]`). Decide the signature (`zip(a, b)` is the obvious one), fix it and fix example 05.
- [ ] **Fixed-width integers.** `i32` does not wrap or trap (`2147483647 + 1` gives `2147483648`). Pick wrap vs. trap vs. checked per type, and implement it. This matters later for compiler parity.
- [ ] **`/` on two ints gives a float.** Intentional or not, decide it ([§Decisions](#decisions-needed-from-you)). Adding an integer-division operator or function is part of the answer.
- [ ] **Exception values.** `except ValueError as e: print(e)` prints an internal object. Support `e.message`, `str(e)`, `e.type`, and a traceback-ish location. Define the exception hierarchy (`Exception` → `ValueError`, `IndexError`, `TypeError`, `RuntimeError`, `OverflowError`, `AttributeError`, `KeyError`, `ZeroDivisionError`, ...).
- [ ] **Division by zero, bad index, missing key**: each should raise a catchable, typed exception with a precise message. Check each.
- [ ] **Immutability is not enforced anywhere today** (`let x = 1; x = 2` is accepted, and `let mut` is parsed then ignored). This is closed by `const` in [Phase 3a](#phase-3a--declarations-remove-let-add-const).
- [ ] **Uncaught exception output.** Currently `error[E3900]: ValueError: boom`. Add the source line, a call stack, and exit code 1.
- [ ] Audit `None` vs. `none` vs. `void` return behavior (what does `print(f())` print for a function with no return?).
- [ ] Fix the stale comments in `examples/` that describe behavior that is no longer true or never was (02, 04, 05, 06, 07). Each should match the golden output.
- [x] **`while` had a hard cap of 1,000,000 iterations** (`RuntimeError: loop iteration guard exceeded`), so a legitimate long loop failed. Removed; `conformance/control_flow/while_beyond_one_million` covers it. The same arbitrary cap still exists for user-defined iterators (`iterator guard exceeded`).
- [ ] **`and` / `or` do not short-circuit.** `false and f()` and `true or f()` still call `f()`. Case: `conformance/basics/short_circuit` (xfail).
- [ ] **Output printed before an uncaught error is lost.** The CLI only prints the program's output when the run succeeds, so `print("before"); raise ...` shows nothing. Case: `conformance/errors/output_before_uncaught_exception` (xfail).
- [ ] **`array_append` mutates its argument**, although `examples/05_collections.mv` documents it as returning a new array and leaving the original unchanged. Decide the semantics (Decision 9), then make the code, the example and [syntax.md](syntax.md) agree. No conformance case pins it until then.
- [ ] `for x in a..b` builds the whole range as a list before looping (memory, and slow for large ranges). Fix in the engine, not the tree-walker.

## Phase 2 — Lexer

- [ ] Number literals: hex `0xFF`, octal `0o17`, binary `0b101`, digit separators `1_000`, float exponent `1e-3`, and a leading-dot or trailing-dot policy.
- [ ] Multi-line **bracketed** expressions: arrays, maps and call arguments continue across lines inside `( ) [ ] { }` (suppress NEWLINE/INDENT while nesting depth > 0). This is the single most-felt gap in real code.
- [ ] Explicit line continuation with `\` (if wanted).
- [ ] Multi-line (triple-quoted) strings as ordinary expressions, not just docstrings (`E0005` today).
- [ ] Raw strings (`r"..."`) and string interpolation, per the decision (f-string, `"${x}"`, or none).
- [ ] Operators: bitwise `& | ^ ~ << >>`, exponent `**`, integer division, and decide the `&&`/`||`/`!` aliases vs. `and`/`or`/`not` only.
- [ ] Unicode: the lexer accepts any Unicode letter in identifiers (it uses `str.isalpha`) without that being a decision. Decide the policy (allowed identifier characters, normalization) and the source encoding (UTF-8), and test non-ASCII text in strings.
- [ ] Windows line endings (`\r\n`) in the indentation logic. Add a test.
- [ ] Golden tests for diagnostic spans (line, column, end column).
- [x] A line indented with only tabs was reported as `E0004 unexpected character '\t'`; it is now `E0001 tabs are not allowed for indentation`, and a line containing only whitespace and tabs is treated as blank. (`conformance/errors/tab_indentation`)

## Phase 3 — Parser and semantic analysis

The target: nothing that parses is rejected later with a generic `E2099`.

- [ ] **Augmented assignment** `+= -= *= /= %=` on names, attributes and elements. The parser already supports it. Add semantics and interpreter support.
- [ ] **`match` / `case`** with literal, wildcard `_`, binding, guard (`case n if n > 0`), and later tuple/array/type patterns. Parser and HLIR lowering exist; the semantic pass and interpreter need the work.
- [ ] **`elif`** (pending the decision). Cheap to add and removes a lot of nesting.
- [ ] **Nested functions** and local helpers (`E2099` today). Decide capture semantics, and make named inner functions and lambdas consistent.
- [ ] **Single-expression lambdas** (`fn(x): x * 2`), pending the decision.
- [ ] **`with` statement** and a context-manager protocol (`__enter__` / `__exit__`), or remove `with` from the grammar.
- [ ] **Conditional expression** (`a if cond else b`).
- [ ] **Tuples** and destructuring: `a, b = f()`, `for k, v in ...`, multiple return values.
- [ ] **Slices** `a[1..3]`, `s[1..]`, plus string indexing and negative-index rules, for strings, arrays and ranges.
- [ ] **Ranges as values**: stepped `0..10 step 2` (or `range(0, 10, 2)`), inclusive vs. exclusive, and ranges stored in variables.
- [ ] **Default and keyword arguments** at call sites, variadics (`*args`), and argument-count errors with a clear message.
- [ ] **Top-level statements / script mode.** `print("hi")` in a file with no `main` currently fails with `E3001`. Decide and implement (a script entry point that is also valid for the compiler).
- [ ] **Structured type annotations.** Annotations are stored as raw strings. Parse them into a type AST (`array[int]`, `map[str, int]`, `f32[]`, function types, optional types) and check them.
- [ ] Generic containers: `array[int]`, `map[str, int]`, with checking where annotated.
- [ ] **Definite assignment** and **unreachable code** diagnostics.
- [ ] Macros: either define expansion semantics (hygiene, argument forms, where they can appear) or cut `macro` down to what is real.
- [ ] Parse-error recovery: report more than the first syntax error per file.
- [ ] Review every `E2099` raise site and turn each into either an implementation or a specific, named diagnostic.

## Phase 3a — Declarations: remove `let`, add `const`

Spec: [syntax.md §5.1](syntax.md#51-declarations). Do this **before** the rest of Phase 3 and before the C engine ([Phase 5](#phase-5--the-c-engine-fast-interpreter)) is built, so the engine never learns about `let`.

Footprint today: 23 `let` uses in `examples/`, the e2e fixtures, 22 sites in the compiler (`LetStmt`, `TypeAttrDecl`, `"let"`), and the tree-sitter grammar in `tree-sitter-manv/`. The roughly 384 uses in `stdlib/src` (and `libc/src/main.mv`, which is treated as library code) are **deliberately out of scope**: the stdlib is parked and will be rewritten in the new syntax rather than migrated.

**Front end**
- [ ] Lexer/tokens: add `const` as a keyword. Keep `let` reserved, so using it raises a targeted migration error (`E1015: 'let' was removed — write 'x = ...' or 'T x = ...'; use 'const' for constants`). Drop the `mut` special case.
- [ ] Parser: allow the type prefix for **every** type. The `C_DECL_TYPES` allow-list in `parser.py` goes away. Disambiguate `IDENT IDENT` (typed declaration) from an expression statement. Support `i64`, `ptr`, user types and `f32[]`-style annotated types as prefixes.
- [ ] Parser: `const [T] NAME = expr` at file level, in function bodies, and in type bodies. The value is required (`E1016`).
- [ ] Parser: type-body fields in the form `int x = 0` / `x = 0` instead of `let x: int = 0`. Update `_parse_type_block` and `TypeAttrDecl`.
- [ ] AST: replace `LetStmt` with a single declaration node carrying `declared_type`, `is_const` and `is_inferred`. Do not keep both.

**Semantics**
- [ ] Inferred declaration: `name = value` declares when `name` is unbound and assigns when it is bound. Define the scope it lands in (Decision 2b).
- [ ] Reject redeclaration in the same scope (`int x = 1` twice) with a clear error.
- [ ] Constant enforcement: reject `=`, augmented assignment, and re-declaration of a `const` name. Reject index and attribute assignment whose root is a `const` name (Decision 2a). Give each its own error code.
- [ ] Reject use-before-declaration and `const` without a value.
- [ ] Constants are compile-time known where possible. Record that in the AST so the compiler can fold them and GPU kernels can capture them later.
- [ ] Type-checking of typed declarations for the newly allowed types (`i64`, `ptr`, user types) with the same `E2002` mismatch behavior as `int`.

**Interpreter / runtime**
- [ ] Bind constants in the environment with a read-only flag, as a second line of defence. Static checks must catch everything first.
- [ ] Module-level constants: evaluate once, in dependency order, and make them importable with `include`.

**Migration**
- [ ] Write a mechanical rewriter (`manv fmt --migrate-let`, or a script) with these rules: `let x: T = v` becomes `T x = v`; `let x = v` becomes `x = v`; `let mut` becomes `x = v`; type-body `let` fields become typed fields. Run it over `examples/`, the e2e fixtures and any `.mv` embedded in Python tests. Keep the tool: it is the template for every future syntax migration (see [Parked: the standard library](#parked-the-standard-library)).
- [ ] Identify bindings that were *meant* to be constant (never reassigned after `let`) and convert them to `const` by hand or with the rewriter's `--infer-const` option.
- [ ] Update `tree-sitter-manv/grammar.js` (and regenerate `src/`) and the LSP completions/semantic tokens: remove `let`, add `const`.
- [ ] Update the `compile_mut_unsupported` fixture: its premise changes with `mut`.
- [ ] Update `README.md`, the `examples/` header comments, and the website docs.

**Tests**
- [ ] Cases for each form (typed, inferred, const, typed const, each of the newly allowed types).
- [ ] Negative cases: reassigning a const, mutating through a const name, const without a value, redeclaration, use of removed `let` (checks the migration message).
- [ ] The whole existing suite must pass after the migration. Output must be byte-identical, since this is a pure syntax change.

---

## Phase 4 — Interpreter runtime and builtins

With no stdlib, **the builtins are the standard library**. Everything in this phase is implemented natively in the C engine, and the list below is the complete core surface until the stdlib redesign. Keep it small and deliberate: whatever lands here is hard to take back out later.

- [ ] **Strings:** indexing, slicing, `len` in characters (decide bytes vs. code points), `+`, `*`, comparison, `in`, and methods or functions: `upper`, `lower`, `strip`, `split`, `join`, `replace`, `find`, `startswith`, `endswith`, `format`, `ord`, `chr`.
- [ ] **Arrays:** methods or functions for append (mutating), insert, remove, pop, extend, reverse, sort, index, contains, copy, and clear. Settle value-vs-reference semantics and document them. Right now `array_append` is pure and `a[i] = v` mutates.
- [ ] **Maps:** iterate keys, values and items, `in`/contains, delete, `get` with a default, and `len`. A deterministic iteration order (insertion order is the usual answer).
- [ ] **Equality, hashing and ordering** for all built-in types, and user-defined `__eq__`, `__hash__`, `__lt__`. Allow maps keyed by more than strings.
- [ ] **Operator overloading** protocol for user types (`__add__`, `__index__`, `__len__`, `__iter__`, ...), so `for x in my_collection` works.
- [ ] **Type conversions**: define and test the full matrix (`int("x")`, `float("1e3")`, `bool("")`, `str(array)`, ...). Raise `ValueError` consistently.
- [ ] **Numerics:** float formatting (shortest round-trip), `inf`/`nan`, integer size (arbitrary precision vs. 64-bit), and `math` coverage.
- [ ] **Inheritance details:** `super`, method resolution order with multiple bases, class-level constants, `isinstance` against base types, and abstract-ish methods.
- [ ] **Closures:** captured variable semantics (by reference vs. by value) and loop-variable capture. Add tests for each.
- [ ] **Recursion limit** with a clean error rather than a Python `RecursionError`. Tail-call policy.
- [ ] **Write down the final builtin list** (name, signature, semantics, error behavior) as a table in the language reference, and mark each builtin "core" or "may move to the stdlib later". Today the set is spread across `print`, `len`, `sum`, `min`, `max`, `sorted`, `enumerate`, `zip`, `array_append`, `map_has_key`, `map_keys`, `map_values`, `type`, `isinstance`, `help` and the `ptr_*` / `c_str` family.
- [ ] Performance work does **not** happen here. Do not tune the Python tree-walker. See Phase 5.
- [ ] **`input()` / stdin**, `args` / `argv`, `exit(code)`, and environment access.

## Phase 5 — The C engine (fast interpreter)

### Why this is a rewrite of the core, not a set of bindings

The slowness is not in a few leaf operations that C could take over. It is in the structure. The current interpreter walks the AST in Python, so every node costs several Python calls, dictionary lookups and allocations. That is where the ~2.2 µs per loop iteration goes. Calling C from that loop for `a + b` would still pay the Python dispatch around it, and the boundary crossing would add to the cost.

To be fast, **the evaluation loop itself must be C**: the host hands over a program once and never sees an individual operation. Python then stays in the toolchain (CLI, LSP, front end) and out of execution.

```
Source ──(Python front end: lex, parse, semantics)──► AST ──► HLIR
                                                              │
                                  (new) HLIR → bytecode lowering, in Python
                                                              ▼
                                            .mvc bytecode file / in-memory image
                                                              ▼
                                      C engine: loader + register VM + runtime
```

The front end and the HLIR already exist and are reused. The new work is the bytecode format, the lowering and the C engine. The Python tree-walker (`interpreter.py`) is kept as a **reference oracle** until the engine passes the whole conformance suite, then retired (Decision 11).

The README's "no foreign-language runtime delegation" policy is about the standard library not wrapping a host language's libraries. A C engine is the implementation of ManV's own runtime, so it does not conflict. Write that distinction into the policy test (`tests/policy/`) so the two don't get confused later.

### Targets

These are goals to measure against, not predictions. No engine code exists yet.

| Milestone | Target | Why |
|-----------|--------|-----|
| **M0** | Startup under 20 ms for hello world. | A standalone executable removes the 0.65 s Python start. For short scripts, startup *is* the speed. |
| **M1** | At or better than CPython 3.13 on the benchmark set below. | That is already ~20× faster than today and is the floor for calling the interpreter fast. |
| **M2** | Roughly 3–5× faster than CPython on typed numeric code. | Well-built register VMs (Lua is the usual comparison) land roughly in this range. Going beyond it means a JIT, which is out of scope. Treat M2 as the stretch goal and re-state it once M1 is measured. |

The lever that can make ManV unusual among interpreted languages is its **optional static types**. `int i = 0` is a promise the front end can check, so the engine can emit specialized instructions (`ADD_INT`, `LT_INT`, `INDEX_F32`) with no tag checks, while untyped code takes the generic path. This only works if annotations are *enforced*, not advisory: an `int` parameter must be verified at the call boundary, or the specialized code is unsound. That makes Decision 5 (integer model) a prerequisite.

### Work items

**Foundations**
- [ ] Settle Decisions 5, 8, 9, 11, 12 and 13 first. They fix the value representation and the memory model, and changing them later means rewriting the engine.
- [ ] Repository layout: `engine/` (C11, no GNU-only syntax except behind `#ifdef`, such as computed goto), a `Makefile` or CMake build, and a thin Python wrapper (`ctypes` or a CPython extension) used by the REPL, tests and tooling. The toolchain on this machine is `gcc` 15.2 (w64devkit). CI builds with gcc and clang on Linux and macOS and adds MSVC on Windows later, so avoid compiler-specific extensions where a portable form exists.
- [x] **Benchmark harness first**, before any engine code, so every change is measured. Done in [benchmarks/](benchmarks/README.md): 11 programs (startup, loop, fib, nested loops, array fill/sum, map ops, string building, method calls, closure calls, exceptions, mandelbrot), each in ManV and CPython with output cross-checked, and recorded baselines. Still to add: n-body and spectral-norm, and Lua as a third reference.
- [ ] Warnings and sanitizer policy: `-Wall -Wextra -Werror`, UBSan/ASan builds in Linux CI, and a fuzzer for the bytecode loader (a corrupt `.mvc` file must produce an error, never a crash).

**Bytecode**
- [ ] Specify the instruction set and file format (`docs/bytecode.md`): version header, constant pool, function table, line table (for diagnostics and debugging), exception-handler table, type table.
- [ ] **Register-based** instructions in three-address form (fewer dispatches than a stack machine).
- [ ] HLIR → bytecode lowering in Python: register allocation, constant pooling, jump patching, `const` folding (Phase 3a supplies the facts it needs), line-table emission.
- [ ] A disassembler (`manv compile --emit bytecode`) and a bytecode verifier in the loader.

**Runtime core**
- [ ] **Value representation:** 64-bit tagged values (NaN-boxing or pointer tagging) with unboxed `int`, `float`, `bool` and `none`. Heap objects carry a small header (type, GC bits).
- [ ] **Dispatch:** `switch` as the portable baseline, computed goto behind `#ifdef __GNUC__`. Add superinstructions only for patterns the benchmarks show are hot.
- [ ] **Calls:** one contiguous value/frame stack, no heap allocation per call, closures via upvalues, argument-count checking, and a recursion limit that raises a catchable error.
- [ ] **Memory management:** refcounting vs. tracing GC (Decision 12). The `Heap` in Python today is only a simulation. Keep the deterministic-GC and GC-stress test modes, since they are valuable, and provide equivalents.
- [ ] **Strings:** immutable, UTF-8 (per Decision 8), with cached hashes and interned short strings.
- [ ] **Arrays:** contiguous storage. Typed arrays (`f32[]`, `int[]`) store raw elements, which also makes later GPU transfer zero-copy.
- [ ] **Maps:** open-addressing hash table with insertion-order iteration.
- [ ] **User types:** field slots resolved at load time where the type is known, with inline caches for method lookup on dynamic receivers. Inheritance and MRO as in Phase 4.
- [ ] **Exceptions:** a handler table per function, typed matching, `finally`, re-raise, and a stack trace built from the line table.
- [ ] **Specialization:** typed opcodes first (the static-type lever above). Add adaptive rewriting of hot generic instructions only if the benchmarks justify the complexity.
- [ ] **Native builtins** from Phase 4, with a stable internal C API for registering them.
- [ ] **`extern "c"` FFI** in the engine: load a shared library (`LoadLibrary`/`dlopen`), resolve symbols, and call through a small set of signature-specific thunks (or libffi). Pointers are raw, unmanaged values.
- [ ] **`syscall`** and `__intrin` mapped onto engine primitives, with the intrinsic list re-derived from what the engine actually needs.

**Tooling integration**
- [ ] `manv run` uses the engine by default. `--engine=python` keeps the oracle available while it exists.
- [ ] The REPL runs on the engine (incremental compile and run, with persistent globals).
- [ ] **Debugging:** DAP and `debug_engine.py` hook the Python interpreter today. Give the engine a debug interface (line breakpoints via the line table, single-step, frame and variable inspection, exception breakpoints) and re-point the DAP server at it.
- [ ] Profiling hooks (`--profile`): instruction counts and time per function.
- [ ] Distribution: a single `manv-vm` executable plus the Python toolchain. A release that needs no Python at all (the front end ported into the engine, or self-hosted in ManV) is a later project. Note it, don't start it.

**Verification**
- [ ] **Differential testing:** run every conformance case and every fuzz-generated program through both the oracle and the engine, and diff stdout, exit code and diagnostic code. Any difference is a bug in one of them, to be resolved against the *spec*, not against either implementation.
- [ ] Hand-written `.expected` files are the authority (Phase 7). Never generate them from the engine under test.
- [ ] A performance regression gate in CI: fail on a slowdown beyond a threshold against the stored baseline.

## Phase 5b — User modules (no stdlib)

`include` is a language feature and stays, for the user's own code. The standard library is not involved.

- [ ] `include` semantics: caching, cycles (clear error), relative includes, name collisions, and what `include X from Y` does when `Y` has a side effect.
- [ ] A defined **visibility** rule (what a module exports; private names).
- [ ] Test modules live under `tests/` (tiny, purpose-written), not in `stdlib/`.
- [ ] `manv init` should scaffold a runnable project that passes `manv test`, with no stdlib dependency.
- [ ] `manv add` / the registry are operational per the README. Leave dependency locking and signing for after the compiler.

## Parked: the standard library

**Do nothing here until the interpreter works and the syntax is settled.** Concretely, until then:

- No edits, fixes, migrations or new modules under `stdlib/` or `libc/`.
- No new feature may depend on the stdlib, and the interpreter must not load it.
- The `std.*` namespace, the "restricted to std sources" intrinsic rule and the versioned-intrinsic scheme are on hold. They are not specified in [syntax.md](syntax.md) for now.
- The existing 23 modules (`assert`, `builtins`, `collections`, `error`, `fmt`, `io`, `iter`, `json`, `libc`, `log`, `main`, `math`, `net`, `option`, `os`, `path`, `pattern`, `random`, `result`, `sort`, `str`, `sys`, `time`) are reference material for the redesign, not something to stay compatible with.

When the time comes, treat it as its own project with these constraints, so that a future language change does not mean editing hundreds of files:

- [ ] **Syntax freeze before authoring.** Write no library code until the grammar is stable (the end of Phase 3 and the Phase 8 reference).
- [ ] **Migration tooling comes first.** The `manv fmt --migrate-*` rewriter from Phase 3a is the model. Any syntax change ships with a rewrite rule that runs over the whole stdlib automatically.
- [ ] **An edition or version marker** in `project.toml` (and a language version in the reference), so old library code keeps compiling while the language moves.
- [ ] **Small and layered:** a minimal `core` that the engine depends on, with everything else (`json`, `net`, `fmt`, ...) as separate packages that version independently. A change to one package must not touch the others.
- [ ] **Decide the intrinsic boundary** (the narrow-intrinsics policy in the README) after the engine exists and its real primitive set is known.
- [ ] Decide which Phase 4 builtins move out of the core and into the library.
- [ ] A compatibility test suite per package, run in CI.

## Phase 6 — Diagnostics and developer experience

- [ ] A diagnostics style guide: one format, stable codes, "help:" lines with fixes. Document each `E####` code in `docs/errors.md`.
- [ ] Source-line context and caret underlines for **runtime** errors, not just compile-time ones.
- [ ] A stack trace for runtime errors (function names with file:line).
- [ ] REPL: multi-line input, history, persistent state across lines, `:help`, `:type`, and `:load`. It already exists, so audit it against the final grammar.
- [ ] A formatter (`manv fmt`). Layout rules are strict enough that this is tractable and valuable.
- [ ] A linter (unused variable, shadowing, unreachable code). With `name = value` now declaring, it must also flag a variable that is assigned but never read, which is the usual symptom of a misspelled name.
- [ ] `manv test` for ManV code: `assert`, test discovery (`test_*` functions), and exit codes.
- [ ] Editor support: the tree-sitter grammar and the LSP live in `tree-sitter-manv/` and `manv lsp`. Keep both in step with the grammar once Phases 2–3 land, so there is a single grammar source of truth or a test that cross-checks them.
- [ ] DAP debugger: re-verify against the final interpreter (breakpoints, stepping, variables). `test_dap_reports_gpu_execution…` currently fails.

## Phase 7 — Conformance suite (do this early, grow it throughout)

- [x] `tests/conformance/**/*.mv` with sibling `.expected` / `.error` / `.exit` / `.xfail` files, one feature per file, expected output written by hand (see its README). 55 cases so far: the rest pass today, 14 are strict xfails that mark known gaps. This suite is what the engines will be held to.
- [ ] A harness that runs every conformance case through the interpreter (done: `tests/test_conformance.py`). Still to do: also run them through the C engine and compiled mode and diff the results.
- [x] A negative-test set: programs that must fail with a specific diagnostic code (`conformance/errors/`, 16 cases). Locations (line/column) are not asserted yet.
- [ ] A test that extracts and runs the fenced `manv` blocks in `syntax.md`.
- [ ] Fuzz the lexer and parser (random bytes must never crash with a Python exception; only `ManvError`).
- [ ] Property tests for arithmetic, conversion and collection semantics.

## Phase 8 — Specification and docs

- [ ] Turn `syntax.md` into a versioned language reference with a formal grammar (EBNF), kept in sync with the parser by a test.
- [ ] A **semantics document** covering evaluation order, scoping, value vs. reference, exceptions, integer behavior and module loading. Without it the compiler has nothing to match.
- [ ] A tutorial (hello world → a real program) based on the cleaned-up `examples/`.
- [ ] Language versioning policy: what `v0.1` promises and what may still break.
- [ ] Update `README.md` ("Current Status") and the website docs to match reality, and trim claims that aren't true yet.

---

## Decisions needed from you

These shape the language, so they are yours to make. Each blocks specific items above. A recommendation is given where there is a clear default.

| # | Decision | Recommendation | Blocks |
|---|----------|----------------|--------|
| 1 | `/` on two `int`s: always float (today), or integer division with a separate `/` for float? | Keep `/` true division, add `//` for integers. It is the Python precedent, and the language already looks Python-shaped. | P1, P2, P4 |
| ~~2~~ | ~~Mutability~~ **Decided:** `let` is removed. `T name = v` and `name = v` are mutable declarations. `const` is the one immutable form. See 2a and 2b for what is left open. | n/a | n/a |
| 2a | **How immutable is a `const`?** (a) only the name can't be rebound, `ITEMS[0] = 1` still works, or (b) the name can't be rebound **and** it can't be mutated through (index/attribute assignment is rejected). | (b), as a static check on the name. It matches what `const` promises and it needs no runtime cost. It cannot see aliases (`x = ITEMS; x[0] = 1`), so true deep immutability would need frozen values or the ownership model later. Document that limit. [syntax.md §5.1](syntax.md#51-declarations) currently assumes (b). | P3a |
| 2b | **Where does `name = v` declare, and what is the scope?** (a) function-wide like Python, (b) block-scoped like C. Also: may an inner scope shadow an outer name, or does assignment always reach the outer one? | Block scope, and assignment to an already-bound outer name assigns to it (no accidental shadowing). It is stricter, it matches the typed `int x = 0` form that already exists, and it keeps names from leaking out of `if`/`for` bodies. Also decide whether to warn on the typo hazard (`totl = total + 1`). | P3a |
| 3 | `elif` or only nested `else: if`? | Add `elif`. Nesting 4 deep for a simple chain works against readability. | P3 |
| 4 | String interpolation: `f"x={x}"`, `"${x}"`, or a native `format` builtin only? | f-strings. | P2, P4 |
| 5 | Integer model: arbitrary precision, 64-bit, or per-type fixed width with wrap/trap? | Per-type fixed width, with `int` as a 64-bit trap-on-overflow default. This keeps engine and compiler identical, and trap-on-overflow is cheap in C (`__builtin_add_overflow`). Arbitrary precision would make every `int` operation a possible allocation and undo the typed-code speedups. | P1, P4, P5, §9 |
| 6 | Script mode: may a file be just statements with no `main`? | Yes for `manv run`, with the semantics "implicit main", so `manv run x.mv` works for small programs. | P3 |
| 7 | Single-expression lambdas: `fn(x): x * 2`? | Yes. Block-only lambdas make `map`/`filter` noisy. | P3 |
| 8 | `String` model: bytes or Unicode code points for `len`/index? | Code points, stored as UTF-8, with a separate `bytes` type for raw data. Indexing by code point is O(n) on UTF-8, so the engine needs an ASCII-only fast path (a flag on the string) to keep it cheap. | P4, P5 |
| 9 | Value vs. reference semantics for `array`/`map`/`type` instances (aliasing on assignment/pass)? A concrete symptom today: `array_append(a, x)` mutates `a`, while `examples/05` documents it as pure. | Reference semantics for heap types, written down plainly, with an explicit `copy()`. This needs to be settled before the ownership model. | P4, P5, §9 |
| 10 | Keep or cut: `class` alias, `True`/`False` aliases, `&&`/`||`/`!` aliases, the `macro` stub, `gpu`/`memory` statements? | Cut the `True`/`False` and `&&`/`||`/`!` aliases (one way to do it). Keep `class` only if it is needed for migration. Defer macros. | P2, P3 |
| 11 | **Where do runtime fixes go?** Phases 1 and 4 are mostly runtime behavior. Fix them in the Python tree-walker now (a quick path to a "working" interpreter, throwaway work later), or write them once in the C engine and keep the tree-walker frozen as an oracle? | Write them once, in the engine. The tree-walker is ~20× slower than CPython and is going away, so effort spent there is spent twice. The cost is a longer wait before the interpreter feels "working". Cap it by building a thin vertical slice first (ints, floats, strings, functions, `if`/`while`/`for`, `print`) and growing from there. Lexer, parser and semantic fixes are shared and stay in Python. | P1, P4, P5 |
| 12 | **Memory management in the engine:** reference counting, tracing GC, or arenas/ownership from the start? | Start with a simple precise tracing GC (mark-sweep). It handles cycles, keeps the value representation simple, and can later be replaced behind the allocator interface. Reference counting costs a write on every copy, which works against the speed goal. Revisit when the ownership model (§9.4) is designed. | P5, §9.4 |
| 13 | **How is the engine hosted?** (a) a standalone `manv-vm` executable that the Python tool launches or that runs `.mvc` files on its own, (b) a CPython extension module, (c) both. | (c), with the executable first. It is the only way to get the startup target (M0), and it is what makes shipping without Python possible later. The extension or `ctypes` wrapper is for tests and the REPL only. | P5 |

---

## 9. After the interpreter: the compiler

**Do not start until the §0 exit criteria are met.** The pipeline already exists in rough form (`Source → AST → HLIR → Graph IR → Kernel IR → Backend`); this phase is about making it *correct*, not about adding backends.

### 9.1 Parity first

- [ ] Run the full conformance suite through compiled mode and diff the output against the interpreter. Any divergence is a compiler bug (this rule is in the README).
- [ ] Decide the compiled artifact for v1: a native executable via LLVM (`clang` required), or an interpreter-embedded bundle first as a stepping stone.
- [ ] HLIR stays the single semantic authority. The C engine's bytecode and the compiler's LLVM IR are both lowered from the same HLIR, so the engine, not a second tree-walker, is the thing the compiled output is diffed against. Retire `hlir_interpreter.py` and `interpreter.py` once the engine passes the conformance suite.
- [ ] Reuse engine work in the compiler: the runtime library for compiled programs (strings, maps, GC, exceptions) should be the *same C code* as the engine's runtime core, not a second implementation.
- [ ] `manv compile --emit ast,hir,hlir` produce stable, documented output (golden-tested).

### 9.2 CPU codegen

- [ ] Finish HLIR → LLVM IR lowering (arrays, maps, strings, exceptions, closures, user types). `test_llvm_lowering_supports_arrays_maps_and_core_len` shows where it stops.
- [ ] A runtime library for the compiled path: allocation and GC (or the ownership model), strings, maps, exceptions (unwinding), panic output.
- [ ] Linking and cross-compilation targets (Windows, Linux, macOS; x86_64, aarch64). Remove the old assembly codegen once LLVM reaches parity.
- [ ] Optimization levels (`-O0`/`-O2`) and debug info (DWARF/PDB) so the DAP debugger works on compiled code.

### 9.3 GPU

- [ ] Region eligibility rules written down and tested (what is allowed in `@gpu`).
- [ ] Kernel IR → PTX / SPIR-V / WGSL, validated on real hardware per backend; the mock tests are not enough.
- [ ] Host-side: memory transfer, launch, synchronization, and error reporting. A guaranteed fallback to CPU with identical results (within a documented float tolerance).
- [ ] GPU library: map/reduce/scan, device buffers and profiling (after the stdlib redesign).

### 9.4 Memory model

- [ ] Ownership, move and borrow semantics: spec first (syntax.md §12), then checker, then codegen.
- [ ] Arenas and pools, and address-space annotations (`global`/`shared`/`local`).

### 9.5 Packaging and release

- [ ] Reproducible dependency resolution and lockfiles, source distributions, signed packages, scoped names.
- [ ] Installers/releases per platform, versioned toolchain, a documented stability policy.
- [ ] Performance targets from the README (memory-bandwidth-bound elementwise and reduction kernels), with benchmarks tracked in CI.

---

## Suggested order of attack

1. **Phase 0**: a green baseline with the stdlib cut out and its tests skipped.
2. **The conformance harness** (the start of Phase 7) and the **benchmark harness** (Phase 5, Foundations). Do these before changing anything, so every later fix has a test and every speedup has a number.
3. **Answer the decisions that fix the design:** 1, 2a, 2b, 5, 8, 9, 11, 12 and 13. Most of the cost of getting these wrong is in the engine.
4. **Phase 3a** (`let` out, `const` in), then **Phase 2 and Phase 3** (multi-line brackets, escapes, augmented assignment, `match`, nested functions). These are all front-end work. They are shared by every engine and unlock real programs. Finish them before the engine so it is built against the final grammar.
5. **Phase 5, the C engine**, as a vertical slice first (ints, floats, strings, functions, control flow, `print`) behind the same conformance suite, then grow it feature by feature. **Phases 1 and 4** are implemented here, in C (Decision 11).
6. **Phase 5b, 6 and 8** (modules, diagnostics, REPL, formatter, docs) in parallel with the previous steps whenever convenient.
7. Check the **§0 exit criteria**, including engine milestone M1. Then start **§9**, the compiler.
8. **Parked: the standard library** is last, and only after step 7 and the syntax freeze.
