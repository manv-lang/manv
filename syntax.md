# ManV Syntax and Philosophy

This document describes what ManV source looks like and why it looks that way.

Every construct is marked with its status. The statuses were checked against the lexer, the parser and the interpreter on 2026-10-26.

| Mark | Meaning |
|------|---------|
| ✅ | Parses, passes semantic analysis and runs in the interpreter. |
| ⚠️ | Parses, but semantic analysis or the interpreter rejects or mishandles it today. |
| 🧭 | Not implemented. Proposed direction, and open to change. |

> **Scope.** There is no standard library for now (it is parked until the interpreter works and the syntax is settled), so this document describes the core language and its native builtins only.
>
> **Declarations are being redesigned.** `let` is removed and `const` is added ([§5.1](#51-declarations)). This document describes the **new** design. The interpreter still implements the old one: it accepts `let` and has no `const`, so everything in §5.1 about `const`, inferred `name = value` declarations and the all-types type prefix is 🧭 until the migration in [TODO.md](TODO.md) (Phase 3a) lands. Examples in `examples/` and the stdlib still use `let` and have not been migrated.

The file `main.mv` and the programs in `examples/` are the working reference. The authoritative grammar is [manv/parser.py](manv/parser.py) and [manv/lexer.py](manv/lexer.py). When this document and those files disagree, the files win. Please fix this document.

---

## 1. Philosophy

### 1.1 Readable like a script, explicit like a systems language

ManV borrows its shape from Python: significant indentation, `:` to open a block, no braces, no semicolons. It borrows its declarations from C: `int x = 1`. The goal is code that reads quickly but never hides what a value *is* or what happens to it.

### 1.2 Explicit semantics

Nothing important should happen invisibly. Moving, copying and converting values should be visible in the source and traceable through the compiler's IR stages. Conversions are written as calls (`int(x)`, `str(x)`), not applied silently. Mismatches are rejected statically (`int x = "s"` is `E2002`).

### 1.3 One meaning, two execution engines

A program can be interpreted for fast feedback or compiled for speed. Both consume the same semantic representation (HLIR). Differences in observable behavior between the two are compiler bugs, not design differences.

**Consequence for syntax:** a feature does not exist until its meaning can be stated precisely enough for both engines to agree on it. Parsing a construct is not the same as supporting it, so this document uses ⚠️ for the gap.

### 1.4 GPU is a normal target

Marking a function with `@gpu` is ordinary decoration, not a separate dialect. The kernel-eligible subset of the language is the same language: bounded loops over arrays, arithmetic, no I/O. Ineligible code falls back to the CPU.

### 1.5 Low-level control when asked for, out of the way when not

The common case is plain values and functions. Raw pointers (`ptr`), C FFI (`extern`), system calls (`syscall`) and compiler intrinsics (`__intrin.*`) exist for the cases that need them. They are visibly different from ordinary code.

### 1.6 A small core; the library comes later

The language core plus a deliberately small set of native builtins ([§11](#11-built-in-functions)) must be enough to write real programs. **There is no standard library for now.** It is redesigned after the interpreter works and the syntax is settled (see "Parked: the standard library" in [TODO.md](TODO.md)), so a syntax change never means editing a large body of library code.

The intended long-term shape is unchanged: the compiler knows a narrow set of primitive operations, and everything else (JSON, math, paths, collections) is ordinary ManV in a library. A new subsystem is never added as a primitive.

### 1.7 Errors are part of the product

Every diagnostic has a stable code (`E0xxx` lexer, `E1xxx` parser, `E2xxx` semantic analysis, `E3xxx` driver and runtime, `E4xxx` project and files), a file, a line and column, and the offending source line. Diagnostic quality is a feature.

---

## 2. Lexical structure

### 2.1 Files and layout ✅

- Source files use the `.mv` extension (`E4001` otherwise).
- Blocks are defined by indentation. Indentation must be **spaces**. Tabs are an error (`E0001`). A dedent to a level that does not match an enclosing block is `E0002`.
- A statement ends at the end of the line. There are no semicolons.
- A block opens with `:` at the end of a header line, followed by an indented body.
- Blank lines and comment-only lines are ignored.
- Within `( )`, `[ ]` or `{ }` a line break is **not** allowed. See [§4.5](#45-collections-and-ranges).

```manv
fn main() -> int:
    if true:
        print("indented block")
    return 0
```

### 2.2 Comments and docstrings ✅

```manv
# A comment runs to the end of the line.

fn f():
    """A docstring: a triple-quoted string that is the only thing on its line(s).
    It must be the first statement of a function, type or file."""
    return
```

- `#` starts a comment.
- Triple-quoted strings (`"""` or `'''`) are accepted **only** as standalone docstrings. Using one inside an expression is `E0005`.
- A string-literal first statement in a function body, a type body or a file becomes its docstring. `help(...)` reads it.

### 2.3 Identifiers and keywords ✅

Identifiers start with a letter or `_` and continue with letters, digits or `_`. The lexer uses Python's `isalpha`, so non-ASCII letters are accepted today by accident, not by decision. Stick to ASCII.

Reserved words (they cannot be used as identifiers):

```
fn const type class impl macro module include from extern as
if else while for in break continue return
try except finally raise with match
and or not true false True False none
int i32 i64 u8 usize float f32 bool str array map ptr void
syscall gpu memory
```

Notes:

- `class` is an accepted alias for `type`.
- `True` and `False` are accepted aliases for `true` and `false`. The canonical spelling is lowercase.
- `case` is **not** reserved. It is recognised only inside a `match` block.
- `let` is **no longer a declaration keyword**, but it stays reserved so that old code gets a targeted error ("`let` was removed: write `x = ...` or `T x = ...`; use `const` for constants") instead of a confusing parse failure. `mut` has no meaning anymore.
- `gpu` and `memory` are reserved for planned statement forms ([§12](#12-reserved-and-planned-syntax)).

### 2.4 Literals

| Literal | Examples | Status |
|---------|----------|--------|
| Integer | `0`, `42`, `9223372036854` | ✅ decimal only |
| Float | `3.14`, `1.0`, `0.5` | ✅ decimal only, requires a digit before and after `.` |
| Boolean | `true`, `false` | ✅ |
| None | `none` | ✅ |
| String | `"hi"`, `'hi'` | ✅ single-line only |
| Array | `[1, 2, 3]` | ✅ one line only |
| Map | `{"a": 1, "b": 2}` | ✅ one line only |
| Hex/octal/binary, `1_000`, `1e3` | `0xFF`, `0b101`, `1_000`, `1e3` | 🧭 |
| Tuple | `(1, 2)` | 🧭 |
| f-string / interpolation | `f"x={x}"` | 🧭 |

**Strings.** Either quote character works. ⚠️ Escape sequences are **not** interpreted. A backslash only makes the next character literal and is then dropped, so `"a\nb"` is the three characters `anb`. Interpreting `\n`, `\t`, `\\`, `\"` and `\xNN` is planned.

**Number edge.** `0..5` lexes as `0`, `..`, `5` (a range), not as a float.

### 2.5 Operators and punctuation ✅

```
+  -  *  /  %                 arithmetic
== != <  <= >  >=             comparison
and or not   &&  ||  !        logic (word forms and symbol forms are interchangeable)
=                             assignment
+= -= *= /= %=                augmented assignment (see §5.2, ⚠️)
..                            range
->                            return-type arrow
@                             decorator
.   ,   :                     member access, separator, block opener
( ) [ ] { }                   grouping, call, index, collection literals
```

No bitwise (`& | ^ ~ << >>`), no exponent (`**`) and no ternary (`x if c else y`) operators exist yet 🧭.

---

## 3. Types

### 3.1 Built-in types

| Type | Meaning | Declaration prefix (`T name = ...`) |
|------|---------|--------------------------------------|
| `int` | integer | ✅ |
| `i32`, `u8`, `usize` | fixed-width integers | ✅ |
| `i64` | 64-bit integer | ✅ (🧭 today: needs `let x: i64`) |
| `float`, `f32` | floating point | ✅ |
| `bool` | boolean | ✅ |
| `str` | string | ✅ |
| `array` | ordered, growable sequence | ✅ |
| `map` | key-to-value dictionary | ✅ |
| `ptr` | raw pointer (for FFI) | ✅ (🧭 today: needs `let p: ptr`) |
| `void` | no value (return type only) | n/a |
| `none` | the single "no value" value | n/a |

User-defined types come from `type` declarations ([§8](#8-user-defined-types)) and use the same prefix form: `Point p = Point(1, 2)`. With `let` gone, the type prefix works for **every** type.

### 3.2 Annotations are strings to the parser

A type annotation is everything between `:` (or `->`) and the following `=`, `,`, `)` or end of line. Examples: `int`, `array`, `f32[]`, `array[int]`. Semantic analysis interprets that text. There is no generic-type grammar yet 🧭.

Typed arrays use a postfix `[]`: `f32[]`. The sized form `int buf[16]` declares an array of `int` with a length.

### 3.3 Typing discipline

Typing is gradual. Annotations are optional on parameters, returns and inferred declarations. Where a type is written, it is checked statically (`int x = "s"` gives `E2002: type mismatch`). Where it is omitted, the type is inferred or deferred to runtime. Implicit numeric conversion is not performed, so use `int(x)` and `float(x)`.

⚠️ Fixed-width integers do not yet wrap or trap on overflow: `i32 b = 2147483647; b = b + 1` yields `2147483648`.

---

## 4. Expressions

### 4.1 Precedence, lowest to highest ✅

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `a .. b` (range) | non-associative |
| 2 | `or`, `\|\|` | left |
| 3 | `and`, `&&` | left |
| 4 | `==`, `!=` | left |
| 5 | `<`, `<=`, `>`, `>=` | left |
| 6 | `+`, `-` | left |
| 7 | `*`, `/`, `%` | left |
| 8 | unary `-`, `!`, `not` | right |
| 9 | call `f(x)`, index `a[i]`, member `a.b` | left |

Comparisons parse left-associatively, so `1 < 2 < 3` is `(1 < 2) < 3`, not a chained comparison.

### 4.2 Arithmetic semantics

- ✅ `/` is **always true division**: `24 / 4` is `6.0`, and `7 / 2` is `3.5`, even for two `int` operands. Use `int(a / b)` to truncate. An integer-division operator (`//`) is an open decision 🧭.
- ✅ `%` takes the sign of the divisor: `-7 % 3` is `2`.
- ✅ `+` concatenates two strings. `"a" + 1` is a `TypeError`, so convert with `str(1)`.
- ✅ Strings compare lexicographically with `<`, `==` and so on.
- ⚠️ `and` / `or` do **not** short-circuit today: in `false and f()` the call to `f()` still happens. They are meant to (the right operand is evaluated only when it can change the result). Tracked in [TODO.md](TODO.md) Phase 1 and covered by `tests/conformance/basics/short_circuit`.

### 4.3 Calls, members and indexing ✅

```manv
f(1, 2)          # call
obj.field        # attribute
obj.method(x)    # method call
a[0]             # index (negative indices count from the end)
m["key"]         # map lookup
shapes.area(r)   # member of an included module
```

Argument lists cannot span lines ⚠️. Keyword arguments are accepted only in decorators.

### 4.4 Function literals ✅

A lambda is the same `fn` form without a name, and it **always has a block body**:

```manv
square = fn(x: int) -> int:
    return x * x

array evens = filter_array(numbers, fn(x: int) -> bool:
    return x % 2 == 0
)
```

The closing `)` of the enclosing call goes on the line after the block. Single-expression lambdas (`fn(x): x * 2`) do not exist 🧭. Lambdas capture variables from the enclosing scope (closures).

### 4.5 Collections and ranges

```manv
array nums = [10, 30, 20]
map person = {"name": "Alice", "age": 30}
nums[1] = 99                  # ✅ index assignment
person["email"] = "a@b.c"     # ✅ add or update key
for i in 0..5:                # ✅ range: 0,1,2,3,4 (end is exclusive)
    print(i)
```

- `a .. b` is a half-open range. It is accepted in `for ... in` ✅. Elsewhere it parses but is not a runtime value ⚠️, so `a[0..2]` fails.
- Missing 🧭: multi-line array, map and call literals, slices `a[1..3]`, stepped ranges `0..10..2`, map iteration `for k in m`, and indexing into strings `s[1]`.
- Collections are managed values. `a[i] = v` mutates in place. ⚠️ `array_append(a, x)` also mutates `a` in place (and returns it), although `examples/05_collections.mv` describes it as returning a new array. Which one is intended is an open decision (Decision 9 in [TODO.md](TODO.md)), so don't rely on either: write `a = array_append(a, x)`, which behaves the same under both.

---

## 5. Statements

### 5.1 Declarations

There is no `let`. A name comes into existence in one of three ways, and only one of them is immutable.

| Form | Meaning | Mutable |
|------|---------|---------|
| `T name = value` | typed declaration | yes |
| `name = value` | declaration with inferred type, when `name` is not already bound | yes |
| `const name = value`<br>`const T name = value` | constant | **no** |

```manv
# Typed declarations: the type is written first (C-like). Works for every type.
int count = 0
array items = []
i64 big = 9223372036854
ptr buf = malloc(64)
Point origin = Point(0, 0)

# Inferred declaration: a first assignment creates the variable.
total = 0
square = fn(x: int) -> int:
    return x * x

# Constants: bound once, never reassigned.
const MAX_RETRIES = 5
const float PI = 3.14159
```

**Typed declarations (🧭 for `i64`, `ptr`, user types; ✅ for the rest).** `T name = value` declares `name` with type `T`, and the value is checked against `T` (`int x = "s"` gives `E2002`). It works for every type in [§3.1](#31-built-in-types) and for user-defined types. Today the interpreter accepts it only for `int i32 str array map u8 usize float f32 bool`.

**Inferred declarations (🧭).** `name = value` is a *declaration* if `name` is not already bound in an enclosing scope, and an *assignment* if it is. Its type is inferred from the value. This is the replacement for `let x = ...`.

**Constants (🧭).**

- `const` requires a value. `const x` alone is an error.
- The type is optional: `const MAX = 5` and `const int MAX = 5` are both legal.
- Reassigning a constant (`MAX = 6`, `MAX += 1`) is a **compile-time** error, even in interpreter mode. Redeclaring it in the same scope is also an error.
- A constant cannot be mutated *through its name* either: `ITEMS[0] = 1`, `CONFIG["k"] = v` and `ORIGIN.x = 1` are rejected where the target is a constant. This is a static check on the name. It does not track aliases. See Decision 2a in [TODO.md](TODO.md).
- `const` is allowed at file level, inside functions and inside `type` bodies. File-level constants are the intended way to share configuration and magic numbers, and they are the natural fit for GPU kernels, since a constant is known at compile time.
- Naming convention: `UPPER_SNAKE_CASE`. This is a convention, not a rule.

**Rules common to all declarations.**

- Declaring a name twice in the same scope is an error (`int x = 1` followed by `int x = 2`). Assign instead: `x = 2`.
- A declaration without a value (`int x`) parses. Semantic analysis decides whether it is legal.
- Reading a name that was never declared is `E2010`, as today.
- Block scoping and shadowing rules are not final. See Decision 2b in [TODO.md](TODO.md).

**The typo hazard.** Because `name = value` declares, a misspelled name on the left (`totl = total + 1`) silently creates a new variable instead of failing. Keep this in view. The linter is expected to flag variables that are assigned and never read (TODO Phase 6). Typed declarations (`int total = ...`) are the explicit, typo-proof form.

**Today.** The interpreter still uses `let name`, `let name: T = value` and `let mut name`. `let mut` is accepted and `mut` ignored. `let` bindings are mutable, so `let x = 1; x = 2` is legal.

### 5.2 Assignment

```manv
x = 5            # ✅ variable (🧭 declares it if unbound, see §5.1)
obj.field = 5    # ✅ attribute
a[i] = 5         # ✅ element
x += 1           # ⚠️ parses, rejected: E2099 unsupported statement 'AugAssignStmt'
```

Augmented assignment (`+= -= *= /= %=`) parses for variables, attributes and elements, but is not yet implemented. Write `x = x + 1` until it is.

### 5.3 Conditionals ✅

```manv
if n < 0:
    print("neg")
else:
    print("non-neg")
```

There is no `elif` or `else if`. Chain by nesting an `if` in the `else` block. Whether to add `elif` is an open decision 🧭.

### 5.4 Loops ✅

```manv
while cond:
    ...
else:            # runs when the loop ends without `break`
    ...

for item in collection:   # collection: array or range
    ...
else:
    ...

break
continue
```

`for` binds a single name. Destructuring (`for k, v in ...`) does not exist 🧭. `enumerate(xs)` yields `[index, value]` arrays.

### 5.5 Pattern matching ⚠️

```manv
match x:
    case 1:
        print("one")
    case n if n > 10:
        print("big")
```

`match` and `case` parse, including guards. The interpreter's semantic pass rejects them (`E2099 unsupported statement 'MatchStmt'`). HLIR lowering already handles `match`, so the gap is in the semantic and interpreter layer.

### 5.6 Exceptions ✅

```manv
try:
    risky()
except ValueError:
    print("bad value")
except IndexError as e:     # `as e` parses; reading e.message is ⚠️ not yet supported
    raise                   # bare raise re-raises the active exception
else:
    print("no exception")   # runs only if the try body did not raise
finally:
    print("always")

raise ValueError("message")
```

- A `try` needs at least one `except` or a `finally` (`E1011`).
- `except` names exactly one type. Multiple clauses are allowed, and the first match wins.
- Built-in exception types seen so far: `ValueError`, `IndexError`, `RuntimeError`, `TypeError`, `AttributeError`, `OverflowError`.
- An uncaught exception terminates the program with an `E3900` diagnostic.

### 5.7 Other statements

| Form | Status |
|------|--------|
| `return` / `return expr` | ✅ |
| expression statement (`f(x)`) | ✅ |
| `with expr as name:` | ⚠️ parses, rejected at semantic analysis |
| nested `fn` inside a function body | ⚠️ parses, rejected (`E2099`). Declare helpers at the top level. |
| `syscall(target, args...)` as statement or expression | ✅ |
| `gpu ...`, `memory ...` | ⚠️ reserved. Parse as "unsupported statement". |

---

## 6. Functions

### 6.1 Declaration ✅

```manv
fn add(a: int, b: int) -> int:
    return a + b

fn greet(name: str):          # no return type: returns none
    print("Hello, " + name)

fn apply(f, x: int) -> int:   # untyped parameter: any value, including a function
    return f(x)
```

- Parameter types and the return type are optional.
- Functions are first-class values: they can be passed, returned, stored in arrays and called.
- No default arguments, keyword arguments at call sites, `*args` or overloading yet 🧭.
- Recursion works.

### 6.2 Entry point ✅

A runnable program defines `fn main() -> int`. The return value is the process exit code (`0` means success). A file with no `main` fails with `E3001`.

⚠️ Top-level statements outside functions parse, but a file with them and no `main` is rejected. A "script mode" is an open decision 🧭.

### 6.3 Decorators ✅

```manv
@gpu
fn add(a: f32[], b: f32[], out: f32[]) -> void:
    for i in 0..len(out):
        out[i] = a[i] + b[i]

@gpu(required=True, mode="kernel")
fn scale(...): ...
```

- A decorator is `@name` or `@name(args, key=value)` on its own line directly above a `fn`.
- Decorators apply only to functions and methods. Anywhere else is `E1012`.
- `@gpu` options seen in tests: `required` and `mode` (`"kernel"`, `"graph"`).
- Method decorators: `@staticmethod`, `@getter`, `@setter` and `@property`.

---

## 7. Modules and includes

### 7.1 `module` ✅

```manv
module pkg.sub.leaf
```

Optional, at the top of a file. It names the file's module (dotted path).

### 7.2 `include` ✅

```manv
include shapes                  # whole module, bound as `shapes`
include shapes as sh            # whole module, aliased
include pkg.sub                 # dotted path, bound as `sub`
include area, scale from shapes # selected names
include area as a from shapes   # selected name, aliased
include Foo from .sibling       # relative include (leading dots)
```

These modules are the user's own `.mv` files. Module lookup order is: the project source root, then `MANV_PATH`. There is no bundled library on the search path while the standard library is parked.

### 7.3 The `std` namespace (parked)

The interpreter currently exposes a `std.<subsystem>.<function>(...)` namespace (`std.json`, `std.rand`, `std.path`, `std.time`). It is **not part of the language** for now and should not be used. It returns with the standard library redesign, if at all.

---

## 8. User-defined types

### 8.1 `type` ✅

```manv
type Point:
    """Optional docstring."""
    int x = 0                       # field with a (required) default
    int y = 0
    const int DIMENSIONS = 2        # type-level constant

    fn __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    fn __repr__(self) -> str:
        return "Point(" + str(self.x) + ", " + str(self.y) + ")"

    fn translate(self, dx: int, dy: int):
        self.x = self.x + dx
        self.y = self.y + dy

type Rectangle(Shape):              # inheritance; several bases allowed: (A, B)
    fn area(self) -> float: ...
```

- A type body contains an optional docstring, then fields, `const` members and `fn` methods. Anything else is `E1006`. Fields are written like declarations: `int x = 0`, or `x = 0` for an inferred type. (🧭 The interpreter currently spells fields `let x: int = 0`.)
- Fields require `= default`. `const` members are read through the type or an instance (`Point.DIMENSIONS`) and cannot be assigned.
- The first parameter of a method is `self`. Methods are called as `p.translate(1, -1)`.
- `__init__` is the constructor and is invoked as `Point(3, 4)`.
- `isinstance(x, Type)` tests membership.
- ⚠️ `print(p)` does not call `__repr__` yet. It prints the interpreter's internal object representation.

### 8.2 `impl` ✅

```manv
impl Point:
    fn magnitude(self) -> float: ...
```

Adds methods to an existing type without editing its declaration. Only `fn` is allowed inside.

### 8.3 Accessors ✅

```manv
type User:
    @getter
    fn name(self) -> str: return self._name

    @setter
    fn name(self, value: str): self._name = value
```

---

## 9. Foreign and system interfaces

### 9.1 `extern` (C FFI) ✅

```manv
extern "c":                              # block form
    fn strlen(s: ptr) -> int
    fn malloc(size: int) -> ptr
    fn free(p: ptr)

extern "c" fn puts(s: ptr) -> int        # single-function form
extern "c" fn printf(fmt: ptr, ...) -> int          # varargs
extern "c" fn c_abs(x: int) -> int as "abs"         # rename: ManV name vs C symbol
```

Pointer helpers: `c_str(s)`, `ptr_null()`, `ptr_is_null(p)`, `ptr_add(p, bytes)`. A `void` return type is written by omitting `-> type`.

### 9.2 `syscall` ✅

```manv
syscall("getpid")
syscall(number_or_name, arg1, arg2)
```

Usable as a statement or an expression. Typed wrappers are a library concern and wait for the standard library redesign.

### 9.3 Intrinsics

`__intrin.name(args)` calls a compiler-known primitive such as `__intrin.sqrt_float(x)`. The interpreter currently versions intrinsics (`core_len@1`) and rejects calls from non-library code with `E2022`. That rule belongs to the parked standard library and is expected to change once the engine's real primitive set is known. For now, treat `__intrin.*` as internal. See [§1.6](#16-a-small-core-the-library-comes-later).

---

## 10. Macros ⚠️

```manv
macro log(msg):
    ...

log!(msg)        # macro call: identifier immediately followed by `!(`
```

`macro` declarations parse as stubs, and calls to them parse as `name!(args)`. Lowering treats a macro body as a function call site. There is no hygiene or expansion model yet 🧭.

---

## 11. Built-in functions

Available without an `include`. With no standard library, these native builtins are the whole library. The list is expected to stay small, and some entries may move out of the core when the library is designed.

| Group | Names |
|-------|-------|
| I/O | `print` |
| Conversion | `int`, `float`, `str`, `bool` |
| Collections | `len`, `sum`, `min`, `max`, `sorted`, `enumerate`, `zip`, `array_append`, `map_has_key`, `map_keys`, `map_values` |
| Types | `type`, `isinstance` |
| Introspection | `help` |
| FFI | `c_str`, `ptr_null`, `ptr_is_null`, `ptr_add` |

Known quirks that are bugs, not design:

- `print(true)` prints `True`, and `print(none)` prints `None`. These are Python spellings leaking through.
- `zip` takes one argument, an array of arrays: `zip([a, b])`. Its output is wrong at the moment.
- Strings have no methods or indexing, and arrays have no methods. Use the helper functions above.

---

## 12. Reserved and planned syntax

These are reserved so they can be added without breaking existing programs.

| Syntax | Intent |
|--------|--------|
| `gpu ...` statement | Explicit GPU region or launch control. |
| `memory ...` statement | Allocation regions, arenas and address-space annotations. |
| Ownership, move and borrow markers | The planned memory model (see README). |
| GPU memory classes (`global`, `shared`, `local`) | First-class on kernel parameters and locals. |

---

## 13. A complete example

```manv
type Account:
    int balance = 0

    fn __init__(self, opening: int):
        self.balance = opening

    fn withdraw(self, amount: int):
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance = self.balance - amount

fn total(accounts: array) -> int:
    int sum_so_far = 0
    for a in accounts:
        sum_so_far = sum_so_far + a.balance
    return sum_so_far

fn main() -> int:
    Account a = Account(100)
    Account b = Account(50)

    try:
        a.withdraw(30)
        b.withdraw(80)
    except ValueError:
        print("withdrawal refused")
    finally:
        print(total([a, b]))   # 120

    return 0
```

---

## 14. Grammar sketch (informal)

```
program      := { decorator* (fn_decl | type_decl | impl_decl | macro_decl
                              | module_decl | extern_decl | statement) }
block        := NEWLINE INDENT statement+ DEDENT
fn_decl      := "fn" IDENT "(" [params] ")" ["->" type] ":" block
lambda       := "fn" "(" [params] ")" ["->" type] ":" block
params       := IDENT [":" type] { "," IDENT [":" type] }
type_decl    := ("type" | "class") IDENT ["(" IDENT {"," IDENT} ")"] ":" NEWLINE INDENT
                [STRING] { (field_decl | const_decl | decorator* fn_decl) } DEDENT
impl_decl    := "impl" IDENT ":" NEWLINE INDENT { decorator* fn_decl } DEDENT
decorator    := "@" NAME ["(" [arg {"," arg}] ")"] NEWLINE
statement    := decl | assign | aug_assign | if | while | for | try | match | with
              | "return" [expr] | "raise" [expr] | "break" | "continue"
              | "include" ... | "syscall" "(" ... ")" | expr
decl         := type IDENT ["[" expr "]"] ["=" expr]        # typed; type is any type name
              | IDENT "=" expr                               # inferred (declares if unbound)
              | const_decl
const_decl   := "const" [type] IDENT "=" expr               # value required
field_decl   := [type] IDENT "=" expr                        # inside a type body
if           := "if" expr ":" block ["else" ":" block]
while        := "while" expr ":" block ["else" ":" block]
for          := "for" IDENT "in" expr ":" block ["else" ":" block]
try          := "try" ":" block { "except" IDENT ["as" IDENT] ":" block }
                ["else" ":" block] ["finally" ":" block]
expr         := range
range        := or [".." or]
or           := and { ("or" | "||") and }
and          := equality { ("and" | "&&") equality }
equality     := comparison { ("==" | "!=") comparison }
comparison   := term { ("<" | "<=" | ">" | ">=") term }
term         := factor { ("+" | "-") factor }
factor       := unary { ("*" | "/" | "%") unary }
unary        := ("-" | "!" | "not") unary | postfix
postfix      := primary { "(" [args] ")" | "[" expr "]" | "." IDENT }
primary      := NUMBER | STRING | "true" | "false" | "none" | IDENT
              | "(" expr ")" | "[" [expr {"," expr}] "]" | "{" [pair {"," pair}] "}" | lambda
```
