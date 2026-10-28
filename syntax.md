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
> **Declarations.** `let` is gone and `const` exists ([§5.1](#51-declarations)). Using `let` is the error `E1015`. The `tools/migrate_let.py` script rewrites old code.

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
- Inside `( )`, `[ ]` or `{ }` a line break does not end the statement and indentation is ignored, so literals and calls can span lines (comments and blank lines are fine there). A bracket that is never closed is `E0007`, reported at the opening bracket. The one place layout still applies inside brackets is a block lambda: a line ending in `:` that contains `fn` opens an indented body, and the closing `)` goes on a line after it (§4.4).

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
| Integer | `0`, `42`, `1_000_000`, `0xFF`, `0o17`, `0b101` | ✅ decimal, hex, octal, binary; `_` between digits |
| Float | `3.14`, `1.0`, `1e3`, `2.5e-3` | ✅ decimal, optional exponent; `_` between digits |
| Boolean | `true`, `false` | ✅ |
| None | `none` | ✅ |
| String | `"hi"`, `'hi'` | ✅ single-line only (see escapes below) |
| Array | `[1, 2, 3]` | ✅ may span lines |
| Map | `{"a": 1, "b": 2}` | ✅ may span lines |
| Tuple | `(1, 2)`, `(1,)`, `()` | ✅ see §4.7 |
| f-string | `f"x={x}"` | ✅ see §4.8 |

**Strings.** Either quote character works. Escape sequences are interpreted (✅): `\n` `\t` `\r` `\0` `\\` `\"` `\'`, `\xNN` (two hex digits) and `\uNNNN` (four hex digits). Any other escape, or a malformed `\x`/`\u`, is `E0006`. A string cannot contain a raw line break; use `\n`.

**Numbers.** A malformed number (`0x` with no digits, `0b102`, `1__0`, `100_`, `1e+`, or letters right after digits such as `12abc`) is `E0008`. A float too large to represent is also `E0008`. `0..5` lexes as `0`, `..`, `5` (a range), not as a float.

### 2.5 Operators and punctuation ✅

```
+  -  *  /  %                 arithmetic
== != <  <= >  >=             comparison
in   not in                   membership (§4.6): substring, array element, map key
and or not   &&  ||  !        logic (word forms and symbol forms are interchangeable)
=                             assignment
+= -= *= /= %=                augmented assignment (§5.2)
..                            range
->                            return-type arrow
@                             decorator
.   ,   :                     member access, separator, block opener
( ) [ ] { }                   grouping, call, index, collection literals
```

No bitwise (`& | ^ ~ << >>`) and no exponent (`**`) operators exist yet 🧭.

**Conditional expression** ✅: `a if cond else b` evaluates `cond` first and then only the chosen branch. It is the loosest operator and right-associative, so `"neg" if x < 0 else "zero" if x == 0 else "pos"` chains without parentheses. The `else` is required (`E1017`). In a `match` pattern the `if` starts the guard, not a conditional.

---

## 3. Types

### 3.1 Built-in types

| Type | Meaning | Declaration prefix (`T name = ...`) |
|------|---------|--------------------------------------|
| `int` | integer | ✅ |
| `i32`, `u8`, `usize` | fixed-width integers | ✅ |
| `i64` | 64-bit integer | ✅ |
| `float`, `f32` | floating point | ✅ |
| `bool` | boolean | ✅ |
| `str` | string | ✅ |
| `array` | ordered, growable sequence | ✅ |
| `map` | key-to-value dictionary | ✅ |
| `ptr` | raw pointer (for FFI) | ✅ |
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
| 1 | `a if c else b` | right |
| 2 | `a .. b` (range) | non-associative |
| 3 | `or`, `\|\|` | left |
| 4 | `and`, `&&` | left |
| 5 | `not` | right |
| 6 | `==`, `!=` | left |
| 7 | `<`, `<=`, `>`, `>=`, `in`, `not in` | left |
| 8 | `+`, `-` | left |
| 9 | `*`, `/`, `%` | left |
| 10 | unary `-`, `!` | right |
| 11 | call `f(x)`, index `a[i]`, member `a.b` | left |

Comparisons parse left-associatively, so `1 < 2 < 3` is `(1 < 2) < 3`, not a chained comparison.

### 4.2 Arithmetic semantics

- ✅ `/` is **always true division**: `24 / 4` is `6.0`, and `7 / 2` is `3.5`, even for two `int` operands. Use `int(a / b)` to truncate. An integer-division operator (`//`) is an open decision 🧭.
- ✅ `%` takes the sign of the divisor: `-7 % 3` is `2`.
- ✅ `+` concatenates two strings. `"a" + 1` is a `TypeError`, so convert with `str(1)`.
- ✅ Strings compare lexicographically with `<`, `==` and so on.
- ✅ `and` / `or` short-circuit: the right operand is evaluated only when it can change the result. The result is always a `bool`. The HLIR (compiled-mode) interpreter does the same.

### 4.3 Calls, members and indexing ✅

```manv
f(1, 2)          # call
obj.field        # attribute
obj.method(x)    # method call
e.type           # a reserved word is fine after the dot (also on the left of `=`)
a[0]             # index (negative indices count from the end)
m["key"]         # map lookup
shapes.area(r)   # member of an included module
```

Argument lists may span lines. Keyword arguments are accepted only in decorators.

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

- `a .. b` is a half-open range. It is accepted in `for ... in` ✅ and as an index, where it slices (§4.6).
- Open-ended slices work: `a[1..]`, `a[..2]`, `a[..]`. Missing 🧭: stepped ranges `0..10..2`. Note that `..` binds tighter than `if/else`, so write `a[(x if c else y)..]`.
- `for` also iterates the characters of a string and the keys of a map (in insertion order).
- Arrays and maps are reference values: assigning or passing one shares it, and `a.copy()` makes an independent shallow copy. `a[i] = v` mutates in place. ⚠️ `array_append(a, x)` also mutates `a` in place (and returns it), although `examples/05_collections.mv` used to describe it as returning a new array; write `a = array_append(a, x)` or, better, `a.append(x)`. Decision 9 in [TODO.md](TODO.md) records these as the recommended defaults.

### 4.6 Strings, indexing, slicing and methods ✅

Strings are sequences of Unicode code points (`len("h\u00e9llo")` is 5) and are immutable.

```manv
s = "hello"
s[1]            # "e"           negative indexes count from the end: s[-1] is "o"
s[1..4]         # "ell"         a..b slices; works on arrays too
s[3..100]       # "lo"          bounds outside the sequence are clamped, not an error
s[10]           # IndexError
```

Methods are called with a dot, like methods of your own types. A mutating method changes the receiver and returns `none`. Calling one with the wrong number of arguments is a `TypeError`.

| Type | Methods |
|------|---------|
| string | `upper()` `lower()` `strip()` `lstrip()` `rstrip()` `split([sep])` `join(items)` `replace(old, new)` `find(sub)` (index, or -1) `startswith(p)` `endswith(p)` `contains(sub)` |
| array | `append(x)` `pop([i])` `insert(i, x)` `extend(items)` `remove(x)` `index(x)` `contains(x)` `count(x)` `reverse()` `sort()` `copy()` `clear()` |
| map | `keys()` `values()` `items()` (an array of `(key, value)` tuples) `get(key[, default])` `has(key)` `remove(key)` `copy()` `clear()` |

`remove` and `index` on a missing value are `ValueError`; `pop` on an empty array and an out-of-range `pop(i)` are `IndexError`; `remove` of a missing map key is `KeyError`. A map entry whose key is a method name (`{"keys": 1}`) is reached with `m["keys"]`, since `m.keys` is the method.

`x in s` is substring search when `s` is a string (the left side must be a string too), element search (with `==`) for an array, and key lookup for a map. `x not in s` is `not (x in s)`. `ord("A")` is 65 and `chr(65)` is `"A"` (code points; a non-single-character string or a non-code-point int is an error).

Not there yet 🧭: sorting with a key function, and the same methods in compiled mode (the HLIR and native paths know none of them). For building text see f-strings in §4.8.

### 4.7 Tuples and unpacking ✅

A tuple is a fixed-size, immutable sequence of values of any type. Write it with parentheses; a one-element tuple needs a trailing comma, and `()` is the empty tuple.

```manv
t = (1, "one", 2.5)
t[0]                  # 1          indexing, negative indexes, slices (t[1..]) work like arrays
len(t)                # 3
(1, 2) == (1, 2)      # true       compared element by element; a tuple never equals an array
(1, 2) < (1, 3)       # true       ordered lexicographically
(1, 2) + (3,)         # (1, 2, 3)
2 in t                # membership
t[0] = 9              # error E2012 (tuples cannot be changed)
```

A tuple of hashable values can be a map key (`grid[(x, y)] = cell`). Printing writes `(1, 'one')`, and `(7,)` for one element.

**Unpacking** binds several names at once, from a tuple or an array. The right side is fully evaluated first, so a swap works:

```manv
a, b = 1, 2
a, b = b, a                  # a is 2, b is 1
q, r = divmod_like(7, 2)     # from a function that does `return x, y`
for k, v in m.items():       # also: for i, x in enumerate(xs)
    print(k, v)
```

- `return a, b` returns a tuple; the parentheses are optional.
- `a, b = value` declares each name that is not bound yet and assigns the others (like a single `name = value`). In `for k, v in ...` the names always belong to the loop body.
- A size mismatch is a `ValueError` and a non-sequence on the right is a `TypeError`. The same name twice is `E2007`; a constant as a target is `E2064`.
- `tuple` can be written as a return type (`-> tuple`). There is no type syntax for the element types yet 🧭.
- `m.items()` and `enumerate` now give tuples (they gave two-element arrays before).

Not there yet 🧭: unpacking into attributes or elements (`xs[i], xs[j] = xs[j], xs[i]`), nested targets (`(a, b), c = ...`), a `*rest` target, tuple patterns in `match`, and tuples in compiled mode.

### 4.8 f-strings ✅

`f"..."` (or `f'...'`) builds a string from text and `{expression}` fields. Each field is evaluated in order and written the way `print` writes it, so `str`, collections, `true`/`false`/`none` and a type's `__str__` all apply.

```manv
name = "ManV"
n = 42
f"hello {name}"              # hello ManV
f"{n} + 1 = {n + 1}"         # any expression: 42 + 1 = 43
f"{'big' if n > 9 else 'small'}"
f"{{not a field}}"           # {not a field}  ('{{' and '}}' are literal braces)
```

After a `:` a field takes a **format spec**, the same mini-language as Python's: `[[fill]align][sign][#][0][width][,][.precision][type]`.

| Spec | Example | Result |
|------|---------|--------|
| `.2f` | `f"{3.14159:.2f}"` | `3.14` |
| `5d`, `<5`, `^6`, `>6` | `f"[{42:>6}]"` | `[    42]` |
| `05` | `f"{42:05}"` | `00042` |
| `x` `X` `b` `#x` | `f"{255:x}"` | `ff` |
| `,` | `f"{1234567:,}"` | `1,234,567` |
| `.0%` | `f"{0.25:.0%}"` | `25%` |
| `*<7` | `f"{'ManV':*<7}"` | `ManV***` |

Numbers use the spec directly. Any other value is turned into its text first, so width, fill and alignment work on it, but a numeric type code on a non-number (`f"{'text':.2f}"`) is a `ValueError`.

Rules and limits:
- Use the other kind of quote inside a field (`f"{d['k']}"`): a field ends the string at the first matching quote like any string literal.
- A `:` inside brackets does not start a spec, so `{ {'a': 1}['a'] }` works.
- Syntax errors in an f-string (`E1018`): an unterminated `{`, an empty `{}`, a lone `}`, or text after the expression.
- Not there yet 🧭: nested specs (`{n:>{width}}`), `!r`, `{x=}`, multi-line and triple-quoted f-strings, and a `format()` method or `%` formatting.

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

**Typed declarations (✅).** `T name = value` declares `name` with type `T`, and the value is checked against `T` (`int x = "s"` gives `E2002`). It works for every type in [§3.1](#31-built-in-types) and for user-defined types. Today the interpreter accepts it only for `int i32 str array map u8 usize float f32 bool`.

**Inferred declarations (✅).** `name = value` is a *declaration* if `name` is not already bound in an enclosing scope, and an *assignment* if it is. Its type is inferred from the value. This is the replacement for `let x = ...`.

**Constants (✅).**

- `const` requires a value. `const x` alone is `E1016`.
- The type is optional: `const MAX = 5` and `const int MAX = 5` are both legal.
- Reassigning a constant (`MAX = 6`, `MAX += 1`) is the compile-time error `E2064`, even in interpreter mode. Redeclaring it in the same scope is `E2007`.
- A constant cannot be mutated *through its name* either: `ITEMS[0] = 1`, `CONFIG["k"] = v` and `ORIGIN.x = 1` are rejected with `E2065` where the target is a constant. This is a static check on the name. It does not track aliases. See Decision 2a in [TODO.md](TODO.md).
- `const` is allowed at file level, inside functions and inside `type` bodies. File-level constants are the intended way to share configuration and magic numbers, and they are the natural fit for GPU kernels, since a constant is known at compile time.
- Naming convention: `UPPER_SNAKE_CASE`. This is a convention, not a rule.

**Rules common to all declarations.**

- Declaring a name twice in the same scope is an error (`int x = 1` followed by `int x = 2`). Assign instead: `x = 2`.
- A declaration without a value (`int x`) parses. Semantic analysis decides whether it is legal.
- Reading a name that was never declared is `E2010`, as today.
- Scope is by block. A name first assigned inside an `if`, loop body or function is local to that block and gone after it (using it later is `E2010`). Assigning to a name that is already bound in an enclosing scope assigns to that variable; it never creates a shadowing one. This is the recommendation of Decision 2b in [TODO.md](TODO.md), still open to change.

**The typo hazard.** Because `name = value` declares, a misspelled name on the left (`totl = total + 1`) silently creates a new variable instead of failing. Keep this in view. The linter is expected to flag variables that are assigned and never read (TODO Phase 6). Typed declarations (`int total = ...`) are the explicit, typo-proof form.

**Types.** A typed declaration accepts any type name: a builtin (`int`, `i64`, `ptr`, ...), a user-defined `type` (`Point p = Point(1, 2)`), or a type with `[]` suffixes (`f32[] xs = [1.0, 2.0]`). The REPL has no semantic pass, so it tracks constants itself and reports the same `E2064`.

### 5.2 Assignment

```manv
x = 5            # ✅ assigns, or declares the variable if it is not bound yet (§5.1)
obj.field = 5    # ✅ attribute
a[i] = 5         # ✅ element
x += 1           # ✅ augmented assignment (also p.x += 1 and a[i] += 1)
```

Augmented assignment (`+= -= *= /= %=`) works on variables, attributes and elements. The target must already exist (`E2003` otherwise).

### 5.3 Conditionals ✅

```manv
if n < 0:
    print("neg")
else:
    print("non-neg")
```

`elif` ✅ chains conditions without nesting:

```manv
if score >= 90:
    print("A")
elif score >= 80:
    print("B")
else:
    print("F")
```

`elif` is reserved. `else if` on one line is not accepted; use `elif`.

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

### 5.5 Pattern matching

```manv
match x:
    case 1:
        print("one")
    case n if n > 10:
        print("big")
```

✅ `match` compares the subject with each `case` in order and runs the first that matches. A pattern is a literal (compared with `==`), `_` (matches anything), a name that is **not** already bound (it captures the subject for the case body and guard), or a name that is bound (compared with `==`). An optional `if` guard must also be true. If no case matches, nothing runs.

Not supported yet 🧭: array, tuple, type and alternative (`1 | 2`) patterns, and exhaustiveness checking.

### 5.6 Exceptions ✅

```manv
try:
    risky()
except ValueError:
    print("bad value")
except IndexError as e:     # `as e` binds the exception; `e.message` is its text, `e.type` its type name
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
| nested `fn` inside a function body | ✅ visible from its declaration on, and can read (and recurse through) the enclosing scope. Declaring the same name twice in one scope is `E2007`. |
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

- A type body contains an optional docstring, then fields, `const` members and `fn` methods. Anything else is `E1006`. Fields are written like declarations: `int x = 0`, or `x = 0` for an inferred type.
- Fields require `= default`. `const` members are read through the type or an instance (`Point.DIMENSIONS`) and cannot be assigned.
- The first parameter of a method is `self`. Methods are called as `p.translate(1, -1)`.
- `__init__` is the constructor and is invoked as `Point(3, 4)`.
- `isinstance(x, Type)` tests membership.
- `print(p)` and `str(p)` use `__str__` if the type defines it, otherwise `__repr__`, otherwise `<Point x=3, y=4>`. Objects inside an array or map are shown with `__repr__`.

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
| I/O | `print` (any number of values, written separated by spaces) |
| Conversion | `int`, `float`, `str`, `bool` |
| Collections | `len`, `sum`, `min`, `max`, `sorted`, `enumerate`, `zip`, `array_append`, `map_has_key`, `map_keys`, `map_values` |
| Types | `type`, `isinstance` |
| Introspection | `help` |
| FFI | `c_str`, `ptr_null`, `ptr_is_null`, `ptr_add` |

How values are written by `print` and `str()`: `true`, `false` and `none`; numbers as usual; a string as itself (but quoted inside an array or map); arrays as `[1, 2]`; maps as `{'k': 1}`. Written text for strings inside collections is not final.

Known quirks that are bugs, not design:

- `zip` takes one argument, an array of arrays: `zip([a, b])`. Its output is wrong at the moment.
- `not` used to bind tighter than comparison (`not a == b` meant `(not a) == b`). It is now looser, as in Python.

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
