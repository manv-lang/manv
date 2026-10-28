# Plan: C-style pointers for ManV

Status: **implemented in the Python interpreter (P0-P6, P9); P7 (function pointers) and P8 (compiled-mode and C-engine parity) are open.** The decisions in §5 were settled with the recommended defaults; where the implementation differs from the first draft, the text says so. The user-facing description is in [syntax.md §9.2](../syntax.md#92-pointers-and-raw-memory-).

## 1. Why the current approach feels complicated

Today `ptr` is one opaque type that is really an integer address, plus five helper functions. What exists (checked in `manv/intrinsics.py`, `syntax.md` §9.1, `examples/09_ffi_extern.mv`):

| Today | Problem |
|-------|---------|
| `ptr` is untyped | `ptr` does not say what it points to, so nothing can be read or written through it from ManV. The only use is passing it back to C. |
| No dereference, no address-of, no indexing | You cannot do `*p`, `&x` or `p[i]`. All memory access has to go through an `extern` C function. |
| `ptr_add(p, bytes)` | Arithmetic is a function call and counts **bytes**. In C, `p + 1` moves by `sizeof(*p)`. |
| `ptr_null()` / `ptr_is_null(p)` | Function calls for what C writes as `NULL` and `p == NULL`. `ptr_null` is registered as returning `int`, so `ptr` and `int` mix in the type checker. |
| `c_str(s)` / `ptr_to_str(p)` | `c_str` returns a Python `bytes` object (not memory with an address), and `ptr_to_str` accepts anything. Ownership and lifetime are not defined. |
| `int buf[16]` | The parser already accepts it, but it makes a managed array padded with `none`, not C memory. It would collide with C arrays. |
| `malloc`/`free` only via `extern` | The runtime cannot know which memory is live, so it cannot report a bad access. It crashes or corrupts memory. |

The goal is to replace that with the C model people already know, instead of a ManV-specific helper API.

## 2. What "copy C's pointers" means here

Copied from C as it is:

| Feature | ManV spelling |
|---------|---------------|
| Pointer types | `int*`, `f32*`, `u8*`, `void*`, `int**`, `const u8*` |
| Address-of | `&x`, `&buf[3]`, `&s.field` |
| Dereference, read and write | `*p`, `*p = 5`, `*p += 1` |
| Indexing | `p[i]` is `*(p + i)` |
| Arithmetic, scaled by the pointee size | `p + n`, `p - n`, `p += 1` (no `++`/`--`, ManV has none) |
| Pointer difference | `q - p` (in elements) |
| Comparison and truthiness | `p == q`, `p < q`, `p == null`, and `if p:` (true when not null, as in C) |
| Null | `null` (a literal, type `void*`, converts to any pointer type) |
| Struct member through a pointer | `p->x` |
| Size | `sizeof(T)`, `sizeof(expr)` |
| Allocation | `malloc`, `calloc`, `realloc`, `free`, `memcpy`, `memmove`, `memset`, `memcmp` as core builtins |
| `void*` | converts implicitly to and from any `T*`, as in C |
| Function pointers | `fn(int) -> int` as a type; `&f` / plain `f` gives one |

Deliberate differences, kept as few as possible:

1. **Casts are written `p as T*`**, not `(T*)p`. `(int*)p` cannot be told apart from a parenthesised expression without knowing the type names, and `as` is already a keyword.
2. **No integer/pointer mixing without a cast.** `int` to pointer needs `n as void*`; a pointer to `usize` needs `p as usize`. C allows more; the type checker here should not.
3. **Memory errors are defined, not undefined behaviour,** in the interpreter and in the engine's default mode (§4). A bad access raises a catchable `PointerError`. C's behaviour is available later as an opt-in `--unchecked` mode.
4. **Pointers never point into managed objects.** `&arr[0]` for a ManV `array`, `&some_str` and `&instance` are compile errors. Pointers point at: raw memory from `malloc`, address-taken locals, `struct` values and C arrays (§3, P5/P6). This keeps the garbage collector free to work without pinning.
5. **No `unsafe` keyword.** The pointer syntax itself is the marker, as in C.

## 3. Plan, in phases

Every phase ends with conformance programs in `tests/conformance/pointers/` (expected output written by hand) and a `syntax.md` / `CHANGELOG.md` update. Nothing prints an address; tests compare pointers or print the values behind them.

### P0. Decide and specify (no code)
- [x] Answer the decisions in §5.
- [x] Write `syntax.md` §9.3 "Pointers" with the table above, the type conversion rules and the error list. Mark each construct 🧭 until it is implemented.
- [x] Add the fixed-width integer types the pointee needs: `i8 i16 i32 i64 u8 u16 u32 u64`, `usize`, `f32`, `f64` (today only `u8 usize i32 i64 f32 float` exist). The overlap with decision 5 (integer model) must be settled first.

### P1. Pointer types in the front end
- [x] Lexer/parser: `T*` in declarations (`int* p = ...`), parameters (`p: int*`), return types and `extern` signatures; `const T*`; `T**`. Parse into a **type AST** (this also delivers the "structured type annotations" item in `TODO.md` Phase 3). A statement starting `IDENT * IDENT =` is a declaration, which `a * b = ...` never is.
- [x] Literal `null`. Keyword `sizeof`. Postfix `->`. Unary `&` and `*` (the lexer already has `&&`; `&x` is a new unary token).
- [x] Semantic pass: a `Pointer(T)` type, the conversion rules (`null` and `void*` convert to any pointer type; `T*` to `U*` needs `as`; pointer plus `int` is a pointer; pointer minus pointer of the same type is `int`), `const` checking (`*p = v` through `const T*` is an error), and a check that `&` is applied to an lvalue.
- [x] Mark **address-taken locals** in the semantic pass (`&x` anywhere makes `x` memory-resident). The interpreter and the compilers both need this fact.
- [x] Error codes: `E2070` deref of a non-pointer, `E2071` address of a non-lvalue, `E2072` assignment through `const`, `E2073` incompatible pointer types, `E2074` address of a managed object.

### P2. Raw memory runtime (Python interpreter first)
- [x] A `Memory` module: `malloc`/`calloc`/`realloc`/`free` over real memory (via `ctypes`), plus a **shadow table** of live blocks `(start, size, state)`. Real addresses keep `extern "c"` interop working, which `examples/09` already relies on.
- [x] Typed load/store for each scalar width, little-endian.
- [x] Checked accesses raise `NullPointerError` (null deref), `PointerError` (out of bounds, use after free, invalid or double `free`, misaligned access). Memory the table does not know (returned by foreign C code) is treated as **unchecked** and documented as such.
- [x] `Pointer` runtime value: `(address, pointee type)`. Equality and ordering by address.

### P3. Operators on pointers
- [x] `*p`, `*p = v`, augmented assignment through a pointer, `p[i]`, `p[i] = v`.
- [x] Scaled arithmetic and pointer difference; comparisons; `p == null`. Truthiness (`if p:`) is **allowed, as in C** (the first draft turned it off; copying C won).
- [x] `&x` for address-taken locals: the interpreter stores them in a small memory slot and routes loads and stores through memory.
- [x] `p as T*`, `p as usize`, `n as void*`.
- [x] `sizeof(T)` for scalars and pointers.
- [x] Core builtins `malloc`, `calloc`, `realloc`, `free`, `memcpy`, `memmove`, `memset`, `memcmp`.

### P4. Strings and C strings
- [x] Define the bridge between managed `str` and `u8*`. Proposal: `cstr(s)` returns a `u8*` to a NUL-terminated copy that the caller frees; `str_from(p)` reads up to the NUL; `str_from(p, n)` reads `n` bytes (UTF-8 decoded). String literals passed straight to an `extern` parameter declared `u8*` convert automatically (static data, no free).
- [x] Keep `c_str` / `ptr_to_str` working for one release with a warning, then remove them.

### P5. Structs and `->`
- [x] A `struct` declaration with C layout (field order, natural alignment, padding), **value semantics** (assignment copies) and `sizeof`. `p->x` is `(*p).x`. Nested structs and arrays of structs.
- [x] `type` stays the managed, dynamic class. A `struct` is for layout-sensitive data and FFI.

### P6. C arrays
- [x] A fixed array `int buf[16]` with C semantics: contiguous, `sizeof(buf)` is 64, and the name decays to `int*` in expressions. This **replaces** today's meaning of `int buf[16]` (a padded managed array), so the existing `array_size` code and its tests are migrated first. `f32[]` stays the managed array (and the GPU buffer type).
- [x] Arrays of structs, arrays of pointers, pointer to array element.

### P7. Function pointers
- [ ] Types `fn(int, int) -> int` and `void*` casts of them, calling through a pointer, and passing a ManV function to C as a callback (a small thunk generated per signature).
- [ ] Depends on the engine's FFI layer (`TODO.md` Phase 5), so this lands last.

### P8. Compiler and engine parity
- [ ] HLIR: `addr_of`, `load`, `store`, `ptr_add`, `ptr_diff`, `sizeof` instructions with explicit element types. LLVM lowering uses `alloca`, `getelementptr`, `load`, `store`; address-taken locals stay in `alloca` and the rest are promoted as today.
- [ ] C engine: `LOAD8..LOAD64`, `STORE8..STORE64`, `ADDR_LOCAL`, `PTR_ADD`; the shadow table is the checked mode, the plain instructions are the `--unchecked` mode. The GC never scans raw memory and never moves it.
- [ ] The conformance suite runs unchanged on every engine.

### P9. Clean-up
- [x] Migrate `examples/09_ffi_extern.mv` to the new syntax and drop `ptr_null`, `ptr_is_null`, `ptr_add` (replaced by `null`, `p == null`, `p + n`).
- [x] `ptr` stays as a deprecated alias of `void*` for one release.
- [x] Update the grammar in `syntax.md`, the tree-sitter grammar and the LSP.

## 4. Conformance tests to write (the executable spec)

`tests/conformance/pointers/`:

| File | Pins down |
|------|-----------|
| `deref_read_write` | `*p`, `*p = v`, `*p += 1` on malloc'd and address-taken memory |
| `address_of_local` | `&x` then writing through the pointer changes `x`; two pointers to one local |
| `arithmetic_scaled` | `p + 1` moves by `sizeof(T)` for `u8*`, `i32*`, `f32*`, `i64*` |
| `indexing_and_difference` | `p[i]`, `q - p`, comparisons |
| `pointer_to_pointer` | `int**` read and write |
| `void_pointer_and_casts` | implicit `void*` conversions, `as T*`, `as usize` |
| `null_handling` | `null` literal, `p == null`, deref raises `NullPointerError` |
| `memory_functions` | `calloc` zeroing, `realloc` keeping contents, `memcpy`, `memmove` overlap, `memset`, `memcmp` |
| `strings_bridge` | `cstr`, `str_from`, UTF-8 round trip |
| `struct_layout` | `sizeof`, padding, `p->field`, nested struct |
| `c_arrays` | decay, `sizeof(buf)`, array of structs |
| `errors/pointer_*` | each `E207x` code above |
| `errors/runtime_*` | out of bounds, use after free, double free, misaligned access, all catchable as `PointerError` |

## 5. Decisions needed

| # | Decision | Recommendation |
|---|----------|----------------|
| P-1 | **Checked or unchecked memory by default?** C: unchecked (undefined behaviour). | Checked everywhere by default, so a conformance test has one defined result. An `--unchecked` mode comes with the engine for speed. |
| P-2 | **Real or simulated memory?** Real memory (`ctypes`/the C heap) or an emulated address space. | Real. It keeps `extern "c"` working and is what the engine and LLVM do anyway. A shadow table gives the checking. |
| P-3 | **Cast syntax:** C's `(T*)p` or `p as T*`. | `p as T*` (the one deliberate deviation, see §2). |
| P-4 | **Member access through a pointer:** `p->x` only (C), or `p.x` auto-dereferences too. | `->` only. Strict C, and no ambiguity with managed objects. |
| P-5 | **Pointers into managed values** (`&arr[i]`, `&str`) allowed or forbidden. | Forbidden (`E2074`). A managed array can be copied to a raw buffer with an explicit builtin. This is what lets the GC stay simple (decision 12). |
| P-6 | **`ptr` keyword:** keep as an alias of `void*` or remove at once. | Alias for one release, then remove. |
| P-7 | **Struct vs `type`:** add `struct`, or give `type` a `@repr(c)` decorator. | A new `struct` keyword. `type` keeps managed semantics, and the two differ in copy behaviour. |
| P-8 | **Fixed-width integers and decision 5 (integer model).** | Settle decision 5 first. Pointee types cannot be defined until the integer widths are. |
| P-9 | **Meaning of `int buf[16]`:** C array (P6) or today's padded managed array. | C array. The old form is rarely used and is migrated. |

## 6. Suggested order

1. Answer P-1 to P-9 (P-8 first).
2. P0, then P1 and P2 together (front end and memory runtime are independent).
3. P3 and P4: the first usable version is "malloc a buffer, fill it through a pointer, read it back, free it, with errors reported cleanly".
4. P5 and P6 (structs and arrays), then P9 for the old helpers.
5. P7 and P8 alongside the C engine and compiler work.

The checked interpreter implementation (P1-P4) can start without the C engine. It is also what defines the behaviour the engine and LLVM must match.

## 6a. Superseded by the memory-model decision

[docs/memory.md](memory.md) adopts Go's memory solution (escape analysis plus a non-moving concurrent tracing collector). Two rules in this plan change:

- **Address-taken locals are never dangling.** `&local` stays valid while a pointer to it is reachable, so returning `&local` is legal. ✅ Implemented in the checked interpreter (`conformance/pointers/escaping_locals`); the old `dangling_local` test is gone.
- **P-5 is resolved: allowed.** ✅ Pointers into managed arrays (`&xs[i]`) and object fields (`&obj.field`) work (`SlotPointer`; `conformance/pointers/managed_pointers`). `&some_str`, `&map[k]` and `&tuple[i]` stay forbidden (`E2074`).
- New rule from the model: a pointer to GC memory must not be stored in `malloc` memory (the collector does not scan it). ✅ Enforced as a `PointerError` (`conformance/pointers/gc_pointer_in_malloc`).

## 7. Implementation notes (where the code differs from the plan above)

- **Address-taken locals** are promoted lazily: the first `&x` moves `x` into a small memory cell (`MemCell` in `interpreter.py`), instead of the semantic pass marking the variable up front. The effect is the same; the cell is freed, and the variable demoted to its last value, when the function returns.
- **`c_str` and `ptr_to_str`** were removed at once together with `ptr_null`, `ptr_is_null` and `ptr_add` (the language has no stable release, so a deprecation period had no users to protect). `ptr` stays as an alias of `void*`.
- **Fixed-width types** `i8 i16 u16 u32 u64 f64` exist for sizes, pointees and fields. The arithmetic model (decision 5) is unchanged: `int` is 64 bits in memory, and values are stored with C's wrap-around when they are written to a narrower slot.
- **`free`** only accepts memory from `malloc`, `calloc`, `realloc` or `cstr`; freeing a local, struct or array is a `PointerError`.
- **Truthiness** of a pointer is allowed (decision list in §2, point 3 of the first draft said off).
- **Not done:** P7 (function pointers), P8 (HLIR/LLVM/C-engine parity; these paths reject pointers, structs and C arrays with `E3011`), the `--unchecked` mode, and the `E2074` check for `&xs[0]` where `xs` was declared `f32[]`, which is only caught when it runs.
