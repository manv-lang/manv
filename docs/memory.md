# Memory model: Go's approach for the compiler

Status: **decided** (Decision 12 in [TODO.md](../TODO.md)). **Steps 1 and 2 of the work plan are implemented in the interpreter**, which is the specification the compiler must match: address-taken locals live while reachable, `&xs[i]` / `&obj.field` work, and storing a GC pointer in `malloc` memory is a `PointerError`. The runtime library, escape analysis and compiler allocation (steps 3 to 7) are not started.

## 1. What "Go's solution" is

Go has no manual `free`, no ownership system and no reference counts. Three things carry the whole design:

1. **Escape analysis at compile time.** A value lives on the stack unless the compiler cannot prove that nothing outlives the function that made it. `&x`, closures, values stored into longer-lived objects and values passed to unknown code *escape* and are allocated on the heap. Most small, short-lived values never touch the heap, so most allocation costs a stack-pointer bump.
2. **A non-moving, concurrent, tri-colour mark-and-sweep garbage collector.** It is not generational and does not compact. Marking runs alongside the program; only two very short stop-the-world pauses remain (starting and finishing the mark). A *hybrid write barrier* (Yuasa deletion plus Dijkstra insertion) keeps the mark correct while the program keeps running, and the program is asked to help mark ("mark assist") when it allocates faster than the collector can keep up.
3. **A size-class allocator** (the design of tcmalloc): small objects come from per-thread caches of fixed-size slots, grouped in *spans* of pages; objects without pointers live in separate "noscan" spans so the collector never reads them; large objects get their own span. The collector is *paced*: the next cycle starts when the heap has grown by a set percentage over the live data of the last one (`GOGC`, default 100), optionally capped by a soft memory limit.

Because objects never move, an **interior pointer** (a pointer into the middle of an object) is legal and keeps the whole object alive, a pointer can be a plain machine address, and the stack can be scanned without perfect knowledge of every slot. Pointers are precise in the heap (each object has a type with a pointer bitmap) and, in Go's compiler, in the stack too (stack maps at safe points).

## 2. Why this fits ManV

| ManV goal | What Go's model gives |
|-----------|-----------------------|
| Fast, "out of the ordinary" speed for a Python-shaped language | Stack allocation of everything that does not escape, bump allocation of the rest, no write cost on pointer copies (unlike reference counting). |
| C-style pointers (`&x`, `p + 1`, `p->f`) | A non-moving collector makes plain addresses and interior pointers safe. Moving collectors would force handles or pinning. |
| Typed arrays (`f32[]`, `i32[]`) and GPU transfer | Pointer-free ("noscan") objects are never traversed, and stay where they are for zero-copy transfer. |
| `struct` with C layout and FFI | A struct is a plain value; passing its address to C is fine because it does not move. |
| Cycles are normal (`Node* next`, objects pointing at each other) | A tracing collector handles them without weak references or a cycle detector. |
| One runtime for the engine and the compiled program (TODO §9.1) | The collector, allocator and type descriptors are one C library used by both. |

## 3. What changes in the language

### 3.1 Taking the address of a local

Today the interpreter frees an address-taken local when its function returns, so a pointer that outlives it faults (`PointerError`, `conformance/pointers/dangling_local`). With Go's model that is wrong: **`&x` of a local is always valid for as long as anything can reach it**. `fn leak() -> int*` that returns `&local` is legal, exactly as in Go; the compiler's escape analysis moves `local` to the heap. The checked interpreter must follow (see section 6, step 1), so the reference behavior and the compiler agree.

### 3.2 Two kinds of memory, as in Go with cgo

| | GC memory | Manual memory |
|---|-----------|---------------|
| Made by | every language value: `str`, `array`, `map`, `type` instances, `struct` values, closures, address-taken locals | `malloc`, `calloc`, `realloc`, `cstr`, C libraries |
| Freed by | the collector, when unreachable | `free` (and only `free`) |
| Traced? | yes | **no** |
| `free()` | `PointerError` ("not allocated by malloc") | allowed once |

The rule inherited from cgo: **a pointer to GC memory must not be stored in manual memory.** The collector does not scan malloc'd blocks, so such a pointer would not keep its target alive. The checked interpreter reports it as a `PointerError` when it can tell; in unchecked builds it is the programmer's responsibility. Putting a GC pointer in a `struct` or array that is itself GC memory is fine and is the normal case.

### 3.3 Decisions it settles in the pointer plan

- **P-5 (pointers into managed values) is reopened.** The rule "never point into a managed object" was chosen to keep the collector simple. A non-moving collector with interior-pointer support removes that reason. Recommendation: allow `&xs[i]` and `&s.field` for arrays and `struct` values (the pointer keeps the whole array alive), keep `&some_str` and `&map[k]` forbidden (strings are immutable and map slots move when the table grows, as in Go).
- **Pointer arithmetic stays**, but within one object. A pointer moved outside the object it came from is a bug in checked mode (`PointerError`) and undefined in unchecked mode. One past the end is allowed for loops and is treated as pointing at the *next* object when scanned, which can only keep too much alive, never too little.
- **`sizeof`, struct layout and `int` as 64-bit** are unchanged.

### 3.4 What stays out

No ownership, borrowing or lifetimes (§9.4 of TODO.md is superseded for the default memory model: arenas and address-space annotations remain candidates for GPU work, not for ordinary code), no destructors, no user-visible finalizers in v1, no weak references.

## 4. The runtime design (`libmanvrt`, C11)

### 4.1 Object layout

- Every heap object has a header word holding a **type descriptor index** and mark/flag bits. Descriptors say: size, whether it contains pointers, and a pointer bitmap (one bit per word) for objects up to 64 words, a pointer-program for larger or array-shaped objects.
- Strings: immutable `{data*, len}` pointing at a noscan UTF-8 buffer (`str` per Decision 8). Arrays: `{data*, len, cap}` header plus a separately allocated element buffer, like Go slices, so growing never moves the header and typed arrays are noscan. Maps: open addressing with insertion-order iteration (TODO Phase 5), buckets in one object.
- A value of type `int`, `float`, `bool` or `none` is unboxed. A tagged-value layout (Decision 5/engine) decides whether a dynamic slot holds a pointer; the bitmap marks such slots as "maybe pointer" and the collector checks the tag.

### 4.2 Allocator

- About 60 size classes from 8 bytes to 32 KB, each class with its own spans of whole pages. Scan and noscan classes are separate spans. Objects under 16 bytes without pointers are packed by a *tiny allocator*. Anything larger than 32 KB is a span of its own.
- A page-to-span table finds the span (and thus the object start and size) of any address in O(1). That table is how interior pointers resolve, and how the collector tells GC memory from malloc memory and from stack addresses.
- Single-threaded to start, with the structure of per-thread cache / central lists / page heap so threads can be added without redesign.

### 4.3 Collector

Milestones, each shippable on its own:

| | Collector | Stack scanning | Notes |
|---|-----------|----------------|-------|
| **G0** | Stop-the-world mark-sweep, precise heap | **Conservative**: every aligned word in the stack and registers that looks like a heap address is a root | Safe only because objects never move. No compiler support needed, so the engine and LLVM output can both use it immediately. Keeps the `deterministic` and `stress` modes. |
| **G1** | Same, plus lazy/incremental sweep and the GOGC-style pacer (`MANV_GC=100`, `MANV_GC=off`, `MANV_MEMLIMIT`) | Conservative | Pause time is mark time only. |
| **G2** | Precise stack roots from stack maps (LLVM statepoints or a compiler-emitted frame table) | Precise | Removes false retention; needed before moving to concurrent marking. |
| **G3** | **Concurrent** tri-colour marking with the hybrid write barrier and mark assist | Precise | The compiler emits a barrier on every pointer store into the heap (not on stack stores). Two short pauses remain. |

The collector lives in the runtime; the C engine's `Heap` and the Python `Heap` simulation expose the same observable operations (`gc_stats`, force a collection, stress mode) so tests can drive all three.

### 4.4 Escape analysis (compiler pass over HLIR)

- Builds a graph of *locations* (allocation sites, locals, parameters, results) and *flow edges* ("p holds the address of x", "store p into *q", "return p", "pass p to f", "capture x in a closure").
- A location escapes when its address reaches the heap, a global, a result, an unknown call or an escaping closure; when its size is unknown or larger than a limit (64 KB for implicit values); or when it is a `malloc`-style allocation (those are manual and never analysed).
- Per-function **summaries** of parameters (does not escape / flows to result / escapes to heap) let callers keep arguments on the stack across calls. Recursion uses a fixed point over strongly connected components.
- Output: each allocation site is tagged `stack` or `heap` in HLIR. `manv compile --report escape` prints the decisions with reasons (the equivalent of `go build -gcflags=-m`). The interpreter ignores the tags, since it has no stack/heap split.
- The analysis must be *sound before it is clever*: when unsure, heap-allocate.

## 5. Observable behavior that must not differ between interpreter, engine and compiler

The conformance suite is the check. The following are therefore specified now:

- An address-taken local remains valid while reachable (no dangling pointers to locals).
- `free` only accepts `malloc`-family memory; freeing GC memory, a local, a struct or an array is a `PointerError`.
- Object identity and equality are independent of where an object lives (stack vs heap is invisible).
- Collection timing is not observable except through `gc_stats()`, `MANV_GC=off` and the stress/deterministic modes.
- No finalizers, so a program cannot observe that an object was collected.

## 6. Work plan

Each step has its own tests, and the language-visible ones update `syntax.md`.

1. ✅ **Interpreter follows the model.** Address-taken locals stay alive while a `Pointer` to them exists: every pointer carries a keep-alive reference to the owner of its memory (a promoted local's `Lifetime`, or the struct or array value), released by Python's own reference counting. Pointers stored in struct fields and C arrays keep their targets through a side table of the container, and copying a struct copies it. Storing a GC pointer in `malloc` memory is a `PointerError`. `conformance/pointers/dangling_local` was replaced by `escaping_locals`; `gc_pointer_in_malloc` pins the new rule. (`memory.py`: `Lifetime`, `Pointer.keep`, `Memory._holders`.)
2. ✅ **P-5 resolved**: `&xs[i]` on a managed array and `&obj.field` on an object give a `SlotPointer` (`conformance/pointers/managed_pointers`), valid while the array keeps its length. This differs from a Go slice pointer, which keeps pointing into the old buffer after a growth; the interpreter reports it instead so the mistake is caught, and the compiler may do either as long as it does not read freed memory. `&str`, `&map[k]` and `&tuple[i]` stay forbidden.
3. **`libmanvrt` G0** in C: allocator with size classes and spans, type descriptors, mark-sweep, conservative stacks, stress mode. Unit tests in C plus a Python `ctypes` harness.
4. **Escape analysis** over HLIR with `--report escape`, golden-tested. Used first by the LLVM backend; the engine can use it to avoid heap allocation of `struct` values.
5. **LLVM backend allocation**: `new`/literals/closures call the runtime or become `alloca` according to the escape tag; pointer-containing types emit descriptors.
6. **G1 pacer**, then **G2 precise stacks**, then **G3 concurrent marking** with barrier emission, each behind the same conformance suite plus a GC torture suite (cycles, large graphs, interior pointers, allocation in loops, `MANV_GC=off`).
7. Share `libmanvrt` with the C engine (TODO Phase 5) so there is one collector.

## 7. Risks

- **Conservative stack scanning (G0/G1)** can retain garbage that looks like a pointer. Acceptable until G2; it never frees live data.
- **Write barriers cost** a few percent on pointer-heavy code. They are only needed from G3 and only on heap stores.
- **Interior pointers and the page table** need a lookup on every mark step; the table must be compact and cache-friendly.
- **Cgo-style rule (GC pointers in malloc memory)** is easy to break silently in unchecked builds. The checked interpreter and a debug runtime mode (`MANV_CHECK=1`) must catch it.
- **`@gpu` code** has its own memory spaces (§9.3 of TODO.md). Device buffers are manual memory from the collector's point of view and are never traced.
