# ManV

ManV (Manipulate Variable) is a modern language and runtime initiative focused on high-performance computing, machine learning workloads, and direct GPU-oriented execution.

It is designed to feel approachable while still exposing a serious compiler architecture for low-level control and future backend expansion.

## Why ManV

- Built for ML and GPU-heavy workloads from day one.
- Hybrid runtime model: fast interpreter iteration with a compiler pipeline for optimization.
- Multi-stage IR architecture that keeps semantics explicit and debuggable.
- Cross-platform toolchain and project workflow.
- Standard library strategy centered on language-authored source backed by internal intrinsics.

## Architecture Value

ManV uses a structured lowering pipeline:

`AST -> HLIR -> Graph IR -> Kernel IR -> backend boundary`

This gives ManV clear separation between:

- Language semantics
- Optimization and graph transformations
- Kernel formation
- Backend-specific code generation and dispatch

The result is a foundation that is easier to validate, test, and scale across CPU and GPU targets.

## Execution Model

ManV currently prioritizes semantic consistency:

- Interpreter mode for fast feedback loops and debugging.
- Compiled mode that remains HLIR-authoritative for parity in v1.
- Kernelization paths with safe fallback when regions are not eligible.

## Tooling

CLI surface:

- `manv init [path] [--std]`
- `manv run [file|project]`
- `manv compile [file|project]`
- `manv build [file|project]`
- `manv repl`
- `manv test [path]`
- `manv dap --transport stdio|tcp [--host 127.0.0.1 --port 4711]`\r\n- `manv lsp --transport stdio|tcp [--host 127.0.0.1 --port 2087]`

Package and registry operations:

- `manv auth login`
- `manv auth status`
- `manv auth logout`
- `manv add <name[@version]>`
- `manv add <git-url>`

Build artifacts are emitted to `.manv/target` by default.

## Standard Library Direction

ManV is moving to a pure language-authored standard library model.

- `__intrin.*` provides the internal compiler/runtime bridge.
- Intrinsics are validated by semantic analysis and lowered through HLIR.
- Runtime behavior stays consistent between interpreter and compiled execution.
- `manv init <path> --std` scaffolds the compiler-shipped ManV `std` source.
- `syscall(...)` is available as both statement and expression form.
- `std` includes typed syscall wrappers: `std_syscall_posix(...)` and `std_syscall_windows(...)`.

Example:

```manv
fn main() -> int:
    let r = syscall("getpid")
    print(r["ok"])
    return 0
```

## Current Status

ManV is an active v1 foundation phase with emphasis on:

- Semantic correctness
- Deterministic diagnostics and testing
- Debugger integration (DAP)
- Language tooling integration (LSP via pygls)
- Backend extensibility for future native GPU execution

Debug adapter design details are documented in `DEBUGGING.md`.
