# ManV std

This project is the compiler-shipped, ManV-authored standard library source.

- The canonical source lives in `src/main.mv`.
- `manv init <path> --std` now copies this bundled source template.
- Privileged runtime/platform operations are accessed through `__intrin.*`.
- Platform-direct operations are available through `syscall(...)` and `__intrin.syscall_invoke(...)`.

## Added functionality

## Purity policy

This std source is the purity target for ManV:

- std logic must be written in ManV source.
- No wrapping/importing host-language stdlib subsystems.
- Intrinsics are limited to narrow primitive capabilities and are versioned/tested.

### Bootstrap modules

The bundled std source now includes importable bootstrap modules at `src/` root
(flat layout, no nested `std/std` package), for example:

- `builtins` for ManV-authored wrappers of core builtin operations
- `str` for string conversion/inspection bootstrap helpers
- additional protocol modules can be added in the same flat style as phases progress

These modules are resolved by interpreter module search order:
project source root, then `MANV_PATH`, then bundled std source.

### Syscall support

ManV now supports:

- `syscall(...)` as a statement
- `syscall(...)` as an expression
- typed std wrappers:
  - `std_syscall_posix(number: int, args: array) -> map`
  - `std_syscall_windows(name: str, args: array) -> map`

Example (expression form):

```manv
fn main() -> int:
    let r = syscall("getpid")
    print(r["ok"])
    print(r["result"])
    return 0
```

Example (typed std wrappers):

```manv
fn main() -> int:
    let posix = std_syscall_posix(39, [])      # Linux getpid syscall number
    let win = std_syscall_windows("getpid", [])
    print(posix["ok"])
    print(win["ok"])
    return 0 
```

### Expanded intrinsic boundary

The runtime intrinsic surface was expanded for compiler-shipped std bootstrap, including:

- core builtins bridge: `core_repr`, `core_hash`, `core_min`, `core_max`, `core_sum`, `core_any`, `core_all`, `core_sorted`, `core_range`, `core_enumerate`, `core_zip`, `core_int`, `core_float`, `core_bool`, `core_str`, `core_iter`, `core_next`
- system/runtime bridge: `sys_capabilities`, `sys_require`
- OS/process bridge: `os_getenv`, `os_setenv`, `os_getcwd`, `os_chdir`, `process_run`
- network/URL bridge: `url_parse`, `http_request`
- syscall bridge: `syscall_invoke`

Versioned intrinsic IDs are supported:

```python
from manv.intrinsics import invoke_intrinsic

assert invoke_intrinsic("core_len@1", [[1, 2, 3]]) == 3
```

## Usage

Initialize a new std project:

```bash
manv init ./std --std
```

Run the bundled std source:

```bash
manv run ./std
```
