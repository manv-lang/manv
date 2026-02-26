# manv

`manv` (manipulate variable) is a Python-implemented language toolchain MVP with:

- Rust-leaning declarations and Python-style indentation blocks
- No Rust-style `mut`; declarations use `let` or C-style typed declarations
- Interpreter execution path for fast iteration (`run`, `repl`)
- Compiler pipeline artifacts through Kernel IR:
  - AST
  - HIR
  - Graph IR (tensor/dataflow DAG)
  - Kernel IR
- Registry-aware package management from CLI (`auth`, `add`)

## Quickstart

```bash
poetry install
poetry run manv init demo
cd demo
poetry run manv run
poetry run manv compile --emit ast,hir,graph,kernel
poetry run manv compile --backend cuda-ptx
poetry run manv build
poetry run manv test
```

## Commands

- `manv init [path]`
- `manv run [file|project]`
- `manv compile [file|project] --emit ast,hir,graph,kernel [--backend none|cuda-ptx] [--optimize/--no-optimize]`
- `manv build [file|project]`
- `manv repl`
- `manv test [path]`\n- `manv dap --transport stdio|tcp [--host 127.0.0.1 --port 4711]`

Registry and dependencies:

- `manv auth login --registry <url> --token <token>`
- `manv auth status`
- `manv auth logout`
- `manv add <name[@version]> [project]`
- `manv add <git-url> [project] [--branch <name>|--tag <name>|--rev <sha>]`

Compile artifacts are emitted to `.manv/target` by default.

## Core Language Features

Variables:

```mv
int x = 10
array p[10]
str f = "hello world"
```

Expressions and control flow:

```mv
if (x > 0) and not false:
    x = x + 1
while x < 20:
    x = x + 2
    if x == 14:
        continue
    if x > 18:
        break
```

Functions and data aggregation:

```mv
fn add(a: int, b: int) -> int:
    return a + b

map m = {"k": 1}
array nums[3] = [1, 2]
nums[2] = add(nums[0], nums[1])
m["k2"] = nums[2]
```


Debug adapter architecture and behavior are documented in DEBUGGING.md.

