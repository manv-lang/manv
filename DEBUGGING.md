# Debugging Architecture (DAP v1)

ManV provides a Debug Adapter Protocol (DAP) server for source-level debugging of the hybrid runtime.

## Runtime model

- Canonical debug target: HLIR interpreter.
- Execution points are instruction-level HLIR IDs with source provenance.
- Graph/KIR compiled regions are debugged in v1 via trace-and-compare diagnostics.

## Provenance chain

Each executable HLIR instruction has:

- stable HLIR id
- source span (`uri`, start/end line+column)
- provenance id links (`ast_id`, `hlir_id`, optional `graph_id`/`kir_id`)

Graph capture and Kernel IR keep this chain so exception stops can report source locations.

## Breakpoint binding policy

Source breakpoints bind to the first executable point on a line.

- Verified: executable point exists.
- Unverified: no executable point found; a message is returned.

Supported per-breakpoint options:

- condition
- hitCondition
- logMessage

## Stepping

Stepping is HLIR-source aware:

- `stepIn`: stop on next source boundary, entering calls.
- `next`: step over calls from current frame depth.
- `stepOut`: run until caller frame boundary.

## Variables and evaluate

Scopes:

- Arguments
- Locals
- Temporaries
- Globals

`evaluate` contexts (`watch`, `hover`, `repl`) are side-effect free in v1.
Allowed expression forms are pure reads/operators with optional `len(...)`.

## Kernel trace-and-compare

When trace-compare is enabled:

1. capture HLIR execution to graph with per-op results
2. lower to Kernel IR and execute via reference backend
3. compare expected vs actual per op
4. on first mismatch, emit `stopped(reason="exception")` with provenance trail

## DAP transport

Supported transports:

- stdio
- tcp

## Example VS Code launch configuration

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "ManV (DAP stdio)",
      "type": "cppdbg",
      "request": "launch",
      "program": "manv",
      "args": ["dap", "--transport", "stdio"],
      "cwd": "${workspaceFolder}",
      "stopAtEntry": true
    }
  ]
}
```

If your client can start TCP adapters directly, run:

```bash
manv dap --transport tcp --host 127.0.0.1 --port 4711
```

Then connect with launch/attach requests through that socket.

## Current limitations

- No native instruction-level stepping (DWARF/PDB bridge is future work).
- No data breakpoints in v1.
- Kernel single-thread/lane stepping UI is deferred.
- Attach is session-id based (runtime registry), not OS-process-level attach.
