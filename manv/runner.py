from __future__ import annotations

from pathlib import Path
import io
import os
import sys
from typing import TextIO

from .compiled_runtime import compile_and_run_program
from .compiler import analyze_program, parse_program
from .diagnostics import ManvError, diag
from .interpreter import Interpreter, RaiseSignal
from .project import discover_target


# Dual-mode runner entrypoint shared by CLI and tests.
def run_target(
    path: str | Path | None,
    stdout: TextIO | None = None,
    mode: str = "interpreter",
    target_name: str = "x86_64-sysv",
    backend: str = "auto",
    device: str | None = None,
    deterministic_gc: bool = False,
    gc_stress: bool = False,
    stable_debug_format: bool = False,
    argv: list[str] | None = None,
) -> int:
    context = discover_target(path)
    return run_file(
        context.entry,
        argv=argv,
        stdout=stdout,
        mode=mode,
        target_name=target_name,
        backend=backend,
        device=device,
        deterministic_gc=deterministic_gc,
        gc_stress=gc_stress,
        stable_debug_format=stable_debug_format,
    )


def run_file(
    path: str | Path,
    stdout: TextIO | None = None,
    mode: str = "interpreter",
    target_name: str = "x86_64-sysv",
    backend: str = "auto",
    device: str | None = None,
    deterministic_gc: bool = False,
    gc_stress: bool = False,
    stable_debug_format: bool = False,
    argv: list[str] | None = None,
) -> int:
    source_path = Path(path).resolve()
    source = source_path.read_text(encoding="utf-8")
    program = parse_program(source, str(source_path))
    for warning in analyze_program(program, str(source_path)):
        print(warning.render(), file=sys.stderr)

    if mode == "interpreter":
        interpreter = Interpreter(
            file=str(source_path),
            stdout=stdout or sys.stdout,
            deterministic_gc=deterministic_gc,
            gc_stress=gc_stress,
            stable_debug_format=stable_debug_format,
        )
        interpreter.argv = list(argv or [])
        try:
            return interpreter.run_main(program)
        except RaiseSignal as rs:
            frame = rs.error.stacktrace[-1] if rs.error.stacktrace else {"line": 1, "column": 1}
            raise ManvError(
                diag(
                    "E3900",
                    f"{rs.error.type_obj.name}: {rs.error.message}",
                    str(source_path),
                    int(frame.get("line", 1)),
                    int(frame.get("column", 1)),
                )
            ) from None
        except ManvError:
            raise
        except Exception as exc:  # a bug in the interpreter, not in the program: say so without a traceback
            if os.environ.get("MANV_DEBUG"):
                raise
            raise ManvError(
                diag("E3999", f"internal error: {type(exc).__name__}: {exc} (set MANV_DEBUG=1 for a traceback)", str(source_path), 1, 1)
            ) from None

    if mode == "compiled":
        out_stream = stdout or io.StringIO()
        result = compile_and_run_program(
            program,
            source_name=str(source_path),
            stdout=out_stream,
            target_name=target_name,
            backend=backend,
            device=device,
            optimize=True,
            capture=False,
            deterministic_gc=deterministic_gc,
            gc_stress=gc_stress,
            stable_debug_format=stable_debug_format,
        )
        return result.exit_code

    raise ManvError(diag("E7001", f"unsupported run mode: {mode}", str(source_path), 1, 1))
