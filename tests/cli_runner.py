"""Run the ManV CLI in-process with captured stdio, for tests.

Importable from any test module: the root ``conftest.py`` puts this directory on ``sys.path``.
"""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass

from manv.cli import main


@dataclass
class CliResult:
    exit_code: int
    stdout: str
    stderr: str
    exception: BaseException | None = None


def invoke(args: list[str], *, env: dict[str, str] | None = None, input: str | None = None) -> CliResult:
    """Run the CLI with captured stdio, a temporary environment and optional stdin."""
    out, err = io.StringIO(), io.StringIO()
    saved_io = (sys.stdout, sys.stderr, sys.stdin)
    saved_env = {key: os.environ.get(key) for key in (env or {})}
    exception: BaseException | None = None
    sys.stdout, sys.stderr, sys.stdin = out, err, io.StringIO(input or "")
    os.environ.update(env or {})
    try:
        exit_code = main(list(args))
    except Exception as exc:  # an uncaught error is a failed run, like a crashed process
        exit_code, exception = 1, exc
    finally:
        sys.stdout, sys.stderr, sys.stdin = saved_io
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return CliResult(exit_code, out.getvalue(), err.getvalue(), exception)
