"""A runtime error is reported at the line where it happened, not at 1:1."""

from __future__ import annotations

from pathlib import Path

import pytest

from cli_runner import invoke

CASES = [
    ("fn main() -> int:\n    print(1)\n    xs = [1]\n    print(xs[5])\n    return 0\n", ":4:"),
    ("fn f(xs: array) -> int:\n    return xs[9]\n\nfn main() -> int:\n    x = f([1])\n    return 0\n", ":2:"),
    ("fn main() -> int:\n    print(1)\n    raise ValueError(\"bad\")\n", ":3:"),
    ("fn main() -> int:\n    print(1)\n    print(f\"{1:z}\")\n    return 0\n", ":3:"),
]


@pytest.mark.parametrize("source,where", CASES)
def test_runtime_error_points_at_its_line(tmp_path: Path, source: str, where: str) -> None:
    program = tmp_path / "prog.mv"
    program.write_text(source, encoding="utf-8")
    result = invoke(["run", str(program)])
    assert result.exit_code == 1
    assert f"prog.mv{where}" in result.stderr, result.stderr
