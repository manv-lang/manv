"""Every program in examples/ runs and prints what its committed `.expected` file says.

The expected files were checked by hand against the comments in each example, so a change in
behaviour shows up here before it makes an example lie.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli_runner import invoke

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PROGRAMS = sorted(p for p in EXAMPLES.glob("*.mv") if p.with_suffix(".expected").exists())


def test_every_example_has_a_golden_file() -> None:
    missing = [p.name for p in sorted(EXAMPLES.glob("*.mv")) if not p.with_suffix(".expected").exists()]
    assert not missing, f"examples without an .expected file: {missing}"


@pytest.mark.parametrize("program", PROGRAMS, ids=lambda p: p.stem)
def test_example_output(program: Path) -> None:
    result = invoke(["run", str(program)])
    assert result.exit_code == 0, result.stderr
    expected = program.with_suffix(".expected").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert result.stdout == expected
