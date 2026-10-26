"""Language-conformance tests: run every ``tests/conformance/**/*.mv`` program and compare.

See tests/conformance/README.md for the file layout and the rules for writing cases.
The expected files are written by hand from the language spec, never generated from the
implementation under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli_runner import invoke

ROOT = Path(__file__).parent / "conformance"
SIDECARS = (".expected", ".error", ".exit", ".xfail")
PROGRAMS = sorted(ROOT.rglob("*.mv"))


def _text(path: Path) -> str:
    # Checkouts may convert line endings; the language's output is always "\n".
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _case_id(program: Path) -> str:
    return program.relative_to(ROOT).with_suffix("").as_posix()


@pytest.mark.parametrize("program", PROGRAMS, ids=_case_id)
def test_conformance(program: Path, request: pytest.FixtureRequest) -> None:
    xfail = program.with_suffix(".xfail")
    if xfail.exists():
        # strict: when the gap is fixed the case XPASSes and fails, which says "delete the .xfail".
        request.applymarker(pytest.mark.xfail(strict=True, reason=_text(xfail).strip()))

    expected_out = program.with_suffix(".expected")
    expected_err = program.with_suffix(".error")
    expected_exit = program.with_suffix(".exit")

    result = invoke(["run", str(program)])

    if expected_err.exists():
        code = _text(expected_err).split()[0]
        assert result.exit_code == 1, f"expected a failing run, got exit {result.exit_code}"
        assert f"error[{code}]" in result.stderr, f"expected diagnostic {code}, stderr was:\n{result.stderr}"
    else:
        want_exit = int(_text(expected_exit).strip()) if expected_exit.exists() else 0
        assert result.exit_code == want_exit, f"exit code {result.exit_code} != {want_exit}\nstderr:\n{result.stderr}"
        assert result.stderr == "", f"unexpected stderr:\n{result.stderr}"

    if expected_out.exists():
        assert result.stdout == _text(expected_out)


def test_conformance_layout() -> None:
    """Every program says what it expects, and no sidecar is left without a program."""
    assert PROGRAMS, "no conformance programs found"

    for program in PROGRAMS:
        has_out = program.with_suffix(".expected").exists()
        has_err = program.with_suffix(".error").exists()
        assert has_out or has_err, f"{_case_id(program)}: needs a .expected or .error file"

    stems = {p.with_suffix("") for p in PROGRAMS}
    for sidecar in ROOT.rglob("*"):
        if sidecar.suffix in SIDECARS:
            assert sidecar.with_suffix("") in stems, f"{sidecar.relative_to(ROOT)}: no matching .mv program"
