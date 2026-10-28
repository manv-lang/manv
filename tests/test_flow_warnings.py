"""Unreachable code is reported as a warning; the program still runs."""

from __future__ import annotations

from pathlib import Path

from cli_runner import invoke


def run(tmp_path: Path, source: str):
    program = tmp_path / "prog.mv"
    program.write_text(source, encoding="utf-8")
    return invoke(["run", str(program)])


def test_code_after_return_is_a_warning_and_the_program_runs(tmp_path: Path) -> None:
    result = run(tmp_path, 'fn f() -> int:\n    return 1\n    print("never")\n\nfn main() -> int:\n    print(f())\n    return 0\n')
    assert result.exit_code == 0
    assert result.stdout == "1\n"
    assert "warning[W2001]" in result.stderr and "prog.mv:3:" in result.stderr


def test_code_after_break_continue_and_raise(tmp_path: Path) -> None:
    source = (
        "fn main() -> int:\n"
        "    for i in 0..3:\n"
        "        break\n"
        "        print(1)\n"
        "    for i in 0..3:\n"
        "        continue\n"
        "        print(2)\n"
        "    try:\n"
        "        raise ValueError(\"x\")\n"
        "        print(3)\n"
        "    except ValueError as e:\n"
        "        pass\n"
        "    return 0\n"
    )
    result = run(tmp_path, source)
    assert result.exit_code == 0
    assert result.stderr.count("W2001") == 3


def test_reachable_code_has_no_warning(tmp_path: Path) -> None:
    result = run(
        tmp_path,
        'fn f(n: int) -> int:\n    if n > 0:\n        return 1\n    return 2\n\nfn main() -> int:\n    print(f(1), f(0))\n    return 0\n',
    )
    assert result.stderr == ""
