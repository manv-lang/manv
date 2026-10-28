"""input(), exit(), args() and env(): a program's connection to its surroundings."""

from __future__ import annotations

from pathlib import Path

from cli_runner import invoke


def run(tmp_path: Path, source: str, *args: str, stdin: str = "", env: dict[str, str] | None = None):
    program = tmp_path / "prog.mv"
    program.write_text(source, encoding="utf-8")
    return invoke(["run", str(program), *args], input=stdin, env=env)


def test_input_reads_lines_and_shows_the_prompt(tmp_path: Path) -> None:
    result = run(
        tmp_path,
        'fn main() -> int:\n    a = input("first? ")\n    b = input()\n    print(a + "|" + b)\n    return 0\n',
        stdin="one\ntwo\r\n",
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout == "first? one|two\n"


def test_input_at_end_of_input_is_a_catchable_eoferror(tmp_path: Path) -> None:
    result = run(
        tmp_path,
        'fn main() -> int:\n    try:\n        input()\n    except EOFError as e:\n        print("EOF")\n    return 0\n',
    )
    assert result.stdout == "EOF\n"


def test_exit_sets_the_exit_code_and_stops_the_program(tmp_path: Path) -> None:
    result = run(tmp_path, 'fn main() -> int:\n    print("before")\n    exit(7)\n    print("after")\n    return 0\n')
    assert result.exit_code == 7
    assert result.stdout == "before\n"


def test_exit_cannot_be_caught(tmp_path: Path) -> None:
    result = run(
        tmp_path,
        'fn main() -> int:\n    try:\n        exit(2)\n    except RuntimeError as e:\n        print("caught")\n    return 0\n',
    )
    assert result.exit_code == 2
    assert result.stdout == ""


def test_exit_without_a_code_is_zero_and_works_in_script_mode(tmp_path: Path) -> None:
    result = run(tmp_path, 'print("hi")\nexit()\nprint("never")\n')
    assert result.exit_code == 0
    assert result.stdout == "hi\n"


def test_args_are_the_words_after_the_program(tmp_path: Path) -> None:
    result = run(tmp_path, "fn main() -> int:\n    print(args())\n    print(len(args()))\n    return 0\n", "alpha", "beta gamma")
    assert result.stdout == "['alpha', 'beta gamma']\n2\n"


def test_args_is_empty_without_arguments(tmp_path: Path) -> None:
    result = run(tmp_path, "fn main() -> int:\n    print(args())\n    return 0\n")
    assert result.stdout == "[]\n"


def test_env_reads_the_environment_with_a_default(tmp_path: Path) -> None:
    result = run(
        tmp_path,
        'fn main() -> int:\n    print(env("MANV_TEST_VALUE"))\n    print(env("MANV_TEST_MISSING", "fallback"))\n    print(env("MANV_TEST_MISSING"))\n    return 0\n',
        env={"MANV_TEST_VALUE": "set"},
    )
    assert result.stdout == "set\nfallback\nnone\n"


def test_an_interpreter_bug_is_reported_without_a_traceback(tmp_path: Path, monkeypatch) -> None:
    from manv import interpreter

    def boom(self, *a, **k):
        raise ValueError("synthetic internal failure")

    monkeypatch.setattr(interpreter.Interpreter, "_run_main", boom)
    result = run(tmp_path, "fn main() -> int:\n    return 0\n")
    assert result.exit_code == 1
    assert "E3999" in result.stderr and "synthetic internal failure" in result.stderr
    assert "Traceback" not in result.stderr
