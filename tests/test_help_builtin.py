from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.runner import run_file


def test_help_builtin_displays_function_class_and_static_method_docs_with_gpu_hlir_path(tmp_path: Path) -> None:
    source = (
        '@gpu\n'
        "fn noop(out: f32[]) -> void:\n"
        "    return\n"
        "\n"
        "class Math:\n"
        '    "Scalar math utilities."\n'
        "    @static_method\n"
        "    fn abs(x: int) -> int:\n"
        '        "Returns the magnitude of x."\n'
        "        return x\n"
        "\n"
        "fn helper(x: int) -> int:\n"
        '    "Helper docs."\n'
        "    return x\n"
        "\n"
        "fn main() -> int:\n"
        "    help(helper)\n"
        "    help(Math)\n"
        "    help(Math.abs)\n"
        "    return 0\n"
    )
    path = tmp_path / "main.mv"
    path.write_text(source, encoding="utf-8")

    out_interpreter = StringIO()
    out_compiled = StringIO()

    assert run_file(path, stdout=out_interpreter, mode="interpreter") == 0
    assert run_file(path, stdout=out_compiled, mode="compiled") == 0

    rendered = out_interpreter.getvalue().strip().splitlines()
    assert rendered == out_compiled.getvalue().strip().splitlines()
    assert rendered == [
        "fn helper(x: i32) -> i32",
        "Helper docs.",
        "class Math",
        "Scalar math utilities.",
        "@static_method fn Math.abs(x: i32) -> i32",
        "Returns the magnitude of x.",
    ]


def test_help_builtin_displays_module_docstring(tmp_path: Path) -> None:
    (tmp_path / "tools.mv").write_text(
        '"""Small helper module."""\n'
        "\n"
        "fn helper() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )
    source = (
        "import tools\n"
        "fn main() -> int:\n"
        "    help(tools)\n"
        "    return 0\n"
    )
    path = tmp_path / "main.mv"
    path.write_text(source, encoding="utf-8")

    out = StringIO()
    assert run_file(path, stdout=out, mode="interpreter") == 0
    assert out.getvalue().strip().splitlines() == [
        "module tools",
        "Small helper module.",
    ]


def test_help_builtin_falls_back_to_runtime_type_for_plain_values(tmp_path: Path) -> None:
    source = (
        "fn main() -> int:\n"
        "    help(42)\n"
        "    return 0\n"
    )
    path = tmp_path / "main.mv"
    path.write_text(source, encoding="utf-8")

    out = StringIO()
    assert run_file(path, stdout=out, mode="interpreter") == 0
    assert out.getvalue().strip().splitlines() == [
        "value of type int",
        "No docstring available.",
    ]
