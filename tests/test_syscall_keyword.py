from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.intrinsics import invoke_intrinsic
from manv.runner import run_file


def test_versioned_intrinsic_id_is_accepted() -> None:
    assert invoke_intrinsic("core_len@1", [[1, 2, 3]]) == 3


def test_syscall_keyword_executes(tmp_path: Path) -> None:
    src = tmp_path / "main.mv"
    src.write_text(
        (
            "fn main() -> int:\n"
            "    syscall(\"getpid\")\n"
            "    return 0\n"
        ),
        encoding="utf-8",
    )
    out = StringIO()
    code = run_file(src, stdout=out, mode="interpreter")
    assert code == 0


def test_syscall_expression_form(tmp_path: Path) -> None:
    src = tmp_path / "main.mv"
    src.write_text(
        (
            "fn main() -> int:\n"
            "    let r = syscall(\"getpid\")\n"
            "    print(r[\"ok\"])\n"
            "    return 0\n"
        ),
        encoding="utf-8",
    )
    out_i = StringIO()
    out_c = StringIO()
    code_i = run_file(src, stdout=out_i, mode="interpreter")
    code_c = run_file(src, stdout=out_c, mode="compiled")
    assert code_i == 0
    assert code_c == 0
    assert out_i.getvalue().strip() in {"True", "true"}
    assert out_i.getvalue() == out_c.getvalue()


def test_bundled_std_has_typed_syscall_wrappers() -> None:
    std_main = ROOT / "std" / "src" / "main.mv"
    text = std_main.read_text(encoding="utf-8")
    assert "fn std_syscall_posix(number: int, args: array) -> map:" in text
    assert "fn std_syscall_windows(name: str, args: array) -> map:" in text
