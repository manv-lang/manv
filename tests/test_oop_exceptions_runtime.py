"""Runtime conformance tests for OOP, exceptions, and module imports.

Why this file exists:
- Locks down Python-like class/exception behavior in the AST interpreter.
- Verifies import/module semantics (including cache + relative imports).
- Guards interpreter/compiled observable equivalence for key language features.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.diagnostics import ManvError
from manv.interpreter import Interpreter
from manv.object_runtime import ExceptionObject, InstanceObject
from manv.runner import run_file


def _run_source(tmp_path: Path, source: str, mode: str = "interpreter") -> tuple[int, str]:
    # Shared harness that compiles/runs a generated source file and captures stdout.
    src = tmp_path / "main.mv"
    src.write_text(source, encoding="utf-8")
    out = StringIO()
    code = run_file(src, stdout=out, mode=mode)
    return code, out.getvalue()


def test_exception_basic_raise_catch(tmp_path: Path) -> None:
    source = (
        "fn main() -> int:\n"
        "    try:\n"
        "        raise ValueError(\"x\")\n"
        "    except ValueError as e:\n"
        "        print(1)\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip() == "1"


def test_exception_hierarchy_and_reraise(tmp_path: Path) -> None:
    source = (
        "fn main() -> int:\n"
        "    try:\n"
        "        try:\n"
        "            raise KeyError(\"k\")\n"
        "        except KeyError as e:\n"
        "            print(2)\n"
        "            raise\n"
        "    except Exception:\n"
        "        print(3)\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["2", "3"]


def test_exception_else_and_finally_order(tmp_path: Path) -> None:
    source = (
        "fn main() -> int:\n"
        "    try:\n"
        "        print(1)\n"
        "    except Exception:\n"
        "        print(2)\n"
        "    else:\n"
        "        print(7)\n"
        "    finally:\n"
        "        print(9)\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["1", "7", "9"]


def test_finally_runs_on_return(tmp_path: Path) -> None:
    source = (
        "fn work() -> int:\n"
        "    try:\n"
        "        return 5\n"
        "    finally:\n"
        "        print(9)\n"
        "\n"
        "fn main() -> int:\n"
        "    print(work())\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["9", "5"]


def test_oop_class_methods_inheritance_and_dynamic_attrs(tmp_path: Path) -> None:
    source = (
        "type Base:\n"
        "    fn init(self, x):\n"
        "        self.x = x\n"
        "\n"
        "    fn value(self):\n"
        "        return self.x\n"
        "\n"
        "type Child(Base):\n"
        "    fn value(self):\n"
        "        return self.x + 10\n"
        "\n"
        "impl Child:\n"
        "    fn add(self, y):\n"
        "        return self.x + y\n"
        "\n"
        "fn main() -> int:\n"
        "    let c = Child(5)\n"
        "    c.z = 99\n"
        "    print(c.value())\n"
        "    print(c.add(2))\n"
        "    print(c.z)\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["15", "7", "99"]


def test_class_keyword_and_dual_init_alias(tmp_path: Path) -> None:
    source = (
        "class C:\n"
        "    fn __init__(self, x):\n"
        "        self.x = x\n"
        "\n"
        "class D:\n"
        "    fn init(self, x):\n"
        "        self.x = x + 1\n"
        "\n"
        "fn main() -> int:\n"
        "    let c = C(4)\n"
        "    let d = D(4)\n"
        "    print(c.x)\n"
        "    print(d.x)\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["4", "5"]


def test_object_equality_identity_default_and_dunder_eq(tmp_path: Path) -> None:
    source = (
        "class P:\n"
        "    fn __init__(self, x):\n"
        "        self.x = x\n"
        "\n"
        "class Q:\n"
        "    fn __init__(self, x):\n"
        "        self.x = x\n"
        "    fn __eq__(self, other):\n"
        "        return self.x == other.x\n"
        "\n"
        "fn main() -> int:\n"
        "    let a = P(1)\n"
        "    let b = P(1)\n"
        "    let q1 = Q(2)\n"
        "    let q2 = Q(2)\n"
        "    print(a == b)\n"
        "    print(q1 == q2)\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["False", "True"]


def test_reflection_builtins_type_isinstance_issubclass_and_id(tmp_path: Path) -> None:
    source = (
        "class Base:\n"
        "    fn __init__(self):\n"
        "        self.x = 1\n"
        "\n"
        "class Child(Base):\n"
        "    fn __init__(self):\n"
        "        self.x = 2\n"
        "\n"
        "fn main() -> int:\n"
        "    let c = Child()\n"
        "    let t = type(c)\n"
        "    print(isinstance(c, Child))\n"
        "    print(isinstance(c, Base))\n"
        "    print(issubclass(Child, Base))\n"
        "    print(t.name)\n"
        "    print(id(c) == id(c))\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["True", "True", "True", "Child", "True"]


def test_import_and_from_import_executes_module_code(tmp_path: Path) -> None:
    util = tmp_path / "util.mv"
    util.write_text(
        (
            "let VALUE = 9\n"
            "fn add(x, y) -> int:\n"
            "    return x + y + VALUE\n"
        ),
        encoding="utf-8",
    )
    source = (
        "import util\n"
        "from util import add as plus\n"
        "fn main() -> int:\n"
        "    print(util.VALUE)\n"
        "    print(plus(1, 2))\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["9", "12"]


def test_import_module_cache_single_initialization(tmp_path: Path) -> None:
    util = tmp_path / "util.mv"
    util.write_text(
        (
            "let INIT_COUNT = 0\n"
            "INIT_COUNT = INIT_COUNT + 1\n"
            "fn count() -> int:\n"
            "    return INIT_COUNT\n"
        ),
        encoding="utf-8",
    )
    source = (
        "import util as u1\n"
        "import util as u2\n"
        "fn main() -> int:\n"
        "    print(u1.count())\n"
        "    print(u2.count())\n"
        "    print(id(u1) == id(u2))\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["1", "1", "True"]


def test_relative_from_import_inside_package(tmp_path: Path) -> None:
    # Package layout:
    # pkg/__init__.mv defines CONST
    # pkg/helpers.mv defines VALUE
    # pkg/mod.mv imports both via relative forms
    pkg = tmp_path / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.mv").write_text("let CONST = 5\n", encoding="utf-8")
    (pkg / "helpers.mv").write_text("let VALUE = 7\n", encoding="utf-8")
    (pkg / "mod.mv").write_text(
        (
            "from . import CONST\n"
            "from .helpers import VALUE\n"
            "fn read() -> int:\n"
            "    return CONST + VALUE\n"
        ),
        encoding="utf-8",
    )
    source = (
        "import pkg.mod as mod\n"
        "fn main() -> int:\n"
        "    print(mod.read())\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["12"]


def test_relative_import_requires_package_context(tmp_path: Path) -> None:
    # Relative imports from entry module should fail with deterministic ImportError.
    source = (
        "from .util import add\n"
        "fn main() -> int:\n"
        "    return 0\n"
    )
    src = tmp_path / "main.mv"
    src.write_text(source, encoding="utf-8")
    with pytest.raises(ManvError) as err:
        run_file(src, mode="interpreter")
    rendered = err.value.render()
    assert "ImportError" in rendered
    assert "relative import requires package context" in rendered


def test_relative_import_requires_from_syntax(tmp_path: Path) -> None:
    # Python-like rule: `import .foo` is invalid; relative imports must use `from`.
    source = (
        "import .foo\n"
        "fn main() -> int:\n"
        "    return 0\n"
    )
    src = tmp_path / "main.mv"
    src.write_text(source, encoding="utf-8")
    with pytest.raises(ManvError) as err:
        run_file(src, mode="interpreter")
    rendered = err.value.render()
    assert "relative import requires 'from ... import ...' form" in rendered


def test_relative_from_import_parent_package_resolution(tmp_path: Path) -> None:
    # Validate `..` resolution from nested package module to parent package module.
    pkg = tmp_path / "pkg"
    sub = pkg / "sub"
    sub.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.mv").write_text("", encoding="utf-8")
    (sub / "__init__.mv").write_text("", encoding="utf-8")
    (pkg / "common.mv").write_text("let V = 40\n", encoding="utf-8")
    (sub / "mod.mv").write_text(
        (
            "from ..common import V\n"
            "fn read() -> int:\n"
            "    return V + 2\n"
        ),
        encoding="utf-8",
    )
    source = (
        "import pkg.sub.mod as mod\n"
        "fn main() -> int:\n"
        "    print(mod.read())\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["42"]


def test_import_cycle_returns_partial_module_objects(tmp_path: Path) -> None:
    (tmp_path / "a.mv").write_text(
        (
            "import b\n"
            "let A = 1\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "b.mv").write_text(
        (
            "import a\n"
            "let B = 2\n"
        ),
        encoding="utf-8",
    )
    source = (
        "import a\n"
        "import b\n"
        "fn main() -> int:\n"
        "    print(a.A)\n"
        "    print(b.B)\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["1", "2"]


def test_import_search_path_prefers_project_over_manv_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Search order must be deterministic: project root first, then MANV_PATH.
    lib = tmp_path / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    (tmp_path / "util.mv").write_text("let VALUE = 10\n", encoding="utf-8")
    (lib / "util.mv").write_text("let VALUE = 99\n", encoding="utf-8")
    monkeypatch.setenv("MANV_PATH", str(lib))

    source = (
        "import util\n"
        "fn main() -> int:\n"
        "    print(util.VALUE)\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["10"]


def test_import_search_path_uses_manv_path_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # When project root misses, MANV_PATH should be used before bundled std.
    lib = tmp_path / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    (lib / "util.mv").write_text("let VALUE = 55\n", encoding="utf-8")
    monkeypatch.setenv("MANV_PATH", str(lib))

    source = (
        "import util\n"
        "fn main() -> int:\n"
        "    print(util.VALUE)\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["55"]


def test_bundled_std_module_import_without_project_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear MANV_PATH to ensure this resolves from bundled std fallback root.
    monkeypatch.setenv("MANV_PATH", "")
    source = (
        "from builtins import sum as bsum\n"
        "from builtins import len as blen\n"
        "fn main() -> int:\n"
        "    print(bsum([1, 2, 3]))\n"
        "    print(blen([7, 8]))\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["6", "2"]


def test_bundled_str_module_import_uses_str_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # `str` module name is intentional; avoid `string` naming in bootstrap API.
    monkeypatch.setenv("MANV_PATH", "")
    source = (
        "import str as s\n"
        "from str import from_value as to_text\n"
        "fn main() -> int:\n"
        "    print(to_text(123))\n"
        "    print(s.length(\"abcd\"))\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source)
    assert code == 0
    assert out.strip().splitlines() == ["123", "4"]


def test_oop_missing_attribute_raises(tmp_path: Path) -> None:
    source = (
        "type T:\n"
        "    fn init(self):\n"
        "        self.x = 1\n"
        "\n"
        "fn main() -> int:\n"
        "    let t = T()\n"
        "    print(t.missing)\n"
        "    return 0\n"
    )
    src = tmp_path / "main.mv"
    src.write_text(source, encoding="utf-8")

    with pytest.raises(ManvError) as err:
        run_file(src, mode="interpreter")
    assert "AttributeError" in err.value.render()


def test_map_and_list_index_semantics(tmp_path: Path) -> None:
    source_ok = (
        "fn main() -> int:\n"
        "    array a = [10, 20, 30]\n"
        "    map m = {\"k\": 9}\n"
        "    print(a[-1])\n"
        "    print(m[\"k\"])\n"
        "    return 0\n"
    )
    code, out = _run_source(tmp_path, source_ok)
    assert code == 0
    assert out.strip().splitlines() == ["30", "9"]

    source_missing = (
        "fn main() -> int:\n"
        "    map m = {\"k\": 9}\n"
        "    print(m[\"x\"])\n"
        "    return 0\n"
    )
    src = tmp_path / "missing.mv"
    src.write_text(source_missing, encoding="utf-8")
    with pytest.raises(ManvError) as err:
        run_file(src, mode="interpreter")
    assert "KeyError" in err.value.render()


def test_interpreter_compiled_equivalence_for_oop_exceptions(tmp_path: Path) -> None:
    source = (
        "type C:\n"
        "    fn init(self, x):\n"
        "        self.x = x\n"
        "\n"
        "fn main() -> int:\n"
        "    let c = C(4)\n"
        "    try:\n"
        "        print(c.x)\n"
        "        raise ValueError(\"e\")\n"
        "    except Exception:\n"
        "        print(8)\n"
        "    finally:\n"
        "        print(9)\n"
        "    return 0\n"
    )
    src = tmp_path / "main.mv"
    src.write_text(source, encoding="utf-8")

    out_i = StringIO()
    out_c = StringIO()
    code_i = run_file(src, stdout=out_i, mode="interpreter")
    code_c = run_file(src, stdout=out_c, mode="compiled")

    assert code_i == code_c
    assert out_i.getvalue() == out_c.getvalue()


def test_syscall_failure_maps_to_oserror(tmp_path: Path) -> None:
    source = (
        "fn main() -> int:\n"
        "    syscall(\"definitely_not_a_syscall\")\n"
        "    return 0\n"
    )
    src = tmp_path / "main.mv"
    src.write_text(source, encoding="utf-8")
    with pytest.raises(ManvError) as err:
        run_file(src, mode="interpreter")
    assert "OSError" in err.value.render()


def test_gc_cycle_collection_and_exception_rooting() -> None:
    interp = Interpreter(file="<mem>", deterministic_gc=True)
    base_count = len(interp.heap._records)

    t = interp._define_type("Node", "object")
    a = interp.heap.allocate("Node", InstanceObject(type_obj=t))
    b = interp.heap.allocate("Node", InstanceObject(type_obj=t))
    a.attrs["peer"] = b
    b.attrs["peer"] = a
    interp.global_env.define("root", a)

    interp.heap.collect("pre")
    assert len(interp.heap._records) >= base_count + 3

    interp.global_env.values.pop("root", None)
    interp.heap.collect("cycle")
    assert len(interp.heap._records) == base_count + 1

    val_err = interp.types["ValueError"]
    payload = interp.heap.allocate("Node", InstanceObject(type_obj=t))
    exc = interp.heap.allocate("Exception", ExceptionObject(type_obj=val_err, message="m", payload=payload))
    interp.active_exceptions.append(exc)
    interp.heap.collect("exception-root")
    assert any(r.payload is payload for r in interp.heap._records.values())

    interp.active_exceptions.pop()
    interp.heap.collect("exception-release")
