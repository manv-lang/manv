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
