"""Integration tests for the C FFI extern declaration system.

Covers:
- AST / parser (no live C calls needed)
- Interpreter / FFI engine (ctypes, conditional on libc availability)
- Stdlib libc module integration
- HLIR metadata
"""
from __future__ import annotations

import os
import sys
import tempfile
from io import StringIO
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv import ast as mast
from manv.compiler import parse_program
from manv.diagnostics import ManvError
from manv.ffi import resolve_lib, load_library, marshal_arg, unmarshal_return, marshal_vararg
from manv.hlir_lowering import lower_ast_to_hlir
from manv.intrinsics import invoke_intrinsic
from manv.memory import NULL, Pointer
from manv.runner import run_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STDLIB_SRC = ROOT.parent / "stdlib" / "src"
if not _STDLIB_SRC.is_dir():
    _STDLIB_SRC = ROOT / "stdlib" / "src"

_HAS_STDLIB = _STDLIB_SRC.is_dir() and (_STDLIB_SRC / "libc.mv").is_file()

# Try loading libc to decide whether live C tests can run.
_HAS_LIBC = False
try:
    load_library("c")
    _HAS_LIBC = True
except Exception:
    pass

_requires_libc = pytest.mark.skipif(not _HAS_LIBC, reason="libc not available on this platform")
_requires_stdlib = pytest.mark.skipif(not _HAS_STDLIB, reason="stdlib/src not found")


def _run_src(src: str, *, mode: str = "interpreter") -> tuple[int, str]:
    """Write *src* to a temp file, run it, return (exit_code, stdout)."""
    with tempfile.NamedTemporaryFile(
        suffix=".mv", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(src)
        fname = f.name
    out = StringIO()
    env_backup = os.environ.copy()
    if _HAS_STDLIB:
        os.environ["MANV_PATH"] = str(_STDLIB_SRC)
    try:
        code = run_file(fname, stdout=out, mode=mode)
    finally:
        os.unlink(fname)
        os.environ.clear()
        os.environ.update(env_backup)
    return code, out.getvalue()


# ---------------------------------------------------------------------------
# Part A — AST / Parser tests (no live C calls needed)
# ---------------------------------------------------------------------------

class TestExternParser:
    def test_extern_fn_decl_single(self):
        """Single-fn form: extern "c" fn strlen(s: ptr) -> int"""
        prog = parse_program('extern "c" fn strlen(s: ptr) -> int', "test.mv")
        assert len(prog.declarations) == 1
        block = prog.declarations[0]
        assert isinstance(block, mast.ExternBlock)
        assert block.lib == "c"
        assert len(block.fns) == 1
        fn = block.fns[0]
        assert fn.name == "strlen"
        assert fn.c_name == "strlen"
        assert fn.varargs is False
        assert fn.return_type == "int"
        assert fn.params[0].type_name == "ptr"

    def test_extern_block_multi_fn(self):
        """Block form with 3 functions."""
        src = (
            'extern "m":\n'
            "    fn sin(x: float) -> float\n"
            "    fn cos(x: float) -> float\n"
            "    fn sqrt(x: float) -> float\n"
        )
        prog = parse_program(src, "test.mv")
        assert len(prog.declarations) == 1
        block = prog.declarations[0]
        assert isinstance(block, mast.ExternBlock)
        assert block.lib == "m"
        assert len(block.fns) == 3
        names = [f.name for f in block.fns]
        assert names == ["sin", "cos", "sqrt"]

    def test_extern_varargs(self):
        """fn printf(fmt: ptr, ...) -> int parses with varargs=True."""
        prog = parse_program('extern "c" fn printf(fmt: ptr, ...) -> int', "test.mv")
        block = prog.declarations[0]
        fn = block.fns[0]
        assert fn.name == "printf"
        assert fn.varargs is True
        assert len(fn.params) == 1  # only the fixed param; ... is not a Param

    def test_extern_alias(self):
        """fn c_strlen(s: ptr) -> int as \"strlen\" gives c_name=\"strlen\"."""
        prog = parse_program(
            'extern "c" fn c_strlen(s: ptr) -> int as "strlen"', "test.mv"
        )
        fn = prog.declarations[0].fns[0]
        assert fn.name == "c_strlen"
        assert fn.c_name == "strlen"

    def test_extern_void_return(self):
        """fn free(p: ptr) with no return type → return_type is None."""
        prog = parse_program('extern "c" fn free(p: ptr)', "test.mv")
        fn = prog.declarations[0].fns[0]
        assert fn.name == "free"
        assert fn.return_type is None

    def test_ptr_type_annotation(self):
        """fn f(p: ptr) -> ptr parses without error."""
        prog = parse_program('extern "c" fn f(p: ptr) -> ptr', "test.mv")
        fn = prog.declarations[0].fns[0]
        assert fn.params[0].type_name == "ptr"
        assert fn.return_type == "ptr"

    def test_extern_block_in_function_body(self):
        """extern block inside a function body registers in local scope."""
        src = (
            'fn main() -> int:\n'
            '    extern "c" fn strlen(s: ptr) -> int\n'
            '    n = strlen(cstr("hi"))\n'
            '    return n\n'
        )
        # Parse only — should not raise
        prog = parse_program(src, "test.mv")
        fn_decl = prog.declarations[0]
        assert isinstance(fn_decl, mast.FnDecl)


# ---------------------------------------------------------------------------
# Part B — FFI engine unit tests
# ---------------------------------------------------------------------------

class TestFfiEngine:
    def test_resolve_lib_c_returns_nonempty(self):
        result = resolve_lib("c")
        assert isinstance(result, str) and result

    def test_resolve_lib_absolute_passthrough(self):
        path = "/usr/lib/libfoo.so"
        assert resolve_lib(path) == path

    def test_marshal_arg_str(self):
        assert marshal_arg("hello", "str") == b"hello"

    def test_marshal_arg_str_bytes_passthrough(self):
        assert marshal_arg(b"hi", "str") == b"hi"

    def test_marshal_arg_int(self):
        assert marshal_arg(42, "int") == 42
        assert marshal_arg(42, "i32") == 42
        assert marshal_arg(42, "i64") == 42

    def test_marshal_arg_float(self):
        result = marshal_arg(3.14, "float")
        assert abs(result - 3.14) < 1e-6

    def test_marshal_arg_bool(self):
        assert marshal_arg(True, "bool") is True
        assert marshal_arg(False, "bool") is False

    def test_marshal_arg_ptr_int(self):
        assert marshal_arg(12345, "ptr") == 12345

    def test_marshal_arg_ptr_none(self):
        assert marshal_arg(None, "ptr") == 0

    def test_unmarshal_return_void(self):
        assert unmarshal_return(42, None) is None
        assert unmarshal_return(42, "void") is None

    def test_unmarshal_return_ptr(self):
        assert unmarshal_return(99, "ptr") == Pointer(99)
        assert unmarshal_return(99, "ptr").foreign
        assert unmarshal_return(None, "ptr") == NULL
        assert unmarshal_return(7, "u8*").pointee == "u8"

    def test_unmarshal_return_str_bytes(self):
        assert unmarshal_return(b"hello", "str") == "hello"

    def test_unmarshal_return_str_none(self):
        assert unmarshal_return(None, "str") == ""

    def test_unmarshal_return_bool(self):
        assert unmarshal_return(1, "bool") is True
        assert unmarshal_return(0, "bool") is False

    def test_marshal_vararg_int(self):
        import ctypes
        result = marshal_vararg(42)
        assert isinstance(result, ctypes.c_longlong)
        assert result.value == 42

    def test_marshal_vararg_float(self):
        import ctypes
        result = marshal_vararg(3.14)
        assert isinstance(result, ctypes.c_double)

    def test_marshal_vararg_str(self):
        import ctypes
        result = marshal_vararg("hi")
        assert isinstance(result, ctypes.c_char_p)
        assert result.value == b"hi"

    def test_marshal_vararg_bytes(self):
        import ctypes
        result = marshal_vararg(b"hi")
        assert isinstance(result, ctypes.c_char_p)


# ---------------------------------------------------------------------------
# Part C — Pointer intrinsics
# ---------------------------------------------------------------------------

class TestPointerIntrinsics:
    """The old helper functions were replaced by `null`, `==`, `+`, `cstr` and `str_from`
    (see tests/conformance/pointers/)."""

    @pytest.mark.parametrize("name", ["ptr_null", "ptr_is_null", "c_str", "ptr_to_str", "ptr_add"])
    def test_old_helpers_are_gone(self, name):
        with pytest.raises(Exception):
            invoke_intrinsic(name, [])


# ---------------------------------------------------------------------------
# Part D — Interpreter / live libc tests
# ---------------------------------------------------------------------------

class TestInterpreterLiveLibc:
    @_requires_libc
    def test_ffi_strlen_interpreter(self, tmp_path):
        src = (
            'extern "c" fn strlen(s: ptr) -> int\n'
            "\n"
            "fn main() -> int:\n"
            '    n = strlen(cstr("hello"))\n'
            "    print(n)\n"
            "    return 0\n"
        )
        code, out = _run_src(src)
        assert code == 0
        assert out.strip() == "5"

    @_requires_libc
    def test_ffi_malloc_free_interpreter(self, tmp_path):
        src = (
            'extern "c":\n'
            "    fn malloc(n: int) -> ptr\n"
            "    fn free(p: ptr)\n"
            "\n"
            "fn main() -> int:\n"
            "    buf = malloc(64)\n"
            "    print((buf == null))\n"
            "    free(buf)\n"
            "    return 0\n"
        )
        code, out = _run_src(src)
        assert code == 0
        assert out.strip() in {"False", "false"}

    @_requires_libc
    def test_ffi_block_form_multiple_fns(self, tmp_path):
        src = (
            'extern "c":\n'
            "    fn strlen(s: ptr) -> int\n"
            "    fn malloc(n: int) -> ptr\n"
            "    fn free(p: ptr)\n"
            "\n"
            "fn main() -> int:\n"
            '    n = strlen(cstr("hi"))\n'
            "    buf = malloc(n)\n"
            "    free(buf)\n"
            "    print(n)\n"
            "    return 0\n"
        )
        code, out = _run_src(src)
        assert code == 0
        assert out.strip() == "2"

    @_requires_libc
    def test_ffi_alias_form(self, tmp_path):
        """extern alias: ManV name differs from C symbol name."""
        src = (
            'extern "c" fn c_len(s: ptr) -> int as "strlen"\n'
            "\n"
            "fn main() -> int:\n"
            '    n = c_len(cstr("hello!"))\n'
            "    print(n)\n"
            "    return 0\n"
        )
        code, out = _run_src(src)
        assert code == 0
        assert out.strip() == "6"

    @_requires_libc
    def test_ffi_unknown_lib_raises(self, tmp_path):
        src = (
            'extern "nonexistentlib999xyzzy" fn bogus_fn()\n'
            "\n"
            "fn main() -> int:\n"
            "    bogus_fn()\n"
            "    return 0\n"
        )
        with pytest.raises(Exception):
            _run_src(src)

    @_requires_libc
    def test_ffi_unknown_symbol_raises(self, tmp_path):
        src = (
            'extern "c" fn totally_nonexistent_fn_abc123()\n'
            "\n"
            "fn main() -> int:\n"
            "    totally_nonexistent_fn_abc123()\n"
            "    return 0\n"
        )
        with pytest.raises(Exception):
            _run_src(src)

    @_requires_libc
    def test_ptr_null_in_interpreter(self, tmp_path):
        src = (
            "fn main() -> int:\n"
            "    p = null\n"
            "    print((p == null))\n"
            "    return 0\n"
        )
        code, out = _run_src(src)
        assert code == 0
        assert out.strip() in {"True", "true"}

    @_requires_libc
    def test_ptr_add_in_interpreter(self, tmp_path):
        src = (
            'extern "c" fn malloc(n: int) -> ptr\n'
            'extern "c" fn free(p: ptr)\n'
            "\n"
            "fn main() -> int:\n"
            "    buf = malloc(16)\n"
            "    p2 = (buf as u8*) + 8\n"
            "    print((p2 == null))\n"
            "    free(buf)\n"
            "    return 0\n"
        )
        code, out = _run_src(src)
        assert code == 0
        assert out.strip() in {"False", "false"}


# ---------------------------------------------------------------------------
# Part E — Stdlib libc integration
# ---------------------------------------------------------------------------

class TestStdlibLibc:
    @_requires_libc
    @_requires_stdlib
    def test_include_strlen_from_libc(self, tmp_path):
        src = (
            "include strlen from libc\n"
            "\n"
            "fn main() -> int:\n"
            '    n = strlen(cstr("hello"))\n'
            "    print(n)\n"
            "    return 0\n"
        )
        code, out = _run_src(src)
        assert code == 0
        assert out.strip() == "5"

    @_requires_libc
    @_requires_stdlib
    def test_include_malloc_free_from_libc(self, tmp_path):
        src = (
            "include malloc, free from libc\n"
            "\n"
            "fn main() -> int:\n"
            "    buf = malloc(128)\n"
            "    print((buf == null))\n"
            "    free(buf)\n"
            "    return 0\n"
        )
        code, out = _run_src(src)
        assert code == 0
        assert out.strip() in {"False", "false"}

    @_requires_stdlib
    def test_libc_module_parses_cleanly(self):
        """libc.mv parses without any errors."""
        libc_src = (_STDLIB_SRC / "libc.mv").read_text(encoding="utf-8")
        prog = parse_program(libc_src, str(_STDLIB_SRC / "libc.mv"))
        # Should contain at least one ExternBlock
        extern_blocks = [d for d in prog.declarations if isinstance(d, mast.ExternBlock)]
        assert len(extern_blocks) >= 1

    @_requires_stdlib
    def test_libc_module_has_malloc_and_free(self):
        libc_src = (_STDLIB_SRC / "libc.mv").read_text(encoding="utf-8")
        prog = parse_program(libc_src, str(_STDLIB_SRC / "libc.mv"))
        fn_names = set()
        for d in prog.declarations:
            if isinstance(d, mast.ExternBlock):
                fn_names.update(fn.name for fn in d.fns)
        assert "malloc" in fn_names
        assert "free" in fn_names
        assert "strlen" in fn_names


# ---------------------------------------------------------------------------
# Part F — HLIR integration
# ---------------------------------------------------------------------------

class TestHlirIntegration:
    def test_extern_block_appears_in_hlir_module(self):
        src = (
            'extern "c":\n'
            "    fn strlen(s: ptr) -> int\n"
            "    fn malloc(n: int) -> ptr\n"
            "\n"
            "fn main() -> int:\n"
            "    return 0\n"
        )
        prog = parse_program(src, "test.mv")
        module = lower_ast_to_hlir(prog, "test.mv")
        assert len(module.extern_fns) == 2
        names = {e.manv_name for e in module.extern_fns}
        assert names == {"strlen", "malloc"}

    def test_extern_decl_metadata_correct(self):
        src = (
            'extern "c" fn printf(fmt: ptr, ...) -> int\n'
            "\n"
            "fn main() -> int:\n"
            "    return 0\n"
        )
        prog = parse_program(src, "test.mv")
        module = lower_ast_to_hlir(prog, "test.mv")
        assert len(module.extern_fns) == 1
        e = module.extern_fns[0]
        assert e.manv_name == "printf"
        assert e.c_name == "printf"
        assert e.lib == "c"
        assert e.return_type == "int"
        assert e.varargs is True

    def test_extern_call_annotated_in_hlir(self):
        src = (
            'extern "c" fn strlen(s: ptr) -> int\n'
            "\n"
            "fn main() -> int:\n"
            '    n = strlen(cstr("hello"))\n'
            "    return n\n"
        )
        prog = parse_program(src, "test.mv")
        module = lower_ast_to_hlir(prog, "test.mv")
        # Find a call instruction with extern=True
        extern_calls = []
        for fn in module.functions:
            for blk in fn.blocks:
                for instr in blk.instructions:
                    if instr.op == "call" and instr.attrs.get("extern"):
                        extern_calls.append(instr)
        assert len(extern_calls) >= 1
        call = extern_calls[0]
        assert call.attrs["extern_c_name"] == "strlen"
        assert call.attrs["extern_lib"] == "c"

    def test_extern_alias_reflected_in_hlir(self):
        """The c_name in HExternDecl matches the `as` alias."""
        src = (
            'extern "c" fn my_len(s: ptr) -> int as "strlen"\n'
            "\n"
            "fn main() -> int:\n"
            "    return 0\n"
        )
        prog = parse_program(src, "test.mv")
        module = lower_ast_to_hlir(prog, "test.mv")
        e = module.extern_fns[0]
        assert e.manv_name == "my_len"
        assert e.c_name == "strlen"

    def test_hlir_module_to_dict_includes_extern_fns(self):
        src = (
            'extern "c" fn free(p: ptr)\n'
            "\n"
            "fn main() -> int:\n"
            "    return 0\n"
        )
        prog = parse_program(src, "test.mv")
        module = lower_ast_to_hlir(prog, "test.mv")
        d = module.to_dict()
        assert "extern_fns" in d
        assert len(d["extern_fns"]) == 1
        assert d["extern_fns"][0]["manv_name"] == "free"
