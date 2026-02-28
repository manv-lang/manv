from __future__ import annotations

from pathlib import Path
import sys

from lsprotocol.types import Position

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.lsp_server import (
    ManvLanguageServer,
    _analyze_document,
    _attribute_chain_at,
    _decorator_completion_items,
    _definition_ranges,
    _intrinsic_completion_items,
    _intrinsic_name_at,
    _signature_for_name,
)


def test_lsp_nested_cuda_intrinsic_completion_and_signature() -> None:
    root_items = _intrinsic_completion_items("__intrin.c")
    assert root_items is not None
    assert any(item.label == "cuda" for item in root_items)

    cuda_items = _intrinsic_completion_items("__intrin.cuda.is")
    assert cuda_items is not None
    assert any(item.label == "is_available" for item in cuda_items)

    analyzed = _analyze_document(Path("dummy.mv").resolve().as_uri(), "fn main() -> int:\n    return 0\n")
    signature = _signature_for_name(analyzed, "__intrin.cuda.is_available")
    assert signature is not None
    assert signature[0].startswith("__intrin.cuda.is_available(")


def test_lsp_decorator_completion_tracks_gpu_surface() -> None:
    names = _decorator_completion_items("@g")
    assert names is not None
    assert [item.label for item in names] == ["gpu"]

    kwargs = _decorator_completion_items("@gpu(re")
    assert kwargs is not None
    assert any(item.label == "required" for item in kwargs)

    # Mode completions are string-valued on purpose because semantics only
    # accepts the explicit "kernel" / "graph" spellings.
    modes = _decorator_completion_items('@gpu(mode="g')
    assert modes is not None
    assert [item.label for item in modes] == ["graph"]


def test_lsp_symbols_show_gpu_signature_and_for_loop_vars(tmp_path: Path) -> None:
    source = (
        "@gpu\n"
        "fn add(a: f32[], b: f32[], out: f32[]) -> void:\n"
        "    for i in 0..len(out):\n"
        "        out[i] = a[i] + b[i]\n"
    )
    path = tmp_path / "main.mv"
    path.write_text(source, encoding="utf-8")

    analyzed = _analyze_document(path.resolve().as_uri(), source)
    add_symbol = next(symbol for symbol in analyzed.symbols if symbol.name == "add")
    loop_symbol = next(symbol for symbol in analyzed.symbols if symbol.name == "i")

    assert '@gpu(required=false, mode="kernel")' in add_symbol.detail
    assert "fn add(a: array[f32], b: array[f32], out: array[f32]) -> none" in add_symbol.detail
    assert loop_symbol.detail == "for i: i32"


def test_lsp_workspace_definitions_include_unopened_files(tmp_path: Path) -> None:
    main_path = tmp_path / "main.mv"
    helper_path = tmp_path / "helper.mv"

    main_source = (
        "fn main() -> int:\n"
        "    return helper(1)\n"
    )
    helper_source = (
        "fn helper(x: int) -> int:\n"
        "    return x\n"
    )

    main_path.write_text(main_source, encoding="utf-8")
    helper_path.write_text(helper_source, encoding="utf-8")

    ls = ManvLanguageServer()
    ls.root_uri = tmp_path.resolve().as_uri()
    main_uri = main_path.resolve().as_uri()
    ls.documents[main_uri] = main_source
    ls.analysis[main_uri] = _analyze_document(main_uri, main_source)

    definitions = _definition_ranges(ls, "helper")
    assert any(uri == helper_path.resolve().as_uri() for uri, _ in definitions)


def test_lsp_attribute_chain_resolves_nested_cuda_intrinsics() -> None:
    line = "    __intrin.cuda.device_count()"
    pos = Position(line=0, character=line.index("device_count") + 3)

    assert _attribute_chain_at(line, pos) == "__intrin.cuda.device_count"
    assert _intrinsic_name_at(line, pos) == "cuda_device_count"
