"""Shared pytest configuration.

Tests that need something this machine may not have are marked, and skipped with a
stated reason instead of failing:

* ``stdlib``      needs the standard library, which is parked until the interpreter
                  works (see TODO.md). Run them anyway with ``--run-stdlib``.
* ``needs_clang`` needs an LLVM toolchain (``clang`` on PATH).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Shared test helpers (tests/cli_runner.py) are importable from every test module.
sys.path.insert(0, str(Path(__file__).parent / "tests"))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-stdlib",
        action="store_true",
        default=False,
        help="also run tests that need the (parked) standard library",
    )


def _have_clang() -> bool:
    from manv.llvm_toolchain import detect_llvm_toolchain

    return detect_llvm_toolchain() is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skips = {}
    if not config.getoption("--run-stdlib"):
        skips["stdlib"] = pytest.mark.skip(
            reason="the standard library is parked until the interpreter works (see TODO.md); use --run-stdlib"
        )
    if not _have_clang():
        skips["needs_clang"] = pytest.mark.skip(reason="clang not found on PATH")

    for item in items:
        for marker, skip in skips.items():
            if marker in item.keywords:
                item.add_marker(skip)
