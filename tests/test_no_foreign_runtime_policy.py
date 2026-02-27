from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tests" / "policy" / "no_foreign_runtime_baseline.json"
STDLIB_PY_ROOT = ROOT / "manv" / "stdlib"
INTRINSICS_FILE = ROOT / "manv" / "intrinsics.py"

BANNED_STDLIB_IMPORT_TOPLEVEL = {
    "json",
    "re",
    "socket",
    "subprocess",
    "urllib",
    "http",
    "asyncio",
    "threading",
    "multiprocessing",
    "pickle",
    "gzip",
    "zipfile",
    "statistics",
    "math",
    "decimal",
    "inspect",
    "importlib",
    "csv",
    "tomllib",
    "configparser",
}

FORBIDDEN_INTRINSIC_PREFIXES = (
    "json_",
    "http_",
    "url_",
    "re_",
    "regex_",
    "socket_",
    "pickle_",
    "gzip_",
    "zip_",
    "compression_",
)


def _load_baseline() -> dict[str, list[str]]:
    raw = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {
        "stdlib_import_violations": list(raw.get("stdlib_import_violations", [])),
        "forbidden_intrinsics": list(raw.get("forbidden_intrinsics", [])),
    }


def _discover_stdlib_import_violations() -> set[str]:
    violations: set[str] = set()
    for path in sorted(STDLIB_PY_ROOT.rglob("*.py")):
        rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in BANNED_STDLIB_IMPORT_TOPLEVEL:
                        violations.add(f"{rel}::{top}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                top = module.split(".")[0] if module else ""
                if top in BANNED_STDLIB_IMPORT_TOPLEVEL:
                    violations.add(f"{rel}::{top}")
    return violations


def _discover_forbidden_intrinsics() -> set[str]:
    tree = ast.parse(INTRINSICS_FILE.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "IntrinsicSpec"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        name = first.value
        if name.startswith(FORBIDDEN_INTRINSIC_PREFIXES):
            out.add(name)
    return out


def test_no_new_foreign_runtime_stdlib_import_violations() -> None:
    baseline = set(_load_baseline()["stdlib_import_violations"])
    current = _discover_stdlib_import_violations()
    new_violations = sorted(current - baseline)
    assert not new_violations, (
        "New foreign-runtime stdlib import violations detected. "
        "Remove them or explicitly update baseline with justification:\n"
        + "\n".join(new_violations)
    )


def test_no_new_forbidden_subsystem_intrinsics() -> None:
    baseline = set(_load_baseline()["forbidden_intrinsics"])
    current = _discover_forbidden_intrinsics()
    new_violations = sorted(current - baseline)
    assert not new_violations, (
        "New forbidden subsystem intrinsics detected. "
        "Use narrow primitive intrinsics instead:\n"
        + "\n".join(new_violations)
    )

