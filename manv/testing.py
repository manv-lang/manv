from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import shutil
import tempfile
import tomllib

from .builder import build_target
from .compiler import compile_target
from .diagnostics import ManvError
from .project import discover_target, init_project
from .repl import run_repl
from .runner import run_target


@dataclass
class CaseResult:
    name: str
    passed: bool
    message: str


@dataclass
class SuiteResult:
    passed: int
    failed: int
    results: list[CaseResult]


def run_e2e_suite(root: str | Path) -> SuiteResult:
    root_path = Path(root).resolve()
    case_files = sorted(root_path.rglob("case.toml"))
    results: list[CaseResult] = []
    for case_file in case_files:
        results.append(_run_case(root_path, case_file))
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    return SuiteResult(passed=passed, failed=failed, results=results)


def _run_case(suite_root: Path, case_file: Path) -> CaseResult:
    data = tomllib.loads(case_file.read_text(encoding="utf-8"))
    name = str(data.get("name", case_file.parent.name))
    command = str(data.get("command", "run"))
    target = str(data.get("target", "."))
    emit = str(data.get("emit", "ast,hir,graph,kernel"))
    backend = str(data.get("backend", "none"))
    abi_target = str(data.get("abi_target", data.get("target_abi", "x86_64-sysv")))
    run_mode = str(data.get("run_mode", "interpreter"))
    optimize = bool(data.get("optimize", True))
    capture = bool(data.get("capture", False))
    expect_exit = int(data.get("expect_exit", 0))
    expect_stdout = _as_list(data.get("expect_stdout_contains", []))
    expect_stderr = _as_list(data.get("expect_stderr_contains", []))
    expect_files = _as_list(data.get("expect_files", []))
    repl_input = str(data.get("repl_input", ""))

    stdout = ""
    stderr = ""
    exit_code = 0

    with tempfile.TemporaryDirectory(prefix="manv-case-") as tmp:
        tmp_suite = Path(tmp) / "suite"
        shutil.copytree(suite_root, tmp_suite)
        rel_case_dir = case_file.parent.relative_to(suite_root)
        case_root = tmp_suite / rel_case_dir
        case_target = (case_root / target).resolve()

        try:
            if command == "run":
                out = io.StringIO()
                exit_code = run_target(case_target, stdout=out, mode=run_mode, target_name=abi_target)
                stdout = out.getvalue()
            elif command == "compile":
                ctx = discover_target(case_target)
                compile_target(
                    ctx.entry,
                    ctx.target_dir,
                    emit=[x.strip() for x in emit.split(",") if x.strip()],
                    backend=backend,
                    optimize=optimize,
                    target_name=abi_target,
                    capture_graph=capture,
                )
            elif command == "build":
                build_target(case_target)
            elif command == "init":
                init_project(case_target)
            elif command == "repl":
                in_stream = io.StringIO(repl_input)
                out_stream = io.StringIO()
                exit_code = run_repl(in_stream, out_stream)
                stdout = out_stream.getvalue()
            else:
                raise RuntimeError(f"unsupported test command: {command}")
        except ManvError as err:
            exit_code = 1
            stderr = err.render()
        except Exception as err:  # pragma: no cover - safety net
            exit_code = 1
            stderr = str(err)

        ok = True
        reasons: list[str] = []

        if exit_code != expect_exit:
            ok = False
            reasons.append(f"expected exit {expect_exit}, got {exit_code}")

        for expected in expect_stdout:
            if expected not in stdout:
                ok = False
                reasons.append(f"stdout missing: {expected}")

        for expected in expect_stderr:
            if expected not in stderr:
                ok = False
                reasons.append(f"stderr missing: {expected}")

        for rel in expect_files:
            artifact = case_root / rel
            if not artifact.exists():
                ok = False
                reasons.append(f"missing file: {rel}")

        if ok:
            return CaseResult(name=name, passed=True, message="ok")
        detail = "; ".join(reasons) if reasons else "failed"
        return CaseResult(name=name, passed=False, message=detail)


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value in (None, ""):
        return []
    return [str(value)]
