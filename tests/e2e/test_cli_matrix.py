from __future__ import annotations

import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.cli import app


runner = CliRunner(mix_stderr=False)
FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixture(tmp_path: Path, name: str) -> Path:
    source = FIXTURES / name
    target = tmp_path / name
    shutil.copytree(source, target)
    return target



def test_init_std_command(tmp_path: Path) -> None:
    target = tmp_path / "std"
    result = runner.invoke(app, ["init", str(target), "--std"])
    assert result.exit_code == 0
    assert "[Init]" in result.stdout
    assert "mode: std" in result.stdout
    assert (target / "project.toml").exists()
    assert (target / ".gitignore").exists()
    assert (target / "src" / "main.mv").exists()

    config = tomllib.loads((target / "project.toml").read_text(encoding="utf-8"))
    assert config["project"]["name"] == "std"
    assert ".manv/" in (target / ".gitignore").read_text(encoding="utf-8")

    run = runner.invoke(app, ["run", str(target)])
    assert run.exit_code == 0
    assert "std ready" in run.stdout


def test_init_command(tmp_path: Path) -> None:
    target = tmp_path / "demo"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0
    assert "[Init]" in result.stdout
    assert "status: initialized" in result.stdout
    assert (target / "project.toml").exists()
    assert (target / ".gitignore").exists()
    assert (target / "src" / "main.mv").exists()
    assert ".manv/" in (target / ".gitignore").read_text(encoding="utf-8")

    suite = runner.invoke(app, ["test", str(target)])
    assert suite.exit_code == 0
    assert "summary: passed=1 failed=0" in suite.stdout


def test_init_command_with_metadata_flags(tmp_path: Path) -> None:
    target = tmp_path / "demo_meta"
    result = runner.invoke(
        app,
        [
            "init",
            str(target),
            "--name",
            "demo-pkg",
            "--description",
            "Demo package",
            "--author",
            "Jane Dev",
            "--python",
            ">=3.12,<3.14",
        ],
    )
    assert result.exit_code == 0

    config = tomllib.loads((target / "project.toml").read_text(encoding="utf-8"))
    assert config["project"]["name"] == "demo-pkg"
    assert config["project"]["description"] == "Demo package"
    assert config["project"]["requires-python"] == ">=3.12,<3.14"
    assert config["project"]["authors"][0]["name"] == "Jane Dev"


def test_auth_login_status_logout(tmp_path: Path) -> None:
    auth_file = tmp_path / "manv_auth.json"
    env = {"MANV_AUTH_FILE": str(auth_file)}

    login = runner.invoke(
        app,
        ["auth", "login", "--registry", "https://registry.example.com", "--token", "tok_live_test_1234"],
        env=env,
    )
    assert login.exit_code == 0
    assert "logged_in" in login.stdout
    assert auth_file.exists()

    status = runner.invoke(app, ["auth", "status"], env=env)
    assert status.exit_code == 0
    assert "https://registry.example.com" in status.stdout
    assert "1234" in status.stdout

    logout = runner.invoke(app, ["auth", "logout"], env=env)
    assert logout.exit_code == 0
    assert "logged_out" in logout.stdout

    status_after = runner.invoke(app, ["auth", "status"], env=env)
    assert status_after.exit_code == 0
    assert "logged_out" in status_after.stdout


def test_add_registry_dependency(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_ok")
    auth_file = tmp_path / "manv_auth.json"
    env = {"MANV_AUTH_FILE": str(auth_file)}

    login = runner.invoke(
        app,
        ["auth", "login", "--registry", "https://registry.example.com", "--token", "tok_live_test_1234"],
        env=env,
    )
    assert login.exit_code == 0

    result = runner.invoke(app, ["add", "tensorx@1.2.3", str(project)], env=env)
    assert result.exit_code == 0
    assert "source: registry" in result.stdout

    manifest = tomllib.loads((project / "project.toml").read_text(encoding="utf-8"))
    deps = manifest.get("tool", {}).get("manv", {}).get("dependencies", {})
    assert "tensorx" in deps
    assert deps["tensorx"]["version"] == "1.2.3"
    assert deps["tensorx"]["registry"] == "https://registry.example.com"


def test_add_git_dependency(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_ok")

    result = runner.invoke(
        app,
        ["add", "https://github.com/manv-lang/math.git", str(project), "--branch", "main"],
    )
    assert result.exit_code == 0
    assert "source: git" in result.stdout

    manifest = tomllib.loads((project / "project.toml").read_text(encoding="utf-8"))
    deps = manifest.get("tool", {}).get("manv", {}).get("dependencies", {})
    assert "math" in deps
    assert deps["math"]["git"] == "https://github.com/manv-lang/math.git"
    assert deps["math"]["branch"] == "main"


def test_run_command(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "run_hello")
    result = runner.invoke(app, ["run", str(project)])
    assert result.exit_code == 0
    assert "Hello, World!" in result.stdout


def test_run_core_language_features(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "run_core_features")
    result = runner.invoke(app, ["run", str(project)])
    assert result.exit_code == 0
    for expected in ["15", "10", "8", "9", "True"]:
        assert expected in result.stdout


def test_compile_outputs_all_ir(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_ok")
    result = runner.invoke(app, ["compile", str(project), "--emit", "ast,hir,graph,kernel"])
    assert result.exit_code == 0
    assert "[Compile]" in result.stdout
    for name in ["main.ast.json", "main.hir.json", "main.graph.json", "main.kernel.json"]:
        assert (project / ".manv" / "target" / name).exists()
    graph_payload = json.loads((project / ".manv" / "target" / "main.graph.json").read_text(encoding="utf-8"))
    assert graph_payload["kind"] == "tensor_dag"
    assert graph_payload["optimization"]["constant_folding"] >= 1
    assert graph_payload["optimization"]["dead_nodes_removed"] >= 1


def test_compile_cuda_ptx_backend(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_ok")
    result = runner.invoke(app, ["compile", str(project), "--backend", "cuda-ptx"])
    assert result.exit_code == 0
    ptx = project / ".manv" / "target" / "main.cuda.ptx"
    assert ptx.exists()
    ptx_source = ptx.read_text(encoding="utf-8")
    assert ".visible .entry" in ptx_source
    assert any(op in ptx_source for op in ["add.s32", "sub.s32", "mul.lo.s32", "ld.global.s32"])


def test_compile_parse_failure(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_parse_error")
    result = runner.invoke(app, ["compile", str(project)])
    assert result.exit_code == 1
    assert "expected ':'" in result.stderr


def test_compile_semantic_failure(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_semantic_error")
    result = runner.invoke(app, ["compile", str(project)])
    assert result.exit_code == 1
    assert "undefined variable 'y'" in result.stderr


def test_compile_break_outside_loop_failure(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_break_outside")
    result = runner.invoke(app, ["compile", str(project)])
    assert result.exit_code == 1
    assert "'break' is only valid inside loops" in result.stderr


def test_build_command_and_bundle_run(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "build_ok")
    build = runner.invoke(app, ["build", str(project)])
    assert build.exit_code == 0
    assert "[Build]" in build.stdout
    bundle = project / "dist" / "build_ok"
    run_file = bundle / "run.py"
    assert run_file.exists()
    proc = subprocess.run([sys.executable, str(run_file)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "Build Hello" in proc.stdout


def test_repl_command() -> None:
    script = "let x = 2\nx + 3\n:q\n"
    result = runner.invoke(app, ["repl"], input=script)
    assert result.exit_code == 0
    assert "5" in result.stdout


def test_test_command_runs_fixture_suite() -> None:
    result = runner.invoke(app, ["test", str(FIXTURES)])
    assert result.exit_code == 0
    assert "[Test]" in result.stdout
    assert "summary: passed=" in result.stdout

def test_repl_does_not_reexecute_previous_input() -> None:
    script = "let x = 1\nx + 1\nx + 2\n:q\n"
    result = runner.invoke(app, ["repl"], input=script)
    assert result.exit_code == 0

    values: list[str] = []
    for line in result.stdout.splitlines():
        match = re.search(r"(\d+)\s*$", line)
        if match and ">>>" in line:
            values.append(match.group(1))

    assert values == ["2", "3"]
