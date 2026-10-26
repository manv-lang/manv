from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib
from dataclasses import dataclass

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manv.builder import host_target_name
from manv.cli import main
from manv.native_toolchain import detect_toolchain
from manv.llvm_toolchain import detect_llvm_toolchain


@dataclass
class CliResult:
    exit_code: int
    stdout: str
    stderr: str
    exception: BaseException | None = None


def invoke(args: list[str], *, env: dict[str, str] | None = None, input: str | None = None) -> CliResult:
    """Run the CLI in-process with captured stdio, a temporary environment and optional stdin."""
    out, err = io.StringIO(), io.StringIO()
    saved_io = (sys.stdout, sys.stderr, sys.stdin)
    saved_env = {key: os.environ.get(key) for key in (env or {})}
    exception: BaseException | None = None
    sys.stdout, sys.stderr, sys.stdin = out, err, io.StringIO(input or "")
    os.environ.update(env or {})
    try:
        exit_code = main(list(args))
    except Exception as exc:  # an uncaught error is a failed run, like a crashed process
        exit_code, exception = 1, exc
    finally:
        sys.stdout, sys.stderr, sys.stdin = saved_io
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return CliResult(exit_code, out.getvalue(), err.getvalue(), exception)


FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixture(tmp_path: Path, name: str) -> Path:
    source = FIXTURES / name
    target = tmp_path / name
    shutil.copytree(source, target)
    return target




def test_init_command(tmp_path: Path) -> None:
    target = tmp_path / "demo"
    result = invoke(["init", str(target)])
    assert result.exit_code == 0
    assert "[Init]" in result.stdout
    assert "status: initialized" in result.stdout
    assert (target / "project.toml").exists()
    assert (target / ".gitignore").exists()
    assert (target / "src" / "main.mv").exists()
    assert ".manv/" in (target / ".gitignore").read_text(encoding="utf-8")

    suite = invoke(["test", str(target)])
    assert suite.exit_code == 0
    assert "summary: passed=1 failed=0" in suite.stdout


def test_init_command_with_metadata_flags(tmp_path: Path) -> None:
    target = tmp_path / "demo_meta"
    result = invoke([
            "init",
            str(target),
            "--name",
            "demo-pkg",
            "--description",
            "Demo package",
            "--author",
            "Jane Dev",
        ],
    )
    assert result.exit_code == 0

    config = tomllib.loads((target / "project.toml").read_text(encoding="utf-8"))
    assert config["project"]["name"] == "demo-pkg"
    assert config["project"]["description"] == "Demo package"
    assert config["project"]["authors"][0]["name"] == "Jane Dev"
    # verify module declaration was written to main.mv (hyphens normalized to underscores)
    main_src = (target / "src" / "main.mv").read_text(encoding="utf-8")
    assert main_src.startswith("module demo_pkg")


@pytest.mark.parametrize(
    ("project_name", "module_name"),
    [
        ("demo-pkg", "demo_pkg"),
        ("My Proj", "My_Proj"),
        ("my.cool project", "my_cool_project"),
        ("1app", "_1app"),
        ("type", "type_"),
        ("h\u00e9llo w\u00f6rld", "h_llo_w_rld"),
        ("---", "app"),
    ],
)
def test_init_command_generates_a_valid_module_name(tmp_path: Path, project_name: str, module_name: str) -> None:
    target = tmp_path / "proj"
    created = invoke(["init", str(target), f"--name={project_name}"])
    assert created.exit_code == 0, created.stderr

    main_src = (target / "src" / "main.mv").read_text(encoding="utf-8")
    assert main_src.startswith(f"module {module_name}\n")
    # The project name itself is kept verbatim in project.toml.
    assert tomllib.loads((target / "project.toml").read_text(encoding="utf-8"))["project"]["name"] == project_name

    # The scaffold must actually parse and run, whatever the name was.
    ran = invoke(["run", str(target)])
    assert ran.exit_code == 0, ran.stderr
    assert "Hello, World!" in ran.stdout


def test_auth_login_status_logout(tmp_path: Path) -> None:
    auth_file = tmp_path / "manv_auth.json"
    env = {"MANV_AUTH_FILE": str(auth_file)}

    login = invoke(["auth", "login", "--registry", "https://registry.example.com", "--token", "tok_live_test_1234"],
        env=env,
    )
    assert login.exit_code == 0
    assert "logged_in" in login.stdout
    assert auth_file.exists()

    status = invoke(["auth", "status"], env=env)
    assert status.exit_code == 0
    assert "https://registry.example.com" in status.stdout
    assert "1234" in status.stdout

    logout = invoke(["auth", "logout"], env=env)
    assert logout.exit_code == 0
    assert "logged_out" in logout.stdout

    status_after = invoke(["auth", "status"], env=env)
    assert status_after.exit_code == 0
    assert "logged_out" in status_after.stdout


def _make_fake_tarball(filename: str, content: bytes) -> bytes:
    """Helper: build a minimal .tar.gz with src/{filename}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=f"src/{filename}")
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def test_add_registry_dependency(tmp_path: Path) -> None:
    from unittest.mock import patch

    project = _copy_fixture(tmp_path, "compile_ok")
    auth_file = tmp_path / "manv_auth.json"
    manv_home = tmp_path / "manv_home"
    env = {"MANV_AUTH_FILE": str(auth_file), "MANV_HOME": str(manv_home)}

    login = invoke(["auth", "login", "--registry", "https://registry.example.com", "--token", "tok_live_test_1234"],
        env=env,
    )
    assert login.exit_code == 0

    fake_archive = _make_fake_tarball("tensorx.mv", b'fn hello() -> str:\n    return "hello"\n')
    with patch("manv.registry.download_package_archive", return_value=fake_archive):
        result = invoke(["add", "tensorx@1.2.3", str(project)], env=env)

    assert result.exit_code == 0
    assert "source: registry" in result.stdout

    manifest = tomllib.loads((project / "project.toml").read_text(encoding="utf-8"))
    deps = manifest.get("tool", {}).get("manv", {}).get("dependencies", {})
    assert "tensorx" in deps
    assert deps["tensorx"]["version"] == "1.2.3"
    assert deps["tensorx"]["registry"] == "https://registry.example.com"

    # Package was installed into the global cache
    assert (manv_home / "packages" / "tensorx" / "1.2.3" / "src" / "tensorx.mv").exists()


def test_add_git_dependency(tmp_path: Path) -> None:
    from unittest.mock import patch

    project = _copy_fixture(tmp_path, "compile_ok")
    manv_home = tmp_path / "manv_home"
    env = {"MANV_HOME": str(manv_home)}

    fake_archive = _make_fake_tarball("sort.mv", b"fn sort(arr) -> array:\n    return arr\n")

    class _FakeResp:
        def read(self) -> bytes:
            return fake_archive
        def __enter__(self) -> _FakeResp:
            return self
        def __exit__(self, *_: object) -> None:
            pass

    with patch("manv.packages.urllib.request.urlopen", return_value=_FakeResp()):
        result = invoke(["add", "https://github.com/manv-lang/math.git", str(project), "--branch", "main"],
            env=env,
        )

    assert result.exit_code == 0
    assert "source: git" in result.stdout

    manifest = tomllib.loads((project / "project.toml").read_text(encoding="utf-8"))
    deps = manifest.get("tool", {}).get("manv", {}).get("dependencies", {})
    assert "math" in deps
    assert deps["math"]["git"] == "https://github.com/manv-lang/math.git"
    assert deps["math"]["branch"] == "main"

    # Package was installed into the cache under a git-{hash} version key
    cache_pkg = manv_home / "packages" / "math"
    installed = list(cache_pkg.iterdir()) if cache_pkg.exists() else []
    assert any(v.name.startswith("git-") for v in installed)


def test_install_command(tmp_path: Path) -> None:
    from unittest.mock import patch

    manv_home = tmp_path / "manv_home"
    env = {"MANV_HOME": str(manv_home)}

    project = tmp_path / "proj"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "main.mv").write_text("module proj\nfn main() -> int:\n    return 0\n")
    (project / "project.toml").write_text(
        "[project]\nname = \"proj\"\nversion = \"0.1.0\"\n\n"
        "[tool.manv]\nentry = \"src/main.mv\"\n\n"
        "[tool.manv.dependencies]\nmath = {version = \"1.0.0\", registry = \"https://registry.example.com\"}\n"
    )

    fake_archive = _make_fake_tarball("math.mv", b"fn pi() -> float:\n    return 3.14159\n")
    with patch("manv.registry.download_package_archive", return_value=fake_archive):
        result = invoke(["install", str(project)], env=env)

    assert result.exit_code == 0
    assert "math" in result.stdout
    assert (manv_home / "packages" / "math" / "1.0.0" / "src" / "math.mv").exists()


def test_setup_installs_stdlib(tmp_path: Path) -> None:
    from unittest.mock import patch

    manv_home = tmp_path / "manv_home"
    env = {"MANV_HOME": str(manv_home)}

    fake_archive = _make_fake_tarball("math.mv", b"fn pi() -> float:\n    return 3.14159\n")

    class _FakeResp:
        def read(self) -> bytes:
            return fake_archive
        def __enter__(self) -> _FakeResp:
            return self
        def __exit__(self, *_: object) -> None:
            pass

    with patch("manv.packages.urllib.request.urlopen", return_value=_FakeResp()):
        result = invoke(["setup"], env=env)

    assert result.exit_code == 0
    assert "installed" in result.stdout
    cache_std = manv_home / "packages" / "std"
    installed = list(cache_std.iterdir()) if cache_std.exists() else []
    assert any(v.name.startswith("git-") for v in installed)


def test_run_command(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "run_hello")
    result = invoke(["run", str(project)])
    assert result.exit_code == 0
    assert "Hello, World!" in result.stdout


def test_run_core_language_features(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "run_core_features")
    result = invoke(["run", str(project)])
    assert result.exit_code == 0
    for expected in ["15", "10", "8", "9", "True"]:
        assert expected in result.stdout


def test_run_backend_report(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "run_hello")
    result = invoke(["run", str(project), "--report", "backend"])
    assert result.exit_code == 0
    assert "[Report:backend]" in result.stdout
    assert '"selected_backend"' in result.stdout
    assert '"requested_host_backend"' in result.stdout
    assert '"resolved_host_backend"' in result.stdout


def test_compile_outputs_all_ir(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_ok")
    result = invoke(["compile", str(project / "src" / "main.mv"), "--emit", "ast,hir,graph,kernel"])
    assert result.exit_code == 0
    assert "[Compile]" in result.stdout
    for name in ["main.ast.json", "main.hir.json", "main.graph.json", "main.kernel.json"]:
        assert (project / "src" / ".manv" / "target" / name).exists()
    graph_payload = json.loads((project / "src" / ".manv" / "target" / "main.graph.json").read_text(encoding="utf-8"))
    assert graph_payload["kind"] == "tensor_dag"
    assert graph_payload["optimization"]["constant_folding"] >= 1
    assert graph_payload["optimization"]["dead_nodes_removed"] >= 1


def test_compile_cuda_ptx_backend(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_ok")
    result = invoke(["compile", str(project / "src" / "main.mv"), "--backend", "cuda-ptx"])
    assert result.exit_code == 0
    ptx = project / "src" / ".manv" / "target" / "main.cuda.ptx"
    assert ptx.exists()
    ptx_source = ptx.read_text(encoding="utf-8")
    assert ptx_source.strip()
    assert ".visible .entry" in ptx_source or "PTX unavailable on this machine" in ptx_source


def test_compile_default_emit_is_native_exe(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_ok")
    result = invoke(["compile", str(project / "src" / "main.mv")])
    # Must succeed (or fail only due to missing LLVM toolchain, not emit config).
    if result.exit_code != 0:
        assert "no LLVM toolchain found" in result.stderr
        return
    assert "[Compile]" in result.stdout
    out_dir = project / "src" / ".manv" / "target"
    # No intermediate IR artifacts should appear with the default emit.
    for ir_suffix in ["ast.json", "hir.json", "hlir.json", "graph.json", "kernel.json"]:
        assert not any(out_dir.glob(f"*.{ir_suffix}")), f"unexpected {ir_suffix} in default emit"


def test_compile_parse_failure(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_parse_error")
    result = invoke(["compile", str(project / "src" / "main.mv")])
    assert result.exit_code == 1
    assert "expected ':'" in result.stderr


def test_compile_semantic_failure(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_semantic_error")
    result = invoke(["compile", str(project / "src" / "main.mv")])
    assert result.exit_code == 1
    assert "undefined variable 'y'" in result.stderr


def test_compile_break_outside_loop_failure(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_break_outside")
    result = invoke(["compile", str(project / "src" / "main.mv")])
    assert result.exit_code == 1
    assert "'break' is only valid inside loops" in result.stderr


def test_build_command_and_bundle_run(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "build_ok")
    build = invoke(["build", str(project)])
    if detect_llvm_toolchain() is None:
        assert build.exit_code == 1
        assert "no LLVM toolchain found (clang)" in build.stderr
        return

    assert build.exit_code == 0
    assert "[Build]" in build.stdout
    shutil.rmtree(project / "src")
    native_exe = project / "build" / host_target_name() / ("build_ok.exe" if sys.platform == "win32" else "build_ok")
    assert native_exe.exists()
    proc = subprocess.run([str(native_exe)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "Build Hello" in proc.stdout


def test_build_host_interp_keeps_mvz(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "build_ok")
    build = invoke(["build", str(project), "--host", "interp"])
    assert build.exit_code == 0
    run_file = project / "build" / host_target_name() / "build_ok.mvz"
    assert run_file.exists()


def test_build_incremental_skips_rebuild(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "build_ok")
    if detect_llvm_toolchain() is None:
        import pytest
        pytest.skip("no LLVM toolchain")

    # First build
    r1 = invoke(["build", str(project)])
    assert r1.exit_code == 0
    assert "status: built" in r1.stdout

    # Second build — nothing changed
    r2 = invoke(["build", str(project)])
    assert r2.exit_code == 0
    assert "up to date" in r2.stdout

    # Modify source
    entry = project / "src" / "main.mv"
    entry.write_text(entry.read_text() + "\n# touched\n")

    # Third build — should rebuild
    r3 = invoke(["build", str(project)])
    assert r3.exit_code == 0
    assert "status: built" in r3.stdout


def test_compile_host_report_and_native_fallback(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_ok")
    result = invoke(["compile", str(project / "src" / "main.mv"), "--report", "backend"])
    assert '"resolved_host_backend"' in result.stdout
    if detect_llvm_toolchain() is None:
        assert result.exit_code == 1
        assert "no LLVM toolchain found (clang)" in result.stderr
        return

    assert result.exit_code == 0
    native_exe = project / "src" / ".manv" / "target" / ("main.exe" if sys.platform == "win32" else "main")
    if detect_toolchain() is not None:
        assert native_exe.exists()


def test_compile_accepts_directory_with_root_main_file(tmp_path: Path) -> None:
    root = tmp_path / "single"
    root.mkdir()
    (root / "main.mv").write_text(
        "fn main() -> int:\n"
        "    return 0\n",
        encoding="utf-8",
    )

    result = invoke(["compile", str(root), "--emit", "ast"])
    assert result.exit_code == 0
    assert "[Compile]" in result.stdout
    assert (root / ".manv" / "target" / "main.ast.json").exists()


def test_compile_rejects_project_root_and_points_to_build(tmp_path: Path) -> None:
    project = _copy_fixture(tmp_path, "compile_ok")
    result = invoke(["compile", str(project)])
    assert result.exit_code == 1
    assert "use 'manv build' for projects or pass src/main.mv explicitly" in result.stderr


def test_repl_command() -> None:
    script = "let x = 2\nx + 3\n:q\n"
    result = invoke(["repl"], input=script)
    assert result.exit_code == 0
    assert "5" in result.stdout


def test_test_command_runs_fixture_suite() -> None:
    result = invoke(["test", str(FIXTURES)])
    assert result.exit_code == 0
    assert "[Test]" in result.stdout
    assert "summary: passed=" in result.stdout

def test_repl_does_not_reexecute_previous_input() -> None:
    script = "let x = 1\nx + 1\nx + 2\n:q\n"
    result = invoke(["repl"], input=script)
    assert result.exit_code == 0

    values: list[str] = []
    for line in result.stdout.splitlines():
        match = re.search(r"(\d+)\s*$", line)
        if match and ">>>" in line:
            values.append(match.group(1))

    assert values == ["2", "3"]
