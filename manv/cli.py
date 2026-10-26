from __future__ import annotations

import argparse
import getpass
import io
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from . import __version__
from .builder import BuildResult, build_target, host_target_name
from .compiler import compile_pipeline_full, compile_target
from .device import render_selection_report
from .dap import DAPServer
from .diagnostics import ManvError, diag
from .gpu_dispatch import backend_selection_report
from .host import HostSelectionRequest, render_joint_backend_report, resolve_host_selection
from .project import discover_compile_target, discover_target, init_project
from .registry import (
    DEFAULT_REGISTRY_URL,
    add_dependency_entry,
    choose_registry_token,
    choose_registry_url,
    clear_registry_auth,
    ensure_manifest,
    infer_name_from_git_url,
    load_registry_auth,
    looks_like_git_spec,
    parse_registry_spec,
    resolve_registry_version,
    save_registry_auth,
)
from .repl import run_repl
from .runner import run_target
from .testing import run_e2e_suite


# Commands that don't need stdlib (meta-operations)
_STDLIB_EXEMPT_COMMANDS = frozenset({"setup", "auth", "version", "add", "install"})


class _Exit(Exception):
    """Stops the running command with an exit code (raised from helpers like `_fail`)."""

    def __init__(self, code: int = 0):
        super().__init__(code)
        self.code = code


def _check_stdlib(command: str | None) -> None:
    """Warn once per invocation if the ManV stdlib is not installed."""
    if command in _STDLIB_EXEMPT_COMMANDS:
        return
    try:
        from .packages import is_stdlib_installed, STDLIB_GIT_URL, STDLIB_PACKAGE_NAME
        if not is_stdlib_installed():
            _echo(
                f"warning: stdlib not installed. Run:\n"
                f"  manv setup\n"
                f"or manually:\n"
                f"  manv add {STDLIB_GIT_URL} --branch main",
                err=True,
            )
    except Exception:
        pass


def _echo(text: object = "", *, err: bool = False, nl: bool = True) -> None:
    print(text, end="\n" if nl else "", file=sys.stderr if err else sys.stdout)


def _prompt(text: str, *, default: str | None = None, hide_input: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    if hide_input:
        value = getpass.getpass(f"{text}{suffix}: ")
    else:
        _echo(f"{text}{suffix}: ", nl=False)
        sys.stdout.flush()
        value = sys.stdin.readline().rstrip("\r\n")
    return value if value or default is None else default


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return getattr(sys.stdout, "isatty", lambda: False)()


_ANSI_FG = {"red": 31, "green": 32, "yellow": 33, "cyan": 36, "bright_black": 90}


def _c(text: str, *, fg: str | None = None, bold: bool = False) -> str:
    """Wrap text in ANSI codes only when color is supported."""
    if not _supports_color():
        return text
    codes = (["1"] if bold else []) + ([str(_ANSI_FG[fg])] if fg else [])
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m" if codes else text


def _fail(err: ManvError) -> None:
    _echo(_c(err.render(), fg="red"), err=True)
    raise _Exit(1)


def _title(text: str) -> None:
    _echo(_c(f"[{text}]", bold=True, fg="cyan"))


def _kv(key: str, value: object) -> None:
    _echo(f"  {_c(key + ':', fg='bright_black')} {value}")


def _ok(text: str) -> None:
    _echo(_c("  " + text, fg="green", bold=True))


def _note(text: str) -> None:
    _echo(_c("  " + text, fg="yellow"))


def _requested_reports(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def _emit_backend_report(
    preferred_backend: str,
    *,
    host_backend: str = "auto",
    device: str | None = None,
    policy: str = "auto",
) -> None:
    _echo("[Report:backend]")
    host_report = resolve_host_selection(
        HostSelectionRequest(
            requested_host_backend=host_backend,
            policy=policy,
        )
    )
    device_report = backend_selection_report(preferred_backend, device=device, policy=policy)
    _echo(render_joint_backend_report(host_report, device_report), nl=False)


def _emit_kernelize_report(target: str | Path, *, optimize: bool = True, abi_target: str = "x86_64-sysv") -> None:
    context = discover_target(target)
    source = context.entry.read_text(encoding="utf-8")
    artifacts = compile_pipeline_full(
        source,
        str(context.entry),
        optimize=optimize,
        target_name=abi_target,
        capture_graph=False,
    )
    _echo("[Report:kernelize]")
    _echo(json.dumps(artifacts["gpu_report"], indent=2, sort_keys=True), nl=False)
    _echo()


def version() -> None:
    _echo(__version__)


def init(
    path: str,
    name: str | None,
    description: str | None,
    author: str | None,
    interactive: bool,
) -> None:
    if interactive:
        base_name = Path(path).resolve().name or "manv-project"
        name = _prompt("Project name", default=(name or base_name)).strip()
        description = _prompt("Description", default=(description or "")).strip()
        author = _prompt("Author", default=(author or "ManV Developer")).strip()

    try:
        ctx = init_project(
            path,
            name=name,
            description=description,
            author=author,
        )
    except ManvError as err:
        _fail(err)
    _title("Init")
    _kv("project", ctx.root)
    _kv("config", ctx.root / "project.toml")
    _kv("entry", ctx.entry)
    _kv("mode", "app")
    _kv("tests", ctx.root / "tests" / "e2e" / "hello_world" / "case.toml")
    _ok("status: initialized")


def run(
    target: str,
    mode: str,
    abi_target: str,
    backend: str,
    device: str | None,
    report: str | None,
    cuda: bool,
    cuda_arch: str,
    cuda_dump_kernels: bool,
    cuda_jit: bool,
    cuda_aot: bool,
    deterministic_gc: bool,
    gc_stress: bool,
    stable_debug_format: bool,
) -> None:
    del cuda, cuda_arch, cuda_dump_kernels, cuda_jit, cuda_aot
    reports = _requested_reports(report)
    if "backend" in reports:
        _emit_backend_report(
            backend,
            host_backend="interp" if mode == "interpreter" else "llvm",
            device=device,
            policy="run",
        )
    if "kernelize" in reports:
        _emit_kernelize_report(target, optimize=True, abi_target=abi_target)
    out = io.StringIO()
    try:
        code = run_target(
            target,
            stdout=out,
            mode=mode,
            target_name=abi_target,
            backend=backend,
            device=device,
            deterministic_gc=deterministic_gc,
            gc_stress=gc_stress,
            stable_debug_format=stable_debug_format,
        )
    except ManvError as err:
        _fail(err)
    text = out.getvalue()
    if text:
        _echo(text, nl=False)
    raise _Exit(code)


def compile_cmd(
    target: str,
    emit: str,
    out: str | None,
    host: str,
    device_backend: str,
    optimize: bool,
    abi_target: str,
    capture: bool,
    device: str | None,
    link_lib: list[str] | None,
    link_path: list[str] | None,
    link_arg: list[str] | None,
    report: str | None,
    cuda_arch: str,
    cuda_dump_kernels: bool,
    cuda_jit: bool,
    cuda_aot: bool,
) -> None:
    del cuda_jit, cuda_aot
    try:
        ctx = discover_compile_target(target)
        out_dir = Path(out).resolve() if out else ctx.target_dir
        reports = _requested_reports(report)
        if "backend" in reports:
            _emit_backend_report(
                "auto" if device_backend == "none" else device_backend,
                host_backend=host,
                device=device,
                policy="compile",
            )
        if "kernelize" in reports:
            _emit_kernelize_report(ctx.entry, optimize=optimize, abi_target=abi_target)
        emit_parts = [item.strip() for item in emit.split(",") if item.strip()]
        written = compile_target(
            ctx.entry,
            out_dir,
            emit=emit_parts,
            backend=device_backend,
            optimize=optimize,
            target_name=abi_target,
            capture_graph=capture,
            host_backend=host,
            cuda_arch=cuda_arch,
            cuda_dump_kernels=cuda_dump_kernels,
            link_libs=tuple(link_lib or []),
            link_paths=tuple(link_path or []),
            link_args=tuple(link_arg or []),
        )
    except ManvError as err:
        _fail(err)
    _title("Compile")
    _kv("source", ctx.entry)
    _kv("host_backend", host)
    _kv("target", abi_target)
    _kv("out_dir", out_dir)
    _echo(_c("  artifacts:", fg="bright_black"))
    for kind, path in written.items():
        _echo(f"    {_c(kind + ':', fg="bright_black"):<18} {path}")
    _ok("status: compiled")


def build(
    target: str,
    out: str | None,
    host: str,
    device_backend: str,
    device: str | None,
    report: str | None,
    portable_cache: bool,
    cuda_arch: str,
    cuda_dump_kernels: bool,
    cuda_jit: bool,
    cuda_aot: bool,
) -> None:
    del cuda_arch, cuda_dump_kernels, cuda_jit, cuda_aot
    reports = _requested_reports(report)
    if "backend" in reports:
        _emit_backend_report(device_backend, host_backend=host, device=device, policy="build")
    if "kernelize" in reports:
        _emit_kernelize_report(target)
    try:
        result = build_target(
            target,
            Path(out).resolve() if out else None,
            portable_cache=portable_cache,
            host_backend=host,
            device_backend=device_backend,
        )
    except ManvError as err:
        _fail(err)
        return
    _title("Build")
    _kv("artifact", result.path)
    _kv("target", host_target_name())
    _kv("host_backend", host)
    _kv("device_backend", device_backend)
    if result.was_rebuilt:
        _note(f"rebuilt  ({result.reason})")
        _ok("status: built")
    else:
        _ok("status: up to date")


def repl() -> None:
    try:
        code = run_repl(sys.stdin, sys.stdout)
    except ManvError as err:
        _fail(err)
        return
    raise _Exit(code)


def test(
    path: str,
    cuda: bool,
) -> None:
    del cuda
    result = run_e2e_suite(path)
    _title("Test")
    _echo(f"  {'result':<8} {'case':<28} detail")
    _echo("  " + "-" * 68)
    for case in result.results:
        case_name = f"'{case.name}'" if case.name else "<unnamed>"
        if case.passed:
            prefix = _c("PASS", fg="green", bold=True)
        else:
            prefix = _c("FAIL", fg="red", bold=True)
        _echo(f"  {prefix:<8} {case_name:<28} {case.message}")
    total = result.passed + result.failed
    _echo("  " + "-" * 68)
    summary = f"summary: passed={result.passed} failed={result.failed} total={total}"
    if result.failed:
        _echo(_c("  " + summary, fg="red"))
        raise _Exit(1)
    _echo(_c("  " + summary, fg="green"))


def auth_login(
    registry: str,
    token: str | None,
) -> None:
    final_token = (token or "").strip()
    if not final_token:
        final_token = _prompt("Registry token", hide_input=True).strip()
    if not final_token:
        _fail(ManvError(diag("E8201", "token cannot be empty", "auth", 1, 1)))

    session = save_registry_auth(registry=registry, token=final_token)
    _title("Registry Auth")
    _ok("status: logged_in")
    _kv("registry", session.registry)
    _kv("saved_at", session.saved_at)


def auth_status() -> None:
    session = load_registry_auth()
    _title("Registry Auth")
    if session is None:
        _echo(_c("  status: logged_out", fg="bright_black"))
        return

    masked = "*" * max(0, len(session.token) - 4) + session.token[-4:]
    _ok("status: logged_in")
    _kv("registry", session.registry)
    _kv("token", masked)
    _kv("saved_at", session.saved_at)


def auth_logout() -> None:
    removed = clear_registry_auth()
    _title("Registry Auth")
    _echo(_c("  status: logged_out", fg="bright_black"))
    _kv("cleared", removed)


def add(
    spec: str,
    target: str,
    git: str | None,
    version: str | None,
    registry: str | None,
    branch: str | None,
    tag: str | None,
    rev: str | None,
) -> None:
    try:
        ctx = discover_target(target)

        manifest_path = ctx.config_path or (ctx.root / "project.toml")
        try:
            entry_rel = str(ctx.entry.relative_to(ctx.root))
        except ValueError:
            entry_rel = ctx.entry.name
        ensure_manifest(manifest_path, project_name=ctx.name, entry_rel=entry_rel)

        selected_refs = [value for value in [branch, tag, rev] if value]
        if len(selected_refs) > 1:
            raise ManvError(diag("E8202", "use only one of --branch, --tag, --rev", str(manifest_path), 1, 1))

        git_url = git.strip() if git else None
        if git_url is None and looks_like_git_spec(spec):
            git_url = spec.strip()

        if git_url is not None:
            dep_name = spec.strip() if git else infer_name_from_git_url(git_url)
            payload: dict[str, str] = {"git": git_url}
            if rev:
                payload["rev"] = rev
            elif tag:
                payload["tag"] = tag
            elif branch:
                payload["branch"] = branch

            add_dependency_entry(manifest_path, dependency_name=dep_name, payload=payload)

            _title("Add")
            _kv("project", ctx.root)
            _kv("dependency", dep_name)
            _kv("source", "git")
            _kv("git", git_url)
            if rev:
                _kv("rev", rev)
            if tag:
                _kv("tag", tag)
            if branch:
                _kv("branch", branch)
            _kv("manifest", manifest_path)

            try:
                from .packages import install_dependency
                version_key = install_dependency(
                    dep_name,
                    payload,
                    registry_url=choose_registry_url(registry),
                    token=choose_registry_token(),
                )
                _kv("installed", version_key)
            except Exception as exc:
                _echo(f"  warning: install failed: {exc}", err=True)
            return

        dep_name, version_from_spec = parse_registry_spec(spec)
        registry_url = choose_registry_url(registry)
        token = choose_registry_token()
        resolved_version = resolve_registry_version(
            dep_name,
            requested=version or version_from_spec,
            registry_url=registry_url,
            token=token,
        )

        payload = {
            "version": resolved_version,
            "registry": registry_url,
        }
        add_dependency_entry(manifest_path, dependency_name=dep_name, payload=payload)

        _title("Add")
        _kv("project", ctx.root)
        _kv("dependency", dep_name)
        _kv("source", "registry")
        _kv("registry", registry_url)
        _kv("version", resolved_version)
        _kv("manifest", manifest_path)

        try:
            from .packages import install_dependency
            version_key = install_dependency(
                dep_name,
                payload,
                registry_url=registry_url,
                token=token,
            )
            _kv("installed", version_key)
        except Exception as exc:
            _echo(f"  warning: install failed: {exc}", err=True)
    except (ManvError, ValueError) as err:
        if isinstance(err, ManvError):
            _fail(err)
        _fail(ManvError(diag("E8203", str(err), str(target), 1, 1)))


def install(
    target: str,
    registry: str | None,
) -> None:
    """Install all dependencies declared in project.toml into the global cache."""
    import tomllib as _tomllib
    from .packages import install_dependency

    try:
        ctx = discover_target(target)
        manifest = ctx.config_path or (ctx.root / "project.toml")
        if not manifest.exists():
            _echo("no project.toml found", err=True)
            raise _Exit(1)

        data = _tomllib.loads(manifest.read_text(encoding="utf-8"))
        tool_manv = data.get("tool", {}).get("manv", {})
        deps: dict = tool_manv.get("dependencies", data.get("dependencies", {}))  # type: ignore[assignment]
        if not isinstance(deps, dict) or not deps:
            _echo("no dependencies declared")
            return

        reg_url = choose_registry_url(registry)
        token = choose_registry_token()

        _title("Install")
        for dep_name, dep_entry in deps.items():
            try:
                version_key = install_dependency(dep_name, dep_entry, registry_url=reg_url, token=token)
                _kv(dep_name, f"{version_key} (installed)")
            except Exception as exc:
                _kv(dep_name, f"FAILED: {exc}")
        _echo("status: done")
    except ManvError as err:
        _fail(err)


def setup() -> None:
    """Install the ManV stdlib from GitHub (https://github.com/manv-lang/stdlib).

    Run this once after installing ManV.  Equivalent to:
      manv add https://github.com/manv-lang/stdlib --branch main
    """
    from .packages import (
        STDLIB_GIT_URL,
        STDLIB_PACKAGE_NAME,
        install_from_github,
        is_stdlib_installed,
    )

    if is_stdlib_installed():
        _echo("stdlib is already installed.")
        return

    _title("Setup")
    _kv("stdlib", STDLIB_GIT_URL)
    _echo("downloading stdlib...")
    try:
        version_key = install_from_github(
            STDLIB_PACKAGE_NAME,
            STDLIB_GIT_URL,
            "main",
            ref_kind="branch",
        )
        _kv("installed", version_key)
        _echo("status: done")
    except Exception as exc:
        _echo(f"error: could not install stdlib: {exc}", err=True)
        raise _Exit(1)


def dap(
    transport: str,
    host: str,
    port: int,
) -> None:
    server = DAPServer()
    if transport == "stdio":
        server.start_stdio()
        return
    if transport == "tcp":
        server.start_tcp(host=host, port=port)
        return
    _fail(ManvError(diag("E8401", f"unsupported dap transport: {transport}", "dap", 1, 1)))



def lsp(
    transport: str,
    host: str,
    port: int,
) -> None:
    try:
        from .lsp_server import start_stdio as lsp_start_stdio
        from .lsp_server import start_tcp as lsp_start_tcp
    except Exception as err:
        _fail(ManvError(diag("E8501", f"failed to load LSP server: {err}", "lsp", 1, 1)))
        return

    if transport == "stdio":
        lsp_start_stdio()
        return
    if transport == "tcp":
        lsp_start_tcp(host=host, port=port)
        return
    _fail(ManvError(diag("E8502", f"unsupported lsp transport: {transport}", "lsp", 1, 1)))

class _HelpFormatter(argparse.HelpFormatter):
    """Default argparse help plus a `(default: ...)` suffix for meaningful defaults."""

    def _get_help_string(self, action: argparse.Action) -> str:
        text = action.help or ""
        shown = action.default not in (None, False, argparse.SUPPRESS)
        if shown and (action.option_strings or action.nargs == "?") and "%(default)" not in text:
            text += " (default: %(default)s)"
        return text


def _flag(parser: argparse.ArgumentParser, name: str, *, default: bool, help: str, negatable: bool) -> None:
    """Add a boolean option. `negatable` also accepts `--no-<name>`."""
    if negatable:
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default, help=help)
    else:
        parser.add_argument(f"--{name}", action="store_true", default=default, help=help)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manv",
        description="ManV language toolchain",
        formatter_class=_HelpFormatter,
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name: str, func, help: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help, description=help, formatter_class=_HelpFormatter, allow_abbrev=False)
        if func is not None:
            sub.set_defaults(func=func)
        return sub

    command("version", version, "print the ManV version")

    p = command("init", init, "scaffold a new project")
    p.add_argument("path", nargs="?", default=".", help="project directory")
    p.add_argument("--name", help="project.name for project.toml")
    p.add_argument("--description", help="project description")
    p.add_argument("--author", help="author display name")
    _flag(p, "interactive", default=False, negatable=False, help="prompt for project metadata (Poetry-style init flow)")

    p = command("run", run, "run a program")
    p.add_argument("target", nargs="?", default=".", help="project directory or .mv file")
    p.add_argument("--mode", default="interpreter", help="execution mode: interpreter|compiled")
    p.add_argument("--target", dest="abi_target", metavar="TARGET", default="x86_64-sysv", help="cpu target ABI")
    p.add_argument("--backend", default="auto", help="runtime backend preference: auto,cuda,rocm,level0,vulkan-spv,directx,webgpu,cpu")
    p.add_argument("--device", help="preferred runtime device id")
    p.add_argument("--report", help="comma separated report kinds: backend,kernelize")
    _flag(p, "cuda", default=False, negatable=False, help="enable CUDA-aware execution paths when available")
    p.add_argument("--cuda-arch", default="sm_80", help="CUDA architecture for JIT compilation")
    _flag(p, "cuda-dump-kernels", default=False, negatable=False, help="dump generated CUDA source alongside PTX")
    _flag(p, "cuda-jit", default=True, negatable=True, help="prefer JIT compilation for CUDA kernels")
    _flag(p, "cuda-aot", default=False, negatable=False, help="request AOT-style artifact emission where supported")
    _flag(p, "deterministic-gc", default=False, negatable=True, help="enable deterministic GC checkpoints")
    _flag(p, "gc-stress", default=False, negatable=True, help="force GC at every safe point")
    _flag(p, "stable-debug-format", default=False, negatable=True, help="stable map/object formatting for debugging")

    p = command("compile", compile_cmd, "compile a program and emit artifacts")
    p.add_argument("target", nargs="?", default=".", help="single .mv file or directory containing main.mv")
    p.add_argument(
        "--emit",
        default="native_exe",
        help=(
            "comma separated list: ast,hir,hlir,graph,capture,kernel,kernel_exec,"
            "abi,host_stub_abi,asm,host_stub,source_map,gpu_report,backend_bundle,ptx,cuda_cpp,hip,spirv,wgsl,hlsl,llvm_ir,native_obj,native_exe"
        ),
    )
    p.add_argument("--out", help="output directory override")
    p.add_argument("--host", default="auto", help="host backend: auto,llvm,interp")
    p.add_argument(
        "--device-backend", "--backend", dest="device_backend", default="none",
        help="device backend target: auto,none,cuda,rocm,level0,vulkan-spv,webgpu,directx,cpu",
    )
    _flag(p, "optimize", default=True, negatable=True, help="enable Graph IR optimization")
    p.add_argument("--target", dest="abi_target", metavar="TARGET", default=host_target_name(), help="cpu target ABI")
    _flag(p, "capture", default=False, negatable=True, help="capture Graph IR by tracing HLIR execution")
    p.add_argument("--device", help="preferred runtime device id for reports")
    p.add_argument("--link-lib", action="append", help="native linker library name (repeatable)")
    p.add_argument("--link-path", action="append", help="native linker search path (repeatable)")
    p.add_argument("--link-arg", action="append", help="raw native linker argument (repeatable)")
    p.add_argument("--report", help="comma separated report kinds: backend,kernelize")
    p.add_argument("--cuda-arch", default="sm_80", help="CUDA architecture for JIT compilation")
    _flag(p, "cuda-dump-kernels", default=False, negatable=False, help="dump generated CUDA source")
    _flag(p, "cuda-jit", default=True, negatable=True, help="prefer JIT compilation for CUDA kernels")
    _flag(p, "cuda-aot", default=False, negatable=False, help="request AOT-style artifact emission where supported")

    p = command("build", build, "build a distribution artifact")
    p.add_argument("target", nargs="?", default=".", help="project directory or .mv file")
    p.add_argument("--out", help="dist directory override")
    p.add_argument("--host", default="auto", help="host backend: auto,llvm,interp")
    p.add_argument(
        "--device-backend", "--backend", dest="device_backend", default="auto",
        help="device backend preference: auto,none,cuda,rocm,level0,vulkan-spv,directx,webgpu,cpu",
    )
    p.add_argument("--device", help="preferred runtime device id for reports")
    p.add_argument("--report", help="comma separated report kinds: backend,kernelize")
    _flag(p, "portable-cache", default=False, negatable=False, help="extract bundled program payload next to the executable")
    p.add_argument("--cuda-arch", default="sm_80", help="CUDA architecture for emitted artifacts")
    _flag(p, "cuda-dump-kernels", default=False, negatable=False, help="dump generated CUDA source")
    _flag(p, "cuda-jit", default=True, negatable=True, help="prefer JIT compilation for CUDA kernels")
    _flag(p, "cuda-aot", default=False, negatable=False, help="request AOT-style artifact emission where supported")

    command("repl", repl, "start the interactive REPL")

    p = command("test", test, "run the end-to-end test suite")
    p.add_argument("path", nargs="?", default=".", help="project root or fixtures root")
    _flag(p, "cuda", default=False, negatable=False, help="run CUDA-aware tests when the environment supports them")

    p = command("add", add, "add a package dependency")
    p.add_argument("spec", help="name[@version] or git URL")
    p.add_argument("target", nargs="?", default=".", help="project directory")
    p.add_argument("--git", help="git repository URL override")
    p.add_argument("--version", help="registry version override")
    p.add_argument("--registry", help="registry base URL override")
    p.add_argument("--branch", help="git branch")
    p.add_argument("--tag", help="git tag")
    p.add_argument("--rev", help="git commit/revision")

    p = command("install", install, "install all dependencies declared in project.toml into the global cache")
    p.add_argument("target", nargs="?", default=".", help="project directory")
    p.add_argument("--registry", help="registry base URL override")

    command("setup", setup, "install the ManV stdlib from GitHub (https://github.com/manv-lang/stdlib)")

    p = command("dap", dap, "start a Debug Adapter Protocol server")
    p.add_argument("--transport", default="stdio", help="debug adapter transport: stdio|tcp")
    p.add_argument("--host", default="127.0.0.1", help="tcp bind host")
    p.add_argument("--port", type=int, default=4711, help="tcp bind port")

    p = command("lsp", lsp, "start a Language Server Protocol server")
    p.add_argument("--transport", default="stdio", help="language server transport: stdio|tcp")
    p.add_argument("--host", default="127.0.0.1", help="tcp bind host")
    p.add_argument("--port", type=int, default=2087, help="tcp bind port")

    auth = command("auth", None, "registry authentication")
    auth_commands = auth.add_subparsers(dest="auth_command", required=True, metavar="COMMAND")

    def auth_command(name: str, func, help: str) -> argparse.ArgumentParser:
        sub = auth_commands.add_parser(name, help=help, description=help, formatter_class=_HelpFormatter, allow_abbrev=False)
        sub.set_defaults(func=func)
        return sub

    p = auth_command("login", auth_login, "save registry credentials")
    p.add_argument("--registry", default=DEFAULT_REGISTRY_URL, help="registry base URL")
    p.add_argument("--token", help="registry token")
    auth_command("status", auth_status, "show the current registry login")
    auth_command("logout", auth_logout, "remove saved registry credentials")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ManV CLI and return the process exit code."""
    parser = build_parser()
    try:
        ns = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits on --help (0) and usage errors (2)
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)

    params = {k: v for k, v in vars(ns).items() if k not in {"command", "auth_command", "func"}}
    _check_stdlib(ns.command)
    try:
        return ns.func(**params) or 0
    except _Exit as exc:
        return exc.code
    except KeyboardInterrupt:
        _echo("Aborted!", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
