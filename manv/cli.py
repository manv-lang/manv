from __future__ import annotations

import io
from pathlib import Path
import sys
from typing import Annotated

import typer

from . import __version__
from .builder import build_target
from .compiler import compile_target
from .dap import DAPServer
from .diagnostics import ManvError, diag
from .project import discover_target, init_project
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


app = typer.Typer(help="ManV language toolchain")
auth_app = typer.Typer(help="Registry authentication")
app.add_typer(auth_app, name="auth")


def _fail(err: ManvError) -> None:
    typer.echo(err.render(), err=True)
    raise typer.Exit(code=1)


def _title(text: str) -> None:
    typer.echo(f"== {text} ==")


def _kv(key: str, value: object) -> None:
    typer.echo(f"  {key:<12} {value}")


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def init(path: Annotated[str, typer.Argument(help="project directory")] = ".") -> None:
    try:
        ctx = init_project(path)
    except ManvError as err:
        _fail(err)
    _title("Init")
    _kv("project", ctx.root)
    _kv("config", ctx.root / "manv.toml")
    _kv("entry", ctx.entry)
    _kv("tests", ctx.root / "tests" / "e2e" / "hello_world" / "case.toml")
    typer.echo("status: initialized")


@app.command()
def run(
    target: Annotated[str, typer.Argument(help="project directory or .mv file")] = ".",
    mode: Annotated[str, typer.Option(help="execution mode: interpreter|compiled")] = "interpreter",
    abi_target: Annotated[str, typer.Option("--target", help="cpu target ABI")] = "x86_64-sysv",
    deterministic_gc: Annotated[bool, typer.Option(help="enable deterministic GC checkpoints")] = False,
    gc_stress: Annotated[bool, typer.Option(help="force GC at every safe point")] = False,
    stable_debug_format: Annotated[bool, typer.Option(help="stable map/object formatting for debugging")] = False,
) -> None:
    out = io.StringIO()
    try:
        code = run_target(
            target,
            stdout=out,
            mode=mode,
            target_name=abi_target,
            deterministic_gc=deterministic_gc,
            gc_stress=gc_stress,
            stable_debug_format=stable_debug_format,
        )
    except ManvError as err:
        _fail(err)
    text = out.getvalue()
    if text:
        typer.echo(text, nl=False)
    raise typer.Exit(code=code)


@app.command("compile")
def compile_cmd(
    target: Annotated[str, typer.Argument(help="project directory or .mv file")] = ".",
    emit: Annotated[
        str,
        typer.Option(
            help=(
                "comma separated list: ast,hir,hlir,graph,capture,kernel,kernel_exec,"
                "abi,host_stub_abi,asm,host_stub,source_map,backend_bundle,ptx,hip,msl,spirv,wgsl,opencl,hlsl,native_obj,native_exe"
            )
        ),
    ] = "ast,hir,hlir,graph,kernel,abi,asm,host_stub",
    out: Annotated[str | None, typer.Option(help="output directory override")] = None,
    backend: Annotated[str, typer.Option(help="backend target: none,cuda,rocm,metal,vulkan-spv,webgpu,opencl,directx,cpu-ref")] = "none",
    optimize: Annotated[bool, typer.Option(help="enable Graph IR optimization")] = True,
    abi_target: Annotated[str, typer.Option("--target", help="cpu target ABI")] = "x86_64-sysv",
    capture: Annotated[bool, typer.Option(help="capture Graph IR by tracing HLIR execution")] = False,
) -> None:
    try:
        ctx = discover_target(target)
        out_dir = Path(out).resolve() if out else ctx.target_dir
        emit_parts = [item.strip() for item in emit.split(",") if item.strip()]
        written = compile_target(
            ctx.entry,
            out_dir,
            emit=emit_parts,
            backend=backend,
            optimize=optimize,
            target_name=abi_target,
            capture_graph=capture,
        )
    except ManvError as err:
        _fail(err)
    _title("Compile")
    _kv("source", ctx.entry)
    _kv("backend", backend)
    _kv("target", abi_target)
    _kv("optimize", optimize)
    _kv("capture", capture)
    _kv("out_dir", out_dir)
    typer.echo("artifacts:")
    for kind, path in written.items():
        typer.echo(f"  - {kind:<12} {path}")
        

@app.command()
def build(
    target: Annotated[str, typer.Argument(help="project directory or .mv file")] = ".",
    out: Annotated[str | None, typer.Option(help="dist directory override")] = None,
) -> None:
    try:
        bundle = build_target(target, Path(out).resolve() if out else None)
    except ManvError as err:
        _fail(err)
    _title("Build")
    _kv("bundle", bundle)
    _kv("run", bundle / "run.py")
    typer.echo("status: built")


@app.command()
def repl() -> None:
    try:
        code = run_repl(sys.stdin, sys.stdout)
    except ManvError as err:
        _fail(err)
        return
    raise typer.Exit(code=code)


@app.command()
def test(path: Annotated[str, typer.Argument(help="project root or fixtures root")] = ".") -> None:
    result = run_e2e_suite(path)
    _title("Test")
    typer.echo(f"{'result':<8} {'case':<28} detail")
    typer.echo("-" * 72)
    for case in result.results:
        prefix = "PASS" if case.passed else "FAIL"
        typer.echo(f"{prefix:<8} {case.name:<28} {case.message}")
    total = result.passed + result.failed
    typer.echo("-" * 72)
    typer.echo(f"summary: passed={result.passed} failed={result.failed} total={total}")
    if result.failed:
        raise typer.Exit(code=1)


@auth_app.command("login")
def auth_login(
    registry: Annotated[str, typer.Option(help="registry base URL")] = DEFAULT_REGISTRY_URL,
    token: Annotated[str | None, typer.Option(help="registry token")] = None,
) -> None:
    final_token = (token or "").strip()
    if not final_token:
        final_token = typer.prompt("Registry token", hide_input=True).strip()
    if not final_token:
        _fail(ManvError(diag("E8201", "token cannot be empty", "auth", 1, 1)))

    session = save_registry_auth(registry=registry, token=final_token)
    _title("Registry Auth")
    _kv("status", "logged_in")
    _kv("registry", session.registry)
    _kv("saved_at", session.saved_at)


@auth_app.command("status")
def auth_status() -> None:
    session = load_registry_auth()
    _title("Registry Auth")
    if session is None:
        _kv("status", "logged_out")
        return

    masked = "*" * max(0, len(session.token) - 4) + session.token[-4:]
    _kv("status", "logged_in")
    _kv("registry", session.registry)
    _kv("token", masked)
    _kv("saved_at", session.saved_at)


@auth_app.command("logout")
def auth_logout() -> None:
    removed = clear_registry_auth()
    _title("Registry Auth")
    _kv("status", "logged_out")
    _kv("cleared", removed)


@app.command()
def add(
    spec: Annotated[str, typer.Argument(help="name[@version] or git URL")],
    target: Annotated[str, typer.Argument(help="project directory")] = ".",
    git: Annotated[str | None, typer.Option(help="git repository URL override")] = None,
    version: Annotated[str | None, typer.Option(help="registry version override")] = None,
    registry: Annotated[str | None, typer.Option(help="registry base URL override")] = None,
    branch: Annotated[str | None, typer.Option(help="git branch")] = None,
    tag: Annotated[str | None, typer.Option(help="git tag")] = None,
    rev: Annotated[str | None, typer.Option(help="git commit/revision")] = None,
) -> None:
    try:
        ctx = discover_target(target)

        manifest_path = ctx.config_path or (ctx.root / "manv.toml")
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
    except (ManvError, ValueError) as err:
        if isinstance(err, ManvError):
            _fail(err)
        _fail(ManvError(diag("E8203", str(err), str(target), 1, 1)))


@app.command()
def dap(
    transport: Annotated[str, typer.Option(help="debug adapter transport: stdio|tcp")] = "stdio",
    host: Annotated[str, typer.Option(help="tcp bind host")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="tcp bind port")] = 4711,
) -> None:
    server = DAPServer()
    if transport == "stdio":
        server.start_stdio()
        return
    if transport == "tcp":
        server.start_tcp(host=host, port=port)
        return
    _fail(ManvError(diag("E8401", f"unsupported dap transport: {transport}", "dap", 1, 1)))


if __name__ == "__main__":
    app()



