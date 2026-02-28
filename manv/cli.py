from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from typing import Annotated

import typer

from . import __version__
from .builder import build_target, host_target_name
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


app = typer.Typer(help="ManV language toolchain")
auth_app = typer.Typer(help="Registry authentication")
app.add_typer(auth_app, name="auth")


def _fail(err: ManvError) -> None:
    typer.echo(err.render(), err=True)
    raise typer.Exit(code=1)


def _title(text: str) -> None:
    typer.echo(f"[{text}]")


def _kv(key: str, value: object) -> None:
    typer.echo(f"{key}: {value}")


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
    typer.echo("[Report:backend]")
    host_report = resolve_host_selection(
        HostSelectionRequest(
            requested_host_backend=host_backend,
            policy=policy,
        )
    )
    device_report = backend_selection_report(preferred_backend, device=device, policy=policy)
    typer.echo(render_joint_backend_report(host_report, device_report), nl=False)


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
    typer.echo("[Report:kernelize]")
    typer.echo(json.dumps(artifacts["gpu_report"], indent=2, sort_keys=True), nl=False)
    typer.echo()


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def init(
    path: Annotated[str, typer.Argument(help="project directory")] = ".",
    std: Annotated[bool, typer.Option("--std", help="initialize the standard-library project scaffold")] = False,
    name: Annotated[str | None, typer.Option("--name", help="project.name for project.toml")] = None,
    description: Annotated[str | None, typer.Option("--description", help="project description")] = None,
    author: Annotated[str | None, typer.Option("--author", help="author display name")]= None,
    requires_python: Annotated[str | None, typer.Option("--python", help="project.requires-python constraint")] = None,
    interactive: Annotated[bool, typer.Option("--interactive", help="prompt for project metadata (Poetry-style init flow)")] = False,
) -> None:
    if interactive:
        base_name = Path(path).resolve().name or "manv-project"
        if std:
            name = "std"
        else:
            name = typer.prompt("Project name", default=(name or base_name)).strip()
        description = typer.prompt(
            "Description",
            default=(description or ("ManV standard library" if std else "")),
        ).strip()
        author = typer.prompt("Author", default=(author or "ManV Developer")).strip()
        requires_python = typer.prompt("Requires Python", default=(requires_python or ">=3.12")).strip()

    if std:
        name = "std"

    try:
        ctx = init_project(
            path,
            std=std,
            name=name,
            description=description,
            author=author,
            requires_python=requires_python,
        )
    except ManvError as err:
        _fail(err)
    _title("Init")
    _kv("project", ctx.root)
    _kv("config", ctx.root / "project.toml")
    _kv("entry", ctx.entry)
    if std:
        _kv("mode", "std")
        _kv("tests", ctx.root / "tests" / "e2e" / "std_smoke" / "case.toml")
    else:
        _kv("mode", "app")
        _kv("tests", ctx.root / "tests" / "e2e" / "hello_world" / "case.toml")
    typer.echo("status: initialized")


@app.command()
def run(
    target: Annotated[str, typer.Argument(help="project directory or .mv file")] = ".",
    mode: Annotated[str, typer.Option(help="execution mode: interpreter|compiled")] = "interpreter",
    abi_target: Annotated[str, typer.Option("--target", help="cpu target ABI")] = "x86_64-sysv",
    backend: Annotated[str, typer.Option("--backend", help="runtime backend preference: auto,cuda,rocm,level0,vulkan-spv,directx,webgpu,cpu")] = "auto",
    device: Annotated[str | None, typer.Option("--device", help="preferred runtime device id")] = None,
    report: Annotated[str | None, typer.Option("--report", help="comma separated report kinds: backend,kernelize")] = None,
    cuda: Annotated[bool, typer.Option("--cuda", help="enable CUDA-aware execution paths when available")] = False,
    cuda_arch: Annotated[str, typer.Option("--cuda-arch", help="CUDA architecture for JIT compilation")] = "sm_80",
    cuda_dump_kernels: Annotated[bool, typer.Option("--cuda-dump-kernels", help="dump generated CUDA source alongside PTX")] = False,
    cuda_jit: Annotated[bool, typer.Option("--cuda-jit", help="prefer JIT compilation for CUDA kernels")] = True,
    cuda_aot: Annotated[bool, typer.Option("--cuda-aot", help="request AOT-style artifact emission where supported")] = False,
    deterministic_gc: Annotated[bool, typer.Option(help="enable deterministic GC checkpoints")] = False,
    gc_stress: Annotated[bool, typer.Option(help="force GC at every safe point")] = False,
    stable_debug_format: Annotated[bool, typer.Option(help="stable map/object formatting for debugging")] = False,
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
        typer.echo(text, nl=False)
    raise typer.Exit(code=code)


@app.command("compile")
def compile_cmd(
    target: Annotated[str, typer.Argument(help="single .mv file or directory containing main.mv")] = ".",
    emit: Annotated[
        str,
        typer.Option(
            help=(
                "comma separated list: ast,hir,hlir,graph,capture,kernel,kernel_exec,"
                "abi,host_stub_abi,asm,host_stub,source_map,gpu_report,backend_bundle,ptx,cuda_cpp,hip,spirv,wgsl,hlsl,llvm_ir,native_obj,native_exe"
            )
        ),
    ] = "ast,hir,hlir,graph,kernel,abi,llvm_ir,native_exe",
    out: Annotated[str | None, typer.Option(help="output directory override")] = None,
    host: Annotated[str, typer.Option("--host", help="host backend: auto,llvm,interp")] = "auto",
    device_backend: Annotated[
        str,
        typer.Option("--device-backend", "--backend", help="device backend target: auto,none,cuda,rocm,level0,vulkan-spv,webgpu,directx,cpu"),
    ] = "none",
    optimize: Annotated[bool, typer.Option(help="enable Graph IR optimization")] = True,
    abi_target: Annotated[str, typer.Option("--target", help="cpu target ABI")] = host_target_name(),
    capture: Annotated[bool, typer.Option(help="capture Graph IR by tracing HLIR execution")] = False,
    device: Annotated[str | None, typer.Option("--device", help="preferred runtime device id for reports")] = None,
    link_lib: Annotated[list[str] | None, typer.Option("--link-lib", help="native linker library name")] = None,
    link_path: Annotated[list[str] | None, typer.Option("--link-path", help="native linker search path")] = None,
    link_arg: Annotated[list[str] | None, typer.Option("--link-arg", help="raw native linker argument")] = None,
    report: Annotated[str | None, typer.Option("--report", help="comma separated report kinds: backend,kernelize")] = None,
    cuda_arch: Annotated[str, typer.Option("--cuda-arch", help="CUDA architecture for JIT compilation")] = "sm_80",
    cuda_dump_kernels: Annotated[bool, typer.Option("--cuda-dump-kernels", help="dump generated CUDA source")] = False,
    cuda_jit: Annotated[bool, typer.Option("--cuda-jit", help="prefer JIT compilation for CUDA kernels")] = True,
    cuda_aot: Annotated[bool, typer.Option("--cuda-aot", help="request AOT-style artifact emission where supported")] = False,
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
    _kv("device_backend", device_backend)
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
    host: Annotated[str, typer.Option("--host", help="host backend: auto,llvm,interp")] = "auto",
    device_backend: Annotated[
        str,
        typer.Option("--device-backend", "--backend", help="device backend preference: auto,none,cuda,rocm,level0,vulkan-spv,directx,webgpu,cpu"),
    ] = "auto",
    device: Annotated[str | None, typer.Option("--device", help="preferred runtime device id for reports")] = None,
    report: Annotated[str | None, typer.Option("--report", help="comma separated report kinds: backend,kernelize")] = None,
    portable_cache: Annotated[bool, typer.Option("--portable-cache", help="extract bundled program payload next to the executable")] = False,
    cuda_arch: Annotated[str, typer.Option("--cuda-arch", help="CUDA architecture for emitted artifacts")] = "sm_80",
    cuda_dump_kernels: Annotated[bool, typer.Option("--cuda-dump-kernels", help="dump generated CUDA source")] = False,
    cuda_jit: Annotated[bool, typer.Option("--cuda-jit", help="prefer JIT compilation for CUDA kernels")] = True,
    cuda_aot: Annotated[bool, typer.Option("--cuda-aot", help="request AOT-style artifact emission where supported")] = False,
) -> None:
    del cuda_arch, cuda_dump_kernels, cuda_jit, cuda_aot
    reports = _requested_reports(report)
    if "backend" in reports:
        _emit_backend_report(device_backend, host_backend=host, device=device, policy="build")
    if "kernelize" in reports:
        _emit_kernelize_report(target)
    try:
        bundle = build_target(
            target,
            Path(out).resolve() if out else None,
            portable_cache=portable_cache,
            host_backend=host,
            device_backend=device_backend,
        )
    except ManvError as err:
        _fail(err)
    _title("Build")
    _kv("artifact", bundle)
    _kv("host_backend", host)
    _kv("device_backend", device_backend)
    _kv("target", host_target_name())
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
def test(
    path: Annotated[str, typer.Argument(help="project root or fixtures root")] = ".",
    cuda: Annotated[bool, typer.Option("--cuda", help="run CUDA-aware tests when the environment supports them")] = False,
) -> None:
    del cuda
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



@app.command()
def lsp(
    transport: Annotated[str, typer.Option(help="language server transport: stdio|tcp")] = "stdio",
    host: Annotated[str, typer.Option(help="tcp bind host")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="tcp bind port")] = 2087,
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

if __name__ == "__main__":
    app()
