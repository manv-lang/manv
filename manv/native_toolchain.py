from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Iterable

from .diagnostics import ManvError, diag
from .targets import TargetSpec


@dataclass(frozen=True)
class NativeToolchain:
    c_compiler: str


@dataclass(frozen=True)
class NativeArtifacts:
    object_path: Path | None
    executable_path: Path | None


def detect_toolchain() -> NativeToolchain | None:
    """Return the first available host C toolchain that can assemble/link."""
    for compiler in ("gcc", "clang", "cc"):
        path = shutil.which(compiler)
        if path:
            return NativeToolchain(c_compiler=path)
    return None


def host_default_target() -> str:
    """Map the host machine to the default native ABI target."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if machine in {"amd64", "x86_64", "x64"}:
        return "x86_64-win64" if system == "windows" else "x86_64-sysv"

    if machine in {"arm64", "aarch64"}:
        return "aarch64-aapcs64"

    # Conservative fallback used only when machine cannot be recognized.
    return "x86_64-sysv"


def ensure_host_target_supported(target: TargetSpec) -> None:
    expected = host_default_target()
    if target.name != expected:
        raise ManvError(
            diag(
                "E5101",
                f"native machine-code emission for target '{target.name}' requires cross-toolchain; host default is '{expected}'",
                target.name,
                1,
                1,
            )
        )


def build_native_artifacts(
    *,
    asm_text: str,
    out_dir: Path,
    stem: str,
    target: TargetSpec,
    emit_object: bool,
    emit_executable: bool,
) -> NativeArtifacts:
    """
    Assemble and optionally link a native binary from compiler-emitted assembly.

    This is intentionally host-toolchain driven for v0.1.0: no cross-linking is attempted.
    """
    if not emit_object and not emit_executable:
        return NativeArtifacts(object_path=None, executable_path=None)

    ensure_host_target_supported(target)

    toolchain = detect_toolchain()
    if toolchain is None:
        raise ManvError(diag("E5102", "no host C toolchain found (gcc/clang/cc)", str(out_dir), 1, 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    asm_path = out_dir / f"{stem}.{target.name}.native.s"
    obj_path = out_dir / f"{stem}.{target.name}.o"
    exe_suffix = ".exe" if platform.system().lower() == "windows" else ""
    exe_path = out_dir / f"{stem}.{target.name}{exe_suffix}"
    asm_path.write_text(asm_text, encoding="utf-8")

    _run(
        [toolchain.c_compiler, "-x", "assembler", "-c", str(asm_path), "-o", str(obj_path)],
        cwd=out_dir,
        err_code="E5103",
        err_prefix="assembly failed",
    )

    if emit_executable:
        _run(
            [toolchain.c_compiler, str(obj_path), "-o", str(exe_path)],
            cwd=out_dir,
            err_code="E5104",
            err_prefix="link failed",
        )

    return NativeArtifacts(
        object_path=obj_path if emit_object or emit_executable else None,
        executable_path=exe_path if emit_executable else None,
    )


def _run(command: Iterable[str], *, cwd: Path, err_code: str, err_prefix: str) -> None:
    proc = subprocess.run(list(command), cwd=str(cwd), capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        return

    message = err_prefix
    stderr = proc.stderr.strip()
    stdout = proc.stdout.strip()
    if stderr:
        message = f"{message}: {stderr}"
    elif stdout:
        message = f"{message}: {stdout}"

    raise ManvError(diag(err_code, message, str(cwd), 1, 1))
