"""System-LLVM toolchain helpers for ManV host compilation.

Why this module exists:
- The host backend uses textual LLVM IR plus the system toolchain instead of
  bundling a Python LLVM binding.
- Keeping toolchain discovery and subprocess calls here prevents the compiler
  pipeline from being littered with platform-specific command assembly.

Important invariants:
- Output naming is deterministic.
- LLVM failure surfaces are translated into stable ManV diagnostics.
- The toolchain wrapper can temporarily fall back to the older assembly path
  when LLVM is unavailable, but only as an internal migration aid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Iterable

from .diagnostics import ManvError, diag
from .native_toolchain import build_native_artifacts
from .targets import TargetSpec


@dataclass(frozen=True)
class LlvmToolchain:
    clang: str
    version_text: str


def detect_llvm_toolchain() -> LlvmToolchain | None:
    clang = shutil.which("clang")
    if clang is None:
        return None
    proc = subprocess.run([clang, "--version"], capture_output=True, text=True, check=False)
    version_text = proc.stdout.splitlines()[0].strip() if proc.returncode == 0 and proc.stdout else "unknown"
    return LlvmToolchain(clang=clang, version_text=version_text)


def build_llvm_artifacts(
    *,
    llvm_ir: str,
    out_dir: Path,
    stem: str,
    target: TargetSpec,
    emit_ir: bool,
    emit_object: bool,
    emit_executable: bool,
    emit_asm: bool = False,
    link_libs: tuple[str, ...] = (),
    link_paths: tuple[str, ...] = (),
    link_args: tuple[str, ...] = (),
    allow_asm_fallback: bool = True,
    fallback_asm_text: str | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    ll_path = out_dir / f"{stem}.{target.name}.ll"
    ll_path.write_text(llvm_ir, encoding="utf-8")
    if emit_ir:
        paths["llvm_ir"] = ll_path

    toolchain = detect_llvm_toolchain()
    if toolchain is None:
        if allow_asm_fallback and fallback_asm_text is not None and (emit_object or emit_executable):
            fallback = build_native_artifacts(
                asm_text=fallback_asm_text,
                out_dir=out_dir,
                stem=stem,
                target=target,
                emit_object=emit_object,
                emit_executable=emit_executable,
            )
            if fallback.object_path is not None:
                paths["native_obj"] = fallback.object_path
            if fallback.executable_path is not None:
                paths["native_exe"] = fallback.executable_path
            return paths
        raise ManvError(diag("E5201", "no LLVM toolchain found (clang)", str(out_dir), 1, 1))

    obj_ext = ".obj" if platform.system().lower() == "windows" else ".o"
    obj_path = out_dir / f"{stem}.{target.name}{obj_ext}"
    asm_path = out_dir / f"{stem}.{target.name}.llvm.s"
    runtime_c_path = out_dir / f"{stem}.{target.name}.runtime.c"
    runtime_obj_path = out_dir / f"{stem}.{target.name}.runtime{obj_ext}"
    exe_suffix = ".exe" if platform.system().lower() == "windows" else ""
    exe_path = out_dir / f"{stem}.{target.name}{exe_suffix}"

    if emit_asm:
        _run(
            [toolchain.clang, "-S", "-x", "ir", str(ll_path), "-o", str(asm_path), "-target", _target_triple(target)],
            cwd=out_dir,
            err_code="E5202",
            err_prefix="llvm assembly emission failed",
        )
        paths["asm"] = asm_path

    if emit_object or emit_executable:
        _run(
            [toolchain.clang, "-c", "-x", "ir", str(ll_path), "-o", str(obj_path), "-target", _target_triple(target)],
            cwd=out_dir,
            err_code="E5203",
            err_prefix="llvm object emission failed",
        )
        paths["native_obj"] = obj_path

    if emit_executable:
        runtime_c_path.write_text(_runtime_support_c(), encoding="utf-8")
        _run(
            [toolchain.clang, "-c", str(runtime_c_path), "-o", str(runtime_obj_path), "-target", _target_triple(target)],
            cwd=out_dir,
            err_code="E5204",
            err_prefix="runtime support compilation failed",
        )
        command = [toolchain.clang, str(obj_path), str(runtime_obj_path), "-o", str(exe_path), "-target", _target_triple(target)]
        for path in link_paths:
            command.extend(["-L", path])
        for lib in link_libs:
            command.append(f"-l{lib}")
        command.extend(link_args)
        _run(command, cwd=out_dir, err_code="E5205", err_prefix="llvm link failed")
        paths["native_exe"] = exe_path

    return paths


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


def _runtime_support_c() -> str:
    return (
        "#include <stdint.h>\n"
        "#include <stdio.h>\n\n"
        "void manv_rt_print_i64(int64_t value) {\n"
        "    printf(\"%lld\\n\", (long long)value);\n"
        "}\n\n"
        "void manv_rt_print_f64(double value) {\n"
        "    printf(\"%.17g\\n\", value);\n"
        "}\n\n"
        "void manv_rt_print_bool(_Bool value) {\n"
        "    puts(value ? \"True\" : \"False\");\n"
        "}\n\n"
        "void manv_rt_print_cstr(const char* value) {\n"
        "    puts(value ? value : \"\");\n"
        "}\n"
    )


def _target_triple(target: TargetSpec) -> str:
    if target.name == "x86_64-sysv":
        return "x86_64-unknown-linux-gnu"
    if target.name == "x86_64-win64":
        return "x86_64-pc-windows-msvc"
    if target.name == "aarch64-aapcs64":
        return "aarch64-unknown-linux-gnu"
    return "unknown-unknown-unknown"
