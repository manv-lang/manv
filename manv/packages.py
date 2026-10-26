"""Global package environment for ManV.

Why this file exists:
- Provides a platform-aware package cache (like pip's site-packages).
- `get_active_env()` returns the active PackageEnv (global only for now).
- Foundation for future venv support: callers use PackageEnv, not raw paths.
- Download/install functions for registry and git/GitHub packages.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


STDLIB_GIT_URL = "https://github.com/manv-lang/stdlib"
STDLIB_PACKAGE_NAME = "std"


@dataclass
class PackageEnv:
    """An environment that holds installed ManV packages.

    kind: "global" | "venv"  (only "global" is implemented now)
    root: base directory of the environment
    packages_dir: root / "packages", one subdir per package name/version
    """

    kind: str
    root: Path
    packages_dir: Path


def get_global_cache_dir() -> Path:
    """Returns the platform-appropriate global ManV cache directory.

    MANV_HOME env var overrides the entire base (like CARGO_HOME).
    Layout:
      Linux   — $XDG_DATA_HOME/manv  or  ~/.local/share/manv
      macOS   — ~/Library/ManV
      Windows — %LOCALAPPDATA%\\ManV
    """
    if manv_home := os.getenv("MANV_HOME", "").strip():
        return Path(manv_home)
    system = platform.system()
    if system == "Windows":
        local_app = os.getenv("LOCALAPPDATA", "").strip()
        base = Path(local_app) if local_app else Path.home() / "AppData" / "Local"
        return base / "ManV"
    if system == "Darwin":
        return Path.home() / "Library" / "ManV"
    # Linux / other POSIX
    xdg = os.getenv("XDG_DATA_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "manv"


def get_active_env() -> PackageEnv:
    """Returns the currently active package environment.

    Future: detect and return a venv-local env when inside a ManV venv.
    For now always returns the global env.
    """
    root = get_global_cache_dir()
    return PackageEnv(kind="global", root=root, packages_dir=root / "packages")


def installed_package_roots(exclude: frozenset[str] = frozenset()) -> list[Path]:
    """Returns src/ directories for every installed package version.

    Expected layout:  <packages_dir>/<name>/<version>/src/
    Returns only directories that actually exist, sorted by name then version.
    Packages named in `exclude` are skipped.
    """
    env = get_active_env()
    if not env.packages_dir.exists():
        return []
    roots: list[Path] = []
    for pkg_dir in sorted(env.packages_dir.iterdir()):
        if not pkg_dir.is_dir() or pkg_dir.name in exclude:
            continue
        for ver_dir in sorted(pkg_dir.iterdir()):
            src = ver_dir / "src"
            if src.is_dir():
                roots.append(src)
    return roots


def is_installed(name: str, version: str) -> bool:
    """Returns True if the given package/version already has a src/ dir in the cache."""
    env = get_active_env()
    return (env.packages_dir / name / version / "src").is_dir()


def is_stdlib_installed() -> bool:
    """Returns True if any version of the ManV stdlib is present in the cache."""
    env = get_active_env()
    pkg_dir = env.packages_dir / STDLIB_PACKAGE_NAME
    if not pkg_dir.exists():
        return False
    return any(
        (ver / "src").is_dir()
        for ver in pkg_dir.iterdir()
        if ver.is_dir()
    )


def package_meta_path(name: str, version: str) -> Path:
    env = get_active_env()
    return env.packages_dir / name / version / "manv-pkg.json"


def _extract_src_from_archive(archive_bytes: bytes, dest_src_dir: Path) -> None:
    """Extracts the src/ subtree from a .tar.gz archive into dest_src_dir.

    Handles two common layouts:
      - src/foo.mv          (src/ at archive root)
      - pkg-1.0.0/src/foo.mv  (one nesting level)
    """
    dest_src_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
        names = tf.getnames()

        # Determine the src/ prefix in the archive
        src_prefix: str | None = None
        for n in names:
            segs = n.split("/")
            if segs[0] == "src":
                src_prefix = "src/"
                break
            if len(segs) > 1 and segs[1] == "src":
                src_prefix = segs[0] + "/src/"
                break

        if src_prefix is None:
            raise ValueError(
                f"archive does not contain a 'src/' directory "
                f"(found top-level entries: {sorted({n.split('/')[0] for n in names})[:5]})"
            )

        for member in tf.getmembers():
            if not member.name.startswith(src_prefix):
                continue
            rel = member.name[len(src_prefix):]
            if not rel:
                continue
            target = dest_src_dir / rel
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                f = tf.extractfile(member)
                if f is not None:
                    target.write_bytes(f.read())


def install_from_registry(
    name: str,
    version: str,
    *,
    registry_url: str,
    token: str | None,
) -> Path:
    """Downloads and installs a registry package. Returns the installed src/ dir.

    Idempotent: skips the download if already installed.
    """
    if is_installed(name, version):
        return get_active_env().packages_dir / name / version / "src"

    from .registry import download_package_archive  # avoid circular at module level

    archive = download_package_archive(name, version, registry_url=registry_url, token=token)

    env = get_active_env()
    src_dir = env.packages_dir / name / version / "src"
    _extract_src_from_archive(archive, src_dir)

    package_meta_path(name, version).write_text(
        json.dumps(
            {"name": name, "version": version, "source": "registry", "registry": registry_url},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return src_dir


def install_from_github(
    name: str,
    git_url: str,
    ref: str,
    *,
    ref_kind: str = "branch",
) -> str:
    """Downloads a GitHub archive tarball and installs it. Returns the version key.

    No git binary required — uses GitHub's archive download endpoint.
    """
    base = git_url.rstrip("/")
    if base.endswith(".git"):
        base = base[:-4]

    if ref_kind in ("branch", "tag"):
        sub = "heads" if ref_kind == "branch" else "tags"
        archive_url = f"{base}/archive/refs/{sub}/{ref}.tar.gz"
    else:
        archive_url = f"{base}/archive/{ref}.tar.gz"

    req = urllib.request.Request(archive_url, headers={"Accept": "application/octet-stream"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            archive_bytes = resp.read()
    except Exception as exc:
        raise RuntimeError(f"could not download GitHub archive from {archive_url}: {exc}") from exc

    short_hash = hashlib.sha1(archive_bytes).hexdigest()[:8]
    version_key = f"git-{short_hash}"

    if is_installed(name, version_key):
        return version_key

    env = get_active_env()
    src_dir = env.packages_dir / name / version_key / "src"
    _extract_src_from_archive(archive_bytes, src_dir)

    package_meta_path(name, version_key).write_text(
        json.dumps(
            {"name": name, "version": version_key, "source": "git",
             "git": git_url, "ref": ref, "ref_kind": ref_kind},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return version_key


def install_from_git(
    name: str,
    git_url: str,
    ref: str | None,
    *,
    ref_kind: str = "branch",
) -> str:
    """Clones a git repository and installs its src/ tree. Returns the version key.

    Requires the `git` binary on PATH.
    """
    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "repo"
        cmd = ["git", "clone", "--depth", "1"]
        if ref and ref_kind in ("branch", "tag"):
            cmd += ["--branch", ref]
        cmd += [git_url, str(clone_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.strip()}")

        hash_result = subprocess.run(
            ["git", "-C", str(clone_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        short_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else "unknown"
        version_key = f"git-{short_hash}"

        if is_installed(name, version_key):
            return version_key

        src_src = clone_dir / "src"
        if not src_src.is_dir():
            raise RuntimeError(f"git repo '{git_url}' has no 'src/' directory")

        env = get_active_env()
        src_dir = env.packages_dir / name / version_key / "src"
        shutil.copytree(src_src, src_dir)

        package_meta_path(name, version_key).write_text(
            json.dumps(
                {"name": name, "version": version_key, "source": "git",
                 "git": git_url, "ref": ref, "ref_kind": ref_kind},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return version_key


def install_dependency(
    name: str,
    dep_entry: dict | str,
    *,
    registry_url: str,
    token: str | None,
) -> str:
    """Installs a single dependency from a project.toml entry. Returns the version key.

    Dispatches to the right install function based on the entry shape:
      - str          → registry, use as version
      - {"git": ...} → GitHub (if github.com) or generic git clone
      - {"version": ...} → registry
    """
    if isinstance(dep_entry, str):
        install_from_registry(name, dep_entry, registry_url=registry_url, token=token)
        return dep_entry

    git_url = dep_entry.get("git")
    if git_url:
        ref = (
            dep_entry.get("rev")
            or dep_entry.get("tag")
            or dep_entry.get("branch")
            or "main"
        )
        ref_kind = (
            "rev" if "rev" in dep_entry
            else ("tag" if "tag" in dep_entry else "branch")
        )
        if "github.com" in git_url:
            return install_from_github(name, git_url, ref, ref_kind=ref_kind)
        else:
            return install_from_git(name, git_url, ref, ref_kind=ref_kind)

    version = str(dep_entry.get("version", "latest"))
    reg = str(dep_entry.get("registry", registry_url))
    install_from_registry(name, version, registry_url=reg, token=token)
    return version
