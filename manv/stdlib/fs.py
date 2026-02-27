from __future__ import annotations

from pathlib import Path
import shutil


def exists(path: str) -> bool:
    return Path(path).exists()


def stat(path: str):
    return Path(path).stat()


def mkdir(path: str, *, parents: bool = True) -> None:
    Path(path).mkdir(parents=parents, exist_ok=True)


def remove(path: str) -> None:
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


def rename(src: str, dst: str) -> None:
    Path(src).rename(dst)


def walk(path: str) -> list[str]:
    root = Path(path)
    return [str(p) for p in root.rglob("*")]
