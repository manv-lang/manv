from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .diagnostics import ManvError, diag


DEFAULT_ENTRY = "src/main.mv"
DEFAULT_TARGET_DIR = ".manv/target"
DEFAULT_DIST_DIR = "dist"


@dataclass
class ProjectContext:
    root: Path
    name: str
    entry: Path
    target_dir: Path
    dist_dir: Path
    config_path: Path | None


def discover_target(path: str | Path | None) -> ProjectContext:
    candidate = Path(path or ".").resolve()
    if candidate.is_file():
        if candidate.suffix != ".mv":
            raise ManvError(diag("E4001", "source file must use .mv extension", str(candidate), 1, 1))
        root = candidate.parent
        return ProjectContext(
            root=root,
            name=candidate.stem,
            entry=candidate,
            target_dir=root / ".manv" / "target",
            dist_dir=root / DEFAULT_DIST_DIR,
            config_path=None,
        )

    if not candidate.exists():
        raise ManvError(diag("E4002", f"path does not exist: {candidate}", str(candidate), 1, 1))

    config_path = candidate / "manv.toml"
    if config_path.exists():
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        package = data.get("package", {})
        build = data.get("build", {})
        entry_rel = package.get("entry", DEFAULT_ENTRY)
        entry = candidate / entry_rel
        if not entry.exists():
            raise ManvError(diag("E4003", f"entry file not found: {entry_rel}", str(config_path), 1, 1))
        return ProjectContext(
            root=candidate,
            name=str(package.get("name", candidate.name)),
            entry=entry,
            target_dir=candidate / str(build.get("target_dir", DEFAULT_TARGET_DIR)),
            dist_dir=candidate / str(build.get("dist_dir", DEFAULT_DIST_DIR)),
            config_path=config_path,
        )

    entry = candidate / DEFAULT_ENTRY
    if not entry.exists():
        raise ManvError(diag("E4004", "missing manv.toml and src/main.mv", str(candidate), 1, 1))
    return ProjectContext(
        root=candidate,
        name=candidate.name,
        entry=entry,
        target_dir=candidate / ".manv" / "target",
        dist_dir=candidate / DEFAULT_DIST_DIR,
        config_path=None,
    )


def init_project(path: str | Path) -> ProjectContext:
    root = Path(path).resolve()
    root.mkdir(parents=True, exist_ok=True)

    src_dir = root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = root / "tests" / "e2e" / "hello_world"
    tests_dir.mkdir(parents=True, exist_ok=True)

    name = root.name
    config_path = root / "manv.toml"
    if not config_path.exists():
        config_path.write_text(
            (
                "[package]\n"
                f"name = \"{name}\"\n"
                "version = \"0.1.0\"\n"
                f"entry = \"{DEFAULT_ENTRY}\"\n\n"
                "[build]\n"
                f"target_dir = \"{DEFAULT_TARGET_DIR}\"\n"
                f"dist_dir = \"{DEFAULT_DIST_DIR}\"\n"
            ),
            encoding="utf-8",
        )

    main_path = src_dir / "main.mv"
    if not main_path.exists():
        main_path.write_text(
            (
                "fn main() -> int:\n"
                "    print(\"Hello, World!\")\n"
                "    return 0\n"
            ),
            encoding="utf-8",
        )

    case_path = tests_dir / "case.toml"
    if not case_path.exists():
        case_path.write_text(
            (
                "name = \"hello_world\"\n"
                "command = \"run\"\n"
                "target = \"../../../\"\n"
                "expect_exit = 0\n"
                "expect_stdout_contains = [\"Hello, World!\"]\n"
            ),
            encoding="utf-8",
        )

    return discover_target(root)
