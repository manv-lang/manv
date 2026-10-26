from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import tomllib

from .diagnostics import ManvError, diag
from .tokens import KEYWORDS


DEFAULT_ENTRY = "src/main.mv"
DEFAULT_TARGET_DIR = ".manv/target"
DEFAULT_DIST_DIR = "dist"
DEFAULT_BUILD_DIR = "build"
DEFAULT_CONFIG_FILE = "project.toml"
LEGACY_CONFIG_FILE = "manv.toml"


@dataclass
class ProjectContext:
    root: Path
    name: str
    entry: Path
    target_dir: Path
    dist_dir: Path
    build_dir: Path
    config_path: Path | None
    build_emit: list[str] = field(default_factory=list)


def discover_compile_target(path: str | Path | None) -> ProjectContext:
    """Resolve the input for `manv compile`.

    Why this exists:
    - `compile` is the single-file developer-facing command, while `build`
      owns project resolution and packaging.
    - Users still reasonably expect `manv compile` inside a directory that
      contains `main.mv` to compile that file without needing to spell it out.
    - When the caller points at a project root, the diagnostic should explain
      the intended split rather than surfacing the generic project loader error.
    """

    candidate = Path(path or ".").resolve()
    if candidate.is_file():
        if candidate.suffix != ".mv":
            raise ManvError(diag("E4001", "source file must use .mv extension", str(candidate), 1, 1))
        root = candidate.parent
        return ProjectContext(
            root=root,
            name=candidate.stem,
            entry=candidate,
            target_dir=root / DEFAULT_TARGET_DIR,
            dist_dir=root / DEFAULT_DIST_DIR,
            build_dir=root / DEFAULT_BUILD_DIR,
            config_path=None,
        )

    if not candidate.exists():
        raise ManvError(diag("E4002", f"path does not exist: {candidate}", str(candidate), 1, 1))

    direct_entry = candidate / "main.mv"
    if direct_entry.exists():
        return ProjectContext(
            root=candidate,
            name=direct_entry.stem,
            entry=direct_entry,
            target_dir=candidate / DEFAULT_TARGET_DIR,
            dist_dir=candidate / DEFAULT_DIST_DIR,
            build_dir=candidate / DEFAULT_BUILD_DIR,
            config_path=None,
        )

    project_entry = candidate / DEFAULT_ENTRY
    if _discover_config(candidate) is not None or project_entry.exists():
        raise ManvError(
            diag(
                "E4005",
                "compile expects a single .mv file or a directory containing main.mv; use 'manv build' for projects or pass src/main.mv explicitly",
                str(candidate),
                1,
                1,
            )
        )

    raise ManvError(
        diag(
            "E4005",
            "compile expects a single .mv file or a directory containing main.mv",
            str(candidate),
            1,
            1,
        )
    )


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
            target_dir=root / DEFAULT_TARGET_DIR,
            dist_dir=root / DEFAULT_DIST_DIR,
            build_dir=root / DEFAULT_BUILD_DIR,
            config_path=None,
        )

    if not candidate.exists():
        raise ManvError(diag("E4002", f"path does not exist: {candidate}", str(candidate), 1, 1))

    config_path = _discover_config(candidate)
    if config_path is not None:
        cfg = _read_project_config(config_path)
        entry_rel = str(cfg.get("entry", DEFAULT_ENTRY))
        entry = candidate / entry_rel
        if not entry.exists():
            raise ManvError(diag("E4003", f"entry file not found: {entry_rel}", str(config_path), 1, 1))

        return ProjectContext(
            root=candidate,
            name=str(cfg.get("name", candidate.name)),
            entry=entry,
            target_dir=candidate / str(cfg.get("target_dir", DEFAULT_TARGET_DIR)),
            dist_dir=candidate / str(cfg.get("dist_dir", DEFAULT_DIST_DIR)),
            build_dir=candidate / str(cfg.get("build_dir", DEFAULT_BUILD_DIR)),
            config_path=config_path,
            build_emit=cfg.get("build_emit", []),  # type: ignore[arg-type]
        )

    entry = candidate / DEFAULT_ENTRY
    if not entry.exists():
        raise ManvError(diag("E4004", "missing project.toml and src/main.mv", str(candidate), 1, 1))
    return ProjectContext(
        root=candidate,
        name=candidate.name,
        entry=entry,
        target_dir=candidate / DEFAULT_TARGET_DIR,
        dist_dir=candidate / DEFAULT_DIST_DIR,
        build_dir=candidate / DEFAULT_BUILD_DIR,
        config_path=None,
    )


def _discover_config(root: Path) -> Path | None:
    project_toml = root / DEFAULT_CONFIG_FILE
    if project_toml.exists():
        return project_toml
    legacy = root / LEGACY_CONFIG_FILE
    if legacy.exists():
        return legacy
    return None


def _read_project_config(path: Path) -> dict[str, object]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    project = data.get("project", {})
    package = data.get("package", {})
    build = data.get("build", {})

    tool = data.get("tool", {})
    tool_manv = tool.get("manv", {}) if isinstance(tool, dict) else {}

    name = _first_non_empty(
        project.get("name") if isinstance(project, dict) else None,
        package.get("name") if isinstance(package, dict) else None,
    )
    entry = _first_non_empty(
        tool_manv.get("entry") if isinstance(tool_manv, dict) else None,
        project.get("entry") if isinstance(project, dict) else None,
        package.get("entry") if isinstance(package, dict) else None,
        DEFAULT_ENTRY,
    )
    target_dir = _first_non_empty(
        tool_manv.get("target-dir") if isinstance(tool_manv, dict) else None,
        tool_manv.get("target_dir") if isinstance(tool_manv, dict) else None,
        build.get("target_dir") if isinstance(build, dict) else None,
        DEFAULT_TARGET_DIR,
    )
    dist_dir = _first_non_empty(
        tool_manv.get("dist-dir") if isinstance(tool_manv, dict) else None,
        tool_manv.get("dist_dir") if isinstance(tool_manv, dict) else None,
        build.get("dist_dir") if isinstance(build, dict) else None,
        DEFAULT_DIST_DIR,
    )
    build_dir = _first_non_empty(
        tool_manv.get("build-dir") if isinstance(tool_manv, dict) else None,
        tool_manv.get("build_dir") if isinstance(tool_manv, dict) else None,
        DEFAULT_BUILD_DIR,
    )

    build_emit: list[str] = []
    if isinstance(tool_manv, dict) and "build-emit" in tool_manv:
        raw = tool_manv["build-emit"]
        if isinstance(raw, list):
            build_emit = [str(x).strip() for x in raw if str(x).strip()]
        elif isinstance(raw, str):
            build_emit = [x.strip() for x in raw.split(",") if x.strip()]

    return {
        "name": str(name or ""),
        "entry": str(entry),
        "target_dir": str(target_dir),
        "dist_dir": str(dist_dir),
        "build_dir": str(build_dir),
        "build_emit": build_emit,
    }


def _first_non_empty(*values: object | None) -> object | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def init_project(
    path: str | Path,
    *,
    name: str | None = None,
    description: str | None = None,
    author: str | None = None,
) -> ProjectContext:
    root = Path(path).resolve()
    root.mkdir(parents=True, exist_ok=True)

    _init_default_project(
        root,
        name=name,
        description=description,
        author=author,
    )

    return discover_target(root)


def _module_identifier(name: str) -> str:
    """Turn a project name into a valid ManV module identifier.

    Any run of characters that is not an ASCII letter, digit or underscore becomes
    a single `_`. A leading digit gets a `_` prefix, and a reserved word (`type`,
    `int`, ...) gets a `_` suffix, so the generated `module <name>` line always parses.
    """
    ident = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip()
    if not ident.strip("_"):
        return "app"
    if ident[0].isdigit():
        ident = "_" + ident
    if ident in KEYWORDS:
        ident += "_"
    return ident


def _write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _default_gitignore_template() -> str:
    return (
        "# ManV build artifacts\n"
        ".manv/\n"
        "build/\n"
        "dist/\n"
        "\n"
        "# Python cache / local env\n"
        "__pycache__/\n"
        "*.py[cod]\n"
        ".pytest_cache/\n"
        ".venv/\n"
    )


def _toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\"", "\\\"")


def _project_toml_template(
    name: str,
    *,
    description: str = "",
    author: str = "ManV Developer",
) -> str:
    return (
        "[project]\n"
        f"name = \"{_toml_string(name)}\"\n"
        "version = \"0.1.0\"\n"
        f"description = \"{_toml_string(description)}\"\n"
        "readme = \"README.md\"\n"
        f"authors = [{{ name = \"{_toml_string(author)}\" }}]\n\n"
        "[tool.manv]\n"
        f"entry = \"{DEFAULT_ENTRY}\"\n"
        f"build-dir = \"{DEFAULT_BUILD_DIR}\"\n"
        f"target-dir = \"{DEFAULT_TARGET_DIR}\"\n"
        f"dist-dir = \"{DEFAULT_DIST_DIR}\"\n"
        "# build-emit = [\"hlir\", \"graph\"]  # uncomment to emit extra build artifacts\n"
    )


def _init_default_project(
    root: Path,
    *,
    name: str | None = None,
    description: str | None = None,
    author: str | None = None,
) -> None:
    src_dir = root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = root / "tests" / "e2e" / "hello_world"
    tests_dir.mkdir(parents=True, exist_ok=True)

    project_name = (name or root.name).strip() or root.name
    project_description = (description or "").strip()
    project_author = (author or "ManV Developer").strip() or "ManV Developer"

    _write_if_missing(root / ".gitignore", _default_gitignore_template())
    _write_if_missing(
        root / DEFAULT_CONFIG_FILE,
        _project_toml_template(
            project_name,
            description=project_description,
            author=project_author,
        ),
    )

    module_ident = _module_identifier(project_name)

    _write_if_missing(
        src_dir / "main.mv",
        (
            f"module {module_ident}\n"
            "\n"
            "fn main() -> int:\n"
            "    print(\"Hello, World!\")\n"
            "    return 0\n"
        ),
    )

    _write_if_missing(
        tests_dir / "case.toml",
        (
            "name = \"hello_world\"\n"
            "command = \"run\"\n"
            "target = \"../../../\"\n"
            "expect_exit = 0\n"
            "expect_stdout_contains = [\"Hello, World!\"]\n"
        ),
    )


