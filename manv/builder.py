from __future__ import annotations

import shutil
from pathlib import Path

from .compiler import compile_target
from .project import discover_target


def build_target(path: str | Path | None, out_dir: Path | None = None) -> Path:
    context = discover_target(path)
    compile_target(context.entry, context.target_dir, emit=["ast", "hir", "graph", "kernel"])

    dist_root = out_dir or context.dist_dir
    bundle_root = dist_root / context.name
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    (bundle_root / "src").mkdir(parents=True, exist_ok=True)
    shutil.copy2(context.entry, bundle_root / "src" / context.entry.name)
    if context.config_path and context.config_path.exists():
        shutil.copy2(context.config_path, bundle_root / "manv.toml")

    package_root = Path(__file__).resolve().parent
    shutil.copytree(package_root, bundle_root / "manv", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    launcher = (
        "from __future__ import annotations\n"
        "from pathlib import Path\n"
        "import sys\n\n"
        "ROOT = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "from manv.runner import run_file\n\n"
        "if __name__ == '__main__':\n"
        f"    raise SystemExit(run_file(ROOT / 'src' / '{context.entry.name}'))\n"
    )
    (bundle_root / "run.py").write_text(launcher, encoding="utf-8")
    (bundle_root / "__main__.py").write_text("from run import *\n", encoding="utf-8")
    return bundle_root
