"""Rewrite `let` declarations to the new declaration syntax.

    python tools/migrate_let.py [--check] PATH [PATH ...]

PATH may be a file or a directory (searched for *.mv and *.py; Python files are scanned for ManV
programs embedded in strings). Rules:

    let x: T = v      ->  T x = v
    let x = v         ->  x = v
    let mut x ... = v ->  as above (`mut` no longer exists)
    let x: T          ->  T x

A binding that was never reassigned can be made a constant by hand (`const x = v`); this tool does
not guess. Line endings of each file are preserved. With --check nothing is written and the exit
code is 1 if any file would change.

This is the model for future syntax migrations: each syntax change ships a rewrite rule like this.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# `let` must not be part of a longer word or a member access (`obj.let`).
_HEAD = r"(?<![\w.])let[ \t]+(?:mut[ \t]+)?"

# let NAME: TYPE = value   (TYPE has no '=', quotes, backslashes or line breaks)
_TYPED = re.compile(_HEAD + r"(\w+)[ \t]*:[ \t]*([^=\n\"'\\]+?)[ \t]*=[ \t]")
# let NAME = value
_INFERRED = re.compile(_HEAD + r"(\w+)[ \t]*=[ \t]")
# let NAME: TYPE   (declaration without a value, ending the line or the string)
_TYPED_NO_VALUE = re.compile(_HEAD + r"(\w+)[ \t]*:[ \t]*([\w\[\]]+)(?=[ \t]*(?:\r?\n|\\n|[\"']|$))", re.M)
# anything left that still looks like a declaration
_LEFTOVER = re.compile(r"(?<![\w.])let[ \t]+(?:mut[ \t]+)?\w+")


def migrate(text: str) -> tuple[str, int]:
    count = 0

    def typed(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{m.group(2)} {m.group(1)} = "

    def inferred(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{m.group(1)} = "

    def typed_no_value(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{m.group(2)} {m.group(1)}"

    text = _TYPED.sub(typed, text)
    text = _INFERRED.sub(inferred, text)
    text = _TYPED_NO_VALUE.sub(typed_no_value, text)
    return text, count


def iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            for pattern in ("*.mv", "*.py"):
                files.extend(p for p in sorted(path.rglob(pattern)) if "node_modules" not in p.parts and "__pycache__" not in p.parts)
        elif path.is_file():
            files.append(path)
        else:
            print(f"skipping {path}: not found", file=sys.stderr)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="report what would change, write nothing")
    args = parser.parse_args()

    changed = 0
    for path in iter_files(args.paths):
        raw = path.read_bytes().decode("utf-8")
        new, count = migrate(raw)
        leftovers = [m.group(0) for m in _LEFTOVER.finditer(new)]
        if count:
            changed += 1
            print(f"{path}: {count} rewritten")
            if not args.check:
                path.write_bytes(new.encode("utf-8"))
        for text in leftovers:
            print(f"{path}: needs a manual look: {text!r}", file=sys.stderr)

    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
