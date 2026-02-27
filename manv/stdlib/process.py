from __future__ import annotations

import subprocess


def run(argv: list[str], *, check: bool = False, capture_output: bool = True, text: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=check, capture_output=capture_output, text=text)


def spawn(argv: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
