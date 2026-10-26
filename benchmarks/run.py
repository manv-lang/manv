"""Benchmark runner: times each program in benchmarks/src on ManV and on CPython.

Every benchmark exists twice, ``name.mv`` and ``name.py``, with the same algorithm. Sizes are
written as ``@N@`` / ``@M@`` placeholders and filled in from SIZES below, so one program can run
at a size the current interpreter finishes in seconds (``small``) or at the size the fast engine
is meant for (``full``). Both versions must print the same output, which doubles as a
correctness check.

    python benchmarks/run.py                       # small sizes, 3 runs each
    python benchmarks/run.py --size full --runs 5
    python benchmarks/run.py --only fib loop
    python benchmarks/run.py --save benchmarks/baseline-small.json
    python benchmarks/run.py --compare benchmarks/baseline-small.json

Times are wall-clock for the whole process, startup included, because startup is part of what
users feel. The reported number is the median of --runs runs.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
REPO = HERE.parent  # `python -m manv.cli` resolves from here

# Placeholder values per benchmark and size. "small" suits the Python tree-walker; "full" is
# sized for a fast engine.
SIZES: dict[str, dict[str, dict[str, int]]] = {
    "startup": {"small": {}, "full": {}},
    "loop": {"small": {"N": 200_000}, "full": {"N": 10_000_000}},
    "fib": {"small": {"N": 24}, "full": {"N": 30}},
    "nested_loops": {"small": {"N": 400}, "full": {"N": 3_000}},
    "array_fill_sum": {"small": {"N": 100_000}, "full": {"N": 2_000_000}},
    "map_ops": {"small": {"N": 30_000}, "full": {"N": 500_000}},
    "string_build": {"small": {"N": 50_000}, "full": {"N": 200_000}},
    "method_calls": {"small": {"N": 100_000}, "full": {"N": 2_000_000}},
    "closure_calls": {"small": {"N": 100_000}, "full": {"N": 2_000_000}},
    "exceptions": {"small": {"N": 30_000}, "full": {"N": 500_000}},
    "mandelbrot": {"small": {"N": 40, "M": 100}, "full": {"N": 400, "M": 200}},
}

TIMEOUT_SECONDS = 600


def engines() -> dict[str, tuple[str, list[str]]]:
    """name -> (source suffix, command prefix). Add the C engine here when it exists."""
    return {
        "manv": (".mv", [sys.executable, "-m", "manv.cli", "run"]),
        "cpython": (".py", [sys.executable]),
    }


def render(name: str, suffix: str, size: str, out_dir: Path) -> Path:
    text = (SRC / f"{name}{suffix}").read_text(encoding="utf-8")
    for key, value in SIZES[name][size].items():
        text = text.replace(f"@{key}@", str(value))
    target = out_dir / f"{name}{suffix}"
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


def run_once(command: list[str], source: Path) -> tuple[float, str]:
    start = time.perf_counter()
    proc = subprocess.run(
        [*command, str(source)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=TIMEOUT_SECONDS,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{source.name} exited {proc.returncode}:\n{proc.stderr.strip()[:800]}")
    return elapsed, proc.stdout.replace("\r\n", "\n").strip()


def fmt(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=["small", "full"], default="small")
    parser.add_argument("--runs", type=int, default=3, help="runs per benchmark and engine (median is reported)")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="run only these benchmarks")
    parser.add_argument("--save", type=Path, help="write results to this JSON file")
    parser.add_argument("--compare", type=Path, help="show how each result changed against this JSON file")
    args = parser.parse_args()

    names = args.only or list(SIZES)
    unknown = [n for n in names if n not in SIZES]
    if unknown:
        parser.error(f"unknown benchmark(s): {', '.join(unknown)}; choose from {', '.join(SIZES)}")

    baseline = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None
    results: dict[str, dict[str, float]] = {}
    failures = 0

    with tempfile.TemporaryDirectory(prefix="manv-bench-") as tmp:
        work = Path(tmp)
        for name in names:
            results[name] = {}
            outputs: dict[str, str] = {}
            for engine, (suffix, command) in engines().items():
                source = render(name, suffix, args.size, work)
                times: list[float] = []
                try:
                    for _ in range(args.runs):
                        elapsed, output = run_once(command, source)
                        times.append(elapsed)
                        outputs[engine] = output
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    print(f"{name} [{engine}] FAILED: {exc}", file=sys.stderr)
                    failures += 1
                    continue
                results[name][engine] = statistics.median(times)
            if len(set(outputs.values())) > 1:
                print(f"{name}: engines printed different output: {outputs}", file=sys.stderr)
                failures += 1

    header = f"{'benchmark':<16}{'manv (s)':>10}{'cpython (s)':>13}{'manv/cpython':>14}"
    if baseline:
        header += f"{'manv vs baseline':>20}"
    print(f"\nsize={args.size}  runs={args.runs}  python={platform.python_version()}  {platform.system()} {platform.machine()}\n")
    print(header)
    print("-" * len(header))
    for name, row in results.items():
        manv, cpy = row.get("manv"), row.get("cpython")
        ratio = f"{manv / cpy:.1f}x" if manv and cpy else "-"
        line = f"{name:<16}{fmt(manv):>10}{fmt(cpy):>13}{ratio:>14}"
        if baseline:
            before = baseline.get("results", {}).get(name, {}).get("manv")
            line += f"{(f'{manv / before:.2f}x' if manv and before else '-'):>20}"
        print(line)

    if args.save:
        merged = results
        if args.only and args.save.exists():
            # A partial run updates the benchmarks it ran and keeps the rest of the file.
            merged = {**json.loads(args.save.read_text(encoding="utf-8")).get("results", {}), **results}
        payload = {
            "size": args.size,
            "runs": args.runs,
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "results": merged,
        }
        args.save.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nsaved {args.save}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
