"""Launch and inspect the Linux Orbbec RGB-D capture backend."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .paths import PROJECT_ROOT


DEFAULT_BINARY = PROJECT_ROOT.parent / "build" / "orbbec_rgbd_capture"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "rgbd_capture" / "current"


def run_capture(binary: Path, output: Path, seconds: int, pairs: int) -> int:
    """Run the native backend and return its process exit code."""
    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        raise FileNotFoundError(
            f"Native capture binary is missing or not executable: {binary}. "
            "Build it with cmake -S . -B build && cmake --build build -j2."
        )
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "--output",
        str(output),
        "--seconds",
        str(seconds),
        "--pairs",
        str(pairs),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT.parent, check=False)
    return completed.returncode


def read_summary(output: Path) -> dict[str, str]:
    """Read the key/value summary emitted after a native capture."""
    summary = output / "summary.txt"
    if not summary.is_file():
        raise FileNotFoundError(f"Capture summary was not found: {summary}")
    values: dict[str, str] = {}
    for line in summary.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--summary", action="store_true",
                        help="Print summary.txt as JSON after capture")
    args = parser.parse_args()
    if args.seconds <= 0 or args.pairs <= 0:
        parser.error("--seconds and --pairs must be positive")
    return args


def main() -> int:
    args = parse_args()
    return_code = run_capture(args.binary.resolve(), args.output.resolve(),
                              args.seconds, args.pairs)
    if args.summary and (args.output / "summary.txt").is_file():
        print(json.dumps(read_summary(args.output.resolve()), indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

