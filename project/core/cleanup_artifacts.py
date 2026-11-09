"""Safely remove reproducible logs and media from a 6D-vision workspace.

The cleaner is deliberately conservative.  It protects reference datasets,
calibration inputs, CAD/model files, descriptor caches, and every RGB image
referenced by a saved accepted pose.  It only performs deletion with
``--apply``; the default command prints a dry-run report.

Examples
--------
    python -m core.cleanup_artifacts
    python -m core.cleanup_artifacts --apply --manifest cleanup_manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import (
    CALIBRATION_ROOT,
    CACHE_ROOT,
    DATA_ROOT,
    MODEL_ROOT,
    PROJECT_ROOT,
    RGBD_CAPTURE_ROOT,
    SAMPLE_ROOT,
    TRACKING_ROOT,
    TRAINING_ROOT,
)


MEDIA_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
LOG_SUFFIXES = {".log", ".jsonl"}
TRACKING_CSV_NAMES = {"frames.csv", "center_xyz.csv"}
# Regression fixtures are retained reference inputs.  Their depth archives and
# calibration snapshots were already protected by type, but these RGB pairs
# must also survive media cleanup because tests exercise the real RGB gates.
PROTECTED_TRACKING_RGB_RELATIVE_PATHS = frozenset({
    "live_depth_6dof/20261101_162802/attempt_003/rgb_before.png",
    "live_depth_6dof/20261101_162802/attempt_003/rgb_after.png",
    "live_depth_6dof/20261101_162307/attempt_015/rgb_before.png",
    "live_depth_6dof/20261101_162307/attempt_015/rgb_after.png",
    "live_depth_6dof/20261101_162307/attempt_024/rgb_before.png",
    "live_depth_6dof/20261101_162307/attempt_024/rgb_after.png",
})

# These are generated runtime capture roots outside project/data.  They do not
# contain reference inputs and are safe to empty as part of this maintenance.
EXTRA_CAPTURE_ROOTS = (
    PROJECT_ROOT.parent / "data" / "rgbd_capture",
    RGBD_CAPTURE_ROOT,
)
LOG_ROOTS = (
    PROJECT_ROOT.parent / "logs",
    PROJECT_ROOT.parent / "Log",
    PROJECT_ROOT / "Log",
)
BUILD_LOG_ROOTS = (
    PROJECT_ROOT.parent / "build",
    PROJECT_ROOT.parent / "OrbbecSDK_C_C++_v1.10.37_20260707_3f75820_linux_x64_release"
    / "OrbbecSDK_v1.10.37" / "Example" / "build",
)


@dataclass(frozen=True)
class Removal:
    path: Path
    reason: str
    size: int


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (path for path in root.rglob("*") if path.is_file())


def _size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else sum(
            child.stat().st_size for child in path.rglob("*") if child.is_file()
        )
    except OSError:
        return 0


def _add(items: dict[Path, Removal], path: Path, reason: str) -> None:
    path = path.resolve()
    if not path.exists():
        return
    current = items.get(path)
    if current is None:
        items[path] = Removal(path, reason, _size(path))


def _protected_tracking_images() -> set[Path]:
    """Return RGB inputs still needed by saved pose references.

    Reference JSON stores absolute paths because captures may be consumed from
    another working directory.  Resolve only existing paths inside the tracking
    root; an invalid historical path cannot accidentally protect an unrelated
    file.
    """
    protected: set[Path] = set()
    for pose_path in TRACKING_ROOT.rglob("rgb_initial_pose.json"):
        try:
            payload = json.loads(pose_path.read_text(encoding="utf-8"))
            image = Path(payload.get("rgb_image", "")).expanduser()
            if image.exists() and _within(image, TRACKING_ROOT):
                protected.add(image.resolve())
        except (OSError, ValueError, TypeError):
            continue

    # These are raw inputs for retained depth reference sessions rather than
    # diagnostic overlays.  Keep them when present even if no pose JSON points
    # back to them.
    for relative in (
        "depth_probe_adjusted_unmirrored/rgb.png",
        "global_pose_trials/trial_20261030_03/rgb.png",
        *PROTECTED_TRACKING_RGB_RELATIVE_PATHS,
    ):
        candidate = TRACKING_ROOT / relative
        if candidate.is_file():
            protected.add(candidate.resolve())
    return protected


def _schedule_directory_contents(items: dict[Path, Removal], root: Path, reason: str) -> None:
    for path in _files(root):
        _add(items, path, reason)


def discover_removals() -> list[Removal]:
    """Build a deterministic, conservative removal plan."""
    items: dict[Path, Removal] = {}

    # Standalone SDK/application logs.
    for root in LOG_ROOTS:
        _schedule_directory_contents(items, root, "standalone log directory")
    for root in BUILD_LOG_ROOTS:
        for path in _files(root):
            if path.suffix.lower() == ".log":
                _add(items, path, "generated build log")

    # Raw smoke captures are reproducible and outside the reference datasets.
    for root in EXTRA_CAPTURE_ROOTS:
        _schedule_directory_contents(items, root, "reproducible RGB-D capture output")

    # Training weights are runtime inputs; training plots and metadata are not.
    for root in (TRAINING_ROOT, PROJECT_ROOT.parent / "runs"):
        for path in _files(root):
            if path.suffix.lower() != ".pt":
                _add(items, path, "training visualization or run metadata")

    # Failed/rejected/incomplete sample attempts are never selected by the
    # normal reference-data workflows and are safe to remove as whole trees.
    for root in (SAMPLE_ROOT, CALIBRATION_ROOT / "rgb_ir_pairs"):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir() and path.name.lower().startswith(("failed_", "rejected_", "review_", "incomplete_")):
                _add(items, path, "failed or rejected sample")

    # Calibration previews can be regenerated from the retained raw pairs.
    for path in _files(CALIBRATION_ROOT):
        name = path.name.lower()
        if (
            path.suffix.lower() in IMAGE_SUFFIXES
            and (
                name.startswith("detections_")
                or name.startswith("validation_preview")
                or name in {"ir_checkerboard_preview.png", "ir_median_preview.png"}
            )
        ):
            _add(items, path, "reproducible calibration preview")

    protected_images = _protected_tracking_images()
    for path in _files(TRACKING_ROOT):
        resolved = path.resolve()
        suffix = path.suffix.lower()
        if resolved in protected_images:
            continue
        if suffix in MEDIA_SUFFIXES:
            _add(items, path, "tracking video")
        elif suffix in LOG_SUFFIXES:
            _add(items, path, "tracking log stream")
        elif suffix in IMAGE_SUFFIXES:
            _add(items, path, "tracking diagnostic image")
        elif path.name in TRACKING_CSV_NAMES or path.name.startswith("track_log_"):
            _add(items, path, "tracking diagnostic table")

    # A directory removal already includes its children; omit descendants to
    # keep the report readable and make application idempotent.
    directories = sorted((item.path for item in items.values() if item.path.is_dir()), key=lambda path: len(path.parts))
    filtered: list[Removal] = []
    for item in sorted(items.values(), key=lambda entry: str(entry.path)):
        if item.path.is_file() and any(_within(item.path, directory) for directory in directories):
            continue
        filtered.append(item)
    return filtered


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def write_manifest(path: Path, removals: list[Removal], *, applied: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "applied": applied,
        "project_root": str(PROJECT_ROOT),
        "removed_count": len(removals),
        "removed_bytes": sum(item.size for item in removals),
        "items": [
            {"path": str(item.path), "reason": item.reason, "size": item.size}
            for item in removals
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remove_empty_directories() -> int:
    count = 0
    roots = [*LOG_ROOTS, *EXTRA_CAPTURE_ROOTS, TRAINING_ROOT, PROJECT_ROOT.parent / "runs",
             SAMPLE_ROOT, CALIBRATION_ROOT / "rgb_ir_pairs", TRACKING_ROOT]
    for root in roots:
        if not root.exists():
            continue
        for directory in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                directory.rmdir()
                count += 1
            except OSError:
                pass
    return count


def apply_removals(removals: list[Removal]) -> None:
    for item in removals:
        if item.path.is_dir():
            shutil.rmtree(item.path)
        else:
            item.path.unlink()
    remove_empty_directories()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually delete the listed artifacts")
    parser.add_argument("--manifest", type=Path, help="write the plan as JSON")
    parser.add_argument("--list", action="store_true", help="print every planned path")
    args = parser.parse_args()

    removals = discover_removals()
    total = sum(item.size for item in removals)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode}: {len(removals)} items, {_format_size(total)}")
    for item in removals:
        if args.list:
            print(f"{_format_size(item.size):>10}  {item.reason:<36} {item.path}")
    if args.manifest:
        write_manifest(args.manifest, removals, applied=args.apply)
        print(f"Manifest: {args.manifest}")
    if args.apply:
        apply_removals(removals)
        print("Removed planned artifacts and empty directories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
