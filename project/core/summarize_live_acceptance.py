"""Summarize one live 6DoF acceptance run without opening the camera."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from .paths import DATA_ROOT


DEFAULT_ROOT = DATA_ROOT / "tracking" / "live_depth_6dof"
HARD_LOSS_STATES = {
    "TARGET LOST",
    "FLOW LOST",
    "PNP FAILED",
    "SEED FAILED",
    "REJECTED QUALITY",
    "REJECTED JUMP",
}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def latest_run(root: Path = DEFAULT_ROOT) -> Path:
    runs = [path for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
    if not runs:
        raise RuntimeError(f"No live runs found under {root}")
    return max(runs, key=lambda path: path.name)


def read_elapsed_values(path: Path) -> list[float]:
    if not path.is_file():
        return []
    values: list[float] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    value = float(row["elapsed_s"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not values or value >= values[-1]:
                    values.append(value)
    except OSError as exc:
        raise RuntimeError(f"Cannot read CSV: {path}: {exc}") from exc
    return values


def summarize_attempt(attempt_dir: Path) -> dict:
    timing_path = attempt_dir / "initialization_timing.json"
    tracking_summary_path = attempt_dir / "tracking" / "summary.json"
    timing = read_json(timing_path) if timing_path.is_file() else {}
    tracking = read_json(tracking_summary_path) if tracking_summary_path.is_file() else None
    elapsed = read_elapsed_values(attempt_dir / "tracking" / "frames.csv")
    gaps = [right - left for left, right in zip(elapsed, elapsed[1:])]

    result = {
        "attempt": attempt_dir.name,
        "initialization_status": timing.get("status"),
        "initialization_source": timing.get("source"),
        "initialization_error": timing.get("error"),
        "tracking_summary": str(tracking_summary_path.resolve()) if tracking is not None else None,
        "tracking": None,
    }
    if tracking is None:
        return result

    status_counts = tracking.get("status_counts") or {}
    display = tracking.get("display") or {}
    display_counts = display.get("status_counts") or {}
    result["tracking"] = {
        "frames": int(tracking.get("frames", 0) or 0),
        "tracking_frames": int(tracking.get("tracking_frames", 0) or 0),
        "tracking_ratio": round(
            float(tracking.get("tracking_frames", 0) or 0) /
            max(int(tracking.get("frames", 0) or 0), 1), 4),
        "duration_s": float(tracking.get("duration_s", 0.0) or 0.0),
        "average_fps": float(tracking.get("average_fps", 0.0) or 0.0),
        "max_frame_interval_s": round(max(gaps, default=0.0), 6),
        "status_counts": status_counts,
        "display_status_counts": display_counts,
        "hard_loss_counts": {
            state: int(status_counts.get(state, 0) or 0)
            for state in sorted(HARD_LOSS_STATES)
            if status_counts.get(state, 0)
        },
        "reacquire_requested": bool(tracking.get("reacquire_requested", False)),
        "final_tracking_valid": bool((tracking.get("final_pose") or {}).get("tracking_valid", False)),
        "max_inliers": int(tracking.get("max_inliers", 0) or 0),
        "timing_ms": tracking.get("timing_ms") or {},
    }
    return result


def summarize_run(run_dir: Path) -> dict:
    if not run_dir.is_dir():
        raise RuntimeError(f"Live run directory does not exist: {run_dir}")
    attempt_dirs = sorted(
        (path for path in run_dir.glob("attempt_*") if path.is_dir()),
        key=lambda path: path.name,
    )
    attempts = [summarize_attempt(path) for path in attempt_dirs]
    init_counts = Counter(
        attempt["initialization_status"] or "missing" for attempt in attempts
    )
    tracking_attempts = [attempt["tracking"] for attempt in attempts
                         if attempt["tracking"] is not None]
    status_counts: Counter[str] = Counter()
    display_counts: Counter[str] = Counter()
    hard_loss_counts: Counter[str] = Counter()
    total_frames = total_tracking_frames = 0
    total_duration = 0.0
    max_gap = 0.0
    reacquire_count = final_valid_count = 0
    for tracking in tracking_attempts:
        total_frames += tracking["frames"]
        total_tracking_frames += tracking["tracking_frames"]
        total_duration += tracking["duration_s"]
        max_gap = max(max_gap, tracking["max_frame_interval_s"])
        status_counts.update(tracking["status_counts"])
        display_counts.update(tracking["display_status_counts"])
        hard_loss_counts.update(tracking["hard_loss_counts"])
        reacquire_count += int(tracking["reacquire_requested"])
        final_valid_count += int(tracking["final_tracking_valid"])

    event_counts: Counter[str] = Counter()
    event_total = 0
    events_path = run_dir / "session_events.jsonl"
    if events_path.is_file():
        try:
            with events_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        event_total += 1
                        stage = event.get("stage")
                        if isinstance(stage, str):
                            event_counts[stage] += 1
        except OSError as exc:
            raise RuntimeError(f"Cannot read events: {events_path}: {exc}") from exc

    result = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir.resolve()),
        "attempts": attempts,
        "initialization": {
            "attempts": len(attempts),
            "ready": init_counts.get("ready", 0),
            "rejected": init_counts.get("rejected", 0),
            "cancelled": init_counts.get("cancelled", 0),
            "other_or_missing": sum(count for status, count in init_counts.items()
                                     if status not in {"ready", "rejected", "cancelled"}),
        },
        "tracking": {
            "attempts": len(tracking_attempts),
            "frames": total_frames,
            "tracking_frames": total_tracking_frames,
            "tracking_ratio": round(total_tracking_frames / max(total_frames, 1), 4),
            "total_duration_s": round(total_duration, 3),
            "effective_fps": round(total_frames / max(total_duration, 1e-9), 3),
            "max_frame_interval_s": round(max_gap, 6),
            "max_frame_interval_ms": round(max_gap * 1000.0, 3),
            "status_counts": dict(status_counts),
            "display_status_counts": dict(display_counts),
            "hard_loss_counts": dict(hard_loss_counts),
            "reacquire_requested": reacquire_count,
            "final_tracking_valid": final_valid_count,
        },
        "events": {
            "total": event_total,
            "by_stage": dict(event_counts),
        },
        "review_flags": [],
    }
    flags = result["review_flags"]
    if not tracking_attempts:
        flags.append("没有生成 tracking/summary.json；可能尚未识别到目标或只进行了搜索。")
    if tracking_attempts and total_tracking_frames == 0:
        flags.append("没有有效 TRACKING 帧。")
    if hard_loss_counts:
        flags.append("出现跟踪丢失或质量拒绝状态，请结合视频动作复核。")
    if max_gap > 0.15:
        flags.append("跟踪循环存在超过 150 ms 的间隔，请检查是否由深度重捕获或相机恢复造成。")
    if event_counts.get("depth_capture_failed", 0):
        flags.append("深度采集失败过，需区分驱动恢复问题和目标姿态初始化问题。")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="One live run directory; defaults to the newest run")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Root containing timestamped live run directories")
    parser.add_argument("--output", type=Path,
                        help="Write JSON summary to this file")
    args = parser.parse_args(argv)
    if args.run_dir is None:
        args.run_dir = latest_run(args.root)
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = summarize_run(args.run_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
