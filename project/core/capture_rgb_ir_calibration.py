"""Capture stable RGB/IR checkerboard pairs for Astra extrinsic calibration."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .calibrate_camera import open_camera, write_image

from .paths import CALIBRATION_ROOT, PROJECT_ROOT

BOARD = (9, 6)
DEFAULT_OUTPUT = CALIBRATION_ROOT / "rgb_ir_pairs"
IR_PROBE = "core.astra_ir_probe"


def find_board(frame: np.ndarray) -> tuple[bool, np.ndarray | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
             | cv2.CALIB_CB_ACCURACY)
    return cv2.findChessboardCornersSB(gray, BOARD, flags)


def next_pair_index(output: Path) -> int:
    indices = []
    for path in output.glob("pair_[0-9][0-9][0-9][0-9]"):
        try:
            indices.append(int(path.name.removeprefix("pair_")))
        except ValueError:
            continue
    return max(indices, default=0) + 1


def read_fresh_frame(cap: cv2.VideoCapture, flush_count: int = 5) -> np.ndarray:
    frame = None
    for _ in range(flush_count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("RGB camera frame read failed.")
    assert frame is not None
    return frame


def reopen_rgb_camera(args: argparse.Namespace) -> tuple[cv2.VideoCapture, np.ndarray]:
    last_error: Exception | None = None
    for _ in range(args.rgb_reopen_attempts):
        cap = None
        try:
            cap = open_camera(args.camera, args.width, args.height)
            return cap, read_fresh_frame(cap, 10)
        except RuntimeError as exc:
            last_error = exc
            if cap is not None:
                cap.release()
            time.sleep(0.5)
    raise RuntimeError(
        f"RGB camera did not recover after {args.rgb_reopen_attempts} attempts") from last_error


def capture_pair(cap: cv2.VideoCapture, before: np.ndarray,
                 before_corners: np.ndarray,
                 args: argparse.Namespace) -> tuple[dict[str, object], cv2.VideoCapture]:
    index = next_pair_index(args.output)
    stem = f"pair_{index:04d}"
    temporary = args.output / f".{stem}_capturing"
    if temporary.exists():
        raise RuntimeError(f"Temporary capture directory already exists: {temporary}")
    temporary.mkdir(parents=True)

    command = [
        sys.executable, "-u", "-m", IR_PROBE,
        "--output", str(temporary),
        "--warmup", str(args.ir_warmup),
        "--frames", str(args.ir_frames),
    ]
    cap.release()
    worker_stdout = ""
    worker_stderr = ""
    worker_returncode = -1
    try:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=args.ir_timeout,
        )
        worker_returncode = completed.returncode
        worker_stdout = completed.stdout.strip()
        worker_stderr = completed.stderr.strip()
    except subprocess.TimeoutExpired as exc:
        worker_stderr = f"IR worker timed out after {args.ir_timeout:.1f}s"
        if exc.stderr:
            worker_stderr += f": {exc.stderr}"
    cap, after = reopen_rgb_camera(args)
    after_found, after_corners = find_board(after)
    ir_metadata_path = temporary / "metadata.json"
    ir_metadata = (json.loads(ir_metadata_path.read_text(encoding="utf-8"))
                   if ir_metadata_path.exists() else {})

    reasons = []
    if worker_returncode != 0:
        reasons.append(f"IR worker failed with exit code {worker_returncode}")
    if not ir_metadata.get("checkerboard_found", False):
        reasons.append("IR checkerboard was not detected")
    if not after_found:
        reasons.append("RGB checkerboard was not detected after IR capture")

    median_drift = float("inf")
    p95_drift = float("inf")
    if after_found:
        distances = np.linalg.norm(
            before_corners.reshape(-1, 2) - after_corners.reshape(-1, 2), axis=1)
        median_drift = float(np.median(distances))
        p95_drift = float(np.percentile(distances, 95))
        if median_drift > args.max_median_drift:
            reasons.append(
                f"RGB corner median drift {median_drift:.3f}px exceeds "
                f"{args.max_median_drift:.3f}px")
        if p95_drift > args.max_p95_drift:
            reasons.append(
                f"RGB corner p95 drift {p95_drift:.3f}px exceeds "
                f"{args.max_p95_drift:.3f}px")

    accepted = not reasons
    write_image(temporary / "rgb.png", before)
    write_image(temporary / "rgb_before.png", before)
    write_image(temporary / "rgb_after.png", after)
    manifest = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "accepted": accepted,
        "pair_index": index,
        "board_size": list(BOARD),
        "square_size_mm": args.square_mm,
        "rgb_corner_median_drift_px": median_drift,
        "rgb_corner_p95_drift_px": p95_drift,
        "rejection_reasons": reasons,
        "ir_worker_stdout": worker_stdout,
        "ir_worker_stderr": worker_stderr,
        "ir_metadata": ir_metadata,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if accepted:
        destination = args.output / stem
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = args.output / f"rejected_{stamp}"
    temporary.rename(destination)
    manifest["path"] = str(destination.resolve())
    return manifest, cap


def draw_preview(frame: np.ndarray, corners: np.ndarray | None,
                 found: bool, accepted_count: int, target_count: int,
                 status: str) -> np.ndarray:
    preview = frame.copy()
    if found:
        cv2.drawChessboardCorners(preview, BOARD, corners, True)
    color = (0, 210, 0) if found else (0, 160, 255)
    cv2.putText(preview, f"RGB corners: {'YES' if found else 'NO'} | pairs {accepted_count}/{target_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(preview, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, color, 1, cv2.LINE_AA)
    return preview


def accepted_pair_count(output: Path) -> int:
    return len(list(output.glob("pair_[0-9][0-9][0-9][0-9]")))


def run_once(cap: cv2.VideoCapture, args: argparse.Namespace
             ) -> tuple[dict[str, object], cv2.VideoCapture]:
    for _ in range(60):
        frame = read_fresh_frame(cap, 1)
        found, corners = find_board(frame)
        if found:
            return capture_pair(cap, frame, corners, args)
    raise RuntimeError("RGB checkerboard was not detected within 60 frames.")


def run_interactive(cap: cv2.VideoCapture, args: argparse.Namespace) -> None:
    window = "Astra RGB-IR calibration"
    tick = 0
    found = False
    corners = None
    status = "Move board; hold still and press S"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("RGB camera frame read failed.")
            if tick % 4 == 0:
                found, corners = find_board(frame)
            tick += 1
            count = accepted_pair_count(args.output)
            cv2.imshow(window, draw_preview(frame, corners, found, count,
                                            args.target_count, status))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                current_found, current_corners = find_board(frame)
                if not current_found:
                    status = "RGB corners not found"
                    continue
                manifest, cap = capture_pair(cap, frame.copy(), current_corners.copy(), args)
                if manifest["accepted"]:
                    status = (f"Saved pair {manifest['pair_index']:04d}; "
                              "move board to a new pose")
                else:
                    status = "Rejected: " + "; ".join(manifest["rejection_reasons"])
                if accepted_pair_count(args.output) >= args.target_count:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--target-count", type=int, default=18)
    parser.add_argument("--ir-warmup", type=int, default=2)
    parser.add_argument("--ir-frames", type=int, default=3)
    parser.add_argument("--ir-timeout", type=float, default=15.0)
    parser.add_argument("--rgb-reopen-attempts", type=int, default=3)
    parser.add_argument("--max-median-drift", type=float, default=0.75)
    parser.add_argument("--max-p95-drift", type=float, default=1.5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.ir_frames < 1 or args.ir_frames % 2 == 0:
        parser.error("--ir-frames must be a positive odd number")
    if args.rgb_reopen_attempts < 1:
        parser.error("--rgb-reopen-attempts must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.camera, args.width, args.height)
    if args.once:
        try:
            manifest, cap = run_once(cap, args)
        finally:
            cap.release()
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        run_interactive(cap, args)


if __name__ == "__main__":
    main()
