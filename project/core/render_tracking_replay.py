"""Render a recorded RGB clip beside its frame-aligned CAD tracking log."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

try:
    import av
except ImportError:  # pragma: no cover - depends on optional video support
    av = None

from .cad_pose_overlay import load_calibration
from .cad_video_tracker import (BOX_DISPLAY_Y_OFFSET_PX, DISPLAY_POSE_STATES,
                               load_box_display_y_offset, load_model_box,
                               render_box_layer)

from .paths import CALIBRATION_ROOT, MODEL_ROOT, SAMPLE_ROOT, TRACKING_ROOT

DEFAULT_EVALUATION = TRACKING_ROOT / "evaluations" / "captured_20261031"
DEFAULT_MODEL = MODEL_ROOT / "energy_unit_centered_mm.stl"
DEFAULT_INTRINSICS = CALIBRATION_ROOT / "camera_intrinsics.npz"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Tracking log has no frames: {path}")
    for index, row in enumerate(rows):
        if int(row["frame_index"]) != index:
            raise ValueError(f"Tracking log frame indices are not sequential at row {index}")
        for field in ("source_elapsed_s", "rvec_x", "rvec_y", "rvec_z",
                      "tx_mm", "ty_mm", "tz_mm"):
            if not np.isfinite(float(row[field])):
                raise ValueError(f"Invalid {field} at frame {index}")
    return rows


def status_text(row: dict[str, str]) -> str:
    if row.get("session_action") == "RELOCALIZED":
        return "RELOCALIZED / VERIFYING"
    return row["status"]


def decorate_frame(frame: np.ndarray, row: dict[str, str], box_corners: np.ndarray,
                   K: np.ndarray, dist: np.ndarray,
                   side_by_side: bool = True,
                   box_display_y_offset_px: float = BOX_DISPLAY_Y_OFFSET_PX) -> np.ndarray:
    height, width = frame.shape[:2]
    tracked = frame.copy()
    status = row["status"]
    if status in DISPLAY_POSE_STATES:
        rvec = np.array([float(row[field]) for field in
                         ("rvec_x", "rvec_y", "rvec_z")], dtype=np.float64)
        tvec = np.array([float(row[field]) for field in
                         ("tx_mm", "ty_mm", "tz_mm")], dtype=np.float64)
        layer = render_box_layer(frame.shape, box_corners, K, dist, rvec, tvec,
                                 box_display_y_offset_px)
        tracked = cv2.addWeighted(tracked, 1.0, layer, 0.72, 0)

    label = status_text(row)
    color = ((78, 211, 132) if status in DISPLAY_POSE_STATES else
             (52, 211, 238) if row.get("session_action") == "RELOCALIZED" else
             (83, 157, 249) if status == "OUT OF VIEW" else (67, 185, 255))
    cv2.rectangle(tracked, (0, height - 72), (width, height), (25, 30, 33), -1)
    cv2.rectangle(tracked, (0, height - 72), (5, height), color, -1)
    cv2.putText(tracked, label, (15, height - 43), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, color, 2, cv2.LINE_AA)
    cv2.putText(tracked, f"t={float(row['source_elapsed_s']):05.2f}s  "
                f"frame {int(row['frame_index']) + 1}",
                (15, height - 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (232, 235, 235), 1, cv2.LINE_AA)

    if not side_by_side:
        return tracked
    raw = frame.copy()
    cv2.rectangle(raw, (0, height - 72), (width, height), (25, 30, 33), -1)
    cv2.putText(raw, "ORIGINAL RGB", (15, height - 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.67, (232, 235, 235), 2, cv2.LINE_AA)
    combined = np.concatenate((raw, tracked), axis=1)
    cv2.line(combined, (width, 0), (width, height - 1), (230, 230, 230), 2)
    return combined


def keyframe_indices(rows: list[dict[str, str]], start_s: float,
                     end_s: float | None) -> list[tuple[str, int]]:
    eligible = [(i, row) for i, row in enumerate(rows)
                if start_s <= float(row["source_elapsed_s"])
                and (end_s is None or float(row["source_elapsed_s"]) <= end_s)]
    choices = (
        ("Before occlusion", "TRACKING", None, 7.0),
        ("Out of view", "OUT OF VIEW", None, 12.0),
        ("Relocalized", None, "RELOCALIZED", 19.5),
        ("Recovered tracking", "TRACKING", None, 23.0),
    )
    selected = []
    for label, status, action, target_time in choices:
        candidates = [(i, row) for i, row in eligible
                      if (status is None or row["status"] == status)
                      and (action is None or row.get("session_action") == action)]
        if not candidates:
            continue
        index, _ = min(candidates, key=lambda item: abs(
            float(item[1]["source_elapsed_s"]) - target_time))
        selected.append((label, index))
    return selected


def write_keyframes(path: Path, images: list[tuple[str, np.ndarray]]) -> None:
    if not images:
        return
    height, width = images[0][1].shape[:2]
    sheet = np.full((2 * (height + 42), 2 * width, 3), 28, dtype=np.uint8)
    for position, (label, frame) in enumerate(images[:4]):
        col = position % 2
        row = position // 2
        x, y = col * width, row * (height + 42)
        cv2.putText(sheet, label, (x + 14, y + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.73, (237, 239, 239), 2, cv2.LINE_AA)
        sheet[y + 42:y + 42 + height, x:x + width] = frame
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", sheet)
    if not ok:
        raise RuntimeError(f"Cannot encode keyframe sheet: {path}")
    encoded.tofile(path)


def render(video: Path, log: Path, output: Path, model: Path, intrinsics: Path,
           keyframes: Path | None = None, start_s: float = 0.0,
           end_s: float | None = None, side_by_side: bool = True,
           box_display_y_offset_px: float = BOX_DISPLAY_Y_OFFSET_PX) -> dict[str, object]:
    if start_s < 0 or end_s is not None and end_s <= start_s:
        raise ValueError("Require 0 <= start_s < end_s")
    if not np.isfinite(box_display_y_offset_px) or abs(box_display_y_offset_px) > 100:
        raise ValueError("Box display Y offset must be finite and within +/-100 pixels")
    if av is None:
        raise RuntimeError(
            "PyAV is required to render video replays. Install it with "
            "python -m pip install av or install project/requirements-linux.txt."
        )
    rows = read_rows(log)
    K, dist, image_size = load_calibration(intrinsics)
    box_corners = load_model_box(model)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid source FPS: {video}")
    width, height = image_size
    output_size = (width * 2 if side_by_side else width, height)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = av.open(str(output), mode="w", options={"movflags": "+faststart"})
        video_stream = writer.add_stream("libx264", rate=Fraction(str(fps)))
        video_stream.width, video_stream.height = output_size
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": "22", "preset": "veryfast"}
    except Exception:
        capture.release()
        raise

    selected_keyframes = dict((index, label) for label, index in
                              keyframe_indices(rows, start_s, end_s)) if keyframes else {}
    sheet_images: list[tuple[str, np.ndarray]] = []
    decoded = written = 0
    try:
        for index, row in enumerate(rows):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended before log frame {index}: {video}")
            decoded += 1
            if frame.shape[1::-1] != image_size:
                raise RuntimeError(f"Video frame {index} size differs from calibration")
            timestamp = float(row["source_elapsed_s"])
            if timestamp < start_s or end_s is not None and timestamp > end_s:
                continue
            decorated = decorate_frame(frame, row, box_corners, K, dist,
                                       side_by_side, box_display_y_offset_px)
            for packet in video_stream.encode(av.VideoFrame.from_ndarray(
                    decorated, format="bgr24")):
                writer.mux(packet)
            written += 1
            if index in selected_keyframes:
                sheet_images.append((selected_keyframes[index], decorated[:, width:].copy()
                                     if side_by_side else decorated.copy()))
        if capture.read()[0]:
            raise RuntimeError(f"Video has more frames than tracking log: {video}")
    finally:
        for packet in video_stream.encode():
            writer.mux(packet)
        writer.close()
        capture.release()
    if written == 0:
        raise RuntimeError("No video frames fall within the selected time range")
    if keyframes:
        labels = {label: image for label, image in sheet_images}
        ordered = [(label, labels[label]) for label, _ in
                   keyframe_indices(rows, start_s, end_s) if label in labels]
        write_keyframes(keyframes, ordered)
    return {"video": str(output.resolve()), "decoded_frames": decoded,
            "written_frames": written, "fps": fps, "size": output_size,
            "keyframes": str(keyframes.resolve()) if keyframes else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", default="session_03")
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keyframes", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--overlay-only", action="store_true")
    parser.add_argument("--box-display-y-offset-px", type=float,
                        help="Display-only vertical box offset in pixels; defaults to saved setting")
    args = parser.parse_args()
    video = args.video or SAMPLE_ROOT / "integration_sessions" / args.session_id / \
        "motion" / "clip_0001" / "rgb.avi"
    log = args.log or args.evaluation / "tracking" / args.session_id / "frames.csv"
    output = args.output or args.evaluation / "videos" / f"{args.session_id}_replay.mp4"
    box_display_y_offset_px = (args.box_display_y_offset_px
                               if args.box_display_y_offset_px is not None
                               else load_box_display_y_offset())
    result = render(video, log, output, args.model, args.intrinsics,
                    args.keyframes, args.start_s, args.end_s,
                    not args.overlay_only, box_display_y_offset_px)
    print(f"Saved {result['video']}: {result['written_frames']} frames, "
          f"{result['size'][0]}x{result['size'][1]} at {result['fps']:.3f} FPS")
    if result["keyframes"]:
        print(f"Saved {result['keyframes']}")


if __name__ == "__main__":
    main()
