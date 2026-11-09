"""Record unannotated Astra RGB motion clips for offline tracking evaluation."""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2

from .cad_video_tracker import resolve_camera_index

from .paths import SAMPLE_ROOT

DEFAULT_OUTPUT = SAMPLE_ROOT / "motion_dataset"
DEFAULT_LABELS = ("translation", "rotation", "occlusion")
MOTION_HINTS = {
    "translation": (
        "Move the object slowly left and right.",
        "Keep both ends visible; leave the camera fixed.",
    ),
    "rotation": (
        "Turn the object slowly on the table.",
        "Show both ends; keep the camera fixed.",
    ),
    "occlusion": (
        "Cover briefly, uncover, then leave and re-enter.",
        "Show the object clearly before and after each event.",
    ),
}


def next_clip_number(folder: Path) -> int:
    numbers = [
        int(match.group(1))
        for path in folder.iterdir()
        if path.is_dir() and (match := re.fullmatch(r"clip_(\d{4})", path.name))
    ]
    return max(numbers, default=0) + 1


def completed_clip_count(folder: Path) -> int:
    count = 0
    for path in folder.glob("clip_????/metadata.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete":
                count += 1
        except (OSError, json.JSONDecodeError):
            continue
    return count


def completed_labels(folder: Path) -> set[str]:
    labels = set()
    for path in folder.glob("clip_????/metadata.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        video_path = path.parent / "rgb.avi"
        if (metadata.get("status") == "complete"
                and isinstance(metadata.get("label"), str)
                and video_path.is_file() and video_path.stat().st_size > 0
                and isinstance(metadata.get("frame_count"), int)
                and metadata["frame_count"] > 0
                and metadata.get("video_decoded_frame_count") == metadata.get("frame_count")):
            labels.add(metadata["label"])
    return labels


def parse_labels(value: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in value.split(","))
    if not labels or any(not label for label in labels):
        raise ValueError("Motion labels must be nonempty comma-separated names.")
    if len(set(labels)) != len(labels):
        raise ValueError("Motion labels must be unique for reliable resume.")
    return labels


def capture_metrics(start_s: float, end_s: float,
                    frame_timestamps_s: list[float], min_duration_s: float,
                    min_capture_fps: float,
                    reason: str) -> dict[str, float | int | str]:
    duration = max(0.0, end_s - start_s)
    count = len(frame_timestamps_s)
    capture_fps = count / duration if duration > 0 else 0.0
    if reason == "error":
        status = "failed"
    elif duration < min_duration_s or count == 0:
        status = "short"
    elif capture_fps < min_capture_fps:
        status = "low_fps"
    else:
        status = "complete"
    return {
        "status": status,
        "stop_reason": reason,
        "frame_count": count,
        "elapsed_s": round(duration, 6),
        "average_capture_fps": round(capture_fps, 3),
        "min_capture_fps": min_capture_fps,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fourcc_name(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))


def inspect_video(path: Path) -> tuple[int, int, float]:
    capture = cv2.VideoCapture(str(path))
    decoded_frames = 0
    reported_frames = 0
    reported_fps = 0.0
    try:
        if capture.isOpened():
            reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded_frames += 1
    finally:
        capture.release()
    return decoded_frames, reported_frames, reported_fps


def retime_avi(path: Path, fps: float, expected_frames: int,
               frame_size: tuple[int, int], expected_duration_s: float) -> None:
    timed_path = path.with_name("rgb_timed.avi")
    source = cv2.VideoCapture(str(path))
    writer = cv2.VideoWriter(
        str(timed_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, frame_size)
    decoded_frames = 0
    try:
        if not source.isOpened() or not writer.isOpened():
            raise RuntimeError("Could not open the AVI timing conversion.")
        while True:
            ok, frame = source.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if (width, height) != frame_size:
                raise RuntimeError("AVI frame size changed during timing conversion.")
            writer.write(frame)
            decoded_frames += 1
    finally:
        source.release()
        writer.release()
    if decoded_frames != expected_frames:
        raise RuntimeError(
            f"Source AVI decoded {decoded_frames} of {expected_frames} frames.")
    verified_frames, _, verified_fps = inspect_video(timed_path)
    if verified_frames != expected_frames or verified_fps <= 0:
        raise RuntimeError(
            f"Timed AVI decoded {verified_frames} of {expected_frames} frames.")
    playback_duration_s = verified_frames / verified_fps
    if abs(playback_duration_s - expected_duration_s) > max(0.5, 0.02 * expected_duration_s):
        raise RuntimeError("Timed AVI playback duration does not match capture.")
    timed_path.replace(path)


class ClipRecorder:
    def __init__(self, folder: Path, label: str, camera_info: dict[str, object],
                 writer_fps: float, min_duration_s: float, min_capture_fps: float,
                 frame_size: tuple[int, int]):
        self.folder = folder
        self.label = label
        self.camera_info = camera_info
        self.writer_fps = writer_fps
        self.min_duration_s = min_duration_s
        self.min_capture_fps = min_capture_fps
        self.frame_size = frame_size
        self.started_at = datetime.now().astimezone().isoformat()
        self.start_s = time.perf_counter()
        self.frame_timestamps_s: list[float] = []
        self.writer = None

        folder.mkdir(parents=False, exist_ok=False)
        path = folder / "rgb.avi"
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), writer_fps, frame_size)
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            write_json(folder / "metadata.json", {
                "status": "failed",
                "error": "OpenCV could not open MJPG AVI writer.",
                "video": str(path.resolve()),
                "camera": camera_info,
            })
            raise RuntimeError(f"Cannot open video writer: {path}")

    def append(self, frame, now_s: float) -> None:
        height, width = frame.shape[:2]
        if (width, height) != self.frame_size:
            raise RuntimeError(
                f"Camera size changed from {self.frame_size} to {(width, height)}")
        self.writer.write(frame)
        self.frame_timestamps_s.append(round(now_s - self.start_s, 6))

    def elapsed(self, now_s: float) -> float:
        return now_s - self.start_s

    def finish(self, reason: str, error: str | None = None) -> dict[str, object]:
        end_s = time.perf_counter()
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        video_path = self.folder / "rgb.avi"
        metrics = capture_metrics(
            self.start_s, end_s, self.frame_timestamps_s,
            self.min_duration_s, self.min_capture_fps, reason)
        frame_count = len(self.frame_timestamps_s)
        elapsed_s = end_s - self.start_s
        output_fps = self.writer_fps
        retimed = False
        if (metrics["status"] == "complete" and frame_count > 0
                and abs(frame_count / self.writer_fps - elapsed_s) > 0.25):
            output_fps = frame_count / elapsed_s
            try:
                retime_avi(video_path, output_fps, frame_count,
                           self.frame_size, elapsed_s)
                retimed = True
            except Exception as exc:
                error = error or f"AVI timing conversion failed: {exc}"
        decoded_frame_count, reported_frame_count, video_fps = inspect_video(video_path)
        playback_duration_s = decoded_frame_count / video_fps if video_fps > 0 else 0.0
        duration_error_s = abs(playback_duration_s - elapsed_s)
        metadata: dict[str, object] = {
            "label": self.label,
            "started_at": self.started_at,
            "ended_at": datetime.now().astimezone().isoformat(),
            "video": str(video_path.resolve()),
            "format": "AVI/MJPG",
            "writer_fps": output_fps if retimed else self.writer_fps,
            "capture_writer_fps": self.writer_fps,
            "video_reported_fps": video_fps,
            "video_playback_duration_s": round(playback_duration_s, 6),
            "video_duration_error_s": round(duration_error_s, 6),
            "video_retimed": retimed,
            "image_size": list(self.frame_size),
            "camera": self.camera_info,
            "video_bytes": video_path.stat().st_size if video_path.exists() else 0,
            "video_decodes_first_frame": decoded_frame_count > 0,
            "video_reported_frame_count": reported_frame_count,
            "video_decoded_frame_count": decoded_frame_count,
            "frame_timestamps_s": self.frame_timestamps_s,
            "rgb_frames_unannotated": True,
            "depth_recorded": False,
            **metrics,
        }
        if decoded_frame_count != len(self.frame_timestamps_s):
            metadata["status"] = "failed"
            error = error or (
                f"AVI decoded {decoded_frame_count} of "
                f"{len(self.frame_timestamps_s)} captured frames."
            )
        if metrics["status"] == "complete" and duration_error_s > max(0.5, 0.02 * elapsed_s):
            metadata["status"] = "failed"
            error = error or (
                f"AVI playback duration differs from capture by {duration_error_s:.2f}s."
            )
        if error and metrics["status"] == "complete":
            metadata["status"] = "failed"
        if error:
            metadata["error"] = error
        write_json(self.folder / "metadata.json", metadata)
        return metadata


def open_astra(args: argparse.Namespace):
    index, name = resolve_camera_index(args.camera_name, args.camera)
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open Astra RGB camera {index}.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera_info = {
        "camera_index": index,
        "camera_name": name,
        "requested_camera_name": args.camera_name,
        "camera_identity_verified": name is not None,
        "backend": "DSHOW",
        "requested_image_size": [args.width, args.height],
        "requested_fps": args.fps,
        "requested_fourcc": "MJPG",
        "reported_fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "reported_fourcc": fourcc_name(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    return cap, camera_info


def draw_status(frame, label: str, stage: int, target: int,
                recorder: ClipRecorder | None, now_s: float, duration_s: float,
                feedback: str = ""):
    display = frame.copy()
    if recorder is None:
        status = f"READY {stage}/{target} | {label.upper()} | {duration_s:.0f}s"
        color = (0, 190, 0)
    else:
        status = f"REC {recorder.elapsed(now_s):.1f}/{duration_s:.0f}s | {label.upper()}"
        color = (0, 0, 255)
        cv2.circle(display, (display.shape[1] - 20, 20), 6, color, -1)
    hints = MOTION_HINTS.get(label, (label, ""))
    lines = (status, *hints, "R: start/stop    Q: quit")
    cv2.rectangle(display, (0, 0), (display.shape[1], 102), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        base_scale = 0.48 if index == 0 else 0.43
        text_width, _ = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, base_scale, 1)[0]
        scale = base_scale * min(1.0, (display.shape[1] - 16) / max(text_width, 1))
        cv2.putText(display, line, (8, 23 + 24 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color if index == 0 else (255, 255, 255), 1, cv2.LINE_AA)
    if feedback:
        height = display.shape[0]
        cv2.rectangle(display, (0, height - 30),
                      (display.shape[1], height), (0, 0, 0), -1)
        cv2.putText(display, feedback, (8, height - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 200, 0) if feedback.startswith("SAVED") else (0, 200, 255),
                    1, cv2.LINE_AA)
    return display


def record_interactive(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    labels = args.labels[:args.target_clips]
    completed = completed_labels(args.output)
    if all(label in completed for label in labels):
        print(f"All {len(labels)} clips are already complete in {args.output}")
        return
    cap = None
    recorder = None
    feedback = ""
    feedback_until_s = 0.0
    camera_info: dict[str, object] = {}
    try:
        cap, camera_info = open_astra(args)
        window = "Astra RGB motion capture"
        print("Check the preview shows the Astra RGB camera.")
        print("R starts/stops a clip; Q closes the window. Clips stop automatically at the requested duration.")
        while True:
            stage_index = next((i for i, label in enumerate(labels)
                                if label not in completed), None)
            if stage_index is None:
                print(f"Completed {len(labels)} clips in {args.output}")
                break
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Astra RGB frame read failed.")
            height, width = frame.shape[:2]
            if (width, height) != (args.width, args.height):
                raise RuntimeError(
                    f"Expected {args.width}x{args.height}, got {width}x{height}.")
            label = labels[stage_index]
            now_s = time.perf_counter()
            if recorder is not None:
                recorder.append(frame, now_s)
                if recorder.elapsed(now_s) >= args.duration:
                    metadata = recorder.finish("duration")
                    print(f"Saved {recorder.folder}: {metadata['status']}, "
                          f"{metadata['frame_count']} frames, "
                          f"{metadata['elapsed_s']:.1f}s, "
                          f"{metadata['average_capture_fps']:.1f} fps")
                    if metadata["status"] == "complete":
                        completed.add(recorder.label)
                    feedback = (f"SAVED {recorder.label}" if metadata["status"] == "complete"
                                else f"RETAKE {recorder.label}: {metadata['status']}")
                    feedback_until_s = time.perf_counter() + 4.0
                    recorder = None
                    stage_index = next((i for i, item in enumerate(labels)
                                        if item not in completed), None)
                    if stage_index is None:
                        print(f"Completed {len(labels)} clips in {args.output}")
                        break
                    label = labels[stage_index]
            cv2.imshow(window, draw_status(
                frame, label, stage_index + 1, args.target_clips,
                recorder, now_s, args.duration,
                feedback if now_s < feedback_until_s else ""))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                if recorder is not None:
                    metadata = recorder.finish("quit")
                    print(f"Saved partial clip {recorder.folder}: {metadata['status']}")
                    recorder = None
                break
            if key in (ord("r"), ord("R")):
                if recorder is not None:
                    metadata = recorder.finish("manual")
                    print(f"Saved {recorder.folder}: {metadata['status']}, "
                          f"{metadata['elapsed_s']:.1f}s")
                    if metadata["status"] == "complete":
                        completed.add(recorder.label)
                    feedback = (f"SAVED {recorder.label}" if metadata["status"] == "complete"
                                else f"RETAKE {recorder.label}: {metadata['status']}")
                    feedback_until_s = time.perf_counter() + 4.0
                    recorder = None
                else:
                    clip_folder = args.output / f"clip_{next_clip_number(args.output):04d}"
                    recorder = ClipRecorder(
                        clip_folder, label, camera_info, args.fps,
                        args.min_duration, args.min_capture_fps, (width, height))
                    feedback = ""
                    print(f"Recording {clip_folder} ({label})")
    except BaseException as exc:
        if recorder is not None:
            recorder.finish("error", str(exc))
        args.output.mkdir(parents=True, exist_ok=True)
        failure_path = args.output / f"capture_error_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        write_json(failure_path, {
            "error": f"{type(exc).__name__}: {exc}",
            "at": datetime.now().astimezone().isoformat(),
            "camera": camera_info,
        })
        raise
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera-name", default="USB Camera")
    parser.add_argument("--camera", type=int, default=None,
                        help="Manual DirectShow index override; current Astra is usually 0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--min-duration", type=float, default=20.0)
    parser.add_argument("--min-capture-fps", type=float, default=15.0)
    parser.add_argument("--target-clips", type=int, default=3)
    parser.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS))
    args = parser.parse_args()
    try:
        args.labels = parse_labels(args.labels)
    except ValueError as exc:
        parser.error(str(exc))
    if args.width < 1 or args.height < 1 or args.fps <= 0:
        parser.error("Image size and FPS must be positive.")
    if args.duration <= 0 or args.min_duration <= 0 or args.min_duration > args.duration:
        parser.error("Require 0 < --min-duration <= --duration.")
    if not 0 < args.min_capture_fps <= args.fps:
        parser.error("Require 0 < --min-capture-fps <= --fps.")
    if args.target_clips < 1 or len(args.labels) < args.target_clips:
        parser.error("Provide at least one label per target clip.")
    args.output = args.output.resolve()
    return args


def main() -> None:
    record_interactive(parse_args())


if __name__ == "__main__":
    main()
