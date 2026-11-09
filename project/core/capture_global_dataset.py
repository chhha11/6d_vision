"""Collect labeled RGB and five-frame depth samples for global pose evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .cad_video_tracker import resolve_camera_index
from .calibrate_camera import open_camera, write_image

from .paths import CALIBRATION_ROOT, PROJECT_ROOT, SAMPLE_ROOT

DEFAULT_OUTPUT = SAMPLE_ROOT / "global_dataset"
DEFAULT_EXTRINSICS = CALIBRATION_ROOT / "rgb_ir" / "rgb_ir_extrinsics.npz"
DEPTH_PROBE = "core.astra_depth_probe"
IMAGE_SIZE = (640, 480)
PLAN_FIELDS = ("id", "kind", "distance_mm", "pose", "position", "background",
               "requirement", "target_present")


@dataclass(frozen=True)
class Sample:
    id: str
    kind: str
    distance_mm: float | None
    pose: str
    position: str
    background: str
    requirement: str
    target_present: bool


def load_plan(path: Path) -> list[Sample]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in PLAN_FIELDS):
            raise ValueError(f"Plan must include: {', '.join(PLAN_FIELDS)}")
        samples = [sample_from_row(row) for row in reader]
    if not samples:
        raise ValueError("Capture plan is empty.")
    ids = [sample.id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Capture plan contains duplicate IDs.")
    return samples


def sample_from_row(row: dict[str, str | None]) -> Sample:
    sample_id = (row.get("id") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
        raise ValueError(f"Invalid sample ID: {sample_id!r}")
    kind = (row.get("kind") or "").strip().lower()
    if kind not in {"positive", "negative"}:
        raise ValueError(f"Sample {sample_id}: kind must be positive or negative.")
    raw_distance = (row.get("distance_mm") or "").strip()
    distance = float(raw_distance) if raw_distance else None
    if kind == "positive" and (distance is None or not math.isfinite(distance) or distance <= 0):
        raise ValueError(f"Sample {sample_id}: positive distance_mm must be positive.")
    if distance is not None and (not math.isfinite(distance) or distance <= 0):
        raise ValueError(f"Sample {sample_id}: distance_mm must be positive.")
    raw_present = (row.get("target_present") or "").strip().lower()
    if raw_present not in {"true", "false"}:
        raise ValueError(f"Sample {sample_id}: target_present must be true or false.")
    target_present = raw_present == "true"
    if kind == "positive" and not target_present:
        raise ValueError(f"Sample {sample_id}: a positive sample must contain the target.")
    return Sample(
        id=sample_id,
        kind=kind,
        distance_mm=distance,
        pose=(row.get("pose") or "").strip(),
        position=(row.get("position") or "").strip(),
        background=(row.get("background") or "").strip(),
        requirement=(row.get("requirement") or "").strip(),
        target_present=target_present,
    )


def completed(sample: Sample, output: Path) -> bool:
    manifest = output / sample.id / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return bool(data.get("accepted", False))
    return (output / f"skipped_{sample.id}.json").is_file()


def next_pending(samples: list[Sample], output: Path) -> Sample | None:
    return next((sample for sample in samples if not completed(sample, output)), None)


def plan_summary(samples: list[Sample], output: Path) -> str:
    positive = [sample for sample in samples if sample.kind == "positive"]
    negative = [sample for sample in samples if sample.kind == "negative"]
    positive_saved = sum((output / sample.id / "manifest.json").is_file()
                         for sample in positive)
    negative_saved = sum((output / sample.id / "manifest.json").is_file()
                         for sample in negative)
    skipped = sum((output / f"skipped_{sample.id}.json").is_file()
                  for sample in samples)
    summary = (f"Saved {positive_saved}/{len(positive)} positive and "
               f"{negative_saved}/{len(negative)} negative samples; "
               f"skipped {skipped}.")
    return summary + (" Skipped positives need replacement captures." if skipped else "")


def next_free_sample_id(output: Path) -> str:
    used = []
    for path in output.glob("sample_[0-9][0-9][0-9][0-9]"):
        if path.is_dir():
            used.append(int(path.name.split("_")[1]))
    for path in output.glob("skipped_sample_[0-9][0-9][0-9][0-9].json"):
        used.append(int(path.stem.split("_")[-1]))
    return f"sample_{max(used, default=0) + 1:04d}"


def read_rgb(cap: cv2.VideoCapture, frames: int = 5) -> np.ndarray:
    frame = None
    for _ in range(frames):
        try:
            ok, frame = cap.read()
        except (OSError, cv2.error) as exc:
            raise RuntimeError("Astra RGB frame read failed.") from exc
        if not ok:
            raise RuntimeError("Astra RGB frame read failed.")
    assert frame is not None
    if frame.shape[:2] != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
        raise RuntimeError(f"RGB image is {frame.shape[1]}x{frame.shape[0]}; expected 640x480.")
    return frame


def reopen_rgb(index: int, attempts: int,
               warmup_frames: int = 10,
               camera_name: str | None = None,
               camera_identity=None,
               camera_fourcc: str | None = None,
               camera_fps: float | None = None) -> tuple[cv2.VideoCapture, np.ndarray]:
    last_error: Exception | None = None
    for attempt_index in range(attempts):
        cap = None
        try:
            selected_index = index
            if camera_name:
                selected_index, selected_name = resolve_camera_index(camera_name, None)
                if selected_name is None or selected_name.casefold() != camera_name.casefold():
                    raise RuntimeError(
                        f"DirectShow camera identity changed: expected {camera_name!r}, "
                        f"got {selected_name!r}")
            open_args = (*IMAGE_SIZE, camera_fourcc, camera_fps)
            cap = (camera_identity.open(open_camera, *open_args)
                   if camera_identity is not None else open_camera(selected_index, *open_args))
            return cap, read_rgb(cap, warmup_frames)
        except (OSError, RuntimeError, cv2.error) as exc:
            last_error = exc
            if cap is not None:
                cap.release()
            if attempt_index + 1 < attempts:
                time.sleep(0.5)
    raise RuntimeError(f"Astra RGB did not recover after {attempts} attempts") from last_error


def map_rgb_roi_to_depth(roi: tuple[int, int, int, int], distance_mm: float,
                         K_rgb: np.ndarray, dist_rgb: np.ndarray,
                         K_depth: np.ndarray, T_depth_rgb: np.ndarray,
                         image_shape: tuple[int, int], margin_px: int = 16
                         ) -> tuple[int, int, int, int]:
    x, y, width, height = roi
    if width <= 0 or height <= 0:
        raise ValueError("Object ROI must have positive width and height.")
    corners = np.array([[x, y], [x + width, y], [x, y + height],
                        [x + width, y + height]], dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(corners, K_rgb, dist_rgb).reshape(-1, 2)
    rgb_points = np.column_stack((normalized * distance_mm,
                                  np.full(4, distance_mm)))
    depth_points = rgb_points @ T_depth_rgb[:3, :3].T + T_depth_rgb[:3, 3]
    if np.any(depth_points[:, 2] <= 0):
        raise ValueError("Object ROI projects behind the depth camera.")
    projected = depth_points[:, :2] / depth_points[:, 2, None]
    projected = projected @ K_depth[:2, :2].T + K_depth[:2, 2]
    h, w = image_shape
    left = max(0, int(np.floor(projected[:, 0].min())) - margin_px)
    right = min(w, int(np.ceil(projected[:, 0].max())) + margin_px)
    top = max(0, int(np.floor(projected[:, 1].min())) - margin_px)
    bottom = min(h, int(np.ceil(projected[:, 1].max())) + margin_px)
    if right <= left or bottom <= top:
        raise ValueError("Object ROI is outside the depth image.")
    return left, top, right - left, bottom - top


def depth_presence(depth_mm: np.ndarray, roi: tuple[int, int, int, int],
                   distance_mm: float, tolerance_mm: float = 200.0,
                   min_pixels: int = 100, min_ratio: float = 0.02) -> dict[str, object]:
    x, y, width, height = roi
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > depth_mm.shape[1] or y + height > depth_mm.shape[0]:
        raise ValueError("Depth ROI must be inside the image.")
    region = depth_mm[y:y + height, x:x + width]
    nearby = region[(region >= distance_mm - tolerance_mm)
                    & (region <= distance_mm + tolerance_mm)]
    count = int(nearby.size)
    ratio = float(count / region.size)
    return {
        "depth_roi_xywh": list(roi),
        "distance_window_mm": [distance_mm - tolerance_mm,
                               distance_mm + tolerance_mm],
        "nearby_pixels": count,
        "nearby_ratio": ratio,
        "nearby_median_mm": float(np.median(nearby)) if count else None,
        "passed": bool(count >= min_pixels and ratio >= min_ratio),
    }


def depth_qc(depth_path: Path, sample: Sample, rgb_roi: tuple[int, int, int, int],
             extrinsics_path: Path) -> dict[str, object]:
    with np.load(depth_path) as archive:
        depth = archive["median_depth_mm"].copy()
        frames = archive["depth_frames_mm"].copy()
    if depth.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0]) or frames.shape != (5, *depth.shape):
        raise ValueError("Expected five 640x480 depth frames and a median frame.")
    with np.load(extrinsics_path) as archive:
        K_rgb = archive["K_rgb"].copy()
        dist_rgb = archive["dist_rgb"].copy()
        K_depth = archive["K_depth"].copy()
        T_depth_rgb = archive["T_depth_rgb"].copy()
    assert sample.distance_mm is not None
    mapped_roi = map_rgb_roi_to_depth(rgb_roi, sample.distance_mm, K_rgb,
                                       dist_rgb, K_depth, T_depth_rgb,
                                       depth.shape)
    median_screen = depth_presence(depth, mapped_roi, sample.distance_mm)
    frame_screens = [depth_presence(frame, mapped_roi, sample.distance_mm)
                     for frame in frames]
    passing_frames = sum(bool(screen["passed"]) for screen in frame_screens)
    return {
        **median_screen,
        "passing_frames": passing_frames,
        "required_passing_frames": 3,
        "frame_nearby_pixels": [screen["nearby_pixels"] for screen in frame_screens],
        "passed": bool(median_screen["passed"] and passing_frames >= 3),
    }


def validate_depth_archive(path: Path) -> None:
    with np.load(path) as archive:
        frames = archive["depth_frames_mm"]
        median = archive["median_depth_mm"]
        expected_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0])
        if (frames.shape != (5, *expected_shape) or median.shape != expected_shape
                or frames.dtype != np.uint16 or median.dtype != np.uint16):
            raise ValueError("Depth archive must contain five 640x480 uint16 frames and median.")


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def select_rgb_camera(args: argparse.Namespace) -> dict[str, object]:
    requested_index = args.camera
    index, name = resolve_camera_index(args.camera_name, requested_index)
    args.camera = index
    return {
        "camera_index": index,
        "camera_name": name,
        "requested_camera_name": args.camera_name,
        "manual_index_override": requested_index is not None,
        "camera_identity_verified": name is not None,
    }


def rgb_stream_settings(cap: cv2.VideoCapture, frame: np.ndarray) -> dict[str, object]:
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    code = [(fourcc >> (8 * index)) & 0xFF for index in range(4)]
    return {
        "backend": "DSHOW",
        "actual_frame_size": [int(frame.shape[1]), int(frame.shape[0])],
        "reported_frame_size": [float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
        "reported_fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "reported_fourcc": "".join(map(chr, code)) if all(32 <= c < 127 for c in code) else None,
        "reported_fourcc_code": fourcc,
        "reported_auto_exposure": float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
        "reported_exposure": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
        "reported_gain": float(cap.get(cv2.CAP_PROP_GAIN)),
        "reported_brightness": float(cap.get(cv2.CAP_PROP_BRIGHTNESS)),
        "reported_contrast": float(cap.get(cv2.CAP_PROP_CONTRAST)),
    }


def capture_one(cap: cv2.VideoCapture, sample: Sample,
                rgb_roi: tuple[int, int, int, int] | None,
                args: argparse.Namespace) -> tuple[dict[str, object], cv2.VideoCapture]:
    if sample.target_present and rgb_roi is None:
        raise ValueError("Draw a bounding box around the visible target before capture.")
    if rgb_roi is not None:
        x, y, width, height = rgb_roi
        if (width <= 0 or height <= 0 or x < 0 or y < 0
                or x + width > IMAGE_SIZE[0] or y + height > IMAGE_SIZE[1]):
            raise ValueError("RGB target box must be inside the 640x480 image.")
    if (args.output / sample.id).exists() or (args.output / f"skipped_{sample.id}.json").exists():
        raise RuntimeError(f"Sample ID has already been used: {sample.id}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temporary = args.output / f".{sample.id}_{stamp}_capturing"
    temporary.mkdir(parents=True)
    calibration_info = None
    calibration_copy = None
    if sample.kind == "positive":
        calibration_copy = temporary / "rgb_depth_extrinsics.npz"
        shutil.copy2(args.extrinsics, calibration_copy)
        calibration_info = {
            "file": calibration_copy.name,
            "sha256": hashlib.sha256(calibration_copy.read_bytes()).hexdigest(),
            "source": str(args.extrinsics),
        }
    before = read_rgb(cap, 3)
    camera_before = rgb_stream_settings(cap, before)
    write_image(temporary / "rgb_before.png", before)
    cap.release()

    command = [sys.executable, "-u", "-m", DEPTH_PROBE,
               "--output", str(temporary / "depth"),
               "--warmup", str(args.depth_warmup), "--frames", "5"]
    worker_result = {"returncode": -1, "stdout": "", "stderr": ""}
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                                   text=True, timeout=args.depth_timeout)
        worker_result = {"returncode": completed.returncode,
                         "stdout": completed.stdout.strip(),
                         "stderr": completed.stderr.strip()}
    except subprocess.TimeoutExpired as exc:
        worker_result["stderr"] = f"Depth worker timed out after {args.depth_timeout}s"
        if exc.stderr:
            worker_result["stderr"] += f": {exc.stderr}"
    except OSError as exc:
        worker_result["stderr"] = f"Depth worker could not start: {exc}"

    try:
        cap, after = reopen_rgb(args.camera, args.rgb_reopen_attempts)
    except RuntimeError as exc:
        write_json(temporary / "manifest.json", {
            "sample": asdict(sample), "accepted": False,
            "failure": str(exc), "depth_worker": worker_result,
            "rgb_camera": {**args.camera_info, "before": camera_before},
            "rgb_depth_calibration": calibration_info,
        })
        failed_path = args.output / f"failed_{sample.id}_{stamp}"
        temporary.rename(failed_path)
        raise RuntimeError(f"RGB did not reopen; attempt preserved at {failed_path}") from exc
    write_image(temporary / "rgb_after.png", after)
    write_image(temporary / "rgb.png", before)
    camera_after = rgb_stream_settings(cap, after)
    reasons = []
    if worker_result["returncode"] != 0:
        reasons.append(f"Depth worker failed: {worker_result['stderr']}")
    depth_archive = temporary / "depth" / "depth_burst.npz"
    depth_metadata = temporary / "depth" / "metadata.json"
    if not depth_archive.is_file() or not depth_metadata.is_file():
        reasons.append("Depth burst or metadata is missing")
    depth_meta: dict[str, object] = {}
    if depth_metadata.is_file():
        try:
            depth_meta = json.loads(depth_metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"Depth metadata could not be read: {exc}")
    if depth_meta.get("depth_mirroring", {}).get("active") is not False:
        reasons.append("Depth mirroring is not confirmed off")
    if depth_meta.get("video_mode", {}).get("width") != IMAGE_SIZE[0] or depth_meta.get("video_mode", {}).get("height") != IMAGE_SIZE[1]:
        reasons.append("Depth image size is not 640x480")
    if depth_meta.get("video_mode", {}).get("depth_unit_mm") != 1:
        reasons.append("Depth units are not confirmed as millimeters")
    if depth_meta.get("captured_frames") != 5:
        reasons.append("Depth burst does not contain five captured frames")
    if depth_meta.get("registration", {}).get("requested_depth_to_color") is not False:
        reasons.append("Depth registration is not confirmed off")
    if depth_archive.is_file():
        try:
            validate_depth_archive(depth_archive)
        except (OSError, KeyError, ValueError) as exc:
            reasons.append(f"Depth archive is invalid: {exc}")

    qc: dict[str, object] | None = None
    if sample.kind == "positive" and depth_archive.is_file() and calibration_copy is not None and not any(
            reason.startswith("Depth archive is invalid") for reason in reasons):
        try:
            qc = depth_qc(depth_archive, sample, rgb_roi, calibration_copy)
        except (OSError, KeyError, ValueError, cv2.error) as exc:
            reasons.append(f"Depth quality check failed: {exc}")
        else:
            if not qc["passed"]:
                reasons.append("Object ROI has too few depth pixels near planned distance")
    accepted = not reasons
    manifest = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "sample": asdict(sample),
        "target_present": sample.target_present,
        "rgb_camera": {**args.camera_info, "before": camera_before,
                       "after": camera_after},
        "rgb_depth_calibration": calibration_info,
        "accepted": accepted,
        "rgb_roi_xywh": list(rgb_roi) if rgb_roi is not None else None,
        "depth_presence_screen": qc,
        "depth_worker": worker_result,
        "depth_metadata": depth_meta,
        "rejection_reasons": reasons,
        "capture_order": "RGB before -> RGB released -> depth burst -> RGB after",
        "quality_note": "Depth presence is a collection check, not a CAD pose validation.",
    }
    write_json(temporary / "manifest.json", manifest)
    destination = (args.output / sample.id if accepted else
                   args.output / f"failed_{sample.id}_{stamp}")
    temporary.rename(destination)
    manifest["path"] = str(destination.resolve())
    return manifest, cap


def skip_sample(sample: Sample, output: Path) -> Path:
    path = output / f"skipped_{sample.id}.json"
    if path.exists() or (output / sample.id).exists():
        raise RuntimeError(f"Sample ID has already been used: {sample.id}")
    write_json(path, {"sample": asdict(sample),
                      "skipped_at": datetime.now().astimezone().isoformat(),
                      "reason": "Skipped at capture window"})
    return path


class RoiSelector:
    def __init__(self, image_size: tuple[int, int], roi: tuple[int, int, int, int] | None = None):
        self.width, self.height = image_size
        self.reset()
        self.roi = roi

    def reset(self) -> None:
        self.roi: tuple[int, int, int, int] | None = None
        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        self.feedback = ""

    def arm(self) -> None:
        self.reset()
        self.feedback = "Drag mouse over target; release to set box"

    @property
    def drag_roi(self) -> tuple[int, int, int, int] | None:
        if self.start is None or self.current is None:
            return None
        x1, x2 = sorted((self.start[0], self.current[0]))
        y1, y2 = sorted((self.start[1], self.current[1]))
        return x1, y1, x2 - x1, y2 - y1

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        point = (max(0, min(int(x), self.width - 1)),
                 max(0, min(int(y), self.height - 1)))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi = None
            self.start = self.current = point
            self.feedback = "Drawing target box"
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.current = point
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            self.current = point
            box = self.drag_roi
            self.start = self.current = None
            if box is not None and box[2] > 0 and box[3] > 0:
                self.roi = box
                self.feedback = "Target box set; press S to capture"
            else:
                self.feedback = "Box not set; drag mouse over target"


def fitted_text_scale(line: str, max_width: int, preferred: float = 0.48) -> float:
    width, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, preferred, 1)[0]
    scale = min(preferred, preferred * max_width / max(width, 1))
    while cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > max_width:
        scale *= 0.98
    return scale


def draw_preview(frame: np.ndarray, sample: Sample, roi: tuple[int, int, int, int] | None,
                 accepted_count: int, total_count: int, status: str,
                 drag_roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    canvas = frame.copy()
    tint = canvas.copy()
    cv2.rectangle(tint, (0, 0), (canvas.shape[1], 142), (10, 10, 10), -1)
    canvas = cv2.addWeighted(canvas, 0.65, tint, 0.35, 0)
    if roi is not None:
        x, y, width, height = roi
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 255, 0), 2)
    if drag_roi is not None:
        x, y, width, height = drag_roi
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 210, 255), 2)
    distance = f"{sample.distance_mm:g}mm" if sample.distance_mm else "distance n/a"
    lines = (
        f"{accepted_count}/{total_count} | {sample.id} {sample.kind} | {distance}",
        f"{sample.pose} | {sample.position} | {sample.background}",
        f"Setup: {sample.requirement}",
        "Target visible: drag to box (B resets) | S capture" if sample.target_present
        else "Target absent: S capture",
        "Hands out; keep object/camera still 5s | N skip | Q quit",
        status,
    )
    for row, line in enumerate(lines):
        scale = fitted_text_scale(line, canvas.shape[1] - 16)
        cv2.putText(canvas, line, (8, 20 + 23 * row),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    1, cv2.LINE_AA)
    return canvas


def announce(sample: Sample, accepted_count: int, total_count: int) -> None:
    print(f"[{accepted_count}/{total_count}] {sample.id} | {sample.kind} | "
          f"distance {sample.distance_mm or '-'} mm | {sample.pose} | "
          f"{sample.position} | {sample.background}", flush=True)
    print(sample.requirement, flush=True)


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    plan = load_plan(args.plan) if args.plan else None
    if plan is not None:
        sample = next_pending(plan, args.output)
        if sample is None:
            print(plan_summary(plan, args.output))
            return
    else:
        sample = Sample(next_free_sample_id(args.output), args.kind, args.distance_mm,
                        args.pose_label, "", "", args.notes,
                        args.target_present or args.kind == "positive")
    if sample.kind == "positive" and not args.extrinsics.is_file():
        raise RuntimeError(f"RGB-depth extrinsics not found: {args.extrinsics}")
    args.camera_info = select_rgb_camera(args)
    accepted_count = len([item for item in plan if (args.output / item.id / "manifest.json").is_file()]) if plan else 0
    total_count = len(plan) if plan else args.target_count
    announce(sample, accepted_count, total_count)
    cap = open_camera(args.camera, *IMAGE_SIZE)
    window = "Astra global pose dataset"
    selector = RoiSelector(IMAGE_SIZE, args.roi)
    status = "Ready"
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, selector.on_mouse)
        while True:
            frame = read_rgb(cap, 1)
            cv2.imshow(window, draw_preview(
                frame, sample, selector.roi, accepted_count, total_count,
                selector.feedback or status, selector.drag_roi))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("b"), ord("B")):
                selector.arm()
                continue
            if key in (ord("n"), ord("N")):
                skip_sample(sample, args.output)
                print(f"Skipped {sample.id}", flush=True)
                sample = next_pending(plan, args.output) if plan else Sample(
                    next_free_sample_id(args.output), args.kind, args.distance_mm,
                    args.pose_label, "", "", args.notes,
                    args.target_present or args.kind == "positive")
                if sample is None:
                    print(plan_summary(plan, args.output), flush=True)
                    break
                selector.reset()
                announce(sample, accepted_count, total_count)
                status = "Next target ready"
                continue
            if key in (ord("s"), ord("S")):
                if sample.target_present and selector.roi is None:
                    selector.feedback = ""
                    status = "Drag mouse over visible target first"
                    continue
                manifest, cap = capture_one(cap, sample, selector.roi, args)
                if manifest["accepted"]:
                    accepted_count += 1
                    print(f"Saved {sample.id}: {manifest['path']}", flush=True)
                    sample = next_pending(plan, args.output) if plan else Sample(
                        next_free_sample_id(args.output), args.kind, args.distance_mm,
                        args.pose_label, "", "", args.notes,
                        args.target_present or args.kind == "positive")
                    if sample is None or (plan is None and accepted_count >= args.target_count):
                        print(plan_summary(plan, args.output) if plan else
                              "Capture target reached.", flush=True)
                        break
                    selector.reset()
                    announce(sample, accepted_count, total_count)
                    status = "Saved; move to next pose"
                else:
                    selector.reset()
                    status = "QC failed; adjust, drag box again, retry"
                    print(f"Rejected {sample.id}: {'; '.join(manifest['rejection_reasons'])}",
                          flush=True)
                if args.once:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path,
                        help="CSV plan with id, kind, distance_mm, pose, position, background, requirement, target_present")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    parser.add_argument("--camera", type=int,
                        help="Manual camera index; overrides --camera-name")
    parser.add_argument("--camera-name", default="USB Camera",
                        help="DirectShow camera name used by default")
    parser.add_argument("--kind", choices=("positive", "negative"), default="positive")
    parser.add_argument("--target-present", action="store_true",
                        help="For manual negative samples where the target is visible")
    parser.add_argument("--distance-mm", type=float)
    parser.add_argument("--pose-label", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--target-count", type=int, default=1)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--depth-warmup", type=int, default=2)
    parser.add_argument("--depth-timeout", type=float, default=20.0)
    parser.add_argument("--rgb-reopen-attempts", type=int, default=5)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.extrinsics = args.extrinsics.resolve()
    if args.plan:
        args.plan = args.plan.resolve()
    elif args.kind == "positive" and (args.distance_mm is None or args.distance_mm <= 0):
        parser.error("--distance-mm is required for positive captures without --plan")
    if args.roi:
        args.roi = tuple(args.roi)
    if args.target_count < 1 or args.depth_warmup < 0 or args.depth_timeout <= 0 or args.rgb_reopen_attempts < 1:
        parser.error("Capture counts and timeouts must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
