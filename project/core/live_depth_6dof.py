"""Initialize the energy unit from verified RGB/CAD references, then track it.

RGB initialization works during camera motion. Unmatched stable views can use
the legacy Astra depth fallback, which releases UVC RGB for a five-frame depth
burst and then reopens it for pose tracking.
"""
from __future__ import annotations

import time

MODULE_STARTED = time.perf_counter()

import argparse
import faulthandler
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .cad_pose_overlay import DEFAULT_INTRINSICS, DEFAULT_MODEL, load_calibration
from .cad_video_tracker import (load_box_display_y_offset, load_model_box,
                               detection_matches_pose, make_edge_points,
                               project, render_box_layer, resolve_camera_index,
                               restore_rotated_box, roll_neutral_box_points,
                               parse_args as parse_tracker_args,
                               run as run_rgb_tracker)
from .calibrate_camera import open_camera, write_image
from .camera_identity import PinnedCamera
from .capture_global_dataset import IMAGE_SIZE, read_rgb, reopen_rgb, validate_depth_archive
from .depth_global_initializer import (RgbConstraint, Settings, load_model_cloud,
                                       run as initialize_depth)
from .depth_pose_handoff import DEFAULT_EXTRINSICS, compose_pose
from .orbbec_rgbd import materialize_depth_burst
from .rgb_pose_recovery import axis_matches_pose, observe_long_axis

from .collection_recording import wrap_camera, collection_key, collection_display
from .paths import DATA_ROOT, PROJECT_ROOT

TRACKER = "core.cad_video_tracker"
WINDOW = "Astra depth 6DoF"
DETECTOR_MODEL = DATA_ROOT / "runs" / "nldy_train-3" / "weights" / "best.pt"
NATIVE_RGBD_CAPTURE = PROJECT_ROOT.parent / "build" / "orbbec_rgbd_capture"
REACQUIRE_EXIT_CODE = 2
DEFAULT_LIVE_DETECTOR_INTERVAL = 2
DEFAULT_LIVE_CAMERA_FPS = 30.0
DEFAULT_LIVE_CAMERA_FOURCC = 'MJPG'
DEFAULT_RGB_FALLBACK_TIMEOUT_S = 1.8
MODULE_IMPORTED = time.perf_counter()
STARTUP_TIMING_LOCK = threading.Lock()


class PreviewCancelled(Exception):
    """The user requested a clean stop while a background stage was running."""


class DepthWorkerUnreapedError(Exception):
    """Do not reopen RGB while a native worker cannot be confirmed stopped."""


def record_event(output_dir: Path, stage: str, error: Exception | None = None,
                 **details: object) -> None:
    """Keep recoverable errors even when the application has no terminal."""
    payload = {"at": datetime.now().isoformat(timespec="milliseconds"),
               "stage": stage, **details}
    if error is not None:
        payload["error"] = str(error)
        payload["traceback"] = "".join(traceback.format_exception(error))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "session_events.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"Cannot save session diagnostics: {exc}", file=sys.stderr, flush=True)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read RGB image: {path}")
    return image


def sample_extrinsics(sample: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    manifest_path = sample / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Offline sample has no calibration manifest; pass --extrinsics")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = manifest.get("rgb_depth_calibration") or {}
    name = calibration.get("file")
    expected_hash = calibration.get("sha256")
    if not isinstance(name, str) or Path(name).name != name or not expected_hash:
        raise RuntimeError("Offline sample has no valid RGB-depth calibration snapshot")
    snapshot = sample / name
    if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError("Offline sample RGB-depth calibration snapshot is missing or changed")
    return snapshot


def validate_capture(rgb: np.ndarray, depth_dir: Path, extrinsics_path: Path,
                     intrinsics_path: Path) -> np.ndarray:
    depth_path = depth_dir / "depth_burst.npz"
    metadata_path = depth_dir / "metadata.json"
    validate_depth_archive(depth_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mode = metadata.get("video_mode", {})
    if (mode.get("width"), mode.get("height")) != IMAGE_SIZE:
        raise RuntimeError("Depth stream size differs from calibrated 640x480")
    if mode.get("depth_unit_mm") != 1:
        raise RuntimeError("Depth data must use 1 mm units")
    if metadata.get("captured_frames") != 5:
        raise RuntimeError("Depth capture must contain five frames")
    if metadata.get("depth_mirroring", {}).get("active") is not False:
        raise RuntimeError("Depth mirroring must be disabled")
    if metadata.get("registration", {}).get("requested_depth_to_color") is not False:
        raise RuntimeError("Depth-to-color registration must be disabled")
    estimate = metadata.get("intrinsics_estimate", {})
    try:
        capture_K_depth = np.array([
            [estimate["fx"], 0.0, estimate["cx"]],
            [0.0, estimate["fy"], estimate["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    except KeyError as exc:
        raise RuntimeError(f"Depth capture lacks intrinsic parameter {exc}") from exc
    with np.load(extrinsics_path) as calibration:
        saved_size = tuple(int(v) for v in calibration["image_size"])
        saved_K_depth = calibration["K_depth"].copy()
        saved_K_rgb = calibration["K_rgb"].copy()
        saved_dist_rgb = calibration["dist_rgb"].copy()
        T_rgb_depth = calibration["T_rgb_depth"].copy()
    K_tracker, dist_tracker, tracker_size = load_calibration(intrinsics_path)
    if (saved_size != IMAGE_SIZE or tracker_size != IMAGE_SIZE or
            rgb.shape[1::-1] != IMAGE_SIZE):
        raise RuntimeError("RGB image or calibration size differs from 640x480")
    if not np.allclose(capture_K_depth, saved_K_depth, rtol=0, atol=1e-3):
        raise RuntimeError("Depth intrinsics differ from the RGB-depth calibration")
    if (not np.allclose(K_tracker, saved_K_rgb, rtol=0, atol=1e-6) or
            not np.allclose(dist_tracker, saved_dist_rgb, rtol=0, atol=1e-6)):
        raise RuntimeError("Tracker RGB intrinsics differ from the RGB-depth calibration")
    if (T_rgb_depth.shape != (4, 4) or not np.isfinite(T_rgb_depth).all() or
            not np.allclose(T_rgb_depth[3], [0, 0, 0, 1])):
        raise RuntimeError("Invalid RGB-depth extrinsic transform")
    return T_rgb_depth


def validate_rgb_stillness(before: np.ndarray, after: np.ndarray,
                           box_corners: np.ndarray, K: np.ndarray,
                           dist: np.ndarray, rvec: np.ndarray,
                           tvec: np.ndarray) -> dict[str, object]:
    if before.shape != after.shape:
        raise RuntimeError("RGB frame size changed during depth capture")
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.float32)
    global_shift, global_response = cv2.phaseCorrelate(gray_before, gray_after)
    camera_points = roll_neutral_box_points(box_corners, rvec, tvec)
    projected = project(camera_points, np.zeros(3), np.zeros(3), K, dist)
    if not np.isfinite(projected).all():
        raise RuntimeError("Initial box projection is invalid")
    x1 = max(0, int(np.floor(projected[:, 0].min() - 8)))
    y1 = max(0, int(np.floor(projected[:, 1].min() - 8)))
    x2 = min(before.shape[1], int(np.ceil(projected[:, 0].max() + 8)))
    y2 = min(before.shape[0], int(np.ceil(projected[:, 1].max() + 8)))
    if x2 - x1 < 32 or y2 - y1 < 32:
        raise RuntimeError("Initial box is too far outside the RGB image")
    target_shift, target_response = cv2.phaseCorrelate(
        gray_before[y1:y2, x1:x2], gray_after[y1:y2, x1:x2])
    feature_check = None
    if (global_response >= .35 and np.linalg.norm(global_shift) <= 3.0
            and target_response < .5 and np.linalg.norm(target_shift) <= 2.0):
        # Restarting the RGB stream changes exposure slightly. If phase
        # correlation loses confidence, require distributed forward/backward
        # feature tracks to confirm that the target stayed still.
        first = gray_before[y1:y2, x1:x2].astype(np.uint8)
        second = gray_after[y1:y2, x1:x2].astype(np.uint8)
        points = cv2.goodFeaturesToTrack(first, 100, .01, 5)
        if points is not None and len(points) >= 12:
            forward, valid, _ = cv2.calcOpticalFlowPyrLK(first, second, points, None)
            if forward is not None and valid is not None:
                backward, back_valid, _ = cv2.calcOpticalFlowPyrLK(second, first, forward, None)
                if backward is not None and back_valid is not None:
                    good = (valid.ravel().astype(bool) & back_valid.ravel().astype(bool)
                            & (np.linalg.norm(backward - points, axis=2).ravel() <= 1.0))
                    if np.count_nonzero(good) >= 12 and np.mean(good) >= .6:
                        motion = np.linalg.norm(forward[good] - points[good], axis=2).ravel()
                        spread = cv2.contourArea(cv2.convexHull(points[good]))
                        feature_check = {'median_px': float(np.median(motion)),
                                         'p90_px': float(np.percentile(motion, 90)),
                                         'spread_px2': float(spread)}
    features_still = (feature_check is not None and feature_check['median_px'] <= 1.0
                      and feature_check['p90_px'] <= 2.0
                      and feature_check['spread_px2'] >= (x2 - x1) * (y2 - y1) * .15)
    if (not np.isfinite([*global_shift, global_response,
                         *target_shift, target_response]).all() or
            np.linalg.norm(global_shift) > 3.0 or global_response < 0.35 or
            np.linalg.norm(target_shift) > 2.0 or (target_response < 0.5 and not features_still)):
        raise RuntimeError(
            "RGB scene changed during depth capture; keep camera and target still, then retry "
            f"(global {np.linalg.norm(global_shift):.1f}px, "
            f"target {np.linalg.norm(target_shift):.1f}px)")
    return {
        "global_shift_px": list(global_shift),
        "global_response": float(global_response),
        "target_shift_px": list(target_shift),
        "target_response": float(target_response),
        "feature_confirmation": feature_check,
    }


def initialize_capture(rgb_path: Path, depth_dir: Path, output_dir: Path,
                       model_path: Path, extrinsics_path: Path,
                       intrinsics_path: Path, settings: Settings,
                       display_y_offset_px: float,
                       rgb_after_path: Path | None = None,
                       diagnostics: dict | None = None) -> Path:
    rgb = read_image(rgb_path)
    T_rgb_depth = validate_capture(rgb, depth_dir, extrinsics_path, intrinsics_path)
    fit_dir = output_dir / "depth_fit"
    if diagnostics is None:
        diagnostics = initialize_depth(depth_dir / "depth_burst.npz",
                                       depth_dir / "metadata.json", model_path,
                                       fit_dir, settings)
    quality = diagnostics.get("pose_quality", {})
    if not quality.get("accepted", False):
        pose = diagnostics.get("pose") or {}
        raise RuntimeError("Depth pose rejected: fitness "
                           f"{pose.get('fitness', 'n/a')}, "
                           f"RMSE {pose.get('inlier_rmse_mm', 'n/a')} mm")
    T_depth_model = np.asarray(diagnostics["pose"]["model_to_camera"], np.float64)
    T_rgb_model = compose_pose(T_rgb_depth, T_depth_model)
    if (not np.isfinite(T_rgb_model).all() or T_rgb_model[2, 3] <= 0 or
            not np.allclose(T_rgb_model[3], [0, 0, 0, 1])):
        raise RuntimeError("Depth pose cannot be transformed into RGB camera coordinates")
    rvec, _ = cv2.Rodrigues(T_rgb_model[:3, :3])
    tvec = T_rgb_model[:3, 3]
    K, dist, _ = load_calibration(intrinsics_path)
    box_corners = load_model_box(model_path)
    if quality.get('requires_rgb_verification', False):
        constraint = diagnostics.get('rgb_constraint') or {}
        box = constraint.get('box')
        observed_axis = observe_long_axis(rgb, box)
        vertices, _, _ = make_edge_points(model_path, 700, 3)
        # The end cap has no reliable image long-axis direction.  A broad
        # metric cap fit plus an independent YOLO extent check can validate
        # that view without requiring three visible parallel side rails.
        cap_verified = quality.get('end_face_geometry', {}).get('accepted', False)
        if (box is None or (observed_axis is None and not cap_verified)
                or not detection_matches_pose(
                box, vertices, rvec.reshape(3), tvec, K, dist) or not axis_matches_pose(
                    observed_axis, rvec.reshape(3), tvec, K, dist)):
            raise RuntimeError('Depth pose rejected: partial geometry conflicts with RGB target')
    stillness = (validate_rgb_stillness(rgb, read_image(rgb_after_path),
                                       box_corners, K, dist, rvec.reshape(3), tvec)
                 if rgb_after_path is not None else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_path = output_dir / "rgb_initial_pose.json"
    payload = {
        "coordinate_frame": "rgb_camera_optical",
        "units": "mm",
        "axis_convention": "X right, Y down, Z forward",
        "rvec_rgb_model_rad": rvec.reshape(3).tolist(),
        "tvec_rgb_model_mm": tvec.tolist(),
        "T_rgb_model": T_rgb_model.tolist(),
        "depth_pose_quality": quality,
        "rgb_stillness": stillness,
        "rgb_image": str((rgb_after_path or rgb_path).resolve()),
        "depth_burst": str((depth_dir / "depth_burst.npz").resolve()),
        "extrinsics": str(extrinsics_path.resolve()),
        "intrinsics": str(intrinsics_path.resolve()),
        "roll_note": "Long-axis roll is ambiguous modulo approximately 120 degrees.",
    }
    pose_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    layer = render_box_layer(rgb.shape, box_corners, K, dist,
                             rvec.reshape(3), tvec, display_y_offset_px)
    overlay = cv2.addWeighted(rgb, 1.0, layer, 0.72, 0)
    cv2.putText(overlay, f"RGB XYZ mm: {tvec[0]:+.1f}, {tvec[1]:+.1f}, {tvec[2]:.1f}",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (40, 220, 255), 2, cv2.LINE_AA)
    write_image(output_dir / "rgb_initial_box.png", overlay)
    print(f"Initial RGB camera XYZ (mm): X={tvec[0]:+.1f}, "
          f"Y={tvec[1]:+.1f}, Z={tvec[2]:.1f}", flush=True)
    return pose_path


def fit_depth_capture(depth_dir: Path, output_dir: Path,
                     model: Path, settings: Settings,
                     rgb_constraint: RgbConstraint | None = None) -> dict:
    return initialize_depth(depth_dir / 'depth_burst.npz', depth_dir / 'metadata.json',
                            model, output_dir / 'depth_fit', settings, rgb_constraint)


def timed_stage(timings: dict, name: str, task):
    started = time.perf_counter()
    try:
        return task()
    finally:
        timings[name + '_s'] = round(time.perf_counter() - started, 4)


def prepare_live_detector(model_path: Path, timings: dict):
    from .detector_process import DetectorProcess

    return DetectorProcess(model_path, timings)


def prepare_live_session(args: argparse.Namespace, camera_index: int, timings: dict):
    """Open RGB while optional YOLO loads on its own persistent worker."""
    from .background_detector import BackgroundDetector

    detector = None
    started = time.perf_counter()
    model_path = getattr(args, 'reacquire_model', None)
    if model_path is not None:
        def load_detector():
            result = prepare_live_detector(model_path, timings)
            timings['detector_ready_s'] = round(time.perf_counter() - started, 4)
            return result
        detector = BackgroundDetector(load_detector, detect_target_box)
    cap = None
    frame = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    identity = getattr(args, 'camera_identity', None)
    try:
        task = lambda: (identity.open(
            open_camera, *IMAGE_SIZE,
            getattr(args, 'camera_fourcc', DEFAULT_LIVE_CAMERA_FOURCC),
            getattr(args, 'camera_fps', DEFAULT_LIVE_CAMERA_FPS))
            if identity is not None else open_camera(
                camera_index, *IMAGE_SIZE,
                getattr(args, 'camera_fourcc', DEFAULT_LIVE_CAMERA_FOURCC),
                getattr(args, 'camera_fps', DEFAULT_LIVE_CAMERA_FPS)), None)
        try:
            cap, _ = run_with_preview(task, frame, 'Opening RGB camera')
        except (OSError, RuntimeError, cv2.error) as exc:
            record_event(args.output_dir, 'initial_rgb_open_failed', exc,
                         camera_index=camera_index)
            cap, frame = recover_rgb(args, camera_index, frame)
        timings['rgb_open_s'] = round(time.perf_counter() - started, 4)
        return wrap_camera(cap, args), detector
    except BaseException:
        if cap is not None:
            cap.release()
        if detector is not None:
            detector.close()
        raise


def prepare_live_geometry(model_path: Path, timings: dict) -> None:
    """Build each reusable CAD cache while RGB searching is already active."""
    timed_stage(timings, 'cad_cloud', lambda: load_model_cloud(model_path))
    timed_stage(timings, 'cad_box', lambda: load_model_box(model_path))
    timed_stage(timings, 'cad_edges', lambda: make_edge_points(model_path, 700, 3))
    from .surface_seeding import load_surface_scene

    timed_stage(timings, 'cad_surface', lambda: load_surface_scene(model_path))


def prepare_live_region(args, timings):
    from .m3t_live import NativeRegionTracker
    K, dist, size = load_calibration(args.intrinsics)
    return timed_stage(timings, 'm3t_prepare', lambda: NativeRegionTracker(
        args.model, K, dist, size, args.output_dir / 'm3t_stream'))


def prepare_rgb_initializer(args, timings):
    from .rgb_global_initializer import prepare_initializer

    started = time.perf_counter()
    initializer = prepare_initializer(args.model, args.intrinsics)
    timings['rgb_reference_views'] = len(initializer.views)
    timings['rgb_references_s'] = round(time.perf_counter() - started, 4)
    return initializer


def save_rgb_initial_pose(output_dir, pose, frame, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / 'rgb_current.png'
    write_image(image_path, frame)
    rotation, _ = cv2.Rodrigues(pose.rvec)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, pose.tvec
    path = output_dir / 'rgb_initial_pose.json'
    path.write_text(json.dumps({
        'coordinate_frame': 'rgb_camera_optical', 'units': 'mm',
        'rvec_rgb_model_rad': pose.rvec.tolist(), 'tvec_rgb_model_mm': pose.tvec.tolist(),
        'T_rgb_model': transform.tolist(), 'rgb_image': str(image_path.resolve()),
        'intrinsics': str(args.intrinsics.resolve()), 'pose_source': 'rgb_reference_pnp',
        'rgb_pose_quality': {'accepted': True, 'inliers': len(pose.points),
                             'inlier_ratio': pose.inlier_ratio, 'rms_px': pose.rms_px,
                             'reference': pose.reference, 'elapsed_ms': pose.elapsed_ms},
        'tracking_seed': {'model_points_mm': pose.points.tolist(),
                          'image_points_px': pose.pixels.tolist()},
        'roll_note': 'Equivalent roll and long-axis endpoint direction remain ambiguous.',
    }, indent=2) + '\n', encoding='utf-8')
    return path


def write_startup_timing(args: argparse.Namespace, timings: dict,
                         started: float, status: str) -> None:
    payload = {**timings, 'status': status,
               'elapsed_s': round(time.perf_counter() - started, 4)}
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with STARTUP_TIMING_LOCK:
            (args.output_dir / 'startup_timing.json').write_text(
                json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    except OSError as exc:
        print(f"Cannot save startup timing: {exc}", file=sys.stderr, flush=True)


def write_initialization_timing(output_dir: Path, timings: dict,
                                started: float, status: str) -> None:
    timings['status'] = status
    timings['total_to_pose_s'] = round(time.perf_counter() - started, 4)
    (output_dir / 'initialization_timing.json').write_text(
        json.dumps(timings, indent=2) + '\n', encoding='utf-8')


def run_depth_worker(depth_dir: Path, warmup: int, timeout_s: float,
                     frame_timeout_s: float = 2.0,
                     cancel_event: threading.Event | None = None,
                     calibration_path: Path | None = None) -> None:
    """Run the Linux Orbbec RGB-D capture behind a hard process boundary.

    ``subprocess.run(timeout=...)`` is convenient but hides the worker log and
    makes it difficult to tell a driver crash from a Python exception. Keep a
    durable log and explicitly reap the child after a timeout so the next RGB
    recovery attempt never competes with a still-running native camera process.
    The native backend saves five paired color/depth frames; the adapter then
    materializes the legacy depth archive consumed by the existing fitter.
    """
    depth_dir.mkdir(parents=True, exist_ok=True)
    native_dir = depth_dir / "native_rgbd"
    command = [str(NATIVE_RGBD_CAPTURE), "--output", str(native_dir),
               "--seconds", "3", "--pairs", "5"]
    log_path = depth_dir / "worker.log"
    started = time.perf_counter()
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(json.dumps({"command": command, "timeout_s": timeout_s,
                              "started_at": datetime.now().isoformat()},
                             ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, text=True,
                                   creationflags=creation_flags)
        try:
            deadline = time.monotonic() + timeout_s
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise PreviewCancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_s)
                try:
                    return_code = process.wait(timeout=min(.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired as exc:
            log.write(f"WATCHDOG timeout after {time.perf_counter() - started:.3f}s\n")
            log.flush()
            progress = depth_dir / "capture_progress.json"
            detail = (progress.read_text(encoding="utf-8")
                      if progress.is_file() else "no native progress marker")
            raise RuntimeError(
                f"Depth capture watchdog timeout after {timeout_s:.1f}s ({detail[-500:]})") from exc
        finally:
            if process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as exc:
                    raise DepthWorkerUnreapedError(
                        'Depth worker did not exit after watchdog kill; RGB remains closed') from exc
    if return_code != 0:
        progress = depth_dir / "capture_progress.json"
        detail = (progress.read_text(encoding="utf-8")
                  if progress.is_file() else "no native progress marker")
        raise RuntimeError(
            f"Depth capture failed (worker exit {return_code}); {detail[-700:]}; "
            f"see {log_path}")
    try:
        materialize_depth_burst(native_dir, depth_dir, warmup=warmup,
                                calibration_path=calibration_path)
    except Exception as exc:
        raise RuntimeError(
            f"Native RGB-D capture completed but its depth archive could not be materialized: {exc}; "
            f"see {log_path}") from exc


def tracker_command(args: argparse.Namespace, pose_path: Path, output_dir: Path,
                    camera_index: int | None) -> list[str]:
    detector_interval = max(int(getattr(args, 'detector_interval',
                                        DEFAULT_LIVE_DETECTOR_INTERVAL)), 1)
    camera_fourcc = getattr(args, 'camera_fourcc', None)
    if camera_fourcc is None:
        camera_fourcc = os.environ.get('CAMERA_FOURCC', DEFAULT_LIVE_CAMERA_FOURCC)
    command = [sys.executable, "-m", TRACKER,
               "--pose", str(pose_path), "--model", str(args.model),
               "--intrinsics", str(args.intrinsics), "--no-static-correction",
               "--rgb-pose-recovery",
               "--fast-pose-recovery",
               "--surface-seeding",
               "--detector-interval", str(detector_interval),
               "--three-d-only-after-init",
               # Hand-held motion needs a little more per-frame headroom than
               # the offline validation defaults. The metric and detector
               # gates still reject implausible geometry and stale identities.
               "--fb-error", "2.5",
               "--max-rotation-step", "22",
               "--max-translation-step", "35",
               "--max-depth", "2400",
               "--output", str(output_dir / "last_pose.json"),
               "--log", str(output_dir / "frames.csv"),
               "--summary", str(output_dir / "summary.json"),
               "--xyz-jsonl", str(output_dir / "center_xyz.jsonl"),
               "--box-display-y-offset-px", str(args.box_display_y_offset_px),
               "--box-roll-mode", getattr(args, 'box_roll_mode', 'true')]
    command.append('--depth-fallback-on-loss' if getattr(
        args, 'depth_fallback_on_loss', True) else '--no-depth-fallback-on-loss')
    if getattr(args, 'region_tracking', False):
        command.append('--region-tracking')
    command.append('--cad-outline' if getattr(args, 'cad_outline', True) else '--no-cad-outline')
    command.append('--show-box' if getattr(args, 'show_box', False) else '--no-show-box')
    command.append('--pose-axes' if getattr(args, 'pose_axes', True) else '--no-pose-axes')
    if getattr(args, 'record_diagnostics', True):
        command.extend(["--record-diagnostics", "--diagnostic-duration-s",
                        str(getattr(args, 'diagnostic_duration_s', 60.))])
    if args.video is not None:
        command.extend(["--video", str(args.video)])
    else:
        assert camera_index is not None
        command.extend(["--camera", str(camera_index), "--reacquire-on-loss"])
        if camera_fourcc:
            command.extend(["--camera-fourcc", camera_fourcc])
        if args.reacquire_model is not None:
            command.extend(["--reacquire-model", str(args.reacquire_model)])
    if args.max_frames:
        command.extend(["--max-frames", str(args.max_frames)])
    return command


def run_tracker(args: argparse.Namespace, pose_path: Path,
                output_dir: Path, camera_index: int | None,
                capture: cv2.VideoCapture | None = None,
                detector=None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = tracker_command(args, pose_path, output_dir, camera_index)
    print(f"Tracking output: {output_dir.resolve()}", flush=True)
    if capture is not None:
        tracker_args = parse_tracker_args(command[3:])
        # Keep the in-process RGB reference library available for fast pose
        # relocalization when the target changes orientation.
        tracker_args.rgb_initializer = getattr(args, 'rgb_initializer', None)
        tracker_args.rgb_initializer_provider = getattr(args, 'rgb_initializer_provider', None)
        tracker_args.region_tracker_provider = getattr(args, 'region_tracker_provider', None)
        identity = getattr(args, 'camera_identity', None)
        if identity is not None:
            tracker_args.camera_identity_name = identity.name
            tracker_args.camera_device_path = identity.device_path
        return run_rgb_tracker(tracker_args, capture=capture, detector=detector,
                               window_name=WINDOW, keep_window=True,
                               keep_capture_on_reacquire=True)
    return subprocess.run(command, cwd=PROJECT_ROOT).returncode


def run_with_preview(task, frame: np.ndarray, message: str,
                     cap: cv2.VideoCapture | None = None,
                     cancel_event: threading.Event | None = None,
                     preview_renderer=None):
    cancelled = False
    preview_error = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task)
        while not future.done():
            if cap is not None and not cancelled and preview_error is None:
                try:
                    frame = read_rgb(cap, 1)
                except (OSError, RuntimeError, cv2.error) as exc:
                    preview_error = exc
            renderer = preview_renderer or preview_frame
            cv2.imshow(WINDOW, renderer(frame, "Stopping..." if cancelled else message))
            key = collection_key(cap, cv2.waitKey(20) & 0xFF)
            if key in (27, ord("q"), ord("Q")) or \
                    cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                cancelled = True
                if cancel_event is not None:
                    cancel_event.set()
        # Do not abandon a thread that owns a native camera. A depth worker
        # can be killed safely; other stages must finish before releasing it.
        try:
            result = future.result()
        except Exception:
            if cancelled:
                raise PreviewCancelled() from None
            raise
        if cancelled or preview_error is not None:
            if isinstance(result, tuple) and result and hasattr(result[0], "release"):
                result[0].release()
            if cancelled:
                raise PreviewCancelled()
            raise preview_error
        return result


def recover_rgb(args: argparse.Namespace, camera_index: int,
                frame: np.ndarray, timings: dict | None = None,
                camera_name: str | None = None):
    """Retry DirectShow reopen errors in the existing window until quit."""
    retry = 0
    started = time.perf_counter()
    while True:
        try:
            task = lambda: reopen_rgb(camera_index, args.rgb_reopen_attempts,
                                      getattr(args, 'rgb_warmup_frames', 2),
                                      camera_name if getattr(args, 'camera_identity', None) is None else None,
                                      getattr(args, 'camera_identity', None),
                                      getattr(args, 'camera_fourcc', DEFAULT_LIVE_CAMERA_FOURCC),
                                      getattr(args, 'camera_fps', DEFAULT_LIVE_CAMERA_FPS))
            preview_options = ({'preview_renderer': args.reference_preview_renderer}
                               if getattr(args, 'reference_preview_renderer', None) else {})
            cap, after = run_with_preview(
                task, frame,
                'Restoring RGB' if retry == 0 else 'Camera recovering | Q: quit',
                **preview_options)
            return wrap_camera(cap, args), after
        except (OSError, RuntimeError, cv2.error) as exc:
            retry += 1
            record_event(args.output_dir, 'rgb_reopen_failed', exc, retry=retry)
            print(f"RGB recovery attempt {retry} failed: {exc}; retrying.", flush=True)
            # Back off in the UI thread, keeping quit/window-close responsive.
            retry_at = time.monotonic() + 1.0
            while time.monotonic() < retry_at:
                renderer = getattr(args, 'reference_preview_renderer', None) or preview_frame
                cv2.imshow(WINDOW, renderer(frame, 'Camera recovering | Q: quit'))
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord('q'), ord('Q')) or \
                        cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    raise PreviewCancelled()
        finally:
            if timings is not None:
                timings['rgb_reopen_s'] = round(time.perf_counter() - started, 4)
                timings['rgb_reopen_retries'] = retry


def preview_frame(frame: np.ndarray, message: str) -> np.ndarray:
    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (25, 30, 33), -1)
    cv2.putText(canvas, message, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (40, 220, 255), 2, cv2.LINE_AA)
    return canvas


def plausible_target_box(box: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64).reshape(4)
    height, width = image_shape[:2]
    box_width, box_height = x2 - x1, y2 - y1
    touches_edge = x1 <= 3 or y1 <= 3 or x2 >= width - 3 or y2 >= height - 3
    aspect_min, aspect_max = ((.18, 5.5) if touches_edge else (.22, 4.5))
    return bool(np.isfinite([x1, y1, x2, y2]).all()
                and min(box_width, box_height) >= 20
                and max(box_width, box_height) <= min(width, height) * .85
                and box_width * box_height <= width * height * .5
                and aspect_min <= box_width / box_height <= aspect_max
                and x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height)


def detect_target_box(detector, frame: np.ndarray,
                      fallback_rotation: int = 2,
                      rotation_hint: dict | None = None) -> np.ndarray | None:
    if hasattr(detector, 'get_box'):
        return detector.get_box(frame, fallback_rotation=fallback_rotation)
    for turn in dict.fromkeys((0, fallback_rotation)):
        source = np.ascontiguousarray(np.rot90(frame, turn)) if turn else frame
        result = detector.predict(source, conf=.15, imgsz=640, verbose=False)[0]
        candidates = sorted(range(len(result.boxes.conf)),
                            key=lambda index: float(result.boxes.conf[index]),
                            reverse=True)
        for index in candidates:
            box = result.boxes.xyxy[index].cpu().numpy()
            box = restore_rotated_box(box, frame.shape, turn)
            if plausible_target_box(box, frame.shape):
                if rotation_hint is not None:
                    rotation_hint['turn'] = turn
                return box
    return None


def estimate_startup_pose(search, frame, box):
    """YOLO may narrow feature extraction but never supplies a pose."""
    height, width = frame.shape[:2]
    region = (np.array([0., 0., float(width), float(height)], np.float32)
              if box is None else box)
    return search.estimate(frame, region, validation_box=box)


def wait_for_3d_target(cap, detector, target_hint):
    """Search CAD/reference 6DoF immediately; YOLO is independent assistance."""
    from .background_detector import BackgroundDetector
    from .rgb_global_initializer import advance_pose

    owns_detector = detector is not None and not hasattr(detector, 'get_box')
    assist = (BackgroundDetector(lambda: detector, detect_target_box)
              if owns_detector else detector)
    # Keep independent full-frame CAD search and YOLO ROI search in flight at
    # the same time.  A serial alternation made startup wait for a failed
    # whole-image search before trying the view that YOLO had already found.
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='initial-3d')
    pending = {}
    searches = {}
    initializer_source = None
    stats = target_hint.setdefault('search_stats', {})
    for name in ('global_checks', 'yolo_roi_checks', 'completed', 'candidates',
                 'expired', 'handoff_rejected', 'depth_fallbacks'):
        stats.setdefault(name, 0)
    search_started = None
    stable_since = None
    anchor = previous_gray = previous_box = None
    try:
        while True:
            frame = read_rgb(cap, 1)
            box = assist.get_box(frame) if assist is not None else None
            initializer = target_hint['initializer_provider']()
            if initializer is not None and initializer is not initializer_source:
                initializer_source = initializer
                searches = {name: initializer.new_search() for name in ('global', 'roi')}
                search_started = time.monotonic()
            # Consume whichever independent search finishes first.  Advance
            # every result to the newest camera frame before accepting it so a
            # moving object never enters tracking with a stale startup pose.
            for name, job in list(pending.items()):
                future, pending_search, source_frame, source_box, submitted_at = job
                if not future.done():
                    continue
                del pending[name]
                stats['completed'] += 1
                try:
                    pose = future.result()
                except (RuntimeError, ValueError, cv2.error) as exc:
                    print(f'3D search retry ({name}): {exc}', flush=True)
                    pose = None
                    pending_search.clear()
                stats['candidates'] += int(pose is not None)
                if pose is not None and time.monotonic() - submitted_at <= .5:
                    pose = advance_pose(pose, source_frame, frame, pending_search.K,
                                        pending_search.dist, pending_search.vertices, source_box)
                    if pose is not None:
                        target_hint.update(rgb_pose=pose, rgb_frame=frame.copy(),
                                           box=None if box is None else box.copy())
                        return True
                    stats['handoff_rejected'] += 1
                elif pose is not None:
                    stats['expired'] += 1
                    pending_search.clear()

            # Keep one global search active continuously.  When YOLO has a
            # current ROI, run a second independent search there as well; the
            # ROI search gets its own temporal bank and cannot corrupt global
            # search state.
            if searches:
                if 'global' not in pending:
                    pending_search = searches['global']
                    source_frame = frame.copy()
                    source_box = None
                    pending['global'] = (executor.submit(estimate_startup_pose,
                                                          pending_search, source_frame,
                                                          source_box),
                                         pending_search, source_frame, source_box,
                                         time.monotonic())
                    target_hint['search_stats']['global_checks'] += 1
                if box is not None and 'roi' not in pending:
                    pending_search = searches['roi']
                    source_frame = frame.copy()
                    source_box = box.copy()
                    pending['roi'] = (executor.submit(estimate_startup_pose,
                                                       pending_search, source_frame,
                                                       source_box),
                                      pending_search, source_frame, source_box,
                                      time.monotonic())
                    target_hint['search_stats']['yolo_roi_checks'] += 1
            gray = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                              None, fx=.5, fy=.5), (3, 3), 0)
            stable = (box is not None and previous_box is not None and previous_gray is not None
                      and target_motion_fraction(previous_gray, gray, box, frame.shape) <= .06
                      and np.linalg.norm(box - previous_box) <= 12.)
            if stable and anchor is not None:
                stable = target_motion_fraction(anchor, gray, box, frame.shape) <= .06
            if stable:
                if stable_since is None:
                    stable_since, anchor = time.monotonic(), gray.copy()
            else:
                stable_since = anchor = None
            previous_gray, previous_box = gray, box
            loading = target_hint.get('references_loading', lambda: False)()
            failed = target_hint.get('references_failed', lambda: False)()
            label = ('Preparing 3D references' if loading else '3D pose search')
            label += (' | YOLO assisting' if box is not None else ' | Q: quit')
            cv2.imshow(WINDOW, collection_display(cap, preview_frame(frame, label)))
            key = collection_key(cap, cv2.waitKey(1) & 0xFF)
            if key in (27, ord('q'), ord('Q')) or \
                    cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False
            now = time.monotonic()
            expired = (search_started is not None and now - search_started >=
                       target_hint.get('rgb_fallback_timeout_s', DEFAULT_RGB_FALLBACK_TIMEOUT_S))
            fully_visible = (box is not None and box[0] > 3 and box[1] > 3 and
                             box[2] < frame.shape[1]-3 and box[3] < frame.shape[0]-3)
            depth_ready = (not loading and (failed or expired) and fully_visible and
                           stable_since is not None and now - stable_since >=
                           target_hint.get('settle_seconds', .35) and
                           now >= target_hint.get('depth_retry_after', 0.))
            # Depth remains a fallback for views that lack reference evidence;
            # neither a missed YOLO hit nor a loading model blocks global PnP.
            if (key == 32 and not loading) or depth_ready:
                stats['depth_fallbacks'] += 1
                if box is not None:
                    target_hint['box'] = box.copy()
                return True
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if owns_detector:
            assist.close()


def wait_for_target(cap: cv2.VideoCapture, detector, target_hint: dict | None = None,
                    message: str = 'Searching target') -> bool:
    if ((target_hint or {}).get('async_detection') and
            (target_hint or {}).get('initializer_provider') is not None):
        return wait_for_3d_target(cap, detector, target_hint)
    if target_hint is not None and target_hint.get('async_detection', False):
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='rgb-search') as executor:
            return _wait_for_target(cap, detector, target_hint, message, executor)
    return _wait_for_target(cap, detector, target_hint, message)


def search_rgb_target(detector, frame, rotation, rotation_hint, initializer):
    box = detect_target_box(detector, frame, rotation, rotation_hint)
    pose = initializer.estimate(frame, box) if initializer is not None else None
    return box, pose


def empty_depth_workspace(attempt_dir: Path) -> bool:
    """Distinguish an empty near-field depth capture from a failed pose fit."""
    try:
        diagnostics = json.loads((attempt_dir / 'depth_fit/diagnostics.json').read_text(
            encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return diagnostics.get('workspace_points') == 0


def failed_depth_retry_seconds(attempt_dir: Path, failures: int = 1) -> float:
    """Allow RGB relocalization time after a failed depth fit."""
    base = 10.0 if empty_depth_workspace(attempt_dir) else 5.0
    return min(base * 2 ** min(max(failures - 1, 0), 3), 30.0)


def target_motion_fraction(previous: np.ndarray, current: np.ndarray,
                           box: np.ndarray, image_shape: tuple[int, ...]) -> float:
    """Fraction of changed target pixels in blurred, half-size gray frames."""
    scale = np.array([current.shape[1] / image_shape[1],
                      current.shape[0] / image_shape[0]] * 2)
    x1, y1, x2, y2 = np.rint(np.asarray(box) * scale).astype(int)
    x1, x2 = np.clip([x1, x2], 0, current.shape[1])
    y1, y2 = np.clip([y1, y2], 0, current.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 1.0
    before = previous[y1:y2, x1:x2].astype(np.int16)
    after = current[y1:y2, x1:x2].astype(np.int16)
    difference = after - before
    # Ignore a uniform exposure adjustment, but retain silhouette/texture
    # changes from rotation even when YOLO's axis-aligned box is unchanged.
    difference -= int(np.median(difference))
    return float(np.mean(np.abs(difference) > 18))


def _wait_for_target(cap, detector, target_hint, message, executor=None) -> bool:
    manual_mode = detector is None
    # The asynchronous detector path cannot be used without YOLO. Keep the
    # camera preview interactive instead of attempting a depth burst from an
    # arbitrary frame.
    if manual_mode:
        executor = None
    consecutive = 0
    iteration = 0
    previous_box = None
    misses = 0
    rotation_hint = {}
    preferred_rotation = None
    stable_since = None
    settle_seconds = float((target_hint or {}).get('settle_seconds', 0.0))
    pending = None
    submitted_at = 0.0
    display_box = None
    previous_gray = None
    settle_reference = None
    pending_frame = None
    pending_initializer = None
    source_initializer = None
    search_initializer = None
    rgb_search_started_at = None
    while True:
        frame = read_rgb(cap, 1)
        completed = executor is None
        box = None
        rgb_pose = None
        provider = (target_hint or {}).get('initializer_provider')
        initializer = provider() if provider is not None else None
        if initializer is not None:
            if initializer is not source_initializer:
                source_initializer = initializer
                search_initializer = (initializer.new_search() if hasattr(initializer, 'new_search')
                                      else initializer)
            initializer = search_initializer
        if executor is None:
            if not manual_mode:
                box = detect_target_box(detector, frame,
                                        preferred_rotation or iteration % 3 + 1, rotation_hint)
                if initializer is not None:
                    rgb_pose = initializer.estimate(frame, box)
            else:
                box = ((np.asarray(target_hint['box'], dtype=np.float32).copy())
                       if target_hint is not None and target_hint.get('box') is not None
                       else None)
        else:
            if pending is not None and pending.done():
                if pending_initializer is not None:
                    box, rgb_pose = pending.result()
                else:
                    box = pending.result()
                completed = time.monotonic() - submitted_at <= .5
                if completed and box is not None:
                    preferred_rotation = rotation_hint.get('turn')
                if not completed:
                    rgb_pose = None
                    if pending_initializer is not None and hasattr(pending_initializer, 'clear'):
                        pending_initializer.clear()
                    # A slow inference must not preserve a confirmation from
                    # an old view of a target that may already have moved.
                    previous_box = display_box = None
                    consecutive = misses = 0
                    preferred_rotation = stable_since = settle_reference = None
                if rgb_pose is not None:
                    from .rgb_global_initializer import advance_pose

                    rgb_pose = advance_pose(rgb_pose, pending_frame, frame,
                                            pending_initializer.K, pending_initializer.dist,
                                            pending_initializer.vertices, box)
                pending = None
        if rgb_pose is not None:
            # These points are measured in the newest RGB frame. No stillness
            # gate or RGB/depth switch is needed to initialize while moving.
            if target_hint is not None:
                target_hint.update(box=box.copy(), rgb_pose=rgb_pose, rgb_frame=frame.copy())
            cv2.imshow(WINDOW, preview_frame(frame, '6DoF found | starting tracking'))
            if collection_key(cap, cv2.waitKey(1) & 0xFF) in (27, ord('q'), ord('Q')) or \
                    cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False
            return True
        if completed:
            iteration += 1
            display_box = box
        gray = None
        image_stable = True
        motion_box = box if box is not None else previous_box
        if settle_seconds > 0:
            gray = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                             None, fx=.5, fy=.5), (3, 3), 0)
            image_stable = bool(previous_gray is not None and motion_box is not None and
                                target_motion_fraction(previous_gray, gray, motion_box,
                                                       frame.shape) <= .06)
            if image_stable and settle_reference is not None:
                # The anchor also catches slow cumulative rotation that is
                # small between individual camera frames.
                image_stable = target_motion_fraction(
                    settle_reference, gray, motion_box, frame.shape) <= .06
            if not image_stable:
                stable_since = settle_reference = None
            previous_gray = gray
        if completed and box is not None:
            misses = 0
            if executor is None:
                preferred_rotation = rotation_hint.get('turn')
            # Confirm the same object. Alternating false positives or a target
            # being carried across the image must not start a stale depth burst.
            if previous_box is not None:
                size = np.maximum(box[2:] - box[:2], 1.0)
                old_size = np.maximum(previous_box[2:] - previous_box[:2], 1.0)
                shift = np.linalg.norm((box[:2] + box[2:] -
                                        previous_box[:2] - previous_box[2:]) * .5)
                stable = shift <= max(6., float(np.linalg.norm(size)) * .08) and \
                    np.max(np.abs(np.log(size / old_size))) <= .12
                consecutive = consecutive + 1 if stable else 1
            else:
                consecutive = 1
            previous_box = box
            if consecutive >= 2 and image_stable:
                if stable_since is None:
                    stable_since = time.monotonic()
                    settle_reference = gray
            else:
                stable_since = settle_reference = None
        elif completed:
            # Tolerate one missed inference without losing an otherwise
            # consistent entry. Two misses reset the confirmation.
            misses += 1
            if misses >= 2:
                previous_box = None
                consecutive = 0
                preferred_rotation = None
                stable_since = settle_reference = None
        label = ('No YOLO model | Space: initialize | R: select target'
                 if manual_mode and display_box is None else
                 'Manual target box | Space: initialize'
                 if manual_mode else
                 '2D target found | locating 6DoF'
                 if display_box is not None and provider is not None and stable_since is None else
                 '2D target found | waiting for motion to settle'
                 if display_box is not None and settle_seconds > 0 and stable_since is None else
                 '2D target found | confirming' if display_box is not None else message)
        display = preview_frame(frame, label)
        if display_box is not None:
            x1, y1, x2, y2 = np.rint(display_box).astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.imshow(WINDOW, collection_display(cap, display))
        key = collection_key(cap, cv2.waitKey(1) & 0xFF)
        if key in (27, ord("q"), ord("Q")):
            return False
        if manual_mode and key in (ord('r'), ord('R')):
            x, y, width, height = cv2.selectROI(
                WINDOW, frame, fromCenter=False, showCrosshair=True)
            if width > 0 and height > 0 and target_hint is not None:
                target_hint['box'] = np.array(
                    [x, y, x + width, y + height], dtype=np.float32)
                display_box = target_hint['box'].copy()
            continue
        settled = (completed and box is not None and consecutive >= 2 and stable_since is not None and
                   time.monotonic() - stable_since >= settle_seconds)
        # When references are still loading, keep RGB live instead of starting
        # a needless depth switch. Once loaded, only stable unmatched views use
        # depth; moving views keep trying current-frame RGB PnP.
        references_loading = bool((target_hint or {}).get('references_loading', lambda: False)())
        references_ready = bool((target_hint or {}).get('references_ready', lambda: False)())
        references_failed = bool((target_hint or {}).get('references_failed', lambda: False)())
        height, width = frame.shape[:2]
        fully_visible = (box is not None and box[0] > 3 and box[1] > 3 and
                         box[2] < width - 3 and box[3] < height - 3)
        rgb_first = provider is not None
        if rgb_first and references_ready and not references_failed and box is not None:
            rgb_search_started_at = rgb_search_started_at or time.monotonic()
        rgb_wait_expired = (rgb_search_started_at is not None and
                            time.monotonic() - rgb_search_started_at >=
                            float((target_hint or {}).get('rgb_fallback_timeout_s',
                                                         DEFAULT_RGB_FALLBACK_TIMEOUT_S)))
        # RGB reference PnP is the fast path. Avoid releasing the camera into a
        # multi-second depth burst while references are loading or still have a
        # short chance to solve the current view. Space remains an override.
        retry_after = float((target_hint or {}).get('depth_retry_after', 0.))
        depth_allowed = ((not rgb_first or references_failed or rgb_wait_expired) and
                         (retry_after <= 0. or time.monotonic() >= retry_after))
        # Space may force depth after references are ready, but while they are
        # loading it would preempt the fast RGB path with a costly camera switch.
        force_depth = key == 32 and not references_loading
        if force_depth or (settled and fully_visible and depth_allowed):
            if target_hint is not None and box is not None:
                target_hint['box'] = box.copy()
            return True
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False
        # Submit only after the completed result has been consumed and all exit
        # conditions have been checked.  Otherwise a successful confirmation
        # starts one more YOLO inference just before returning; the executor
        # context then waits for that needless inference and creates a visible
        # pause during the 2D -> depth handoff.
        if executor is not None and pending is None:
            rotation_hint = {}
            submitted_at = time.monotonic()
            pending_initializer = initializer
            pending_frame = frame.copy()
            if initializer is not None:
                pending = executor.submit(search_rgb_target, detector, pending_frame,
                                          preferred_rotation or iteration % 3 + 1,
                                          rotation_hint, initializer)
            else:
                pending = executor.submit(detect_target_box, detector, pending_frame,
                                          preferred_rotation or iteration % 3 + 1,
                                          rotation_hint)


def wait_to_retry(cap: cv2.VideoCapture, reason: str,
                  auto_retry: bool = False,
                  scene_reference: np.ndarray | None = None) -> bool:
    print(f"Initialization failed: {reason}", flush=True)
    print(("Waiting for the target to enter; press Space to retry or Q to quit."
           if scene_reference is not None else
           "Retrying automatically; press Q to quit."
           if auto_retry else
           "Press Space in the camera window to retry, or Q to quit."), flush=True)
    retry_at = time.monotonic() + 1.0
    reference_gray = (cv2.cvtColor(scene_reference, cv2.COLOR_BGR2GRAY)
                      if scene_reference is not None else None)
    previous_gray = None
    stable_since = None
    while True:
        frame = read_rgb(cap, 1)
        message = ("Waiting for target" if reference_gray is not None else
                   "Reacquiring depth" if auto_retry else "Detection failed")
        cv2.imshow(WINDOW, preview_frame(frame, message))
        key = collection_key(cap, cv2.waitKey(1) & 0xFF)
        if key in (27, ord("q"), ord("Q")):
            return False
        if key == 32:
            return True
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False
        if reference_gray is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            changed = np.count_nonzero(cv2.absdiff(gray, reference_gray) > 25)
            motion = (np.count_nonzero(cv2.absdiff(gray, previous_gray) > 15)
                      if previous_gray is not None else gray.size)
            if changed >= 900 and motion <= 300:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= .35:
                    return True
            else:
                stable_since = None
            previous_gray = gray
        elif auto_retry and time.monotonic() >= retry_at:
            return True


def run_live(args: argparse.Namespace) -> int:
    cv2.setNumThreads(2)
    startup_started = time.perf_counter()
    startup_timings: dict = {'module_import_s': round(MODULE_IMPORTED - MODULE_STARTED, 4)}
    camera_index, camera_name = resolve_camera_index(args.camera_name, args.camera)
    # DirectShow can reorder USB cameras when a capture is released on Windows,
    # so pin the physical device there. Linux uses the native/OpenCV backend and
    # does not need the Windows-only COM identity packages.
    args.camera_identity = (PinnedCamera.bind(camera_index, camera_name)
                            if sys.platform == 'win32' else None)
    camera_name = args.camera_identity.name if args.camera_identity is not None else camera_name
    if args.camera_identity is not None:
        record_event(args.output_dir, 'rgb_identity_pinned', camera_index=camera_index,
                     camera_name=camera_name, device_path=args.camera_identity.device_path)
    print(f"Astra RGB camera: {camera_index} ({camera_name or 'manual index'})", flush=True)
    cap = None
    detector = None
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    geometry_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='startup')
    reference_future = (geometry_executor.submit(prepare_rgb_initializer, args, startup_timings)
                        if getattr(args, 'rgb_first', False) else None)
    region_future = (geometry_executor.submit(prepare_live_region, args, startup_timings)
                     if getattr(args, 'region_tracking', False) else None)
    region_reported_error = False

    def current_region_tracker():
        nonlocal region_reported_error
        if region_future is None or not region_future.done():
            return None
        try:
            return region_future.result()
        except Exception as exc:
            if not region_reported_error:
                region_reported_error = True
                record_event(args.output_dir, 'm3t_preparation_failed', exc)
                print(f'M3T preparation unavailable: {exc}; RGB recovery remains active', flush=True)
            return None

    args.region_tracker_provider = current_region_tracker
    try:
        cap, detector = prepare_live_session(args, camera_index, startup_timings)
        startup_timings['search_ready_s'] = round(time.perf_counter() - startup_started, 4)
        startup_timings['module_to_search_s'] = round(
            time.perf_counter() - getattr(args, 'module_started', startup_started), 4)
        write_startup_timing(args, startup_timings, startup_started,
                             'camera_ready_3d_search')
    except PreviewCancelled:
        geometry_executor.shutdown(wait=True, cancel_futures=True)
        ready_region = current_region_tracker()
        if ready_region is not None:
            ready_region.close()
        cv2.destroyAllWindows()
        return 0
    except Exception:
        geometry_executor.shutdown(wait=True, cancel_futures=True)
        ready_region = current_region_tracker()
        if ready_region is not None:
            ready_region.close()
        cv2.destroyAllWindows()
        raise

    def prepare_geometry():
        prepare_live_geometry(args.model, startup_timings)
        startup_timings['geometry_ready_s'] = round(time.perf_counter() - startup_started, 4)
        write_startup_timing(args, startup_timings, startup_started, 'geometry_ready')

    geometry_future = geometry_executor.submit(prepare_geometry)

    def current_rgb_initializer():
        if getattr(args, 'rgb_initializer', None) is not None:
            reference_state['ready'] = True
            return args.rgb_initializer
        if reference_future is None:
            reference_state['failed'] = True
            return None
        if not reference_future.done():
            return None
        try:
            args.rgb_initializer = reference_future.result()
            reference_state['ready'] = bool(getattr(args.rgb_initializer, 'views', ()))
            reference_state['failed'] = not reference_state['ready']
            write_startup_timing(args, startup_timings, startup_started, 'rgb_references_ready')
            return args.rgb_initializer
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            reference_state['failed'] = True
            record_event(args.output_dir, 'rgb_reference_preparation_failed', exc)
            return None
    args.rgb_initializer_provider = current_rgb_initializer
    attempt = 0
    automatic_recovery = False
    manual_start = detector is None and getattr(args, 'manual_start', False)
    needs_target_input = detector is not None or manual_start or getattr(args, 'rgb_first', False)
    await_target = needs_target_input
    search_message = ('3D pose search | Space: depth fallback' if getattr(args, 'rgb_first', False) else
                      'No YOLO model | place target, Space to initialize'
                      if manual_start else 'Searching target')
    reference_state = {'ready': getattr(args, 'rgb_initializer', None) is not None,
                       'failed': False}
    target_hint = {'async_detection': True, 'settle_seconds': .35}
    if getattr(args, 'rgb_first', False):
        target_hint.update(initializer_provider=current_rgb_initializer,
                           references_loading=lambda: reference_future is not None and
                           not reference_future.done(),
                           references_ready=lambda: reference_state['ready'],
                           references_failed=lambda: reference_state['failed'],
                           rgb_fallback_timeout_s=getattr(
                               args, 'rgb_fallback_timeout_s', DEFAULT_RGB_FALLBACK_TIMEOUT_S))

    def reset_target_hint():
        settings = {key: target_hint[key] for key in
                    ('async_detection', 'settle_seconds', 'initializer_provider',
                     'references_loading', 'references_ready', 'references_failed',
                     'rgb_fallback_timeout_s', 'depth_retry_after', 'depth_fit_failures',
                     'search_stats')
                    if key in target_hint}
        target_hint.clear()
        target_hint.update(settings)
    try:
        while True:
            if await_target:
                try:
                    if not wait_for_target(cap, detector, target_hint, search_message):
                        return 0
                except (OSError, RuntimeError, cv2.error) as exc:
                    if 'frame read failed' not in str(exc).lower():
                        raise
                    record_event(args.output_dir, 'rgb_search_read_failed', exc)
                    cap.release()
                    cap = None
                    cap, _ = recover_rgb(args, camera_index,
                                          np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8),
                                          camera_name=camera_name)
                    reset_target_hint()
                    target_hint.update(async_detection=True, settle_seconds=.35)
                    continue
                await_target = False
            attempt += 1
            attempt_dir = args.output_dir / f"attempt_{attempt:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            initialization_started = time.perf_counter()
            timings = {'started_at': datetime.now().isoformat(timespec='milliseconds')}
            rgb_pose = target_hint.pop('rgb_pose', None)
            if rgb_pose is not None:
                target_hint.pop('depth_fit_failures', None)
                target_hint.pop('depth_retry_after', None)
                if 'first_3d_pose_s' not in startup_timings:
                    startup_timings['first_3d_pose_s'] = round(time.perf_counter() - startup_started, 4)
                    startup_timings['search_stats'] = target_hint.get('search_stats', {})
                    write_startup_timing(args, startup_timings, startup_started, 'first_3d_pose')
                rgb_frame = target_hint.pop('rgb_frame')
                pose_path = save_rgb_initial_pose(attempt_dir, rgb_pose, rgb_frame, args)
                timings.update(source='rgb_reference_pnp', rgb_pose_ms=rgb_pose.elapsed_ms)
                write_initialization_timing(attempt_dir, timings, initialization_started, 'ready')
                try:
                    tracker_result = run_tracker(args, pose_path, attempt_dir / 'tracking',
                                                  camera_index, cap, detector)
                except (OSError, RuntimeError, cv2.error) as exc:
                    if 'frame read failed' not in str(exc).lower():
                        raise
                    record_event(args.output_dir, 'tracker_rgb_read_failed', exc, attempt=attempt)
                    cap = None
                    tracker_result = REACQUIRE_EXIT_CODE
                if tracker_result != REACQUIRE_EXIT_CODE:
                    cap = None
                    return tracker_result
                if cap is None:
                    cap, _ = recover_rgb(args, camera_index, rgb_frame,
                                          camera_name=camera_name)
                automatic_recovery = True
                await_target = needs_target_input
                continue
            print("Hold the camera and target still during RGB-depth initialization.",
                  flush=True)
            try:
                for _ in range(args.preview_frames):
                    frame = read_rgb(cap, 1)
                    cv2.imshow(WINDOW, preview_frame(frame, "Preparing depth detection"))
                    key = collection_key(cap, cv2.waitKey(1) & 0xFF)
                    if key in (27, ord("q"), ord("Q")):
                        return 0
            except (OSError, RuntimeError, cv2.error) as exc:
                record_event(args.output_dir, 'rgb_preview_read_failed', exc, attempt=attempt)
                timings['error'] = str(exc)
                write_initialization_timing(attempt_dir, timings,
                                             initialization_started, 'rejected')
                cap.release()
                cap = None
                cap, _ = recover_rgb(args, camera_index,
                                      np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8),
                                      camera_name=camera_name)
                reset_target_hint()
                target_hint.update(async_detection=True, settle_seconds=.35)
                await_target = needs_target_input
                continue
            # The confirmed search and preview already provide fresh RGB frames.
            before = frame
            before_path = attempt_dir / "rgb_before.png"
            write_image(before_path, before)
            calibration_snapshot = attempt_dir / "rgb_depth_extrinsics.npz"
            shutil.copy2(args.extrinsics, calibration_snapshot)
            cap.release()
            cap = None
            timings['rgb_preparation_s'] = round(
                time.perf_counter() - initialization_started, 4)
            worker_error = None
            cancel_depth = threading.Event()
            rgb_constraint = None
            if target_hint.get('box') is not None:
                with np.load(calibration_snapshot) as calibration:
                    rgb_constraint = RgbConstraint(
                        np.asarray(target_hint['box'], np.float64),
                        calibration['K_rgb'].copy(), calibration['dist_rgb'].copy(),
                        calibration['T_rgb_depth'].copy())
            try:
                run_with_preview(
                    lambda: timed_stage(timings, 'depth_capture', lambda: run_depth_worker(
                        attempt_dir / "depth", args.depth_warmup, args.depth_timeout_s,
                        getattr(args, 'depth_frame_timeout_s', 2.0), cancel_depth,
                        args.extrinsics)),
                    before, "Capturing depth | Q: quit", None, cancel_depth)
            except (OSError, RuntimeError, subprocess.TimeoutExpired, PreviewCancelled) as exc:
                if isinstance(exc, PreviewCancelled):
                    write_initialization_timing(attempt_dir, timings,
                                                 initialization_started, 'cancelled')
                    return 0
                worker_error = exc
                record_event(args.output_dir, 'depth_capture_failed', exc, attempt=attempt)
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='depth-fit') as executor:
                def fit_prepared_geometry():
                    # Search and the depth burst can overlap CAD preparation;
                    # fitting waits for the shared caches instead of building
                    # them concurrently a second time.
                    geometry_future.result()
                    return fit_depth_capture(
                        attempt_dir / 'depth', attempt_dir, args.model,
                        args.settings, rgb_constraint)

                fit_future = (executor.submit(
                    timed_stage, timings, 'depth_fit', fit_prepared_geometry)
                    if worker_error is None else None)
                cap, after = recover_rgb(args, camera_index, before, timings, camera_name)
                after_path = attempt_dir / "rgb_after.png"
                write_image(after_path, after)
                try:
                    if worker_error is not None:
                        raise RuntimeError(str(worker_error))
                    pose_path = run_with_preview(
                        lambda: timed_stage(timings, 'pose_handoff', lambda: initialize_capture(
                            before_path, attempt_dir / "depth", attempt_dir,
                            args.model, calibration_snapshot, args.intrinsics,
                             args.settings, args.box_display_y_offset_px, after_path,
                             fit_future.result())),
                        after, "Estimating pose", cap)
                except (OSError, RuntimeError, ValueError, KeyError, cv2.error) as exc:
                    timings['error'] = str(exc)
                    record_event(args.output_dir, 'initialization_rejected', exc,
                                 attempt=attempt)
                    write_initialization_timing(attempt_dir, timings,
                                                 initialization_started, 'rejected')
                    if 'frame read failed' in str(exc).lower():
                        cap.release()
                        cap = None
                        cap, _ = recover_rgb(args, camera_index, after,
                                              camera_name=camera_name)
                        reset_target_hint()
                        target_hint.update(async_detection=True, settle_seconds=.35)
                        await_target = needs_target_input
                        continue
                    no_target = str(exc).startswith("No cluster matches the energy-unit dimensions")
                    if (detector is not None or getattr(args, 'rgb_first', False)) and (
                            worker_error is not None or no_target or
                            str(exc).startswith("Depth pose rejected") or
                            "RGB scene changed during depth capture" in str(exc)):
                        # Any rejected depth handoff must leave RGB time to
                        # search. Partial-pose conflicts used to bypass this
                        # backoff and repeatedly close RGB for the same fit.
                        if (getattr(args, 'rgb_first', False) and reference_state['ready']):
                            failures = target_hint.get('depth_fit_failures', 0) + 1
                            target_hint['depth_fit_failures'] = failures
                            cooldown = failed_depth_retry_seconds(attempt_dir, failures)
                            target_hint['depth_retry_after'] = time.monotonic() + cooldown
                            record_event(args.output_dir, 'failed_depth_rgb_retry',
                                         attempt=attempt, failures=failures, cooldown_s=cooldown,
                                         reason=str(exc))
                        print(f"Initialization failed: {exc}; searching for target again.",
                              flush=True)
                        await_target = True
                        target_hint['settle_seconds'] = .35
                        search_message = ('Depth capture retry | searching target'
                                          if worker_error is not None else
                                          '2D target found | waiting for valid depth'
                                          if no_target or str(exc).startswith('Depth pose rejected')
                                          else 'Target moved | confirming again')
                        continue
                    if not wait_to_retry(cap, str(exc),
                                         automatic_recovery or no_target or worker_error is not None,
                                         after if no_target else None):
                        return 1
                    continue
            write_initialization_timing(attempt_dir, timings,
                                         initialization_started, 'ready')
            target_hint.pop('depth_fit_failures', None)
            target_hint.pop('depth_retry_after', None)
            print(f"Depth initialization ready in {timings['total_to_pose_s']:.2f} s "
                  f"(capture {timings.get('depth_capture_s', 0):.2f} s; "
                  f"RGB reopen {timings.get('rgb_reopen_s', 0):.2f} s; "
                  f"fit {timings.get('depth_fit_s', 0):.2f} s, overlapping RGB)", flush=True)
            try:
                tracker_result = run_tracker(args, pose_path,
                                             attempt_dir / "tracking", camera_index,
                                             cap, detector)
            except (OSError, RuntimeError, cv2.error) as exc:
                if 'frame read failed' not in str(exc).lower():
                    raise
                record_event(args.output_dir, 'tracker_rgb_read_failed', exc,
                             attempt=attempt)
                # The tracker releases its camera in finally, including when
                # DirectShow temporarily stops returning frames.
                cap = None
                tracker_result = REACQUIRE_EXIT_CODE
            if tracker_result != REACQUIRE_EXIT_CODE:
                cap = None  # tracker releases it on normal exit
                return tracker_result
            print("Tracker requested a fresh depth pose; restarting capture.",
                  flush=True)
            # A relocalization request retains the already open RGB stream.
            # Avoid a redundant release/open before search releases it again
            # for the actual depth burst.
            if cap is None:
                cap, _ = recover_rgb(args, camera_index, after, camera_name=camera_name)
            automatic_recovery = True
            await_target = needs_target_input
            target_hint['settle_seconds'] = .35
    except PreviewCancelled:
        return 0
    finally:
        geometry_executor.shutdown(wait=True, cancel_futures=True)
        ready_region = current_region_tracker()
        if ready_region is not None:
            ready_region.close()
        if geometry_future.done() and not geometry_future.cancelled():
            error = geometry_future.exception()
            if error is not None:
                record_event(args.output_dir, 'startup_geometry_failed', error)
        if reference_future is not None and reference_future.done() and not reference_future.cancelled():
            error = reference_future.exception()
            if error is not None:
                record_event(args.output_dir, 'rgb_reference_preparation_failed', error)
        if cap is not None:
            cap.release()
        if detector is not None and hasattr(detector, 'close'):
            detector.close()
            record_event(args.output_dir, 'background_detector_closed',
                         stats=detector.stats, detector_error=detector.error)
        cv2.destroyAllWindows()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-name", default="USB Camera")
    parser.add_argument("--camera", type=int,
                        help="RGB camera index; overrides camera name selection")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--extrinsics", type=Path,
                        help="RGB-depth calibration; offline samples use their saved snapshot")
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--offline-sample", type=Path,
                        help="Initialize from an existing RGB-depth capture without a camera")
    parser.add_argument("--video", type=Path,
                        help="Track a recorded RGB clip after --offline-sample")
    parser.add_argument("--multi-target", action="store_true",
                        help="Detect all energy units with separate IDs and RGB/CAD poses")
    parser.add_argument("--headless", action="store_true",
                        help="Run multi-target mode without a preview window")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--region-tracking", action=argparse.BooleanOptionalAction, default=False,
                        help="Experimental M3T contour assistance; disabled after recorded-video regressions")
    parser.add_argument("--rgb-first", action=argparse.BooleanOptionalAction, default=True,
                        help="Use verified RGB/CAD references to initialize 6DoF during motion")
    parser.add_argument("--rgb-fallback-timeout-s", type=float,
                        default=DEFAULT_RGB_FALLBACK_TIMEOUT_S,
                        help="Seconds to try current-frame RGB PnP before stable unmatched views use depth")
    parser.add_argument("--camera-fourcc", default=os.environ.get(
                        'CAMERA_FOURCC', DEFAULT_LIVE_CAMERA_FOURCC),
                        help="Camera pixel format, normally MJPG; empty leaves driver default")
    parser.add_argument("--camera-fps", type=float, default=float(os.environ.get(
                        'CAMERA_FPS', DEFAULT_LIVE_CAMERA_FPS)),
                        help="Requested camera FPS; the driver may choose a lower supported rate")
    parser.add_argument("--detector-interval", type=int,
                        default=DEFAULT_LIVE_DETECTOR_INTERVAL,
                        help="Background RGB recovery check interval during stable tracking")
    parser.add_argument("--depth-fallback-on-loss", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="After sustained RGB loss, perform one depth-verified recovery attempt")
    parser.add_argument("--manual-start", action=argparse.BooleanOptionalAction, default=True,
                        help="Without a YOLO model, keep preview open until Space starts initialization")
    parser.add_argument("--preview-frames", type=int, default=1)
    parser.add_argument("--depth-warmup", type=int, default=2)
    parser.add_argument("--depth-timeout-s", type=float, default=8.0,
                        help="Hard limit for the isolated depth worker")
    parser.add_argument("--depth-frame-timeout-s", type=float, default=2.0,
                        help="OpenNI wait limit for one depth frame")
    parser.add_argument("--rgb-reopen-attempts", type=int, default=4)
    parser.add_argument("--rgb-warmup-frames", type=int, default=2,
                        help="RGB frames to read after switching back from depth")
    parser.add_argument("--reacquire-model", type=Path,
                        default=(DETECTOR_MODEL if DETECTOR_MODEL.is_file() else None),
                        help="RGB detector for stale-pose checks during live tracking; "
                             "if omitted and the bundled model is unavailable, live mode "
                             "continues without automatic detector-based reacquisition")
    parser.add_argument("--min-depth-mm", type=float, default=400.,
                        help="Minimum usable depth for live initialization (Astra near range)")
    # Live Astra acceptance covers the observed near-field working distance
    # around 0.5 m while retaining headroom for the planned acceptance range.
    parser.add_argument("--max-depth-mm", type=float, default=2400.0)
    parser.add_argument("--box-display-y-offset-px", type=float)
    parser.add_argument("--box-roll-mode", choices=("neutral", "true"),
                        default="true",
                        help="true displays measured 6DoF; neutral only changes the displayed box")
    parser.add_argument("--cad-outline", action=argparse.BooleanOptionalAction, default=True,
                        help="Show CAD silhouette and model axes; Z follows the long axis")
    parser.add_argument("--show-box", action=argparse.BooleanOptionalAction, default=False,
                        help="Show the optional metric model bounding box in single-target mode")
    parser.add_argument("--pose-axes", action=argparse.BooleanOptionalAction, default=True,
                        help="Show model X red, Y green, Z blue axes with the contour")
    parser.add_argument("--record-diagnostics", action=argparse.BooleanOptionalAction, default=True,
                        help="Save raw RGB and synchronized poses for the most recent 60 seconds")
    parser.add_argument("--diagnostic-duration-s", type=float, default=60.)
    args = parser.parse_args(argv)
    if args.multi_target and args.offline_sample is not None:
        parser.error("--multi-target uses RGB camera/video, not --offline-sample")
    if args.multi_target and args.reacquire_model is None:
        parser.error("--multi-target requires --reacquire-model weights")
    if args.multi_target and args.region_tracking:
        parser.error("--region-tracking is only supported in single-target mode")
    if args.headless and not args.multi_target:
        parser.error("--headless requires --multi-target")
    if args.video is not None and args.offline_sample is None and not args.multi_target:
        parser.error("--video requires --offline-sample")
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be non-negative")
    if not np.isfinite(args.diagnostic_duration_s) or args.diagnostic_duration_s <= 0:
        parser.error("--diagnostic-duration-s must be finite and positive")
    if args.max_frames < 0 or args.preview_frames < 1 or args.depth_warmup < 0:
        parser.error("Frame counts must be non-negative, with at least one preview frame")
    if (not np.isfinite([args.depth_timeout_s, args.depth_frame_timeout_s]).all()
            or args.depth_timeout_s <= 0 or args.depth_frame_timeout_s <= 0
            or args.rgb_reopen_attempts < 1):
        parser.error("Depth timeouts and RGB reopen attempts must be positive")
    if args.rgb_warmup_frames < 1:
        parser.error("--rgb-warmup-frames must be positive")
    if not np.isfinite(args.rgb_fallback_timeout_s) or args.rgb_fallback_timeout_s < 0:
        parser.error("--rgb-fallback-timeout-s must be finite and non-negative")
    if args.detector_interval < 1:
        parser.error("--detector-interval must be positive")
    if not np.isfinite(args.camera_fps) or args.camera_fps < 0:
        parser.error("--camera-fps must be non-negative")
    if args.camera_fourcc and len(args.camera_fourcc) != 4:
        parser.error("--camera-fourcc must contain exactly four characters")
    if (not np.isfinite(args.min_depth_mm) or not np.isfinite(args.max_depth_mm) or
            args.min_depth_mm <= 0 or args.min_depth_mm >= args.max_depth_mm):
        parser.error("Require 0 < min-depth-mm < max-depth-mm")
    args.settings = replace(Settings(), min_depth_mm=args.min_depth_mm,
                            max_depth_mm=args.max_depth_mm)
    if args.reacquire_model is not None and not args.reacquire_model.is_file():
        parser.error(f"Reacquisition detector does not exist: {args.reacquire_model}")
    if args.box_display_y_offset_px is None:
        args.box_display_y_offset_px = load_box_display_y_offset()
    if (not np.isfinite(args.box_display_y_offset_px) or
            abs(args.box_display_y_offset_px) > 100):
        parser.error("Box display Y offset must be finite and within +/-100 pixels")
    if args.output_dir is None:
        args.output_dir = DATA_ROOT / "tracking" / "live_depth_6dof" / \
            datetime.now().strftime("%Y%m%d_%H%M%S")
    return args


def main() -> int:
    args = parse_args()
    args.module_started = MODULE_STARTED
    native_log = None
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        native_log = (args.output_dir / 'native_fault.log').open('a', encoding='utf-8')
        faulthandler.enable(file=native_log)
    except (OSError, RuntimeError):
        faulthandler.enable()
    try:
        if getattr(args, 'multi_target', False):
            from .multi_target_live import run_multi_target
            return run_multi_target(args)
        if args.offline_sample is not None:
            sample = args.offline_sample
            args.extrinsics = sample_extrinsics(sample, args.extrinsics)
            after_path = sample / "rgb_after.png"
            pose_path = initialize_capture(
                sample / "rgb_before.png", sample / "depth", args.output_dir,
                args.model, args.extrinsics, args.intrinsics,
                args.settings, args.box_display_y_offset_px,
                after_path if after_path.is_file() else None)
            print(f"Saved initial pose: {pose_path.resolve()}")
            if args.video is None:
                return 0
            return run_tracker(args, pose_path, args.output_dir, None)
        args.extrinsics = args.extrinsics or DEFAULT_EXTRINSICS
        return run_live(args)
    except Exception as exc:
        # A third-party detector/driver exception must leave a diagnostic file
        # instead of looking like a window that simply flashed closed.
        record_event(args.output_dir, 'session_error', exc)
        try:
            (args.output_dir / 'last_error.txt').write_text(
                ''.join(traceback.format_exception(exc)), encoding='utf-8')
        except OSError:
            pass
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if native_log is not None:
            faulthandler.disable()
            native_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
