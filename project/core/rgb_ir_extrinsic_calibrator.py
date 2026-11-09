"""Offline RGB-to-IR checkerboard calibration for the Astra camera pair."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .paths import DATA_ROOT

DEFAULT_PAIRS = DATA_ROOT / "calibration" / "rgb_ir_pairs"
DEFAULT_RGB_INTRINSICS = DATA_ROOT / "calibration" / "camera_intrinsics.npz"
DEFAULT_DEPTH_METADATA = (DATA_ROOT / "tracking" / "depth_probe_adjusted_unmirrored"
                          / "metadata.json")
DEFAULT_OUTPUT = DATA_ROOT / "calibration" / "rgb_ir"
DEFAULT_BOARD = (9, 6)


@dataclass(frozen=True)
class PairDetection:
    name: str
    object_points: np.ndarray
    ir_points: np.ndarray
    rgb_points: np.ndarray
    corner_order: str


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def checkerboard_object_points(board_size: tuple[int, int], square_mm: float) -> np.ndarray:
    columns, rows = board_size
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_mm
    return points


def normalize_ir(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image.copy()
    finite = image[np.isfinite(image) & (image > 0)]
    if not finite.size:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def find_corners(image: np.ndarray, board_size: tuple[int, int],
                 infrared: bool = False) -> tuple[bool, np.ndarray | None, np.ndarray]:
    if infrared:
        gray = normalize_ir(image)
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8, copy=False)
    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
             | cv2.CALIB_CB_ACCURACY)
    variants = [gray]
    if infrared:
        variants.extend([
            255 - gray,
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
        ])
    for variant in variants:
        found, points = cv2.findChessboardCornersSB(variant, board_size, flags)
        if found:
            return True, points.astype(np.float32), variant
    return False, None, variants[-1]


def align_corner_order(ir_points: np.ndarray, rgb_points: np.ndarray,
                       ir_size: tuple[int, int], rgb_size: tuple[int, int],
                       board_size: tuple[int, int]) -> tuple[np.ndarray, str, dict[str, float]]:
    expected = board_size[0] * board_size[1]
    if len(ir_points) != expected or len(rgb_points) != expected:
        raise ValueError("Corner count does not match the checkerboard dimensions.")
    ir = ir_points.reshape(-1, 2)
    rgb = rgb_points.reshape(-1, 2)
    ir_scale = np.array(ir_size, dtype=np.float64)
    rgb_scale = np.array(rgb_size, dtype=np.float64)
    scores = {
        "original": float(np.median(np.linalg.norm(ir / ir_scale - rgb / rgb_scale, axis=1))),
        "rotated_180": float(np.median(
            np.linalg.norm(ir[::-1] / ir_scale - rgb / rgb_scale, axis=1))),
    }
    order = min(scores, key=scores.get)
    aligned = ir if order == "original" else ir[::-1]
    return aligned.reshape(-1, 1, 2).astype(np.float32), order, scores


def discover_pairs(folder: Path) -> tuple[list[tuple[str, Path, Path]], list[dict[str, str]]]:
    pair_folders = sorted(
        path for path in folder.glob("pair_*") if path.is_dir())
    if pair_folders:
        pairs = []
        missing = []
        for pair_folder in pair_folders:
            rgb_path = pair_folder / "rgb.png"
            ir_path = pair_folder / "ir_median_uint16.png"
            absent = [name for name, path in (("rgb.png", rgb_path),
                                               ("ir_median_uint16.png", ir_path))
                      if not path.is_file()]
            if absent:
                missing.append({
                    "name": pair_folder.name,
                    "reason": f"missing {', '.join(absent)}",
                })
            else:
                pairs.append((pair_folder.name, rgb_path, ir_path))
        if not pairs:
            raise RuntimeError(f"No complete pair_XXXX captures in {folder}")
        return pairs, missing

    rgb_folder = folder / "rgb"
    ir_folder = folder / "ir"
    if not rgb_folder.is_dir() or not ir_folder.is_dir():
        raise RuntimeError(f"Expected {rgb_folder} and {ir_folder}")
    rgb = {path.stem: path for path in rgb_folder.glob("*.png")}
    ir = {path.stem: path for path in ir_folder.glob("*.png")}
    names = sorted(rgb.keys() & ir.keys())
    missing = [
        *({"name": name, "reason": "missing IR image"} for name in sorted(rgb.keys() - ir.keys())),
        *({"name": name, "reason": "missing RGB image"} for name in sorted(ir.keys() - rgb.keys())),
    ]
    if not names:
        raise RuntimeError(f"No identically named RGB/IR PNG pairs in {folder}")
    return [(name, rgb[name], ir[name]) for name in names], missing


def pair_capture_rejection(pair_folder: Path, board_size: tuple[int, int],
                           square_mm: float, image_size: tuple[int, int]) -> str | None:
    manifest_path = pair_folder / "manifest.json"
    if not manifest_path.is_file():
        return "missing capture manifest"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ir_metadata = manifest["ir_metadata"]
        if manifest["accepted"] is not True:
            return "capture manifest marks pair as rejected"
        if tuple(manifest["board_size"]) != board_size:
            return "checkerboard dimensions differ from calibration settings"
        if not math.isclose(float(manifest["square_size_mm"]), square_mm,
                            rel_tol=0.0, abs_tol=1e-6):
            return "checkerboard square size differs from calibration settings"
        if ir_metadata["ir_mirroring"]["active"] is not False:
            return "IR stream was mirrored"
        mode = ir_metadata["video_mode"]
        if (int(mode["width"]), int(mode["height"])) != image_size:
            return "IR capture dimensions differ from calibration images"
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return f"invalid capture manifest: {exc}"
    return None


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.degrees(math.acos(float(value))))


def solve_calibration(objects: list[np.ndarray], ir_points: list[np.ndarray],
                      rgb_points: list[np.ndarray], image_size: tuple[int, int],
                      K_rgb: np.ndarray, dist_rgb: np.ndarray) -> dict[str, object]:
    if len(objects) < 3 or len(objects) != len(ir_points) or len(objects) != len(rgb_points):
        raise ValueError("Calibration requires equal object, IR and RGB view lists.")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
    ir_rms, K_ir, dist_ir, _, _ = cv2.calibrateCamera(
        objects, ir_points, image_size, None, None, criteria=criteria)
    stereo = cv2.stereoCalibrate(
        objects,
        ir_points,
        rgb_points,
        K_ir,
        dist_ir,
        K_rgb.copy(),
        dist_rgb.copy(),
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    stereo_rms, K_ir, dist_ir, K_rgb_out, dist_rgb_out, R_rgb_ir, t_rgb_ir, E, F = stereo
    T_rgb_ir = np.eye(4, dtype=np.float64)
    T_rgb_ir[:3, :3] = R_rgb_ir
    T_rgb_ir[:3, 3] = t_rgb_ir.ravel()
    T_ir_rgb = np.linalg.inv(T_rgb_ir)
    return {
        "ir_rms_px": float(ir_rms),
        "stereo_rms_px": float(stereo_rms),
        "K_ir": K_ir,
        "dist_ir": dist_ir,
        "K_rgb": K_rgb_out,
        "dist_rgb": dist_rgb_out,
        "R_rgb_ir": R_rgb_ir,
        "t_rgb_ir_mm": t_rgb_ir.reshape(3),
        "T_rgb_ir": T_rgb_ir,
        "T_ir_rgb": T_ir_rgb,
        "essential": E,
        "fundamental": F,
    }


def solve_depth_calibration(objects: list[np.ndarray], depth_pixels: list[np.ndarray],
                            rgb_points: list[np.ndarray], image_size: tuple[int, int],
                            K_depth: np.ndarray, K_rgb: np.ndarray,
                            dist_rgb: np.ndarray) -> dict[str, object]:
    if len(objects) < 3 or len(objects) != len(depth_pixels) or len(objects) != len(rgb_points):
        raise ValueError("Calibration requires equal object, depth-pixel and RGB view lists.")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-10)
    dist_depth = np.zeros((1, 5), dtype=np.float64)
    stereo = cv2.stereoCalibrate(
        objects,
        depth_pixels,
        rgb_points,
        np.asarray(K_depth, dtype=np.float64).copy(),
        dist_depth,
        K_rgb.copy(),
        dist_rgb.copy(),
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    rms, K_depth, dist_depth, K_rgb_out, dist_rgb_out, R, t, E, F = stereo
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R
    transform[:3, 3] = t.ravel()
    return {
        "depth_stereo_rms_px": float(rms),
        "K_depth": K_depth,
        "dist_depth": dist_depth,
        "K_rgb": K_rgb_out,
        "dist_rgb": dist_rgb_out,
        "R_rgb_depth": R,
        "t_rgb_depth_mm": t.reshape(3),
        "T_rgb_depth": transform,
        "T_depth_rgb": np.linalg.inv(transform),
        "essential_depth": E,
        "fundamental_depth": F,
    }


def project_camera_points(points: np.ndarray, K: np.ndarray,
                          dist: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        K,
        dist,
    )
    return projected.astype(np.float32)


def point_rms(measured: np.ndarray, predicted: np.ndarray) -> float:
    delta = measured.reshape(-1, 2) - predicted.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def cross_reprojection_errors(detection: PairDetection, calibration: dict[str, object]) -> dict[str, float]:
    obj = detection.object_points.astype(np.float64)
    K_ir = np.asarray(calibration["K_ir"])
    dist_ir = np.asarray(calibration["dist_ir"])
    K_rgb = np.asarray(calibration["K_rgb"])
    dist_rgb = np.asarray(calibration["dist_rgb"])
    R_rgb_ir = np.asarray(calibration["R_rgb_ir"])
    t_rgb_ir = np.asarray(calibration["t_rgb_ir_mm"]).reshape(1, 3)

    ok, rvec_ir, tvec_ir = cv2.solvePnP(obj, detection.ir_points, K_ir, dist_ir)
    if not ok:
        raise RuntimeError(f"IR board pose failed for {detection.name}")
    R_ir_board, _ = cv2.Rodrigues(rvec_ir)
    points_ir = obj @ R_ir_board.T + tvec_ir.reshape(1, 3)
    points_rgb = points_ir @ R_rgb_ir.T + t_rgb_ir
    predicted_rgb = project_camera_points(points_rgb, K_rgb, dist_rgb)
    ir_to_rgb_rms = point_rms(detection.rgb_points, predicted_rgb)

    ok, rvec_rgb, tvec_rgb = cv2.solvePnP(obj, detection.rgb_points, K_rgb, dist_rgb)
    if not ok:
        raise RuntimeError(f"RGB board pose failed for {detection.name}")
    R_rgb_board, _ = cv2.Rodrigues(rvec_rgb)
    points_rgb_from_board = obj @ R_rgb_board.T + tvec_rgb.reshape(1, 3)
    points_ir_from_rgb = (points_rgb_from_board - t_rgb_ir) @ R_rgb_ir
    predicted_ir = project_camera_points(points_ir_from_rgb, K_ir, dist_ir)
    rgb_to_ir_rms = point_rms(detection.ir_points, predicted_ir)
    combined = math.sqrt((ir_to_rgb_rms ** 2 + rgb_to_ir_rms ** 2) / 2.0)
    return {
        "ir_pose_to_rgb_rms_px": ir_to_rgb_rms,
        "rgb_pose_to_ir_rms_px": rgb_to_ir_rms,
        "symmetric_cross_rms_px": float(combined),
    }


def board_geometry_ir(detection: PairDetection,
                      calibration: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    ok, rvec, tvec = cv2.solvePnP(
        detection.object_points,
        detection.ir_points,
        np.asarray(calibration["K_ir"]),
        np.asarray(calibration["dist_ir"]),
    )
    if not ok:
        raise RuntimeError(f"IR board pose failed for {detection.name}")
    rotation, _ = cv2.Rodrigues(rvec)
    object_center = detection.object_points.mean(axis=0).astype(np.float64)
    center_ir = rotation @ object_center + tvec.reshape(3)
    normal_ir = rotation[:, 2]
    return center_ir, normal_ir


def summarize_view_diversity(centers: np.ndarray, normals: np.ndarray) -> dict[str, object]:
    centers = np.asarray(centers, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or normals.shape != centers.shape:
        raise ValueError("Centers and normals must both have shape (views, 3).")
    if not len(centers):
        raise ValueError("At least one view is required.")
    normalized = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    cosines = np.clip(np.abs(normalized @ normalized.T), 0.0, 1.0)
    max_normal_angle = float(np.degrees(np.arccos(cosines)).max())
    spans = np.ptp(centers, axis=0)
    return {
        "board_center_span_ir_mm": spans.tolist(),
        "board_depth_min_mm": float(centers[:, 2].min()),
        "board_depth_max_mm": float(centers[:, 2].max()),
        "max_board_normal_separation_deg": max_normal_angle,
    }


def make_detection_preview(rgb: np.ndarray, ir_display: np.ndarray,
                           rgb_points: np.ndarray, ir_points: np.ndarray,
                           label: str, board_size: tuple[int, int]) -> np.ndarray:
    rgb_panel = rgb.copy() if rgb.ndim == 3 else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    ir_panel = cv2.cvtColor(ir_display, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(rgb_panel, board_size, rgb_points, True)
    cv2.drawChessboardCorners(ir_panel, board_size, ir_points, True)
    panel = np.hstack((rgb_panel, ir_panel))
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (255, 255, 255), -1)
    cv2.putText(panel, f"{label} | RGB left / IR right", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def calibrate(pairs_folder: Path, rgb_intrinsics_path: Path,
              depth_metadata_path: Path, output: Path,
              board_size: tuple[int, int] = DEFAULT_BOARD, square_mm: float = 25.0,
              min_pairs: int = 12) -> dict[str, object]:
    pairs, rejected = discover_pairs(pairs_folder)
    with np.load(rgb_intrinsics_path) as archive:
        K_rgb = archive["K"].copy()
        dist_rgb = archive["dist"].copy()
        expected_rgb_size = tuple(int(value) for value in archive["image_size"])
    depth_metadata = json.loads(depth_metadata_path.read_text(encoding="utf-8"))
    if depth_metadata.get("depth_mirroring", {}).get("active") is not False:
        raise RuntimeError("Depth metadata must describe an unmirrored OpenNI stream.")
    depth_mode = depth_metadata["video_mode"]
    depth_size = (int(depth_mode["width"]), int(depth_mode["height"]))
    if depth_size != expected_rgb_size:
        raise RuntimeError(
            f"Depth image size {depth_size} does not match RGB {expected_rgb_size}.")
    depth_intrinsics = depth_metadata["intrinsics_estimate"]
    K_depth = np.array([
        [depth_intrinsics["fx"], 0.0, depth_intrinsics["cx"]],
        [0.0, depth_intrinsics["fy"], depth_intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    template = checkerboard_object_points(board_size, square_mm)
    detections: list[PairDetection] = []
    previews: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    order_details: dict[str, dict[str, object]] = {}

    for name, rgb_path, ir_path in pairs:
        if rgb_path.parent.name.startswith("pair_"):
            reason = pair_capture_rejection(
                rgb_path.parent, board_size, square_mm, expected_rgb_size)
            if reason:
                rejected.append({"name": name, "reason": reason})
                continue
        rgb = read_image(rgb_path, cv2.IMREAD_COLOR)
        ir = read_image(ir_path, cv2.IMREAD_UNCHANGED)
        rgb_size = (rgb.shape[1], rgb.shape[0])
        ir_size = (ir.shape[1], ir.shape[0])
        if rgb_size != expected_rgb_size or ir_size != expected_rgb_size:
            rejected.append({"name": name, "reason": (
                f"image size mismatch: RGB {rgb_size}, IR {ir_size}, expected {expected_rgb_size}")})
            continue
        if image_size is None:
            image_size = rgb_size
        rgb_found, rgb_corners, _ = find_corners(rgb, board_size, infrared=False)
        ir_found, ir_corners, ir_display = find_corners(ir, board_size, infrared=True)
        if not rgb_found or not ir_found:
            missing = "RGB and IR" if not rgb_found and not ir_found else ("RGB" if not rgb_found else "IR")
            rejected.append({"name": name, "reason": f"checkerboard not detected in {missing}"})
            continue
        aligned_ir, order, scores = align_corner_order(
            ir_corners, rgb_corners, ir_size, rgb_size, board_size)
        detection = PairDetection(name, template.copy(), aligned_ir, rgb_corners, order)
        detections.append(detection)
        order_details[name] = {"selected": order, "normalized_median_scores": scores}
        previews.append(make_detection_preview(
            rgb, ir_display, rgb_corners, aligned_ir, name, board_size))

    if len(detections) < min_pairs:
        raise RuntimeError(
            f"Only {len(detections)} valid RGB/IR checkerboard pairs; need at least {min_pairs}.")
    calibration = solve_calibration(
        [item.object_points for item in detections],
        [item.ir_points for item in detections],
        [item.rgb_points for item in detections],
        image_size,
        K_rgb,
        dist_rgb,
    )
    depth_calibration = solve_depth_calibration(
        [item.object_points for item in detections],
        [item.ir_points for item in detections],
        [item.rgb_points for item in detections],
        image_size,
        K_depth,
        K_rgb,
        dist_rgb,
    )
    depth_cross_calibration = {
        "K_ir": depth_calibration["K_depth"],
        "dist_ir": depth_calibration["dist_depth"],
        "K_rgb": depth_calibration["K_rgb"],
        "dist_rgb": depth_calibration["dist_rgb"],
        "R_rgb_ir": depth_calibration["R_rgb_depth"],
        "t_rgb_ir_mm": depth_calibration["t_rgb_depth_mm"],
    }
    views = []
    board_centers = []
    board_normals = []
    for detection in detections:
        center_ir, normal_ir = board_geometry_ir(detection, calibration)
        board_centers.append(center_ir)
        board_normals.append(normal_ir)
        ir_errors = cross_reprojection_errors(detection, calibration)
        depth_errors = cross_reprojection_errors(detection, depth_cross_calibration)
        views.append({
            "name": detection.name,
            "corner_order": detection.corner_order,
            "board_center_ir_mm": center_ir.tolist(),
            "board_normal_ir": normal_ir.tolist(),
            **ir_errors,
            "depth_pose_to_rgb_rms_px": depth_errors["ir_pose_to_rgb_rms_px"],
            "rgb_pose_to_depth_rms_px": depth_errors["rgb_pose_to_ir_rms_px"],
            "depth_symmetric_cross_rms_px": depth_errors["symmetric_cross_rms_px"],
        })
    cross_values = np.array([item["symmetric_cross_rms_px"] for item in views])
    depth_cross_values = np.array([
        item["depth_symmetric_cross_rms_px"] for item in views])
    diversity = summarize_view_diversity(
        np.asarray(board_centers), np.asarray(board_normals))
    center_span = diversity["board_center_span_ir_mm"]
    ir_baseline_mm = float(np.linalg.norm(calibration["t_rgb_ir_mm"]))
    ir_rotation_deg = rotation_angle_deg(calibration["R_rgb_ir"])
    depth_baseline_mm = float(np.linalg.norm(depth_calibration["t_rgb_depth_mm"]))
    depth_rotation_deg = rotation_angle_deg(depth_calibration["R_rgb_depth"])
    ir_cross_rms = float(np.sqrt(np.mean(cross_values ** 2)))
    ir_cross_p95 = float(np.percentile(cross_values, 95))
    depth_cross_rms = float(np.sqrt(np.mean(depth_cross_values ** 2)))
    depth_cross_p95 = float(np.percentile(depth_cross_values, 95))
    quality = {
        "accepted": bool(
            len(detections) >= min_pairs
            and calibration["ir_rms_px"] <= 0.8
            and calibration["stereo_rms_px"] <= 0.8
            and ir_cross_rms <= 1.5
            and ir_cross_p95 <= 3.0
            and depth_calibration["depth_stereo_rms_px"] <= 0.8
            and depth_cross_rms <= 1.5
            and depth_cross_p95 <= 3.0
            and 10.0 <= ir_baseline_mm <= 100.0
            and 10.0 <= depth_baseline_mm <= 100.0
            and ir_rotation_deg <= 10.0
            and depth_rotation_deg <= 10.0
            and center_span[0] >= 80.0
            and center_span[1] >= 60.0
            and center_span[2] >= 120.0
            and diversity["max_board_normal_separation_deg"] >= 15.0
        ),
        "minimum_pairs": min_pairs,
        "ir_rms_target_px": 0.8,
        "ir_stereo_rms_target_px": 0.8,
        "depth_stereo_rms_target_px": 0.8,
        "cross_rms_target_px": 1.5,
        "cross_p95_target_px": 3.0,
        "baseline_range_mm": [10.0, 100.0],
        "relative_rotation_max_deg": 10.0,
        "board_center_span_targets_ir_mm": [80.0, 60.0, 120.0],
        "board_normal_separation_target_deg": 15.0,
    }
    report = {
        "definition_ir": "T_rgb_ir maps a point modeled by calibrated K_ir/dist_ir into RGB coordinates.",
        "definition_depth": "T_rgb_depth maps an unmirrored OpenNI depth point modeled by K_depth and zero distortion into RGB coordinates.",
        "pose_handoff": "T_rgb_model = T_rgb_depth @ T_depth_model",
        "board_size_inner_corners": list(board_size),
        "square_size_mm": square_mm,
        "image_size": list(image_size),
        "valid_pairs": len(detections),
        "rejected_pairs": rejected,
        "rgb_intrinsics": str(rgb_intrinsics_path.resolve()),
        "depth_intrinsics_metadata": str(depth_metadata_path.resolve()),
        "ir_rms_px": calibration["ir_rms_px"],
        "ir_stereo_rms_px": calibration["stereo_rms_px"],
        "ir_symmetric_cross_rms_px": ir_cross_rms,
        "ir_symmetric_cross_p95_px": ir_cross_p95,
        "depth_stereo_rms_px": depth_calibration["depth_stereo_rms_px"],
        "depth_symmetric_cross_rms_px": depth_cross_rms,
        "depth_symmetric_cross_p95_px": depth_cross_p95,
        "cross_error_scope": "In-sample diagnostic only; independent held-out depth projection validation is still required.",
        "ir_baseline_mm": ir_baseline_mm,
        "ir_relative_rotation_deg": ir_rotation_deg,
        "depth_baseline_mm": depth_baseline_mm,
        "depth_relative_rotation_deg": depth_rotation_deg,
        "view_diversity": diversity,
        "K_ir": calibration["K_ir"].tolist(),
        "dist_ir": calibration["dist_ir"].tolist(),
        "K_rgb": calibration["K_rgb"].tolist(),
        "dist_rgb": calibration["dist_rgb"].tolist(),
        "R_rgb_ir": calibration["R_rgb_ir"].tolist(),
        "t_rgb_ir_mm": calibration["t_rgb_ir_mm"].tolist(),
        "T_rgb_ir": calibration["T_rgb_ir"].tolist(),
        "T_ir_rgb": calibration["T_ir_rgb"].tolist(),
        "K_depth": depth_calibration["K_depth"].tolist(),
        "dist_depth": depth_calibration["dist_depth"].tolist(),
        "R_rgb_depth": depth_calibration["R_rgb_depth"].tolist(),
        "t_rgb_depth_mm": depth_calibration["t_rgb_depth_mm"].tolist(),
        "T_rgb_depth": depth_calibration["T_rgb_depth"].tolist(),
        "T_depth_rgb": depth_calibration["T_depth_rgb"].tolist(),
        "quality": quality,
        "corner_order": order_details,
        "views": views,
        "capture_requirement": "Original 640x480 images with both RGB and IR mirroring disabled; no resize or crop.",
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "rgb_ir_extrinsics.npz",
        K_ir=calibration["K_ir"],
        dist_ir=calibration["dist_ir"],
        K_rgb=calibration["K_rgb"],
        dist_rgb=calibration["dist_rgb"],
        R_rgb_ir=calibration["R_rgb_ir"],
        t_rgb_ir_mm=calibration["t_rgb_ir_mm"],
        T_rgb_ir=calibration["T_rgb_ir"],
        T_ir_rgb=calibration["T_ir_rgb"],
        K_depth=depth_calibration["K_depth"],
        dist_depth=depth_calibration["dist_depth"],
        R_rgb_depth=depth_calibration["R_rgb_depth"],
        t_rgb_depth_mm=depth_calibration["t_rgb_depth_mm"],
        T_rgb_depth=depth_calibration["T_rgb_depth"],
        T_depth_rgb=depth_calibration["T_depth_rgb"],
        E=calibration["essential"],
        F=calibration["fundamental"],
        E_depth=depth_calibration["essential_depth"],
        F_depth=depth_calibration["fundamental_depth"],
        image_size=np.array(image_size),
        board_size=np.array(board_size),
        square_size_mm=np.array(square_mm),
        ir_rms_px=np.array(calibration["ir_rms_px"]),
        ir_stereo_rms_px=np.array(calibration["stereo_rms_px"]),
        depth_stereo_rms_px=np.array(depth_calibration["depth_stereo_rms_px"]),
    )
    (output / "rgb_ir_extrinsics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    for index in range(0, len(previews), 5):
        page = np.vstack(previews[index:index + 5])
        write_png(output / f"detections_{index // 5 + 1}.png", page)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                        help="Folder containing pair_XXXX/rgb.png and ir_median_uint16.png")
    parser.add_argument("--rgb-intrinsics", type=Path, default=DEFAULT_RGB_INTRINSICS)
    parser.add_argument("--depth-metadata", type=Path, default=DEFAULT_DEPTH_METADATA,
                        help="Metadata from an unmirrored OpenNI depth capture")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--board-columns", type=int, default=DEFAULT_BOARD[0])
    parser.add_argument("--board-rows", type=int, default=DEFAULT_BOARD[1])
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--min-pairs", type=int, default=12)
    args = parser.parse_args()
    if args.board_columns < 2 or args.board_rows < 2:
        parser.error("Checkerboard dimensions must both be at least two.")
    if args.square_mm <= 0:
        parser.error("--square-mm must be positive")
    if args.min_pairs < 3:
        parser.error("--min-pairs must be at least three")
    return args


def main() -> None:
    args = parse_args()
    report = calibrate(
        args.pairs.resolve(),
        args.rgb_intrinsics.resolve(),
        args.depth_metadata.resolve(),
        args.output.resolve(),
        (args.board_columns, args.board_rows),
        args.square_mm,
        args.min_pairs,
    )
    print(f"Valid pairs: {report['valid_pairs']}")
    print(f"IR RMS: {report['ir_rms_px']:.4f} px")
    print(f"IR stereo RMS: {report['ir_stereo_rms_px']:.4f} px")
    print(f"Depth stereo RMS: {report['depth_stereo_rms_px']:.4f} px")
    print(f"Depth baseline: {report['depth_baseline_mm']:.2f} mm")
    print(f"Accepted: {report['quality']['accepted']}")


if __name__ == "__main__":
    main()
