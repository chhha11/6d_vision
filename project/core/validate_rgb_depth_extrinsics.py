"""Validate RGB-depth extrinsics on a held-out checkerboard with depth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from . import rgb_ir_extrinsic_calibrator as calibrator

from .paths import DATA_ROOT

DEFAULT_CAPTURE = DATA_ROOT / "calibration" / "rgb_ir_validation" / "held_out_001"
DEFAULT_EXTRINSICS = DATA_ROOT / "calibration" / "rgb_ir" / "rgb_ir_extrinsics.npz"
BOARD = (9, 6)


def validate_capture_contract(rgb: np.ndarray, ir: np.ndarray, depth: np.ndarray,
                              ir_metadata: dict[str, object],
                              depth_metadata: dict[str, object],
                              saved_size: tuple[int, int],
                              saved_K_depth: np.ndarray) -> None:
    actual_sizes = [(image.shape[1], image.shape[0]) for image in (rgb, ir, depth)]
    if any(size != saved_size for size in actual_sizes):
        raise RuntimeError(
            f"RGB/IR/depth image sizes {actual_sizes} differ from calibrated {saved_size}.")
    for stream, metadata, mirror_key in (
        ("IR", ir_metadata, "ir_mirroring"),
        ("depth", depth_metadata, "depth_mirroring"),
    ):
        mode = metadata.get("video_mode", {})
        if (mode.get("width"), mode.get("height")) != saved_size:
            raise RuntimeError(f"{stream} metadata dimensions differ from calibrated size.")
        if metadata.get(mirror_key, {}).get("active") is not False:
            raise RuntimeError(f"{stream} stream must be unmirrored.")
    if depth_metadata["video_mode"].get("depth_unit_mm") != 1:
        raise RuntimeError("Depth data must use 1 mm units.")
    if ir_metadata.get("device", {}).get("uri") != depth_metadata.get("device", {}).get("uri"):
        raise RuntimeError("IR and depth captures came from different devices.")
    estimate = depth_metadata.get("intrinsics_estimate", {})
    try:
        capture_K_depth = np.array([
            [estimate["fx"], 0.0, estimate["cx"]],
            [0.0, estimate["fy"], estimate["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    except KeyError as exc:
        raise RuntimeError(f"Depth metadata lacks intrinsic parameter {exc}.") from exc
    if not np.allclose(capture_K_depth, saved_K_depth, rtol=0.0, atol=1e-3):
        raise RuntimeError("Capture depth intrinsics differ from calibrated K_depth.")


def rays_from_matrix(pixels: np.ndarray, K: np.ndarray,
                     dist: np.ndarray | None = None) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    if dist is None:
        x = (pixels[:, 0, 0] - K[0, 2]) / K[0, 0]
        y = (pixels[:, 0, 1] - K[1, 2]) / K[1, 1]
        return np.column_stack((x, y, np.ones_like(x)))
    normalized = cv2.undistortPoints(pixels, K, dist).reshape(-1, 2)
    return np.column_stack((normalized, np.ones(len(normalized))))


def fit_plane(points: np.ndarray, threshold_mm: float = 3.0
              ) -> tuple[np.ndarray, np.ndarray]:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.utility.random.seed(0)
    equation, indices = cloud.segment_plane(
        distance_threshold=threshold_mm, ransac_n=3, num_iterations=2000)
    equation = np.asarray(equation, dtype=np.float64)
    if equation[3] > 0:
        equation *= -1.0
    return equation, np.asarray(indices, dtype=np.int32)


def intersect_plane(rays: np.ndarray, plane: np.ndarray) -> np.ndarray:
    denominator = rays @ plane[:3]
    if np.any(np.abs(denominator) < 1e-8):
        raise RuntimeError("A checkerboard ray is parallel to the fitted plane.")
    scale = -plane[3] / denominator
    return rays * scale[:, None]


def project_rgb(points_depth: np.ndarray, transform: np.ndarray,
                K_rgb: np.ndarray, dist_rgb: np.ndarray) -> np.ndarray:
    points_rgb = (points_depth @ transform[:3, :3].T
                  + transform[:3, 3].reshape(1, 3))
    return calibrator.project_camera_points(points_rgb, K_rgb, dist_rgb)


def adjacent_spacing(points: np.ndarray) -> dict[str, float]:
    grid = points.reshape(BOARD[1], BOARD[0], 3)
    horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()
    vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()
    return {
        "horizontal_median_mm": float(np.median(horizontal)),
        "vertical_median_mm": float(np.median(vertical)),
        "combined_p95_abs_error_mm": float(np.percentile(
            np.abs(np.concatenate((horizontal, vertical)) - 25.0), 95)),
    }


def evaluate_model(name: str, depth: np.ndarray, roi_pixels: np.ndarray,
                   corner_pixels: np.ndarray, ray_K: np.ndarray,
                   ray_dist: np.ndarray | None, measured_rgb: np.ndarray,
                   T_rgb_depth: np.ndarray, T_depth_rgb: np.ndarray,
                   K_rgb: np.ndarray, dist_rgb: np.ndarray
                   ) -> tuple[dict[str, object], np.ndarray]:
    roi_rays = rays_from_matrix(roi_pixels, ray_K, ray_dist)
    rows = roi_pixels[:, 1].astype(int)
    cols = roi_pixels[:, 0].astype(int)
    points = roi_rays * depth[rows, cols].astype(np.float64)[:, None]
    plane, inliers = fit_plane(points)
    residuals = np.abs(points @ plane[:3] + plane[3])
    corner_rays = rays_from_matrix(corner_pixels, ray_K, ray_dist)
    corner_points = intersect_plane(corner_rays, plane)
    predicted = project_rgb(corner_points, T_rgb_depth, K_rgb, dist_rgb)
    wrong_direction = project_rgb(corner_points, T_depth_rgb, K_rgb, dist_rgb)
    errors = np.linalg.norm(
        measured_rgb.reshape(-1, 2) - predicted.reshape(-1, 2), axis=1)
    wrong_errors = np.linalg.norm(
        measured_rgb.reshape(-1, 2) - wrong_direction.reshape(-1, 2), axis=1)
    result: dict[str, object] = {
        "name": name,
        "roi_depth_points": int(len(points)),
        "plane_inlier_points": int(len(inliers)),
        "plane_inlier_ratio": float(len(inliers) / len(points)),
        "plane_equation": plane.tolist(),
        "plane_residual_median_mm": float(np.median(residuals)),
        "plane_residual_p95_mm": float(np.percentile(residuals, 95)),
        "rgb_projection_error_median_px": float(np.median(errors)),
        "rgb_projection_error_rms_px": float(np.sqrt(np.mean(errors ** 2))),
        "rgb_projection_error_p95_px": float(np.percentile(errors, 95)),
        "inverse_transform_error_median_px": float(np.median(wrong_errors)),
        "spacing": adjacent_spacing(corner_points),
    }
    return result, predicted


def draw_validation(rgb: np.ndarray, measured: np.ndarray,
                    predicted: np.ndarray) -> np.ndarray:
    preview = rgb.copy()
    for actual, estimate in zip(measured.reshape(-1, 2), predicted.reshape(-1, 2)):
        a = tuple(np.rint(actual).astype(int))
        e = tuple(np.rint(estimate).astype(int))
        cv2.line(preview, a, e, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(preview, a, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(preview, e, 2, (0, 0, 255), -1, cv2.LINE_AA)
    return preview


def validate(capture: Path, extrinsics_path: Path) -> dict[str, object]:
    rgb = calibrator.read_image(capture / "rgb_after.png")
    ir = calibrator.read_image(capture / "ir" / "ir_median_uint16.png")
    with np.load(capture / "depth" / "depth_burst.npz") as archive:
        depth = archive["median_depth_mm"].copy()
    depth_metadata = json.loads(
        (capture / "depth" / "metadata.json").read_text(encoding="utf-8"))
    ir_metadata = json.loads(
        (capture / "ir" / "metadata.json").read_text(encoding="utf-8"))
    with np.load(extrinsics_path) as saved:
        K_ir = saved["K_ir"]
        dist_ir = saved["dist_ir"]
        K_rgb = saved["K_rgb"]
        dist_rgb = saved["dist_rgb"]
        K_depth = saved["K_depth"]
        saved_size = tuple(int(value) for value in saved["image_size"])
        T_rgb_depth = saved["T_rgb_depth"]
        T_depth_rgb = saved["T_depth_rgb"]
    validate_capture_contract(
        rgb, ir, depth, ir_metadata, depth_metadata, saved_size, K_depth)
    found_rgb, rgb_corners, _ = calibrator.find_corners(rgb, BOARD)
    found_ir, ir_corners, _ = calibrator.find_corners(ir, BOARD, infrared=True)
    if not found_rgb or not found_ir:
        raise RuntimeError("The held-out RGB or IR checkerboard was not detected.")
    ir_corners, order, order_scores = calibrator.align_corner_order(
        ir_corners, rgb_corners, saved_size, saved_size, BOARD)

    polygon = cv2.convexHull(np.rint(ir_corners).astype(np.int32))
    roi_mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, polygon.reshape(-1, 2), 1)
    valid = (roi_mask > 0) & (depth >= 400) & (depth <= 1200)
    rows, cols = np.nonzero(valid)
    roi_pixels = np.column_stack((cols, rows)).astype(np.float64)
    if len(roi_pixels) < 500:
        raise RuntimeError("Too few valid checkerboard depth pixels for validation.")

    models = [
        ("openni_fov", K_depth, None),
        ("calibrated_ir", K_ir, dist_ir),
    ]
    evaluations = []
    predictions = {}
    for name, K, dist in models:
        evaluation, prediction = evaluate_model(
            name, depth, roi_pixels, ir_corners.reshape(-1, 2), K, dist,
            rgb_corners, T_rgb_depth, T_depth_rgb, K_rgb, dist_rgb)
        evaluations.append(evaluation)
        predictions[name] = prediction
    best = evaluations[0]
    preview = draw_validation(rgb, rgb_corners, predictions[str(best["name"])])
    calibrator.write_png(capture / "depth_to_rgb_validation.png", preview)
    accepted = bool(
        float(best["plane_inlier_ratio"]) >= 0.80
        and float(best["plane_residual_median_mm"]) <= 1.5
        and float(best["plane_residual_p95_mm"]) <= 3.0
        and float(best["rgb_projection_error_rms_px"]) <= 1.0
        and float(best["rgb_projection_error_p95_px"]) <= 2.0
        and float(best["spacing"]["combined_p95_abs_error_mm"]) <= 1.5
        and float(best["inverse_transform_error_median_px"])
        > float(best["rgb_projection_error_median_px"]) * 2.0
    )
    result = {
        "capture": str(capture.resolve()),
        "extrinsics": str(extrinsics_path.resolve()),
        "checkerboard_corner_order": order,
        "corner_order_scores": order_scores,
        "models": evaluations,
        "selected_depth_ray_model": best["name"],
        "accepted": accepted,
        "quality_thresholds": {
            "plane_inlier_ratio_min": 0.80,
            "plane_residual_median_max_mm": 1.5,
            "plane_residual_p95_max_mm": 3.0,
            "rgb_projection_rms_max_px": 1.0,
            "rgb_projection_p95_max_px": 2.0,
            "spacing_p95_abs_error_max_mm": 1.5,
        },
        "acceptance_note": (
            "Independent held-out RGB/IR/depth validation. RGB-before was not used "
            "because the board had not yet been placed in that frame."
        ),
    }
    (capture / "validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(args.capture.resolve(), args.extrinsics.resolve())
    best = result["models"][0]
    print(f"Depth ray model: {result['selected_depth_ray_model']}")
    print(f"Plane residual median/p95: {best['plane_residual_median_mm']:.3f}/"
          f"{best['plane_residual_p95_mm']:.3f} mm")
    print(f"RGB projection median/RMS/p95: {best['rgb_projection_error_median_px']:.3f}/"
          f"{best['rgb_projection_error_rms_px']:.3f}/"
          f"{best['rgb_projection_error_p95_px']:.3f} px")
    print(f"Accepted: {result['accepted']}")


if __name__ == "__main__":
    main()
