"""Estimate camera motion from the fixed checkerboard in RGB-D samples."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .calibrate_camera import BOARD, object_points, read_image

from .paths import DATA_ROOT

DEFAULT_DATASET = DATA_ROOT / "samples" / "board_reference"
DEFAULT_INTRINSICS = DATA_ROOT / "calibration" / "camera_intrinsics.npz"
MAX_BOARD_CORNER_MEDIAN_DRIFT_PX = 0.75
MAX_BOARD_CORNER_P95_DRIFT_PX = 1.5


def matrix4(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    return transform


def camera_motion_in_first_frame(first_T_camera_board: np.ndarray,
                                 current_T_board_camera: np.ndarray) -> np.ndarray:
    return first_T_camera_board @ current_T_board_camera


def reprojection_rms(object_pts: np.ndarray, image_pts: np.ndarray,
                     rvec: np.ndarray, tvec: np.ndarray,
                     K: np.ndarray, dist: np.ndarray) -> float:
    projected, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
    errors = projected.reshape(-1, 2) - image_pts.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


def detect_board_corners(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
             | cv2.CALIB_CB_ACCURACY)
    found, corners = cv2.findChessboardCornersSB(gray, BOARD, flags)
    return corners if found else None


def board_corner_drift(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(
        before.reshape(-1, 2) - after.reshape(-1, 2), axis=1)
    return {
        "median_px": float(np.median(distances)),
        "p95_px": float(np.percentile(distances, 95)),
    }


def estimate_board_pose(image: np.ndarray, object_pts: np.ndarray,
                        K: np.ndarray, dist: np.ndarray) -> dict[str, object] | None:
    corners = detect_board_corners(image)
    if corners is None:
        return None
    ok, rvec, tvec = cv2.solvePnP(object_pts, corners, K, dist)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    T_camera_board = matrix4(rotation, tvec)
    T_board_camera = np.linalg.inv(T_camera_board)

    grid = np.indices((BOARD[1], BOARD[0]))
    fit_mask = (grid[0] + grid[1]).ravel() % 2 == 0
    held_ok, held_rvec, held_tvec = cv2.solvePnP(
        object_pts[fit_mask], corners[fit_mask], K, dist)
    held_rms = (reprojection_rms(object_pts[~fit_mask], corners[~fit_mask],
                                 held_rvec, held_tvec, K, dist)
                if held_ok else None)
    xy = corners.reshape(-1, 2)
    return {
        "corners": xy,
        "reprojection_rms_px": reprojection_rms(
            object_pts, corners, rvec, tvec, K, dist),
        "held_out_corners_rms_px": held_rms,
        "T_camera_board": T_camera_board,
        "T_board_camera": T_board_camera,
        "camera_center_board_mm": T_board_camera[:3, 3].tolist(),
        "board_row_direction_px": (xy[BOARD[0] - 1] - xy[0]).tolist(),
        "board_column_direction_px": (xy[-BOARD[0]] - xy[0]).tolist(),
        "board_corners_bbox_xyxy": [
            float(xy[:, 0].min()), float(xy[:, 1].min()),
            float(xy[:, 0].max()), float(xy[:, 1].max()),
        ],
    }


def roi_overlaps_bbox(roi: list[int] | None, bbox: list[float]) -> bool | None:
    if roi is None:
        return None
    x, y, w, h = roi
    left, top, right, bottom = bbox
    return max(x, left) < min(x + w, right) and max(y, top) < min(y + h, bottom)


def analyze(dataset: Path, intrinsics: Path) -> dict[str, object]:
    with np.load(intrinsics) as calibration:
        K = calibration["K"].copy()
        dist = calibration["dist"].copy()
        image_size = tuple(int(v) for v in calibration["image_size"])
        board_size = tuple(int(v) for v in calibration["board_size"])
        square_mm = float(calibration["square_size_mm"])
    if board_size != BOARD or K.shape != (3, 3) or square_mm <= 0:
        raise ValueError("Intrinsics do not match the 9x6 checkerboard setup.")

    samples: list[dict[str, object]] = []
    object_pts = object_points(square_mm)
    for folder in sorted(dataset.glob("sample_????")):
        if not folder.is_dir():
            continue
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("accepted", False):
            continue
        image = read_image(folder / "rgb.png")
        if image is None or (image.shape[1], image.shape[0]) != image_size:
            raise ValueError(f"Unreadable RGB or wrong size: {folder}")
        after_image = read_image(folder / "rgb_after.png")
        if after_image is None or (after_image.shape[1], after_image.shape[0]) != image_size:
            raise ValueError(f"Unreadable RGB after frame or wrong size: {folder}")
        pose = estimate_board_pose(image, object_pts, K, dist)
        after_corners = detect_board_corners(after_image)
        result: dict[str, object] = {
            "sample_id": folder.name,
            "captured_at": manifest.get("captured_at"),
            "rgb": f"{folder.name}/rgb.png",
            "depth": f"{folder.name}/depth/depth_burst.npz",
            "target_roi_xywh": manifest.get("rgb_roi_xywh"),
            "board_detected": pose is not None,
            "board_detected_after_depth": after_corners is not None,
        }
        if pose is not None:
            corners = pose.pop("corners")
            if after_corners is not None:
                drift = board_corner_drift(corners, after_corners)
                result["rgb_before_after_board_corner_drift_px"] = drift
                result["rgb_depth_pair_stable"] = bool(
                    drift["median_px"] <= MAX_BOARD_CORNER_MEDIAN_DRIFT_PX
                    and drift["p95_px"] <= MAX_BOARD_CORNER_P95_DRIFT_PX)
            else:
                result["rgb_depth_pair_stable"] = False
            result["board_corner_count"] = len(corners)
            result["board_inner_corners_px"] = corners.tolist()
            result["target_roi_overlaps_board_corner_bbox"] = roi_overlaps_bbox(
                manifest.get("rgb_roi_xywh"), pose["board_corners_bbox_xyxy"])
            result.update(pose)
        samples.append(result)
        status = (f"RMS {pose['reprojection_rms_px']:.3f} px" if pose
                  else "board not detected")
        print(f"{folder.name}: {status}")

    if not samples:
        raise ValueError(f"No accepted sample_???? captures in {dataset}")
    detected = [sample for sample in samples if sample["board_detected"]]
    if not detected:
        raise ValueError("No checkerboards detected in accepted samples.")
    first = detected[0]
    T_first_board = first["T_camera_board"]
    first_row = np.asarray(first["board_row_direction_px"])
    first_column = np.asarray(first["board_column_direction_px"])
    warnings: list[str] = []
    for sample in detected:
        relative = camera_motion_in_first_frame(
            T_first_board, sample["T_board_camera"])
        rvec, _ = cv2.Rodrigues(relative[:3, :3])
        sample["T_first_camera_current_camera"] = relative
        sample["camera_displacement_first_camera_mm"] = relative[:3, 3].tolist()
        sample["camera_displacement_norm_mm"] = float(np.linalg.norm(relative[:3, 3]))
        sample["camera_rotation_from_first_deg"] = float(np.linalg.norm(rvec) * 180 / np.pi)
        consistent = (np.dot(first_row, sample["board_row_direction_px"]) > 0
                      and np.dot(first_column, sample["board_column_direction_px"]) > 0)
        sample["corner_order_consistent_with_first"] = bool(consistent)
        if not consistent:
            warnings.append(f"{sample['sample_id']}: checkerboard corner order may have flipped.")
        if sample["target_roi_overlaps_board_corner_bbox"]:
            warnings.append(f"{sample['sample_id']}: target ROI overlaps checkerboard corners.")
        if not sample["rgb_depth_pair_stable"]:
            warnings.append(
                f"{sample['sample_id']}: checkerboard moved between RGB frames or was lost after depth capture.")
    if len(detected) != len(samples):
        warnings.append(f"Checkerboard missing in {len(samples) - len(detected)} accepted samples.")

    rms_values = [sample["reprojection_rms_px"] for sample in detected]
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset.resolve()),
        "intrinsics": str(intrinsics.resolve()),
        "board_inner_corners": list(BOARD),
        "square_size_mm": square_mm,
        "square_size_note": "Eight printed squares measured approximately 200 mm by ruler.",
        "image_size": list(image_size),
        "rgb_depth_pair_stability_thresholds_px": {
            "board_corner_median_max": MAX_BOARD_CORNER_MEDIAN_DRIFT_PX,
            "board_corner_p95_max": MAX_BOARD_CORNER_P95_DRIFT_PX,
        },
        "reference_sample_id": first["sample_id"],
        "transform_convention": (
            "T_destination_source maps homogeneous source coordinates to destination coordinates; "
            "T_first_camera_current_camera locates each current camera in the first camera frame. "
            "Board origin is the first detected inner corner, X follows a row, Y follows a column, Z is normal."
        ),
        "scope_note": (
            "The board and object were intended to remain fixed. Board PnP gives relative camera motion, "
            "but the absolute board-to-object transform was not independently measured; "
            "these captures are not absolute 6DoF object ground truth."
        ),
        "summary": {
            "accepted_samples": len(samples),
            "board_detected_samples": len(detected),
            "rgb_depth_stable_samples": sum(
                sample.get("rgb_depth_pair_stable", False) for sample in samples),
            "mean_reprojection_rms_px": float(np.mean(rms_values)),
            "max_reprojection_rms_px": float(np.max(rms_values)),
            "warnings": warnings,
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.dataset / "board_pose_report.json"
    report = analyze(args.dataset, args.intrinsics)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                 default=lambda value: value.tolist()
                                 if isinstance(value, np.ndarray) else value),
                      encoding="utf-8")
    print(f"Saved {output}: {report['summary']['board_detected_samples']}/"
          f"{report['summary']['accepted_samples']} boards detected")


if __name__ == "__main__":
    main()
