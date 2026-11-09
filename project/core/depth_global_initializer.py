"""Offline Astra depth initialization for the energy-unit CAD model."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .paths import MODEL_ROOT, TRACKING_ROOT

DEFAULT_SESSION = TRACKING_ROOT / "depth_probe_adjusted_unmirrored"
DEFAULT_DEPTH = DEFAULT_SESSION / "depth_burst.npz"
DEFAULT_METADATA = DEFAULT_SESSION / "metadata.json"
DEFAULT_MODEL = MODEL_ROOT / "energy_unit_centered_mm.stl"
DEFAULT_OUTPUT = TRACKING_ROOT / "depth_global_current"


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RgbConstraint:
    box: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    T_rgb_depth: np.ndarray

    def contains(self, points):
        camera = points @ self.T_rgb_depth[:3, :3].T + self.T_rgb_depth[:3, 3]
        uv, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), self.K, self.dist)
        uv = uv.reshape(-1, 2)
        margin = np.maximum((self.box[2:] - self.box[:2]) * .08, 5.)
        return ((camera[:, 2] > 0) & np.isfinite(uv).all(axis=1)
                & np.all(uv >= self.box[:2] - margin, axis=1)
                & np.all(uv <= self.box[2:] + margin, axis=1))


@dataclass(frozen=True)
class Settings:
    min_depth_mm: float = 550.0
    max_depth_mm: float = 1200.0
    max_abs_x_mm: float = 350.0
    min_y_mm: float = -250.0
    max_y_mm: float = 300.0
    voxel_mm: float = 4.0
    plane_threshold_mm: float = 4.0
    plane_iterations: int = 1000
    min_plane_points: int = 800
    min_plane_fraction: float = 0.10
    max_planes: int = 2
    cluster_eps_mm: float = 10.0
    cluster_retry_eps_mm: float = 20.0
    cluster_min_points: int = 20
    candidate_min_points: int = 250
    candidate_long_min_mm: float = 125.0
    candidate_long_max_mm: float = 175.0
    candidate_mid_min_mm: float = 60.0
    candidate_mid_max_mm: float = 115.0
    candidate_thickness_min_mm: float = 10.0
    candidate_thickness_max_mm: float = 100.0
    icp_roll_step_deg: float = 30.0
    icp_coarse_distance_mm: float = 20.0
    icp_fine_distance_mm: float = 8.0
    axis_direction_min_fitness_margin: float = 0.03
    axis_direction_min_rmse_margin_mm: float = 1.0


@dataclass
class ClusterCandidate:
    label: int
    point_count: int
    center_mm: np.ndarray
    axes_camera: np.ndarray
    robust_extents_mm: np.ndarray
    full_extents_mm: np.ndarray
    score: float
    accepted: bool

    def json_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "point_count": self.point_count,
            "center_mm": self.center_mm.tolist(),
            "long_axis_camera": self.axes_camera[:, 0].tolist(),
            "axes_camera": self.axes_camera.tolist(),
            "robust_extents_mm": self.robust_extents_mm.tolist(),
            "full_extents_mm": self.full_extents_mm.tolist(),
            "score": self.score,
            "accepted": self.accepted,
        }


def nonzero_median(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 3 or not frames.shape[0]:
        raise ValueError("Depth frames must have shape (frames, height, width).")
    return np.ma.median(np.ma.masked_equal(frames, 0), axis=0).filled(0).astype(np.uint16)


def depth_to_points(depth_mm: np.ndarray, intrinsics: Intrinsics,
                    mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if depth_mm.ndim != 2:
        raise ValueError("Depth image must be two-dimensional.")
    valid = depth_mm > 0
    if mask is not None:
        if mask.shape != depth_mm.shape:
            raise ValueError("Depth mask shape does not match the image.")
        valid &= mask
    rows, cols = np.nonzero(valid)
    z = depth_mm[rows, cols].astype(np.float64)
    x = (cols - intrinsics.cx) * z / intrinsics.fx
    y = (rows - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z)), np.column_stack((cols, rows))


def robust_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 3:
        raise ValueError("At least three points are required for PCA.")
    mean = points.mean(axis=0)
    eigenvalues, axes = np.linalg.eigh(np.cov(points - mean, rowvar=False))
    axes = axes[:, np.argsort(eigenvalues)[::-1]]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
    dominant = int(np.argmax(np.abs(axes[:, 0])))
    if axes[dominant, 0] < 0:
        axes[:, 0] *= -1.0
        axes[:, 2] *= -1.0
    projected = (points - mean) @ axes
    low, high = np.percentile(projected, [1.0, 99.0], axis=0)
    center = mean + axes @ ((low + high) / 2.0)
    return center, axes, high - low, np.ptp(projected, axis=0)


def candidate_from_points(label: int, points: np.ndarray,
                          settings: Settings) -> ClusterCandidate:
    center, axes, robust_extents, full_extents = robust_pca(points)
    long, middle, thickness = robust_extents
    accepted = bool(
        len(points) >= settings.candidate_min_points
        and settings.candidate_long_min_mm <= long <= settings.candidate_long_max_mm
        and settings.candidate_mid_min_mm <= middle <= settings.candidate_mid_max_mm
        and settings.candidate_thickness_min_mm <= thickness <= settings.candidate_thickness_max_mm
    )
    score = (
        abs(long - 150.0) / 25.0
        + abs(middle - 90.0) / 35.0
        + 0.20 * abs(center[0]) / settings.max_abs_x_mm
        + 25.0 / max(math.sqrt(len(points)), 1.0)
    )
    if not accepted:
        score += 100.0
    return ClusterCandidate(label, len(points), center, axes, robust_extents,
                            full_extents, float(score), accepted)


def make_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def remove_large_planes(cloud: o3d.geometry.PointCloud,
                        settings: Settings) -> tuple[o3d.geometry.PointCloud, list[dict[str, object]]]:
    import open3d as o3d

    remaining = cloud
    initial_count = len(remaining.points)
    min_inliers = max(settings.min_plane_points,
                      int(math.ceil(initial_count * settings.min_plane_fraction)))
    planes: list[dict[str, object]] = []
    o3d.utility.random.seed(0)
    for _ in range(settings.max_planes):
        if len(remaining.points) < min_inliers:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=settings.plane_threshold_mm,
            ransac_n=3,
            num_iterations=settings.plane_iterations,
        )
        if len(inliers) < min_inliers:
            break
        planes.append({
            "equation": [float(value) for value in model],
            "inlier_points": len(inliers),
        })
        remaining = remaining.select_by_index(inliers, invert=True)
    return remaining, planes


def find_candidates(cloud: o3d.geometry.PointCloud,
                    settings: Settings) -> tuple[np.ndarray, list[ClusterCandidate]]:
    labels = np.asarray(cloud.cluster_dbscan(
        eps=settings.cluster_eps_mm,
        min_points=settings.cluster_min_points,
        print_progress=False,
    ))
    points = np.asarray(cloud.points)
    candidates = [
        candidate_from_points(int(label), points[labels == label], settings)
        for label in sorted(set(labels.tolist()))
        if label >= 0 and int(np.count_nonzero(labels == label)) >= settings.cluster_min_points
    ]
    candidates.sort(key=lambda item: item.score)
    return labels, candidates


def rotation_hypotheses(candidate: ClusterCandidate, roll_step_deg: float):
    """Yield CAD-to-camera rotations for both side and end-face observations.

    In an end-face view the visible point cloud is approximately a 90 mm disk
    (two large PCA axes) with a 20 mm depth thickness.  The first PCA axis is
    therefore a diameter, not the 150 mm CAD long axis.  Treat the thin PCA
    normal as an additional long-axis candidate in that case.  This is gated
    by the partial geometry dimensions so ordinary side views keep the fast,
    single family of hypotheses.
    """
    axis_families = [("side", candidate.axes_camera[:, 0],
                      candidate.axes_camera[:, 1])]
    extents = np.asarray(candidate.robust_extents_mm, dtype=np.float64)
    end_face = bool(extents[0] < 120. and extents[1] >= 45.
                    and extents[2] < 35.)
    if end_face:
        # Model z is the physical long axis.  The observed thin direction is
        # the cap normal, while either broad direction can seed model x.
        axis_families.insert(0, ("end_face", candidate.axes_camera[:, 2],
                                 candidate.axes_camera[:, 0]))
    for axis_source, long_axis, transverse in axis_families:
        long_axis = np.asarray(long_axis, dtype=np.float64).copy()
        transverse = np.asarray(transverse, dtype=np.float64)
        long_axis /= max(np.linalg.norm(long_axis), 1e-12)
        transverse = transverse - long_axis * float(np.dot(transverse, long_axis))
        transverse /= max(np.linalg.norm(transverse), 1e-12)
        for axis_sign in (1, -1):
            z_axis = long_axis * axis_sign
            x_axis = transverse - z_axis * float(np.dot(transverse, z_axis))
            x_axis /= max(np.linalg.norm(x_axis), 1e-12)
            y_axis = np.cross(z_axis, x_axis)
            for roll_deg in np.arange(0.0, 360.0, roll_step_deg):
                angle = math.radians(float(roll_deg))
                x_rotated = math.cos(angle) * x_axis + math.sin(angle) * y_axis
                y_rotated = -math.sin(angle) * x_axis + math.cos(angle) * y_axis
                rotation = np.column_stack((x_rotated, y_rotated, z_axis))
                yield axis_sign, float(roll_deg), rotation, axis_source


def initial_translation(model_points: np.ndarray, observed_points: np.ndarray,
                        rotation_model_to_camera: np.ndarray) -> np.ndarray:
    rotated = model_points @ rotation_model_to_camera.T
    translation = np.zeros(3, dtype=np.float64)
    for coordinate in (0, 1):
        model_bounds = np.percentile(rotated[:, coordinate], [1.0, 99.0])
        observed_bounds = np.percentile(observed_points[:, coordinate], [1.0, 99.0])
        translation[coordinate] = observed_bounds.mean() - model_bounds.mean()
    translation[2] = (
        np.percentile(observed_points[:, 2], 3.0)
        - np.percentile(rotated[:, 2], 1.0)
    )
    return translation


def end_face_evidence(points: np.ndarray, pose: dict[str, object]) -> dict[str, object]:
    """Check that a fitted partial cloud spans a CAD end cap, not a rail patch.

    The known model supplies the hidden length.  Its position and axis must
    still be supported by the observed cap diameter, plane and CAD residual.
    Semantic RGB detection is required separately at handoff.
    """
    transform = np.asarray(pose["model_to_camera"], dtype=np.float64)
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    cap = np.abs(np.abs(local[:, 2]) - 72.) <= 5.
    cap_points = local[cap]
    support = float(np.mean(cap)) if len(local) else 0.
    extents = (np.diff(np.percentile(cap_points[:, :2], [1, 99], axis=0), axis=0)[0]
               if len(cap_points) >= 20 else np.zeros(2))
    radial_support = (float(np.mean(np.linalg.norm(cap_points[:, :2], axis=1) <= 50.))
                      if len(cap_points) else 0.)
    accepted = bool(pose.get("axis_source") == "end_face" and support >= .5
                    and np.all(extents >= 65.) and np.all(extents <= 102.)
                    and radial_support >= .9)
    return {"accepted": accepted, "support_fraction": support,
            "transverse_extents_mm": extents.tolist(),
            "radial_support_fraction": radial_support}


def invert_transform(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


@lru_cache(maxsize=2)
def _cached_model_cloud(model_path: str, mtime_ns: int, size: int):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(model_path)
    if mesh.is_empty():
        raise RuntimeError(f"Cannot read CAD mesh: {model_path}")
    o3d.utility.random.seed(0)
    return mesh.sample_points_uniformly(number_of_points=30000).voxel_down_sample(2.5)


def load_model_cloud(model_path: Path):
    path = model_path.resolve()
    stat = path.stat()
    return _cached_model_cloud(str(path), stat.st_mtime_ns, stat.st_size)


def cad_icp_hypotheses(model_path: Path, observed_points: np.ndarray,
                       candidate: ClusterCandidate, settings: Settings) -> tuple[list[dict[str, object]], o3d.geometry.PointCloud]:
    import open3d as o3d

    model_cloud = load_model_cloud(model_path)
    observed_cloud = make_cloud(observed_points)
    model_points = np.asarray(model_cloud.points)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
    results: list[dict[str, object]] = []
    for axis_sign, roll_deg, rotation, axis_source in rotation_hypotheses(
            candidate, settings.icp_roll_step_deg):
        if axis_source == "side" and results:
            cap_best = min(results, key=lambda item: (-item['fitness'], item['inlier_rmse_mm']))
            if (cap_best['fitness'] >= .98 and cap_best['inlier_rmse_mm'] <= 2.5
                    and end_face_evidence(observed_points, cap_best)['accepted']):
                # A complete, precise cap fit already explains the cloud.
                # Avoid paying for all side hypotheses on every end-on view.
                break
        if axis_source == "end_face":
            # The measured disk is an end surface, not the model center.
            # Place the near CAD cap at its measured plane before ICP.  Using
            # the full CAD bounding-box center can start 20-40 mm sideways
            # from the cap and trap ICP on the cylinder wall.
            normal = rotation[:, 2]
            cap_center = candidate.center_mm.copy()
            cap_center += normal * (np.median(observed_points @ normal)
                                    - float(cap_center @ normal))
            cap_sign = -1. if float(normal @ cap_center) > 0 else 1.
            translation = cap_center - normal * (72. * cap_sign)
        else:
            translation = initial_translation(model_points, observed_points, rotation)
        model_to_camera = np.eye(4, dtype=np.float64)
        model_to_camera[:3, :3] = rotation
        model_to_camera[:3, 3] = translation
        camera_to_model = invert_transform(model_to_camera)
        coarse = o3d.pipelines.registration.registration_icp(
            observed_cloud, model_cloud, settings.icp_coarse_distance_mm,
            camera_to_model,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria,
        )
        fine = o3d.pipelines.registration.registration_icp(
            observed_cloud, model_cloud, settings.icp_fine_distance_mm,
            coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria,
        )
        refined_model_to_camera = invert_transform(fine.transformation)
        rotation_vector, _ = cv2.Rodrigues(refined_model_to_camera[:3, :3])
        rotation_vector = rotation_vector.ravel()
        results.append({
            "axis_sign": axis_sign,
            "axis_source": axis_source,
            "initial_roll_deg": roll_deg,
            "fitness": float(fine.fitness),
            "inlier_rmse_mm": float(fine.inlier_rmse),
            "model_to_camera": refined_model_to_camera.tolist(),
            "translation_mm": refined_model_to_camera[:3, 3].tolist(),
            "tvec_mm": refined_model_to_camera[:3, 3].tolist(),
            "rvec_rad": rotation_vector.tolist(),
            "rvec_deg": np.degrees(rotation_vector).tolist(),
            "rotation_model_to_camera": refined_model_to_camera[:3, :3].tolist(),
        })
    results.sort(key=lambda item: (-float(item["fitness"]), float(item["inlier_rmse_mm"])))
    return results, model_cloud


def depth_preview(depth_mm: np.ndarray, selected_points: np.ndarray,
                  candidate: ClusterCandidate, intrinsics: Intrinsics) -> np.ndarray:
    valid = depth_mm[(depth_mm > 0) & (depth_mm < 1200)]
    low, high = (np.percentile(valid, [1, 99]) if valid.size else (0.0, 1.0))
    normalized = np.clip((depth_mm.astype(np.float32) - low) / max(high - low, 1.0), 0.0, 1.0)
    gray = np.round((1.0 - normalized) * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    preview[(depth_mm == 0) | (depth_mm >= 1200)] = 0
    z = selected_points[:, 2]
    cols = np.rint(intrinsics.fx * selected_points[:, 0] / z + intrinsics.cx).astype(int)
    rows = np.rint(intrinsics.fy * selected_points[:, 1] / z + intrinsics.cy).astype(int)
    inside = ((cols >= 0) & (cols < depth_mm.shape[1])
              & (rows >= 0) & (rows < depth_mm.shape[0]))
    overlay = np.zeros(depth_mm.shape, dtype=np.uint8)
    overlay[rows[inside], cols[inside]] = 255
    overlay = cv2.dilate(overlay, np.ones((5, 5), np.uint8))
    preview[overlay > 0] = (0, 255, 0)
    projected = np.column_stack((cols[inside], rows[inside]))
    if len(projected):
        left, top = projected.min(axis=0)
        right, bottom = projected.max(axis=0)
        cv2.rectangle(preview, (int(left), int(top)), (int(right), int(bottom)),
                      (255, 255, 255), 1, cv2.LINE_AA)
    half = candidate.axes_camera[:, 0] * candidate.robust_extents_mm[0] / 2.0
    endpoints = np.vstack((candidate.center_mm - half, candidate.center_mm + half))
    axis_cols = intrinsics.fx * endpoints[:, 0] / endpoints[:, 2] + intrinsics.cx
    axis_rows = intrinsics.fy * endpoints[:, 1] / endpoints[:, 2] + intrinsics.cy
    cv2.line(preview, tuple(np.rint([axis_cols[0], axis_rows[0]]).astype(int)),
             tuple(np.rint([axis_cols[1], axis_rows[1]]).astype(int)),
             (0, 255, 255), 2, cv2.LINE_AA)
    return preview


def registration_preview(depth_mm: np.ndarray, observed_points: np.ndarray,
                         aligned_model_points: np.ndarray,
                         candidate: ClusterCandidate,
                         intrinsics: Intrinsics) -> np.ndarray:
    preview = depth_preview(depth_mm, observed_points, candidate, intrinsics)
    observed_mask = np.all(preview == (0, 255, 0), axis=2)
    model_mask = np.zeros(depth_mm.shape, dtype=np.uint8)
    z = aligned_model_points[:, 2]
    valid = z > 0
    cols = np.rint(intrinsics.fx * aligned_model_points[valid, 0] / z[valid]
                    + intrinsics.cx).astype(int)
    rows = np.rint(intrinsics.fy * aligned_model_points[valid, 1] / z[valid]
                    + intrinsics.cy).astype(int)
    inside = ((cols >= 0) & (cols < depth_mm.shape[1])
              & (rows >= 0) & (rows < depth_mm.shape[0]))
    model_mask[rows[inside], cols[inside]] = 255
    model_mask = cv2.dilate(model_mask, np.ones((3, 3), np.uint8))
    preview[model_mask > 0] = (0, 0, 255)
    preview[observed_mask & (model_mask > 0)] = (0, 255, 255)
    return preview


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def run(depth_path: Path, metadata_path: Path, model_path: Path,
        output: Path, settings: Settings,
        rgb_constraint: RgbConstraint | None = None) -> dict[str, object]:
    import open3d as o3d

    output.mkdir(parents=True, exist_ok=True)
    with np.load(depth_path) as archive:
        if "depth_frames_mm" in archive:
            depth = nonzero_median(archive["depth_frames_mm"])
            frame_count = int(archive["depth_frames_mm"].shape[0])
        elif "median_depth_mm" in archive:
            depth = archive["median_depth_mm"].copy()
            frame_count = 1
        else:
            raise RuntimeError(f"No depth array in {depth_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_mode = metadata.get("video_mode", {})
    if video_mode.get("depth_unit_mm") != 1:
        raise ValueError("Depth units must be millimeters.")
    if metadata.get("depth_mirroring", {}).get("active") is not False:
        raise ValueError("Depth mirroring must be confirmed off.")
    if metadata.get("registration", {}).get("requested_depth_to_color") is not False:
        raise ValueError("Depth-to-color registration must be off.")
    if depth.shape != (video_mode.get("height"), video_mode.get("width")):
        raise ValueError("Depth array shape does not match its capture metadata.")
    calibration = metadata["intrinsics_estimate"]
    intrinsics = Intrinsics(*(float(calibration[key]) for key in ("fx", "fy", "cx", "cy")))
    workspace_mask = (
        (depth >= settings.min_depth_mm)
        & (depth <= settings.max_depth_mm)
    )
    points, _ = depth_to_points(depth, intrinsics, workspace_mask)
    points = points[
        (np.abs(points[:, 0]) <= settings.max_abs_x_mm)
        & (points[:, 1] >= settings.min_y_mm)
        & (points[:, 1] <= settings.max_y_mm)
    ]
    if rgb_constraint is not None and len(points):
        points = points[rgb_constraint.contains(points)]
    cloud = (make_cloud(points).voxel_down_sample(settings.voxel_mm)
             if len(points) else make_cloud(points))
    remaining, planes = remove_large_planes(cloud, settings)
    labels, candidates = (find_candidates(remaining, settings)
                          if len(remaining.points) else (np.empty(0, dtype=int), []))
    accepted = [candidate for candidate in candidates if candidate.accepted]
    cluster_eps_used = settings.cluster_eps_mm
    if (not accepted and len(remaining.points)
            and settings.cluster_retry_eps_mm > settings.cluster_eps_mm):
        retry_settings = replace(settings, cluster_eps_mm=settings.cluster_retry_eps_mm)
        retry_labels, retry_candidates = find_candidates(remaining, retry_settings)
        retry_accepted = [candidate for candidate in retry_candidates if candidate.accepted]
        if retry_accepted:
            labels, candidates, accepted = retry_labels, retry_candidates, retry_accepted
            cluster_eps_used = settings.cluster_retry_eps_mm
    partial_depth = False
    if not accepted and rgb_constraint is not None and len(remaining.points):
        # A near-range Astra burst can miss dark endcaps. Only try partial
        # geometry inside an independently confirmed RGB detection. Handoff
        # must additionally verify its RGB extent and long axis.
        partial_settings = replace(settings, candidate_long_min_mm=80.,
                                   candidate_mid_min_mm=45.,
                                   cluster_eps_mm=settings.cluster_retry_eps_mm)
        partial_labels, partial_candidates = find_candidates(remaining, partial_settings)
        partial_accepted = [candidate for candidate in partial_candidates if candidate.accepted]
        if partial_accepted:
            labels, candidates, accepted = partial_labels, partial_candidates, partial_accepted
            cluster_eps_used = partial_settings.cluster_eps_mm
            partial_depth = True
        if not accepted:
            # A nearly head-on cap may be thinner than 10 mm in depth.  Only
            # admit that special case when BOTH diameters match the known cap;
            # keep the existing thickness gate for all other partial clouds.
            cap_settings = replace(partial_settings, candidate_mid_min_mm=65.,
                                   candidate_long_max_mm=115.,
                                   candidate_thickness_min_mm=5.,
                                   candidate_thickness_max_mm=35.)
            cap_labels, cap_candidates = find_candidates(remaining, cap_settings)
            cap_accepted = [candidate for candidate in cap_candidates if candidate.accepted]
            if cap_accepted:
                labels, candidates, accepted = cap_labels, cap_candidates, cap_accepted
                cluster_eps_used = cap_settings.cluster_eps_mm
                partial_depth = True
    if not accepted:
        diagnostics = {
            "input": {
                "depth": str(depth_path.resolve()),
                "metadata": str(metadata_path.resolve()),
                "model": str(model_path.resolve()),
                "depth_frames": frame_count,
            },
            "coordinate_system": "camera X right, Y down, Z forward; millimeters",
            "intrinsics": asdict(intrinsics),
            "settings": asdict(settings),
            "cluster_eps_used_mm": cluster_eps_used,
            "workspace_points": len(points),
            "voxel_points": len(cloud.points),
            "remaining_points": len(remaining.points),
            "removed_planes": planes,
            "clusters": [candidate.json_dict() for candidate in candidates],
            "selected_cluster": None,
            "pose": None,
            "pose_quality": {"accepted": False},
            "rejection_reason": "No cluster matches the energy-unit dimensions.",
        }
        (output / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8")
        raise RuntimeError("No cluster matches the energy-unit dimensions.")
    selected = accepted[0]
    remaining_points = np.asarray(remaining.points)
    selected_points = remaining_points[labels == selected.label]
    icp_results, model_cloud = cad_icp_hypotheses(
        model_path, selected_points, selected, settings)
    best_pose = icp_results[0]
    cap_evidence = end_face_evidence(selected_points, best_pose)

    selected_cloud = make_cloud(selected_points)
    selected_cloud.paint_uniform_color((0.1, 0.8, 0.2))
    o3d.io.write_point_cloud(str(output / "selected_candidate.ply"), selected_cloud)
    aligned_model = o3d.geometry.PointCloud(model_cloud)
    aligned_model.transform(np.asarray(best_pose["model_to_camera"], dtype=np.float64))
    aligned_model.paint_uniform_color((0.85, 0.25, 0.1))
    o3d.io.write_point_cloud(str(output / "aligned_model.ply"), aligned_model)
    write_png(output / "segmentation_preview.png",
              depth_preview(depth, selected_points, selected, intrinsics))
    write_png(output / "registration_preview.png",
              registration_preview(depth, selected_points,
                                   np.asarray(aligned_model.points), selected,
                                   intrinsics))

    opposite_axis = [item for item in icp_results
                     if item["axis_source"] == best_pose["axis_source"]
                     and item["axis_sign"] != best_pose["axis_sign"]]
    opposite_best = min(
        opposite_axis,
        key=lambda item: (-float(item["fitness"]), float(item["inlier_rmse_mm"])),
    )
    accepted_pose = bool(float(best_pose["fitness"]) >= 0.95
                         and float(best_pose["inlier_rmse_mm"]) <= 3.0
                         and (best_pose["axis_source"] != "end_face"
                              or cap_evidence["accepted"]))
    opposite_fitness_margin = (
        float(best_pose["fitness"]) - float(opposite_best["fitness"])
    )
    opposite_rmse_margin = (
        float(opposite_best["inlier_rmse_mm"])
        - float(best_pose["inlier_rmse_mm"])
    )
    axis_direction_confident = bool(
        opposite_fitness_margin >= settings.axis_direction_min_fitness_margin
        or opposite_rmse_margin >= settings.axis_direction_min_rmse_margin_mm
    )
    quality = {
        "accepted": accepted_pose,
        "translation_and_axis_line_confident": accepted_pose,
        "long_axis_direction_confident": axis_direction_confident,
        "fully_disambiguated": False,
        "roll_ambiguous": True,
        "roll_symmetry_degrees": 120.0,
        "best_vs_second_rmse_margin_mm": float(icp_results[1]["inlier_rmse_mm"]
                                                - best_pose["inlier_rmse_mm"]),
        "best_vs_opposite_axis_fitness_margin": opposite_fitness_margin,
        "best_vs_opposite_axis_rmse_margin_mm": opposite_rmse_margin,
        "pose_coordinate_frame": "Astra depth camera",
        "partial_depth": partial_depth,
        "requires_rgb_verification": partial_depth,
        "end_face_geometry": cap_evidence,
    }

    diagnostics: dict[str, object] = {
        "input": {
            "depth": str(depth_path.resolve()),
            "metadata": str(metadata_path.resolve()),
            "model": str(model_path.resolve()),
            "depth_frames": frame_count,
        },
        "coordinate_system": "camera X right, Y down, Z forward; millimeters",
        "intrinsics": asdict(intrinsics),
        "settings": asdict(settings),
        "cluster_eps_used_mm": cluster_eps_used,
        "workspace_points": len(points),
        "voxel_points": len(cloud.points),
        "remaining_points": len(remaining.points),
        "removed_planes": planes,
        "clusters": [candidate.json_dict() for candidate in candidates],
        "selected_cluster": selected.json_dict(),
        "pose": best_pose,
        "pose_quality": quality,
        "pose_hypotheses": icp_results,
        "rgb_constraint": ({"box": rgb_constraint.box.tolist()}
                           if rgb_constraint is not None else None),
        "symmetry_note": "Roll is only observable modulo the CAD's approximate 120 degree symmetry.",
    }
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = run(args.depth.resolve(), args.metadata.resolve(),
                      args.model.resolve(), args.output.resolve(), Settings())
    selected = diagnostics["selected_cluster"]
    pose = diagnostics["pose"]
    print(f"Candidate points: {selected['point_count']}")
    print(f"Observed extents: {np.round(selected['robust_extents_mm'], 2).tolist()} mm")
    print(f"ICP fitness: {pose['fitness']:.4f}; RMSE: {pose['inlier_rmse_mm']:.3f} mm")
    print(f"Model center: {np.round(pose['translation_mm'], 2).tolist()} mm")
    print(f"Saved diagnostics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
