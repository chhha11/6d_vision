"""Transform a depth-camera CAD pose into the calibrated RGB camera frame."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .cad_pose_overlay import load_edges, project_overlay

from .paths import CALIBRATION_ROOT, MODEL_ROOT, TRACKING_ROOT

DEFAULT_DIAGNOSTICS = TRACKING_ROOT / "depth_global_current" / "diagnostics.json"
DEFAULT_EXTRINSICS = CALIBRATION_ROOT / "rgb_ir" / "rgb_ir_extrinsics.npz"
DEFAULT_MODEL = MODEL_ROOT / "energy_unit_centered_mm.stl"
DEFAULT_OUTPUT = TRACKING_ROOT / "depth_global_current" / "rgb_pose_overlay.png"


def compose_pose(T_rgb_depth: np.ndarray, T_depth_model: np.ndarray) -> np.ndarray:
    T_rgb_depth = np.asarray(T_rgb_depth, dtype=np.float64)
    T_depth_model = np.asarray(T_depth_model, dtype=np.float64)
    if T_rgb_depth.shape != (4, 4) or T_depth_model.shape != (4, 4):
        raise ValueError("Both poses must be 4x4 homogeneous transforms.")
    return T_rgb_depth @ T_depth_model


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def paired_capture_inputs(diagnostics: dict[str, object],
                          extrinsics_path: Path | None,
                          image_path: Path | None) -> tuple[Path, Path, bool]:
    depth_path = Path(diagnostics["input"]["depth"])
    sample_dir = depth_path.parent.parent if depth_path.parent.name == "depth" else None
    manifest_path = sample_dir / "manifest.json" if sample_dir is not None else None
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path is not None and manifest_path.is_file() else None)
    if image_path is None:
        if manifest is None:
            raise ValueError("No paired RGB image was recorded; pass --image explicitly.")
        image_path = sample_dir / "rgb_before.png"
    if extrinsics_path is None:
        if manifest is None:
            extrinsics_path = DEFAULT_EXTRINSICS
        else:
            calibration = manifest.get("rgb_depth_calibration") or {}
            filename = calibration.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Sample has no valid RGB-depth calibration snapshot.")
            extrinsics_path = sample_dir / filename
            expected_hash = calibration.get("sha256")
            if not extrinsics_path.is_file() or hashlib.sha256(
                    extrinsics_path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError("RGB-depth calibration snapshot is missing or changed.")
    paired_capture = bool(
        manifest is not None
        and image_path.resolve().parent == sample_dir.resolve()
        and extrinsics_path.resolve().parent == sample_dir.resolve()
    )
    return extrinsics_path, image_path, paired_capture


def run(diagnostics_path: Path, extrinsics_path: Path | None,
        image_path: Path | None,
        model_path: Path, output_path: Path, max_edges: int = 3500
        ) -> dict[str, object]:
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if not diagnostics.get("pose_quality", {}).get("accepted", False):
        raise RuntimeError("Depth global pose was not accepted.")
    extrinsics_path, image_path, paired_capture = paired_capture_inputs(
        diagnostics, extrinsics_path, image_path)
    T_depth_model = np.asarray(
        diagnostics["pose"]["model_to_camera"], dtype=np.float64)
    with np.load(extrinsics_path) as extrinsics:
        T_rgb_depth = extrinsics["T_rgb_depth"].copy()
        K_rgb = extrinsics["K_rgb"].copy()
        dist_rgb = extrinsics["dist_rgb"].copy()
        image_size = tuple(extrinsics["image_size"]) if "image_size" in extrinsics else None
    T_rgb_model = compose_pose(T_rgb_depth, T_depth_model)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read RGB image: {image_path}")
    if image_size is not None and image_size != (image.shape[1], image.shape[0]):
        raise ValueError("RGB image dimensions differ from the calibration.")
    vertices, edges = load_edges(model_path, max_edges)
    rvec, _ = cv2.Rodrigues(T_rgb_model[:3, :3])
    tvec = T_rgb_model[:3, 3]
    overlay = project_overlay(
        image, vertices, edges, K_rgb, dist_rgb, rvec.ravel(), tvec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(output_path, overlay)
    result = {
        "definition": "T_rgb_model = T_rgb_depth @ T_depth_model",
        "diagnostics": str(diagnostics_path.resolve()),
        "extrinsics": str(extrinsics_path.resolve()),
        "rgb_image": str(image_path.resolve()),
        "model": str(model_path.resolve()),
        "output": str(output_path.resolve()),
        "T_depth_model": T_depth_model.tolist(),
        "T_rgb_depth": T_rgb_depth.tolist(),
        "T_rgb_model": T_rgb_model.tolist(),
        "rvec_rgb_model_rad": rvec.ravel().tolist(),
        "tvec_rgb_model_mm": tvec.tolist(),
        "depth_pose_quality": diagnostics["pose_quality"],
        "paired_capture": paired_capture,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--extrinsics", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-edges", type=int, default=3500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        args.diagnostics.resolve(),
        args.extrinsics.resolve() if args.extrinsics is not None else None,
        args.image.resolve() if args.image is not None else None,
        args.model.resolve(), args.output.resolve(),
        args.max_edges)
    print(f"RGB tvec: {np.round(result['tvec_rgb_model_mm'], 3).tolist()} mm")
    print(f"Saved {result['output']}")


if __name__ == "__main__":
    main()
