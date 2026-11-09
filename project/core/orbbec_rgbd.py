"""Read and normalize RGB-D pairs produced by the native Orbbec backend."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


REQUIRED_PAIR_FIELDS = {
    "pair_index",
    "color_file",
    "depth_file",
    "color_timestamp_us",
    "depth_timestamp_us",
    "color_system_timestamp_us",
    "depth_system_timestamp_us",
    "system_timestamp_delta_us",
    "color_width",
    "color_height",
    "depth_width",
    "depth_height",
    "depth_value_scale",
}


@dataclass(frozen=True)
class RgbdPair:
    """One color/depth pair emitted by a single OrbbecSDK FrameSet."""

    index: int
    color: np.ndarray
    depth_raw: np.ndarray
    depth_scale: float
    color_timestamp_us: int
    depth_timestamp_us: int
    color_system_timestamp_us: int
    depth_system_timestamp_us: int

    @property
    def depth_mm(self) -> np.ndarray:
        """Return the depth image in millimeters as float32 values."""
        return self.depth_raw.astype(np.float32) * np.float32(self.depth_scale)

    @property
    def system_timestamp_delta_us(self) -> int:
        return self.color_system_timestamp_us - self.depth_system_timestamp_us


def _read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def _safe_member(root: Path, name: str, field: str) -> Path:
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes RGB-D capture directory: {name!r}") from exc
    return candidate


class RgbdCapture:
    """Indexed reader for a native capture directory.

    The reader intentionally trusts pair membership from ``pairs.csv`` rather
    than trying to re-match color and depth timestamps after capture.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        pairs_path = self.root / "pairs.csv"
        if not pairs_path.is_file():
            raise FileNotFoundError(f"RGB-D pair metadata is missing: {pairs_path}")
        with pairs_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = REQUIRED_PAIR_FIELDS - fields
            if missing:
                raise ValueError(f"pairs.csv is missing fields: {', '.join(sorted(missing))}")
            self._rows = [dict(row) for row in reader]
        if not self._rows:
            raise ValueError(f"RGB-D capture has no paired rows: {pairs_path}")

    def __len__(self) -> int:
        return len(self._rows)

    def _row(self, index: int) -> dict[str, str]:
        if index < 0 or index >= len(self):
            raise IndexError(f"RGB-D pair index out of range: {index}")
        row = self._rows[index]
        if int(row["pair_index"]) != index:
            raise ValueError(
                f"pairs.csv is not contiguous at row {index}: pair_index={row['pair_index']}"
            )
        return row

    def read(self, index: int) -> RgbdPair:
        row = self._row(index)
        color_path = _safe_member(self.root / "color", row["color_file"], "color_file")
        depth_path = _safe_member(self.root / "depth", row["depth_file"], "depth_file")
        color = _read_image(color_path, cv2.IMREAD_COLOR)
        depth = _read_image(depth_path, cv2.IMREAD_UNCHANGED)
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError(f"Depth image must be a single-channel uint16 PNG: {depth_path}")
        expected_color = (int(row["color_height"]), int(row["color_width"]))
        expected_depth = (int(row["depth_height"]), int(row["depth_width"]))
        if color.shape[:2] != expected_color:
            raise ValueError(f"Color shape {color.shape[:2]} differs from metadata {expected_color}")
        if depth.shape != expected_depth:
            raise ValueError(f"Depth shape {depth.shape} differs from metadata {expected_depth}")
        scale = float(row["depth_value_scale"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid depth_value_scale: {scale!r}")
        return RgbdPair(
            index=index,
            color=color,
            depth_raw=depth,
            depth_scale=scale,
            color_timestamp_us=int(row["color_timestamp_us"]),
            depth_timestamp_us=int(row["depth_timestamp_us"]),
            color_system_timestamp_us=int(row["color_system_timestamp_us"]),
            depth_system_timestamp_us=int(row["depth_system_timestamp_us"]),
        )

    def iter_pairs(self, start: int = 0, stop: int | None = None):
        """Yield pairs in CSV order without changing their pairing."""
        if start < 0 or start > len(self):
            raise ValueError("start must be within the capture")
        end = len(self) if stop is None else stop
        if end < start or end > len(self):
            raise ValueError("stop must be between start and the capture length")
        for index in range(start, end):
            yield self.read(index)

    def read_summary(self) -> dict[str, str]:
        summary_path = self.root / "summary.txt"
        if not summary_path.is_file():
            return {}
        values: dict[str, str] = {}
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        return values


def nonzero_median(frames: np.ndarray) -> np.ndarray:
    """Compute a uint16 median while ignoring invalid zero depth pixels."""
    if frames.ndim != 3 or frames.shape[0] == 0 or frames.dtype != np.uint16:
        raise ValueError("Depth frames must be a non-empty uint16 array of shape (N, H, W).")
    return np.ma.median(np.ma.masked_equal(frames, 0), axis=0).filled(0).astype(np.uint16)


def depth_stats(depth_mm: np.ndarray) -> dict[str, object]:
    valid = depth_mm[depth_mm > 0]
    result: dict[str, object] = {
        "valid_pixels": int(valid.size),
        "total_pixels": int(depth_mm.size),
        "valid_ratio": float(valid.size / depth_mm.size),
    }
    if valid.size:
        percentiles = np.percentile(valid, [1, 5, 25, 50, 75, 95, 99])
        result["depth_mm_percentiles"] = {
            key: float(value)
            for key, value in zip(("p01", "p05", "p25", "p50", "p75", "p95", "p99"), percentiles)
        }
        result["depth_mm_min"] = float(valid.min())
        result["depth_mm_max"] = float(valid.max())
    return result


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def _intrinsics_from_native(capture: RgbdCapture, width: int, height: int) -> dict[str, float]:
    params_path = capture.root / "camera_params.json"
    if not params_path.is_file():
        raise FileNotFoundError(
            f"Native camera parameters are missing: {params_path}; rebuild the capture backend"
        )
    params = json.loads(params_path.read_text(encoding="utf-8"))
    intrinsic = params.get("depth_intrinsic", {})
    values = {key: float(intrinsic[key]) for key in ("fx", "fy", "cx", "cy")}
    if (int(intrinsic.get("width", width)), int(intrinsic.get("height", height))) != (width, height):
        raise ValueError("Native depth intrinsic size differs from the captured depth images")
    if not all(math.isfinite(value) and value > 0 for value in values.values()):
        raise ValueError(f"Invalid native depth intrinsic parameters: {values}")
    return values


def _calibrated_intrinsics(path: Path, width: int, height: int) -> dict[str, float]:
    with np.load(path) as calibration:
        matrix = calibration["K_depth"].copy()
        size = tuple(int(value) for value in calibration["image_size"])
    if size != (width, height) or matrix.shape != (3, 3):
        raise ValueError("Calibrated depth intrinsics do not match the native depth profile")
    return {
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
    }


def materialize_depth_burst(capture_dir: Path, output_dir: Path, warmup: int = 0,
                            calibration_path: Path | None = None) -> Path:
    """Convert a native capture into the archive consumed by depth fitting."""
    capture = RgbdCapture(capture_dir)
    if len(capture) != 5:
        raise ValueError(f"Depth initialization requires exactly five pairs, got {len(capture)}")
    pairs = list(capture.iter_pairs())
    raw_frames = np.stack([pair.depth_raw for pair in pairs])
    scales = {pair.depth_scale for pair in pairs}
    if len(scales) != 1:
        raise ValueError("Depth value scale changed during the five-frame capture")
    scale = pairs[0].depth_scale
    frames = np.rint(raw_frames.astype(np.float32) * np.float32(scale))
    frames = np.clip(frames, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    median = nonzero_median(frames)
    height, width = median.shape
    native_intrinsics = _intrinsics_from_native(capture, width, height)
    intrinsics = (_calibrated_intrinsics(Path(calibration_path), width, height)
                  if calibration_path is not None else native_intrinsics)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "depth_burst.npz",
                        depth_frames_mm=frames, median_depth_mm=median)
    _write_png(output_dir / "depth_median_uint16.png", median)
    metadata = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "backend": "OrbbecSDK v1.10.37 native RGB-D FrameSet",
        "source_capture": str(capture.root),
        "native_intrinsics": native_intrinsics,
        "intrinsics_source": (str(Path(calibration_path).resolve())
                               if calibration_path is not None else "native_camera_params.json"),
        "video_mode": {
            "width": width,
            "height": height,
            "fps": 30,
            "pixel_format": "Y16",
            "depth_unit_mm": 1,
            "depth_value_scale": scale,
        },
        "registration": {
            "requested_depth_to_color": False,
            "depth_to_color_supported": False,
            "active_mode": "OFF",
        },
        "depth_mirroring": {"driver_default": False, "active": False},
        "intrinsics_estimate": intrinsics,
        "warmup_frames": int(warmup),
        "captured_frames": 5,
        "frames": [
            {"index": pair.index, "system_timestamp_delta_us": pair.system_timestamp_delta_us,
             **depth_stats(frame.astype(np.float32) * np.float32(scale))}
            for pair, frame in zip(pairs, frames)
        ],
        "median_frame": depth_stats(median.astype(np.float32)),
        "median_center_50_percent": depth_stats(
            median[height // 4:3 * height // 4, width // 4:3 * width // 4].astype(np.float32)
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir
