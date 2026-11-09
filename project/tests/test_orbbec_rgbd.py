import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.orbbec_rgbd import RgbdCapture, materialize_depth_burst


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode test image: {path}")
    encoded.tofile(path)


class OrbbecRgbdTest(unittest.TestCase):
    def make_capture(self, root: Path, pairs: int = 5) -> None:
        (root / "color").mkdir(parents=True)
        (root / "depth").mkdir()
        with (root / "pairs.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "pair_index", "color_file", "depth_file", "color_timestamp_us",
                "depth_timestamp_us", "color_system_timestamp_us",
                "depth_system_timestamp_us", "frame_timestamp_delta_us",
                "system_timestamp_delta_us", "color_width", "color_height",
                "depth_width", "depth_height", "depth_value_scale",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(pairs):
                color_name = f"{index:06d}.jpg"
                depth_name = f"{index:06d}.png"
                write_image(root / "color" / color_name,
                            np.full((2, 3, 3), index, dtype=np.uint8))
                write_image(root / "depth" / depth_name,
                            np.full((2, 3), 1000 + index, dtype=np.uint16))
                writer.writerow({
                    "pair_index": index,
                    "color_file": color_name,
                    "depth_file": depth_name,
                    "color_timestamp_us": index,
                    "depth_timestamp_us": index + 1,
                    "color_system_timestamp_us": 10 + index,
                    "depth_system_timestamp_us": 20 + index,
                    "frame_timestamp_delta_us": -1,
                    "system_timestamp_delta_us": -10,
                    "color_width": 3,
                    "color_height": 2,
                    "depth_width": 3,
                    "depth_height": 2,
                    "depth_value_scale": 0.5,
                })
        (root / "camera_params.json").write_text(json.dumps({
            "depth_intrinsic": {
                "fx": 577.8, "fy": 577.9, "cx": 1.5, "cy": 0.5,
                "width": 3, "height": 2,
            },
        }), encoding="utf-8")

    def test_reader_preserves_pair_membership_and_applies_scale(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_capture(root, pairs=1)
            capture = RgbdCapture(root)
            pair = capture.read(0)
            self.assertEqual(len(capture), 1)
            self.assertEqual(pair.color.shape, (2, 3, 3))
            self.assertEqual(pair.depth_raw.dtype, np.uint16)
            np.testing.assert_allclose(pair.depth_mm, 500.0)
            self.assertEqual(pair.system_timestamp_delta_us, -10)

    def test_materialize_depth_burst_writes_legacy_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "capture"
            output = Path(folder) / "depth"
            self.make_capture(root)
            materialize_depth_burst(root, output, warmup=2)
            with np.load(output / "depth_burst.npz") as archive:
                self.assertEqual(archive["depth_frames_mm"].shape, (5, 2, 3))
                self.assertEqual(archive["depth_frames_mm"].dtype, np.uint16)
                self.assertEqual(int(archive["median_depth_mm"][0, 0]), 501)
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["captured_frames"], 5)
            self.assertEqual(metadata["video_mode"]["depth_unit_mm"], 1)
            self.assertAlmostEqual(metadata["intrinsics_estimate"]["fx"], 577.8)

    def test_materialize_can_keep_calibrated_depth_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "capture"
            output = Path(folder) / "depth"
            calibration = Path(folder) / "extrinsics.npz"
            self.make_capture(root)
            np.savez(calibration, K_depth=np.array([
                [500.0, 0.0, 1.0], [0.0, 501.0, 1.0], [0.0, 0.0, 1.0]
            ]), image_size=np.array([3, 2]))
            materialize_depth_burst(root, output, calibration_path=calibration)
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["intrinsics_estimate"]["cx"], 1.0)
            self.assertEqual(metadata["native_intrinsics"]["cx"], 1.5)


if __name__ == "__main__":
    unittest.main()
