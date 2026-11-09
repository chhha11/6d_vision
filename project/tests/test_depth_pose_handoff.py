import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from core import depth_pose_handoff as handoff
from core.paths import DATA_ROOT


class DepthPoseHandoffTest(unittest.TestCase):
    def test_compose_pose_applies_depth_to_rgb_transform(self):
        T_rgb_depth = np.eye(4)
        T_rgb_depth[:3, 3] = [10, -2, 3]
        T_depth_model = np.eye(4)
        T_depth_model[:3, 3] = [1, 2, 100]
        actual = handoff.compose_pose(T_rgb_depth, T_depth_model)
        np.testing.assert_allclose(actual[:3, 3], [11, 0, 103])

    def test_compose_pose_rejects_non_homogeneous_shape(self):
        with self.assertRaises(ValueError):
            handoff.compose_pose(np.eye(3), np.eye(4))

    def test_paired_capture_uses_own_rgb_and_calibration_snapshot(self):
        sample = DATA_ROOT / "samples" / "global_dataset" / "P001"
        diagnostics = {"input": {"depth": str(sample / "depth" / "depth_burst.npz")}}
        calibration, image, paired = handoff.paired_capture_inputs(
            diagnostics, None, None)
        self.assertEqual(calibration, sample / "rgb_depth_extrinsics.npz")
        self.assertEqual(image, sample / "rgb_before.png")
        self.assertTrue(paired)

    def test_paired_capture_rejects_changed_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            sample = Path(folder)
            depth = sample / "depth"
            depth.mkdir()
            (sample / "rgb_depth_extrinsics.npz").write_bytes(b"changed")
            (sample / "manifest.json").write_text(json.dumps({
                "rgb_depth_calibration": {
                    "file": "rgb_depth_extrinsics.npz",
                    "sha256": hashlib.sha256(b"original").hexdigest(),
                },
            }), encoding="utf-8")
            diagnostics = {"input": {"depth": str(depth / "depth_burst.npz")}}
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                handoff.paired_capture_inputs(diagnostics, None, None)

    def test_unpaired_probe_requires_explicit_image(self):
        diagnostics = {"input": {"depth": str(
            DATA_ROOT / "tracking" / "depth_probe_adjusted_unmirrored" / "depth_burst.npz")}}
        with self.assertRaisesRegex(ValueError, "pass --image"):
            handoff.paired_capture_inputs(diagnostics, None, None)


if __name__ == "__main__":
    unittest.main()
