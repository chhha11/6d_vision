import unittest
import json
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.evaluate_captured_data import (
    calibration_path,
    discover_samples,
    endpoint_axis_line_check,
    endpoint_center_check,
    evaluate_sample,
    projection_check,
    replay_integration_sessions,
    summarize,
)
from core.paths import DATA_ROOT


ROOT = DATA_ROOT


class CapturedDataEvaluationTest(unittest.TestCase):
    def test_inventory_excludes_failed_and_review_captures(self):
        samples = discover_samples(
            ROOT / "samples", {"global", "board", "integration"})
        counts = Counter((item.group, item.kind, item.target_present)
                         for item in samples)
        self.assertEqual(counts[("global", "positive", True)], 27)
        self.assertEqual(counts[("global", "negative", False)], 4)
        self.assertEqual(counts[("global", "negative", True)], 3)
        self.assertEqual(counts[("board", "positive", True)], 10)
        self.assertEqual(counts[("integration", "positive", True)], 10)
        self.assertNotIn("global/P023", {item.sample_id for item in samples})
        self.assertIn("global/sample_0001", {item.sample_id for item in samples})

    def test_capture_calibration_snapshot_matches_manifest(self):
        sample = next(item for item in discover_samples(ROOT / "samples", {"global"})
                      if item.sample_id == "global/P001")
        path, source = calibration_path(sample, ROOT / "missing.npz")
        self.assertEqual(source, "capture_snapshot")
        self.assertTrue(path.is_file())

    def test_projection_check_uses_saved_roi_only_for_scoring(self):
        pose = np.eye(4)
        pose[2, 3] = 1000
        intrinsics = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
        vertices = np.array([[-10, -10, -10], [10, 10, 10]], dtype=float)
        result = projection_check(pose, intrinsics, np.zeros(5),
                                  vertices, [45, 45, 10, 10])
        self.assertTrue(result["roi_center_inside"])
        self.assertEqual(result["roi_center_error_px"], 0.0)
        self.assertGreater(result["roi_bbox_iou"], 0.0)

    def test_summary_keeps_target_present_negatives_separate(self):
        rows = [
            dict(sample_id="global/P001", group="global", kind="positive",
                 target_present=True, status="accepted", roi_center_inside=True),
            dict(sample_id="global/P002", group="global", kind="positive",
                 target_present=True, status="rejected_quality", roi_center_inside=True),
            dict(sample_id="global/N001", group="global", kind="negative",
                 target_present=False, status="accepted"),
            dict(sample_id="global/N005", group="global", kind="negative",
                 target_present=True, status="accepted"),
        ]
        result = summarize(rows)
        self.assertEqual(result["global_positive"]["depth_pose_accepted"], 1)
        self.assertEqual(result["global_positive"]["accepted_center_inside_manual_roi"], 1)
        self.assertEqual(result["global_target_absent_negative"]["false_acceptances"], 1)
        self.assertEqual(result["global_target_present_negative"]["accepted_poses"], 1)

    def test_rgb_projection_error_does_not_erase_depth_acceptance(self):
        sample = next(item for item in discover_samples(ROOT / "samples", {"global"})
                      if item.sample_id == "global/P001")
        pose = np.eye(4)
        pose[2, 3] = 800
        diagnostics = {
            "pose_quality": {"accepted": True},
            "pose": {"fitness": 1.0, "inlier_rmse_mm": 2.0,
                     "translation_mm": [0, 0, 800],
                     "model_to_camera": pose.tolist()},
            "selected_cluster": {"point_count": 500},
            "cluster_eps_used_mm": 10.0,
        }
        with tempfile.TemporaryDirectory() as folder:
            with patch("core.depth_global_initializer.run", return_value=diagnostics):
                row = evaluate_sample(sample, Path(folder), ROOT / "models" / "energy_unit_centered_mm.stl",
                                      ROOT / "missing.npz", np.empty((0, 3)), object())
        self.assertEqual(row["status"], "accepted")
        self.assertEqual(row["rgb_projection_status"], "error")
        self.assertEqual(summarize([row])["global_positive"]["depth_pose_accepted"], 1)

    def test_no_candidate_retains_generated_rejection_diagnostics(self):
        sample = next(item for item in discover_samples(ROOT / "samples", {"global"})
                      if item.sample_id == "global/N001")

        def reject(_depth, _metadata, _model, output, _settings):
            output.mkdir(parents=True, exist_ok=True)
            (output / "diagnostics.json").write_text(json.dumps({
                "pose_quality": {"accepted": False},
                "rejection_reason": "No cluster matches the energy-unit dimensions.",
            }), encoding="utf-8")
            raise RuntimeError("No cluster matches the energy-unit dimensions.")

        with tempfile.TemporaryDirectory() as folder:
            stale_pose = Path(folder) / "samples" / "global" / "N001" / "rgb_tracker_pose.json"
            stale_pose.parent.mkdir(parents=True)
            stale_pose.write_text("old pose", encoding="utf-8")
            with patch("core.depth_global_initializer.run", side_effect=reject):
                row = evaluate_sample(sample, Path(folder), ROOT / "models" / "energy_unit_centered_mm.stl",
                                      ROOT / "missing.npz", np.empty((0, 3)), object())
            self.assertEqual(row["status"], "no_candidate")
            self.assertIn("No cluster matches", row["rejection_reason"])
            self.assertTrue(Path(row["diagnostics"]).exists())
            self.assertFalse(stale_pose.exists())

    def test_endpoint_check_catches_lateral_drift_with_valid_depth(self):
        result = endpoint_center_check([-294, -40, 785], [29, -23, 778], 50.0)
        self.assertGreater(result["end_anchor_center_error_mm"], 300)
        self.assertFalse(result["end_anchor_center_check_passed"])
        self.assertEqual(result["end_anchor_center_delta_xyz_mm"], [-323.0, -17.0, 7.0])

    def test_endpoint_axis_ignores_symmetric_axis_direction(self):
        anchor = np.eye(4)
        flipped = endpoint_axis_line_check([np.pi, 0, 0], anchor.tolist(), 15.0)
        self.assertAlmostEqual(flipped["end_anchor_axis_line_error_deg"], 0.0)
        self.assertTrue(flipped["end_anchor_axis_line_check_passed"])
        tilted = endpoint_axis_line_check([np.pi / 4, 0, 0], anchor.tolist(), 15.0)
        self.assertFalse(tilted["end_anchor_axis_line_check_passed"])

    def test_replay_skips_endpoint_scoring_without_valid_final_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            data_root = Path(folder) / "data"
            output = Path(folder) / "evaluation"
            session = data_root / "integration_sessions" / "session_01"
            session.mkdir(parents=True)
            (session / "session_manifest.json").write_text(json.dumps({
                "motion_rgb": {"video": "motion/clip_0001/rgb.avi"},
            }), encoding="utf-8")
            end_pose = np.eye(4)
            end_pose[2, 3] = 800
            rows = [
                dict(sample_id="integration/session_01/start", group="integration",
                     session_id="session_01", tracker_pose="start_pose.json"),
                dict(sample_id="integration/session_01/end", group="integration",
                     session_id="session_01", status="accepted",
                     rgb_center_mm=[0, 0, 800], T_rgb_model=end_pose.tolist()),
            ]

            def replay(_command, **_kwargs):
                summary = output / "tracking" / "session_01" / "summary.json"
                summary.write_text(json.dumps({
                    "frames": 5, "tracking_frames": 2, "video_reached_eof": False,
                    "final_pose": {"tracking_valid": False, "tvec_mm": [0, 0, 800],
                                   "rvec": [0, 0, 0]},
                    "end_nominal_depth_error_mm": 0.0,
                    "end_nominal_depth_check_passed": True,
                }), encoding="utf-8")
                return type("Completed", (), {"returncode": 0})()

            with patch("core.evaluate_captured_data.subprocess.run", side_effect=replay):
                result = replay_integration_sessions(
                    data_root, output, rows, ROOT / "models" / "energy_unit_centered_mm.stl",
                    ROOT / "calibration" / "camera_intrinsics.npz", 50.0, 15.0)[0]
            self.assertEqual(result["endpoint_check_status"], "skipped_untrusted_final_pose")
            self.assertNotIn("end_anchor_center_and_axis_consistent", result)
            self.assertTrue(result["end_nominal_depth_check_raw"])
            self.assertIsNone(result["end_nominal_depth_check_passed"])

    def test_board_summary_compares_poses_in_first_camera_frame(self):
        first = np.eye(4)
        second = np.eye(4)
        second[0, 3] = 100
        first_pose = np.eye(4)
        first_pose[:3, 3] = [0, 0, 800]
        second_pose = np.eye(4)
        second_pose[:3, 3] = [-100, 0, 800]
        rows = [
            dict(sample_id="board/sample_0001", group="board", kind="positive",
                 target_present=True, status="accepted", T_rgb_model=first_pose.tolist()),
            dict(sample_id="board/sample_0002", group="board", kind="positive",
                 target_present=True, status="accepted", T_rgb_model=second_pose.tolist()),
            dict(sample_id="board/sample_0003", group="board", kind="positive",
                 target_present=True, status="accepted", T_rgb_model=second_pose.tolist()),
        ]
        board_report = {"samples": [
            dict(sample_id="sample_0001", board_detected=True,
                 rgb_depth_pair_stable=True, corner_order_consistent_with_first=True,
                 T_first_camera_current_camera=first.tolist()),
            dict(sample_id="sample_0002", board_detected=True,
                 rgb_depth_pair_stable=True, corner_order_consistent_with_first=True,
                 T_first_camera_current_camera=second.tolist()),
            dict(sample_id="sample_0003", board_detected=True,
                 rgb_depth_pair_stable=False, corner_order_consistent_with_first=True,
                 T_first_camera_current_camera=first.tolist()),
        ]}
        result = summarize(rows, board_report)["board"]
        self.assertEqual(result["comparable_poses"], 2)
        self.assertAlmostEqual(result["center_scatter_p95_mm"], 0.0)
        self.assertAlmostEqual(result["axis_line_scatter_median_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
