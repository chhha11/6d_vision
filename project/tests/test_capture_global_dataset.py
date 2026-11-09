import argparse
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import capture_global_dataset as capture


class CaptureGlobalDatasetTest(unittest.TestCase):
    def test_reopen_releases_failed_camera_on_opencv_exception(self):
        frame = np.zeros((480, 640, 3), np.uint8)

        class Camera:
            def __init__(self, fail):
                self.fail = fail
                self.released = False

            def read(self):
                if self.fail:
                    raise capture.cv2.error('native RGB read interrupted')
                return True, frame

            def release(self):
                self.released = True

        failed, recovered = Camera(True), Camera(False)
        with patch.object(capture, 'open_camera', side_effect=[failed, recovered]) as open_camera, \
                patch.object(capture.time, 'sleep') as pause:
            camera, _ = capture.reopen_rgb(3, 2, 1)
        self.assertTrue(failed.released)
        self.assertIs(camera, recovered)
        self.assertFalse(recovered.released)
        pause.assert_called_once_with(.5)
        self.assertEqual([call.args[0] for call in open_camera.call_args_list], [3, 3])

    def test_plan_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.csv"
            path.write_text(
                "id,kind,distance_mm,pose,position,background,requirement,target_present\n"
                "p01,positive,700,side,center,plain,first,true\n"
                "p01,positive,850,yaw30,left,plain,second,true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate IDs"):
                capture.load_plan(path)

    def test_plan_distinguishes_target_presence_from_negative_kind(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plan.csv"
            path.write_text(
                "id,kind,distance_mm,pose,position,background,requirement,target_present\n"
                "p01,positive,700,side,center,plain,target,true\n"
                "n01,negative,850,empty,center,plain,empty,false\n"
                "n02,negative,1100,out_of_range,center,plain,far,true\n",
                encoding="utf-8",
            )
            samples = capture.load_plan(path)
            self.assertEqual([sample.target_present for sample in samples],
                             [True, False, True])

    def test_camera_name_is_resolved_once_and_manual_index_is_preserved(self):
        named = argparse.Namespace(camera=None, camera_name="USB Camera")
        with patch.object(capture, "resolve_camera_index",
                          return_value=(2, "USB Camera")) as resolver:
            info = capture.select_rgb_camera(named)
        resolver.assert_called_once_with("USB Camera", None)
        self.assertEqual(named.camera, 2)
        self.assertEqual(info["camera_name"], "USB Camera")
        self.assertTrue(info["camera_identity_verified"])

        manual = argparse.Namespace(camera=3, camera_name="USB Camera")
        with patch.object(capture, "resolve_camera_index",
                          return_value=(3, None)) as resolver:
            info = capture.select_rgb_camera(manual)
        resolver.assert_called_once_with("USB Camera", 3)
        self.assertEqual(manual.camera, 3)
        self.assertTrue(info["manual_index_override"])

    def test_preview_text_scale_fits_frame_width(self):
        line = "Setup: " + "long requirement " * 20
        scale = capture.fitted_text_scale(line, 624)
        width = capture.cv2.getTextSize(line, capture.cv2.FONT_HERSHEY_SIMPLEX,
                                        scale, 1)[0][0]
        self.assertLessEqual(width, 624)

    def test_mouse_drag_sets_roi_without_blocking_preview(self):
        selector = capture.RoiSelector((640, 480))
        selector.on_mouse(capture.cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
        selector.on_mouse(capture.cv2.EVENT_MOUSEMOVE, 40, 30, 0, None)
        self.assertEqual(selector.drag_roi, (40, 30, 60, 70))
        self.assertIsNone(selector.roi)
        selector.on_mouse(capture.cv2.EVENT_LBUTTONUP, 40, 30, 0, None)
        self.assertEqual(selector.roi, (40, 30, 60, 70))
        self.assertIsNone(selector.drag_roi)
        selector.arm()
        self.assertIsNone(selector.roi)
        self.assertIn("Drag mouse", selector.feedback)

    def test_plan_resumes_after_completed_and_skipped_samples(self):
        samples = [
            capture.Sample("p01", "positive", 700, "side", "center", "plain", "first", True),
            capture.Sample("p02", "positive", 850, "yaw30", "left", "plain", "second", True),
            capture.Sample("p03", "negative", None, "empty", "center", "plain", "third", False),
        ]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "p01").mkdir()
            (output / "p01" / "manifest.json").write_text('{"accepted": true}', encoding="utf-8")
            (output / "skipped_p02.json").write_text("{}", encoding="utf-8")
            self.assertEqual(capture.next_pending(samples, output), samples[2])
            self.assertEqual(capture.next_free_sample_id(output), "sample_0001")

    def test_depth_presence_rejects_missing_target_depth(self):
        image = np.full((100, 100), 2500, dtype=np.uint16)
        image[20:30, 20:30] = 0
        result = capture.depth_presence(image, (10, 10, 50, 50), 1100)
        self.assertFalse(result["passed"])
        self.assertEqual(result["nearby_pixels"], 0)

    def test_depth_presence_accepts_enough_nearby_pixels(self):
        image = np.zeros((100, 100), dtype=np.uint16)
        image[25:45, 30:50] = 820
        result = capture.depth_presence(image, (10, 10, 60, 60), 850)
        self.assertTrue(result["passed"])
        self.assertEqual(result["nearby_pixels"], 400)
        self.assertAlmostEqual(result["nearby_median_mm"], 820.0)

    def test_rgb_roi_is_mapped_into_depth_frame(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        mapped = capture.map_rgb_roi_to_depth(
            (300, 220, 40, 30), 850, K, np.zeros((1, 5)),
            K, np.eye(4), (480, 640), margin_px=0)
        self.assertEqual(mapped, (300, 220, 40, 30))

    def test_depth_qc_requires_three_valid_frames(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        sample = capture.Sample("p01", "positive", 850, "side", "center", "plain",
                                "target", True)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            extrinsics = root / "extrinsics.npz"
            depth_file = root / "depth_burst.npz"
            np.savez(extrinsics, K_rgb=K, dist_rgb=np.zeros((1, 5)),
                     K_depth=K, T_depth_rgb=np.eye(4))
            frames = np.zeros((5, 480, 640), dtype=np.uint16)
            frames[:2, 200:240, 300:340] = 850
            median = np.zeros((480, 640), dtype=np.uint16)
            median[200:240, 300:340] = 850
            np.savez(depth_file, depth_frames_mm=frames, median_depth_mm=median)
            result = capture.depth_qc(depth_file, sample, (290, 190, 60, 60), extrinsics)
            self.assertFalse(result["passed"])
            self.assertEqual(result["passing_frames"], 2)
            frames[2, 200:240, 300:340] = 850
            np.savez(depth_file, depth_frames_mm=frames, median_depth_mm=median)
            result = capture.depth_qc(depth_file, sample, (290, 190, 60, 60), extrinsics)
            self.assertTrue(result["passed"])
            self.assertEqual(result["passing_frames"], 3)

    def test_failed_positive_is_preserved_but_negative_with_target_can_pass(self):
        class FakeCamera:
            def release(self):
                pass

            def get(self, property_id):
                return {
                    capture.cv2.CAP_PROP_FRAME_WIDTH: 640,
                    capture.cv2.CAP_PROP_FRAME_HEIGHT: 480,
                    capture.cv2.CAP_PROP_FPS: 30,
                }.get(property_id, 0)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((5, 480, 640), dtype=np.uint16)
        metadata = {
            "depth_mirroring": {"active": False},
            "video_mode": {"width": 640, "height": 480, "depth_unit_mm": 1},
            "registration": {"requested_depth_to_color": False},
            "captured_frames": 5,
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
            extrinsics = root / "extrinsics.npz"
            np.savez(extrinsics, K_rgb=K, dist_rgb=np.zeros((1, 5)),
                     K_depth=K, T_depth_rgb=np.eye(4))
            args = argparse.Namespace(output=root, extrinsics=extrinsics, camera=0,
                                      depth_warmup=2, depth_timeout=20,
                                      rgb_reopen_attempts=5,
                                      camera_info={"camera_index": 0,
                                                   "camera_name": "USB Camera"})

            def fake_depth_worker(command, **_kwargs):
                depth_dir = Path(command[command.index("--output") + 1])
                depth_dir.mkdir()
                np.savez(depth_dir / "depth_burst.npz", depth_frames_mm=depth,
                         median_depth_mm=depth[0])
                (depth_dir / "metadata.json").write_text(json.dumps(metadata),
                                                          encoding="utf-8")
                if depth_dir.parent.name.startswith(".P002_"):
                    extrinsics.write_bytes(b"source changed after snapshot")
                return subprocess.CompletedProcess(command, 0, "", "")

            positive = capture.Sample("P001", "positive", 700, "side", "center",
                                      "normal", "target", True)
            positive_good = capture.Sample("P002", "positive", 700, "side",
                                           "center", "normal", "target", True)
            negative = capture.Sample("N007", "negative", 1100, "out_of_range",
                                      "center", "normal", "far", True)
            with patch.object(capture, "read_rgb", return_value=frame), \
                    patch.object(capture, "reopen_rgb", return_value=(FakeCamera(), frame)), \
                    patch.object(capture.subprocess, "run", side_effect=fake_depth_worker):
                failed, _ = capture.capture_one(FakeCamera(), positive,
                                                 (290, 190, 60, 60), args)
                accepted, _ = capture.capture_one(FakeCamera(), negative,
                                                   (290, 190, 60, 60), args)
                depth[:3, 200:240, 300:340] = 700
                accepted_positive, _ = capture.capture_one(
                    FakeCamera(), positive_good, (290, 190, 60, 60), args)

            self.assertFalse(failed["accepted"])
            self.assertTrue(Path(failed["path"]).is_dir())
            self.assertIsNone(failed["depth_presence_screen"]["nearby_median_mm"])
            calibration_copy = Path(failed["path"]) / "rgb_depth_extrinsics.npz"
            self.assertTrue(calibration_copy.is_file())
            self.assertEqual(failed["rgb_depth_calibration"]["sha256"],
                             hashlib.sha256(calibration_copy.read_bytes()).hexdigest())
            self.assertTrue(accepted["accepted"])
            self.assertTrue(accepted["target_present"])
            self.assertEqual(accepted["rgb_camera"]["camera_index"], 0)
            self.assertEqual(accepted["rgb_camera"]["before"]["actual_frame_size"],
                             [640, 480])
            self.assertEqual(accepted["rgb_camera"]["after"]["reported_fps"], 30)
            self.assertIsNone(accepted["rgb_depth_calibration"])
            self.assertEqual(accepted["sample"]["kind"], "negative")
            self.assertEqual(accepted["rgb_roi_xywh"], [290, 190, 60, 60])
            self.assertTrue(accepted_positive["accepted"])
            self.assertTrue((root / "P002" / "rgb_depth_extrinsics.npz").is_file())
            self.assertEqual(capture.next_pending([positive, negative], root), positive)

    def test_plan_auto_advances_after_successful_capture(self):
        class FakeCamera:
            def release(self):
                pass

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / "plan.csv"
            plan.write_text(
                "id,kind,target_present,distance_mm,pose,position,background,requirement\n"
                "P001,positive,true,700,yaw_0,center,normal,first\n"
                "P002,positive,true,850,yaw_30,left,normal,second\n",
                encoding="utf-8",
            )
            extrinsics = root / "extrinsics.npz"
            extrinsics.touch()
            args = argparse.Namespace(
                plan=plan, output=root / "output", extrinsics=extrinsics,
                camera=None, camera_name="USB Camera", roi=(290, 190, 60, 60),
                once=False, kind="positive", distance_mm=None, pose_label="",
                notes="", target_count=2, target_present=False,
            )
            camera = FakeCamera()
            seen = []

            def fake_capture(cap, sample, roi, options):
                seen.append((sample.id, roi))
                destination = options.output / sample.id
                destination.mkdir()
                (destination / "manifest.json").write_text(
                    '{"accepted": true}', encoding="utf-8")
                return {"accepted": True, "path": str(destination)}, cap

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            callbacks = []
            keys = iter((ord("s"), ord("b"), "drag", ord("s")))

            def wait_key(_delay):
                key = next(keys)
                if key == "drag":
                    callback = callbacks[0]
                    callback(capture.cv2.EVENT_LBUTTONDOWN, 290, 190, 0, None)
                    callback(capture.cv2.EVENT_MOUSEMOVE, 350, 250, 0, None)
                    callback(capture.cv2.EVENT_LBUTTONUP, 350, 250, 0, None)
                    return -1
                return key

            with patch.object(capture, "resolve_camera_index",
                              return_value=(2, "USB Camera")) as resolver, \
                    patch.object(capture, "open_camera", return_value=camera) as opener, \
                    patch.object(capture, "read_rgb", return_value=frame) as reader, \
                    patch.object(capture, "capture_one", side_effect=fake_capture), \
                    patch.object(capture.cv2, "namedWindow"), \
                    patch.object(capture.cv2, "imshow"), \
                    patch.object(capture.cv2, "setMouseCallback",
                                 side_effect=lambda _window, callback: callbacks.append(callback)), \
                    patch.object(capture.cv2, "selectROI",
                                 side_effect=AssertionError("Blocking ROI selection used")), \
                    patch.object(capture.cv2, "waitKey", side_effect=wait_key), \
                    patch.object(capture.cv2, "destroyAllWindows"):
                capture.run(args)

            resolver.assert_called_once_with("USB Camera", None)
            opener.assert_called_once_with(2, 640, 480)
            self.assertEqual(reader.call_count, 4)
            self.assertEqual(seen, [("P001", (290, 190, 60, 60)),
                                    ("P002", (290, 190, 60, 60))])


if __name__ == "__main__":
    unittest.main()
