import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core import collect_tracking_evidence as collect
from core import live_depth_6dof as live


class CollectTrackingEvidenceTest(unittest.TestCase):
    def test_failed_reference_never_publishes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attempt = root / 'references/failure/attempt_001'
            attempt.mkdir(parents=True)
            collect.write_json(attempt / 'rgb_initial_pose.json', dict(depth_pose_quality={'accepted': False}))
            destination = root / 'library/session/attempt_001'
            with self.assertRaisesRegex(RuntimeError, 'stillness'):
                collect.publish_reference(attempt, destination, SimpleNamespace())
            self.assertFalse(destination.exists())

    def test_motion_tolerated_pose_cannot_be_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            attempt = Path(folder) / 'failure/attempt_001'
            attempt.mkdir(parents=True)
            collect.write_json(attempt / 'rgb_initial_pose.json', dict(
                depth_pose_quality={'accepted': True}, rgb_stillness=dict(
                    motion_tolerated=True, global_response=1, global_shift_px=[0, 0], target_shift_px=[0, 0])))
            with self.assertRaisesRegex(RuntimeError, 'stillness'):
                collect.validate_reference(attempt, SimpleNamespace())

    def test_publish_rewrites_paths_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attempt = root / 'references/failure/attempt_001'
            attempt.mkdir(parents=True)
            collect.write_json(attempt / 'rgb_initial_pose.json', dict(
                rgb_image=str(attempt / 'rgb_after.png'), depth_burst=str(attempt / 'depth/depth_burst.npz'),
                extrinsics=str(attempt / 'extrinsics.npz'), intrinsics=str(attempt / 'intrinsics.npz')))
            destination = root / 'library/session/attempt_001'
            with patch.object(collect, 'validate_reference', return_value={'sift': 24}):
                collect.publish_reference(attempt, destination, SimpleNamespace())
                with self.assertRaises(FileExistsError):
                    collect.publish_reference(attempt, destination, SimpleNamespace())
            payload = json.loads((destination / 'rgb_initial_pose.json').read_text())
            self.assertEqual(payload['rgb_image'], str(destination / 'rgb_after.png'))
            self.assertEqual(payload['intrinsics'], str(destination / 'intrinsics.npz'))
            self.assertTrue(payload['collection_review']['accepted'])

    def test_rejected_depth_reopens_rgb_and_preserves_error(self):
        class Camera:
            released = False
            def release(self):
                self.released = True
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calibration = root / 'extrinsics.npz'
            np.savez(calibration, K_rgb=np.eye(3), dist_rgb=np.zeros(5), T_rgb_depth=np.eye(4))
            intrinsics = root / 'intrinsics.npz'
            np.savez(intrinsics, K=np.eye(3), dist=np.zeros(5), image_size=[640, 480])
            args = SimpleNamespace(extrinsics=calibration, intrinsics=intrinsics, depth_warmup=2,
                                   depth_timeout_s=8, depth_frame_timeout_s=2)
            before, after = Camera(), Camera()
            frame = np.zeros((480, 640, 3), np.uint8)
            attempt = root / 'references/failure/attempt_001'
            with patch.object(live, 'read_rgb', return_value=frame), \
                 patch.object(live, 'run_with_preview', side_effect=lambda task, *_, **_kw: task()), \
                 patch.object(live, 'run_depth_worker', autospec=True, side_effect=RuntimeError('depth failed')), \
                 patch.object(live, 'recover_rgb', return_value=(after, frame)):
                cap, counts, error = collect.capture_reference(before, attempt, (10, 10, 100, 100), args, 0, None)
            self.assertIs(cap, after)
            self.assertTrue(before.released)
            self.assertIsNone(counts)
            self.assertEqual(error, 'depth failed')
            self.assertFalse(json.loads((attempt / 'collection_check.json').read_text())['accepted'])
            self.assertFalse((attempt / 'rgb_initial_pose.json').exists())

    def test_resume_rejects_different_calibration_before_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            collect.write_json(root / 'session.json', dict(mode='references', files={
                'model': {'sha256': 'changed'}}))
            with patch.object(collect, 'run_references') as camera:
                with self.assertRaisesRegex(ValueError, 'changed model'):
                    collect.main(['references', '--output', str(root)])
                camera.assert_not_called()


if __name__ == '__main__':
    unittest.main()
