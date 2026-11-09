import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from core.evaluate_m3t import load_recording, lost_spans, pose_meters, read_native_poses, write_config


class M3TGeometryTest(unittest.TestCase):
    def row(self, index=0, valid=True):
        return {'valid': valid, 'video_frame_index': index, 'rvec_rad': [.4, -.2, .1],
                'tvec_mm': [50., -30., 500.], 'status': 'TRACKING'}

    def test_metric_pose_preserves_rotation_and_ignores_display_compensation(self):
        row = self.row()
        row['box_display_y_offset_px'] = 12
        row['display_pose'] = {'tvec_mm': [999, 999, 999]}
        T = pose_meters(row)
        np.testing.assert_allclose(T[:3, :3], cv2.Rodrigues(np.array(row['rvec_rad']))[0])
        np.testing.assert_allclose(T[:3, 3], [.05, -.03, .5])
        np.testing.assert_allclose(T[3], [0, 0, 0, 1])
        self.assertEqual(row['tvec_mm'], [50., -30., 500.])

    def test_rejects_invalid_nonfinite_and_behind_camera_initializations(self):
        rows = [self.row(valid=False), dict(self.row(), tvec_mm=[0, 0, -10]),
                dict(self.row(), rvec_rad=[0, np.nan, 0])]
        for row in rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                pose_meters(row)

    def test_recording_can_start_with_seed_but_rejects_missing_sync(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            diag = root / 'attempt_001/tracking/diagnostics_test'
            diag.mkdir(parents=True)
            (diag / 'rgb.avi').touch()
            (diag / 'metadata.json').write_text(json.dumps({'frame_timestamps_s': [0, .1]}))
            rows = [self.row(0, False), self.row(1, True)]
            pose_file = diag / 'poses.jsonl'
            pose_file.write_text('\n'.join(json.dumps(row) for row in rows))
            _, _, loaded = load_recording(root)
            self.assertFalse(loaded[0]['valid'])
            rows[1]['video_frame_index'] = 2
            pose_file.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, 'Non-contiguous'):
                load_recording(root)

    def test_config_keeps_original_intrinsics_and_initial_source_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            K = np.array([[530., 0, 316], [0, 531, 241], [0, 0, 1]])
            d = np.array([.1, -.15, .001, -.002, -.13])
            path = root / 'config.yml'
            write_config(path, video=root/'rgb.avi', mesh=root/'cad.obj',
                         metadata={'camera_matrix': K, 'distortion': d},
                         initial_row=self.row(1), output=root, cache=root/'region.bin',
                         divides=3, image_size=1000, max_frames=8)
            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            try:
                np.testing.assert_array_equal(fs.getNode('camera_matrix').mat(), K)
                np.testing.assert_array_equal(fs.getNode('distortion').mat().ravel(), d)
                np.testing.assert_allclose(fs.getNode('initial_pose_m').mat()[:3, 3], [.05, -.03, .5])
                self.assertEqual(fs.getNode('start_frame').real(), 1)
            finally:
                fs.release()

    def test_native_frames_cannot_silently_shift_or_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'poses.csv'
            path.write_text('frame_index,api_ok,processing_ms,rx,ry,rz,tx_mm,ty_mm,tz_mm\n'
                            '1,1,2,0,0,0,50,-30,500\n2,1,3,0,0,0,51,-30,500\n')
            self.assertEqual(len(read_native_poses(path, start_frame=1)), 2)
            with self.assertRaises(ValueError):
                read_native_poses(path, start_frame=0)

    def test_loss_count_uses_tracking_state_not_display_alias(self):
        rows = [dict(status='TRACKING', display_status='TRACKING'),
                dict(status='TARGET LOST', display_status='SEARCHING'),
                dict(status='TARGET LOST', display_status='SEARCHING'),
                dict(status='TRACKING', display_status='TRACKING')]
        spans = lost_spans(rows, [0, .1, .2, .4])
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]['duration_s'], .3)
        self.assertEqual(spans[0]['start_frame'], 1)
        self.assertEqual(spans[0]['end_frame_exclusive'], 3)
        shifted = lost_spans(rows, [0, .1, .2, .4], start_frame=10)
        self.assertEqual(shifted[0]['start_frame'], 11)
        self.assertEqual(shifted[0]['end_frame_exclusive'], 13)


if __name__ == '__main__':
    unittest.main()
